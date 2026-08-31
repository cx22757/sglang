#!/usr/bin/env python3
"""Compare per-layer KV digests across populate-write and replay-write.

Parses ``[KVSUM]`` lines from a DSV4 server.log produced with
``SGLANG_DSV4_KV_DIGEST=1`` (see ``DeepseekV4AscendAttnBackend._kv_digest``) and,
for each request index and each digest layer, compares the populate-write
digest (ntok-max) against the replay-write digest (ntok-max).

Because replay hits the radix cache (cached ~= prefix length), the replay-write
digest covers the SAME token range as the populate-write digest, so equal
numbers prove the cached KV bytes are unchanged between populate and replay for
that layer. A DIFF means that layer's cached KV changed (or the pool slot was
reused) -- the smoking gun for a write-side corruption.

Usage:
  python3 analyze_kvsum.py <server.log> [--rid-filter <substr>]

Output: per-request per-layer OK / DIFF / ntok-DIFF / MISSING, plus a summary.
Exit code 0 when every compared layer matches, 1 otherwise.
"""
import argparse
import re
import sys
from collections import defaultdict

_LINE = re.compile(
    r"\[KVSUM\] phase=(?P<phase>\w+) rid=(?P<rid>\S+) layer=(?P<layer>\d+) "
    r"ntok=(?P<ntok>\d+) swa=(?P<swa>\S+) c4=(?P<c4>\S+) c128=(?P<c128>\S+)"
    r"(?:\s+state=(?P<state>\S+))?(?:\s+idx=(?P<idx>\S+))?"
    r"(?:\s+sidecar=(?P<sidecar>\S+))?"
    r"(?:\s+swabt=(?P<swabt>\S+))?"
    r"(?:\s+c4bt=(?P<c4bt>\S+))?(?:\s+c128bt=(?P<c128bt>\S+))?"
    r"(?:\s+idxbt=(?P<idxbt>\S+))?"
)
_RID = re.compile(r"-(?P<kind>pop|replay)-(\d+)$")

# swa / c4 / c128 value tokens: a number string, "None" (no data for this
# ratio pool on this layer), or "ERR:...".
_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _val_status(x: str, y: str) -> str:
    """Classify a (populate, replay) field pair.

    Returns ``OK`` / ``DIFF`` / ``ERR`` / ``SUSPECT``. ``SUSPECT`` marks
    inf/nan values: they are garbage even when both sides agree, so a
    consistent-but-inf comparison must not pass as OK.
    """
    if x == "None" and y == "None":
        return "OK"
    if x.startswith("ERR") or y.startswith("ERR"):
        return "ERR"
    # inf/nan are not numeric (miss _NUM) -- detect before the fallback
    # string compare so "inf" == "inf" is not reported as OK.
    if x in ("inf", "-inf", "nan") or y in ("inf", "-inf", "nan"):
        return "SUSPECT"
    if _NUM.match(x) and _NUM.match(y):
        return "OK" if float(x) == float(y) else "DIFF"
    return "OK" if x == y else "DIFF"


def parse(path: str, rid_filter: str):
    # data[i][kind][phase][layer] = (ntok, swa, c4, c128)  -- ntok-max kept
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(path, errors="ignore") as f:
        for line in f:
            m = _LINE.search(line)
            if not m:
                continue
            rid = m.group("rid")
            if rid_filter and rid_filter not in rid:
                continue
            rm = _RID.search(rid)
            if not rm:
                continue
            i = int(rm.group(2))
            kind = rm.group(1)
            phase = m.group("phase")
            layer = int(m.group("layer"))
            ntok = int(m.group("ntok"))
            vals = (ntok, m.group("swa"), m.group("c4"), m.group("c128"),
                    m.group("state") or "None", m.group("idx") or "None",
                    m.group("sidecar") or "None", m.group("swabt") or "None",
                    m.group("c4bt") or "None", m.group("c128bt") or "None",
                    m.group("idxbt") or "None")
            cur = data[i][kind][phase].get(layer)
            if cur is None or ntok > cur[0]:
                data[i][kind][phase][layer] = vals
    return data


def _cmp_vals(a, b):
    """Compare (ntok, swa, c4, c128, state, idx) tuples.

    Returns (result, detail), result in {"OK", "DIFF", "SUSPECT", "MISSING"}.
    """
    if a is None or b is None:
        return "MISSING", "MISSING"
    if a[0] != b[0]:
        return "DIFF", f"ntok-DIFF ({a[0]} vs {b[0]})"
    fields = (
        "swa", "c4", "c128", "state", "idx", "sidecar",
        "swabt", "c4bt", "c128bt", "idxbt",
    )
    statuses = []
    parts = []
    for name, x, y in zip(fields, a[1:], b[1:]):
        st = _val_status(x, y)
        statuses.append(st)
        if st == "OK":
            parts.append(f"{name} OK")
        elif st == "DIFF":
            parts.append(f"{name} {x}!= {y}")
        elif st == "SUSPECT":
            parts.append(f"{name} SUSPECT({x}/{y})")
        else:
            parts.append(f"{name} ERR")
    if "DIFF" in statuses:
        return "DIFF", " ".join(parts)
    if "SUSPECT" in statuses:
        return "SUSPECT", " ".join(parts)
    return "OK", " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="server.log path")
    ap.add_argument("--rid-filter", default="", help="only rids containing substr")
    a = ap.parse_args()

    data = parse(a.log, a.rid_filter)
    requests = sorted(data)
    if not requests:
        print(f"no [KVSUM] rows parsed from {a.log}")
        sys.exit(2)

    total_layers = 0
    diff_layers = 0
    suspect_layers = 0
    missing = 0
    for i in requests:
        pop_layers = data[i].get("pop", {}).get("write", {})
        rep_layers = data[i].get("replay", {}).get("write", {})
        layers = sorted(set(pop_layers) | set(rep_layers))
        if not layers:
            print(f"req {i}: no write digests (pop/replay missing)")
            continue
        for layer in layers:
            pw = pop_layers.get(layer)
            rw = rep_layers.get(layer)
            total_layers += 1
            if pw is None or rw is None:
                missing += 1
                print(f"  req {i} layer {layer}: MISSING "
                      f"(pop={pw is not None}, replay={rw is not None})")
                continue
            result, detail = _cmp_vals(pw, rw)
            if result == "DIFF":
                diff_layers += 1
                print(f"  req {i} layer {layer}: DIFF [{detail}]")
            elif result == "SUSPECT":
                suspect_layers += 1
                print(f"  req {i} layer {layer}: SUSPECT [{detail}]")
            # OK silent

    print("-" * 60)
    print(f"requests={len(requests)} layers_compared={total_layers} "
          f"diffs={diff_layers} suspect={suspect_layers} missing={missing}")
    if diff_layers:
        print("RESULT: some layers' cached KV differ between populate and replay")
        sys.exit(1)
    if suspect_layers:
        print("RESULT: consistent but SUSPECT (inf/nan) values in "
              f"{suspect_layers} layers")
        sys.exit(2)
    if missing:
        print("RESULT: all compared layers consistent; some entries missing")
        sys.exit(1)
    print("RESULT: ALL layers' cached KV identical between populate and replay")
    sys.exit(0)


if __name__ == "__main__":
    main()

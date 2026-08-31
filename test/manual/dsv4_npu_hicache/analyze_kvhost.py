#!/usr/bin/env python3
"""Compare host-side C4/C128 digests: backup (pre-Mooncake) vs prefetch (post-Mooncake).

backup_digest == prefetch_digest  -> Mooncake round-trip bitwise correct;
                                     corruption is in D2H (backup side) or H2D (prefetch side)
backup_digest != prefetch_digest  -> Mooncake write or read corrupts data

Usage:
  python3 -u analyze_kvhost.py <server.log>

The log must contain [KVHOST] lines produced with SGLANG_DSV4_KV_DIGEST=1.
"""
import re
import sys
from collections import defaultdict

print = __import__("functools").partial(print, flush=True)

_LINE = re.compile(
    r"\[KVHOST\] phase=(?P<phase>\S+) pool=(?P<pool>\S+) "
    r"key0=(?P<key0>\S+) nkeys=(?P<nkeys>\d+) digest=(?P<digest>\S+)"
)

# prefetch phase names for KV-derived (C4) and sidecar (C128)
_PREFETCH_PHASES = {"prefetch_kv", "prefetch_sidecar"}


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_kvhost.py <server.log>")
        sys.exit(1)

    # records[(pool, key0)] = {"backup": digest, "prefetch": digest}
    # keep last occurrence per (pool, key0, phase)
    records: dict = defaultdict(dict)

    with open(sys.argv[1], errors="ignore") as f:
        for line in f:
            m = _LINE.search(line)
            if not m:
                continue
            phase = m.group("phase")
            pool = m.group("pool")
            key0 = m.group("key0")
            digest = m.group("digest")
            bucket = "prefetch" if phase in _PREFETCH_PHASES else phase
            records[(pool, key0)][bucket] = digest

    if not records:
        print("No [KVHOST] lines found in log.")
        sys.exit(2)

    total = diffs = errors = missing = 0
    by_pool: dict = defaultdict(lambda: {"diffs": 0, "total": 0})

    for (pool, key0), phases in sorted(records.items()):
        backup = phases.get("backup")
        prefetch = phases.get("prefetch")
        if backup is None or prefetch is None:
            missing += 1
            continue
        total += 1
        by_pool[pool]["total"] += 1
        if backup.startswith("ERR") or prefetch.startswith("ERR"):
            errors += 1
            print(f"  ERR  pool={pool} key0={key0} backup={backup} prefetch={prefetch}")
        elif backup != prefetch:
            diffs += 1
            by_pool[pool]["diffs"] += 1
            print(f"  DIFF pool={pool} key0={key0} backup={backup} prefetch={prefetch}")

    print("-" * 60)
    for pool, stats in sorted(by_pool.items()):
        print(f"  {pool}: {stats['diffs']}/{stats['total']} diffs")
    print(f"\npairs_compared={total} diffs={diffs} errors={errors} missing_half={missing}")

    if diffs:
        print(
            "\nRESULT: Mooncake corrupts data in write or read path\n"
            "  -> bug is between host pool and Mooncake (batch_set_v2 or batch_get_v2)"
        )
        sys.exit(1)
    elif total == 0:
        print("\nRESULT: no backup/prefetch pairs to compare (check log filtering)")
        sys.exit(2)
    else:
        print(
            "\nRESULT: Mooncake round-trip bitwise correct\n"
            "  -> corruption is in D2H (device->host before backup)\n"
            "  or H2D (host->device after prefetch)"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()

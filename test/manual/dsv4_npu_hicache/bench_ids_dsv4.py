#!/usr/bin/env python3
"""DSV4 perf-test single-cell driver (bench_ids adaptation).

Runs ONE matrix cell against a RUNNING server (:30000, pin L1=384K/rank):

  1. generate N deterministic-random DISTINCT prefixes (seed_base+i, first token
     distinct -> no shared radix prefix; pure function of seed -> populate /
     measure / replay regenerate byte-identical inputs);
  2. populate (concurrent, index-order) capturing first-round output_ids;
  3. measure via NATIVE sglang.bench_serving (monkeypatched get_dataset) for
     TTFT / Input TPS (no self-computed stats);
  4. correctness replay (post-measure): re-send each populate input, compare
     output token-by-token with the first-round output;
  5. hit verification: sum #new-token/#cached-token from server.log within the
     measure window.

Routing: --route free (default; both populate and measure omit routed_dp_rank,
server round-robin assigns by arrival; populate & measure submit in index order
so a replayed prefix lands on the same rank). --route roundrobin sets
routed_dp_rank=i%dp_ranks on populate/measure/replay; use only if
free-routing hits fail.

Usage (inside container cx-dsv4):
  python3 -u bench_ids_dsv4.py --input-len 32768 --num-prompts 282 \
      --output-len 1 --concurrency 16 --tag W1.5_32K_c16 \
      --server-log <path> --seed-base 60000 --model-path <model-dir>
  # no-cache baseline (C3): --skip-populate
  # stage split (independent): --populate-only / --measure-only /
  # --skip-populate / --skip-measure / --skip-replay — any combination runs
  # only the wanted stages (cold: --skip-populate --skip-replay).
Kill policy: NEVER kills the operator's server; on failure leaves it running and
prints the log path + manual-stop hint, then exits non-zero.
"""
import argparse
import functools
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import sglang.bench_serving as bs
import sglang.benchmark.serving as serving
from sglang.benchmark.datasets.common import DatasetRow

print = functools.partial(print, flush=True)

G = 128 * 16  # 2048 raw tokens per complete C128 group
MODEL = "/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp"
BASE = "http://127.0.0.1:30000"


def _post_generate(input_ids, rid: str, max_new_tokens: int,
                   routed_dp_rank, timeout: float = 900.0):
    """POST /generate; return (output_ids, cached_tokens). Raises on failure."""
    payload = {
        "rid": rid,
        "input_ids": [int(x) for x in input_ids],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": False,
    }
    if routed_dp_rank is not None:
        payload["routed_dp_rank"] = int(routed_dp_rank)
    req = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except Exception as e:
        raise RuntimeError(f"request failed: {e!r}") from e
    try:
        result = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"bad json: {e!r} body={body[:300]!r}") from e
    meta = result.get("meta_info", {})
    return result.get("output_ids", []), meta.get("cached_tokens", 0)


def _make_ids(vocab_size, special, seed, target):
    """Deterministic random distinct ids (pure function of seed)."""
    rng = random.Random(seed)
    first = seed % vocab_size
    while first in special:
        first += 1  # deterministic; just shifts the first token
    ids = [first]
    while len(ids) < target:
        t = rng.randrange(vocab_size)
        if t not in special:
            ids.append(t)
    return ids


def _read_log_lines(server_log, needle, start_line=0):
    out = []
    with open(server_log, errors="ignore") as f:
        for k, line in enumerate(f):
            if k < start_line:
                continue
            if needle in line:
                out.append(line)
    return out


def _log_line_count(server_log):
    try:
        with open(server_log, errors="ignore") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _hit_from_log(server_log, start_line):
    """Sum #new-token/#cached-token in server.log lines >= start_line."""
    new = cac = 0
    for line in _read_log_lines(server_log, "#cached-token", start_line):
        m = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
        if m:
            new += int(m.group(1))
            cac += int(m.group(2))
    return new, cac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-len", type=int, required=True,
                    help="group-aligned cached prefix length P = N*G (request "
                         "input = P+1 for hit=100; = P+Q for hit=50).")
    ap.add_argument("--model-path", required=True,
                    help="local model/tokenizer directory")
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True, help="1 or 1024")
    ap.add_argument("--cmp-len", type=int, default=None,
                    help="how many leading output tokens to compare in replay "
                         "(default: same as --output-len)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--hit", type=int, choices=[50, 100], default=100)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--seed-base", type=int, default=60000)
    ap.add_argument("--route", choices=["free", "roundrobin"], default="free")
    ap.add_argument("--dp-ranks", type=int, default=16,
                    help="backend DP rank count for roundrobin routing (default: 16)")
    ap.add_argument("--pop-conc", type=int, default=8,
                    help="concurrency of the populate loop")
    ap.add_argument("--skip-populate", action="store_true", help="C3 no-cache")
    ap.add_argument("--populate-only", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--skip-measure", action="store_true",
                    help="skip measure (native bench) and its hit verification; "
                         "populate then replay only, to isolate measure's "
                         "interference on replay")
    ap.add_argument("--skip-replay", action="store_true",
                    help="skip the correctness replay entirely")
    ap.add_argument("--save-first-out", type=str, default=None,
                    help="save populate output_ids to this JSON file")
    ap.add_argument("--load-first-out", type=str, default=None,
                    help="load populate output_ids from this JSON file for replay")
    a = ap.parse_args()
    cmp_len = a.output_len if a.cmp_len is None else a.cmp_len
    if a.output_len <= 0:
        ap.error("--output-len must be positive")
    if a.dp_ranks <= 0:
        ap.error("--dp-ranks must be positive")
    if not 1 <= cmp_len <= a.output_len:
        ap.error("--cmp-len must satisfy 1 <= cmp-len <= output-len")

    # ---- input generation (deterministic distinct) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    vocab = tok.vocab_size
    special = set(tok.all_special_ids or [])
    N = a.num_prompts
    P = (a.input_len // G) * G  # group-aligned full request length (measure)
    assert P == a.input_len, f"input_len {a.input_len} not a multiple of G={G}"
    # --hit controls the populate prefix length. Measure still sends the full
    # input length, so hit=50 means half cached and half new.
    pop_len = ((P * a.hit) // 100) // G * G
    seeds = [a.seed_base + i for i in range(N)]
    prefixes = [_make_ids(vocab, special, s, P) for s in seeds]
    pop_inputs = [pref[:pop_len] for pref in prefixes]
    if a.hit == 100:
        # N*G + 1: the extra token absorbs sglang's longest-hit-minus-1, so the
        # cached portion is exactly N*G (all complete groups).
        meas_inputs = [pref + [pref[-1]] for pref in prefixes]
    else:  # 50: first half cached prefix + second half new suffix
        soff = a.seed_base + 10_000_000
        suffixes = [_make_ids(vocab, special, soff + i, P - pop_len) for i in range(N)]
        meas_inputs = [
            pref[:pop_len] + suf for pref, suf in zip(prefixes, suffixes)
        ]
    ranks = (
        [i % a.dp_ranks for i in range(N)]
        if a.route == "roundrobin"
        else [None] * N
    )
    first_out: list = [None] * N

    def _populate():
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=a.pop_conc) as ex:
            futs = [
                ex.submit(_post_generate, pop_inputs[i], f"{a.tag}-pop-{i}",
                          a.output_len, ranks[i])
                for i in range(N)
            ]
            for i, f in enumerate(futs):
                try:
                    out, _ = f.result()
                    first_out[i] = out
                except Exception as e:
                    print(f"  [populate] [{i}] FAILED: {e!r}")
                    raise
                if (i + 1) % a.pop_conc == 0:
                    print(f"  [populate] {i + 1}/{N} done ({time.perf_counter() - t0:.0f}s)")
        print(f"  [populate] {N} prefixes done in {time.perf_counter() - t0:.0f}s")

    def _measure():
        # roundrobin: pin routed_dp_rank on the native-bench measure too, so each
        # request lands on the same rank that populated it (free routing scatters
        # concurrent arrivals -> measure cached=0). extra_request_body merges into
        # the top-level /generate payload, same as _post_generate's field.
        rows = [
            DatasetRow(
                prompt=ids,
                prompt_len=len(ids),
                output_len=a.output_len,
                extra_request_body=(
                    {"routed_dp_rank": i % a.dp_ranks}
                    if a.route == "roundrobin"
                    else {}
                ),
            )
            for i, ids in enumerate(meas_inputs)
        ]
        # Patch the name run_benchmark actually resolves: sglang.benchmark.serving
        # imports get_dataset into ITS module namespace (bench_serving is a
        # star-import shim, so assigning bs.get_dataset has no effect there).
        serving.get_dataset = lambda args, tokenizer, model_id=None: rows
        args = SimpleNamespace(
            backend="sglang", base_url=BASE, host="127.0.0.1", port=30000,
            model=a.model_path, served_model_name=None, tokenizer=None,
            dataset_name="custom", dataset_path="", num_prompts=N,
            request_rate=float("inf"), max_concurrency=a.concurrency, seed=1,
            ready_check_timeout_sec=60,
            disable_tqdm=True, disable_stream=False, disable_ignore_eos=False,
            extra_request_body=None, lora_name=None,
            lora_request_distribution="uniform", lora_zipf_alpha=1.5,
            profile=False, profile_start_step=None, profile_steps=None,
            pd_separated=False, flush_cache=False, warmup_requests=0,
            output_file=f"result_{a.tag}.jsonl", output_details=False,
            tag=a.tag, apply_chat_template=False, tokenize_prompt=False,
            cache_report=False, plot_throughput=False,
            image_count=1, image_resolution="1080p",
            random_input_len=1024, random_output_len=1024, random_range_ratio=0.0,
            sharegpt_output_len=None, return_logprob=False,
            return_routed_experts=False, top_logprobs_num=0,
            token_ids_logprob=None, logprob_start_len=-1,
            temperature=0.0, top_p=1.0, use_trace_timestamps=False,
            mooncake_slowdown_factor=1.0, mooncake_num_rounds=1)
        bs.run_benchmark(args)

    def _replay():
        bad = 0
        diverged = []
        with ThreadPoolExecutor(max_workers=a.pop_conc) as ex:
            futs = [
                ex.submit(_post_generate, pop_inputs[i], f"{a.tag}-replay-{i}",
                          a.output_len, ranks[i])
                for i in range(N)
            ]
            for i, f in enumerate(futs):
                try:
                    out, cac = f.result()
                except Exception as e:
                    print(f"  [replay] [{i}] FAILED: {e!r}")
                    raise
                first_cmp = (
                    first_out[i][:cmp_len] if first_out[i] is not None else None
                )
                replay_cmp = out[:cmp_len]
                if first_cmp is not None and (
                    len(first_cmp) != cmp_len
                    or len(replay_cmp) != cmp_len
                    or replay_cmp != first_cmp
                ):
                    bad += 1
                    if len(diverged) < 5:
                        idx = next(
                            (
                                k
                                for k, (first_token, replay_token) in enumerate(
                                    zip(first_cmp, replay_cmp)
                                )
                                if first_token != replay_token
                            ),
                            -1,
                        )
                        diverged.append((i, idx, first_cmp, replay_cmp, cac))
        for i, idx, first_cmp, replay_cmp, cac in diverged:
            print(f"  [replay] [{i}] DIVERGED at tok#{idx}: first={first_cmp} "
                  f"replay={replay_cmp} cached={cac}")
        print(f"  [replay] {N - bad}/{N} identical (cached check done)")
        if bad:
            print(f"  [replay] NOTE: {bad} outputs diverged (recorded, cell continues)")

    if not a.skip_populate and not a.measure_only:
        _populate()
        if a.save_first_out:
            with open(a.save_first_out, "w") as output_file:
                json.dump(first_out, output_file)
            print(f"  [first_out] saved {len(first_out)} entries -> "
                  f"{a.save_first_out}")
    if a.load_first_out is not None and not a.populate_only:
        with open(a.load_first_out) as input_file:
            loaded = json.load(input_file)
        for i in range(N):
            if i < len(loaded):
                first_out[i] = loaded[i]
        print(f"  [first_out] loaded {len(loaded)} entries from {a.load_first_out}")
    if not a.populate_only and not a.skip_measure:
        log_measure_start = _log_line_count(a.server_log)
        _measure()
        new, cac = _hit_from_log(a.server_log, log_measure_start)
        tot = new + cac
        hit = cac * 100.0 / tot if tot else 0.0
        print(f"  [verify] measure-window new={new} cached={cac} "
              f"achieved_hit={hit:.1f}%")
    if not a.populate_only and not a.skip_replay:
        _replay()

    print("  CELL DONE (server LEFT RUNNING)")


if __name__ == "__main__":
    main()

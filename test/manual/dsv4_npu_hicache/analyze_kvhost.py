#!/usr/bin/env python3
"""Check D2H/H2D copies and compare host bytes across Mooncake PUT/GET.

Usage:
  python3 -u analyze_kvhost.py <server.log>

Enable logs with SGLANG_DSV4_KV_DIGEST=1.
"""

import re
import sys
from collections import defaultdict

_LINE = re.compile(
    r"\[KVHOST\] phase=(?P<phase>put|get) rank=(?P<rank>\d+) "
    r"pp=(?P<pp>\d+) pool=(?P<pool>\S+) key=(?P<key>\S+) "
    r"action=(?P<action>\S+) bytes=(?P<bytes>\d+) "
    r"result=(?P<result>-?\d+) digest=(?P<digest>\S+)"
)
_COPY_LINE = re.compile(
    r"\[KVCOPY\] direction=(?P<direction>D2H|H2D) pool=(?P<pool>\S+) "
    r"pages=(?P<pages>\d+) src_rows=(?P<src_rows>\S+) "
    r"dst_rows=(?P<dst_rows>\S+) bytes=(?P<bytes>\d+) "
    r"src=(?P<src>\S+) dst=(?P<dst>\S+) match=(?P<match>[01])"
)
_POOL_LINE = re.compile(
    r"\[KVPOOL\] phase=(?P<phase>put|get) rank=(?P<rank>\d+) "
    r"pp=(?P<pp>\d+) pool=(?P<pool>\S+) key_sig=(?P<key_sig>\S+) "
    r"keys=(?P<keys>\d+) rows=(?P<rows>\S+) bytes=(?P<bytes>\d+) "
    r"digest=(?P<digest>\S+)"
)


def main():
    if len(sys.argv) != 2:
        print("usage: analyze_kvhost.py <server.log>")
        sys.exit(2)

    puts = {}
    skipped = set()
    gets = []

    with open(sys.argv[1], errors="ignore") as log_file:
        for line in log_file:
            match = _LINE.search(line)
            if not match:
                continue
            record = match.groupdict()
            identity = (
                record["rank"],
                record["pp"],
                record["pool"],
                record["key"],
            )
            value = (int(record["bytes"]), record["digest"])
            if record["phase"] == "put":
                if record["action"] == "created":
                    puts.setdefault(identity, value)
                elif record["action"] == "skipped_exists":
                    skipped.add(identity)
            else:
                gets.append(
                    (
                        identity,
                        record["action"],
                        int(record["result"]),
                        value,
                    )
                )

    if not puts and not gets:
        print("No new-format [KVHOST] lines found.")
        sys.exit(2)

    compared = 0
    diffs = 0
    io_errors = 0
    unverified = 0
    by_pool = defaultdict(lambda: [0, 0])
    copy_results = []
    pool_puts = {}
    pool_gets = []

    with open(sys.argv[1], errors="ignore") as log_file:
        for line in log_file:
            match = _COPY_LINE.search(line)
            if match:
                copy_results.append(match.groupdict())
            match = _POOL_LINE.search(line)
            if match:
                record = match.groupdict()
                identity = (
                    record["rank"], record["pp"], record["pool"],
                    record["key_sig"],
                )
                value = (
                    int(record["keys"]), int(record["bytes"]), record["digest"]
                )
                if record["phase"] == "put":
                    pool_puts.setdefault(identity, value)
                else:
                    pool_gets.append((identity, value))

    for identity, action, result, get_value in gets:
        rank, pp, pool, key = identity
        byte_count, digest = get_value
        if action != "read" or result != byte_count or digest.startswith("ERR:"):
            io_errors += 1
            print(
                f"IO_ERROR rank={rank} pp={pp} pool={pool} key={key} "
                f"action={action} bytes={byte_count} result={result} digest={digest}"
            )
            continue

        put_value = puts.get(identity)
        if put_value is None:
            unverified += 1
            reason = "skipped_exists" if identity in skipped else "missing_put"
            print(
                f"UNVERIFIED rank={rank} pp={pp} pool={pool} "
                f"key={key} reason={reason}"
            )
            continue

        compared += 1
        by_pool[pool][1] += 1
        if put_value != get_value:
            diffs += 1
            by_pool[pool][0] += 1
            print(
                f"DIFF rank={rank} pp={pp} pool={pool} key={key} "
                f"put={put_value} get={get_value}"
            )

    print("-" * 72)
    for pool, (pool_diffs, pool_total) in sorted(by_pool.items()):
        print(f"{pool}: {pool_diffs}/{pool_total} diffs")
    print(
        f"compared={compared} diffs={diffs} io_errors={io_errors} "
        f"unverified={unverified}"
    )

    copy_mismatches = 0
    copy_by_pool = defaultdict(lambda: [0, 0])
    copy_directions = {record["direction"] for record in copy_results}
    if copy_results:
        print("-" * 72)
        for record in copy_results:
            identity = (record["direction"], record["pool"])
            copy_by_pool[identity][1] += 1
            if record["match"] != "1":
                copy_mismatches += 1
                copy_by_pool[identity][0] += 1
                print(
                    f"COPY_DIFF direction={record['direction']} "
                    f"pool={record['pool']} pages={record['pages']} "
                    f"src_rows={record['src_rows']} dst_rows={record['dst_rows']} "
                    f"src={record['src']} dst={record['dst']}"
                )
        for (direction, pool), (bad, total) in sorted(copy_by_pool.items()):
            print(f"{direction} {pool}: {bad}/{total} copy diffs")
        print(
            f"copy_compared={len(copy_results)} copy_diffs={copy_mismatches}"
        )

    pool_compared = 0
    pool_mismatches = 0
    pool_unverified = 0
    kvpool_by_pool = defaultdict(lambda: [0, 0])
    if pool_gets:
        print("-" * 72)
        for identity, get_value in pool_gets:
            rank, pp, pool, key_sig = identity
            put_value = pool_puts.get(identity)
            if put_value is None:
                pool_unverified += 1
                print(
                    f"POOL_UNVERIFIED rank={rank} pp={pp} pool={pool} "
                    f"key_sig={key_sig}"
                )
                continue
            pool_compared += 1
            kvpool_by_pool[pool][1] += 1
            if put_value != get_value or get_value[2].startswith("ERR:"):
                pool_mismatches += 1
                kvpool_by_pool[pool][0] += 1
                print(
                    f"POOL_DIFF rank={rank} pp={pp} pool={pool} "
                    f"key_sig={key_sig} put={put_value} get={get_value}"
                )
        for pool, (bad, total) in sorted(kvpool_by_pool.items()):
            print(f"{pool}: {bad}/{total} full-pool diffs")
        print(
            f"pool_compared={pool_compared} pool_diffs={pool_mismatches} "
            f"pool_unverified={pool_unverified}"
        )

    if diffs or io_errors or copy_mismatches or pool_mismatches:
        print("RESULT: KV copy, Mooncake data, or I/O mismatch")
        sys.exit(1)
    if (
        compared == 0
        or unverified
        or pool_unverified
        or copy_directions != {"D2H", "H2D"}
        or not pool_gets
    ):
        if copy_directions != {"D2H", "H2D"}:
            print(f"MISSING KVCOPY directions: {sorted({'D2H', 'H2D'} - copy_directions)}")
        if not pool_gets:
            print("MISSING KVPOOL get snapshots")
        print("RESULT: inconclusive; use a fresh extra_backend_tag and rerun")
        sys.exit(2)
    print("RESULT: logged KV copy and host -> Mooncake -> host bytes are exact")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare per-key host bytes before Mooncake PUT and after Mooncake GET.

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

    if diffs or io_errors:
        print("RESULT: KVHOST mismatch or incomplete Mooncake I/O")
        sys.exit(1)
    if compared == 0 or unverified:
        print("RESULT: inconclusive; use a fresh extra_backend_tag and rerun")
        sys.exit(2)
    print("RESULT: host -> Mooncake -> host bytes are exact")


if __name__ == "__main__":
    main()

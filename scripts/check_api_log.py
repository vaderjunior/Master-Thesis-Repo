"""
scripts/check_api_log.py - integrity and attribution for the API call log.
Read-only, zero API calls.
Run: python -m scripts.check_api_log

WHY THIS EXISTS. The project's cost figure goes into the thesis, and it is
currently wrong in two ways at once. PHASE8_SIGNOFF Part 0 reports "80617 API
calls across the phase" - the same number a raw line count of this file gave
on 2026-08-14, which means it is the PROJECT total mislabelled as a PHASE
total. The handover and the Phase 7 build guide both carry ~35,000, which is
staler still. Part 0 already flags its own figure as arithmetic over the runs
rather than a transcribed count.

WHY IT IS AN INTEGRITY CHECK AND NOT JUST A COUNTER. On 2026-08-14 a naive
json.loads over every line raised JSONDecodeError, and a run of 927 votes plus
49 repairs (976 expected) added only 968 lines. Both point the same way: this
file is appended from 8 worker threads and a torn or interleaved write loses
or corrupts a line. Nothing downstream depends on it - attribution lives on
the ItemResult stamps, and every analysis script reads those - so no result is
affected. But a cost figure quoted from a file with known torn writes has to
be quoted as a lower bound, and that is only possible if the corruption is
counted rather than crashed on.

The schema changed mid-project: early lines have no `provider` key. Anything
grouping by provider must tolerate its absence rather than KeyError on the
first line of the file.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LOG = Path("experiments/results/api_log.jsonl")

# Phase boundaries as recorded in the sign-offs. Used only for attribution of
# the call count; nothing else reads them.
# The Phase 8 sign-off says "working sessions of 2026-08-06 to 2026-08-11",
# but 14,526 calls landed on 2026-08-12. By content those are the section 3.12
# KB re-runs (de_dev / de_legal / severity / types under kbv3) and the section
# 3.13 new-label runs - both Phase 8 sections, written up in the sign-off,
# executed the day after it was dated. Arithmetic: 2700 + 3150 + 810 + 1854 +
# 5604 = 14118 plus repairs. Phase 7's first call is 2026-08-14.
PHASES = [
    ("Phase 8 (SQ3, KB re-runs, new label)", "2026-08-06", "2026-08-13"),
    ("Phase 7 (encoder baselines)", "2026-08-14", "2099-01-01"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-bad", type=int, default=5)
    args = ap.parse_args()

    ok, bad = [], []
    total = 0
    for i, line in enumerate(LOG.read_text(encoding="utf-8",
                                           errors="replace").splitlines()):
        if not line.strip():
            continue
        total += 1
        try:
            ok.append(json.loads(line))
        except (ValueError, TypeError):
            bad.append((i + 1, line))

    print(f"\n{'=' * 74}\nAPI LOG INTEGRITY   {LOG}\n{'=' * 74}")
    print(f"  non-empty lines   {total}")
    print(f"  parsed            {len(ok)}")
    print(f"  CORRUPT           {len(bad)}")
    if bad:
        print("\n  Torn or interleaved writes. The log is appended from 8")
        print("  worker threads; if that append is not lock-protected, two")
        print("  threads can write into the same buffer flush. No result is")
        print("  affected - attribution lives on the ItemResult stamps - but")
        print("  any call total from this file is a LOWER BOUND.")
        for ln, txt in bad[:args.show_bad]:
            print(f"    line {ln}: {txt[:160]!r}")

    dated = [r for r in ok if isinstance(r, dict) and "timestamp" in r]
    print(f"\n  with timestamp    {len(dated)}")
    if not dated:
        print("  no timestamps; per-phase attribution not possible")
        return

    days = Counter(datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d")
                   for r in dated)
    first = min(r["timestamp"] for r in dated)
    last = max(r["timestamp"] for r in dated)
    print(f"  first call        "
          f"{datetime.fromtimestamp(first).strftime('%Y-%m-%d %H:%M')}")
    print(f"  last call         "
          f"{datetime.fromtimestamp(last).strftime('%Y-%m-%d %H:%M')}")
    print(f"  span              {(last - first) / 86400:.1f} days over "
          f"{len(days)} active day(s)")

    print(f"\n{'-' * 74}\nCALLS PER DAY")
    for d in sorted(days):
        print(f"  {d}   {days[d]:>7}   {'#' * min(60, days[d] // 400)}")

    print(f"\n{'-' * 74}\nPER PHASE (by the dates recorded in the sign-offs)")
    for label, lo, hi in PHASES:
        n = sum(v for k, v in days.items() if lo <= k <= hi)
        print(f"  {label:34} {n:>8}   ({lo} to {hi})")
    earliest = min(days)
    pre = sum(v for k, v in days.items() if k < PHASES[0][1])
    print(f"  {'everything before Phase 8':34} {pre:>8}   "
          f"({earliest} to {PHASES[0][1]})")
    print(f"  {'PROJECT TOTAL (lower bound)':34} {len(ok):>8}")

    print(f"\n{'-' * 74}\nBY MODEL")
    for m, n in Counter(r.get("model", "?") for r in ok).most_common():
        print(f"  {str(m):44} {n:>8}")

    prov = Counter(r.get("provider", "(unstamped, pre-schema)") for r in ok)
    print(f"\n{'-' * 74}\nBY PROVIDER")
    for p, n in prov.most_common():
        print(f"  {str(p):44} {n:>8}")

    lat = [r["latency_s"] for r in ok if isinstance(r.get("latency_s"),
                                                    (int, float))]
    if lat:
        lat_s = sorted(lat)
        print(f"\n{'-' * 74}\nLATENCY over {len(lat)} calls")
        print(f"  median {lat_s[len(lat_s) // 2]:.2f}s   "
              f"p90 {lat_s[int(len(lat_s) * 0.9)]:.2f}s   "
              f"p99 {lat_s[int(len(lat_s) * 0.99)]:.2f}s   "
              f"max {lat_s[-1]:.2f}s")
        # NOT GPU-hours. This is the sum of per-call latency across 8
        # concurrent workers on a server doing continuous batching, so it
        # over-counts GPU time and over-counts wall clock by roughly 8x.
        # Quote it as "cumulative call latency" or divide by the worker count
        # for an approximate wall-clock figure.
        print(f"  cumulative call latency {sum(lat) / 3600:.1f} h "
              f"(~{sum(lat) / 3600 / 8:.1f} h wall clock at 8 workers)")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
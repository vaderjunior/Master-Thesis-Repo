"""
scripts/inspect_results.py - look at what the model actually returned.

Latency, output length, thinking traces, repairs, and the raw text of a few
runs. Everything here comes from the stored raw_runs; no API calls.

  python -m scripts.inspect_results experiments/results/slice1_en_dev_eval_main_live.jsonl
  python -m scripts.inspect_results <path> --show 3
"""

import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--show", type=int, default=2, help="raw outputs to print")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.path).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(rows)} item-arm results\n")

    lat = [x for r in rows for x in r["latencies"]]
    chars = [len(o) for r in rows for run in r["raw_runs"] for o in run]
    thinking = sum(1 for r in rows for run in r["raw_runs"] for o in run
                   if "<think>" in o.lower())
    fenced = sum(1 for r in rows for run in r["raw_runs"] for o in run
                 if "```" in o)

    print(f"  latency   mean {statistics.mean(lat):5.1f}s   "
          f"median {statistics.median(lat):5.1f}s   "
          f"min {min(lat):5.1f}s   max {max(lat):5.1f}s")
    print(f"  output    mean {statistics.mean(chars):5.0f} chars   "
          f"max {max(chars)} chars")
    print(f"  thinking traces  {thinking}/{len(chars)} outputs")
    print(f"  code fences      {fenced}/{len(chars)} outputs")
    print(f"  repairs          {sum(r['repairs'] for r in rows)}")
    print(f"  parse failures   {sum(r['parse_failures'] for r in rows)}")
    print(f"  normalisations   {sum(r['normalisations'] for r in rows)}")
    print(f"  gate normalised  {sum(r['gate_normalised'] for r in rows)}")
    print(f"  uncertain        {sum(1 for r in rows if r['uncertain'])}")
    print(f"  arms             { {r['arm'] for r in rows} }")

    print(f"\n{'=' * 72}\nRAW OUTPUTS\n{'=' * 72}")
    for r in rows[:args.show]:
        print(f"\n--- {r['item_id']}  [{r['arm']}] ---")
        for i, run in enumerate(r["raw_runs"]):
            print(f"  run {i} ({r['latencies'][i]}s):")
            for attempt in run:
                print("    " + attempt.replace("\n", "\n    ")[:900])
        print(f"  voted: {r['result']}")
        print(f"  agreement: {r['agreement']}")


if __name__ == "__main__":
    main()
"""
scripts/check_sentinel_flips.py - the over-steering measurement.
Read-only, zero API calls.
Run: python -m scripts.check_sentinel_flips --runs sq3_r1_fb sq3_r1_ctl

WHAT IT ASKS. The sentinel is 84 pool items the system got right in all three
round-0 replicates, frozen and never corrected. Does editing the knowledge
base break any of them? That is the cost side of feedback: a system that
learns one thing by forgetting another has not improved.

THE BASELINE IS THE WHOLE POINT. The sentinel was SELECTED on r1/r2/r3, so its
flip rate measured on those runs is zero by construction and means nothing.
sq3_round0_r4 is an independent replicate under the unchanged KB, and it
flipped 1 of 84 (micro 0.012, macro 0.019). Every post-edit count is read
against that, not against zero.

POWER, so a null is not over-read. At n=84 with a 1.2% baseline the 95%
interval runs to roughly 6%, i.e. about 5 flips. Below that the honest
statement is "no detectable over-steering at n=84", not "no over-steering".

MICRO AND MACRO. Membership is decided by what the model already gets right,
so the sentinel is unbalanced by construction: stereotyping 24, grievance 17,
inferiority 14, incitement 13, irony 7, explicit 6, threatening 4. A raw count
is weighted toward whichever labels the model already handles, so a per-label
rate averaged over labels at MIN_SUPPORT is reported beside it, exactly as
every other macro number in this project.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.check_sq3_coverage import PROCESSED, local_gold_sets
from src.hsrag.metrics import MIN_SUPPORT

RESULTS = Path("experiments/results")
SENTINEL = Path("experiments") / "sq3_sentinel_r0.json"
SUBSET = "en_dev_eval_sq3_types"
GOLD_COL, DIM = "hate_types", "hate_type"
BASELINE_RUN = "sq3_round0_r4"


def load(stem: str) -> dict:
    path = RESULTS / f"{stem}_live.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" in r or r.get("arm") != "rag":
            continue
        out[r["item_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", default=BASELINE_RUN)
    args = ap.parse_args()

    s = json.loads(SENTINEL.read_text(encoding="utf-8"))
    ids, support = s["sentinel_ids"], s["sentinel_label_support"]
    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    pool = df[df["sq3_role"] == "pool"].reset_index(drop=True)
    gold = local_gold_sets(pool, GOLD_COL)

    print(f"\n{'=' * 74}\nsentinel: {len(ids)} items, criterion={s['criterion']}"
          f", frozen under kb {s['kb_version'][:8]}")
    print(f"  selected on: {', '.join(s['runs'])}")
    print(f"  baseline:    {args.baseline} (independent of that selection)")

    averaged = sorted(l for l, n in support.items() if n >= MIN_SUPPORT)
    thin = sorted(l for l, n in support.items() if n < MIN_SUPPORT)

    rows = []
    for stem in [args.baseline] + [r for r in args.runs if r != args.baseline]:
        res = load(stem)
        missing = [i for i in ids if i not in res]
        assert not missing, f"{stem} is missing {len(missing)} sentinel items"

        flips = []
        for i in ids:
            r = res[i].get("result")
            pred = set((r or {}).get(DIM) or [])
            if not (gold[i] <= pred):          # `contains`, as frozen
                flips.append(i)
        fc = Counter(l for i in flips for l in gold[i])
        macro = (sum(fc.get(l, 0) / support[l] for l in averaged)
                 / len(averaged)) if averaged else 0.0
        rows.append((stem, len(flips), len(flips) / len(ids), macro, fc))

    print(f"\n  {'run':16} {'flips':>6} {'micro':>7} {'macro':>7}   "
          f"vs baseline")
    base_n = rows[0][1]
    for stem, n, micro, macro, _ in rows:
        delta = "" if stem == args.baseline else f"{n - base_n:+d} flips"
        print(f"  {stem:16} {n:6} {micro:7.3f} {macro:7.3f}   {delta}")

    print(f"\n  per-label flips (support in brackets, "
          f"MIN_SUPPORT={MIN_SUPPORT}):")
    print(f"  {'label':16} {'n':>4} " + " ".join(f"{r[0][-6:]:>10}"
                                                 for r in rows))
    for label in averaged + thin:
        mark = "" if label in averaged else "  (excluded from macro)"
        print(f"  {label:16} {support[label]:4} "
              + " ".join(f"{r[4].get(label, 0):10}" for r in rows) + mark)

    print(f"\n  Power: at n={len(ids)} with a {rows[0][2]:.3f} baseline, a run"
          f"\n  needs roughly {base_n + 5} or more flips to be distinguishable"
          f"\n  from no change. Below that, report 'no detectable"
          f"\n  over-steering at n={len(ids)}'.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
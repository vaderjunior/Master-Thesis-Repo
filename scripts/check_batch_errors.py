"""
scripts/check_batch_errors.py - per-batch correctness across runs.
Read-only, zero API calls.
Run: python -m scripts.check_batch_errors

WHY. Two things this answers that no other script does.

1. THE PRE-REGISTERED ERRORS-PER-BATCH TRAJECTORY. If feedback works at all,
   later batches should contain fewer errors. Comparing round n's batch to
   round n-1's confounds batch composition with KB state; comparing THE SAME
   batch under different KBs does not, and round 0 classified all 467 items so
   the comparison is free.

2. THE UNCORRECTED BATCHES ARE A CLEAN STRATUM NOBODY WAS USING. During round
   1 only b1 is in either KB, so b2, b3 and b4 - 84 items - are as
   never-corrected as held_out. score_by_role --role batches lumps all four
   together and is therefore contaminated by b1; split out, the rest add real
   power. The stratum shrinks by one batch per round, which is why the table
   marks which cells are contaminated rather than leaving it to memory.

CONTAMINATION IS INFERRED FROM THE RUN NAME, deliberately and visibly: a
sq3_r{n}_{arm} run used a KB carrying that arm's corrections from rounds
1..n, so batches b1..bn are seen for it. Inferring from the name is fragile,
so the marker is printed rather than silently applied, and a cell marked seen
should be read as a write-back sanity check and never as a result.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from scripts.apply_feedback import batches, load_results
from scripts.check_sq3_coverage import PROCESSED, local_gold_sets

SUBSET = "en_dev_eval_sq3_types"
GOLD_COL, DIM = "hate_types", "hate_type"
DEFAULT_RUNS = ["sq3_round0_r1", "sq3_round0_r2", "sq3_round0_r3",
                "sq3_round0_r4", "sq3_r1_fb", "sq3_r1_ctl",
                "sq3_r1_fb_k3", "sq3_r1_ctl_k3",
                "sq3_r2_fb", "sq3_r2_ctl",
                "sq3_r3_fb", "sq3_r3_ctl"]


def seen_through(stem: str) -> int:
    """Highest batch index already written into this run's KB, or 0."""
    if "round0" in stem:
        return 0
    m = re.search(r"_r(\d+)_", stem)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    bs = batches(df)
    res = {s: load_results(s) for s in args.runs}

    print(f"\n{'=' * 78}\ncontains-correct rate per batch")
    print(f"  * = that batch is IN this run's KB - a sanity check, not a "
          f"result\n")
    print(f"  {'batch':10} " + " ".join(f"{s.replace('sq3_', ''):>14}"
                                        for s in args.runs))

    for name, bd in sorted(bs.items()):
        gold = local_gold_sets(bd, GOLD_COL)
        idx = int(name[1:])
        cells = []
        for s in args.runs:
            r = res[s]
            miss = [i for i in gold if i not in r]
            assert not miss, f"{s} missing {len(miss)} items of {name}"
            ok = sum(1 for i in gold
                     if r[i]["result"]
                     and gold[i] <= set(r[i]["result"][DIM] or []))
            mark = "*" if idx <= seen_through(s) else " "
            cells.append(f"{ok:2}/{len(gold):2} {ok / len(gold):.3f}{mark}")
        print(f"  {name} n={len(gold):<4} " + " ".join(f"{c:>14}"
                                                       for c in cells))

    # The clean aggregate: every batch not yet written into that run's KB.
    print(f"\n  {'unseen':10} " + " ".join(f"{'':>14}" for _ in args.runs))
    for s in args.runs:
        tot = ok = 0
        for name, bd in bs.items():
            if int(name[1:]) <= seen_through(s):
                continue
            gold = local_gold_sets(bd, GOLD_COL)
            tot += len(gold)
            ok += sum(1 for i in gold
                      if res[s][i]["result"]
                      and gold[i] <= set(res[s][i]["result"][DIM] or []))
        print(f"    {s:22} {ok:3}/{tot:3}  {ok / max(tot, 1):.3f}  "
              f"(batches after b{seen_through(s)})")
    print(f"\n  Compare like with like: a round-1 run's unseen aggregate "
          f"covers\n  b2-b4 while a round-0 run's covers b1-b4, so read the "
          f"per-batch\n  rows for the paired comparison.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
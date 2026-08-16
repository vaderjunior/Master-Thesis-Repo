"""
scripts/check_sq3_consistency.py - Krippendorff's alpha per round.
Read-only, zero API calls.
Run: python -m scripts.check_sq3_consistency

WHY THIS EXISTS. The thesis locks two definitions: adaptability is delta
macro-F1 after knowledge-base edits with the LLM frozen, and consistency is
inter-run agreement plus alignment with the annotation guidelines. Every SQ3
number so far measures the first. SQ3 is the only experiment in the project
that can touch BOTH, because it holds items, model and prompt fixed while
changing only the knowledge base - so any movement in inter-run agreement is
attributable to the KB edit.

WHAT IT ASKS. Does grounding the model in its own corrections make it more
self-consistent? A system that scores the same but agrees with itself more is
better in a way macro-F1 cannot show, and one that scores the same while
agreeing with itself LESS has been destabilised by the edit.

IT COSTS NOTHING because Phase 5 stored every raw run rather than only the
vote. alpha here is computed across the three sampled runs of a single
classification, so it measures decoding stability under a fixed KB, not
agreement between KB versions.

UNITS follow analysis._alpha: the gate is one nominal unit per item; a
multilabel dimension is decomposed into one binary unit per (item, label)
pair, because alpha needs one value per unit per coder and a SET is not a
value; severity is ordinal.

READ IT AGAINST THE ROUND-0 SPREAD, not against an absolute threshold. Four
round-0 replicates under an unchanged KB give the range alpha wanders in when
nothing has been edited, and only movement outside that range is a KB effect.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from scripts.apply_feedback import load_results
from scripts.check_sq3_coverage import PROCESSED
from src.hsrag.analysis import consistency

SUBSET = "en_dev_eval_sq3_types"
BASELINE = ["sq3_round0_r1", "sq3_round0_r2", "sq3_round0_r3", "sq3_round0_r4"]
ROUNDS = [("r1", "sq3_r1_fb", "sq3_r1_ctl"),
          ("r2", "sq3_r2_fb", "sq3_r2_ctl"),
          ("r3", "sq3_r3_fb", "sq3_r3_ctl"),
          ("r4", "sq3_r4_fb", "sq3_r4_ctl")]
ABLATION = [("r1 k=3", "sq3_r1_fb_k3", "sq3_r1_ctl_k3")]
# Three replicates of each arm at the round-4 KB state. Every comparison above
# is between arms whose KBs differ; these differ only in decoding, so they are
# the within-condition variance alpha has lacked. The macro-F1 and sentinel
# gaps both vanished under exactly this test.
REPLICATES = [("r4 rep2", "sq3_r4_fb_rep2", "sq3_r4_ctl_rep2"),
              ("r4 rep3", "sq3_r4_fb_rep3", "sq3_r4_ctl_rep3")]
DIMS = ["gate", "hate_type", "target_group", "severity"]


def taxonomy_labels() -> dict:
    tax = yaml.safe_load(
        Path("config/taxonomy.yaml").read_text(encoding="utf-8"))["dimensions"]
    return {d: list(tax[d]["labels"]) for d in
            ("target_group", "hate_type", "legal")}


def fmt(cell: dict) -> str:
    """alpha=None means perfect agreement (expected disagreement is zero, so
    alpha is undefined). Printing 0.0 there would read as total disagreement
    when the truth is the opposite."""
    a = cell.get("alpha")
    return "  1.000*" if a is None else f"{a:8.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="fixed",
                    choices=["fixed", "all"],
                    help="fixed = held_out + pool, the 350 items never "
                         "corrected in any round, matching the learning curve")
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    if args.role == "fixed":
        keep = set(df.loc[df["sq3_role"].isin(["held_out", "pool"]),
                          "id"].astype(str))
    else:
        keep = set(df["id"].astype(str))
    labels = taxonomy_labels()

    print(f"\n{'=' * 78}\nKrippendorff's alpha across the {args.role} stratum "
          f"({len(keep)} items), n_votes=3")
    print(f"  * = perfect agreement, alpha undefined\n")

    def alpha_for(stem: str) -> dict:
        rows = [r for i, r in load_results(stem).items() if i in keep]
        return consistency(rows, labels)

    print(f"  {'run':16} " + " ".join(f"{d:>12}" for d in DIMS))

    base = {}
    for stem in BASELINE:
        c = alpha_for(stem)
        base[stem] = c
        print(f"  {stem:16} " + " ".join(f"{fmt(c[d]):>12}" for d in DIMS))

    ranges = {}
    for d in DIMS:
        vals = [c[d]["alpha"] for c in base.values() if c[d]["alpha"] is not None]
        ranges[d] = (min(vals), max(vals)) if vals else (None, None)
    print(f"  {'round-0 range':16} " + " ".join(
        f"{(f'{lo:.3f}-{hi:.3f}' if lo is not None else '-'):>12}"
        for lo, hi in (ranges[d] for d in DIMS)))

    print()
    for tag, fb, ctl in ROUNDS + ABLATION + REPLICATES:
        for arm, stem in (("fb", fb), ("ctl", ctl)):
            c = alpha_for(stem)
            marks = []
            for d in DIMS:
                a, (lo, hi) = c[d]["alpha"], ranges[d]
                out = fmt(c[d])
                # Flag only what sits outside the range an unchanged KB
                # produces. Anything inside it is decoding noise.
                if a is not None and lo is not None and (a < lo or a > hi):
                    out = out.strip() + ("v" if a < lo else "^")
                marks.append(out)
            print(f"  {tag + ' ' + arm:16} " + " ".join(f"{m:>12}"
                                                        for m in marks))

    print(f"\n  ^ above / v below the round-0 range. Anything inside that")
    print(f"  range is decoding noise, not a knowledge-base effect.")
    print(f"\n  Note: target_group and severity have no gold on this subset,")
    print(f"  so their alpha measures prediction stability only - the model")
    print(f"  still emits them under full-schema output, and a KB edit that")
    print(f"  destabilised an unrelated dimension would show up here.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
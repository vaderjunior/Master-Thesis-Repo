"""
scripts/score_by_role.py - re-score existing results on one sq3_role.
Read-only, zero API calls.
Run: python -m scripts.score_by_role --role held_out

WHY. The learning curve is macro-F1 on the held_out role, not on the whole
subset. Those are different measurements with different noise floors: the
467-item subset carries 68 gold positives per label, held_out carries ~17.
The per-dimension variance floor finding says fewer positives per label means
a flipped prediction moves the macro average further, so the floor that
actually governs SQ3 is the held_out floor and it is not the one printed by
score_run over the full subset.

Round 0 was run on all 467 items, so this needs no new calls: score_all skips
any result whose item_id is absent from the gold frame, so filtering the FRAME
by role filters the scoring population with no change to the results file.

WHY IT REPORTS PER-LABEL SPREAD. On the full subset the macro spread across
the three round-0 replicates is 0.021 while `threatening` alone spans 0.125.
The macro average hides label-level noise, and any per-label claim - which is
what a per-label recovery experiment makes - has to be read against a
per-label floor.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.score_run import report
from src.hsrag.metrics import score_all

RESULTS = Path("experiments/results")
PROCESSED = Path("data/processed")
DIM = "hate_type"


def load_results(stem: str) -> list:
    path = RESULTS / f"{stem}_live.jsonl"
    return [r for r in (json.loads(l) for l
                        in path.read_text(encoding="utf-8").splitlines()
                        if l.strip())
            if "_manifest" not in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="en_dev_eval_sq3_types")
    ap.add_argument("--role", default="held_out",
                    help="sq3_role to score, 'all' for the whole subset, or "
                         "'clean' for every item not yet in the KB at --round")
    ap.add_argument("--round", type=int, default=0,
                    help="with --role clean: batches b1..bN are excluded as "
                         "corrected; everything later is still unseen")
    ap.add_argument("--runs", nargs="*",
                    default=["sq3_round0_r1", "sq3_round0_r2",
                             "sq3_round0_r3"])
    ap.add_argument("--arm", default="rag")
    ap.add_argument("--full-report", action="store_true",
                    help="print score_run's full report for each run too")
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"{args.subset}.parquet")
    if args.role == "fixed":
        # held_out + pool: the 350 items never corrected in ANY round, so the
        # learning curve is measured on identical items every time. The
        # `clean` stratum shrinks as batches are corrected (434 -> 406 -> 378),
        # which means a round-over-round comparison on `clean` compares
        # different item sets and confounds learning with composition.
        df = df[df["sq3_role"].isin(["held_out", "pool"])].reset_index(drop=True)
    elif args.role == "clean":
        # THE PRIMARY MEASUREMENT. held_out and pool are never corrected, and
        # so is every batch not yet written into the KB - during round 1 that
        # is b2-b4, 84 more items. Reading one stratum alone is unreliable at
        # this scale: on round 1 the three clean strata gave fb - ctl of
        # +0.017, +0.022 and -0.072 on contains, which pool to a one-item
        # difference over 434. The stratum-level swing is noise, not signal,
        # so the pooled figure is the one to report.
        from scripts.apply_feedback import batches
        keep = set(df.loc[df["sq3_role"].isin(["held_out", "pool"]), "id"])
        for name, bd in batches(df).items():
            if int(name[1:]) > args.round:
                keep |= set(bd["id"])
        df = df[df["id"].isin(keep)].reset_index(drop=True)
    elif args.role != "all":
        assert "sq3_role" in df.columns, f"{args.subset} has no sq3_role"
        df = df[df["sq3_role"] == args.role].reset_index(drop=True)
    assert len(df), f"no items for role={args.role}"

    print(f"\n{'=' * 74}\n{args.subset} [role={args.role}]: {len(df)} items, "
          f"arm={args.arm}")

    rows, per_label = {}, {}
    for stem in args.runs:
        results = load_results(stem)
        # gate_mapping=strict: the subset is single-class so the gate number is
        # undefined anyway, and asking for 'both' would print two undefined
        # numbers instead of one.
        scores = score_all(df, results, gate_mapping="strict")
        assert args.arm in scores, f"{stem} has no {args.arm} arm"
        ht = scores[args.arm][DIM]
        rows[stem] = {
            "n": ht["n_items"],
            "macro_f1": ht["macro_f1"],
            "micro_f1": ht["micro_f1"],
            "exact": ht["subset_accuracy"],
            "contains": ht["extra"]["gold_subset_of_pred"],
            "pred_labels": ht["extra"]["mean_pred_labels"],
            "excluded": ht["labels_excluded"],
        }
        per_label[stem] = {l: v["f1"] for l, v in ht["per_label"].items()}
        if args.full_report:
            print(f"\n--- {stem} ---")
            report(scores)

    # ------------------------------------------------------------- headline
    print(f"\n  {'run':22} {'n':>4} {'macro':>7} {'micro':>7} {'exact':>7} "
          f"{'contains':>9} {'pred':>6}")
    for stem, r in rows.items():
        print(f"  {stem:22} {r['n']:4} {r['macro_f1']:7.3f} "
              f"{r['micro_f1']:7.3f} {r['exact']:7.3f} {r['contains']:9.3f} "
              f"{r['pred_labels']:6.2f}")

    def spread(key):
        vals = [r[key] for r in rows.values() if r[key] is not None]
        return (min(vals), max(vals), sum(vals) / len(vals),
                max(vals) - min(vals)) if vals else (None,) * 4

    print(f"\n  {'metric':22} {'min':>7} {'max':>7} {'mean':>7} "
          f"{'SPREAD':>8}")
    for key in ("macro_f1", "micro_f1", "exact", "contains", "pred_labels"):
        lo, hi, mu, sp = spread(key)
        print(f"  {key:22} {lo:7.3f} {hi:7.3f} {mu:7.3f} {sp:8.3f}")

    # A label excluded by MIN_SUPPORT in one replicate but not another would
    # change what the macro average even averages over, which is the same
    # trap fixed_labels exists for in the bootstrap.
    excluded = {stem: r["excluded"] for stem, r in rows.items() if r["excluded"]}
    if excluded:
        print(f"\n  WARNING - labels excluded by MIN_SUPPORT: {excluded}")
        print("    The macro average is not over the same label set in every"
              "\n    replicate, so the spread above mixes two quantities.")

    # ------------------------------------------------------------ per label
    labels = sorted({l for d in per_label.values() for l in d})
    print(f"\n  per-label F1 across {len(args.runs)} replicates"
          f"  (the macro floor hides these):")
    print(f"  {'label':22} " + " ".join(f"{s[-2:]:>7}" for s in args.runs)
          + f" {'SPREAD':>8}")
    worst = None
    for label in labels:
        vals = [per_label[s].get(label) for s in args.runs]
        got = [v for v in vals if v is not None]
        sp = max(got) - min(got) if got else 0.0
        worst = (label, sp) if worst is None or sp > worst[1] else worst
        print(f"  {label:22} "
              + " ".join(f"{v:7.3f}" if v is not None else f"{'-':>7}"
                         for v in vals)
              + f" {sp:8.3f}")

    lo, hi, mu, macro_sp = spread("macro_f1")
    # This is a noise floor ONLY when the runs given are replicates of one
    # condition. Passed a mixture of conditions it is the EFFECT, and reading
    # it as a threshold would hide exactly the difference being measured -
    # which it did twice before this caveat was added.
    print(f"\n  spread across the {len(args.runs)} runs given: macro "
          f"{macro_sp:.3f}, worst per-label {worst[1]:.3f} ({worst[0]})")
    print(f"  -> a NOISE FLOOR only if these runs are replicates of ONE"
          f"\n     condition. Across different conditions this is the effect,"
          f"\n     not a threshold.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
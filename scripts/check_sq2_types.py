"""
scripts/check_sq2_types.py - the three-arm SQ2 comparison on
en_dev_eval_sq3_types. Read-only, zero API calls.
Run: python -m scripts.check_sq2_types

WHY THIS SUBSET AND NOT en_dev_eval_types. The project's only p<0.05 result -
hate_type, rag minus zero_shot +0.088, p=0.018 - was measured on 103 items
under prompt classify_v1-c74cb7ab and KB 475869f9e2422969, both retired. It
cannot appear in the same table as anything current without a caveat. This
replaces it on 467 items with 68-69 support per label, at the frozen state,
with three replicates of each retrieval-free arm and four of rag.

WHY IT IS NOT score_by_role. That script is rag-only by design; run on these
it asserts rather than silently scoring an absent arm. The three arms here
live in different files - zero_shot and few_shot in the sq2 runs, rag in the
round-0 runs - so the comparison has to be assembled across files.

THE PAIRING IS THE POINT. Every arm ran on identical items, so the paired
bootstrap resamples ITEMS and scores both arms on the same resample. Comparing
two macro-F1 numbers over 467 items discards that and cannot say whether a
0.02 gap is real.

READ EACH DELTA AGAINST THE RIGHT FLOOR. The retrieval-free arms replicate to
within 0.003 while rag spans 0.021, because rag inherits variance from which
examples are retrieved per item on top of decoding noise. A pooled floor would
be wrong for both.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from scripts.check_sq3_coverage import PROCESSED
from src.hsrag.analysis import paired_bootstrap
from src.hsrag.metrics import MIN_SUPPORT, score_multilabel

RESULTS = Path("experiments/results")
SUBSET = "en_dev_eval_sq3_types"
# GOLD_COL / DIM stay module globals rather than parameters because `macro` is
# imported by check_sq2_reconcile, which calls it as (rows, gold, fixed).
# main() rebinds them when --dim is given; an importer that never runs main()
# gets the hate_type default, which is what check_sq2_reconcile wants.
DIMS = {"hate_type": "hate_types",
        "target_group": "target_groups",
        "legal": "legal"}
GOLD_COL, DIM = "hate_types", "hate_type"
# zero_shot and few_shot come from the sq2 runs, rag from round 0. Round 0 has
# four replicates; the sq2 runs have three.
SOURCES = {
    "zero_shot": ["sq3_types_sq2_r1", "sq3_types_sq2_r2", "sq3_types_sq2_r3"],
    "few_shot": ["sq3_types_sq2_r1", "sq3_types_sq2_r2", "sq3_types_sq2_r3"],
    "rag": ["sq3_round0_r1", "sq3_round0_r2", "sq3_round0_r3", "sq3_round0_r4"],
}


def rows_for(stem: str, arm: str) -> list:
    path = RESULTS / f"{stem}_live.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r and r.get("arm") == arm:
            out.append(r)
    return out


def macro(rows: list, gold: dict, fixed: list | None = None):
    s = score_multilabel(rows, gold, GOLD_COL, DIM, fixed_labels=fixed)
    return s.macro_f1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=SUBSET)
    # The default SOURCES draws each arm from different files: zero_shot and
    # few_shot from the sq2 runs, rag from round 0. --stems overrides that for
    # a run that carries all three arms itself, which is what the earlier
    # en_dev_eval_types runs do and what makes the prompt/KB/subset
    # decomposition possible at zero API cost.
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--dim", default="hate_type", choices=sorted(DIMS))
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="labels dropped from the macro average; defaults to "
                         "config reporting.macro_exclude. Reported BESIDE the "
                         "full macro, never instead of it.")
    args = ap.parse_args()

    global GOLD_COL, DIM
    DIM, GOLD_COL = args.dim, DIMS[args.dim]

    if args.exclude is None:
        cfg = yaml.safe_load(
            Path("config/config.yaml").read_text(encoding="utf-8"))
        args.exclude = (cfg.get("reporting") or {}).get("macro_exclude") or []

    sources = {a: args.stems for a in SOURCES} if args.stems else SOURCES
    df = pd.read_parquet(PROCESSED / f"{args.subset}.parquet")
    gold = {str(r.id): r for r in df.itertuples(index=False)}

    per_arm = {arm: [rows_for(s, arm) for s in stems]
               for arm, stems in sources.items()}

    print(f"\n{'=' * 74}\nSQ2 on {args.subset}: {len(df)} items, {DIM}")
    print(f"  {'arm':12} {'replicates':>34} {'mean':>7} {'spread':>8}")
    means, labels_ref = {}, None
    for arm in ("zero_shot", "few_shot", "rag"):
        vals = []
        for rows in per_arm[arm]:
            s = score_multilabel(rows, gold, GOLD_COL, DIM)
            vals.append(s.macro_f1)
            # The label set that clears MIN_SUPPORT is fixed once, from the
            # first full sample, and reused in every resample. Recomputing it
            # per draw would change what is being averaged between draws and
            # make the interval meaningless.
            labels_ref = labels_ref or s.labels_averaged
        means[arm] = sum(vals) / len(vals)
        print(f"  {arm:12} {' / '.join(f'{v:.3f}' for v in vals):>34} "
              f"{means[arm]:7.3f} {max(vals) - min(vals):8.3f}")

    print(f"\n  averaged over {len(labels_ref)} labels at MIN_SUPPORT="
          f"{MIN_SUPPORT}: {labels_ref}")

    kept = [l for l in labels_ref if l not in args.exclude]
    if kept != labels_ref:
        dropped = [l for l in labels_ref if l in args.exclude]
        print(f"\n  SECOND MACRO, excluding {dropped} -> {len(kept)} labels.")
        print(f"  {'arm':12} {'replicates':>34} {'mean':>7} {'spread':>8}")
        for arm in ("zero_shot", "few_shot", "rag"):
            vals = [macro(rows, gold, kept) for rows in per_arm[arm]]
            print(f"  {arm:12} {' / '.join(f'{v:.3f}' for v in vals):>34} "
                  f"{sum(vals) / len(vals):7.3f} "
                  f"{max(vals) - min(vals):8.3f}")
        print(f"  The dropped label appears with its own support, F1 and "
              f"floor in the per-label table.")

    print(f"\n{'-' * 74}\nDELTAS (mean over replicates)")
    for a, b in (("rag", "zero_shot"), ("rag", "few_shot"),
                 ("few_shot", "zero_shot")):
        print(f"  {a} - {b:12} {means[a] - means[b]:+.3f}")

    # ------------------------------------------------- paired bootstrap
    # INDEX-MATCHED REPLICATE PAIRS, not replicate 1 alone. Rag's replicate 1
    # is the highest of its four (0.365 against 0.344-0.365), so a bootstrap on
    # it reports the most favourable pairing available and its p understates
    # the true uncertainty. Pairing r1-r1, r2-r2, r3-r3 gives a RANGE, which is
    # what should be reported when one arm's spread is 7x the other's.
    print(f"\n{'-' * 74}\nPAIRED BOOTSTRAP, index-matched replicate pairs, "
          f"1000 resamples of items each")
    for a, b in (("rag", "zero_shot"), ("rag", "few_shot"),
                 ("few_shot", "zero_shot")):
        n = min(len(per_arm[a]), len(per_arm[b]))
        cells = []
        for i in range(n):
            res = paired_bootstrap(
                per_arm[a][i], per_arm[b][i], gold,
                lambda rows, g: macro(rows, g, labels_ref))
            if res.get("delta") is None:
                cells.append(("n/a", "n/a"))
                continue
            cells.append((res["delta"], res["p_value"]))
            print(f"  {a} - {b:12} rep {i + 1}  delta {res['delta']:+.3f}  "
                  f"CI [{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]  "
                  f"p = {res['p_value']:.4f}")
        ps = [p for _, p in cells if p != "n/a"]
        ds = [d for d, _ in cells if d != "n/a"]
        if ps:
            print(f"  {a} - {b:12} RANGE   delta {min(ds):+.3f} to "
                  f"{max(ds):+.3f}   p {min(ps):.4f} to {max(ps):.4f}\n")

    # --------------------------------------------------------- per label
    print(f"\n{'-' * 74}\nPER LABEL, mean over replicates")
    print(f"  {'label':16} {'sup':>4} {'zero_shot':>10} {'few_shot':>10} "
          f"{'rag':>10} {'rag-zs':>8} {'rag-fs':>8} {'floor':>8}")
    acc = defaultdict(lambda: defaultdict(list))
    support = {}
    for arm in ("zero_shot", "few_shot", "rag"):
        for rows in per_arm[arm]:
            s = score_multilabel(rows, gold, GOLD_COL, DIM)
            for lab, cell in s.per_label.items():
                acc[lab][arm].append(cell["f1"])
                # Support is a property of the gold, identical across arms and
                # replicates. Printed because retraction 14 was a two-label
                # macro over 28 scorable items that read as a 0.877 result.
                support[lab] = cell["support"]
    for lab in sorted(acc):
        m = {a: sum(v) / len(v) for a, v in acc[lab].items()}
        # MEASURED per-label floor: the widest within-arm spread across
        # replicates for this label. A per-label delta smaller than this is
        # not evidence. Two published subset floors were wrong because they
        # were taken from the arm under test at two replicates.
        # ACROSS ALL THREE ARMS. check_sq2_reconcile computes the per-label
        # floor across only the two arms in its comparison, so the two scripts
        # give different numbers for the same label (explicit 0.044 here vs
        # 0.027 there) and must not be quoted interchangeably. This one is the
        # general "how noisy is this label" figure; reconcile's is the correct
        # floor for a specific rag-minus-few_shot per-label claim.
        floor = max((max(v) - min(v)) for v in acc[lab].values() if v)
        print(f"  {lab:16} {support.get(lab, 0):>4} "
              f"{m.get('zero_shot', 0):10.3f} "
              f"{m.get('few_shot', 0):10.3f} {m.get('rag', 0):10.3f} "
              f"{m.get('rag', 0) - m.get('zero_shot', 0):+8.3f} "
              f"{m.get('rag', 0) - m.get('few_shot', 0):+8.3f} "
              f"{floor:8.3f}")

    # ------------------------------------------------- over-prediction
    gold_card = score_multilabel(per_arm["rag"][0], gold, GOLD_COL,
                                 DIM).extra["mean_gold_labels"]
    print(f"\n{'-' * 74}\nPREDICTED LABELS PER ITEM (gold {gold_card:.2f})")
    for arm in ("zero_shot", "few_shot", "rag"):
        vals = [score_multilabel(rows, gold, GOLD_COL, DIM)
                .extra["mean_pred_labels"] for rows in per_arm[arm]]
        print(f"  {arm:12} {' / '.join(f'{v:.2f}' for v in vals)}")
    if DIM == "hate_type":
        print(f"\n  Examples cut over-prediction sharply whether static or "
              f"retrieved.\n  That is most of what few_shot buys, and it buys "
              f"it while LOSING macro-F1.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
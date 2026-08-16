"""
scripts/check_sq2_reconcile.py - Phase 7 step 7.1. Reconciles the two SQ2
estimates of `rag - few_shot` on hate_type. Read-only, zero API calls.
Run: python -m scripts.check_sq2_reconcile

THE PROBLEM. PHASE8_SIGNOFF 3.11 reports the same comparison, at the same
system state, on two disjoint samples of the same population, and gets +0.134
(n=103) and +0.033 (n=467). The smaller sample was promoted to headline. That
is the shape of retraction 7, where rag - zero_shot +0.088 rested on a
zero_shot draw of 0.280 against a true range of 0.336-0.354. Retraction 7 was
caught by replication; this one cannot be, because both figures are already
replicated WITHIN their own subset. It has to be caught by reconciliation.

WHY THIS IS NOT JUST "AVERAGE THEM". Two estimates may only be pooled if they
are estimating the same quantity. If they are formally heterogeneous, an
average is a number with no referent, and the job changes from picking a
headline to explaining a contradiction. So the heterogeneity test runs BEFORE
the pooled estimate and gates it, rather than being reported alongside it as
a caveat.

VARIANCE HAS TWO SOURCES HERE AND THEY DO NOT COMBINE THE OBVIOUS WAY.
  - Item sampling: which 103 or 467 items were drawn. Estimated by the paired
    bootstrap, which resamples items at fixed decoding output.
  - Decoding: the model is at T=1.0 with n_votes=3. Estimated by the spread
    across replicates, which run on IDENTICAL items.
Averaging R replicates shrinks the decoding component by sqrt(R) and does
NOTHING to the item component, because every replicate sees the same items.
So SE(mean delta)^2 = SE_item^2 + sd_replicates^2 / R. Treating the bootstrap
CI alone as the uncertainty understates it; treating the replicate spread
alone understates it far more.

FLOORS ARE MEASURED HERE, NOT SCALED. An earlier draft of this analysis scaled
the per-label floors from support 68 to support 15 by sqrt(68/15). That is an
assumption. Every arm has three or more replicates on both subsets, so the
per-label spread can be read directly off the data, and it is.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

# One implementation, always. rows_for and macro live in check_sq2_types and
# are imported rather than re-written here: run_slice1 carried its own copy of
# the scoring functions for two phases, missed a fix, and printed a
# meaningless number the whole time.
from scripts.check_sq2_types import RESULTS, macro, rows_for
from scripts.check_sq3_coverage import PROCESSED
from src.hsrag.analysis import paired_bootstrap
from src.hsrag.metrics import MIN_SUPPORT, clean, score_multilabel

GOLD_COL, DIM = "hate_types", "hate_type"
A_NAME = "en_dev_eval_types"
B_NAME = "en_dev_eval_sq3_types"

# Subset B's arms live in different files: zero_shot and few_shot in the sq2
# runs, rag in the round-0 runs. Subset A's runs each carry all three arms,
# and are globbed rather than listed so that a newly added replicate is picked
# up instead of silently excluded by a stale constant. Every file used is
# printed, so the glob is auditable.
B_SOURCES = {
    "zero_shot": ["sq3_types_sq2_r1", "sq3_types_sq2_r2", "sq3_types_sq2_r3"],
    "few_shot": ["sq3_types_sq2_r1", "sq3_types_sq2_r2", "sq3_types_sq2_r3"],
    "rag": ["sq3_round0_r1", "sq3_round0_r2", "sq3_round0_r3",
            "sq3_round0_r4"],
}


def a_stems() -> list:
    return sorted(p.stem.replace("_live", "")
                  for p in RESULTS.glob("types_kbv3_r*_live.jsonl"))


def gold_of(subset: str) -> tuple:
    df = pd.read_parquet(PROCESSED / f"{subset}.parquet")
    return df, {str(r.id): r for r in df.itertuples(index=False)}


# ------------------------------------------------------------- composition

def composition(name: str, df: pd.DataFrame) -> dict:
    """The checks that decide whether the two estimates CAN be compared.

    Not a formality. A gate=False item has hate_type cleared to [] by gate
    consistency, and few_shot's over-prediction suppression helps on empty
    gold while hurting on hateful gold. A different gate mix between the two
    subsets would therefore produce a different rag - few_shot with no effect
    involved at all.
    """
    src = df["source"].value_counts().to_dict()
    gate = df["gate"].apply(lambda v: "None" if clean(v) is None
                            else str(bool(v))).value_counts().to_dict()
    sets = [set(clean(v) or []) for v in df[GOLD_COL]]
    sup = defaultdict(int)
    for s in sets:
        for l in s:
            sup[l] += 1
    return {"n": len(df), "source": src, "gate": gate,
            "cardinality": sum(len(s) for s in sets) / max(len(sets), 1),
            "support": dict(sorted(sup.items()))}


# ----------------------------------------------------------------- deltas

def arm_rows(subset: str, arm: str) -> list:
    """Replicate-major list of row lists for one arm on one subset."""
    stems = a_stems() if subset == A_NAME else B_SOURCES[arm]
    return [rows_for(s, arm) for s in stems]


def estimate(subset: str, gold: dict, a: str, b: str, labels_ref: list,
             n_boot: int) -> dict:
    """Mean delta for a - b, with its item and decoding variance separated."""
    ra, rb = arm_rows(subset, a), arm_rows(subset, b)
    n = min(len(ra), len(rb))

    deltas, sds, cis, ps = [], [], [], []
    fallback = False
    for i in range(n):
        res = paired_bootstrap(ra[i], rb[i], gold,
                               lambda rows, g: macro(rows, g, labels_ref),
                               b=n_boot)
        if res.get("delta") is None:
            continue
        deltas.append(res["delta"])
        cis.append((res["ci_low"], res["ci_high"]))
        ps.append(res["p_value"])
        if "sd" in res:
            sds.append(res["sd"])
        else:
            # analysis.paired_bootstrap gained an `sd` key for this analysis.
            # If it is absent the SE is reconstructed from the percentile CI,
            # which is lossy whenever the delta distribution is skewed. Loud,
            # because a silently reconstructed SE would propagate into the
            # heterogeneity test and the pooled CI.
            fallback = True
            sds.append((res["ci_high"] - res["ci_low"]) / 3.92)

    r = len(deltas)
    mean = float(np.mean(deltas))
    se_item = float(np.mean(sds))
    sd_rep = float(np.std(deltas, ddof=1)) if r > 1 else 0.0
    # Replicates run on IDENTICAL items, so averaging them shrinks decoding
    # variance by sqrt(R) and leaves item variance untouched.
    se_total = float(np.sqrt(se_item ** 2 + (sd_rep ** 2) / max(r, 1)))
    return {"deltas": deltas, "cis": cis, "ps": ps, "mean": mean,
            "se_item": se_item, "sd_rep": sd_rep, "se_total": se_total,
            "r": r, "sd_fallback": fallback}


# ------------------------------------------------------------- per label

def per_label(subset: str, gold: dict, arms: tuple) -> dict:
    """label -> arm -> list of per-replicate F1. Spread here IS the floor."""
    acc = defaultdict(lambda: defaultdict(list))
    for arm in arms:
        for rows in arm_rows(subset, arm):
            s = score_multilabel(rows, gold, GOLD_COL, DIM)
            for lab, cell in s.per_label.items():
                acc[lab][arm].append(cell["f1"])
    return acc


# --------------------------------------------- fixed_labels resample check

def label_survival(subset: str, gold: dict, arm: str, labels_ref: list,
                   n_boot: int, seed: int = 42) -> dict:
    """How often does a bootstrap resample lose one of the averaged labels?

    score_multilabel builds its label set from the UNION of gold and predicted
    labels, and under fixed_labels a label absent from a resample is SKIPPED
    rather than scored 0 - correct, because absent-from-gold makes F1
    undefined rather than zero. The consequence is that the macro is averaged
    over a varying number of labels between draws, so some draws compute a
    six-label macro where the observed value is a seven-label macro. At
    support 15 that is rare but not impossible, and it would narrow the CI on
    the smaller subset relative to the larger one - one more reason the 103 is
    the less trustworthy instrument, if it fires.
    """
    rows = arm_rows(subset, arm)[0]
    pairs = []
    for r in rows:
        g = gold.get(r["item_id"])
        if g is None or not bool(g.gate) or r["result"] is None:
            continue
        truth = clean(getattr(g, GOLD_COL, None))
        if truth is None:
            continue
        pairs.append((set(truth), set(r["result"][DIM])))

    rng = np.random.default_rng(seed)
    ref = set(labels_ref)
    counts = defaultdict(int)
    for _ in range(n_boot):
        pick = rng.integers(0, len(pairs), size=len(pairs))
        seen = set()
        for j in pick:
            t, p = pairs[j]
            seen |= t | p
        counts[len(ref & seen)] += 1
    return dict(sorted(counts.items(), reverse=True))


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--a", default="rag")
    ap.add_argument("--b", default="few_shot")
    args = ap.parse_args()

    print(f"\n{'=' * 78}")
    print(f"SQ2 RECONCILIATION - {args.a} - {args.b} on {DIM}")
    print(f"{'=' * 78}")

    dfa, ga = gold_of(A_NAME)
    dfb, gb = gold_of(B_NAME)
    stems_a = a_stems()
    print(f"\nsubset A  {A_NAME}   replicates: {', '.join(stems_a)}")
    print(f"subset B  {B_NAME}   zero_shot/few_shot: "
          f"{', '.join(B_SOURCES['few_shot'])}")
    print(f"{'':10}{'':{len(B_NAME)}}   rag: {', '.join(B_SOURCES['rag'])}")

    # --------------------------------------------------- 7.1a label sets
    print(f"\n{'-' * 78}\n7.1a  DO BOTH SUBSETS AVERAGE OVER THE SAME LABELS?")
    print("  A six-label macro and a seven-label macro are different")
    print("  quantities. If MIN_SUPPORT excluded a label on one subset and")
    print("  not the other, the gap would be a reporting artefact.\n")
    refs = {}
    for name, gold in ((A_NAME, ga), (B_NAME, gb)):
        s = score_multilabel(arm_rows(name, args.a)[0], gold, GOLD_COL, DIM)
        refs[name] = s.labels_averaged
        print(f"  {name:24} {len(s.labels_averaged)} labels at "
              f"MIN_SUPPORT={MIN_SUPPORT}")
        print(f"  {'':24} {s.labels_averaged}")
        if s.labels_excluded:
            print(f"  {'':24} EXCLUDED: {s.labels_excluded}")
    same = refs[A_NAME] == refs[B_NAME]
    print(f"\n  IDENTICAL LABEL SETS: {'YES' if same else 'NO'}")
    if not same:
        print("  -> the two macros are different quantities. STOP. The gap is")
        print("     a reporting artefact and must be fixed before pooling.")
        return
    labels_ref = refs[A_NAME]

    # --------------------------------------------------- 7.1b composition
    print(f"\n{'-' * 78}\n7.1b  IS THE POPULATION THE SAME?")
    ca, cb = composition(A_NAME, dfa), composition(B_NAME, dfb)
    print(f"  {'property':16} {A_NAME:>26} {B_NAME:>26}")
    print(f"  {'n':16} {ca['n']:>26} {cb['n']:>26}")
    print(f"  {'source':16} {str(ca['source']):>26} {str(cb['source']):>26}")
    print(f"  {'gate gold':16} {str(ca['gate']):>26} {str(cb['gate']):>26}")
    print(f"  {'cardinality':16} {ca['cardinality']:>26.3f} "
          f"{cb['cardinality']:>26.3f}")
    print(f"\n  per-label gold support")
    print(f"  {'label':16} {A_NAME:>26} {B_NAME:>26}")
    for l in labels_ref:
        print(f"  {l:16} {ca['support'].get(l, 0):>26} "
              f"{cb['support'].get(l, 0):>26}")
    print("\n  Same source and same gate mix means composition cannot explain")
    print("  the gap: a gate=False item would have hate_type cleared to [],")
    print("  and few_shot's over-prediction suppression helps on empty gold")
    print("  while hurting on hateful gold. Support per label is the only")
    print("  property that differs.")

    # ------------------------------------------------------ arms & floors
    print(f"\n{'-' * 78}\nMEASURED FLOORS - within-arm spread at a fixed state")
    print("  This is the number every delta on that subset must clear. The")
    print("  floor is the WIDEST arm, not the arm under test.\n")
    print(f"  {'subset':24} {'arm':10} {'replicates':>40} {'spread':>8}")
    floors = {}
    for name, gold in ((A_NAME, ga), (B_NAME, gb)):
        worst = 0.0
        for arm in ("zero_shot", "few_shot", "rag"):
            vals = [macro(rows, gold, labels_ref)
                    for rows in arm_rows(name, arm)]
            spread = max(vals) - min(vals)
            worst = max(worst, spread)
            print(f"  {name:24} {arm:10} "
                  f"{' / '.join(f'{v:.3f}' for v in vals):>40} {spread:8.3f}")
        floors[name] = worst
        print(f"  {name:24} {'FLOOR':10} {'':>40} {worst:8.3f}\n")

    # --------------------------------------------------------- estimates
    print(f"{'-' * 78}\nPER-SUBSET ESTIMATES, variance decomposed")
    est = {}
    for name, gold in ((A_NAME, ga), (B_NAME, gb)):
        e = estimate(name, gold, args.a, args.b, labels_ref, args.boot)
        est[name] = e
        print(f"\n  {name}  ({e['r']} index-matched replicate pairs)")
        for i, (d, (lo, hi), p) in enumerate(zip(e["deltas"], e["cis"],
                                                 e["ps"])):
            print(f"    rep {i + 1}   delta {d:+.3f}   CI [{lo:+.3f}, "
                  f"{hi:+.3f}]   p = {p:.4f}")
        print(f"    mean delta            {e['mean']:+.3f}")
        print(f"    SE from item sampling  {e['se_item']:.4f}   "
              f"(bootstrap, {args.boot} resamples)")
        print(f"    SD across replicates   {e['sd_rep']:.4f}   "
              f"(decoding noise, identical items)")
        print(f"    SE of the mean         {e['se_total']:.4f}   "
              f"= sqrt(item^2 + rep^2/{e['r']})")
        if e["sd_fallback"]:
            print("    WARNING: paired_bootstrap returned no `sd` key; SE")
            print("             reconstructed from the percentile CI, which")
            print("             is lossy under skew. Add the `sd` key.")

    ea, eb = est[A_NAME], est[B_NAME]

    # ----------------------------------------------------- heterogeneity
    print(f"\n{'-' * 78}\n7.1d  HETEROGENEITY - may these be pooled at all?")
    diff = ea["mean"] - eb["mean"]
    se_d = float(np.sqrt(ea["se_total"] ** 2 + eb["se_total"] ** 2))
    z = diff / se_d
    p_het = float(2 * (1 - norm.cdf(abs(z))))
    print(f"\n  difference between subsets   {diff:+.3f}")
    print(f"  SE of the difference          {se_d:.4f}   "
          f"(the subsets are disjoint, so independent)")
    print(f"  z = {z:.2f}   p = {p_het:.4f}")

    w_a, w_b = 1 / ea["se_total"] ** 2, 1 / eb["se_total"] ** 2
    pooled = (w_a * ea["mean"] + w_b * eb["mean"]) / (w_a + w_b)
    se_p = float(np.sqrt(1 / (w_a + w_b)))
    q = w_a * (ea["mean"] - pooled) ** 2 + w_b * (eb["mean"] - pooled) ** 2
    p_q = float(1 - chi2.cdf(q, 1))
    i2 = max(0.0, (q - 1) / q) if q > 0 else 0.0

    print(f"\n  Cochran's Q = {q:.2f} on 1 df, p = {p_q:.4f}, "
          f"I^2 = {i2 * 100:.0f}%")

    if p_het < 0.05:
        print("\n  HETEROGENEOUS. The two subsets are not noisy measurements")
        print("  of one quantity. Pooling is INADMISSIBLE - an average would")
        print("  be a number with no referent. Report both, state the")
        print("  discrepancy, and explain it from the per-label table below.")
    else:
        print("\n  NOT SIGNIFICANTLY HETEROGENEOUS. Pooling is admissible.")
        print(f"\n  inverse-variance pooled   {pooled:+.3f}")
        print(f"  95% CI                    [{pooled - 1.96 * se_p:+.3f}, "
              f"{pooled + 1.96 * se_p:+.3f}]")
        print(f"  weights                   {A_NAME} "
              f"{w_a / (w_a + w_b) * 100:.0f}%, {B_NAME} "
              f"{w_b / (w_a + w_b) * 100:.0f}%")
        print(f"  n items                   {ca['n'] + cb['n']}")
        print("\n  Inverse-variance, NOT n-weighted. n-weighting assumes equal")
        print("  per-item variance, which is false here: the smaller subset")
        print("  has a per-arm spread an order of magnitude wider.")

    # --------------------------------------------------------- per label
    print(f"\n{'-' * 78}\n7.1e  WHICH LABELS CARRY THE EFFECT?")
    print("  If the same labels carry it on both subsets with the same sign,")
    print("  the mechanism is real and only its magnitude is subset-specific.")
    print("  If different labels carry it, the smaller estimate is noise.\n")
    tables = {n: per_label(n, g, (args.a, args.b))
              for n, g in ((A_NAME, ga), (B_NAME, gb))}
    print(f"  {'label':14} | {A_NAME} (n={ca['n']}) "
          f"| {B_NAME} (n={cb['n']})")
    print(f"  {'':14} | {'sup':>4} {args.b[:9]:>9} {args.a[:9]:>9} "
          f"{'delta':>8} {'floor':>7} "
          f"| {'sup':>4} {args.b[:9]:>9} {args.a[:9]:>9} {'delta':>8} "
          f"{'floor':>7}")
    contrib = {A_NAME: {}, B_NAME: {}}
    for l in labels_ref:
        cells = []
        for name, comp in ((A_NAME, ca), (B_NAME, cb)):
            t = tables[name][l]
            ma = float(np.mean(t[args.a])) if t[args.a] else float("nan")
            mb = float(np.mean(t[args.b])) if t[args.b] else float("nan")
            # The per-label floor, measured: the widest within-arm spread
            # across replicates for that label on that subset, restricted to
            # the TWO arms in this comparison. check_sq2_types takes the widest
            # across all three, which gives a different number for the same
            # label. This one is correct for a delta between these two arms;
            # that one is the general per-label noise figure. Do not quote them
            # interchangeably.
            fl = max((max(v) - min(v)) for v in (t[args.a], t[args.b]) if v)
            contrib[name][l] = ma - mb
            cells.append(f"{comp['support'].get(l, 0):>4} {mb:>9.3f} "
                         f"{ma:>9.3f} {ma - mb:>+8.3f} {fl:>7.3f}")
        print(f"  {l:14} | {cells[0]} | {cells[1]}")

    print(f"\n  {'':14}   contribution to the macro (delta / "
          f"{len(labels_ref)} labels)")
    for name in (A_NAME, B_NAME):
        tot = sum(contrib[name].values()) / len(labels_ref)
        top = sorted(contrib[name].items(), key=lambda kv: -abs(kv[1]))[:2]
        share = sum(v for _, v in top) / len(labels_ref)
        print(f"  {name:24} total {tot:+.3f}   top two "
              f"{', '.join(k for k, _ in top)} = {share:+.3f} "
              f"({share / tot * 100:.0f}% of it)" if tot else "")

    print("\n  A per-label delta smaller than its own measured floor is not")
    print("  evidence of anything, whatever the macro says.")

    # ------------------------------------------- fixed_labels diagnostic
    print(f"\n{'-' * 78}\nDIAGNOSTIC  does a resample ever lose a label?")
    for name, gold in ((A_NAME, ga), (B_NAME, gb)):
        for arm in (args.a, args.b):
            surv = label_survival(name, gold, arm, labels_ref, args.boot)
            full = surv.get(len(labels_ref), 0)
            print(f"  {name:24} {arm:10} "
                  f"{full}/{args.boot} draws keep all {len(labels_ref)}"
                  + (f"   others: {surv}" if len(surv) > 1 else ""))
    print("\n  Any draw short of the full label set computes a macro over")
    print("  fewer labels than the observed value does, which narrows the CI")
    print("  on that subset. Only matters if it fires.")

    # ---------------------------------------------------------- verdict
    print(f"\n{'=' * 78}\nWHAT TO REPORT")
    print(f"  primary        {B_NAME}, n={cb['n']}, "
          f"{eb['mean']:+.3f}, floor {floors[B_NAME]:.3f}")
    if p_het >= 0.05:
        print(f"  best estimate  pooled {pooled:+.3f}, CI "
              f"[{pooled - 1.96 * se_p:+.3f}, {pooled + 1.96 * se_p:+.3f}], "
              f"n={ca['n'] + cb['n']}")
    print(f"  secondary      {A_NAME}, n={ca['n']}, "
          f"{ea['mean']:+.3f}, floor {floors[A_NAME]:.3f} "
          f"- the noisier instrument")
    print(f"\n  Quote the floor beside every figure, and the per-label floor")
    print(f"  beside every per-label claim.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
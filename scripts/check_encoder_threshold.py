"""
scripts/check_encoder_threshold.py - is the encoder's deficit a capability gap
or a decision-policy artefact? Read-only, zero API calls, no GPU.

  python -m scripts.check_encoder_threshold
  python -m scripts.check_encoder_threshold --arm full --head target_group

THE QUESTION. Section 11 of sq2_log.md, extended by 7.6, says prediction
cardinality predicts the macro-F1 ordering ACROSS MODEL FAMILIES: whichever
system predicts a number of labels closer to the gold cardinality wins, on
three of three multilabel dimensions.

  hate_type     gold 1.02   encoder 0.63 (off 0.39)   RAG 1.74 (off 0.72)   encoder wins
  legal         gold 0.55   encoder 0.49 (off 0.06)   RAG 1.01 (off 0.46)   encoder wins
  target_group  gold 2.23   encoder 1.20 (off 1.03)   RAG 1.68 (off 0.55)   RAG wins

That makes target_group the one encoder result that might not be a capability
gap at all. The encoder predicts barely half the labels the gold has, at a
threshold FIXED AT 0.5 and never tuned - which was the correct
pre-registration, because tuning it on anything would hand the encoder a
fitting opportunity the LLM never had. But it means the -0.093 could be a
decision-policy gap wearing a capability gap's clothes.

The raw sigmoid outputs were stored in run_predictions precisely so this costs
no GPU.

FOUR OPERATING POINTS, AND ONLY ONE OF THEM IS A RESULT.

  0.5            the pre-registered primary. The number that goes in the
                 table.
  gold-matched   the threshold at which the encoder predicts as many labels
                 per item as the gold has. Answers "how good is the ranking,
                 independent of where the cut sits".
  RAG-matched    the threshold at which it predicts as many as the LLM does.
                 The like-for-like decision-policy comparison, and the one
                 that decides the question above.
  eval optimum   UPPER BOUND, NOT A RESULT. Chosen by maximising macro-F1 on
                 the scoring data itself. Reported so the ceiling is visible
                 and labelled so it is never quoted as performance.

The first three are chosen WITHOUT looking at macro-F1 - two by matching a
cardinality that was fixed before the encoder existed, one by pre-registration.
Only the fourth peeks, and it is marked.

WHY NOT JUST ADOPT THE BEST THRESHOLD. Because the LLM has no threshold to
tune. It emits a label set and that is the answer. Tuning the encoder's cut
and not the LLM's would be the same asymmetry as early-stopping on the eval
subsets, which 7.5 refused for the same reason.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("experiments/results")
PROCESSED = Path("data/processed")

# arm, head -> (subset, dimension, gold column, RAG cardinality from the
# frozen-config rag runs). RAG figures are transcribed from
# check_encoder_vs_llm on 2026-08-17.
SPEC = {
    ("full", "target_group"): ("en_dev_eval_targets", "target_groups", 1.68),
    ("full", "hate_type"): ("en_dev_eval_sq3_types", "hate_types", 1.74),
    ("de", "legal"): ("de_legal_dev_eval", "legal", 1.01),
    ("kb", "target_group"): ("en_dev_eval_targets", "target_groups", 1.68),
    ("kb", "hate_type"): ("en_dev_eval_sq3_types", "hate_types", 1.74),
}


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def load(arm, seed, subset):
    p = RESULTS / f"encoder_{arm}_seed{seed}_live.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" in r:
            continue
        if (r.get("encoder_meta") or {}).get("subset") != subset:
            continue
        out.append(r)
    return out


def macro_at(P, G, labels, thr, min_support=10):
    """Macro-F1 at a threshold, over labels clearing MIN_SUPPORT.

    Mirrors metrics.score_multilabel's averaging rule so the number is
    comparable with everything else in the project: a label with fewer than
    ten gold positives is reported but not averaged, because a per-label F1
    from two items is noise wearing a number's clothes.
    """
    f1s, card = [], float((P >= thr).sum(axis=1).mean())
    for k, lab in enumerate(labels):
        t, p = G[:, k] > 0.5, P[:, k] >= thr
        sup = int(t.sum())
        if sup < min_support:
            continue
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        fn = int((t & ~p).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return (float(np.mean(f1s)) if f1s else 0.0), card


def thr_for_cardinality(P, target):
    """Threshold whose mean predicted-label count is closest to `target`.

    Searched on a fixed grid rather than solved: the mapping is a step
    function, so many thresholds give the same cardinality and the grid keeps
    the choice reproducible.
    """
    grid = np.linspace(0.01, 0.99, 99)
    cards = np.array([(P >= t).sum(axis=1).mean() for t in grid])
    return float(grid[int(np.argmin(np.abs(cards - target)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    import yaml
    tax = yaml.safe_load(
        Path("config/taxonomy.yaml").read_text(encoding="utf-8"))

    print(f"\n{'=' * 92}\nENCODER THRESHOLD SENSITIVITY   seeds "
          f"{args.seeds}\n{'=' * 92}")
    print("\nOnly the 0.5 column is a result. gold- and RAG-matched are chosen")
    print("by cardinality, never by macro-F1. The eval optimum is an UPPER")
    print("BOUND obtained by tuning on the scoring data and is not "
          "performance.\n")

    for (arm, head), (subset, gold_col, rag_card) in SPEC.items():
        if args.arm and arm != args.arm:
            continue
        if args.head and head != args.head:
            continue
        labels = list(tax["dimensions"][head]["labels"])
        df = pd.read_parquet(PROCESSED / f"{subset}.parquet")
        gold = {str(r.id): set(clean(getattr(r, gold_col, None)) or [])
                for r in df.itertuples(index=False)
                if clean(getattr(r, gold_col, None)) is not None
                and (head == "legal" or bool(r.gate))}
        gold_card = np.mean([len(v) for v in gold.values()])

        print(f"{'-' * 92}\n{arm} / {head}   {subset}   "
              f"{len(gold)} scorable items   gold cardinality "
              f"{gold_card:.2f}   RAG {rag_card:.2f}")
        rows = []
        for seed in args.seeds:
            recs = [r for r in load(arm, seed, subset)
                    if r["item_id"] in gold and r.get("run_predictions")]
            if not recs:
                continue
            ids = [r["item_id"] for r in recs]
            P = np.array([[r["run_predictions"][0][head][l] for l in labels]
                          for r in recs])
            G = np.array([[1.0 if l in gold[i] else 0.0 for l in labels]
                          for i in ids])
            # GATE MASK. run_predictions holds the RAW sigmoid outputs, before
            # Result.gate_consistency clears target_group, hate_type and
            # severity on any item whose gate head said not-hate. Scoring the
            # raw probabilities measures THE HEAD; scoring after the mask
            # measures THE SYSTEM, which is what check_encoder_vs_llm reports
            # and what the LLM is compared against.
            #
            # They differ by a lot. On en_dev_eval_targets the target_group
            # head scores 0.680 alone and 0.553 deployed, because the gate
            # head says not-hate on roughly a third of a subset where every
            # item is gold-hateful. `legal` is exempt from gate consistency
            # and is the one dimension where the two agree to three decimals -
            # which is what confirmed the diagnosis.
            gate_ok = np.array([bool((r.get("result") or {}).get("hate"))
                                for r in recs])
            if head == "legal":
                gate_ok = np.ones(len(recs), dtype=bool)

            t_gold = thr_for_cardinality(P, gold_card)
            t_rag = thr_for_cardinality(P, rag_card)
            grid = np.linspace(0.01, 0.99, 99)
            scored = [(macro_at(P, G, labels, t)[0], t) for t in grid]
            best_f, t_best = max(scored)

            r05 = macro_at(P, G, labels, 0.5)
            rg = macro_at(P, G, labels, t_gold)
            rr = macro_at(P, G, labels, t_rag)
            # As deployed: zero every probability on a gate-negative item, so
            # thresholding reproduces exactly what gate_consistency does.
            Pm = P * gate_ok[:, None]
            dep = macro_at(Pm, G, labels, 0.5)
            rows.append((seed, r05, (t_gold, *rg), (t_rag, *rr),
                         (t_best, best_f), dep,
                         float((~gate_ok).mean())))

        if not rows:
            print("  no stored probabilities for this head")
            continue
        print(f"  {'seed':>4} | {'thr 0.5':>16} | {'gold-matched':>22} "
              f"| {'RAG-matched':>22} | {'eval optimum':>18}")
        print(f"  {'':>4} | {'macro':>7} {'card':>8} | {'thr':>5} {'macro':>7} "
              f"{'card':>7} | {'thr':>5} {'macro':>7} {'card':>7} "
              f"| {'thr':>5} {'macro':>11}")
        agg = {"05": [], "gold": [], "rag": [], "best": [], "dep": [],
               "gfn": []}
        for seed, r05, rg, rr, rb, dep, gfn in rows:
            print(f"  {seed:>4} | {r05[0]:>7.3f} {r05[1]:>8.2f} "
                  f"| {rg[0]:>5.2f} {rg[1]:>7.3f} {rg[2]:>7.2f} "
                  f"| {rr[0]:>5.2f} {rr[1]:>7.3f} {rr[2]:>7.2f} "
                  f"| {rb[0]:>5.2f} {rb[1]:>11.3f}")
            for k, v in (("05", r05[0]), ("gold", rg[1]), ("rag", rr[1]),
                         ("best", rb[1]), ("dep", dep[0]), ("gfn", gfn)):
                agg[k].append(v)
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        print(f"  {'mean':>4} | {m['05']:>7.3f} {'':>8} | {'':>5} "
              f"{m['gold']:>7.3f} {'':>7} | {'':>5} {m['rag']:>7.3f} "
              f"{'':>7} | {'':>5} {m['best']:>11.3f}")
        print(f"\n  HEAD ALONE (raw)     {m['05']:.3f}   at thr 0.5")
        print(f"  AS DEPLOYED          {m['dep']:.3f}   after "
              f"gate_consistency clears {m['gfn']:.1%} of items")
        print(f"  cost of the gate     {m['dep'] - m['05']:+.3f}")

        # ------------------------------------------------------- verdict
        print(f"\n  RAG on this dimension: "
              + {"target_group": "0.646", "hate_type": "0.357",
                 "legal": "0.599"}[head])
        base = {"target_group": 0.646, "hate_type": 0.357,
                "legal": 0.599}[head]
        d05, ddep = m["05"] - base, m["dep"] - base
        drag = m["rag"] - base
        print(f"  delta, head alone     {d05:+.3f}")
        print(f"  delta, as deployed    {ddep:+.3f}   <- the system-level "
              f"result, comparable with the LLM")
        print(f"  delta at RAG-matched  {drag:+.3f}   (head alone)")
        if d05 >= 0 > ddep:
            print("  -> THE HEAD BEATS RAG; THE SYSTEM DOES NOT. The loss is "
                  "the GATE head's\n     errors propagating through "
                  "gate_consistency, not this head's\n     ranking. A cost of "
                  "the separate-model-per-head design that the\n     LLM, "
                  "producing all five dimensions in one pass, does not pay.")
        if d05 < 0 <= drag:
            print("  -> THE DEFICIT IS A DECISION-POLICY ARTEFACT. At equal "
                  "prediction\n     cardinality the encoder is not behind. "
                  "Report the 0.5 figure as the\n     result and this as the "
                  "diagnosis.")
        elif d05 < 0 and drag < 0:
            print("  -> THE DEFICIT SURVIVES cardinality matching, so it is a "
                  "capability gap\n     and not a threshold choice. This is a "
                  "BOUNDARY CONDITION on the\n     cardinality account: "
                  "matching cardinality is necessary but not\n     "
                  "sufficient.")
        else:
            print("  -> the encoder is ahead at 0.5 already; cardinality "
                  "matching is\n     reported for completeness.")
        print()

    print(f"{'=' * 92}\nHOW TO READ THIS")
    print("  The 0.5 column is what goes in the 7.6 table. It was fixed before")
    print("  training and never tuned, because the LLM has no threshold to")
    print("  tune and adjusting only the encoder's would be the same asymmetry")
    print("  as early-stopping on the eval subsets - which 7.5 refused.")
    print("  The other columns diagnose WHY a delta has the sign it does.")
    print(f"{'=' * 92}\n")


if __name__ == "__main__":
    main()
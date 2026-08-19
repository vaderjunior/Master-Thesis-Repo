"""
scripts/check_encoder_vs_llm.py - the Phase 7.6 comparison table.
Read-only, zero API calls, no GPU.

  python -m scripts.check_encoder_vs_llm
  python -m scripts.check_encoder_vs_llm --seeds 0 1 2

WHAT THIS IS FOR, AND WHAT IT IS NOT. Build-guide 7.6 pre-registers the
result: the encoder will probably beat RAG on raw accuracy, plausibly by a
wide margin on hate_type. That is the EXPECTED and HONEST outcome, consistent
with Ghorbanpour et al. 2025 on prompting lagging fine-tuned encoders. Arm A is
the SUPERVISION CEILING, not a competitor. The thesis's claim is about
adaptation cost and 7.10 is where that is measured. This table has to be
readable as a ceiling, which is why the cost columns sit beside the accuracy
ones rather than in a separate section.

WHY VALIDATION MACRO-F1 IS NOT IN THIS TABLE. Each head's validation split is
a random 10% of its NATURAL training distribution; the eval subsets are
LABEL-BALANCED draws from dev. hate_type training runs grievance 1078 against
threatening 475 while en_dev_eval_sq3_types is 68-69 per label, so a balanced
macro weights exactly the labels the model is worst at. Validation numbers are
optimistic and are reported nowhere.

READ EVERY DELTA AGAINST THE RIGHT FLOOR. Two floors matter and they are not
the same quantity:
  LLM floor      spread across sampled runs at T=1.0, decoding noise
  encoder floor  spread across training seeds, initialisation noise
Both are measured here. A delta smaller than EITHER is not evidence. Phase 7
has already corrected two published floors that were taken from the arm under
test at two replicates, so the rule is the widest arm at three or more, quoted
with its replicate count.

PREDICTION CARDINALITY IS A COLUMN, NOT A FOOTNOTE. Section 11 of sq2_log.md:
retrieval tightens the prediction set, which helps where the model
over-predicts (hate_type 2.44 -> 1.77 against gold 1.02, legal 1.23 -> 1.01
against gold 0.55) and hurts where it under-predicts (target_group 1.89 ->
1.68 against gold 2.23). An encoder at a fixed 0.5 threshold has its own
cardinality, and comparing macro-F1 without it compares decision policies as
much as models.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("experiments/results")
PROCESSED = Path("data/processed")
MODELS = Path("models/encoder")

# head -> (subset, dimension, LLM rag stems at the frozen config)
#
# LLM stems are the CURRENT-STATE rag runs only. legal_dev_peasec* sit at KB
# c34e780c and main_peasec_kbv2 at the same, so neither belongs here; the
# comparison is against b3e1a021 with k_definitions flat 2, which 7.3 froze.
SPEC = {
    ("full", "hate"): ("en_dev_eval_main", "gate",
                       ["main_base_r0", "main_base_r0_rep2",
                        "main_base_r0_rep3", "main_base_r0_rep4"]),
    ("full", "target_group"): ("en_dev_eval_targets", "target_group",
                               ["targets_peasec_kbv3",
                                "targets_peasec_kbv3_r2",
                                "targets_peasec_kbv3_r3"]),
    ("full", "hate_type"): ("en_dev_eval_sq3_types", "hate_type",
                            ["sq3_round0_r1", "sq3_round0_r2",
                             "sq3_round0_r3", "sq3_round0_r4"]),
    ("full", "severity"): ("en_dev_eval_severity", "severity",
                           ["severity_kbv3_r1", "severity_kbv3_r2"]),
    ("de", "hate"): ("de_dev_eval", "gate",
                     ["de_dev_kbv3_r1", "de_dev_kbv3_r2"]),
    ("de", "legal"): ("de_legal_dev_eval", "legal",
                      ["de_legal_kbv3_r1", "de_legal_kbv3_r2",
                       "de_legal_kbv3_r3"]),
}
GOLD_COL = {"target_group": "target_groups", "hate_type": "hate_types",
            "legal": "legal"}


def load(stem: str, arm: str | None = None, subset: str | None = None):
    p = RESULTS / f"{stem}_live.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" in r:
            continue
        if arm and r.get("arm") != arm:
            continue
        if subset and (r.get("encoder_meta") or {}).get("subset") != subset:
            continue
        out.append(r)
    return out


def score(rows, gold, dim):
    from src.hsrag.metrics import (score_gate, score_multilabel,
                                   score_severity)
    if not rows:
        return None, None
    if dim == "gate":
        s = score_gate(rows, gold)
        return s.macro_f1, None
    if dim == "severity":
        s = score_severity(rows, gold)
        return s.macro_f1, None
    s = score_multilabel(rows, gold, GOLD_COL[dim], dim)
    return s.macro_f1, s.extra.get("mean_pred_labels")


def fmt(vals):
    ok = [v for v in vals if v is not None]
    if not ok:
        return "-", None, None
    return (" / ".join(f"{v:.3f}" for v in ok),
            sum(ok) / len(ok), max(ok) - min(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print(f"\n{'=' * 96}\nENCODER vs LLM   seeds {args.seeds}\n{'=' * 96}")
    print("\nArm A is the SUPERVISION CEILING, not a competitor. The thesis "
          "claim is adaptation\ncost (7.10), not peak accuracy. Read this "
          "table as a ceiling.\n")

    rows_out = []
    for (arm, head), (subset, dim, llm_stems) in SPEC.items():
        df = pd.read_parquet(PROCESSED / f"{subset}.parquet")
        gold = {str(r.id): r for r in df.itertuples(index=False)}

        # ---------------------------------------------------- the LLM side
        llm_f, llm_c = [], []
        for stem in llm_stems:
            f, c = score(load(stem, arm="rag"), gold, dim)
            if f is not None:
                llm_f.append(f)
            if c is not None:
                llm_c.append(c)

        # ------------------------------------------------ the encoder side
        for e_arm in ([arm] if arm == "de" else [arm, "kb"]):
            enc_f, enc_c = [], []
            for s in args.seeds:
                rows = load(f"encoder_{e_arm}_seed{s}", subset=subset)
                f, c = score(rows, gold, dim)
                if f is not None:
                    enc_f.append(f)
                if c is not None:
                    enc_c.append(c)
            if not enc_f:
                continue
            e_str, e_mean, e_spread = fmt(enc_f)
            l_str, l_mean, l_spread = fmt(llm_f)
            # Training cost, summed across the seeds actually present. This is
            # a cell of the 7.10 table and it only exists because meta.json
            # recorded it at training time; collecting it later means
            # retraining.
            secs, n_train = 0.0, None
            for s in args.seeds:
                mp = MODELS / e_arm / head / f"seed{s}" / "meta.json"
                if mp.exists():
                    m = json.loads(mp.read_text(encoding="utf-8"))
                    secs += m["train_seconds"]
                    n_train = m["n_train"]
            rows_out.append({
                "arm": f"encoder_{e_arm}", "head": head, "subset": subset,
                "n": len(df), "enc": e_str, "enc_mean": e_mean,
                "enc_spread": e_spread,
                "llm": l_str, "llm_mean": l_mean, "llm_spread": l_spread,
                "enc_card": (sum(enc_c) / len(enc_c)) if enc_c else None,
                "llm_card": (sum(llm_c) / len(llm_c)) if llm_c else None,
                "gold_card": (
                    float(np.mean([len(v) if v is not None and
                                   hasattr(v, "__len__") else 0
                                   for v in df[GOLD_COL[dim]]]))
                    if dim in GOLD_COL and GOLD_COL[dim] in df.columns
                    else None),
                "n_train": n_train, "train_s": secs,
            })

    # ------------------------------------------------------------- report
    print(f"{'arm':13} {'head':13} {'n':>5} {'encoder by seed':>24} "
          f"{'mean':>7} {'spr':>6} {'RAG by run':>26} {'mean':>7} "
          f"{'spr':>6} {'delta':>7}")
    print("-" * 96)
    for r in rows_out:
        d = (r["enc_mean"] - r["llm_mean"]) if r["llm_mean"] is not None else None
        lm = f"{r['llm_mean']:.3f}" if r["llm_mean"] is not None else "-"
        ls = f"{r['llm_spread']:.3f}" if r["llm_spread"] is not None else "-"
        dd = f"{d:+.3f}" if d is not None else "-"
        print(f"{r['arm']:13} {r['head']:13} {r['n']:>5} {r['enc']:>24} "
              f"{r['enc_mean']:>7.3f} {r['enc_spread']:>6.3f} "
              f"{r['llm']:>26} {lm:>7} {ls:>6} {dd:>7}")

    print(f"\n{'-' * 96}\nIS THE DELTA READABLE? A delta must clear BOTH "
          f"floors to count.")
    for r in rows_out:
        if r["llm_mean"] is None:
            continue
        d = r["enc_mean"] - r["llm_mean"]
        worst = max(r["enc_spread"], r["llm_spread"] or 0.0)
        verdict = ("READABLE" if abs(d) > worst else
                   "INSIDE THE FLOOR - not evidence")
        print(f"  {r['arm']:13} {r['head']:13} delta {d:+.3f}  "
              f"vs widest floor {worst:.3f}  ->  {verdict}")

    print(f"\n{'-' * 96}\nPREDICTION CARDINALITY  (section 11 of sq2_log.md)")
    print(f"  {'arm':13} {'head':13} {'gold':>6} {'encoder':>8} {'RAG':>8}")
    for r in rows_out:
        if r["gold_card"] is None:
            continue
        ec = f"{r['enc_card']:.2f}" if r["enc_card"] is not None else "-"
        lc = f"{r['llm_card']:.2f}" if r["llm_card"] is not None else "-"
        print(f"  {r['arm']:13} {r['head']:13} {r['gold_card']:>6.2f} "
              f"{ec:>8} {lc:>8}")
    print("\n  Retrieval tightens the prediction set: a gain where the model")
    print("  over-predicts, a loss where it under-predicts. The encoder at a")
    print("  fixed 0.5 threshold has its own cardinality, and macro-F1 "
          "compares\n  decision policies as much as models without this "
          "column.")

    print(f"\n{'-' * 96}\nTRAINING COST  (feeds the 7.10 table)")
    print(f"  {'arm':13} {'head':13} {'n_train':>8} {'train_s':>9} "
          f"{'per seed':>9}")
    tot = 0.0
    for r in rows_out:
        tot += r["train_s"]
        print(f"  {r['arm']:13} {r['head']:13} "
              f"{(r['n_train'] or 0):>8} {r['train_s']:>9.1f} "
              f"{r['train_s'] / max(len(args.seeds), 1):>9.1f}")
    print(f"  {'TOTAL':13} {'':13} {'':>8} {tot:>9.1f} "
          f"({tot / 3600:.2f} GPU-hours)")
    print("\n  The LLM's column in 7.10 is ZERO training seconds and ZERO new")
    print(f"  labelled examples. That contrast is the thesis's claim made\n"
          f"  concrete.\n{'=' * 96}\n")


if __name__ == "__main__":
    main()
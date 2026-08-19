"""
scripts/check_new_label.py - Phase 7.10, the adaptability comparison.
Read-only, zero API calls, no GPU.

  python -m scripts.check_new_label

THE CENTREPIECE. Phase 8 measured what it costs to give the system a label it
was not configured for: remove `hate_type/grievance` from taxonomy.yaml,
restore it, and measure the difference with the LLM frozen.

  zero_shot   0.276 -> 0.337   +0.060   p = 0.0008   33-0 on the 68 target items
  rag         0.301 -> 0.357   +0.057   p = 0.104    43-0 on the 68

The edit is ONE LINE in a configuration file and no retraining. The encoder
cannot do this at all: a classification head's output dimension is fixed at
training time, so adding a label is a retrain from scratch. This script runs
the identical manipulation on the encoder and prices it.

WHAT IS HELD CONSTANT, AND WHY IT MATTERS. The six-label arm borrows the gate,
target_group and severity heads from the seven-label arm at the SAME SEED, so
they are byte-identical. That is not convenience: the gate costs hate_type
-0.092 on this subset through gate_consistency, so an unmatched gate would
swamp the effect being measured. The ONLY difference between the arms is the
hate_type output space.

THE 68 GRIEVANCE-GOLD ITEMS ARE SCORED, NOT SKIPPED. Their gold stays
['grievance'], the six-label arm cannot predict it, and it is penalised for the
miss - exactly as the LLM was when the label was removed from its label space.
Skipping them would give the encoder an advantage the LLM never had, and the
LLM's +0.060 exists precisely BECAUSE those items were scored: the arithmetic
only closes over seven labels with grievance at F1 0.000, not over six.

THE MECHANICAL MINIMUM IS THE THING TO CHECK. A seventh label at F1 x raises a
six-label macro by x/7 if nothing else changes. Phase 8 found the LLM's
observed gain sat almost exactly there (0.378/7 = 0.054 against +0.060), and
read it as the label space being close to modular rather than a set of mutually
competing options. Whether the encoder behaves the same way is the interesting
question, because the encoder has no fixed label budget to displace from.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("experiments/results")
PROCESSED = Path("data/processed")
MODELS = Path("models/encoder")
SUBSET = "en_dev_eval_sq3_types"
DIM, GOLD_COL = "hate_type", "hate_types"
SEEDS = [0, 1, 2]

# Transcribed from PHASE8_SIGNOFF section 3.13. Two arms, only one a clean
# pathway test: zero_shot receives no examples and no retrieved context, so
# removing the label changes exactly ONE thing, the label space. rag
# additionally loses the definition and 10 labelled examples.
LLM = {
    "zero_shot": {"removed": [0.276, 0.275], "restored": [0.337, 0.336, 0.334],
                  "grievance_f1": 0.378, "better": 77, "worse": 40,
                  "p": 0.0008, "target_better": 33, "target_worse": 0},
    "rag": {"removed": [0.292, 0.309],
            "restored": [0.365, 0.344, 0.358, 0.362],
            "grievance_f1": 0.411, "better": 71, "worse": 52,
            "p": 0.104, "target_better": 43, "target_worse": 0},
}


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def load(stem):
    p = RESULTS / f"{stem}_live.jsonl"
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" in r:
            continue
        if (r.get("encoder_meta") or {}).get("subset") != SUBSET:
            continue
        out[r["item_id"]] = r
    return out


def main():
    from scipy.stats import binomtest
    from src.hsrag.metrics import score_multilabel

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    gold = {str(r.id): set(clean(getattr(r, GOLD_COL, None)) or [])
            for r in df.itertuples(index=False)
            if clean(getattr(r, GOLD_COL, None)) is not None and bool(r.gate)}
    grievance_ids = sorted(i for i, g in gold.items() if "grievance" in g)

    print(f"\n{'=' * 88}\n7.10  THE NEW-LABEL COMPARISON\n{'=' * 88}")
    print(f"\n{SUBSET}: {len(gold)} scorable items, "
          f"{len(grievance_ids)} with grievance gold")
    print("\nArm A is the SUPERVISION CEILING. This section is not about which")
    print("system is more accurate - it is about what it COSTS to add a label.")

    # ------------------------------------------------------- macro-F1
    print(f"\n{'-' * 88}\nMACRO-F1 over 7 labels, grievance included in both "
          f"arms")
    print("  The six-label arm scores grievance at F1 0.000 on 68 gold items,")
    print("  exactly as the LLM did when the label was removed from its label")
    print("  space. Skipping those items would hand the encoder an advantage.")
    res = {}
    for tag, label in (("", "7-label (restored)"),
                       ("_nogrievance", "6-label (removed)")):
        vals, per_lab, cards = [], defaultdict(list), []
        for s in SEEDS:
            rows = list(load(f"encoder_full{tag}_seed{s}").values())
            sc = score_multilabel(rows, {str(r.id): r for r in
                                         df.itertuples(index=False)},
                                  GOLD_COL, DIM)
            vals.append(sc.macro_f1)
            cards.append(sc.extra["mean_pred_labels"])
            for l, c in sc.per_label.items():
                per_lab[l].append(c["f1"])
        res[tag] = {"vals": vals, "per": per_lab, "card": cards}
        print(f"  {label:22} {' / '.join(f'{v:.3f}' for v in vals)}"
              f"   mean {np.mean(vals):.3f}   spread "
              f"{max(vals) - min(vals):.3f}   card "
              f"{np.mean(cards):.2f}")

    enc_delta = float(np.mean(res[""]["vals"]) -
                      np.mean(res["_nogrievance"]["vals"]))
    paired = [a - b for a, b in zip(res[""]["vals"],
                                    res["_nogrievance"]["vals"])]
    print(f"\n  ENCODER DELTA          {enc_delta:+.3f}   "
          f"per seed {' / '.join(f'{d:+.3f}' for d in paired)}")
    for arm, d in LLM.items():
        ld = float(np.mean(d["restored"]) - np.mean(d["removed"]))
        print(f"  LLM {arm:11}        {ld:+.3f}   "
              f"(p = {d['p']}, {d['target_better']}-{d['target_worse']} on "
              f"the {len(grievance_ids)} target items)")

    # -------------------------------------------- the mechanical minimum
    g_f1 = float(np.mean(res[""]["per"].get("grievance", [0.0])))
    print(f"\n{'-' * 88}\nIS THE GAIN MECHANICAL?")
    print("  A 7th label at F1 x raises a 6-label macro by x/7 if nothing else")
    print("  changes. Phase 8 found the LLM almost exactly there and read it as")
    print("  the label space being close to MODULAR rather than a set of")
    print("  mutually competing options.")
    print(f"  encoder grievance F1 {g_f1:.3f}  ->  mechanical minimum "
          f"{g_f1 / 7:+.3f}   observed {enc_delta:+.3f}   "
          f"excess {enc_delta - g_f1 / 7:+.3f}")
    for arm, d in LLM.items():
        ld = float(np.mean(d["restored"]) - np.mean(d["removed"]))
        print(f"  LLM {arm:11} F1 {d['grievance_f1']:.3f}  ->  "
              f"{d['grievance_f1'] / 7:+.3f}   observed {ld:+.3f}   "
              f"excess {ld - d['grievance_f1'] / 7:+.3f}")

    # --------------------------------------------------------- per label
    print(f"\n{'-' * 88}\nPER LABEL, mean over {len(SEEDS)} seeds")
    print(f"  {'label':16} {'6-label':>9} {'7-label':>9} {'delta':>8} "
          f"{'floor':>8}")
    labels = sorted(set(res[""]["per"]) | set(res["_nogrievance"]["per"]))
    for l in labels:
        a = res["_nogrievance"]["per"].get(l, [0.0])
        b = res[""]["per"].get(l, [0.0])
        fl = max(max(a) - min(a), max(b) - min(b))
        mark = "  <- the restored label" if l == "grievance" else ""
        print(f"  {l:16} {np.mean(a):>9.3f} {np.mean(b):>9.3f} "
              f"{np.mean(b) - np.mean(a):>+8.3f} {fl:>8.3f}{mark}")

    # DISPLACEMENT, tested per seed and paired rather than read off the means.
    # The arms share three borrowed heads and the same items, so the only
    # thing varying between the paired deltas is the seed - which makes a
    # paired t over seeds the right instrument and sign consistency alone
    # (p = 0.125 at three seeds) far too weak to lean on.
    #
    # This is the measurement behind the negative excess above. It should not
    # happen: the LLM displaces because it has a fixed label budget of ~1.7
    # per item, but the encoder has seven INDEPENDENT sigmoids and no budget.
    # The mechanism has to be the shared trunk - all seven output units read
    # one roberta-base representation, and training with grievance in the
    # label set reshapes it.
    six = [l for l in labels if l != "grievance"]
    per_seed = []
    for k in range(len(SEEDS)):
        a = float(np.mean([res["_nogrievance"]["per"][l][k] for l in six
                           if l in res["_nogrievance"]["per"]]))
        b = float(np.mean([res[""]["per"][l][k] for l in six
                           if l in res[""]["per"]]))
        per_seed.append(b - a)
    md, sd = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
    se = sd / np.sqrt(len(per_seed))
    print(f"\n  DISPLACEMENT onto the other six labels")
    print(f"    per seed  {' / '.join(f'{d:+.4f}' for d in per_seed)}")
    print(f"    mean {md:+.4f}   SD {sd:.4f}   SE {se:.4f}   "
          f"t({len(per_seed) - 1}) = {md / se:.1f}")
    print(f"    magnitude / spread = "
          f"{abs(md) / max(max(per_seed) - min(per_seed), 1e-9):.1f}x")
    print(f"    sum of the six deltas / 7 = "
          f"{sum(np.mean(res['']['per'][l]) - np.mean(res['_nogrievance']['per'][l]) for l in six if l in res['']['per']) / 7:+.4f}"
          f"   <- reconciles the excess above")

    # ------------------------------------- the 68 items, paired per item
    print(f"\n{'-' * 88}\nTHE {len(grievance_ids)} GRIEVANCE-GOLD ITEMS, "
          f"paired per item")
    print("  Correctness under `contains` (gold subset of predicted), the same")
    print("  criterion every SQ3 measurement used. The LLM was 33-0 (zero_shot)")
    print("  and 43-0 (rag): perfect one-sided separation on the target label.")
    tot_b = tot_w = 0
    for s in SEEDS:
        A = load(f"encoder_full_seed{s}")             # restored
        B = load(f"encoder_full_nogrievance_seed{s}")  # removed
        b = w = 0
        for i in grievance_ids:
            if i not in A or i not in B:
                continue
            ok_a = gold[i] <= set(A[i]["result"][DIM])
            ok_b = gold[i] <= set(B[i]["result"][DIM])
            b += ok_a and not ok_b
            w += ok_b and not ok_a
        tot_b, tot_w = tot_b + b, tot_w + w
        print(f"  seed {s}   restored better on {b:>3}, removed better on "
              f"{w:>3}")
    # PER SEED, and the pooled figure is labelled non-independent. The three
    # seeds score THE SAME 68 items, so pooling triples n artificially and the
    # pooled p is not a valid significance statement. Per seed is already
    # overwhelming: 29-0 is p = 1.9e-9.
    p = binomtest(tot_b, tot_b + tot_w, 0.5).pvalue if tot_b + tot_w else 1.0
    print(f"  pooled    {tot_b}-{tot_w}, nominal p = {p:.4g}  <- NOT a valid "
          f"significance test:")
    print(f"            the three seeds score the same {len(grievance_ids)} "
          f"items, so n is tripled. Quote a")
    print(f"            per-seed count instead; each is already p < 1e-6.")

    # --------------------------------------------------------- orphaning
    print(f"\n{'-' * 88}\nORPHANING: where do the {len(grievance_ids)} items "
          f"go when grievance is unavailable?")
    print("  The LLM did not leave them unlabelled: 60 of 68 (zero_shot) and 54")
    print("  (rag) went to `stereotyping`. It has an opinion about these texts")
    print("  and lacks the word for it.")
    orph = Counter()
    for s in SEEDS:
        B = load(f"encoder_full_nogrievance_seed{s}")
        for i in grievance_ids:
            if i in B:
                for l in B[i]["result"][DIM] or ["(nothing predicted)"]:
                    orph[l] += 1
    n = len(SEEDS)
    for l, c in orph.most_common():
        print(f"  {l:24} {c / n:>6.1f} of {len(grievance_ids)} items "
              f"({c / n / len(grievance_ids):>5.1%})")

    # -------------------------------------------------------- cost table
    print(f"\n{'=' * 88}\nTHE COST OF ADDING ONE LABEL\n{'=' * 88}")
    m7 = json.loads((MODELS / "full" / "hate_type" / "seed0" /
                     "meta.json").read_text(encoding="utf-8"))
    m6 = json.loads((MODELS / "full_nogrievance" / "hate_type" / "seed0" /
                     "meta.json").read_text(encoding="utf-8"))
    d7 = json.loads((Path("data/encoder/full") /
                     "hate_type_meta.json").read_text(encoding="utf-8"))
    d6 = json.loads((Path("data/encoder/full_nogrievance") /
                     "hate_type_meta.json").read_text(encoding="utf-8"))
    n_ex = (d7["n_train"] + d7["n_val"]) - (d6["n_train"] + d6["n_val"])
    secs = sum(json.loads((MODELS / "full" / "hate_type" / f"seed{s}" /
                           "meta.json").read_text(encoding="utf-8")
                          )["train_seconds"] for s in SEEDS)
    print(f"  {'':34} {'LLM + RAG':>22} {'Encoder':>22}")
    rows = [
        ("edit required", "one line in taxonomy.yaml",
         "change output dim, retrain"),
        ("new labelled examples", "0", f"{n_ex}"),
        ("training time", "0 s", f"{secs:.0f} s over {len(SEEDS)} seeds"),
        ("inference to evaluate", "~26 min (1401 calls)", "~2.5 s"),
        ("artefacts invalidated", "none", f"{len(SEEDS)} checkpoints"),
        ("macro-F1 gained", f"+0.060 / +0.057", f"{enc_delta:+.3f}"),
    ]
    for k, a, b in rows:
        print(f"  {k:34} {a:>22} {b:>22}")
    print(f"\n  The macro-F1 row is the point: BOTH systems gain roughly the")
    print(f"  same amount. The difference is entirely in the rows above it.")
    print(f"\n  CAVEAT, from the sign-off's own list of what to attack:")
    print(f"  `grievance` is not literally new to the model - white-grievance")
    print(f"  discourse is common in English and certainly in pretraining. This")
    print(f"  measures A LABEL THE SYSTEM WAS NOT CONFIGURED FOR, not one the")
    print(f"  model had never seen, and the design cannot distinguish the two.")
    print(f"{'=' * 88}\n")


if __name__ == "__main__":
    main()
"""
scripts/check_guideline_edit.py - the two links of the guideline-edit study.
Read-only, zero API calls.
Run: python -m scripts.check_guideline_edit

WHY TWO LINKS AND NOT ONE NUMBER. Rewording a guideline changes its embedding,
so it changes RETRIEVAL as well as salience. If the reworded version reaches
more items, a drop in false positives is "shown more often" rather than "obeyed
more when shown", and the two cannot be told apart from an aggregate. Phase 6
established this twice - def-target_group-other reached 6 of 16 items at
k_definitions=7 and was used on 1 of those 6 - and reporting one link without
the other is what made those results unreadable until they were split.

  Link 1  retrieval: on how many items does the guideline appear at all?
  Link 2  prediction: on the items where it appears under BOTH knowledge
          bases, does the answer change?

Link 2 runs on the intersection precisely so that a change in link 1 cannot be
mistaken for a change in adherence.

THE GUIDELINE IS DECISIVE AND DIRECTIONAL. guide-profanity-without-target
implies gate=False when it applies. So it should help on gold NOT-HATE items
and, if over-applied, hurt on gold HATE items. Accuracy is therefore reported
SPLIT BY GOLD LABEL, never pooled: retrieval fetches guidelines by resemblance
rather than by applicability, and a pooled rate would average the cases the
guideline addresses together with the cases where it is simply noise.

BOTH BASELINE REPLICATES ARE USED. On this subset a difference of one or two
items is inside replicate noise, so a single baseline would not support a
claim either way.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from scripts.check_sq3_coverage import PROCESSED
from src.hsrag.metrics import effective_gate

RESULTS = Path("experiments/results")
TARGET = "guide-profanity-without-target"
SUBSET = "en_dev_eval_main"
BASE = ["main_base_r0", "main_base_r0_rep2"]
EDIT = "main_guide_directive"


def load(stem: str) -> dict:
    path = RESULTS / f"{stem}_live.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r and r.get("arm") == "rag":
            out[r["item_id"]] = r
    return out


def retrieved_on(res: dict) -> set:
    return {i for i, r in res.items()
            if TARGET in (r.get("retrieved", {}).get("guidelines") or [])}


def correct(res: dict, gold: dict, ids, mapping: str) -> dict:
    """item -> was the gate right, or None where it cannot be scored."""
    out = {}
    for i in ids:
        r = res.get(i)
        truth = effective_gate(gold[i], mapping)
        if r is None or r.get("result") is None or truth is None:
            out[i] = None
            continue
        out[i] = bool(r["result"]["hate"]) == truth
    return out


def main():
    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    gold = {str(r.id): r for r in df.itertuples(index=False)}
    runs = {s: load(s) for s in BASE + [EDIT]}

    print(f"\n{'=' * 74}\nGUIDELINE EDIT: {TARGET}")
    print(f"  subset {SUBSET}, {len(df)} items")
    print(f"  base   {', '.join(BASE)}  (kb b3e1a021777f7b85)")
    print(f"  edit   {EDIT}  (kb 9149ce2f7c48bfa0, one record's text changed)")

    # ------------------------------------------------------------- link 1
    got = {s: retrieved_on(r) for s, r in runs.items()}
    print(f"\n{'-' * 74}\nLINK 1 - RETRIEVAL")
    for s in BASE + [EDIT]:
        benign = sum(1 for i in got[s] if gold[i].gate is False)
        print(f"  {s:22} retrieved on {len(got[s]):4} items "
              f"({benign} gold not-hate, where it applies)")

    base_ids = set.intersection(*[got[s] for s in BASE])
    both = base_ids & got[EDIT]
    print(f"\n  retrieved under BOTH base replicates: {len(base_ids)}")
    print(f"  and also under the edit:               {len(both)}")
    print(f"  base-only {len(base_ids - got[EDIT]):4}   "
          f"edit-only {len(got[EDIT] - base_ids):4}")
    if not both:
        print("\n  No intersection - link 2 cannot be computed.")
        return

    # ------------------------------------------------------------- link 2
    for mapping in ("strict", "lenient"):
        print(f"\n{'-' * 74}\nLINK 2 - PREDICTION on the {len(both)}-item "
              f"intersection  [{mapping}]")
        cor = {s: correct(runs[s], gold, both, mapping)
               for s in BASE + [EDIT]}

        for side, want in (("gold not-hate (guideline applies)", False),
                           ("gold hate (guideline is noise here)", True)):
            ids = [i for i in both
                   if effective_gate(gold[i], mapping) is want]
            if not ids:
                continue
            print(f"\n  {side}, n={len(ids)}")
            for s in BASE + [EDIT]:
                ok = sum(1 for i in ids if cor[s][i])
                n = sum(1 for i in ids if cor[s][i] is not None)
                print(f"    {s:22} {ok:3}/{n:3}  {ok / max(n, 1):.3f}")

        # Paired against each baseline replicate separately. Pooling the two
        # baselines would treat one system state as two independent
        # observations of the item.
        print(f"\n  paired, edit vs each baseline replicate:")
        for s in BASE:
            a, b = cor[EDIT], cor[s]
            only_e = sum(1 for i in both if a[i] and b[i] is False)
            only_b = sum(1 for i in both if b[i] and a[i] is False)
            disc = only_e + only_b
            p = binomtest(only_e, disc, 0.5).pvalue if disc else 1.0
            print(f"    vs {s:22} only edit {only_e:3}  only base {only_b:3}  "
                  f"discordant {disc:3}  p = {p:.4f}")

    # -------------------------------------------------- whole-subset context
    print(f"\n{'-' * 74}\nWHOLE SUBSET, for context (strict)")
    print(f"  {'run':22} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    for s in BASE + [EDIT]:
        c = {k: 0 for k in ("TP", "FP", "FN", "TN")}
        for i, g in gold.items():
            r = runs[s].get(i)
            truth = effective_gate(g, "strict")
            if r is None or r.get("result") is None or truth is None:
                continue
            pred = bool(r["result"]["hate"])
            c["TP" if truth and pred else "FP" if pred else
              "FN" if truth else "TN"] += 1
        print(f"  {s:22} {c['TP']:4} {c['FP']:4} {c['FN']:4} {c['TN']:4}")

    print(f"\n  A drop in false positives that appears in link 2 on the "
          f"intersection is\n  adherence. One that appears only in the whole-"
          f"subset totals while link 1\n  shows the edit reaching more items "
          f"is reach, not adherence.\n{'=' * 74}\n")


if __name__ == "__main__":
    main()
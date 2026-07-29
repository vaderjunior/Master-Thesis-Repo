"""
scripts/analyse_slice1.py - diagnose Slice 1. No API calls.

The headline number (gate macro-F1 ~0.71) hides the actual finding: false
positives outnumber false negatives roughly 5 to 1, in EVERY arm including
zero_shot, which uses no retrieval at all. So the over-flagging is not caused
by retrieval, and Finding E is not the explanation. This script asks what is.

  python -m scripts.analyse_slice1
  python -m scripts.analyse_slice1 --show 4
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RESULTS = Path("experiments/results")


def clean(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="slice1_en_dev_eval_main_live.jsonl")
    ap.add_argument("--subset", default="en_dev_eval_main")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_parquet(Path("data/processed") / f"{args.subset}.parquet").head(args.n)
    gold = {str(r.id): r for r in df.itertuples(index=False)}

    rows = [json.loads(l) for l in
            (RESULTS / args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_arm = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["item_id"]] = r
    arms = ["zero_shot", "few_shot", "rag"]

    # --- 1. where do the false positives live? -----------------------------
    # If they concentrate in hatexplain, the cause is the offensive -> not-hate
    # mapping (a gold-definition choice), not model over-sensitivity.
    print(f"\n{'=' * 72}\nGATE ERRORS BY SOURCE DATASET\n{'=' * 72}")
    benign = Counter(g.source for g in gold.values() if not g.gate)
    hateful = Counter(g.source for g in gold.values() if g.gate)
    print(f"{'source':16} {'benign':>7} {'hateful':>8}   " +
          "  ".join(f"{a:>14}" for a in arms))
    print(f"{'':16} {'':>7} {'':>8}   " +
          "  ".join(f"{'FP    FN':>14}" for _ in arms))
    for src in sorted(benign | hateful):
        cells = []
        for arm in arms:
            fp = sum(1 for i, r in by_arm[arm].items()
                     if gold[i].source == src and not gold[i].gate
                     and r["result"] and r["result"]["hate"])
            fn = sum(1 for i, r in by_arm[arm].items()
                     if gold[i].source == src and gold[i].gate
                     and r["result"] and not r["result"]["hate"])
            rate = fp / benign[src] * 100 if benign[src] else 0
            cells.append(f"{fp:3} ({rate:3.0f}%) {fn:3}")
        print(f"{src:16} {benign[src]:7} {hateful[src]:8}   " + "  ".join(cells))

    # --- 2. is the model over-predicting labels generally? -----------------
    print(f"\n{'=' * 72}\nLABELS PREDICTED PER HATEFUL ITEM (vs gold)\n{'=' * 72}")
    for field, col in [("target_group", "target_groups"), ("hate_type", "hate_types")]:
        g_n = [len(clean(g.__getattribute__(col)) or [])
               for g in gold.values()
               if g.gate and clean(g.__getattribute__(col)) is not None]
        line = f"  {field:14} gold {sum(g_n) / max(len(g_n), 1):.2f}"
        for arm in arms:
            p_n = [len(r["result"][field]) for i, r in by_arm[arm].items()
                   if r["result"] and r["result"]["hate"]
                   and clean(gold[i].__getattribute__(col)) is not None]
            line += f"   {arm} {sum(p_n) / max(len(p_n), 1):.2f}"
        print(line)

    # --- 3. where do the arms actually disagree? ---------------------------
    print(f"\n{'=' * 72}\nARM DISAGREEMENT ON THE GATE\n{'=' * 72}")
    common = set.intersection(*(set(by_arm[a]) for a in arms))
    preds = {a: {i: by_arm[a][i]["result"]["hate"]
                 for i in common if by_arm[a][i]["result"]} for a in arms}
    agree = sum(1 for i in common
                if len({preds[a].get(i) for a in arms}) == 1)
    print(f"  all three agree on {agree}/{len(common)} items "
          f"({agree / len(common):.0%})")
    for a in arms:
        for b in arms:
            if a < b:
                same = sum(1 for i in common if preds[a].get(i) == preds[b].get(i))
                print(f"  {a:10} vs {b:10} {same / len(common):.0%}")

    # --- 4. qualitative: rag right, zero_shot wrong, and the reverse -------
    for label, right, wrong in [("RAG WINS", "rag", "zero_shot"),
                                ("RAG LOSES", "zero_shot", "rag")]:
        print(f"\n{'=' * 72}\n{label} (gate)\n{'=' * 72}")
        cases = [i for i in common
                 if preds[right].get(i) == gold[i].gate
                 and preds[wrong].get(i) != gold[i].gate]
        print(f"  {len(cases)} such items\n")
        for i in cases[:args.show]:
            g = gold[i]
            r = by_arm["rag"][i]
            print(f"  --- {i} [{g.source}]  gold hate={bool(g.gate)} ---")
            print(f"      {' '.join(str(g.text).split())[:160]}")
            print(f"      rag reasoning: {r['result']['reasoning'][:160]}")
            print(f"      retrieved: "
                  f"defs={r['retrieved'].get('definitions')} "
                  f"guides={r['retrieved'].get('guidelines')}")
            print()

    # --- 5. parse failures: load or content? -------------------------------
    print(f"\n{'=' * 72}\nPARSE FAILURES\n{'=' * 72}")
    for arm in arms:
        bad = [r for r in by_arm[arm].values() if r["parse_failures"]]
        print(f"  {arm:10} {sum(r['parse_failures'] for r in bad):3} failed runs "
              f"across {len(bad):3} items   "
              f"sources={dict(Counter(gold[r['item_id']].source for r in bad))}")


if __name__ == "__main__":
    main()
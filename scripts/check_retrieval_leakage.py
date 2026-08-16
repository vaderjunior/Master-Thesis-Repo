"""
scripts/check_retrieval_leakage.py - did a KB near-duplicate actually reach the
prompt, and does removing the affected items change the result?
Read-only, zero API calls.

  python -m scripts.check_retrieval_leakage --subset de_legal_dev_eval \
      --dim legal --results legal_dev_peasec legal_dev_peasec_r2 \
      legal_dev_peasec_r3 de_legal_kbv3_r1 de_legal_kbv3_r2

WHAT THIS IS FOR. check_leakage --only kb_example found that 6 of the 175
items in de_legal_dev_eval have a near-copy of themselves in kb/records.jsonl,
one at cosine 0.9977. Because _render_gold prints an example's gold labels
into the prompt, a retrieved near-copy hands the model the answer to the item
it is being asked about.

BUT PRESENCE IS NOT RETRIEVAL. The KB holds 320 examples and retrieval takes
5. A 0.9977 twin will probably win a dense slot, and "probably" is not a
measurement. Every ItemResult stores the ids it actually retrieved, so this
costs nothing and settles it. Assuming it rather than checking it would be the
same error as the sorted-id retrieval comparison in Phase 8, which could not
see a reordering and made an uninterpretable FP drop look real.

HOW THE HOLE OPENED, because the fix belongs at the cause. Rule 1 says KB
examples come from TRAIN only, and build_kb_records asserts exactly that. The
assert is correct and it passed. BoTox re-annotates DeTox tweets, so one tweet
exists as detox-<id> in de_train (legitimately TRAIN for the German gate
splits) and as botox-<id> in de_legal_dev (held out for the legal dimension).
The assert checks the split an example was drawn from; it cannot see every
other split the same text also appears in. Exact-text intersection confirms
it: de_legal_train x de_train share 10 texts, de_legal_dev x de_train share 2.

WHY THE KB IS NOT REBUILT. Editing kb/records.jsonl changes kb_version and
puts every one of the 68 runs in the comparability ledger into a different
group from every future run. The affected items are few, identified, and can
be excluded at scoring time, which costs nothing and destroys nothing. The
number to report is both: with and without.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.check_sq3_coverage import PROCESSED
from src.hsrag.metrics import score_gate, score_multilabel

RESULTS = Path("experiments/results")
GOLD_COL = {"legal": "legal", "hate_type": "hate_types",
            "target_group": "target_groups"}


def retrieved_ids(row: dict) -> set:
    """Every record id that reached this item's prompt, across all buckets.

    Deliberately flattens rather than reading only `examples`: a near-copy
    could in principle be stored under another kind, and a leak found in the
    wrong bucket is still a leak. Defensive about the container type because
    the stamp's shape predates this script.
    """
    ret = row.get("retrieved") or {}
    if isinstance(ret, dict):
        out = set()
        for v in ret.values():
            if isinstance(v, list):
                out |= {str(x) for x in v}
        return out
    if isinstance(ret, list):
        return {str(x) for x in ret}
    return set()


def load(stem: str) -> list:
    p = RESULTS / f"{stem}_live.jsonl"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--dim", required=True,
                    choices=["legal", "hate_type", "target_group", "gate"])
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--report", default="experiments/leakage_report_kb_example.json")
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    entry = rep["subsets"].get(args.subset)
    if entry is None:
        print(f"{args.subset} not in {args.report}")
        return
    flagged = {t["eval_id"]: t for t in entry["top"]
               if t["cosine"] >= args.threshold}

    df = pd.read_parquet(PROCESSED / f"{args.subset}.parquet")
    gold = {str(r.id): r for r in df.itertuples(index=False)}

    print(f"\n{'=' * 78}\nRETRIEVAL LEAKAGE  {args.subset}  dim={args.dim}")
    print(f"{'=' * 78}")
    print(f"  {len(df)} items, {len(flagged)} with a KB near-copy at "
          f"cosine >= {args.threshold}")
    if not flagged:
        print("  nothing flagged; this subset is clean on the KB side.\n")
        return
    for eid, t in sorted(flagged.items(), key=lambda kv: -kv[1]["cosine"]):
        print(f"    {eid:28} {t['cosine']:.4f}  {t['train_id']}")

    # ------------------------------------------------ was it retrieved?
    print(f"\n{'-' * 78}\nDID THE TWIN ACTUALLY REACH THE PROMPT?")
    print("  Only retrieval-using arms can leak. zero_shot and few_shot get")
    print("  no retrieved context at all, so they are the control by")
    print("  construction.\n")
    hit_total = {}
    for stem in args.results:
        rows = load(stem)
        arms = sorted({r["arm"] for r in rows})
        for arm in arms:
            by_id = {r["item_id"]: r for r in rows if r["arm"] == arm}
            hits, checked = [], 0
            for eid, t in flagged.items():
                r = by_id.get(eid)
                if r is None:
                    continue
                checked += 1
                if str(t["train_id"]) in retrieved_ids(r):
                    hits.append(eid)
            hit_total[(stem, arm)] = hits
            flag = "  <-- LEAK" if hits else ""
            print(f"  {stem:24} {arm:10} {len(hits)}/{checked} "
                  f"twin(s) retrieved{flag}")
            for eid in hits:
                print(f"      {eid}  <- {flagged[eid]['train_id']}")

    leaked = sorted({e for v in hit_total.values() for e in v})
    print(f"\n  union across all runs and arms: {len(leaked)} item(s) "
          f"actually served their own twin")
    if not leaked:
        print("  -> presence in the KB, but retrieval never selected it.")
        print("     The exposure is bounded at zero for these runs. Say so")
        print("     explicitly rather than leaving it as an open worry.")

    # ------------------------------------------------- does it matter?
    drop = set(leaked) if leaked else set(flagged)
    label = ("actually-retrieved" if leaked else
             "KB-near-copy (retrieval unconfirmed, conservative)")
    print(f"\n{'-' * 78}\nRE-SCORED WITHOUT THE {len(drop)} "
          f"{label} ITEM(S)")
    gold_clean = {k: v for k, v in gold.items() if k not in drop}

    for stem in args.results:
        rows = load(stem)
        for arm in sorted({r["arm"] for r in rows}):
            sub = [r for r in rows if r["arm"] == arm]
            if args.dim == "gate":
                a = score_gate(sub, gold)
                b = score_gate(sub, gold_clean)
                va = a.macro_f1 if a.macro_f1 is not None else float("nan")
                vb = b.macro_f1 if b.macro_f1 is not None else float("nan")
                print(f"  {stem:24} {arm:10} macro {va:.3f} -> {vb:.3f}  "
                      f"({vb - va:+.3f})")
            else:
                col = GOLD_COL[args.dim]
                a = score_multilabel(sub, gold, col, args.dim)
                b = score_multilabel(sub, gold_clean, col, args.dim)
                fa = a.macro_f1 or 0.0
                fb = b.macro_f1 or 0.0
                print(f"  {stem:24} {arm:10} "
                      f"macro {fa:.3f} -> {fb:.3f} ({fb - fa:+.3f})   "
                      f"exact {a.subset_accuracy:.3f} -> "
                      f"{b.subset_accuracy:.3f} "
                      f"({b.subset_accuracy - a.subset_accuracy:+.3f})   "
                      f"hamming {a.hamming_loss:.3f} -> "
                      f"{b.hamming_loss:.3f}")

    print(f"\n{'=' * 78}\nHOW TO READ THIS")
    print("  A drop concentrated in the retrieval arm is the signature of")
    print("  leakage: only that arm could have been shown the answer. A drop")
    print("  in every arm is the items being easy, which is a sampling")
    print("  property and not a leak. Report the affected ids either way -")
    print("  the exposure is a fact about the data whether or not it moved")
    print(f"  the number.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
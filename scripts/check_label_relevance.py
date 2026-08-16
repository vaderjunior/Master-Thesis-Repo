"""
scripts/check_label_relevance.py - is retrieval label-relevant, per dimension?
Read-only, zero API calls.
Run: python -m scripts.check_label_relevance

THE CLAIM THIS TESTS. SQ3 argues that embedding similarity encodes SUBJECT
MATTER while `hate_type` encodes RHETORICAL MODE, so similarity-ranked
retrieval is structurally misaligned with that label scheme and corrections
cannot transfer. As stated in the chapter that is an argument about one
dimension. It becomes a measurement if the same diagnostic is applied to a
TOPICAL dimension and comes out differently.

`target_group` is topical: race, religion, gender, national origin are subject
matter, and its definitions are lexically loaded with the group names that
appear verbatim in the text. `hate_type` is not: irony, grievance, inferiority
are manners of speaking, and two ironic posts about different subjects sit far
apart in embedding space while an ironic post and a direct threat about the
same subject sit close.

PREDICTION, stated before running: target_group well above chance, hate_type
near it. If that holds, the topical/non-topical distinction is supported on two
dimensions rather than argued from one, and the label-relevance measurement
becomes a PRE-FLIGHT DIAGNOSTIC - run it on a taxonomy before building a
feedback loop and it says whether to expect transfer.

CHANCE IS PER ITEM, not a global constant. An item whose gold label is common
in the KB pool is likely to draw a sharing example by luck; one whose label is
rare is not. Chance is therefore computed per item from the pool composition
and then averaged, and examples that never annotate the dimension stay in the
denominator because a retrieved example carrying None cannot hit.

This runs on the SAME `retrieved` hit ids the model actually saw, so it
measures the retrieval that produced the reported numbers rather than a
re-run.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.apply_feedback import load_results
from scripts.check_sq3_coverage import PROCESSED
from src.hsrag.metrics import clean

KB_RECORDS = Path("kb") / "records.jsonl"
# (run stem, subset, dimension, gold column)
CASES = [
    ("sq3_round0_r1", "en_dev_eval_sq3_types", "hate_type", "hate_types"),
    ("sq3_round0_r4", "en_dev_eval_sq3_types", "hate_type", "hate_types"),
    ("targets_peasec_kbv3", "en_dev_eval_targets", "target_group",
     "target_groups"),
    ("targets_kdef7", "en_dev_eval_targets", "target_group", "target_groups"),
]
EXAMPLE_GROUP = "Labelled examples"


def kb_examples(lang: str = "en") -> dict:
    """id -> meta, for retrievable example records of one language."""
    out = {}
    for line in KB_RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["kind"] == "example" and r["lang"] == lang:
            out[r["id"]] = r.get("meta", {})
    return out


def run_case(stem: str, subset: str, dim: str, col: str) -> dict | None:
    path = PROCESSED / f"{subset}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    gold = {}
    for r in df.itertuples(index=False):
        v = clean(getattr(r, col, None))
        # Fine-grained dimensions are annotated only on hateful items, matching
        # the filter every scorer in the project applies.
        if v and bool(r.gate):
            gold[str(r.id)] = set(v)
    if not gold:
        return None

    pool = kb_examples()
    pool_labels = {i: set(m.get(col) or []) for i, m in pool.items()}
    n_pool = len(pool_labels)
    # SOURCE-STYLE CONFOUND. Only Implicit Hate examples carry hate_type gold,
    # and en_dev_eval_sq3_types is 100% Implicit Hate. Retrieval matching on
    # source style - short, implicit, tweet-like - would surface annotated
    # examples at above the pool rate and pick up label relevance for free.
    # If the retrieved-annotated share far exceeds the pool-annotated share,
    # the selection is by source, not by label.
    ann = {i for i, m in pool.items() if m.get(col) is not None}
    pool_ann_rate = len(ann) / n_pool

    res = load_results(stem)
    hits, chances, k_seen, n, k_ann = 0, 0.0, [], 0, []
    for iid, g in gold.items():
        item = res.get(iid)
        if item is None:
            continue
        ex = item.get("retrieved", {}).get(EXAMPLE_GROUP, [])
        if not ex:
            continue
        n += 1
        k_seen.append(len(ex))
        hits += any(pool_labels.get(e, set()) & g for e in ex)
        k_ann.append(sum(1 for e in ex if e in ann) / len(ex))
        # P(at least one of k random draws shares a label with this item).
        share = sum(1 for lab in pool_labels.values() if lab & g) / n_pool
        chances += 1.0 - (1.0 - share) ** len(ex)

    if not n:
        return None
    return {"n": n, "k": sum(k_seen) / len(k_seen), "pool": n_pool,
            "observed": hits / n, "chance": chances / n,
            "ratio": (hits / n) / (chances / n) if chances else float("inf"),
            "ret_ann": sum(k_ann) / len(k_ann), "pool_ann": pool_ann_rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None,
                    help="run stems to restrict to")
    args = ap.parse_args()

    print(f"\n{'=' * 78}\nLabel-relevance of RETRIEVED EXAMPLES, per dimension")
    print(f"  does a retrieved example carry the item's gold label, and how "
          f"often would\n  chance alone manage it?\n")
    print(f"  {'run':22} {'dimension':14} {'n':>4} {'k':>4} "
          f"{'observed':>9} {'chance':>8} {'ratio':>7} {'ret.ann':>8} "
          f"{'pool.ann':>9}")

    for stem, subset, dim, col in CASES:
        if args.cases and stem not in args.cases:
            continue
        try:
            r = run_case(stem, subset, dim, col)
        except FileNotFoundError:
            print(f"  {stem:22} {dim:14} (results file not found)")
            continue
        if r is None:
            print(f"  {stem:22} {dim:14} (no scorable items)")
            continue
        print(f"  {stem:22} {dim:14} {r['n']:4} {r['k']:4.1f} "
              f"{r['observed']:9.3f} {r['chance']:8.3f} {r['ratio']:7.2f}x "
              f"{r['ret_ann']:8.3f} {r['pool_ann']:9.3f}")

    print(f"\n  For comparison, the same diagnostic on SQ3's CORRECTIONS "
          f"(feedback bucket,\n  held_out, k=1): 0.149/0.152, 0.198/0.149, "
          f"0.215/0.148, 0.273/0.147 at pool\n  sizes 20/33/48/60, i.e. 0.98x "
          f"rising to 1.86x chance.")
    print(f"\n  A dimension near 1.00x is one similarity cannot select for. A "
          f"dimension\n  well above it is one where retrieval carries real "
          f"label signal, and where\n  feedback write-back would be expected "
          f"to transfer.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
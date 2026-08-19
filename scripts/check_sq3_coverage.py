"""
scripts/check_sq3_coverage.py - the 8.1 coverage audit. Read-only, zero API
calls. Run: python -m scripts.check_sq3_coverage

KNOWN LIMITATION: `--primary` controls which dimension is REPORTED, not which
gold column is READ. The reads are hardcoded to `hate_types`, so
`--primary legal` raises KeyError on de_legal_* subsets, which have no such
column. It also applies a `gate=True` filter that is wrong for `legal` - BoTox
never annotates the gate, so de_legal_* is 100% gate=None and a gated scan
would report zero scorable items. Legal coverage was obtained with a one-off
command instead (de_legal_dev_eval 175 items, support 27/36/34, cardinality
0.554; de_legal_test_eval 177 items, support 30/30/29, cardinality 0.503).
Not fixed because nothing in Phase 7 depends on it.

WHY THIS IS A GATE AND NOT A STEP. en_dev_eval_sq3_feedback was drawn as a
naturally-distributed slice of EN dev, stratified on (source, gate). Nothing
in its construction guarantees it can support a hate_type experiment. The
comparable slice, en_dev_eval_main, carries target_group gold on 28 of its
150 items. Thinness is the expectation, not the risk case.

WHY THE OBVIOUS COUNT IS THE WRONG ONE. "Items with non-None hate_types gold"
is NOT the number the partition is sized against. metrics.score_multilabel
skips any item where bool(gate) is false, and bool(None) is False - so an item
annotated for hate_type by a dataset that never annotates the gate produces no
score at all. The number that matters is the number of items that actually
reach the scorer, so this script obtains it by CALLING score_multilabel on a
synthetic all-items-predicted run rather than re-implementing its filter.
run_slice1.py once carried a private copy of the scoring functions, missed a
fix, and printed a meaningless number for two phases. One implementation,
always.

Where per-item detail is needed that the scorer does not return - the label
cardinality histogram - it is computed locally AND its size is asserted
against the scorer's own n_items. A derived count that cannot be reconciled
with the authoritative one is not evidence, so a mismatch stops the script.

This script decides nothing. It reports the numbers that make the 8.1
checkpoint decidable.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.hsrag.metrics import (MIN_SUPPORT, clean, score_gate,
                               score_multilabel, score_severity)

PROCESSED = Path("data/processed")
SUBSET = "en_dev_eval_sq3_feedback"
PARENT = "en_dev"

# Guide 8.1's partition, as item counts. The gate is this SUM, not the
# guide's ~230 pivot threshold: 230 items cannot be cut into 80 + 120 + 120.
# Generalisation neighbours are a subset of held-out and cost no extra items.
PARTITION = {"sentinel": 80, "held_out": 120, "feedback_batches": 4 * 30}
REQUIRED = sum(PARTITION.values())

# Every EN dev subset, for the disjointness re-assert and the unused-pool
# count. Re-asserted rather than trusted: make_eval_subsets asserts
# disjointness at creation, but the parquets could have been regenerated.
EN_DEV_SUBSETS = ["en_dev_eval_sq1_tune", "en_dev_eval_main",
                  "en_dev_eval_sq3_feedback", "en_dev_eval_targets",
                  "en_dev_eval_types", "en_dev_eval_severity",
                  "en_dev_eval_sq3_types"]

# gold column name in the parquet -> prediction field name in Result.
# They differ (plural vs singular) and score_multilabel takes both.
DIMS = [("gate", "gate", "gate"),
        ("target_group", "target_groups", "target_group"),
        ("hate_type", "hate_types", "hate_type"),
        ("severity", "severity", "severity"),
        ("legal", "legal", "legal")]


def dummy_rows(df: pd.DataFrame) -> list:
    """A synthetic run in which every item has a valid, well-formed prediction.

    The predictions are never compared to anything. The point is the ITEM
    FILTERING the scoring functions apply on the way to counting: feeding this
    through score_* returns exactly the population each dimension would be
    scored on in a real run, obtained from the real filter.
    """
    return [{"item_id": str(i),
             "result": {"hate": True, "target_group": [], "hate_type": [],
                        "severity": "low", "legal": []},
             "uncertain": False}
            for i in df["id"]]


def gold_map(df: pd.DataFrame) -> dict:
    """Same construction score_all uses, so the same objects reach the same
    functions - including the namedtuple-not-Record detail that getattr()
    defaults in score_multilabel exist to handle."""
    return {str(r.id): r for r in df.itertuples(index=False)}


def annotated(df: pd.DataFrame, col: str) -> int:
    """The naive count: gold present, regardless of whether it can be scored."""
    if col not in df.columns:
        return 0
    if col == "gate":
        return int(df["gate"].notna().sum())
    return sum(1 for v in df[col] if clean(v) is not None)


def local_gold_sets(df: pd.DataFrame, col: str) -> dict:
    """Local view of a multilabel dimension's scoring population, for detail
    score_multilabel does not return.

    Mirrors score_multilabel's filter deliberately. Its SIZE is asserted
    against that function's n_items by the caller - if this predicate ever
    drifts from the real one, the assert fires rather than the report being
    quietly wrong.
    """
    out = {}
    for r in df.itertuples(index=False):
        if not bool(r.gate):
            continue
        v = clean(getattr(r, col, None))
        if v is None:
            continue
        out[str(r.id)] = set(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--primary", default="hate_type",
                    help="dimension the partition is sized against")
    args = ap.parse_args()

    path = PROCESSED / f"{args.subset}.parquet"
    df = pd.read_parquet(path)
    rows, gold = dummy_rows(df), gold_map(df)

    print(f"\n{'=' * 70}\n{args.subset}: {len(df)} items")
    print(f"  file: {path}")
    if args.subset == "en_dev_eval_sq3_feedback":
        # True only of the make_eval_subsets slices: SUBSETS asks for 350 and
        # partition()'s apportionment writes 351. Printed unconditionally it
        # misinformed the en_dev_eval_sq3_types run, which a different script
        # drew with no such request.
        print(f"  (make_eval_subsets asked for 350; largest-remainder "
              f"apportionment in partition() is why the written size differs)")
    if "sq3_role" in df.columns:
        # The role column is a new field four downstream steps depend on, so
        # it gets an explicit check that it survives the parquet round-trip -
        # the same rule that exists because `legal` was silently dropped by
        # four functions without any of them crashing.
        print("  sq3_role: " + ", ".join(
            f"{k}={v}" for k, v in sorted(Counter(df["sq3_role"]).items())))

    # ---------------------------------------------------------- composition
    print("\n  source composition:")
    for src, n in sorted(Counter(df["source"]).items()):
        print(f"    {src:16} {n:5}  ({n / len(df) * 100:4.1f}%)")

    g = Counter("None" if pd.isna(v) else str(bool(v)) for v in df["gate"])
    print("\n  gate gold:")
    for k in ("True", "False", "None"):
        print(f"    {k:16} {g.get(k, 0):5}")

    # ------------------------------------------------ annotated vs scorable
    print(f"\n  ANNOTATED vs SCORABLE per dimension")
    print(f"    'annotated' = gold is not None.")
    print(f"    'scorable'  = survives the real filter in metrics.py, i.e."
          f" what a run would actually score.")
    print(f"    {'dimension':16} {'annotated':>10} {'scorable':>10} {'lost':>7}")

    scorable = {
        "gate": score_gate(rows, gold, "strict").n_scored,
        "severity": score_severity(rows, gold).n_items,
    }
    dim_scores = {}
    for name, gold_col, pred_field in DIMS:
        if name in ("gate", "severity"):
            continue
        ds = score_multilabel(rows, gold, gold_col, pred_field)
        dim_scores[name] = ds
        scorable[name] = ds.n_items

    for name, gold_col, _ in DIMS:
        a, s = annotated(df, gold_col), scorable[name]
        print(f"    {name:16} {a:10} {s:10} {a - s:7}")

    print("\n    A nonzero 'lost' column is the bool(gate) filter: items"
          "\n    annotated for a dimension by a dataset that does not annotate"
          "\n    the gate score nothing, because bool(None) is False.")

    # ------------------------------------------------ the primary dimension
    primary = args.primary
    gold_col = dict((n, c) for n, c, _ in DIMS)[primary]
    print(f"\n{'-' * 70}\nPRIMARY DIMENSION: {primary}")

    sets = local_gold_sets(df, gold_col)
    assert len(sets) == scorable[primary], (
        f"local predicate disagrees with metrics.py: {len(sets)} vs "
        f"{scorable[primary]}. The filter in local_gold_sets() has drifted "
        f"from score_multilabel(). Fix before trusting anything below.")

    ds = dim_scores[primary]
    mean_local = sum(len(v) for v in sets.values()) / max(len(sets), 1)
    mean_auth = ds.extra.get("mean_gold_labels", 0.0)
    assert abs(mean_local - mean_auth) < 1e-9, (
        f"cardinality mismatch: local {mean_local} vs scorer {mean_auth}")
    print(f"  scorable items: {len(sets)}  "
          f"(cross-checked against score_multilabel.n_items)")

    print(f"\n  per-label support (MIN_SUPPORT = {MIN_SUPPORT}):")
    counts = Counter(l for v in sets.values() for l in v)
    if not counts:
        print("    NONE - this dimension has no gold on this subset")
    for label, n in sorted(counts.items(), key=lambda kv: kv[1]):
        flag = "" if n >= MIN_SUPPORT else "  << below MIN_SUPPORT"
        print(f"    {label:20} {n:5}{flag}")

    print(f"\n  label cardinality per item (mean {mean_local:.3f}):")
    for k, n in sorted(Counter(len(v) for v in sets.values()).items()):
        print(f"    {k} label(s): {n:5}")
    print("\n    This is the input to the open decision the round protocol"
          "\n    cannot be written without: what counts as WRONG for a"
          "\n    multilabel dimension. Under exact set match, gold averaging"
          "\n    ~1 label against a model predicting 1.7-2.4 makes almost"
          "\n    every item an error and the sentinel near-empty. Under"
          "\n    gold-in-predicted-set, most items are 'correct' and a flip"
          "\n    to wrong is a much weaker event. It changes sentinel size,"
          "\n    batch composition and what over-steering means.")

    print("\n  source composition of the scorable items:")
    src_of = {str(r.id): r.source for r in df.itertuples(index=False)}
    for src, n in sorted(Counter(src_of[i] for i in sets).items()):
        print(f"    {src:16} {n:5}")

    # ------------------------------------------------------ the partition
    print(f"\n{'-' * 70}\nPARTITION, on {primary}-scorable items")
    if "sq3_role" in df.columns:
        # A frozen partition exists in the file, so the guide's nominal sizes
        # are history. Printing them here would show numbers no run will use.
        print("    frozen sq3_role partition:")
        for k, v in sorted(Counter(df["sq3_role"]).items()):
            print(f"      {k:18} {v:5}")
    else:
        print("    guide 8.1 nominal:")
        for k, v in PARTITION.items():
            print(f"      {k:18} {v:5}")
        print(f"      {'REQUIRED':18} {REQUIRED:5}")
    print(f"    {'AVAILABLE':20} {len(sets):5}")

    print("\n  sentinel pool arithmetic (8.4 draws from items correct in ALL"
          "\n  replicates, so the draw shrinks and 80 is the post-shrinkage"
          "\n  target, not the pool):")
    for rate in (0.2, 0.3, 0.4, 0.5, 0.6):
        print(f"    all-replicate-correct rate {rate:.0%}  ->  "
              f"pool of {round(80 / rate):4} needed for an 80-item sentinel")

    # ------------------------------------------- disjointness + supplement
    print(f"\n{'-' * 70}\nDISJOINTNESS (re-asserted, not trusted)")
    ids, present = {}, []
    for name in EN_DEV_SUBSETS:
        p = PROCESSED / f"{name}.parquet"
        if not p.exists():
            print(f"    {name:28} MISSING")
            continue
        ids[name] = set(pd.read_parquet(p, columns=["id"])["id"])
        present.append(name)
        print(f"    {name:28} {len(ids[name]):5} items")
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            overlap = ids[a] & ids[b]
            assert not overlap, f"{a} and {b} overlap on {len(overlap)} items"
    print("    all pairs disjoint: OK")

    parent = pd.read_parquet(PROCESSED / f"{PARENT}.parquet")
    used = set().union(*ids.values()) if ids else set()
    unused = parent[~parent["id"].isin(used)]
    u_rows, u_gold = dummy_rows(unused), gold_map(unused)
    u_scorable = score_multilabel(u_rows, u_gold, gold_col, primary).n_items
    print(f"\n  supplement pool (8.1's pivot, if the partition does not close):"
          f"\n    {PARENT}: {len(parent)} items, {len(used)} already in a "
          f"subset, {len(unused)} unused"
          f"\n    of the unused, {u_scorable} are {primary}-scorable")

    # PER-LABEL SUPPORT IN THE SUPPLEMENT POOL. The total is not the binding
    # number - the rarest label is. A 320-item balanced draw needs ~45 per
    # label, and en_dev_eval_types already skimmed 15 per label off this pool.
    # Estimating the tail from the 31-item subsample would be the same
    # small-sample reading that has twice been mistaken for an effect here,
    # so it is counted directly.
    print("\n  per-label support IN THE UNUSED POOL"
          "\n  (the rarest label is the constraint, not the total):")
    for _name, _col in (("hate_type", "hate_types"),
                        ("target_group", "target_groups")):
        _sets = local_gold_sets(unused, _col)
        _cnt = Counter(l for v in _sets.values() for l in v)
        print(f"\n    {_name}: {len(_sets)} scorable items")
        for _label, _n in sorted(_cnt.items(), key=lambda kv: kv[1]):
            print(f"      {_label:20} {_n:6}")
        if _cnt:
            _r, _rn = min(_cnt.items(), key=lambda kv: kv[1])
            print(f"      -> rarest '{_r}' at {_rn}; a balanced draw over "
                  f"{len(_cnt)} labels caps at {_rn * len(_cnt)} items")

    _sev = Counter(v for v in (clean(getattr(r, "severity", None))
                               for r in unused.itertuples(index=False)
                               if bool(r.gate)) if v)
    print(f"\n    severity: {sum(_sev.values())} scorable items")
    for _label, _n in sorted(_sev.items(), key=lambda kv: kv[1]):
        print(f"      {_label:20} {_n:6}")
    if _sev:
        _r, _rn = min(_sev.items(), key=lambda kv: kv[1])
        print(f"      -> rarest '{_r}' at {_rn}; a balanced draw over "
              f"{len(_sev)} bands caps at {_rn * len(_sev)} items")

    _gu = Counter("None" if pd.isna(v) else str(bool(v)) for v in unused["gate"])
    print(f"\n    gate: {len(unused)} items - True {_gu.get('True', 0)}, "
          f"False {_gu.get('False', 0)}, None {_gu.get('None', 0)}")

    # ------------------------------------------------------------- verdict
    ok = len(sets) >= REQUIRED
    print(f"\n{'=' * 70}\nCHECKPOINT: ", end="")
    if ok:
        print(f"PASS - {len(sets)} scorable >= {REQUIRED} required.")
        print("  The partition closes on the existing subset. Size the"
              "\n  sentinel against the pool arithmetic above before"
              "\n  committing to 80.")
    else:
        print(f"PIVOT - {len(sets)} scorable < {REQUIRED} required "
              f"(short by {REQUIRED - len(sets)}).")
        print(f"  Draw a supplementary label-stratified subset from the"
              f"\n  {len(unused)} unused {PARENT} items ({u_scorable} scorable),"
              f"\n  disjoint from all existing slices, same seed protocol."
              f"\n  Do NOT borrow the unrun items from en_dev_eval_main: it is"
              f"\n  the reporting slice and KB write-back contaminates it"
              f"\n  permanently.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
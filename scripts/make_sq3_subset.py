"""
scripts/make_sq3_subset.py - draws en_dev_eval_sq3_types, the supplementary
subset SQ3's primary dimension runs on. Zero API calls.
Run: python -m scripts.make_sq3_subset --dry-run

WHY A NEW SUBSET. check_sq3_coverage found en_dev_eval_sq3_feedback carries 31
hate_type-scorable items against a partition needing ~450. That slice was
drawn stratified on (source, gate) from a pool where only Implicit Hate's
hateful items carry hate_type gold, so its thinness is structural, not a
sampling accident. Guide 8.1's pre-specified pivot is a supplementary
label-stratified draw from the unused dev pool, which preserves every
disjointness guarantee.

WHY THIS IS A SEPARATE SCRIPT AND NOT A LINE IN make_eval_subsets. That script
draws its pool with draw_pool(parent, sum(slices.values()), SEED). Adding an
entry to SUBSETS changes that sum, which changes the pool, which changes every
existing en_dev subset. The frozen subsets are what makes every result in the
project comparable. This script only ever reads them.

WHY THE PARTITION IS FROZEN HERE BUT THE SENTINEL IS NOT.
make_eval_subsets' docstring says SQ3's internal split cannot be frozen at
draw time because the sentinel is defined as items a run got right, and that
run does not exist yet. That is correct about the SENTINEL and wrong about the
POOL it is drawn from. Freezing pool / held_out / batches here makes
disjointness and label balance structural; the sentinel is still selected at
run time, from within the pool, from items correct in all replicates.

WHY IT STRATIFIES ON THE LABEL. make_eval_subsets.partition stratifies on
(source, gate). Here both are constant - every item is implicit_hate with
gate=True, because that is what carrying hate_type gold means - so it would
degenerate to a random cut and could leave a label at 5 in held-out, where
macro-F1 reports noise rather than performance.

SIZING, from check_sq3_sizing on three T=1.0 replicates sharing KB
475869f9 and prompt c74cb7ab:

  criterion   per-run   all-3 (r)   pool for an 80-item sentinel
  contains      ~0.49      0.427                  187
  exact         0.117      0.049                 1648

so `contains` (gold label present in the predicted set) is the criterion that
both selects errors and defines the sentinel. `exact` is reported alongside as
the strict secondary, not used to select. The full argument is in
experiment_log.md; the short version is that at exact's 0.117 accuracy a
30-item batch holds ~3 correct items, so the matched control cannot be drawn
from the same batch and would have to come from TRAIN, confounding
error-drivenness with split.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.check_sq3_coverage import (EN_DEV_SUBSETS, PROCESSED,
                                        local_gold_sets)
from scripts.make_eval_subsets import _largest_remainder, clean, label_stratified
from src.hsrag.metrics import MIN_SUPPORT

PARENT = "en_dev"
NAME = "en_dev_eval_sq3_types"
GOLD_COL = "hate_types"
SEED = 42

# Frozen roles. batches is ONE role of 120, not four of 30: guide 8.9 gives
# each replicate its own batch-order seed, so splitting into rounds is the
# runner's job. Freezing four batches here would remove the independence the
# replicates exist to measure.
HELD_OUT = 120
BATCHES = 4 * 30
SENTINEL_TARGET = 80
R_CONTAINS = 0.427          # measured, see module docstring


def partition_by_label(df: pd.DataFrame, sizes: dict, seed: int) -> dict:
    """Disjoint named parts, stratified on the hate_type gold label.

    Items are grouped by their exact label SET, not by a primary label: ~3% of
    Implicit Hate items carry two types, and assigning those to one label
    would put the same item in a different stratum depending on sort order.
    Apportioning inside each group keeps every part's label composition equal
    to the parent's.

    `sizes` must sum to len(df); the apportionment is exact, so the returned
    parts partition the input with nothing left over.
    """
    assert sum(sizes.values()) == len(df), (
        f"sizes sum to {sum(sizes.values())}, subset has {len(df)}")
    names = list(sizes)
    total = sum(sizes.values())
    buckets = {n: [] for n in names}

    keyed = df.assign(_lbl=[",".join(sorted(clean(v) or []))
                            for v in df[GOLD_COL]])
    for _, group in keyed.groupby("_lbl"):
        group = group.sample(frac=1.0, random_state=seed)
        exact = {n: len(group) * sizes[n] / total for n in names}
        take = _largest_remainder(exact, len(group))
        i = 0
        for n in names:
            if take[n] > 0:
                buckets[n].append(group.iloc[i:i + take[n]])
            i += take[n]

    out = {}
    for n, parts in buckets.items():
        d = pd.concat(parts) if parts else keyed.iloc[0:0]
        out[n] = (d.drop(columns=["_lbl"])
                  .sample(frac=1.0, random_state=seed)
                  .reset_index(drop=True))
    return out


def label_counts(df: pd.DataFrame) -> Counter:
    return Counter(l for v in df[GOLD_COL] for l in (clean(v) or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=68,
                    help="items per hate_type label; the rarest label caps it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite (frozen means frozen)")
    args = ap.parse_args()

    out_path = PROCESSED / f"{NAME}.parquet"
    if out_path.exists() and not args.force and not args.dry_run:
        print(f"SKIP: {out_path} exists (use --force to overwrite)")
        return

    # ------------------------------------------------- pool and exclusions
    parent = pd.read_parquet(PROCESSED / f"{PARENT}.parquet")
    used = set()
    for name in EN_DEV_SUBSETS:
        p = PROCESSED / f"{name}.parquet"
        if p.exists():
            used |= set(pd.read_parquet(p, columns=["id"])["id"])
    print(f"{PARENT}: {len(parent)} items, {len(used)} already in a subset")

    avail = label_counts(parent[~parent["id"].isin(used)]
                         [lambda d: d["gate"] == True])  # noqa: E712
    rarest, rn = min(avail.items(), key=lambda kv: kv[1])
    print(f"unused pool, per label: "
          + ", ".join(f"{k}={v}" for k, v in sorted(avail.items(),
                                                    key=lambda kv: kv[1])))
    print(f"rarest '{rarest}' at {rn} -> per_label caps at {rn}")
    assert args.per_label <= rn, (
        f"--per-label {args.per_label} exceeds the {rn} available for "
        f"'{rarest}'. A balanced draw cannot be filled.")

    # ------------------------------------------------------------ the draw
    # Reuses make_eval_subsets.label_stratified unchanged: same rarest-first
    # fill, same seed, same exclusion semantics as every dimension subset in
    # the project. A second implementation of subset drawing is exactly the
    # duplication that made run_slice1's private scorer print a meaningless
    # number for two phases.
    sub = label_stratified(parent, GOLD_COL, args.per_label, SEED, used)
    assert not sub.empty, "draw came back empty"

    print(f"\n{'=' * 70}\n{NAME}: {len(sub)} items")
    counts = label_counts(sub)
    for label, n in sorted(counts.items(), key=lambda kv: kv[1]):
        print(f"    {label:20} {n:5}")
    card = Counter(len(clean(v) or []) for v in sub[GOLD_COL])
    print(f"  cardinality: "
          + ", ".join(f"{k} label(s)={v}" for k, v in sorted(card.items())))
    print(f"  sources: {dict(Counter(sub['source']))}")
    print(f"  gate: {dict(Counter(bool(g) for g in sub['gate']))}")

    # ------------------------------------------------------ the partition
    pool_n = len(sub) - HELD_OUT - BATCHES
    need = round(SENTINEL_TARGET / R_CONTAINS)
    print(f"\n  partition: pool {pool_n} / held_out {HELD_OUT} / "
          f"batches {BATCHES}")
    print(f"  sentinel needs a pool of {need} at r={R_CONTAINS} "
          f"(contains, measured on 3 replicates)")
    assert pool_n >= need, (
        f"pool {pool_n} < {need} needed for a {SENTINEL_TARGET}-item "
        f"sentinel. Raise --per-label, or lower HELD_OUT / BATCHES / "
        f"SENTINEL_TARGET - but change one deliberately, not to make an "
        f"assert pass.")

    parts = partition_by_label(
        sub, {"pool": pool_n, "held_out": HELD_OUT, "batches": BATCHES}, SEED)

    print(f"\n  {'role':12} {'n':>5}  per-label")
    for role, d in parts.items():
        c = label_counts(d)
        thin = [k for k, v in c.items() if v < MIN_SUPPORT]
        print(f"  {role:12} {len(d):5}  "
              + ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
              + (f"   << below MIN_SUPPORT: {thin}" if thin else ""))

    # held_out is the only role scored as a macro average, so it is the only
    # one where a thin label silently degrades the headline number.
    thin = [k for k, v in label_counts(parts["held_out"]).items()
            if v < MIN_SUPPORT]
    assert not thin, (
        f"held_out labels below MIN_SUPPORT: {thin}. The learning curve would "
        f"average noise on these. Raise --per-label or HELD_OUT.")

    # -------------------------------------------------------- the asserts
    ids = {r: set(d["id"]) for r, d in parts.items()}
    roles = list(ids)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            assert not ids[a] & ids[b], f"{a} and {b} overlap"
    assert sum(len(v) for v in ids.values()) == len(sub), "roles lost items"
    assert not set(sub["id"]) & used, "overlaps an existing subset"

    scorable = local_gold_sets(sub, GOLD_COL)
    assert len(scorable) == len(sub), (
        f"{len(sub) - len(scorable)} drawn items are not hate_type-scorable")
    print(f"\n  disjoint from all {len(EN_DEV_SUBSETS)} existing subsets: OK")
    print(f"  roles mutually disjoint and exhaustive: OK")
    print(f"  all {len(sub)} items hate_type-scorable: OK")

    # --------------------------------------------------------------- write
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return

    tagged = pd.concat([d.assign(sq3_role=role) for role, d in parts.items()])
    tagged = tagged.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    tagged.to_parquet(out_path, index=False)
    print(f"\n  wrote {out_path}  ({len(tagged)} items, sq3_role column)")
    print(f"  ADD TO EN_DEV_SUBSETS in check_sq3_coverage.py so future "
          f"disjointness checks see it.")


if __name__ == "__main__":
    main()
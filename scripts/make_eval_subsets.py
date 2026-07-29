"""
scripts/make_eval_subsets.py - frozen evaluation subsets.

WHY: EN test is 12,041 items. At ~3 s per call x n self-consistency votes x
N arms, scoring the full split is not runnable. Every system from Phase 5
onward - all LLM arms and the Phase 7 encoder baselines - scores on exactly
these items. That is what makes cross-system comparison paired: McNemar and
the paired bootstrap need item-level pairing, which only exists if every
system saw the same items.

FROZEN MEANS FROZEN. Refuses to overwrite without --force. Seed 42, matching
make_splits.py.

STRATIFICATION IS ON (source, gate), NOT gate ALONE. Each taxonomy dimension
comes from exactly one dataset - hate_type from implicit_hate, severity from
mhs, target_group from hatexplain/mhs/detox. A sample balanced on gate but
blind to source can leave a dimension with almost no evaluable items, and
which dimension gets starved would be down to luck. Stratifying on source
keeps every dataset's share of each subset equal to its share of the parent
split.

THE ENGLISH DEV POOL IS SPLIT INTO DISJOINT NAMED SLICES. One dev set cannot
serve as tuning set, headline reporting set and adaptability set at once:
selecting a retrieval configuration on the same items later used to measure
adaptability is circular. sq1_tune is tuned on, main is reported from,
sq3_feedback carries the feedback rounds. The slice partition is itself
stratified, so the three are comparable to each other and to the parent.

SQ3's internal split is NOT frozen here. The over-steering sentinel is defined
as items the system classified correctly on the first stable run, which is not
knowable before that run exists. Sentinel, correction items and generalisation
neighbours are all selected at run time from within sq3_feedback.

The per-dimension and per-label counts printed at the end are the point of
this script as much as the parquets are: they say what each dimension can
actually support before any API budget is spent on it.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
SEED = 42

# (lang, split) -> {slice_name or None: n}
# None means "no slices" and writes {lang}_{split}_eval.parquet
SUBSETS = {
    ("en", "dev"): {"sq1_tune": 250, "main": 300, "sq3_feedback": 350},
    ("en", "test"): {None: 500},
    ("de", "dev"): {None: 150},
    ("de", "test"): {None: 300},
}

LIST_DIMS = ["target_groups", "hate_types"]

# label-stratified dimension subsets: {(lang, split): per_label}
DIM_SUBSETS = {
    ("en", "dev"): 15,
    ("en", "test"): 25,
}
DIM_COLS = {"targets": "target_groups", "types": "hate_types",
            "severity": "severity"}
MIN_SUPPORT = 10          # labels below this are reported, not averaged


def label_stratified(df, col, per_label, seed, exclude_ids):
    """Draw up to per_label items carrying each label of `col`.

    WHY THIS EXISTS: every fine-grained dimension comes from one dataset and
    is annotated only on hateful items, so a naturally-distributed 300-item
    subset leaves the rarest hate_type with 2 positives. Macro-F1 over a
    2-item label is not reportable, and growing the natural subset enough to
    fix it costs roughly 7x the API budget. These subsets are deliberately
    label-balanced and stated as such - which is more honest than presenting
    an incidental 27-item sample as a natural distribution.

    Rarest label first: a multilabel item drawn for a rare label often also
    carries a common one, so filling rare labels first minimises total items
    (and therefore calls) for the same coverage.
    """
    pool = df[~df["id"].isin(exclude_ids)].reset_index(drop=True)

    labels_by_idx = {}
    for i, v in enumerate(pool[col]):
        v = clean(v)
        if not v:                       # None (never annotated) or [] (empty)
            continue
        labels_by_idx[i] = [v] if isinstance(v, str) else list(v)

    counts = Counter(l for ls in labels_by_idx.values() for l in ls)
    if not counts:
        return pool.iloc[0:0]

    rng = np.random.default_rng(seed)
    chosen = set()
    for label in sorted(counts, key=lambda l: counts[l]):
        have = sum(1 for i in chosen if label in labels_by_idx[i])
        need = per_label - have
        if need <= 0:
            continue
        cands = [i for i, ls in labels_by_idx.items()
                 if label in ls and i not in chosen]
        rng.shuffle(cands)
        chosen.update(cands[:need])

    out = pool.iloc[sorted(chosen)]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def clean(val):
    """Parquet round-trip: None comes back as NaN, lists as numpy arrays.

    The None / [] / ["race"] three-way distinction has to survive this, so
    every read of a label column goes through here. None means the dataset
    never annotated the dimension; [] means annotated and empty.
    """
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def _largest_remainder(exact: dict, total: int) -> dict:
    """Apportion `total` slots given exact fractional targets.

    floor() alone loses items and would silently return fewer than asked for.
    Leftover slots go to the entries with the biggest fractional parts - the
    standard apportionment fix, deterministic given a fixed key order.
    """
    take = {k: int(np.floor(v)) for k, v in exact.items()}
    short = total - sum(take.values())
    order = sorted(exact, key=lambda k: (exact[k] - take[k], str(k)), reverse=True)
    for k in order[:short]:
        take[k] += 1
    return take


def draw_pool(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample of n items on (source, gate), proportional."""
    strata = list(df.groupby(["source", "gate"], dropna=False))
    total = len(df)
    exact = {key: len(g) / total * n for key, g in strata}
    take = _largest_remainder(exact, n)

    parts = []
    for key, group in strata:
        k = min(take[key], len(group))
        if k > 0:
            parts.append(group.sample(n=k, random_state=seed))

    out = pd.concat(parts)
    # shuffle so "the first 150 items" is a valid sub-sample rather than one
    # stratum sitting at the top of the file
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def partition(pool: pd.DataFrame, slices: dict, seed: int) -> dict:
    """Split a pool into disjoint named slices, stratified on (source, gate).

    Cutting the shuffled pool into contiguous blocks would be random but not
    stratified - each slice's composition would drift a few points from the
    parent. Splitting inside each stratum keeps all slices comparable.
    """
    names = list(slices)
    total = sum(slices.values())
    buckets = {n: [] for n in names}

    for _, group in pool.groupby(["source", "gate"], dropna=False):
        group = group.sample(frac=1.0, random_state=seed)
        exact = {n: len(group) * slices[n] / total for n in names}
        take = _largest_remainder(exact, len(group))
        i = 0
        for n in names:
            if take[n] > 0:
                buckets[n].append(group.iloc[i:i + take[n]])
            i += take[n]

    out = {}
    for n, parts in buckets.items():
        df = pd.concat(parts) if parts else pool.iloc[0:0]
        out[n] = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def report(df: pd.DataFrame, name: str, parent: pd.DataFrame) -> None:
    """Print what this subset can and cannot measure."""
    print(f"\n{'=' * 68}\n{name}: {len(df)} items (parent split: {len(parent)})")

    print("\n  composition (subset % vs parent %):")
    for src in sorted(parent["source"].unique()):
        sub = (df["source"] == src).mean() * 100
        par = (parent["source"] == src).mean() * 100
        print(f"    {src:15} {sub:5.1f}%  vs {par:5.1f}%")
    print(f"    {'gate=True':15} {df['gate'].mean() * 100:5.1f}%  "
          f"vs {parent['gate'].mean() * 100:5.1f}%")

    print("\n  evaluable items per dimension:")
    print(f"    {'gate':16} {len(df):4}  (every dataset annotates it)")
    for col in LIST_DIMS + ["severity"]:
        vals = [clean(v) for v in df[col]]
        annotated = sum(1 for v in vals if v is not None)
        hateful = sum(1 for v, g in zip(vals, df["gate"]) if v is not None and g)
        print(f"    {col:16} {annotated:4}  annotated, {hateful:4} of those hateful")

    print("\n  positive items per label "
          "(macro-F1 is only as good as its thinnest label):")
    for col in LIST_DIMS:
        counter = Counter()
        for v in df[col]:
            v = clean(v)
            if v:
                counter.update(v)
        line = (", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
                if counter else "none")
        print(f"    {col:16} {line}")

    sev = Counter(v for v in (clean(x) for x in df["severity"]) if v)
    line = (", ".join(f"{k}={v}" for k, v in sorted(sev.items()))
            if sev else "none")
    print(f"    {'severity':16} {line}")


def out_path(lang: str, split: str, slice_name) -> Path:
    stem = f"{lang}_{split}_eval"
    if slice_name is not None:
        stem += f"_{slice_name}"
    return PROCESSED / f"{stem}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing subsets (frozen means frozen)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the diagnostics, write nothing")
    args = ap.parse_args()

    for (lang, split), slices in SUBSETS.items():
        paths = {s: out_path(lang, split, s) for s in slices}
        dim_paths = [PROCESSED / f"{lang}_{split}_eval_{n}.parquet"
                     for n in DIM_COLS] if DIM_SUBSETS.get((lang, split)) else []

        exists = any(p.exists() for p in list(paths.values()) + dim_paths)
        if exists and not args.force and not args.dry_run:
            print(f"SKIP {lang}_{split}: subsets already exist "
                  f"(use --force to overwrite)")
            continue

        parent = pd.read_parquet(PROCESSED / f"{lang}_{split}.parquet")
        pool = draw_pool(parent, sum(slices.values()), SEED)

        if list(slices) == [None]:
            written = {None: pool}
        else:
            written = partition(pool, slices, SEED)
            # disjointness is the whole point of slicing - assert it
            ids = [set(df["id"]) for df in written.values()]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    assert not ids[i] & ids[j], "slices overlap"
            assert sum(len(s) for s in ids) == len(pool), "slice sizes drifted"

        for slice_name, df in written.items():
            label = f"{lang}_{split}_eval" + (f"_{slice_name}" if slice_name else "")
            report(df, label, parent)
            if args.dry_run:
                continue
            df.to_parquet(paths[slice_name], index=False)
            print(f"\n  wrote {paths[slice_name]}")

        # --- label-stratified dimension subsets ---
        # The naturally-distributed subsets above can report gate. They cannot
        # report target_group / hate_type / severity: every fine-grained
        # dimension comes from a single dataset and is annotated only on
        # hateful items, so the rarest hate_type lands at 2 positives. These
        # subsets are drawn label-balanced instead, disjoint from the slices
        # above and from each other.
        per_label = DIM_SUBSETS.get((lang, split))
        if not per_label:
            continue

        used = set()
        for df in written.values():
            used |= set(df["id"])

        for name, col in DIM_COLS.items():
            sub = label_stratified(parent, col, per_label, SEED, used)
            if sub.empty:
                print(f"\n{'=' * 68}\n{lang}_{split}_eval_{name}: EMPTY - "
                      f"no positive labels available for {col}")
                continue
            used |= set(sub["id"])       # keep dimension subsets disjoint too
            report(sub, f"{lang}_{split}_eval_{name}", parent)
            if args.dry_run:
                continue
            path = PROCESSED / f"{lang}_{split}_eval_{name}.parquet"
            sub.to_parquet(path, index=False)
            print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
"""
scripts/make_botox_splits.py - frozen splits for the BoTox legal dimension.

SEPARATE FILES, NOT FOLDED INTO de_train/dev/test. BoTox annotates ONLY
criminal relevance, so it shares no dimension with the existing German data.
Merging it would change the contents of the German splits and invalidate the
german_dev baseline (macro-F1 0.777 / 0.755 / 0.766 at kb 475869f9e2422969),
which every German adaptability measurement is compared against.

Stratified on the class-0 flag rather than on any single class, because 72
items carry more than one class and a per-class stratification would double-
count them. Seed 42, matching make_splits.py.

The leakage filter lives in the loader: rows whose text appears in any
held-out split are dropped before they reach this script.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.hsrag.data.botox import load

PROCESSED = Path("data/processed")
SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = {s: PROCESSED / f"de_legal_{s}.parquet"
             for s in ("train", "dev", "test")}
    if any(p.exists() for p in paths.values()) and not args.force:
        print("SKIP: de_legal splits already exist (use --force)")
        return

    records = load()
    df = pd.DataFrame([{
        "id": r.id, "text": r.text, "lang": r.lang, "source": r.source,
        "gate": r.gate, "target_groups": r.target_groups,
        "hate_types": r.hate_types, "severity": r.severity,
        "legal": r.legal, "raw": r.raw,
    } for r in records])

    # Cross-dataset dedup by normalised text, as in make_splits.py. The loader
    # already removed held-out overlaps; this catches duplicates within BoTox.
    df["_norm"] = df["text"].map(lambda t: " ".join(str(t).lower().split()))
    before = len(df)
    df = df.drop_duplicates("_norm", keep="first").drop(columns="_norm")
    if before != len(df):
        print(f"  deduped {before - len(df)} repeated texts")

    # Stratify on "is criminally relevant at all". Per-class stratification
    # would double-count the 72 multi-class items.
    df["_stratum"] = df["legal"].map(lambda v: bool(v))
    rng = np.random.default_rng(SEED)
    parts = {"train": [], "dev": [], "test": []}

    for _, group in df.groupby("_stratum"):
        group = group.sample(frac=1.0, random_state=SEED)
        n = len(group)
        n_train = int(n * RATIOS[0])
        n_dev = int(n * RATIOS[1])
        parts["train"].append(group.iloc[:n_train])
        parts["dev"].append(group.iloc[n_train:n_train + n_dev])
        parts["test"].append(group.iloc[n_train + n_dev:])

    for split, chunks in parts.items():
        out = (pd.concat(chunks)
               .drop(columns="_stratum")
               .sample(frac=1.0, random_state=SEED)
               .reset_index(drop=True))
        counts = Counter(l for v in out["legal"] for l in v)
        counts["(class 0)"] = sum(1 for v in out["legal"] if not v)
        print(f"\n{split:6} {len(out):5} items   {dict(counts)}")
        out.to_parquet(paths[split], index=False)
        print(f"  wrote {paths[split]}")


if __name__ == "__main__":
    main()
"""
Build frozen train/dev/test splits from all loaders.

THREE HARD RULES (these protect the validity of every later experiment):
  1. KB examples come from TRAIN only. Never dev, never test.
  2. TEST is touched only by final experiment runs. All tuning is on DEV.
  3. The SQ3 human-feedback sentinel set is drawn from DEV.

WHY FROZEN
  Splits are made ONCE, with seed=42, and saved. If they were regenerated
  ad hoc, results from different weeks wouldn't be comparable, and KB
  examples could leak into test -> invalid RAG results ("the model
  retrieved the answer" rather than "the model generalised").

SPLIT METHOD
  Per language: 70/15/15, stratified on `gate` only. Full multi-label
  iterative stratification is overkill at these sizes; gate-stratification
  keeps the hateful/not ratio stable across splits, which is what matters
  most. Note this simplification in the thesis.

  GAHD ships its own split, but we re-split uniformly so the KB-from-train
  rule and the leakage test hold identically across all datasets. GAHD's
  original split stays in `raw`.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.hsrag.data.hatexplain import load_hatexplain
from src.hsrag.data.mhs import load_mhs
from src.hsrag.data.implicit_hate import load_implicit_hate
from src.hsrag.data.gahd import load_gahd
from src.hsrag.data.detox import load_detox

SEED = 42
OUT_DIR = Path("data/processed")
FIG_DIR = Path("experiments/results/figures")

LOADERS = [
    load_hatexplain,
    load_mhs,
    load_implicit_hate,
    load_gahd,
    load_detox,
]


def record_to_dict(r) -> dict:
    """Flatten a Record for parquet. Nested list/None columns are kept as
    Python objects; raw is JSON-stringified so parquet doesn't choke on the
    nested dict."""
    return {
        "id": r.id,
        "text": r.text,
        "lang": r.lang,
        "source": r.source,
        "gate": r.gate,
        "target_groups": r.target_groups,   # list | None
        "hate_types": r.hate_types,         # list | None
        "severity": r.severity,             # str | None
        "raw": json.dumps(r.raw, ensure_ascii=False),
    }


def load_all() -> pd.DataFrame:
    rows = []
    for loader in LOADERS:
        records, _ = loader()
        rows.extend(record_to_dict(r) for r in records)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} records total")

    # --- deduplicate by normalised text (prevents cross-dataset leakage) ---
    import re
    df["_norm"] = df["text"].map(
        lambda t: re.sub(r"\s+", " ", str(t).strip().lower())
    )
    before = len(df)
    df = df.drop_duplicates(subset="_norm", keep="first").drop(columns="_norm")
    dropped = before - len(df)
    print(f"Dropped {dropped} duplicate texts (kept first occurrence)")

    print(f"  by language:\n{df['lang'].value_counts().to_string()}")
    return df


def split_one_language(df_lang: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """70/15/15, stratified on gate. Two-step: first carve off test,
    then split the rest into train/dev."""
    # step 1: 85% temp / 15% test
    temp, test = train_test_split(
        df_lang, test_size=0.15, random_state=SEED, stratify=df_lang["gate"]
    )
    # step 2: of the 85%, take 15/85 as dev -> 70/15 overall
    train, dev = train_test_split(
        temp, test_size=0.15 / 0.85, random_state=SEED, stratify=temp["gate"]
    )
    return {"train": train, "dev": dev, "test": test}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all()

    for lang in sorted(df["lang"].unique()):
        df_lang = df[df["lang"] == lang].reset_index(drop=True)
        splits = split_one_language(df_lang)

        print(f"\n=== {lang} ===")
        for split_name, split_df in splits.items():
            path = OUT_DIR / f"{lang}_{split_name}.parquet"
            split_df.to_parquet(path, index=False)

            gate_true = (split_df["gate"] == True).sum()
            print(f"  {split_name:5} {len(split_df):6} records "
                  f"({gate_true/len(split_df)*100:.1f}% hateful) -> {path.name}")

            # per-split source histogram, saved as json for the record
            hist = Counter(split_df["source"])
            hist_path = FIG_DIR / f"{lang}_{split_name}_sources.json"
            hist_path.write_text(json.dumps(dict(hist), indent=2))

    print("\nDone. Splits frozen with seed=42.")


if __name__ == "__main__":
    main()
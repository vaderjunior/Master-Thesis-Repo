"""
Leakage test: no text may appear in more than one split within a language.
Runs on every `pytest`. If this ever fails, RAG results are invalid.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path("data/processed")
LANGS = ["en", "de"]
SPLITS = ["train", "dev", "test"]


def normalise(text: str) -> str:
    """Collapse whitespace and lowercase, so near-duplicates that differ
    only in spacing/case still count as the same text."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


@pytest.mark.parametrize("lang", LANGS)
def test_no_text_leakage_between_splits(lang):
    texts = {}
    for split in SPLITS:
        path = PROCESSED / f"{lang}_{split}.parquet"
        if not path.exists():
            pytest.skip(f"{path} not built yet")
        df = pd.read_parquet(path)
        texts[split] = set(df["text"].map(normalise))

    # every pair of splits must be disjoint
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                overlap = texts[a] & texts[b]
                assert not overlap, (
                    f"{lang}: {len(overlap)} texts leak between {a} and {b}. "
                    f"Example: {next(iter(overlap))[:80]!r}"
                )


@pytest.mark.parametrize("lang", LANGS)
def test_splits_nonempty(lang):
    for split in SPLITS:
        path = PROCESSED / f"{lang}_{split}.parquet"
        if not path.exists():
            pytest.skip(f"{path} not built yet")
        df = pd.read_parquet(path)
        assert len(df) > 0, f"{lang}_{split} is empty"
"""
Frozen evaluation subsets: containment and disjointness.

WHY THESE EXIST: every system from Phase 5 onward - all LLM arms and the
Phase 7 encoder baselines - scores on these exact items. That is what makes
cross-system comparison paired; McNemar and the paired bootstrap need
item-level pairing, which only exists if every system saw the same items. If
a subset silently drifts from its parent split, or two slices overlap, every
downstream comparison is quietly invalid.

The slice partition matters for a second reason: selecting a retrieval
configuration on sq1_tune and then measuring adaptability on sq3_feedback is
only non-circular if the two are actually disjoint. Asserted, not assumed.
"""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path("data/processed")

# subset stem -> parent split stem
SUBSETS = {
    "en_dev_eval_sq1_tune": "en_dev",
    "en_dev_eval_main": "en_dev",
    "en_dev_eval_sq3_feedback": "en_dev",
    "en_dev_eval_targets": "en_dev",
    "en_dev_eval_types": "en_dev",
    "en_dev_eval_severity": "en_dev",
    "en_test_eval": "en_test",
    "en_test_eval_targets": "en_test",
    "en_test_eval_types": "en_test",
    "en_test_eval_severity": "en_test",
    "de_dev_eval": "de_dev",
    "de_test_eval": "de_test",
}

# every subset drawn from the same parent must be pairwise disjoint
FAMILIES = {
    "en_dev": [s for s, p in SUBSETS.items() if p == "en_dev"],
    "en_test": [s for s, p in SUBSETS.items() if p == "en_test"],
}


def load(stem):
    return pd.read_parquet(PROCESSED / f"{stem}.parquet")


@pytest.mark.parametrize("stem", list(SUBSETS))
def test_subset_exists_and_nonempty(stem):
    assert len(load(stem)) > 0


@pytest.mark.parametrize("stem,parent_stem", list(SUBSETS.items()))
def test_subset_is_contained_in_parent(stem, parent_stem):
    """By id AND by text: an id could in principle be reused across splits."""
    sub, parent = load(stem), load(parent_stem)
    assert set(sub["id"]) <= set(parent["id"])
    norm = lambda df: {" ".join(str(t).lower().split()) for t in df["text"]}
    assert norm(sub) <= norm(parent)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_subsets_from_one_parent_are_disjoint(family):
    """sq1_tune is tuned on, main is reported from, sq3_feedback carries the
    feedback rounds, and the dimension subsets score the fine-grained labels.
    Overlap between any two makes the corresponding claim circular."""
    stems = FAMILIES[family]
    ids = {s: set(load(s)["id"]) for s in stems}
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            assert not ids[a] & ids[b], f"{a} and {b} overlap"


@pytest.mark.parametrize("stem,parent_stem", list(SUBSETS.items()))
def test_no_cross_split_text_leakage(stem, parent_stem):
    """Holds by construction (parents are already deduped and disjoint), but a
    future resampling bug would show up here first."""
    lang = stem.split("_")[0]
    other = "test" if "_dev" in parent_stem else "dev"
    sub_texts = {" ".join(str(t).lower().split()) for t in load(stem)["text"]}
    other_texts = {" ".join(str(t).lower().split())
                   for t in load(f"{lang}_{other}")["text"]}
    assert not sub_texts & other_texts
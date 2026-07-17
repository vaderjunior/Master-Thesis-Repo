"""
Retrieval sanity probes. Locks in that the KB surfaces sensible knowledge for
a few clear cases, and that it separates hate from non-hate by distance.

NOT a quality benchmark - a regression guard so a broken KB rebuild (wrong
model, empty collection, bad ingest) gets caught on every pytest run.

Only probes that genuinely passed the 3.5 eyeball are asserted here. The
known misses (quoted-slur guideline, offensive-not-hate) are logged in
experiment_log.md as SQ1 motivation, NOT tested - they're problems Phase 4
fixes, not regressions to guard.
"""

import json
from pathlib import Path

import pytest
import yaml
from sentence_transformers import SentenceTransformer

from src.hsrag.kb import load_collection

config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
kb_cfg = config["kb"]

# module-level: load once, reuse across tests
model = SentenceTransformer(kb_cfg["embedding_model"])
col = load_collection(Path(kb_cfg["chroma_path"]), kb_cfg["embedding_model"])


def _parse_target_groups(meta: dict) -> list[str]:
    """target_groups comes back from Chroma as a JSON string (or the
    '__none__' sentinel). Return a plain list for easy checking."""
    raw = meta.get("target_groups", "__none__")
    if raw == "__none__" or not isinstance(raw, str):
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []


def _query(text: str, k: int = 3):
    """Return (metadatas, ids, documents, distances) for the top-k hits."""
    q = model.encode([text], normalize_embeddings=True)
    hits = col.query(query_embeddings=q.tolist(), n_results=k)
    return (
        hits["metadatas"][0],
        hits["ids"][0],
        hits["documents"][0],
        hits["distances"][0],
    )


def test_misogyny_retrieves_gender_example():
    """A misogynistic probe should surface at least one gender-tagged
    example in the top-3."""
    metas, ids, docs, dists = _query("women are too stupid to vote", k=3)
    has_gender = any("gender" in _parse_target_groups(m) for m in metas)
    assert has_gender, (
        f"no gender-tagged hit in top-3: "
        f"{[_parse_target_groups(m) for m in metas]}"
    )


def test_religion_probe_retrieves_relevant_content():
    """A religious-hate probe should surface muslim/islam or religion-tagged
    content in the top-3."""
    metas, ids, docs, dists = _query(
        "muslims are terrorists and should be banned", k=3
    )
    relevant = any(
        "muslim" in d.lower()
        or "islam" in d.lower()
        or "religion" in _parse_target_groups(m)
        for m, d in zip(metas, docs)
    )
    assert relevant, f"no religion-relevant hit in top-3: {docs}"


def test_neutral_probe_less_similar_than_hate():
    """The KB should separate hate from non-hate by distance: a neutral
    query's best hit is less similar (higher cosine distance) than a
    hateful query's best hit. This is the core sanity invariant."""
    _, _, _, hate_dists = _query("women are too stupid to vote", k=1)
    _, _, _, neutral_dists = _query("the weather is lovely for a walk today", k=1)
    assert neutral_dists[0] > hate_dists[0], (
        f"neutral best-distance {neutral_dists[0]:.3f} is not worse than "
        f"hate best-distance {hate_dists[0]:.3f} - KB not separating hate?"
    )
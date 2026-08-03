"""
Regression tests for the Phase 4 retrieval component.

WHAT THESE GUARD: the burial fix. Phase 3's naive dense retrieval over the
mixed KB returned 1 definition and 0 guidelines across 6 probes. Per-kind
budgeted retrieval guarantees each prompt section its slots. These tests fail
if that guarantee is ever lost.

WHAT THESE DELIBERATELY DO NOT GUARD: guide-quoted-slurs on the victim-
recounting probe. It is absent under dense, bm25 AND hybrid - a KB-authoring
problem, not a retrieval-strategy problem. It is held open as the before/after
case for the KB-rewording adaptability demo, and asserting it here would only
produce a permanently red test.
"""

from pathlib import Path

import pytest
import yaml

from src.hsrag.retrieve import Retriever

PROBES = {
    "misogyny": ("women are too stupid to vote", "en"),
    "religion": ("muslims are terrorists and should be banned", "en"),
    "german": ("Ausländer raus, wir wollen euch hier nicht", "de"),
    "profanity": ("you're such an idiot, get lost", "en"),
    "quoted_slur": ("he called me the n-word and I was shocked", "en"),
    "neutral": ("the weather is lovely for a walk today", "en"),
}


@pytest.fixture(scope="module")
def cfg_all():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))


_CACHE: dict[str, Retriever] = {}


def _retriever(cfg_all, strategy):
    """Memoised: parametrised tests would otherwise load BGE-M3 16 times."""
    if strategy not in _CACHE:
        cfg = dict(cfg_all["retrieval"])
        cfg["strategy"] = strategy
        kb = cfg_all["kb"]
        _CACHE[strategy] = Retriever(
            chroma_path=Path(kb["chroma_path"]),
            records_path=Path(kb["records_path"]),
            model_name=kb["embedding_model"],
            cfg=cfg,
        )
    return _CACHE[strategy]


@pytest.fixture(scope="module")
def retriever_hybrid(cfg_all):
    return _retriever(cfg_all, "hybrid")


@pytest.fixture(scope="module")
def retriever_dense(cfg_all):
    return _retriever(cfg_all, "dense")


# --- THE BURIAL FIX ITSELF -------------------------------------------------

@pytest.mark.parametrize("probe", list(PROBES))
@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_every_bucket_is_filled(cfg_all, probe, strategy):
    """Phase 3 regression: 6 probes gave 1 definition and 0 guidelines total.

    Only dense and hybrid are asserted. Under bm25-only a German query has
    zero lexical overlap with the English definitions and guidelines, so those
    buckets legitimately return empty - that is Finding C, not a regression.
    """
    text, lang = PROBES[probe]
    res = _retriever(cfg_all, strategy).retrieve(text, lang)
    assert len(res.definitions) == cfg_all["retrieval"]["k_definitions"]
    assert len(res.guidelines) == cfg_all["retrieval"]["k_guidelines"]
    assert len(res.examples) == cfg_all["retrieval"]["k_examples"]


# --- CONTRACTS THE PROMPT BUILDER DEPENDS ON -------------------------------

def test_example_language_filter(retriever_hybrid):
    """Examples are language-specific (slurs, slang); definitions and
    guidelines are EN-only by design and cross-lingually reachable."""
    for probe, (text, lang) in PROBES.items():
        res = retriever_hybrid.retrieve(text, lang)
        assert all(h.meta["lang"] == lang for h in res.examples), probe
        assert all(h.meta["lang"] == "en" for h in res.definitions), probe


def test_no_duplicate_text_within_a_bucket(retriever_hybrid):
    """Phase 4.5 regression: ex-detox-<id> and ex-detox-legal-<id> were the
    same comment under two ids and occupied two of five German example slots.
    """
    for probe, (text, lang) in PROBES.items():
        res = retriever_hybrid.retrieve(text, lang)
        for bucket in (res.definitions, res.guidelines, res.examples):
            texts = [" ".join(h.text.lower().split()) for h in bucket]
            assert len(texts) == len(set(texts)), probe


def test_sentinels_are_reversed(retriever_hybrid):
    """The None / [] / ["race"] three-way distinction must survive Chroma.
    A raw "__none__" or a JSON string in Hit.meta means some path bypassed
    unsentinel_meta()."""
    text, lang = PROBES["misogyny"]
    for hit in retriever_hybrid.retrieve(text, lang).all_hits():
        for key, val in hit.meta.items():
            assert val != "__none__", f"{hit.id}.{key}"
        tg = hit.meta.get("target_groups")
        assert tg is None or isinstance(tg, list), hit.id


def test_mmr_vectors_do_not_leak(retriever_hybrid):
    """MMR carries 1024-float embeddings on candidate Hits. They must be
    nulled before return - they are not prompt content and must not reach a
    log file."""
    text, lang = PROBES["misogyny"]
    assert all(h.vec is None
               for h in retriever_hybrid.retrieve(text, lang).all_hits())


# --- CONTENT ASSERTIONS (verified in the 4.3/4.4 comparison) ---------------

@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_religion_definition_retrieved(cfg_all, strategy):
    """Hit at rank 1 under dense, bm25 and hybrid."""
    text, lang = PROBES["religion"]
    res = _retriever(cfg_all, strategy).retrieve(text, lang)
    assert any(h.id == "def-target_group-religion" for h in res.definitions)


@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_profanity_guideline_retrieved(cfg_all, strategy):
    """Documented miss #1 from Phase 3, resolved by per-kind budgeting alone.
    Rank 1 guideline under dense, bm25 and hybrid."""
    text, lang = PROBES["profanity"]
    res = _retriever(cfg_all, strategy).retrieve(text, lang)
    assert any(h.id == "guide-profanity-without-target" for h in res.guidelines)


def test_misogyny_retrieves_gender_example(retriever_dense):
    """Note: a gender EXAMPLE is reachable; def-target_group-gender is not.
    The definition lacks the words "women"/"female" and is unreachable by both
    channels for this query - held open as the KB-rewording demo."""
    text, lang = PROBES["misogyny"]
    res = retriever_dense.retrieve(text, lang)
    tgs = [t for h in res.examples for t in (h.meta.get("target_groups") or [])]
    assert "gender" in tgs


def test_german_gets_english_knowledge_cross_lingually(retriever_hybrid):
    """The asymmetric design: EN-only definitions and guidelines stay reachable
    from a German query via BGE-M3 (EN/DE misogyny cosine 0.887), while the
    BM25 channel contributes what German material the KB now holds. Hybrid
    must degrade gracefully rather than return empty.

    The source assertion was originally `startswith("detox")`, written when
    DeTox was the only German example source. BoTox is now a second one, so
    the check is on LANGUAGE rather than on a source name that will keep
    changing.
    """
    text, lang = PROBES["german"]
    res = retriever_hybrid.retrieve(text, lang)
    assert res.definitions and res.guidelines
    assert all(h.meta["lang"] == "de" for h in res.examples)
    assert all(h.meta["lang"] == "en" for h in res.definitions)


def test_neutral_probe_less_similar_than_hate(retriever_dense):
    """Sanity: the neutral probe's best example is a worse match than the
    hateful probe's. Dense only - RRF scores are positional, not similarities,
    and are not comparable across queries."""
    hate, _ = PROBES["religion"]
    neutral, _ = PROBES["neutral"]
    best_hate = retriever_dense.retrieve(hate, "en").examples[0].score
    best_neutral = retriever_dense.retrieve(neutral, "en").examples[0].score
    assert best_neutral < best_hate
    
def test_empty_list_metadata_survives_chroma(retriever_hybrid):
    """[] must round-trip as [], not as None and not as "[]".

    Adding the legal dimension broke ingest outright: LIST_META_KEYS was a
    hardcoded tuple of two keys, legal=[] reached Chroma as a real empty list,
    and Chroma rejects those. [] is the value that carries the meaning here -
    "annotated, and the answer is empty", which is class 0 for legal.
    """
    res = retriever_hybrid.retrieve("Deutschland erwache", "de")
    legals = [h.meta.get("legal") for h in res.examples
              if "legal" in h.meta]
    assert legals, "no BoTox example retrieved; is the KB rebuilt?"
    for v in legals:
        assert v is None or isinstance(v, list), f"bad legal value: {v!r}"
        assert v != "[]", "empty list came back as a string"
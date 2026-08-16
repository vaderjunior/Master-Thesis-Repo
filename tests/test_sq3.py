"""
Guards for the SQ3 feedback machinery.

WHY THESE AND NOT MORE. This project's canonical failure is a new dimension
silently dropped by a function that does not crash: adding `legal` broke four
of them and discarded 974 of 1,350 predictions while producing a report that
looked exactly like a model which had never learned the task. SQ3 adds a new
record KIND, a new retrieval BUCKET, a new prompt GROUP and a new frozen
PARTITION, so each of those hops gets an explicit test that it survives.

Everything here is pure or reads artefacts already on disk. Tests that need a
built Chroma index skip when it is absent, so the suite still passes on a
fresh clone.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.hsrag.classify import Arms
from src.hsrag.kb import LIST_META_KEYS, _flatten_meta, unsentinel_meta
from src.hsrag.metrics import score_multilabel
from src.hsrag.prompt import PromptContext, build_prompt
from src.hsrag.retrieve import Hit, RetrievalResult, Retriever

PROCESSED = Path("data/processed")
KB = Path("kb")
SUBSET = PROCESSED / "en_dev_eval_sq3_types.parquet"
RECORDS = KB / "records_sq3_fb_r4.jsonl"
CHROMA = KB / "chroma_sq3_fb_r4"
SENTINEL = Path("experiments") / "sq3_sentinel_r0.json"
LOG = KB / "feedback_log.jsonl"

needs_subset = pytest.mark.skipif(not SUBSET.exists(), reason="no SQ3 subset")
needs_records = pytest.mark.skipif(not RECORDS.exists(), reason="no SQ3 KB")
# SKIP_CHROMA_TESTS is an escape hatch: chromadb is SQLite-backed and several
# PersistentClient instances in one process can deadlock below the Python
# level, where Ctrl+C cannot interrupt. The test below opens only one, but the
# suite already opens another on kb/chroma in test_retrieval_probes.
needs_chroma = pytest.mark.skipif(
    not CHROMA.exists() or bool(os.environ.get("SKIP_CHROMA_TESTS")),
    reason="no SQ3 index, or SKIP_CHROMA_TESTS is set")
needs_log = pytest.mark.skipif(not LOG.exists(), reason="no feedback log")


def _hit(rec_id="fb-deadbeef0001", kind="feedback"):
    return Hit(id=rec_id, kind=kind, text="a corrected example",
               meta={"kind": kind, "lang": "en", "gate": True,
                     "hate_types": ["irony"], "target_groups": None,
                     "severity": None, "illustrative_only": False},
               score=0.5, via="rrf")


# ------------------------------------------------- the record kind survives

def test_flatten_and_unsentinel_round_trip_a_feedback_record():
    """Chroma cannot store None or lists, so kb.py encodes them. The gold
    labels on a correction are exactly the payload that must survive, and
    this is the path that broke when `legal` was added."""
    rec = {"id": "fb-x", "kind": "feedback", "lang": "en",
           "source": "sq3-feedback-r1", "dimension": None, "label": None,
           "meta": {"gate": True, "hate_types": ["irony", "grievance"],
                    "target_groups": None, "severity": None,
                    "illustrative_only": False, "round": 1,
                    "origin_item_id": "implicit_hate-1", "criterion": "contains",
                    "feedback_arm": "feedback"}}
    back = unsentinel_meta(_flatten_meta(rec))
    assert back["kind"] == "feedback"
    assert back["hate_types"] == ["irony", "grievance"]
    # None must come back as None, not as the string sentinel or as [].
    assert back["target_groups"] is None
    assert back["round"] == 1 and back["feedback_arm"] == "feedback"


def test_hate_types_is_a_list_meta_key():
    """LIST_META_KEYS is derived from the taxonomy. If a multilabel dimension
    stopped being listed there, a correction's labels would reach Chroma as a
    real list and ingest would fail - or worse, an empty one would."""
    assert "hate_types" in LIST_META_KEYS


@needs_records
def test_every_kind_in_the_sq3_kb_is_registered():
    """check_kb_schema counts per kind from VALID_KINDS. An unregistered kind
    is validated but not counted, which once reported a 380-record file as
    360 - exactly the base KB size, so it looked correct."""
    from scripts.check_kb_schema import VALID_KINDS
    kinds = {json.loads(l)["kind"] for l
             in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert kinds <= VALID_KINDS, f"unregistered kind(s): {kinds - VALID_KINDS}"
    assert "feedback" in kinds


# ------------------------------------------------- the bucket and the prompt

def test_feedback_group_reaches_the_rendered_prompt():
    """Retrieval and rendering are separate hops. A correction that is
    retrieved but never rendered would produce a clean, plausible, entirely
    meaningless null."""
    res = RetrievalResult(definitions=[], guidelines=[], examples=[],
                          feedback=[_hit()])
    ctx = PromptContext.from_retrieval(res)
    assert any(hits and hits[0].kind == "feedback"
               for _, hits in ctx.example_groups)
    user = build_prompt("some text", "en", ctx)[1]["text"]
    assert "a corrected example" in user
    assert "hate_type=[irony]" in user


def test_feedback_is_an_extra_group_not_a_replacement():
    """The feedback slot is additive. If it were carved out of k_examples,
    round 0 would render 5 regular examples and round 1 only 4, and that drop
    would ride along with the intervention as a second uncontrolled change."""
    regular = [_hit(f"ex-{i}", "example") for i in range(5)]
    res = RetrievalResult(examples=regular, feedback=[_hit()])
    ctx = PromptContext.from_retrieval(res)
    counts = {name: len(hits) for name, hits in ctx.example_groups}
    assert sum(counts.values()) == 6, counts
    assert 5 in counts.values() and 1 in counts.values()


def test_all_hits_includes_feedback():
    """all_hits() feeds logging and the debug CLI. A bucket missing from it is
    invisible to every tool that inspects retrieval."""
    res = RetrievalResult(feedback=[_hit()])
    assert [h.id for h in res.all_hits()] == ["fb-deadbeef0001"]


@needs_records
def test_few_shot_never_samples_a_correction():
    """few_shot is the static control arm and must stay frozen across rounds.
    Arms._build_fewshot filters on kind == "example"; a correction leaking in
    would make the control arm change with the KB."""
    arms = Arms(retriever=None, records_path=RECORDS, k_examples=5, seed=42)
    ctx = arms.context("few_shot", "any text", "en")
    ids = [h.id for _, hits in ctx.example_groups for h in hits]
    assert ids and not any(i.startswith("fb-") for i in ids)


@needs_chroma
def test_zero_budget_yields_no_feedback_and_leaves_examples_intact():
    """The deliberate-break case. If k_examples_feedback is 0 the bucket must
    be empty AND the regular example count untouched. Verified live before
    round 1 ran, and kept here so it cannot regress silently."""
    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    base = {**cfg["retrieval"]}
    probe = "women are too stupid to vote"

    # ONE Retriever, cfg mutated between calls. Building two opened two
    # PersistentClients, and with test_retrieval_probes already holding one on
    # kb/chroma the full suite deadlocked below the Python level where Ctrl+C
    # cannot reach. Standalone it passed in 24s, so the fault only appears in
    # the combined run - which is the one that matters.
    r = Retriever(chroma_path=CHROMA, records_path=RECORDS,
                  model_name=cfg["kb"]["embedding_model"],
                  cfg={**base, "k_examples_feedback": 0})
    off = r.retrieve(probe, "en")
    assert off.feedback == []

    r.cfg["k_examples_feedback"] = 1
    on = r.retrieve(probe, "en")
    assert len(on.feedback) == 1
    assert on.feedback[0].kind == "feedback"
    assert len(on.examples) == len(off.examples) == base["k_examples"]


# ----------------------------------------------------------- the partition

@needs_subset
def test_sq3_roles_are_disjoint_and_exhaustive():
    df = pd.read_parquet(SUBSET)
    assert "sq3_role" in df.columns
    counts = df["sq3_role"].value_counts().to_dict()
    assert set(counts) == {"pool", "held_out", "batches"}
    assert sum(counts.values()) == len(df)
    assert df["id"].is_unique


@needs_subset
def test_batch_split_is_deterministic_disjoint_and_exhaustive():
    """Both arms must receive the identical batch in the same round. A
    non-deterministic split would silently give them different items and the
    matched control would stop being matched."""
    from scripts.apply_feedback import batches
    df = pd.read_parquet(SUBSET)
    a, b = batches(df), batches(df)
    assert {k: list(v["id"]) for k, v in a.items()} == \
           {k: list(v["id"]) for k, v in b.items()}

    ids = [set(v["id"]) for v in a.values()]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not ids[i] & ids[j]
    role = df[df["sq3_role"] == "batches"]
    assert sum(len(s) for s in ids) == len(role)


@needs_subset
def test_local_gold_sets_agrees_with_the_real_scorer():
    """check_sq3_coverage mirrors score_multilabel's item filter to get
    per-item detail the scorer does not return. If the two ever drift, every
    coverage and sizing number derived from it is wrong."""
    from scripts.check_sq3_coverage import local_gold_sets
    df = pd.read_parquet(SUBSET)
    rows = [{"item_id": str(i),
             "result": {"hate": True, "hate_type": []}} for i in df["id"]]
    gold = {str(r.id): r for r in df.itertuples(index=False)}
    n = score_multilabel(rows, gold, "hate_types", "hate_type").n_items
    assert len(local_gold_sets(df, "hate_types")) == n


# ------------------------------------------------ the rule that cannot break

@needs_subset
@needs_log
def test_corrections_come_only_from_the_batches_role():
    """THE rule. held_out and pool are the never-corrected sets the learning
    curve and the sentinel are measured on. A correction from either turns
    retrieval into a lookup of the answer and voids every result after it."""
    df = pd.read_parquet(SUBSET)
    allowed = set(df.loc[df["sq3_role"] == "batches", "id"].astype(str))
    origins = {json.loads(l)["meta"]["origin_item_id"] for l
               in LOG.read_text(encoding="utf-8").splitlines() if l.strip()}
    # TRAIN top-ups are not in the subset at all and are legitimately absent.
    in_subset = origins & set(df["id"].astype(str))
    assert in_subset <= allowed, f"leaked: {sorted(in_subset - allowed)[:5]}"


@needs_subset
@needs_log
def test_sentinel_items_were_never_corrected():
    if not SENTINEL.exists():
        pytest.skip("sentinel not frozen yet")
    sent = set(json.loads(SENTINEL.read_text(encoding="utf-8"))["sentinel_ids"])
    origins = {json.loads(l)["meta"]["origin_item_id"] for l
               in LOG.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert not sent & origins


@needs_log
def test_both_arms_add_the_same_number_of_records_each_round():
    """The matched-size control only controls for KB growth if the counts
    match exactly. Unequal counts would confound error-selection with size -
    the single thing this arm exists to rule out."""
    from collections import Counter
    per = Counter()
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)["meta"]
        per[(m["round"], m["feedback_arm"])] += 1
    rounds = {r for r, _ in per}
    for r in rounds:
        assert per[(r, "feedback")] == per[(r, "control")], \
            f"round {r}: {per[(r, 'feedback')]} vs {per[(r, 'control')]}"
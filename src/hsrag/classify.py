"""
The single classification entry point, and the result record.

ONE CODE PATH, THREE ARMS. zero_shot, few_shot and rag differ ONLY in the
PromptContext handed to build_prompt. Prompt assembly, sampling, parsing,
repair and voting are byte-identical across arms. That is what makes the SQ2
comparison clean: any difference in the numbers is attributable to the context,
because nothing else varied.

WHY few_shot EXISTS AS A SEPARATE ARM: zero_shot vs rag alone cannot separate
"retrieval helped" from "having examples at all helped". few_shot supplies the
same NUMBER of examples, statically sampled with a fixed seed and no retrieval,
so the rag-minus-few_shot difference isolates retrieval itself. It carries no
definitions and no guidelines - those are retrieved knowledge, and handing them
to the control arm would give away the thing under test.

WHY ItemResult CARRIES SO MANY STAMPS: adaptability is defined as delta
macro-F1 across knowledge-base edits with the LLM frozen. A result that cannot
be attributed to a specific KB state, prompt wording and model is not evidence
for or against that claim - it is unfalsifiable. Every stamp here exists to
make a number traceable to the exact system state that produced it.
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
import threading
import time

from src.hsrag.parse import RunResult, run_once
from src.hsrag.prompt import (PromptContext, build_prompt, prompt_version,
                              taxonomy_version)
from src.hsrag.retrieve import Hit
from src.hsrag.schema import Result
from src.hsrag.vote import aggregate

ARMS = ("zero_shot", "few_shot", "rag")


def text_hash(text: str) -> str:
    """Identifies the exact string classified, independent of the item id.

    Item ids are ours; the text is the thing the model saw. If a split is ever
    rebuilt and ids shift, the hash still pins what was actually sent.
    """
    return hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def code_version() -> str:
    """Content hash over the library source.

    THE HOLE THIS CLOSES. prompt_version hashes the template; taxonomy_version
    hashes the label space. Neither sees the library code. run_slice1 is
    resumable, so a run interrupted, patched and resumed writes items produced
    by two different versions of the code into ONE file under identical
    stamps, and make_comparability reports no issue because every stamp it
    checks genuinely is constant.

    Not hypothetical. legal_dev_peasec (2026-08-02) carries `legal` in the
    vote on its last 25 items and not on its first 150. It stopped at 150
    because --n defaulted there with no manifest limit; by the time it was
    resumed, both that bug and the missing `legal = majority_labels("legal")`
    in vote.py had been fixed. Model, prompt, KB, temperature and workers were
    identical throughout. Its legal macro-F1 read 0.171 against 0.609 for its
    replicate, and it sat in the ledger as a valid third replicate for twelve
    days.

    Cached for the process lifetime, which is the correct semantics rather
    than an optimisation: Python cannot swap loaded modules mid-run, so the
    code that produced item 1 is the code that produced item 500. An edit
    lands on the NEXT process, and that is exactly the boundary worth
    stamping.

    Same CRLF caveat as prompt_version: git normalises line endings on
    checkout, so this hash is not comparable across a fresh clone.
    """
    h = hashlib.sha256()
    for p in sorted(Path("src/hsrag").rglob("*.py")):
        h.update(p.as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


@dataclass
class ItemResult:
    """One item, one arm, n sampled runs, one voted prediction."""

    # identity
    item_id: str
    text_hash: str
    lang: str
    arm: str
    
    # Load and time context. Added Phase 6: the parse-failure rate rose during
    # the Slice 1 run and the later portion ran concurrently, but ItemResult
    # carried no timestamp, so the question could only be answered with file
    # position as a proxy. Stamped from now on so it is answerable directly.
    timestamp: float = 0.0
    workers: int = 1
    temperature: float | None = None

    # attribution - the falsifiability stamps
    active_model: str | None = None
    prompt_version: str | None = None
    # The label space is read from taxonomy.yaml at render time and reaches
    # every prompt in every arm, so a taxonomy edit changes the system while
    # prompt_version stays put. Stamped on ALL arms, unlike kb_version:
    # zero_shot consults no knowledge base but does receive the label space.
    taxonomy_version: str | None = None
    # Hash of src/hsrag/*.py. The resumable runner means one results file can
    # span a code change; this is the only stamp that can see it.
    code_version: str | None = None
    kb_version: str | None = None            # None for zero_shot: no KB was used
    retrieval_config: dict | None = None     # the RESOLVED dict, not the file
    retrieved: dict = field(default_factory=dict)   # bucket -> [hit ids]

    # prediction
    result: dict | None = None               # voted Result, as a plain dict
    uncertain: bool = False
    n_valid: int = 0
    n_runs: int = 0
    agreement: dict = field(default_factory=dict)

    # honesty numbers
    parse_failures: int = 0
    repairs: int = 0
    normalisations: int = 0
    gate_normalised: int = 0

    # raw material: one list of attempts per run. alpha (Phase 6) needs the
    # per-run predictions, and the n=1 arm is scored from run 0 alone.
    raw_runs: list = field(default_factory=list)
    run_predictions: list = field(default_factory=list)
    latencies: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Arms:
    """Builds the PromptContext for each arm. Holds the retriever once."""

    def __init__(self, retriever=None, records_path: Path | None = None,
                 k_examples: int = 5, seed: int = 42):
        self.retriever = retriever
        self.k_examples = k_examples
        self._fewshot: dict[str, list[Hit]] = {}
        if records_path is not None:
            self._build_fewshot(Path(records_path), seed)
        # Retrieval touches a shared SentenceTransformer and Chroma collection.
        # The API call is the bottleneck (~30 s vs ~50 ms), so serialising
        # retrieval costs almost nothing and removes a whole class of
        # hard-to-reproduce threading bug.
        self._lock = threading.Lock()

    def _build_fewshot(self, records_path: Path, seed: int) -> None:
        """Sample a fixed few-shot set per language, once, deterministically.

        Drawn from the same KB example pool the rag arm retrieves from, so the
        two arms differ in HOW examples are chosen, not in what pool they came
        from. illustrative_only records are excluded: they are background legal
        context, not labelled evidence, and including them would make the
        control arm's examples weaker in a way unrelated to retrieval.
        """
        records = [json.loads(line) for line in
                   records_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        for lang in ("en", "de"):
            pool = [r for r in records
                    if r["kind"] == "example" and r["lang"] == lang
                    and not r.get("meta", {}).get("illustrative_only")]
            rng = random.Random(seed)
            picked = rng.sample(pool, min(self.k_examples, len(pool)))
            self._fewshot[lang] = [
                Hit(id=r["id"], kind=r["kind"], text=r["text"],
                    meta={**r.get("meta", {}), "lang": r["lang"],
                          "source": r["source"], "dimension": r["dimension"],
                          "label": r["label"], "kind": r["kind"]},
                    score=0.0, via="static")
                for r in picked
            ]

    def context(self, arm: str, text: str, lang: str) -> PromptContext:
        if arm == "zero_shot":
            return PromptContext.zero_shot()
        if arm == "few_shot":
            hits = self._fewshot.get(lang, [])
            return PromptContext(
                example_groups=[("Labelled examples", hits)] if hits else [])
        if arm == "rag":
            if self.retriever is None:
                raise ValueError("rag arm needs a retriever")
            with self._lock:
                res = self.retriever.retrieve(text, lang)
            return PromptContext.from_retrieval(res)
        raise ValueError(f"unknown arm: {arm}")


def _hit_ids(ctx: PromptContext) -> dict:
    """Retrieval provenance.

    Needed by the SQ3 generalisation probe and the Finding-G rewording demo:
    both ask which knowledge was in front of the model when it decided, and
    neither can be answered after the fact without this.
    """
    out = {"definitions": [h.id for h in ctx.definitions],
           "guidelines": [h.id for h in ctx.guidelines]}
    for name, hits in ctx.example_groups:
        out[name] = [h.id for h in hits]
    return out


def classify(item_id: str, text: str, lang: str, arm: str, client, arms: Arms,
             n_votes: int = 3, max_repairs: int = 2,
             pinned_model: str | None = None,
             kb_version: str | None = None, workers: int = 1, temperature: float | None = None) -> ItemResult:
    """Classify one item under one arm. n sampled runs, then a vote."""
    ctx = arms.context(arm, text, lang)
    messages = build_prompt(text, lang, ctx)

    out = ItemResult(
        item_id=item_id, text_hash=text_hash(text), lang=lang, arm=arm,
        prompt_version=prompt_version(),
        taxonomy_version=taxonomy_version(),
        code_version=code_version(),
        # zero_shot and few_shot consult no knowledge base and run no
        # retrieval, so stamping a kb_version on them would imply a dependency
        # that does not exist.
        kb_version=kb_version if arm == "rag" else None,
        retrieval_config=dict(arms.retriever.cfg) if arm == "rag" else None,
        retrieved=_hit_ids(ctx),
        n_runs=n_votes,
        timestamp=time.time(),
        workers=workers,
        temperature=temperature,
    )

    runs: list[RunResult] = []
    for _ in range(n_votes):
        r = run_once(client, messages, max_repairs=max_repairs)
        runs.append(r)
        out.raw_runs.append(r.raw_outputs)
        out.latencies.append(round(r.latency_s, 2))
        out.repairs += r.repairs
        out.normalisations += r.normalisations
        out.gate_normalised += int(r.gate_normalised)
        out.parse_failures += int(r.parse_failure)
        out.run_predictions.append(
            r.result.model_dump() if r.result is not None else None)

    if pinned_model is not None and client.active_model != pinned_model:
        # Fail loudly. A result silently produced by a different model is
        # worse than no result: it looks valid and is not comparable.
        raise RuntimeError(
            f"expected pinned model {pinned_model}, got {client.active_model}")
    out.active_model = client.active_model

    vote = aggregate(runs)
    out.result = vote.result.model_dump() if vote.result is not None else None
    out.uncertain = vote.uncertain
    out.n_valid = vote.n_valid
    out.agreement = vote.agreement
    return out
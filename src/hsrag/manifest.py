"""
Experiment manifests: an experiment IS its manifest.

WHY: Slice 1 was configured by command-line flags plus whatever config.yaml
happened to contain at the time. That is fine for one run and impossible for a
suite - six months from now, "which retrieval config produced this number" has
to be answerable from the results file alone, not from shell history.

RESOLVE AND STAMP. The manifest on disk is partial: it names a subset and an
arm list and leaves the rest to config.yaml. resolve() fills every gap and
stamps the run-time facts (kb_version, prompt_version), producing a complete
description that is written into the results header. Editing the YAML later
cannot retroactively change what a past run did.

The cost estimator exists because the Phase 5 estimate was wrong by 8x and
someone confirmed an 11-hour job believing it was 80 minutes.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml

MANIFESTS = Path("experiments/manifests")
PROCESSED = Path("data/processed")
CONFIG = Path("config/config.yaml")

VALID_ARMS = {"zero_shot", "few_shot", "rag"}
VALID_GATE_MAPPING = {"strict", "lenient", "both"}

# Providers that may receive German data. DeTox arrived under Zenodo terms
# with restrictions, so German items must not leave TU or PEASEC
# infrastructure (decision Q8). Asserted rather than remembered.
GERMAN_SAFE_PROVIDERS = {"tudagpt", "peasec", "mock"}

# Measured, not guessed: 13-25 s/call at 4 workers across the Slice 1 run,
# varying with server load. 20 is the midpoint. Any tighter estimate is
# unreliable and this number should stay conservative.
DEFAULT_SECONDS_PER_CALL = 20.0


@dataclass
class Manifest:
    name: str
    subset: str
    arms: list[str]
    provider: str = "tudagpt"
    pinned_model: str | None = None
    allow_fallback: bool = False
    temperature: float = 1.0
    n_votes: int = 3
    max_repair_retries: int = 2
    workers: int = 4
    limit: int | None = None
    seed: int = 42
    description: str = ""

    # rag only; None means "take config.yaml as-is"
    retrieval: dict | None = None

    # scoring options, resolved here so a result set records how it is meant
    # to be read rather than leaving that to whoever scores it later
    gate_mapping: str = "both"
    hate_type_scoring: str = "multilabel"

    # stamped at resolve time, never written by hand
    kb_version: str | None = None
    prompt_version: str | None = None
    n_items: int = 0
    lang: str = ""

    def validate(self) -> None:
        bad = set(self.arms) - VALID_ARMS
        assert not bad, f"unknown arms: {bad}"
        assert self.arms, "manifest has no arms"
        assert self.gate_mapping in VALID_GATE_MAPPING, self.gate_mapping
        assert self.n_votes >= 1, "n_votes must be at least 1"

        path = PROCESSED / f"{self.subset}.parquet"
        assert path.exists(), f"subset not found: {path}"

        # Q8: data governance is an assertion, not a note in a log file.
        if self.lang == "de":
            assert self.provider in GERMAN_SAFE_PROVIDERS, (
                f"provider '{self.provider}' may not receive German data "
                f"(DeTox Zenodo terms). Allowed: {sorted(GERMAN_SAFE_PROVIDERS)}")

        # A test subset is touched only by final runs. Not forbidden here, but
        # it must never happen by accident.
        if "_test_" in self.subset:
            print(f"  !! WARNING: {self.subset} is a TEST subset. "
                  f"Tuning must happen on dev.")

    @property
    def calls(self) -> int:
        return self.n_items * len(self.arms) * self.n_votes

    def estimate(self, seconds_per_call: float = DEFAULT_SECONDS_PER_CALL) -> dict:
        # seconds_per_call is WALL-CLOCK throughput at the configured worker
        # count, not per-worker latency, so it must not be divided by workers
        # again - the parallelism is already inside the measured number.
        wall = self.calls * seconds_per_call
        return {"calls": self.calls, "seconds": wall, "minutes": wall / 60,
                "hours": wall / 3600, "seconds_per_call": seconds_per_call}

    def hash(self) -> str:
        """Identity of the resolved manifest. Two runs with the same hash are
        the same experiment; a different hash means something changed."""
        payload = {k: v for k, v in asdict(self).items()
                   if k not in ("description", "name")}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def header(self) -> str:
        """First line of the results file: a self-describing result set."""
        return json.dumps({"_manifest": asdict(self), "_hash": self.hash()},
                          ensure_ascii=False)


def load(name_or_path: str) -> Manifest:
    path = Path(name_or_path)
    if not path.exists():
        path = MANIFESTS / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Manifest(**raw)


def resolve(m: Manifest, retriever=None) -> Manifest:
    """Fill gaps from config.yaml and stamp run-time facts.

    Called once before a run. After this the manifest is complete, and the
    complete version is what gets written to the results header - so a later
    edit to the YAML or to config.yaml cannot silently change the description
    of a run that already happened.
    """
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    if m.pinned_model is None:
        m.pinned_model = cfg["classify"]["pinned_model"]
    if m.retrieval is None and "rag" in m.arms:
        m.retrieval = dict(cfg["retrieval"])

    df = pd.read_parquet(PROCESSED / f"{m.subset}.parquet")
    if m.limit:
        df = df.head(m.limit)
    m.n_items = len(df)
    langs = set(df["lang"])
    assert len(langs) == 1, f"subset mixes languages: {langs}"
    m.lang = langs.pop()

    from src.hsrag.prompt import prompt_version
    m.prompt_version = prompt_version()

    # kb_version only where a knowledge base is actually consulted. Stamping
    # it on a zero-shot manifest would imply a dependency that does not exist.
    if "rag" in m.arms and retriever is not None:
        m.kb_version = retriever.col.metadata.get("kb_version")

    m.validate()
    return m


def describe(m: Manifest, seconds_per_call: float = DEFAULT_SECONDS_PER_CALL) -> str:
    e = m.estimate(seconds_per_call)
    lines = [
        f"manifest      {m.name}  [{m.hash()}]",
        f"  {m.description}" if m.description else "",
        f"subset        {m.subset}  ({m.n_items} items, lang={m.lang})",
        f"arms          {', '.join(m.arms)}",
        f"model         {m.provider} / {m.pinned_model}  "
        f"fallback={'ON' if m.allow_fallback else 'off'}  T={m.temperature}",
        f"sampling      n_votes={m.n_votes}  repairs<={m.max_repair_retries}  "
        f"workers={m.workers}",
        f"prompt        {m.prompt_version}",
        f"kb            {m.kb_version or '(not used)'}",
        f"retrieval     {m.retrieval if m.retrieval else '(not used)'}",
        f"scoring       gate_mapping={m.gate_mapping}  "
        f"hate_type={m.hate_type_scoring}",
        "",
        f"COST          {e['calls']} calls "
        f"({m.n_items} items x {len(m.arms)} arms x {m.n_votes} votes)",
        f"              ~{e['minutes']:.0f} min ({e['hours']:.1f} h) "
        f"at {e['seconds_per_call']:.0f} s/call with {m.workers} workers",
        f"              measured range 13-25 s/call; treat any tighter "
        f"estimate as unreliable",
    ]
    return "\n".join(l for l in lines if l)
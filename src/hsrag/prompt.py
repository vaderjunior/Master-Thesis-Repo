"""
Prompt assembly: turn retrieved knowledge into the message list for the LLM.

PURE FUNCTION BY DESIGN. build_prompt() performs no retrieval and makes no API
call - it only renders what it is handed. That separation is what lets
check_prompt.py inspect a full prompt for zero cost, and it is what makes the
SQ2 comparison clean: zero-shot, few-shot and RAG differ ONLY in the context
they pass in, never in the code path.

WHY example_groups IS A LIST OF GROUPS AND NOT ONE LIST:
Phase 4 Finding E showed retrieval hands the model only hateful evidence for
non-hateful input (5/5 gate=True on both not-hate probes). Phase 8's fix is a
reserved negative-example budget; a separate legal-illustration budget is a
second candidate. Both are new GROUPS, not a rewrite. Legal illustrations
already have to render differently today, so the multi-group shape is needed
regardless - it is built general rather than retrofitted.

NO SCORES ANYWHERE. RRF scores are positional, not similarities, and are not
comparable across queries or strategies. Nothing in the prompt displays or
compares them, and the builder never assumes how a bucket was produced.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.hsrag.retrieve import Hit

TEMPLATE = Path("prompts/classify_v1.txt")
TAXONOMY = Path("config/taxonomy.yaml")

_SECTION = re.compile(r"^=== (\w+) ===$", re.MULTILINE)


@dataclass
class PromptContext:
    """What the prompt renders. Empty everywhere = the zero-shot arm."""
    definitions: list[Hit] = field(default_factory=list)
    guidelines: list[Hit] = field(default_factory=list)
    example_groups: list[tuple[str, list[Hit]]] = field(default_factory=list)

    @classmethod
    def zero_shot(cls) -> "PromptContext":
        return cls()

    @classmethod
    def from_retrieval(cls, res) -> "PromptContext":
        """RetrievalResult -> PromptContext. Examples split into groups on
        illustrative_only: legal illustrations are background context on German
        legal framing, not labelled evidence, and must not be counted or
        rendered as such."""
        regular = [h for h in res.examples if not h.meta.get("illustrative_only")]
        legal = [h for h in res.examples if h.meta.get("illustrative_only")]
        groups = []
        if regular:
            groups.append(("Labelled examples", regular))
        if legal:
            groups.append(("Legal context illustrations", legal))
        return cls(definitions=res.definitions, guidelines=res.guidelines,
                   example_groups=groups)


def _load_template() -> dict[str, str]:
    """Split the template file on '=== NAME ===' markers."""
    raw = TEMPLATE.read_text(encoding="utf-8")
    parts = _SECTION.split(raw)
    # parts[0] is anything before the first marker; then name, body, name, body
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}

def _flatten(text: str) -> str:
    """Collapse internal whitespace in example text.

    Social-media text carries embedded newlines and blank lines. Rendered
    verbatim inside a markdown list, a blank line inside an example visually
    terminates the list and the model can lose track of where one example ends
    and the next begins. German examples are the longest and worst affected.
    The text's content is unchanged; only its line structure is normalised.
    """
    return " ".join(str(text).split())

def prompt_version() -> str:
    """Template name + content hash. A reworded prompt is a different version,
    so results can never be silently attributed to the wrong wording - the same
    reasoning as kb_version."""
    h = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()[:8]
    return f"{TEMPLATE.stem}-{h}"


def render_label_space(tax: dict | None = None) -> str:
    """The allowed values for each dimension, read from taxonomy.yaml.

    ALWAYS PRESENT, IN EVERY ARM INCLUDING ZERO-SHOT. The label space is task
    specification, not retrieved knowledge - a model that does not know the
    permitted values is not doing the task at all. What varies between SQ2 arms
    is the retrieved knowledge, never the task definition.
    """
    tax = tax or yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    lines = []
    for dim_name, dim in tax["dimensions"].items():
        dtype = dim["type"]
        if dtype == "binary":
            lines.append(f"- {dim_name} ({dtype}): true or false")
            continue
        labels = dim["labels"]
        names = list(labels) if isinstance(labels, dict) else list(labels)
        lines.append(f"- {dim_name} ({dtype}): {', '.join(names)}")
    return "\n".join(lines)


def _render_gold(meta: dict) -> str:
    """Gold labels of one example.

    Renders only dimensions that are NOT None. None means the source dataset
    never annotated that dimension; [] means it did and the answer is empty.
    Printing 'target_group: none' for a GAHD example would assert something the
    data never claimed, and the whole point of the None / [] distinction is to
    avoid exactly that.
    """
    bits = [f"hate={str(bool(meta.get('gate'))).lower()}"]
    for key, out in [("target_groups", "target_group"),
                     ("hate_types", "hate_type")]:
        val = meta.get(key)
        if val is not None:
            bits.append(f"{out}=[{', '.join(val)}]")
    if meta.get("severity") is not None:
        bits.append(f"severity={meta['severity']}")
    return ", ".join(bits)


def build_prompt(text: str, lang: str, ctx: PromptContext) -> list[dict]:
    """Render the message list. No retrieval, no API, no I/O beyond templates."""
    tpl = _load_template()
    system = tpl["SYSTEM"].format(label_space=render_label_space())

    parts = []

    if ctx.definitions:
        parts.append("## Label definitions\n" + "\n".join(
            # dimension AND label: "gender" is a target_group value while
            # "threatening" is a hate_type value, and they would otherwise
            # render identically. "other" is genuinely ambiguous across
            # dimensions, and severity definitions would render as bare "low".
            f"- {h.meta.get('dimension')}"
            + (f" / {h.meta['label']}" if h.meta.get("label") else "")
            + f": {_flatten(h.text)}"
            for h in ctx.definitions))

    if ctx.guidelines:
        parts.append("## Annotation guidelines\n" + "\n".join(
            f"- {h.text}" for h in ctx.guidelines))

    for name, hits in ctx.example_groups:
        if not hits:
            continue                      # empty groups render nothing
        legal = any(h.meta.get("illustrative_only") for h in hits)
        # The few_shot arm's examples are sampled statically and have nothing
        # to do with the input, so the retrieval caution is false there. via
        # is "static" for those hits and a retrieval channel otherwise.
        static = any(h.via == "static" for h in hits)
        key = ("ILLUSTRATIONS_CAUTION" if legal
               else "STATIC_EXAMPLES_CAUTION" if static
               else "EXAMPLES_CAUTION")
        caution = tpl[key]
        rows = []
        for h in hits:
            if h.meta.get("illustrative_only"):
                # paragraph comes from meta.stgb - these are mostly §185, and
                # calling them §130 examples would misdescribe the data
                tag = h.meta.get("stgb", "unspecified")
                rows.append(f'- "{_flatten(h.text)}"\n  flagged under StGB §{tag}')
            else:
                rows.append(f'- "{_flatten(h.text)}"\n  labels: {_render_gold(h.meta)}')
        parts.append(f"## {name}\n{caution}\n\n" + "\n".join(rows))

    parts.append(f"## Text to classify (language: {lang})\n{text}")

    return [
        {"role": "system", "text": system},
        {"role": "user", "text": "\n\n".join(parts)},
    ]
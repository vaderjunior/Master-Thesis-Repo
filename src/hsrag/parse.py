"""
Parse an LLM response into a validated Result, with bounded repair retries.

WHY THIS EXISTS (assumption A4): constrained decoding needs logit access,
which this remote API does not provide, so the output cannot be made
structurally valid by construction. The substitute is schema-validated
decoding with bounded repair. The price is that parse failure becomes a real
event, and the honest response is to bound the retries by config and REPORT
the failure rate rather than retry until something sticks.

THREE LAYERS, DELIBERATELY ORDERED:
  1. extraction  - strip the wrapping the model put around its JSON
  2. normalisation (schema.normalise_raw) - repair presentation drift
  3. repair retry - spend an API call only when 1 and 2 cannot save it

Layers 1 and 2 are free. A repair retry costs a call and adds latency, so it
is the last resort, not the first. Every layer is counted separately, so the
log can distinguish "the model needed prompting again" from "the model typed
the label with a capital letter".
"""

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.hsrag.schema import Result, normalise_raw

# Some models emit reasoning traces before the answer. The pinned model
# advertises output ["text", "thought"], so it CAN do this even though it has
# not to date; a change in server-side defaults would otherwise surface as an
# unexplained spike in the parse-failure rate.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(raw: str) -> str:
    """Pull the JSON object out of whatever the model wrapped it in.

    Handles: reasoning traces, markdown fences, and prose either side
    ("Sure! Here is your answer: {...} Let me know if..."). Brace matching is
    depth-aware and string-aware, because a brace inside the reasoning text
    would otherwise end the object early.
    """
    text = _THINK.sub("", raw or "").strip()
    text = _FENCE.sub("", text).strip()

    start = text.find("{")
    if start == -1:
        return text                      # no object at all; let json.loads fail

    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]                  # unbalanced; let json.loads report it


@dataclass
class RunResult:
    """One sampled run. The accounting fields are the reported honesty numbers."""
    result: Result | None            # None => parse_failure
    raw_outputs: list[str] = field(default_factory=list)
    repairs: int = 0                 # API calls spent on repair
    normalisations: int = 0          # free fixes applied before validation
    gate_normalised: bool = False    # model contradicted itself on the gate
    parse_failure: bool = False
    error: str | None = None
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None


def parse_response(raw: str) -> tuple[Result | None, int, str | None]:
    """(result, normalisations, error). No API calls."""
    try:
        obj = json.loads(extract_json(raw))
    except json.JSONDecodeError as e:
        return None, 0, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, 0, f"expected a JSON object, got {type(obj).__name__}"

    obj, n = normalise_raw(obj)
    try:
        return Result(**obj), n, None
    except ValidationError as e:
        return None, n, f"schema violation: {e.errors()[0]['msg']} " \
                        f"at {e.errors()[0]['loc']}"


REPAIR_TEMPLATE = (
    "Your previous output failed validation: {error}\n"
    "Return ONLY the corrected JSON object, with no prose and no code fences."
)


def run_once(client, messages: list[dict], max_repairs: int = 2) -> RunResult:
    """One sampled run: call, parse, repair up to max_repairs, give up.

    Repair is multi-turn - the failed output is shown back to the model as its
    own turn, followed by the error. Telling a model what it got wrong without
    showing it what it wrote leaves it repairing something it cannot see.
    """
    import time

    out = RunResult(result=None)
    convo = list(messages)

    for attempt in range(max_repairs + 1):
        t0 = time.time()
        raw = client.complete(convo)
        out.latency_s += time.time() - t0
        out.raw_outputs.append(raw)

        result, n, err = parse_response(raw)
        out.normalisations += n

        if result is not None:
            out.result = result
            out.gate_normalised = result.gate_normalised
            out.repairs = attempt
            return out

        out.error = err
        if attempt == max_repairs:
            break
        convo = convo + [
            {"role": "assistant", "text": raw},
            {"role": "user", "text": REPAIR_TEMPLATE.format(error=err)},
        ]

    out.repairs = max_repairs
    out.parse_failure = True
    return out
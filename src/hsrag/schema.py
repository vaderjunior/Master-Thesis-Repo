"""
Output schema: Pydantic model generated from taxonomy.yaml.

TAXONOMY AS DATA, AGAIN. The enums are built at import time from
config/taxonomy.yaml. Adding a label to the taxonomy changes the prompt (via
render_label_space), the knowledge base (via build_kb_records) AND the accepted
output schema, with no code edit anywhere. That is the same mechanism the
adaptability claim rests on, extended to validation.

WHY VALIDATION AND NOT CONSTRAINED DECODING (assumption A4): constrained
decoding needs logit access, which a remote API does not provide. The
substitute is schema-validated decoding with bounded repair retries, and the
price is that parse failure becomes a real, reportable rate rather than
something the decoder made impossible.

SOFT FAILURES ARE NORMALISED AND COUNTED, NOT REJECTED. A model that says
hate=false but fills in a target group has contradicted itself. Rejecting the
output loses an otherwise usable answer; silently fixing it hides a real
behaviour. Fixing it and counting it gives robustness AND measurability.
"""

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

TAXONOMY = Path("config/taxonomy.yaml")


def _labels(tax: dict, dim: str) -> list[str]:
    labels = tax["dimensions"][dim]["labels"]
    return list(labels) if isinstance(labels, dict) else list(labels)


_tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))

# Enums are defined BEFORE the model that annotates against them: Pydantic
# resolves annotations at class-creation time, the same import-order trap hit
# in Phase 4 with rrf_fuse and Hit.
TargetGroup = Enum("TargetGroup", {l: l for l in _labels(_tax, "target_group")},
                   type=str)
HateType = Enum("HateType", {l: l for l in _labels(_tax, "hate_type")}, type=str)
Severity = Enum("Severity", {l: l for l in _labels(_tax, "severity")}, type=str)

# Ordinal position for the severity median in the self-consistency vote (5.5).
# Read from taxonomy order, not hardcoded, so a reordered or extended severity
# scale does not silently invert the median.
SEVERITY_ORDER = {s: i for i, s in enumerate(_labels(_tax, "severity"))}

FIELD_ALIASES = {
    "target_groups": "target_group",
    "hate_types": "hate_type",
    "is_hate": "hate",
    "hateful": "hate",
}


class Result(BaseModel):
    """One classification output."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    reasoning: str
    hate: bool
    target_group: list[TargetGroup] = []
    hate_type: list[HateType] = []
    severity: Severity | None = None

    _gate_normalised: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def gate_consistency(self):
        """hate=false must mean no sub-labels.

        Kept as a normalisation rather than a hard error: the gate is the
        dimension every dataset annotates and the one every metric depends on,
        so an otherwise-parseable answer is worth keeping. The flag is what
        makes the frequency of this contradiction reportable.
        """
        if not self.hate and (self.target_group or self.hate_type
                              or self.severity is not None):
            self._gate_normalised = True
            self.target_group = []
            self.hate_type = []
            self.severity = None
        return self

    @property
    def gate_normalised(self) -> bool:
        return self._gate_normalised


def normalise_raw(obj: dict) -> tuple[dict, int]:
    """Repair superficial format drift before validation. Returns (obj, count).

    These are presentation differences, not disagreements about the label:
    "Sexual Orientation" and "sexual_orientation" are the same answer typed
    differently. Repairing them by prompting again would spend an API call to
    fix whitespace. Every repair is counted, so the rate is reportable and a
    model that drifts constantly is visible rather than invisible.

    Deliberately NOT repaired: unknown labels. An out-of-vocabulary value is a
    genuine schema violation and must reach the repair loop.
    """
    n = 0
    out = {}

    for key, val in obj.items():
        k = str(key).strip().lower()
        canonical = FIELD_ALIASES.get(k, k)
        if canonical != key:
            n += 1
        out[canonical] = val

    def norm_label(v):
        nonlocal n
        if not isinstance(v, str):
            return v
        s = v.strip().lower().replace(" ", "_").replace("-", "_")
        if s != v:
            n += 1
        return s

    for field in ("target_group", "hate_type"):
        val = out.get(field)
        if val is None:                 # absent or explicit null -> empty list
            if field in out:
                n += 1
            out[field] = []
        elif isinstance(val, str):      # single value sent unwrapped
            out[field] = [norm_label(val)]
            n += 1
        elif isinstance(val, list):
            out[field] = [norm_label(v) for v in val]

    if isinstance(out.get("severity"), str):
        s = norm_label(out["severity"])
        # "none"/"null"/"" as a string means absent, not a label
        out["severity"] = None if s in {"none", "null", "n/a", ""} else s

    if isinstance(out.get("hate"), str):
        s = out["hate"].strip().lower()
        if s in {"true", "yes", "1"}:
            out["hate"], n = True, n + 1
        elif s in {"false", "no", "0"}:
            out["hate"], n = False, n + 1

    return out, n
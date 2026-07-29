"""Schema, normalisation and the gate-consistency rule."""

import pytest
from pydantic import ValidationError

from src.hsrag.schema import (SEVERITY_ORDER, Result, TargetGroup,
                              normalise_raw)


def test_valid_output_parses():
    r = Result(reasoning="targets women", hate=True,
               target_group=["gender"], hate_type=["inferiority"],
               severity="medium")
    assert r.hate and r.target_group == ["gender"]
    assert not r.gate_normalised


def test_not_hate_with_empty_fields_is_untouched():
    r = Result(reasoning="just rude", hate=False)
    assert r.target_group == [] and r.severity is None
    assert not r.gate_normalised


def test_gate_inconsistency_is_normalised_and_flagged():
    """The model contradicted itself. Keep the answer, clear the sub-labels,
    count it - the frequency of this is a reportable number."""
    r = Result(reasoning="contradictory", hate=False,
               target_group=["race"], severity="high")
    assert r.target_group == [] and r.hate_type == [] and r.severity is None
    assert r.gate_normalised


def test_unknown_label_raises():
    """Out-of-vocabulary is a real violation and must reach the repair loop,
    not be quietly coerced."""
    with pytest.raises(ValidationError):
        Result(reasoning="x", hate=True, target_group=["klingon"])


def test_extra_field_raises():
    with pytest.raises(ValidationError):
        Result(reasoning="x", hate=True, confidence=0.9)


@pytest.mark.parametrize("raw,expected", [
    ({"target_group": ["Sexual Orientation"]}, ["sexual_orientation"]),
    ({"target_group": "race"}, ["race"]),          # unwrapped single value
    ({"target_group": None}, []),                  # explicit null
    ({"target_groups": ["race"]}, ["race"]),       # plural alias
])
def test_normalisation_of_label_fields(raw, expected):
    out, n = normalise_raw({"reasoning": "x", "hate": True, **raw})
    assert out["target_group"] == expected
    assert n > 0


@pytest.mark.parametrize("val", ["none", "None", "null", ""])
def test_severity_none_strings_become_null(val):
    out, _ = normalise_raw({"reasoning": "x", "hate": False, "severity": val})
    assert out["severity"] is None


def test_string_booleans_are_coerced():
    out, n = normalise_raw({"reasoning": "x", "hate": "TRUE"})
    assert out["hate"] is True and n > 0


def test_clean_output_counts_zero_normalisations():
    """A well-formed answer must not inflate the normalisation rate."""
    out, n = normalise_raw({"reasoning": "x", "hate": True,
                            "target_group": ["race"], "hate_type": [],
                            "severity": "low"})
    assert n == 0


def test_normalised_output_validates():
    out, _ = normalise_raw({"reasoning": "x", "hate": "true",
                            "target_groups": ["Sexual Orientation"],
                            "severity": "None"})
    r = Result(**out)
    assert r.target_group == ["sexual_orientation"] and r.severity is None


def test_severity_order_matches_taxonomy():
    """5.5's median vote depends on this ordering being real, not assumed."""
    assert SEVERITY_ORDER == {"low": 0, "medium": 1, "high": 2}


def test_enums_come_from_taxonomy():
    assert {e.value for e in TargetGroup} == {
        "race", "religion", "gender", "sexual_orientation",
        "disability", "national_origin", "other"}
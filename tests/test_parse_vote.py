"""Extraction, repair loop, and the self-consistency vote."""

import pytest

from src.hsrag.client import MockClient
from src.hsrag.parse import RunResult, extract_json, parse_response, run_once
from src.hsrag.schema import Result
from src.hsrag.vote import aggregate

GOOD = ('{"reasoning": "r", "hate": true, "target_group": ["race"], '
        '"hate_type": [], "severity": "low"}')


# --- extraction ------------------------------------------------------------

@pytest.mark.parametrize("wrapped", [
    GOOD,
    f"```json\n{GOOD}\n```",
    f"Sure! Here is your answer:\n{GOOD}\nLet me know if you need more.",
    f"<think>hmm, is this hateful?</think>\n{GOOD}",
    f"<think>a }} brace in the trace</think>{GOOD}",
])
def test_extraction_survives_wrapping(wrapped):
    assert parse_response(wrapped)[0] is not None


def test_extraction_is_string_aware():
    """A brace inside reasoning text must not terminate the object early."""
    raw = ('{"reasoning": "they said {this} and it was bad", "hate": false, '
           '"target_group": [], "hate_type": [], "severity": null}')
    assert extract_json(raw) == raw
    assert parse_response(raw)[0] is not None


def test_unparseable_returns_error_not_exception():
    result, _, err = parse_response("{hate: true")
    assert result is None and err


# --- repair loop -----------------------------------------------------------

def test_repair_recovers_after_one_round():
    client = MockClient(broken=True)
    out = run_once(client, [{"role": "user", "text": "x"}], max_repairs=2)
    assert out.ok and out.repairs == 1 and not out.parse_failure
    assert len(out.raw_outputs) == 2      # every attempt is retained


def test_clean_output_spends_no_repairs():
    out = run_once(MockClient(), [{"role": "user", "text": "x"}], max_repairs=2)
    assert out.ok and out.repairs == 0 and len(out.raw_outputs) == 1


def test_repair_gives_up_and_is_bounded():
    """A permanently broken client must stop at max_repairs, not loop."""
    class AlwaysBroken:
        active_model = "broken"
        def complete(self, messages):
            return "not json at all"

    out = run_once(AlwaysBroken(), [{"role": "user", "text": "x"}], max_repairs=2)
    assert not out.ok and out.parse_failure
    assert out.repairs == 2 and len(out.raw_outputs) == 3   # 1 + 2 retries


# --- vote ------------------------------------------------------------------

def _run(hate, tg=(), ht=(), sev=None):
    return RunResult(result=Result(reasoning="r", hate=hate,
                                   target_group=list(tg), hate_type=list(ht),
                                   severity=sev))


def _failed():
    return RunResult(result=None, parse_failure=True)


def test_majority_gate():
    v = aggregate([_run(True), _run(True), _run(False)])
    assert v.result.hate and not v.uncertain and v.n_valid == 3


def test_multilabel_needs_strict_majority():
    """race in 2 of 3 is kept; gender in 1 of 3 is not - a single outlier run
    must not inflate recall."""
    v = aggregate([_run(True, tg=["race"]), _run(True, tg=["race", "gender"]),
                   _run(True, tg=[])])
    assert v.result.target_group == ["race"]


def test_severity_median_is_lower_on_even_pools():
    v = aggregate([_run(True, sev="low"), _run(True, sev="high")])
    assert v.result.severity == "low"


def test_severity_median_odd():
    v = aggregate([_run(True, sev="low"), _run(True, sev="high"),
                   _run(True, sev="high")])
    assert v.result.severity == "high"


def test_gate_tie_is_uncertain_and_resolves_to_not_hate():
    """Reachable when parse failures leave an even valid pool."""
    v = aggregate([_run(True, tg=["race"]), _run(False), _failed()])
    assert v.uncertain and not v.result.hate
    assert v.result.target_group == [] and v.n_valid == 2


def test_failures_shrink_the_pool_but_do_not_vote():
    v = aggregate([_run(True), _run(True), _failed()])
    assert v.result.hate and v.n_valid == 2 and v.n_runs == 3


def test_all_runs_failed_yields_no_result():
    v = aggregate([_failed(), _failed()])
    assert v.result is None and v.n_valid == 0


def test_aggregate_enforces_gate_consistency():
    """Runs can agree on a target while disagreeing about the gate; the
    aggregate must not emit hate=false with sub-labels."""
    v = aggregate([_run(True, tg=["race"]), _run(False), _run(False)])
    assert not v.result.hate and v.result.target_group == []
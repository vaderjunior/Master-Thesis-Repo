"""
Aggregate n sampled runs into one prediction.

WHY VOTE AT ALL: temperature is 1.0 (Q3), so runs genuinely differ. The vote
is the stabiliser, and consistency (Krippendorff's alpha, Phase 6) is measured
across these same raw runs. That is why every raw run is stored rather than
just the aggregate - alpha needs them, and the n=1 cost-vs-stability arm is
obtained for free by scoring only the first run.

PARSE FAILURES SHRINK THE POOL, THEY DO NOT VOTE. A run that produced no valid
output has no opinion. This can leave an even pool, which is why the gate tie
case is specified rather than assumed away.


"""

from dataclasses import dataclass, field

from src.hsrag.schema import SEVERITY_ORDER, Result


@dataclass
class VoteResult:
    result: Result | None            # None => every run failed to parse
    n_valid: int = 0
    n_runs: int = 0
    uncertain: bool = False          # gate tie
    agreement: dict = field(default_factory=dict)   # per-dimension, for logging


def _median_severity(values: list[str]) -> str | None:
    """Lower median of the ordinal severity scale.

    Ordinal, so mean is meaningless (the gap low->medium is not a number) and
    mode discards information when all runs disagree. With an even pool the
    LOWER of the two middle values is taken: severity drives no downstream
    gating, and over-stating harm is the worse error of the two. Fixed choice,
    documented, tested.

    The ordering comes from taxonomy.yaml via SEVERITY_ORDER, not a hardcoded
    map, so a reordered or extended scale cannot silently invert the median.
    """
    if not values:
        return None
    ranks = sorted(SEVERITY_ORDER[v] for v in values)
    rank = ranks[(len(ranks) - 1) // 2]
    return next(k for k, v in SEVERITY_ORDER.items() if v == rank)


def aggregate(runs: list) -> VoteResult:
    """Majority gate, >half multi-label, median severity."""
    valid = [r.result for r in runs if getattr(r, "ok", False)]
    out = VoteResult(result=None, n_valid=len(valid), n_runs=len(runs))
    if not valid:
        return out

    n = len(valid)
    yes = sum(1 for r in valid if r.hate)
    uncertain = (yes * 2 == n)          # exact tie; only possible on even pools
    hate = yes * 2 > n

    out.uncertain = uncertain
    out.agreement["hate"] = max(yes, n - yes) / n

    # Multi-label: a value is kept only if MORE than half the valid runs chose
    # it. Union would inflate recall by rewarding a single outlier run; a
    # strict majority keeps the vote a vote.
    def majority_labels(field_name: str) -> list[str]:
        counts = {}
        for r in valid:
            for v in getattr(r, field_name):
                counts[v] = counts.get(v, 0) + 1
        kept = sorted(v for v, c in counts.items() if c * 2 > n)
        out.agreement[field_name] = (
            min(c / n for c in counts.values()) if counts else 1.0)
        return kept

    target_group = majority_labels("target_group")
    hate_type = majority_labels("hate_type")
    legal = majority_labels("legal")

    sevs = [r.severity for r in valid if r.severity is not None]
    severity = _median_severity(sevs)
    out.agreement["severity"] = len(sevs) / n if sevs else 1.0

    # A tie resolves to not-hate: the conservative direction for a
    # false-positive-sensitive task, and it is counted separately so the
    # resolution never hides inside the F1.
    if uncertain:
        hate = False

    # Gate consistency applies to the AGGREGATE too. Individual runs are
    # normalised at validation, but a vote can still produce hate=false with
    # sub-labels when runs disagree about the gate while agreeing about the
    # target.
    #
    # `legal` is deliberately excluded, as in Result.gate_consistency:
    # criminal relevance is independent of the hate gate. A section 185 insult
    # aimed at one private individual is criminally relevant and not hate
    # speech, and clearing it here would erase a legitimate answer.
    if not hate:
        target_group, hate_type, severity = [], [], None

    # Reasoning from the first valid run that AGREES with the vote, not simply
    # run 0. When runs disagree on the gate, run 0's rationale can argue the
    # opposite of the voted label - observed in mhs-41979, a "not hate" vote
    # carrying a rationale arguing for hate. Harmless for scoring, which never
    # reads it, but actively misleading as a thesis example. Falls back to
    # run 0 when no valid run matches, which is possible when a tie forced the
    # resolution to not-hate.
    agreeing = [r for r in valid if r.hate == hate]
    reasoning = (agreeing[0] if agreeing else valid[0]).reasoning

    out.result = Result(
        reasoning=reasoning,
        hate=hate, target_group=target_group,
        hate_type=hate_type, severity=severity, legal=legal,
    )

    # DIAGNOSTIC, not an assertion. A dimension where every valid run said
    # something and the aggregate says nothing usually means this function was
    # not updated when the taxonomy grew - legal was dropped for 974 of 1350
    # runs that way, and the symptom was a plausible-looking zero rather than
    # a crash. A legitimate majority vote can also empty a field, so this is
    # recorded rather than raised.
    out.agreement["_dropped"] = [
        f for f in ("target_group", "hate_type", "legal")
        if any(getattr(r, f) for r in valid) and not getattr(out.result, f)]

    return out
"""
Scoring. Everything Phase 5 learned about how these labels have to be handled,
in one place.

WHY A MODULE AND NOT INLINE FUNCTIONS: Slice 1 scored itself inside the run
script. That is fine for one experiment and wrong for a dozen, because each
one would re-implement the same subtleties and get them subtly differently.

EVERY FUNCTION RETURNS NUMBERS PLUS SUPPORT, NEVER A BARE FLOAT. A macro-F1
that does not say which labels it averaged is not interpretable, and Phase 5
produced several that would have been misread without their support counts.

FOUR RULES THIS MODULE ENFORCES:

  1. MASKING IS DATA-DRIVEN (decision Q5). A dimension is scorable for an item
     if and only if that item's gold is not None. None means the source
     dataset never annotated the dimension; [] means it did and the answer is
     empty. That distinction was preserved from Phase 2 through parquet,
     Chroma and the prompt specifically so that scoring could rely on it here.
     German masks itself as a consequence, with no special-casing anywhere.

  2. MIN_SUPPORT. A label with fewer than 10 gold positives is reported with
     its support and excluded from the macro average. A per-label F1 computed
     from two items is noise wearing a number's clothes.

  3. THE GATE IS SCORED OVER BOTH CLASSES. The task is false-positive
     sensitive, and positive-class F1 alone is blind to over-flagging of
     benign text - which Slice 1 showed is this system's dominant error mode
     at roughly 5 false positives per false negative.

  4. PER-SOURCE FALSE-POSITIVE RATE IS A STANDARD OUTPUT. Slice 1 found the FP
     rate varying from 10% (implicit_hate) to 61% (hatexplain). That
     decomposition is what separates a model prior from an annotation-mapping
     artefact, so it is a column in every gate table rather than an occasional
     ad-hoc analysis.
"""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import pandas as pd

from src.hsrag.schema import SEVERITY_ORDER

MIN_SUPPORT = 10
ARMS = ("zero_shot", "few_shot", "rag")


# --------------------------------------------------------------- utilities

def clean(val):
    """Parquet round-trip: NaN -> None, numpy array -> list.

    Every read of a gold label column goes through here, because the
    None / [] / ["race"] distinction is what rule 1 above depends on and
    parquet does not preserve it natively.
    """
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def hx_original_class(raw_json: str) -> str | None:
    """HateXplain's own three-way label: normal | offensive | hatespeech.

    Phase 2 mapped offensive -> gate=False on a strict reading and kept the
    annotator votes in Record.raw so the decision stayed reversible. This is
    where that reversibility is spent.
    """
    try:
        votes = json.loads(raw_json).get("votes") or []
    except (TypeError, ValueError, AttributeError):
        return None
    return Counter(votes).most_common(1)[0][0] if votes else None


def effective_gate(row, mapping: str = "strict") -> bool:
    """Gold gate under the chosen reading of 'offensive' (decision Q2).

    strict  - offensive is NOT hate (the Phase 2 mapping, current default)
    lenient - offensive IS hate

    Only HateXplain has this three-way distinction; MHS thresholds a
    continuous score and Implicit Hate is binary. So the two readings differ
    on roughly 9 of 150 items in en_dev_eval_main, and the delta between them
    must not be read as if it applied to the whole subset.

    SCOPE: this affects GATE scoring only. Sub-dimensions always use the
    strict gold gate to decide which items are hateful, so that a sensitivity
    analysis about one annotation decision cannot leak into unrelated
    dimensions.
    """
    # None means the dataset never annotated the gate - BoTox annotates only
    # criminal relevance - and must not collapse to False. bool(None) is
    # False, which would score 150 unannotated items as benign and report
    # every hate prediction on them as a false positive.
    if row.gate is None:
        return None
    if mapping == "strict" or row.source != "hatexplain":
        return bool(row.gate)
    return True if hx_original_class(row.raw) == "offensive" else bool(row.gate)


# ------------------------------------------------------------------ scores

@dataclass
class LabelScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int          # gold positives
    predicted: int        # predicted positives
    tp: int
    fp: int
    fn: int
    averaged: bool        # did it clear MIN_SUPPORT


@dataclass
class DimensionScore:
    dimension: str
    macro_f1: float | None            # None when no label clears support
    micro_f1: float
    per_label: dict = field(default_factory=dict)
    labels_averaged: list = field(default_factory=list)
    labels_excluded: list = field(default_factory=list)
    n_items: int = 0                  # scorable items (gold not None, hateful)
    n_unscored: int = 0               # items with no valid prediction
    subset_accuracy: float = 0.0      # exact set match
    hamming_loss: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class GateScore:
    macro_f1: float
    mapping: str
    per_class: dict = field(default_factory=dict)     # "hate"/"not_hate" -> prf
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    per_source: dict = field(default_factory=dict)
    n_scored: int = 0
    n_unscored: int = 0
    n_uncertain: int = 0
    macro_f1_confident_only: float | None = None      # uncertain items dropped
    note: str | None = None


def score_gate(rows: list, gold: dict, mapping: str = "strict") -> GateScore:
    """Binary gate, macro-F1 over both classes, decomposed by source dataset.

    Items with no valid prediction (every run failed to parse) are counted and
    EXCLUDED rather than scored as not-hate, which would reward the system for
    failing on hateful text.

    Uncertain items (a gate tie among the valid runs) resolve to not-hate for
    the primary number - conservative, documented - and are additionally
    reported on their own line and excluded from a confident-only variant.
    Zero have ever occurred, so this path is exercised by fixtures rather than
    by waiting for it to fire.
    """
    out = GateScore(macro_f1=0.0, mapping=mapping)
    scored, confident = [], []

    for r in rows:
        g = gold.get(r["item_id"])
        if g is None:
            continue
        if r["result"] is None:
            out.n_unscored += 1
            continue
        truth = effective_gate(g, mapping)
        if truth is None:      # dimension not annotated by this dataset
            continue
        pred = bool(r["result"]["hate"])
        scored.append((truth, pred, g.source))
        if r.get("uncertain"):
            out.n_uncertain += 1
        else:
            confident.append((truth, pred))

    out.n_scored = len(scored)
    if not scored:
        return out

    for name, cls in (("hate", True), ("not_hate", False)):
        tp = sum(1 for t, p, _ in scored if t == cls and p == cls)
        fp = sum(1 for t, p, _ in scored if t != cls and p == cls)
        fn = sum(1 for t, p, _ in scored if t == cls and p != cls)
        out.per_class[name] = dict(zip(("precision", "recall", "f1"),
                                       _prf(tp, fp, fn)))
        out.per_class[name].update(tp=tp, fp=fp, fn=fn)

    out.macro_f1 = sum(c["f1"] for c in out.per_class.values()) / 2
    # A subset can be single-class by construction: the label-stratified
    # dimension subsets are 100% gold-hateful, because fine-grained labels
    # only exist on hateful items. Macro-F1 over two classes is undefined
    # there - the absent class has no true positives available - and
    # averaging its forced 0.0 would halve the number for a reason unrelated
    # to performance.
    present = {t for t, _, _ in scored}
    if len(present) < 2:
        only = present.pop()
        out.macro_f1 = None
        out.note = (f"single-class subset: every gold item has gate={only}. "
                    f"Macro-F1 over both classes is undefined; recall on the "
                    f"present class is "
                    f"{out.per_class['hate' if only else 'not_hate']['recall']:.3f}.")
    out.tp = sum(1 for t, p, _ in scored if t and p)
    out.fp = sum(1 for t, p, _ in scored if not t and p)
    out.fn = sum(1 for t, p, _ in scored if t and not p)
    out.tn = sum(1 for t, p, _ in scored if not t and not p)

    for src in sorted({s for _, _, s in scored}):
        sub = [(t, p) for t, p, s in scored if s == src]
        benign = sum(1 for t, _ in sub if not t)
        hateful = sum(1 for t, _ in sub if t)
        fp = sum(1 for t, p in sub if not t and p)
        fn = sum(1 for t, p in sub if t and not p)
        out.per_source[src] = {
            "n": len(sub), "benign": benign, "hateful": hateful,
            "fp": fp, "fn": fn,
            "fp_rate": fp / benign if benign else None,
            "fn_rate": fn / hateful if hateful else None,
        }

    if confident and len(confident) < len(scored):
        f1s = []
        for cls in (True, False):
            tp = sum(1 for t, p in confident if t == cls and p == cls)
            fp = sum(1 for t, p in confident if t != cls and p == cls)
            fn = sum(1 for t, p in confident if t == cls and p != cls)
            f1s.append(_prf(tp, fp, fn)[2])
        out.macro_f1_confident_only = sum(f1s) / 2

    return out


def score_multilabel(rows: list, gold: dict, gold_col: str, pred_field: str,
                     min_support: int = MIN_SUPPORT,
                     fixed_labels: list | None = None) -> DimensionScore:
    """Multilabel dimension, scored on items whose gold for it is not None.

    GOLD-HATEFUL ITEMS ONLY, EXCEPT FOR `legal`. Including non-hateful items
    would fill every label with true negatives and drive F1 toward 1.0 without
    measuring anything, so the number reads as: given the item is hateful,
    does the system identify the right labels. `legal` is exempt because
    criminal relevance is independent of the hate gate and BoTox does not
    annotate the gate at all.

    THE LABEL SET IS THE UNION of gold and predicted labels. Scoring only
    gold-observed labels would make a spurious prediction of a label absent
    from this subset free. Union costs nothing: a predicted-only label has
    support 0, so MIN_SUPPORT excludes it from the macro average while it
    still appears in the per-label table and counts against micro-F1.

    fixed_labels IS FOR BOOTSTRAPPING. Recomputing the MIN_SUPPORT filter on
    every resample admits a different label set each draw, so the quantity
    being bootstrapped changes definition between draws and the resulting CI
    is meaningless. Callers that resample pass the label set computed once
    from the full sample. A fixed label absent from a given resample is
    SKIPPED, not scored 0: absent-from-gold makes F1 undefined rather than
    zero, and scoring it 0 would drag the macro average down for a label the
    resample never had a chance to get right.
    """
    out = DimensionScore(dimension=pred_field, macro_f1=None, micro_f1=0.0)
    pairs = []

    for r in rows:
        g = gold.get(r["item_id"])
        if g is None:
            continue
        # Fine-grained dimensions are scored on gold-hateful items only:
        # including benign ones fills every label with true negatives and
        # drives F1 toward 1.0 without measuring anything.
        #
        # `legal` is the exception. BoTox never annotates the gate - criminal
        # relevance is not a hate-speech gate - so gating on it would skip
        # every item and score nothing. Its own [] (class 0, annotated and not
        # criminally relevant) versus ["insult_defamation"] distinction
        # already carries that information.
        if pred_field != "legal" and not bool(g.gate):
            continue
        # getattr with a default, not a bare getattr: splits created before a
        # dimension existed have no column for it at all. The English parquets
        # predate `legal`, and a missing column means the same thing as a None
        # value - this dataset never annotated the dimension - so it is
        # skipped rather than raising.
        truth = clean(getattr(g, gold_col, None))
        if truth is None:
            continue
        if r["result"] is None:
            out.n_unscored += 1
            continue
        pairs.append((set(truth), set(r["result"][pred_field])))

    out.n_items = len(pairs)
    if not pairs:
        return out

    labels = sorted({l for t, _ in pairs for l in t} |
                    {l for _, p in pairs for l in p})
    tot_tp = tot_fp = tot_fn = 0

    for label in labels:
        tp = sum(1 for t, p in pairs if label in t and label in p)
        fp = sum(1 for t, p in pairs if label not in t and label in p)
        fn = sum(1 for t, p in pairs if label in t and label not in p)
        support = sum(1 for t, _ in pairs if label in t)
        pr, rc, f1 = _prf(tp, fp, fn)
        out.per_label[label] = asdict(LabelScore(
            label=label, precision=pr, recall=rc, f1=f1, support=support,
            predicted=tp + fp, tp=tp, fp=fp, fn=fn,
            averaged=support >= min_support))
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn

    if fixed_labels is None:
        out.labels_averaged = [l for l in labels
                               if out.per_label[l]["averaged"]]
    else:
        out.labels_averaged = [l for l in fixed_labels if l in out.per_label]
    out.labels_excluded = [l for l in labels if l not in out.labels_averaged]

    if out.labels_averaged:
        out.macro_f1 = (sum(out.per_label[l]["f1"] for l in out.labels_averaged)
                        / len(out.labels_averaged))
    out.micro_f1 = _prf(tot_tp, tot_fp, tot_fn)[2]

    out.subset_accuracy = sum(1 for t, p in pairs if t == p) / len(pairs)
    out.hamming_loss = (sum(len(t ^ p) for t, p in pairs)
                        / (len(pairs) * max(len(labels), 1)))

    # Context that makes the primary number readable. Slice 1 found gold
    # hate_type is exactly 1.00 labels per item (Implicit Hate assigns one
    # type per post) while the system predicts ~2.1, so multilabel F1 is
    # partly penalising a data property. The 'gold label is somewhere in the
    # predicted set' figure is the ceiling that over-prediction buys, and the
    # gap between the two IS the cost of over-predicting.
    out.extra = {
        "mean_gold_labels": sum(len(t) for t, _ in pairs) / len(pairs),
        "mean_pred_labels": sum(len(p) for _, p in pairs) / len(pairs),
        "any_overlap_rate": sum(1 for t, p in pairs if t & p) / len(pairs),
        "gold_subset_of_pred": sum(1 for t, p in pairs if t <= p) / len(pairs),
    }
    return out


def score_severity(rows: list, gold: dict,
                   min_support: int = MIN_SUPPORT) -> DimensionScore:
    """Severity: single-valued and ORDINAL, so it is scored as multiclass with
    ordinal extras rather than shoehorned into the multilabel path.

    Exact-match accuracy alone would treat predicting 'high' for a 'low' item
    as identical to predicting 'medium'. Mean absolute rank error and
    within-one accuracy say how far wrong a wrong answer is, which is the
    thing an ordinal scale is for.
    """
    out = DimensionScore(dimension="severity", macro_f1=None, micro_f1=0.0)
    pairs = []

    for r in rows:
        g = gold.get(r["item_id"])
        if g is None or not bool(g.gate):
            continue
        truth = clean(g.severity)
        if truth is None:
            continue
        if r["result"] is None:
            out.n_unscored += 1
            continue
        pairs.append((truth, r["result"]["severity"]))

    out.n_items = len(pairs)
    if not pairs:
        return out

    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs if p},
                    key=lambda l: SEVERITY_ORDER.get(l, 99))
    for label in labels:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        support = sum(1 for t, _ in pairs if t == label)
        pr, rc, f1 = _prf(tp, fp, fn)
        out.per_label[label] = asdict(LabelScore(
            label=label, precision=pr, recall=rc, f1=f1, support=support,
            predicted=tp + fp, tp=tp, fp=fp, fn=fn,
            averaged=support >= min_support))

    out.labels_averaged = [l for l in labels if out.per_label[l]["averaged"]]
    out.labels_excluded = [l for l in labels if not out.per_label[l]["averaged"]]
    if out.labels_averaged:
        out.macro_f1 = (sum(out.per_label[l]["f1"] for l in out.labels_averaged)
                        / len(out.labels_averaged))

    exact = sum(1 for t, p in pairs if t == p)
    out.micro_f1 = exact / len(pairs)          # multiclass micro-F1 == accuracy
    out.subset_accuracy = exact / len(pairs)

    ranked = [(SEVERITY_ORDER[t], SEVERITY_ORDER[p])
              for t, p in pairs if p in SEVERITY_ORDER]
    out.extra = {
        "n_predicted_null": sum(1 for _, p in pairs if p is None),
        "mean_abs_rank_error": (sum(abs(a - b) for a, b in ranked) / len(ranked)
                                if ranked else None),
        "within_one_accuracy": (sum(1 for a, b in ranked if abs(a - b) <= 1)
                                / len(ranked) if ranked else None),
    }
    return out


def calibration(rows: list, gold: dict, mapping: str = "strict") -> dict:
    """Expected calibration error on the gate, using self-consistency vote
    share as the confidence proxy.

    STATED AS A PROXY, NOT A PROBABILITY. This API exposes no logprobs, so
    confidence is 'how many of the valid runs agreed'. At n_votes=3 that can
    only be 2/3 or 1 (or 1/2 on a tie), so the reliability curve has two or
    three points and should be read as a coarse check, not a smooth curve.
    """
    buckets = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        g = gold.get(r["item_id"])
        if g is None or r["result"] is None:
            continue
        # The gate is not annotated on every dataset (BoTox annotates only
        # criminal relevance), and calibration against a None truth value
        # scores every prediction wrong.
        if effective_gate(g, mapping) is None:
            continue
        conf = r.get("agreement", {}).get("hate")
        if conf is None:
            continue
        b = buckets[round(float(conf), 3)]
        b["n"] += 1
        b["correct"] += int(bool(r["result"]["hate"]) == effective_gate(g, mapping))

    total = sum(b["n"] for b in buckets.values())
    if not total:
        return {"ece": None, "bins": {}, "n": 0}

    ece = 0.0
    bins = {}
    for conf, b in sorted(buckets.items()):
        acc = b["correct"] / b["n"]
        ece += b["n"] / total * abs(acc - conf)
        bins[conf] = {"n": b["n"], "accuracy": acc, "confidence": conf}
    return {"ece": ece, "bins": bins, "n": total,
            "note": "vote share proxy; n_votes=3 gives 2-3 distinct values"}


def honesty(rows: list) -> dict:
    """The numbers that say how much to trust everything above."""
    runs = sum(r["n_runs"] for r in rows)
    lat = [x for r in rows for x in r["latencies"]]
    return {
        "n_items": len(rows),
        "total_runs": runs,
        "parse_failures": sum(r["parse_failures"] for r in rows),
        "parse_failure_rate": sum(r["parse_failures"] for r in rows) / max(runs, 1),
        "repairs": sum(r["repairs"] for r in rows),
        "normalisations": sum(r["normalisations"] for r in rows),
        "gate_normalised": sum(r["gate_normalised"] for r in rows),
        "uncertain": sum(1 for r in rows if r["uncertain"]),
        "no_prediction": sum(1 for r in rows if r["result"] is None),
        "mean_latency_s": sum(lat) / max(len(lat), 1),
        "workers": sorted({r.get("workers", 1) for r in rows}),
        "models": dict(Counter(r["active_model"] for r in rows)),
        "kb_versions": sorted({r["kb_version"] for r in rows if r["kb_version"]}),
        "prompt_versions": sorted({r["prompt_version"] for r in rows}),
    }


# ------------------------------------------------------------- entry point

def score_all(df: pd.DataFrame, results: list, gate_mapping: str = "both",
              min_support: int = MIN_SUPPORT) -> dict:
    """Score every arm present in `results` against gold in `df`."""
    gold = {str(r.id): r for r in df.itertuples(index=False)}
    by_arm = defaultdict(list)
    for r in results:
        if "_manifest" in r:               # results-file header line
            continue
        by_arm[r["arm"]].append(r)

    mappings = (["strict", "lenient"] if gate_mapping == "both"
                else [gate_mapping])

    out = {}
    for arm, rows in by_arm.items():
        out[arm] = {
            "gate": {m: asdict(score_gate(rows, gold, m)) for m in mappings},
            "target_group": asdict(score_multilabel(
                rows, gold, "target_groups", "target_group", min_support)),
            "hate_type": asdict(score_multilabel(
                rows, gold, "hate_types", "hate_type", min_support)),
            "legal": asdict(score_multilabel(
                rows, gold, "legal", "legal", min_support)),
            "severity": asdict(score_severity(rows, gold, min_support)),
            "calibration": calibration(rows, gold, mappings[0]),
            "honesty": honesty(rows),
        }
    return out
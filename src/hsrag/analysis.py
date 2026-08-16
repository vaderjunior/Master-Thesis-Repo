"""
analysis.py 
Consistency, significance, and the guideline instrument.

Everything here runs on stored ItemResults. No API calls, ever - which is why
all of it could be built and validated before spending anything.

WHY SIGNIFICANCE IS NOT OPTIONAL HERE: Slice 1's three arms agreed on 135 of
150 gate decisions. The disagreement set IS the signal, and comparing two
macro-F1 numbers computed over 150 items says nothing about whether a 0.02
gap is real. Paired tests use the fact that every arm saw identical items.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.hsrag.schema import SEVERITY_ORDER

DECISIVE_PATH = Path("config/decisive_guidelines.yaml")


# ============================================================ 6.4 consistency

def _alpha(units: list[list], ordinal: bool = False) -> dict:
    """Krippendorff's alpha over a list of units, each a list of coder values.

    UNIT OF ANALYSIS (the thing to state in the thesis):
      gate      - one unit per item, values True/False, nominal
      multilabel- one unit per (item, label) pair, values present/absent,
                  nominal. Alpha wants one value per unit per coder; a SET is
                  not a value, so the set is decomposed into independent
                  binary decisions. This is the cleanest reading and it makes
                  the number comparable across dimensions.
      severity  - one unit per item, values low/medium/high, ORDINAL, so that
                  disagreeing low-vs-high counts more than low-vs-medium.

    Units with fewer than two valid values contribute nothing (one coder
    cannot disagree with itself). Parse failures shrink units rather than
    voting, exactly as in the self-consistency vote.

    Returns alpha=None when every coder gave the same value everywhere: D_e is
    then 0 and alpha is 0/0, undefined. Reporting 0.0 there would suggest
    total disagreement when the truth is total agreement.
    """
    usable = [[v for v in u if v is not None] for u in units]
    usable = [u for u in usable if len(u) >= 2]
    if not usable:
        return {"alpha": None, "n_units": 0, "note": "no unit had 2+ values"}

    values = sorted({v for u in usable for v in u},
                    key=lambda v: SEVERITY_ORDER.get(v, v) if ordinal else str(v))
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)

    # coincidence matrix
    o = np.zeros((k, k))
    for u in usable:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[u[a]], idx[u[b]]] += 1.0 / (m - 1)

    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return {"alpha": None, "n_units": len(usable), "note": "too few values"}

    if ordinal:
        # ordinal difference: squared distance through the marginal mass
        # between the two categories, so low-vs-high > low-vs-medium
        delta = np.zeros((k, k))
        for c in range(k):
            for kk in range(k):
                lo, hi = min(c, kk), max(c, kk)
                s = n_c[lo:hi + 1].sum() - (n_c[c] + n_c[kk]) / 2
                delta[c, kk] = s ** 2
    else:
        delta = 1.0 - np.eye(k)

    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))

    if d_e == 0:
        return {"alpha": None, "n_units": len(usable), "n_values": int(n),
                "note": "perfect agreement (expected disagreement is zero, "
                        "so alpha is undefined)"}
    return {"alpha": 1.0 - d_o / d_e, "n_units": len(usable),
            "n_values": int(n), "categories": [str(v) for v in values]}


def consistency(rows: list, taxonomy_labels: dict) -> dict:
    """Krippendorff's alpha per dimension, across the stored raw runs.

    This is the 'consistency' half of the thesis's two anchoring definitions
    (adaptability = delta macro-F1 after KB edits; consistency = inter-run
    agreement plus guideline alignment). It costs nothing because Phase 5
    stored every run rather than only the vote.
    """
    out = {}

    out["gate"] = _alpha([[p["hate"] if p else None for p in r["run_predictions"]]
                          for r in rows])

    for field, labels in (("target_group", taxonomy_labels["target_group"]),
                          ("hate_type", taxonomy_labels["hate_type"]),
                          ("legal", taxonomy_labels["legal"])):
        units = []
        for r in rows:
            for label in labels:
                units.append([(label in p[field]) if p else None
                              for p in r["run_predictions"]])
        out[field] = _alpha(units)
        out[field]["unit"] = "(item, label) binary decision"

    out["severity"] = _alpha(
        [[p["severity"] if p and p["severity"] else None
          for p in r["run_predictions"]] for r in rows], ordinal=True)
    out["severity"]["unit"] = "item, ordinal"
    return out


def single_run_view(rows: list, run_index: int = 0) -> list:
    """Re-cast results as if n_votes had been 1 (decision Q3).

    The cost-versus-stability comparison comes free because every raw run was
    stored. If n=1 scores nearly as well as n=3, the vote is buying stability
    at 3x the API cost and that trade is worth stating explicitly.
    """
    out = []
    for r in rows:
        copy = dict(r)
        preds = r["run_predictions"]
        copy["result"] = preds[run_index] if run_index < len(preds) else None
        copy["uncertain"] = False          # a single run cannot tie
        copy["n_runs"] = 1
        copy["n_valid"] = int(copy["result"] is not None)
        out.append(copy)
    return out


# =========================================================== 6.5 significance

def mcnemar(rows_a: list, rows_b: list, gold: dict, effective_gate,
            mapping: str = "strict") -> dict:
    """Exact McNemar on paired gate correctness.

    The arms agreed on 90% of Slice 1's gate decisions, so the concordant
    pairs carry no information about which arm is better - only the items
    where exactly one arm was right do. McNemar tests precisely those, which
    is why it is the right instrument and a raw F1 comparison is not.

    Discordant counts are reported alongside p, because with b + c small no
    test can find anything and that fact is more informative than the p-value.
    """
    from scipy.stats import binomtest

    a = {r["item_id"]: r for r in rows_a}
    b = {r["item_id"]: r for r in rows_b}
    n00 = n01 = n10 = n11 = 0

    for item_id in set(a) & set(b):
        g = gold.get(item_id)
        if g is None or a[item_id]["result"] is None or b[item_id]["result"] is None:
            continue
        truth = effective_gate(g, mapping)
        ok_a = bool(a[item_id]["result"]["hate"]) == truth
        ok_b = bool(b[item_id]["result"]["hate"]) == truth
        if ok_a and ok_b:
            n11 += 1
        elif ok_a and not ok_b:
            n10 += 1
        elif not ok_a and ok_b:
            n01 += 1
        else:
            n00 += 1

    disc = n10 + n01
    p = binomtest(n10, disc, 0.5).pvalue if disc else 1.0
    return {"both_correct": n11, "both_wrong": n00,
            "only_a_correct": n10, "only_b_correct": n01,
            "discordant": disc, "p_value": p,
            "note": ("no discordant pairs; the arms are indistinguishable "
                     "on this data" if not disc else
                     "exact binomial on discordant pairs only")}


def paired_bootstrap(rows_a: list, rows_b: list, gold: dict, score_fn,
                     b: int = 1000, seed: int = 42) -> dict:
    """Bootstrap CI for a metric delta, resampling ITEMS.

    RESAMPLE ITEMS FIRST, THEN SCORE. Scoring a masked dimension and then
    resampling the survivors would hold the effective n fixed across
    resamples and understate the variance that comes from which items are
    scorable at all.

    Both arms are scored on the SAME resampled items every iteration; that
    pairing is the entire point, and it only works because every arm ran on
    identical items.
    """
    a = {r["item_id"]: r for r in rows_a}
    bb = {r["item_id"]: r for r in rows_b}
    ids = sorted(set(a) & set(bb) & set(gold))
    if not ids:
        return {"delta": None, "note": "no shared items"}

    obs_a = score_fn([a[i] for i in ids], gold)
    obs_b = score_fn([bb[i] for i in ids], gold)
    if obs_a is None or obs_b is None:
        return {"delta": None, "note": "metric undefined on the full sample"}

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(b):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        sample = [ids[i] for i in pick]
        sa = score_fn([a[i] for i in sample], gold)
        sb = score_fn([bb[i] for i in sample], gold)
        if sa is not None and sb is not None:
            deltas.append(sa - sb)

    if not deltas:
        return {"delta": obs_a - obs_b, "note": "metric undefined in resamples"}

    d = np.array(deltas)
    n_a = int((d > 0).sum())
    n_tie = int((d == 0).sum())
    n_b = int((d < 0).sum())

    # Two-sided bootstrap p with a mid-p tie adjustment. Ties at exactly zero
    # are common when few items are scorable - both arms score identically on
    # the drawn items - and without the adjustment that mass counts on BOTH
    # sides and forces p to 1.0 whatever the observed effect.
    frac_le = (n_b + 0.5 * n_tie) / len(d)
    frac_ge = (n_a + 0.5 * n_tie) / len(d)
    p = min(1.0, 2 * min(frac_le, frac_ge))

    return {"score_a": obs_a, "score_b": obs_b, "delta": obs_a - obs_b,
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            # SD of the resampled delta distribution. Added for the Phase 7
            # SQ2 reconciliation: comparing the 103-item and 467-item
            # estimates needs a standard error, and reconstructing one from
            # the percentile CI is lossy when the distribution is skewed.
            # Additive only - no existing caller reads this key.
            "sd": float(d.std(ddof=1)),
            "p_value": float(p), "n_items": len(ids),
            "n_resamples": len(deltas),
            "favour_a": n_a, "ties": n_tie, "favour_b": n_b}


# ============================================== 6.6 the guideline instrument

def guideline_effect(rag_rows: list, zero_rows: list, gold: dict,
                     effective_gate, mapping: str = "strict") -> dict:
    """Does putting a decisive guideline in the prompt change the answer?

    NOT AN ADHERENCE RATE. Retrieval fetches guidelines by resemblance, not
    applicability: guide-threats-and-incitement was retrieved on 41 Slice 1
    items and the gold label contradicted its implication on 27. Scoring those
    as violations would count correct predictions as failures.

    Instead, for each guideline G:
      - gate accuracy on items where G was retrieved, SPLIT BY GOLD LABEL, so
        the retrieval-noise population is visible rather than averaged in
      - the same accuracy for zero_shot on those SAME items. zero_shot never
        saw G, so the difference is the causal effect of placing G in the
        prompt, paired and free.

    A guideline that helps shows a positive delta on the gold side it argues
    for. A guideline that is retrieved and ignored shows roughly zero. A
    guideline that is retrieved when it does not apply shows a negative delta
    on the opposite side.
    """
    decisive = yaml.safe_load(DECISIVE_PATH.read_text(encoding="utf-8"))["decisive"]
    zero = {r["item_id"]: r for r in zero_rows}
    out = {}

    for guide, implies in decisive.items():
        cells = {}
        for side, want in (("gold_hate", True), ("gold_not_hate", False)):
            sub = [r for r in rag_rows
                   if guide in r["retrieved"].get("guidelines", [])
                   and r["item_id"] in gold
                   and effective_gate(gold[r["item_id"]], mapping) is want]
            scored = [r for r in sub if r["result"] is not None]
            rag_ok = sum(1 for r in scored
                         if bool(r["result"]["hate"]) is want)

            paired = [r for r in scored if r["item_id"] in zero
                      and zero[r["item_id"]]["result"] is not None]
            zs_ok = sum(1 for r in paired
                        if bool(zero[r["item_id"]]["result"]["hate"]) is want)

            cells[side] = {
                "n": len(scored),
                "rag_correct": rag_ok,
                "rag_accuracy": rag_ok / len(scored) if scored else None,
                "n_paired": len(paired),
                "zero_shot_correct": zs_ok,
                "zero_shot_accuracy": zs_ok / len(paired) if paired else None,
                "delta": ((rag_ok / len(scored)) - (zs_ok / len(paired)))
                if scored and paired else None,
            }

        total = sum(c["n"] for c in cells.values())
        applies = cells["gold_hate" if implies else "gold_not_hate"]["n"]
        out[guide] = {
            "implies_hate": implies,
            "retrieved_on": total,
            "gold_agrees_with_implication": applies,
            "gold_contradicts": total - applies,
            "contradiction_rate": (total - applies) / total if total else None,
            **cells,
        }
    return out
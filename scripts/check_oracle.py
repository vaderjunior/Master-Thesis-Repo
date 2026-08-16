"""
scripts/check_oracle.py - paired analysis of the oracle injection.
Read-only, zero API calls.
Run: python -m scripts.check_oracle

WHY PAIRED AND NOT MACRO-F1. Conditions A and B ran on the identical 121
held-out items with the identical KB, prompt, model, temperature and vote
count; only the contents of the single feedback slot differ. Comparing two
macro-F1 numbers over 121 items discards that pairing and is blunt: the
held-out macro floor is 0.049, so a real effect can sit under it. The items
where exactly one condition is correct are the entire signal, which is the
same reasoning analysis.mcnemar applies to the gate.

WHAT THE CONDITIONS ARE.
  A - a correction whose gold hate_type intersects the item's own gold label,
      chosen USING that gold label. An oracle: no deployed system could do
      this. It is an upper bound on what perfect correction selection buys.
  B - a correction whose labels do not intersect the item's gold, matched in
      count and position. Without B, A confounds "a relevant correction" with
      "a correction at all".

THE MIXTURE CHECK. If the model copies the label of whatever correction it is
shown, then real performance should be a mixture of A and B weighted by how
often retrieval actually delivers a relevant one:

    predicted = B + p * (A - B)

where p is the measured label-relevance of retrieved corrections. That rate
was measured independently by check_sq3_reachability on the same 121 items and
is transcribed below. If the prediction tracks the observed round-by-round
figures, the SQ3 null is not "nothing happens" - it is help and harm
cancelling at the achieved hit rate, which is a quantitative account rather
than an absence.

VERIFY THE INJECTION FIRST. Every claim here rests on A actually having been
label-matched and B not. That is checkable from the stored `injected` id and
is asserted rather than assumed.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from scripts.apply_feedback import load_results
from scripts.check_sq3_coverage import PROCESSED, local_gold_sets

SUBSET = "en_dev_eval_sq3_types"
GOLD_COL, DIM = "hate_types", "hate_type"
RECORDS = Path("kb") / "records_sq3_fb_r4.jsonl"
# Transcribed from check_sq3_reachability on the same 121 held-out items,
# at correction-pool sizes 20 / 33 / 48 / 60.
LABEL_HIT = {"sq3_r1_fb": 0.149, "sq3_r2_fb": 0.198,
             "sq3_r3_fb": 0.215, "sq3_r4_fb": 0.273}
# Four round-0 replicates on held_out gave macro 0.386/0.338/0.363/0.350 and
# contains 0.488/0.438/0.463/0.463.
FLOOR_CONTAINS = 0.050


def correct(stem: str, gold: dict) -> dict:
    res = load_results(stem)
    out = {}
    for i in gold:
        r = res[i].get("result")
        out[i] = float(gold[i] <= set((r or {}).get(DIM) or []))
    return out


def paired(a: dict, b: dict, name_a: str, name_b: str) -> None:
    ids = sorted(set(a) & set(b))
    only_a = sum(1 for i in ids if a[i] and not b[i])
    only_b = sum(1 for i in ids if b[i] and not a[i])
    both = sum(1 for i in ids if a[i] and b[i])
    neither = len(ids) - only_a - only_b - both
    disc = only_a + only_b
    p = binomtest(only_a, disc, 0.5).pvalue if disc else 1.0
    print(f"  {name_a} vs {name_b}")
    print(f"    both {both:4}   neither {neither:4}   "
          f"only {name_a} {only_a:3}   only {name_b} {only_b:3}")
    print(f"    discordant {disc:3}   exact binomial p = {p:.4f}   "
          f"rate {a and sum(a[i] for i in ids) / len(ids):.3f} vs "
          f"{sum(b[i] for i in ids) / len(ids):.3f}\n")


def main():
    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    items = df[df["sq3_role"] == "held_out"].reset_index(drop=True)
    gold = local_gold_sets(items, GOLD_COL)

    labels = {r["id"]: set(r["meta"].get("hate_types") or []) for r in
              (json.loads(l) for l in
               RECORDS.read_text(encoding="utf-8").splitlines() if l.strip())
              if r["kind"] == "feedback"}

    print(f"\n{'=' * 74}\nORACLE INJECTION, {len(gold)} held-out items")

    # --------------------------------------------- verify the manipulation
    for cond, want in (("A", True), ("B", False)):
        res = load_results(f"oracle_{cond}")
        bad = []
        for i, g in gold.items():
            inj = res[i]["retrieved"].get("injected") or []
            assert len(inj) == 1, f"{cond}/{i}: {len(inj)} injected records"
            if bool(labels[inj[0]] & g) is not want:
                bad.append(i)
        assert not bad, (f"condition {cond} mis-assigned on {len(bad)} items; "
                         f"the manipulation did not happen as intended")
        print(f"  condition {cond}: all {len(gold)} injections "
              f"{'share' if want else 'do not share'} the item's gold label")

    # ------------------------------------------------------ paired tests
    c = {s: correct(s, gold) for s in
         ("oracle_A", "oracle_B", "sq3_r4_fb", "sq3_r4_ctl")}
    print(f"\n{'-' * 74}\nPAIRED, `contains` correctness on identical items\n")
    paired(c["oracle_A"], c["oracle_B"], "oracle_A", "oracle_B")
    paired(c["oracle_A"], c["sq3_r4_fb"], "oracle_A", "sq3_r4_fb")
    paired(c["sq3_r4_fb"], c["oracle_B"], "sq3_r4_fb", "oracle_B")

    # -------------------------------------------------- the mixture check
    a_rate = sum(c["oracle_A"].values()) / len(gold)
    b_rate = sum(c["oracle_B"].values()) / len(gold)
    print(f"{'-' * 74}\nMIXTURE CHECK   predicted = B + p x (A - B)")
    print(f"  A = {a_rate:.3f}   B = {b_rate:.3f}   A - B = "
          f"{a_rate - b_rate:+.3f}\n")
    print(f"  {'round':10} {'p':>6} {'predicted':>10} {'observed':>9} "
          f"{'diff':>7}")
    for stem, p in LABEL_HIT.items():
        obs = sum(correct(stem, gold).values()) / len(gold)
        pred = b_rate + p * (a_rate - b_rate)
        print(f"  {stem:10} {p:6.3f} {pred:10.3f} {obs:9.3f} "
              f"{obs - pred:+7.3f}")

    print(f"\n  Read every diff against the held-out `contains` floor of "
          f"{FLOOR_CONTAINS:.3f},\n  measured from four round-0 replicates. A "
          f"mixture that tracks within that\n  floor means the SQ3 null is "
          f"help and harm cancelling at the achieved\n  selection rate, not an "
          f"absence of effect.")
    print(f"\n  NOTE the label-relevance rates are measured on the RETRIEVED "
          f"correction\n  under normal operation, while A and B inject one "
          f"directly, so the\n  mixture assumes the model treats an injected "
          f"and a retrieved record\n  alike. Prompt shape is identical by "
          f"construction; the code path is not.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
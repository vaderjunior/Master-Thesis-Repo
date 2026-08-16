"""
scripts/check_feedback_gate_origin.py - why might the feedback arm have fewer
gate false negatives? Read-only, zero API calls.
Run: python -m scripts.check_feedback_gate_origin

THE OBSERVATION. Across every k=1 run, gate false negatives on the all-hateful
SQ3 subset were fb 59 / 64 / 62 / 54 against ctl 66 / 65 / 68 / 69 and round-0
68 / 65 / 67 / 69. max(fb)=64 is below min(everything else)=65: the four
lowest of twelve values, p = 1/C(12,4) = 0.002.

THE PROPOSED MECHANISM. Corrections are selected under `contains`: the gold
hate_type was absent from the predicted set. One way that happens is the
system calling the item not-hate at all, which zeroes hate_type through gate
consistency. So the feedback arm's records should be ENRICHED for items whose
gate was predicted False, while the control's records are items the system
already got right and therefore mostly gate=True. Seeing "this was hateful and
you said it was not" 60 times would push the gate.

WHAT THIS SCRIPT DOES. For every record in feedback_log.jsonl, find the run
that produced the classification it was selected from - round 1 selected
against the shared round-0 KB, round n>=2 against each arm's own round n-1 run
- and report what that run predicted for the gate.

IF THE ARMS DO NOT DIFFER HERE, the mechanism is wrong and the balanced-set
experiment is testing a story with no support. Cheaper to find out now.

TRAIN top-ups have no prediction to look up (they were never classified) and
are counted separately rather than silently dropped.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.apply_feedback import load_results

FEEDBACK_LOG = Path("kb") / "feedback_log.jsonl"
ARM_TAG = {"feedback": "fb", "control": "ctl"}


def source_run(arm: str, round_n: int) -> str:
    """The run whose predictions this record was selected against.

    Round 1 used one shared round-0 run for both arms, because both arms had
    the identical unedited KB at that point. From round 2 each arm reads its
    own previous run - an error of the feedback arm is an error of the
    feedback KB.
    """
    if round_n == 1:
        return "sq3_round0_r1"
    return f"sq3_r{round_n - 1}_{ARM_TAG[arm]}"


def main():
    records = [json.loads(l) for l
               in FEEDBACK_LOG.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    print(f"\n{'=' * 74}\n{FEEDBACK_LOG}: {len(records)} entries")

    needed = {source_run(r["meta"]["feedback_arm"], r["meta"]["round"])
              for r in records}
    res = {s: load_results(s) for s in sorted(needed)}
    print(f"  source runs: {', '.join(sorted(needed))}\n")

    tally = defaultdict(Counter)
    for r in records:
        arm, rnd = r["meta"]["feedback_arm"], r["meta"]["round"]
        run = res[source_run(arm, rnd)]
        item = run.get(r["meta"]["origin_item_id"])
        if item is None:
            # TRAIN top-up: never classified, so there is no gate prediction.
            tally[(arm, rnd)]["train_topup"] += 1
            continue
        pred = item.get("result")
        if pred is None:
            tally[(arm, rnd)]["no_prediction"] += 1
        elif bool(pred["hate"]):
            tally[(arm, rnd)]["gate_true"] += 1
        else:
            tally[(arm, rnd)]["gate_false"] += 1

    keys = ["gate_false", "gate_true", "train_topup", "no_prediction"]
    print(f"  {'arm':10} {'round':>5} " + " ".join(f"{k:>13}" for k in keys))
    totals = defaultdict(Counter)
    for (arm, rnd), c in sorted(tally.items()):
        print(f"  {arm:10} {rnd:5} " + " ".join(f"{c[k]:13}" for k in keys))
        totals[arm] += c

    print()
    for arm, c in sorted(totals.items()):
        n = sum(c.values())
        classified = c["gate_false"] + c["gate_true"]
        share = c["gate_false"] / classified if classified else 0.0
        print(f"  {arm:10} TOTAL " + " ".join(f"{c[k]:13}" for k in keys)
              + f"   ({n} records, {share:.3f} of classified were gate=False)")

    fb, ctl = totals.get("feedback", Counter()), totals.get("control", Counter())
    f_cl = fb["gate_false"] + fb["gate_true"]
    c_cl = ctl["gate_false"] + ctl["gate_true"]
    print(f"\n  Mechanism check: the feedback arm's records are"
          f" {fb['gate_false'] / max(f_cl, 1):.3f} gate=False,"
          f"\n  the control's {ctl['gate_false'] / max(c_cl, 1):.3f}."
          f"\n  A large gap supports the proposed mechanism - the feedback"
          f"\n  arm repeatedly shows the model items it wrongly called"
          f"\n  not-hate. A small gap refutes it, and the fewer gate false"
          f"\n  negatives need a different explanation or none.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
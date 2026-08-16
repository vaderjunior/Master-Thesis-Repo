"""
scripts/make_sq3_sentinel.py - freeze the over-steering sentinel.
Read-only on results, writes one JSON. Zero API calls.
Run: python -m scripts.make_sq3_sentinel --dry-run

WHAT THE SENTINEL IS. Items the system got right BEFORE any feedback, frozen,
never corrected, re-scored every round. The metric is flips-to-wrong: does
correcting one thing break another.

WHY ALL-k AND NOT ONE RUN. The Phase 5 spec said "items the system got right
on the first stable run". At T=1.0 that is not a stable property: on the pool
role, 120 of 229 items are correct in at least one of three replicates but
only 84 in all three. A single-run sentinel would sweep in those 36 items,
which then flip back for reasons that have nothing to do with feedback and
would be read as over-steering.

WHY THE BASELINE NEEDS A FOURTH RUN. The sentinel is SELECTED on r1/r2/r3, so
its flip rate measured on those same three runs is zero by construction. A
fourth independent replicate under an unchanged KB is the no-edit control, and
the difference between its flip rate and a post-edit round's is the actual
over-steering measurement. Selecting on three runs and measuring on the same
three is regression to the mean wearing a result's clothes.

WHY LABEL BALANCE IS CHECKED. Membership is decided by what the model gets
right, and it does not get every label right equally: on held_out, per-label
F1 ranges from 0.125 (incitement) to 0.583 (irony). So the sentinel is
unbalanced BY CONSTRUCTION, and a flip count over it is weighted toward
whichever labels the model already handles. That has to be reported with the
number, not discovered afterwards.

THE REMAINDER IS NOT WASTE. Pool items that miss the sentinel are never
corrected and never re-scored as sentinel, which makes them functionally
identical to held_out. They are written out as a second never-corrected
stratum. They are NOT a random sample - they are exactly the items that failed
all-k, so they are enriched for hard cases. Fixed across rounds, so deltas
stay valid, but the absolute level is lower and the stratum must be labelled
as enriched wherever it is reported.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.check_sq3_coverage import PROCESSED, local_gold_sets

RESULTS = Path("experiments/results")
OUT = Path("experiments") / "sq3_sentinel_r0.json"
SUBSET = "en_dev_eval_sq3_types"
ROLE = "pool"
DIM, GOLD_COL = "hate_type", "hate_types"
CRITERION = "contains"      # decision of 2026-08-06; see experiment_log


def load(stem: str) -> dict:
    path = RESULTS / f"{stem}_live.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" in r or r.get("arm") != "rag":
            continue
        out[r["item_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*",
                    default=["sq3_round0_r1", "sq3_round0_r2",
                             "sq3_round0_r3"])
    ap.add_argument("--cap-per-label", type=int, default=None,
                    help="cap membership per label to force balance. Off by "
                         "default: capping discards real measurements, and "
                         "reporting the imbalance is more honest than hiding "
                         "it behind a balanced-looking number.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    pool = df[df["sq3_role"] == ROLE].reset_index(drop=True)
    gold = local_gold_sets(pool, GOLD_COL)
    print(f"\n{'=' * 74}\n{SUBSET} [{ROLE}]: {len(gold)} scorable items")

    runs = {s: load(s) for s in args.runs}
    for s, r in runs.items():
        missing = set(gold) - set(r)
        assert not missing, f"{s} is missing {len(missing)} pool items"

    # Attribution. A sentinel frozen under one system state and re-scored
    # under another measures the state change, not over-steering.
    for field in ("kb_version", "prompt_version", "active_model",
                  "temperature"):
        vals = {str(it[field]) for r in runs.values() for it in r.values()}
        assert len(vals) == 1, f"runs disagree on {field}: {sorted(vals)}"
        print(f"  {field:16} {vals.pop()}")

    # ------------------------------------------------------ the membership
    def ok(item, iid) -> bool:
        res = item.get("result")
        if res is None:
            return False
        pred = set(res.get(DIM) or [])
        return (gold[iid] <= pred) if CRITERION == "contains" else (
            pred == gold[iid])

    per_run = {s: {i: ok(r[i], i) for i in gold} for s, r in runs.items()}
    allk = sorted(i for i in gold if all(per_run[s][i] for s in per_run))
    anyk = sorted(i for i in gold if any(per_run[s][i] for s in per_run))
    remainder = sorted(set(gold) - set(allk))

    print(f"\n  criterion={CRITERION}, k={len(runs)}")
    for s in args.runs:
        print(f"    {s:22} {sum(per_run[s].values()):4} correct "
              f"({sum(per_run[s].values()) / len(gold):.3f})")
    print(f"    {'correct in ALL k':22} {len(allk):4}  <- the sentinel")
    print(f"    {'correct in >=1':22} {len(anyk):4}")
    print(f"    {'swept in by a 1-run draw':22} {len(anyk) - len(allk):4}")
    print(f"    {'remainder (never corrected, enriched hard)':>22} "
          f"{len(remainder):4}")

    # ------------------------------------------------------- label balance
    pool_c = Counter(l for v in gold.values() for l in v)
    sent_c = Counter(l for i in allk for l in gold[i])
    print(f"\n  label balance - membership is decided by what the model gets"
          f"\n  right, so the sentinel is unbalanced BY CONSTRUCTION:")
    print(f"    {'label':20} {'pool':>6} {'sentinel':>9} {'rate':>7}")
    for label in sorted(pool_c):
        p, s = pool_c[label], sent_c.get(label, 0)
        print(f"    {label:20} {p:6} {s:9} {s / p:7.3f}")
    if sent_c:
        lo = min(sent_c.values())
        hi = max(sent_c.values())
        print(f"    -> {hi / max(lo, 1):.1f}x between the best- and "
              f"worst-represented label")
        thin = [l for l in pool_c if sent_c.get(l, 0) < 10]
        if thin:
            print(f"    -> below 10 members, so no per-label flip claim is "
                  f"possible for: {thin}")

    if args.cap_per_label:
        keep, seen = [], Counter()
        for i in allk:                      # allk is id-sorted, deterministic
            labels = gold[i]
            if all(seen[l] < args.cap_per_label for l in labels):
                keep.append(i)
                seen.update(labels)
        print(f"\n  --cap-per-label {args.cap_per_label}: "
              f"{len(allk)} -> {len(keep)} items")
        allk = keep

    # ------------------------------------------------------------- write
    payload = {
        "criterion": CRITERION,
        "k": len(runs),
        "runs": list(args.runs),
        "subset": SUBSET,
        "role": ROLE,
        "kb_version": next(iter(runs.values()))[allk[0]]["kb_version"],
        "prompt_version": next(iter(runs.values()))[allk[0]]["prompt_version"],
        "sentinel_ids": allk,
        "remainder_ids": remainder,
        "sentinel_label_support": dict(sent_c),
        "note": ("Sentinel selected on these runs, so its flip rate measured "
                 "on them is 0 by construction. The no-edit baseline needs a "
                 "fourth independent replicate under the same KB."),
    }
    if args.dry_run:
        print(f"\n  --dry-run: would write {OUT}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}  ({len(allk)} sentinel, {len(remainder)} "
          f"remainder)")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
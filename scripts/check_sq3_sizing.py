"""
scripts/check_sq3_sizing.py - fixes the last free parameter before the SQ3
subset can be drawn. Read-only, zero API calls.
Run: python -m scripts.check_sq3_sizing

WHAT IT DECIDES. 8.4 draws the over-steering sentinel from items the system
gets right in ALL replicates, not in one run, because at T=1.0 with a 0.058
floor on hate_type a single-run "correct" includes items that were right by
luck. That makes the sentinel POOL larger than the sentinel, by a factor
nobody has measured. The supplementary subset cannot be drawn until it is:

    N          = sentinel_pool + held_out + feedback_batches
    N          <= 7 * (rarest label in the unused dev pool)     [balanced draw]
    pool * r   = 80                                            [8.4's target]

so r, the all-replicate-correct rate, decides whether the guide's partition
survives. This script measures r instead of assuming it.

WHY IT COSTS NOTHING. en_dev_eval_types is the same dimension, label-balanced,
103 items, and has been run four times at T=1.0. Those results are on disk
with per-item voted predictions AND per-vote raw predictions, so both levels
of replication are already measured. A number this load-bearing should be
transcribed, not estimated - the D7 rule applies hardest to the parameter
everything downstream is sized against.

WHY IT REFUSES TO POOL SILENTLY. The candidate runs straddle the KB v1 -> v2
edit and several prompt_version changes. Averaging a stability rate across
system states is the mismatch COMPARABILITY.md exists to catch, so the stamps
are printed and heterogeneity is warned about rather than absorbed.

en_dev_eval_types is MEASURED FROM here, never written to. It is an SQ2
reporting subset and KB write-back would contaminate it exactly as it would
en_dev_eval_main.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from scripts.check_sq3_coverage import (EN_DEV_SUBSETS, PROCESSED,
                                        local_gold_sets)
from src.hsrag.metrics import score_multilabel

RESULTS = Path("experiments/results")
DEFAULT_SUBSET = "en_dev_eval_types"   # same dimension, already replicated
DEFAULT_GLOB = "*types*_live.jsonl"
PARENT = "en_dev"
DIM, GOLD_COL = "hate_type", "hate_types"

# Guide 8.1's partition. sentinel here is the TARGET, not the pool.
SENTINEL_TARGET = 80
HELD_OUT = 120
FEEDBACK_BATCHES = 4 * 30


def load_run(path: Path) -> dict:
    """Read a results jsonl into {arm: [ItemResult dicts]}.

    First line is the resolved manifest, not a result - the runner writes it
    there so a results file is self-describing. score_all skips it the same
    way.
    """
    by_arm = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "_manifest" in rec:
            continue
        by_arm[rec["arm"]].append(rec)
    return by_arm


def stamps(items: list) -> dict:
    """The attribution stamps, collapsed. Distinct values are kept as sets so
    a run that changed model or KB mid-flight is visible rather than averaged."""
    return {k: sorted({str(i.get(k)) for i in items})
            for k in ("temperature", "kb_version", "prompt_version",
                      "active_model")}


def correct_map(items: list, gold: dict, criterion: str) -> dict:
    """item_id -> was this item correct, under one criterion.

    exact     - predicted set equals gold set
    contains  - gold set is a subset of the predicted set

    Gold on this dimension is single-label by construction (Implicit Hate
    assigns one type per post), so 'contains' means the one gold label is
    somewhere in a prediction that may carry two or three others. The two
    criteria are not a detail: 'contains' scores an over-predicting model as
    correct, and over-prediction is the largest measured gap on this
    dimension, so an error-selection rule built on 'contains' would never
    correct the thing feedback is most able to move.

    An item with no valid prediction is not correct. It cannot enter a
    sentinel either way, so folding it in here rather than excluding it keeps
    the rate honest about what a real round would find.
    """
    out = {}
    for it in items:
        iid = it["item_id"]
        if iid not in gold:
            continue
        res = it.get("result")
        if res is None:
            out[iid] = False
            continue
        pred = set(res.get(DIM) or [])
        out[iid] = (pred == gold[iid]) if criterion == "exact" else (
            gold[iid] <= pred)
    return out


def vote_correct_map(items: list, gold: dict, criterion: str) -> dict:
    """Same, but requiring EVERY individual vote to be correct.

    This is decoding noise alone, with the self-consistency vote removed. The
    gap between this and the voted rate says how much of the system's apparent
    stability the vote is supplying - which matters because 8.4's sentinel is
    built on voted output.
    """
    out = {}
    for it in items:
        iid = it["item_id"]
        if iid not in gold:
            continue
        runs = [r for r in it.get("run_predictions") or []]
        if not runs or any(r is None for r in runs):
            out[iid] = False
            continue
        ok = []
        for r in runs:
            pred = set(r.get(DIM) or [])
            ok.append((pred == gold[iid]) if criterion == "exact"
                      else (gold[iid] <= pred))
        out[iid] = all(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="rag")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="explicit file stems; default is every run found")
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--role", default=None,
                    help="restrict to one sq3_role. The sentinel is drawn "
                         "from pool, so r must be measured on pool - "
                         "measuring it over the whole subset would mix in "
                         "items no sentinel can ever contain.")
    args = ap.parse_args()

    # ------------------------------------------------------------ the gold
    df = pd.read_parquet(PROCESSED / f"{args.subset}.parquet")
    gold = local_gold_sets(df, GOLD_COL)
    n_auth = score_multilabel(
        [{"item_id": str(i), "result": {"hate": True, DIM: []}, }
         for i in df["id"]],
        {str(r.id): r for r in df.itertuples(index=False)},
        GOLD_COL, DIM).n_items
    assert len(gold) == n_auth, (
        f"local predicate disagrees with metrics.py: {len(gold)} vs {n_auth}")

    # Role filtering happens AFTER the cross-check above, so the predicate is
    # still verified against the whole subset even when one role is measured.
    if args.role:
        assert "sq3_role" in df.columns, f"{args.subset} has no sq3_role"
        keep = {str(i) for i in df.loc[df["sq3_role"] == args.role, "id"]}
        assert keep, f"no items with sq3_role={args.role}"
        gold = {k: v for k, v in gold.items() if k in keep}

    card = Counter(len(v) for v in gold.values())
    _lbl = args.subset + (f" [{args.role}]" if args.role else "")
    print(f"\n{'=' * 74}\nGOLD: {_lbl}, {len(gold)} {DIM}-scorable items")
    print(f"  label cardinality: "
          + ", ".join(f"{k} label(s)={v}" for k, v in sorted(card.items())))

    # ------------------------------------------------------- the inventory
    files = sorted(RESULTS.glob(args.glob))
    if args.runs:
        files = [f for f in files if f.stem.replace("_live", "") in args.runs]

    runs = {}
    print(f"\n{'-' * 74}\nINVENTORY  (arm filter: {args.arm})")
    print(f"  {'run':28} {'arm':10} {'n':>4} {'T':>5}  kb / prompt / model")
    for f in files:
        for arm, items in load_run(f).items():
            st = stamps(items)
            mark = "*" if arm == args.arm else " "
            print(f" {mark}{f.stem.replace('_live', ''):28} {arm:10} "
                  f"{len(items):4} {'/'.join(st['temperature']):>5}  "
                  f"{'/'.join(s[:8] for s in st['kb_version'])} | "
                  f"{'/'.join(s[-8:] for s in st['prompt_version'])} | "
                  f"{'/'.join(s.split('/')[-1] for s in st['active_model'])}")
            if arm == args.arm:
                runs[f.stem.replace("_live", "")] = items
    print("  * = included in the stability computation below")

    if len(runs) < 2:
        print(f"\nNeed at least 2 runs on arm={args.arm}. Found {len(runs)}.")
        return

    # Heterogeneity warning. Pooling across system states is exactly the
    # mismatch COMPARABILITY.md exists to prevent, so it is surfaced rather
    # than absorbed into the average.
    all_items = [i for items in runs.values() for i in items]
    st = stamps(all_items)
    hetero = {k: v for k, v in st.items() if len(v) > 1}
    if hetero:
        print(f"\n  WARNING - the selected runs do not share a system state:")
        for k, v in hetero.items():
            print(f"    {k}: {v}")
        print("    The pooled rate below mixes these. Read it as a range, and"
              "\n    prefer the per-run column when they disagree materially.")

    # -------------------------------------------------------- correctness
    ids = set.intersection(*[{i["item_id"] for i in v} for v in runs.values()])
    ids &= set(gold)
    print(f"\n{'-' * 74}\nCORRECTNESS  ({len(ids)} items common to all "
          f"{len(runs)} runs)")
    print(f"  {'run':28} {'exact':>8} {'contains':>9} {'mean pred':>10} "
          f"{'no pred':>8}")

    maps = {"exact": {}, "contains": {}}
    for name, items in runs.items():
        row = [name]
        for crit in ("exact", "contains"):
            m = {k: v for k, v in correct_map(items, gold, crit).items()
                 if k in ids}
            maps[crit][name] = m
            row.append(sum(m.values()) / len(m))
        mp = [len(set((i.get("result") or {}).get(DIM) or []))
              for i in items if i["item_id"] in ids and i.get("result")]
        nop = sum(1 for i in items if i["item_id"] in ids and not i.get("result"))
        print(f"  {row[0]:28} {row[1]:8.3f} {row[2]:9.3f} "
              f"{sum(mp) / max(len(mp), 1):10.2f} {nop:8}")

    # ----------------------------------------------------- the rate itself
    print(f"\n{'-' * 74}\nACROSS-RUN STABILITY  (k = {len(runs)} replicates)")
    print(f"  {'criterion':12} {'all k':>10} {'>=1 run':>10} "
          f"{'shrinkage':>11}")
    rates = {}
    for crit in ("exact", "contains"):
        per = maps[crit]
        allk = sum(1 for i in ids if all(per[n][i] for n in per))
        anyk = sum(1 for i in ids if any(per[n][i] for n in per))
        rates[crit] = allk / len(ids)
        print(f"  {crit:12} {allk:5} ({rates[crit]:.0%}) {anyk:5} "
              f"({anyk / len(ids):.0%}) {anyk - allk:6} items")
    print("\n  Shrinkage is the count of items right in at least one run and"
          "\n  wrong in another - the ones a single-run sentinel would have"
          "\n  swept in, and the reason 8.4 draws from all-replicate-correct.")

    print(f"\n  within-run vote stability (decoding noise, vote removed):")
    for crit in ("exact", "contains"):
        vals = []
        for name, items in runs.items():
            m = {k: v for k, v in vote_correct_map(items, gold, crit).items()
                 if k in ids}
            vals.append(sum(m.values()) / len(m))
        print(f"    {crit:12} every vote correct: "
              f"{min(vals):.3f}-{max(vals):.3f} across runs")

    # ---------------------------------------------------------- the ceiling
    # Only meaningful BEFORE the subset is drawn. Once sq3_role exists the
    # partition is frozen, and printing a draw ceiling over it would describe
    # a decision already made - the same error as printing "make_eval_subsets
    # asked for 350" over a file that script never created.
    if "sq3_role" in df.columns:
        print(f"\n{'=' * 74}\n{args.subset} has a frozen partition; "
              f"draw-ceiling and sizing sections skipped.\n{'=' * 74}\n")
        return

    parent = pd.read_parquet(PROCESSED / f"{PARENT}.parquet")
    used = set()
    for name in EN_DEV_SUBSETS:
        p = PROCESSED / f"{name}.parquet"
        if p.exists():
            used |= set(pd.read_parquet(p, columns=["id"])["id"])
    unused = parent[~parent["id"].isin(used)]
    pool_sets = local_gold_sets(unused, GOLD_COL)
    pool = Counter(l for v in pool_sets.values() for l in v)

    # ITEMS and LABEL INSTANCES are different numbers - a few items carry two
    # hate_types - and the ladder counts label instances while the partition
    # is sized in items. Printing both stops a 13-item gap being read as a
    # 13-item shortfall.
    print(f"\n{'-' * 74}\nDRAW CEILING  (unused {PARENT}: {len(pool_sets)} "
          f"{DIM}-scorable items, {sum(pool.values())} label instances)")
    print("  Gold is single-label, so a balanced draw is ~7 x per_label and"
          "\n  the rarest label caps it. Trimming the rarest label does not"
          "\n  necessarily raise the cap - the ladder says whether it does:")
    ladder = sorted(pool.items(), key=lambda kv: kv[1])
    for i in range(len(ladder)):
        kept = ladder[i:]
        cap = kept[0][1] * len(kept)
        drop = ", ".join(k for k, _ in ladder[:i]) or "none"
        print(f"    drop {drop:34} -> {len(kept)} labels x {kept[0][1]:3} "
              f"= {cap:4} items")

    cap = ladder[0][1] * len(ladder)
    max_pool = cap - HELD_OUT - FEEDBACK_BATCHES

    # ----------------------------------------------------------- the verdict
    print(f"\n{'=' * 74}\nSIZING")
    print(f"  held_out {HELD_OUT} + feedback_batches {FEEDBACK_BATCHES} "
          f"+ sentinel_pool  <=  {cap}")
    print(f"  -> sentinel_pool <= {max_pool}")
    for crit in ("exact", "contains"):
        r = rates[crit]
        need = SENTINEL_TARGET / r if r else float("inf")
        gets = max_pool * r
        ok = need <= max_pool
        print(f"\n  {crit}: r = {r:.3f}")
        print(f"    pool needed for a {SENTINEL_TARGET}-item sentinel: "
              f"{need:.0f}")
        print(f"    largest sentinel a {max_pool}-item pool yields: {gets:.0f}")
        if ok:
            n_total = need + HELD_OUT + FEEDBACK_BATCHES
            print(f"    -> CLOSES. Draw per_label = {n_total / 7:.0f} "
                  f"(~{n_total:.0f} items), partitioned "
                  f"{need:.0f} pool / {HELD_OUT} held-out / "
                  f"{FEEDBACK_BATCHES} batches")
        else:
            print(f"    -> DOES NOT CLOSE. Levers: smaller sentinel target,"
                  f"\n       fewer rounds, or a smaller held-out set.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
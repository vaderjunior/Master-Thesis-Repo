"""
scripts/check_generalisation.py - guide 8.5, the generalisation probe.
Read-only, zero API calls. Embeds locally with BGE-M3.
Run: python -m scripts.check_generalisation

THE QUESTION THE LEARNING CURVE CANNOT ANSWER. A flat aggregate is consistent
with two very different worlds: corrections help nobody, or corrections help a
small neighbourhood and the effect is diluted across 350 items. This separates
them by asking whether benefit DECAYS WITH DISTANCE from a correction.

WHY DISTANCE BANDS AND NOT JUST "TOP-5 NEIGHBOURS". The Phase 5 spec asked for
each corrected item's top-5 nearest unseen items, scored before and after. As
written that is close to circular: top-5 by BGE-M3 is the same space the
retriever ranks in, so those neighbours are near-selected to retrieve the new
record, and any before/after difference is partly guaranteed. Banding the whole
stratum by distance and looking for DECAY removes that: a uniform shift across
all bands is a generic post-edit effect, and only a gradient is generalisation.

WHY ITEM-CENTRIC RATHER THAN CORRECTION-CENTRIC. Taking top-5 per correction
produces overlapping sets - one item can be a near neighbour of three
corrections and a mid neighbour of ten - so items get counted repeatedly with
no principled weight. Ranking each item by its similarity to its NEAREST
correction gives disjoint bands and counts every item exactly once.

THE CONTROL ARM IS THE POINT. Its 60 records sit in the same bucket and are
just as retrievable, so it gets its own bands around its own records. If the
near band improves equally for both arms, proximity to ANY added record
explains it and error-selection does not. Only a steeper gradient for the
feedback arm is generalisation from correction.

BASELINE IS THE MEAN OF FOUR ROUND-0 REPLICATES, per item, so a band's
"before" is not one noisy draw. Correctness is `contains`, matching the
criterion every other SQ3 measurement uses.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from scripts.apply_feedback import load_results
from scripts.check_sq3_coverage import PROCESSED, local_gold_sets

SUBSET = "en_dev_eval_sq3_types"
GOLD_COL, DIM = "hate_types", "hate_type"
FEEDBACK_LOG = Path("kb") / "feedback_log.jsonl"
BASELINE = ["sq3_round0_r1", "sq3_round0_r2", "sq3_round0_r3", "sq3_round0_r4"]
ARMS = {"feedback": "sq3_r4_fb", "control": "sq3_r4_ctl"}


def correctness(stem: str, gold: dict) -> dict:
    """item_id -> 1.0 / 0.0 under `contains`."""
    res = load_results(stem)
    out = {}
    for i in gold:
        r = res[i].get("result")
        pred = set((r or {}).get(DIM) or [])
        out[i] = float(gold[i] <= pred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--band-source", default="feedback",
                    choices=["feedback", "control"],
                    help="whose records define the distance bands; both arms "
                         "are then scored on the SAME items")
    ap.add_argument("--round", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    fixed = df[df["sq3_role"].isin(["held_out", "pool"])].reset_index(drop=True)
    gold = local_gold_sets(fixed, GOLD_COL)
    ids = sorted(gold)
    texts = {str(r.id): str(r.text) for r in fixed.itertuples(index=False)}
    print(f"\n{'=' * 78}\nfixed stratum: {len(ids)} never-corrected items")

    # Corrections up to the requested round, per arm, deduplicated on id
    # because the same text corrected twice is one record (upsert semantics).
    log = [json.loads(l) for l
           in FEEDBACK_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_arm = {}
    for arm in ARMS:
        recs = {r["id"]: r for r in log
                if r["meta"]["feedback_arm"] == arm
                and r["meta"]["round"] <= args.round}
        by_arm[arm] = list(recs.values())
        print(f"  {arm:9} {len(recs)} records through round {args.round}")

    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    model = SentenceTransformer(cfg["kb"]["embedding_model"])
    item_vecs = model.encode([texts[i] for i in ids],
                             normalize_embeddings=True, batch_size=16)

    # Baseline: per-item mean over the four round-0 replicates.
    base = [correctness(s, gold) for s in BASELINE]
    base_mean = {i: sum(b[i] for b in base) / len(base) for i in ids}
    print(f"  baseline: mean of {len(BASELINE)} round-0 replicates, "
          f"overall {sum(base_mean.values()) / len(ids):.3f}")

    # LEAKAGE GUARD. draw_train_topup excludes texts already in the base KB
    # and in the current batch, but NOT texts in the fixed stratum. A TRAIN
    # item that near-duplicates a held-out item hands that arm a free answer.
    # The first run of this script showed the control's nearest band reaching
    # 0.987 cosine while the feedback arm's reached 0.749, which is the
    # signature of exactly that.
    for arm in ARMS:
        rv = model.encode([r["text"] for r in by_arm[arm]],
                          normalize_embeddings=True, batch_size=16)
        S = item_vecs @ rv.T
        hits = np.argwhere(S > 0.95)
        if len(hits):
            print(f"\n  LEAKAGE WARNING - {arm}: {len(hits)} record/item "
                  f"pair(s) above 0.95 cosine")
            for a, b in hits[:5]:
                rec = by_arm[arm][b]
                print(f"    sim {S[a, b]:.3f}  [{rec['source']}]")
                print(f"      item {ids[a]}: {texts[ids[a]][:88]}")
                print(f"      rec  {rec['id']}: {rec['text'][:88]}")

    # Bands from ONE arm's records, both arms scored on the same items, so
    # every row is a paired comparison. Arm-specific bands put the two arms on
    # different item sets with band baselines differing by up to 0.19, and no
    # band-level comparison was possible.
    src = by_arm[args.band_source]
    rvecs = model.encode([r["text"] for r in src],
                         normalize_embeddings=True, batch_size=16)
    sim = (item_vecs @ rvecs.T).max(axis=1)
    order = np.argsort(-sim)
    after = {arm: correctness(stem, gold) for arm, stem in ARMS.items()}

    n = len(ids)
    edges = [round(n * b / args.bands) for b in range(args.bands + 1)]
    print(f"\n{'-' * 78}\nbands by distance to the {args.band_source} arm's "
          f"{len(src)} records; both arms on the same items")
    print(f"  {'band':12} {'n':>4} {'sim range':>15} {'before':>8} "
          f"{'fb':>8} {'ctl':>8} {'fb-ctl':>8}")
    diffs = []
    for b in range(args.bands):
        idx = order[edges[b]:edges[b + 1]]
        band = [ids[j] for j in idx]
        lo, hi = sim[idx].min(), sim[idx].max()
        before = sum(base_mean[i] for i in band) / len(band)
        a_fb = sum(after["feedback"][i] for i in band) / len(band)
        a_ct = sum(after["control"][i] for i in band) / len(band)
        diffs.append(a_fb - a_ct)
        label = ("nearest" if b == 0 else
                 "farthest" if b == args.bands - 1 else f"band {b + 1}")
        print(f"  {label:12} {len(band):4} {hi:6.3f} - {lo:6.3f} "
              f"{before:8.3f} {a_fb:8.3f} {a_ct:8.3f} {a_fb - a_ct:+8.3f}")
    print(f"\n  fb-ctl gradient (nearest - farthest): "
          f"{diffs[0] - diffs[-1]:+.3f}")

    # PERMUTATION TEST. At 70 items per band a difference of a few points is
    # what resampling alone produces, and the farthest band coming second is
    # not what any distance-decay story predicts. Shuffling band membership
    # while holding the per-item results fixed says how often chance produces
    # a nearest band this favourable, and a gradient this steep.
    rng = np.random.default_rng(42)
    per_item = np.array([after["feedback"][i] - after["control"][i]
                         for i in ids])
    obs_near, obs_grad = diffs[0], diffs[0] - diffs[-1]
    size = edges[1] - edges[0]
    hits_near = hits_grad = 0
    trials = 10000
    for _ in range(trials):
        p = rng.permutation(n)
        near = per_item[p[:size]].mean()
        far = per_item[p[edges[-2]:]].mean()
        hits_near += near >= obs_near
        hits_grad += (near - far) >= obs_grad
    print(f"  permutation, {trials} shuffles of band membership:")
    print(f"    nearest band >= {obs_near:+.3f}: p = "
          f"{(hits_near + 1) / (trials + 1):.3f}")
    print(f"    gradient    >= {obs_grad:+.3f}: p = "
          f"{(hits_grad + 1) / (trials + 1):.3f}")

    print(f"\n{'=' * 78}")
    print("HOW TO READ THIS. Generalisation from correction requires the")
    print("feedback arm's delta to DECAY with distance AND its gradient to")
    print("exceed the control's. A positive gradient in both arms means")
    print("proximity to any added record explains it, not error-selection. A")
    print("flat gradient in both means the benefit, if any, is uniform across")
    print("the stratum and is not generalisation from a neighbourhood.")
    print("Read every delta against the fixed stratum's round-0 floor of")
    print("0.033 macro / 0.046 contains, and note that a band of 70 items has")
    print("a wider floor still than the 350-item stratum those were measured")
    print(f"on.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
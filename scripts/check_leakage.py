"""
scripts/check_leakage.py - near-duplicate leakage between the TRAIN splits and
every eval subset. Read-only, zero API calls. Embeds locally with BGE-M3.

  python -m scripts.check_leakage --smoke     # 2000 train rows, ~1 min, verify it works
  python -m scripts.check_leakage             # the real scan

WHY THIS MATTERS NOW AND DID NOT BEFORE. Every disjointness guarantee in this
project operates on ids or on exact text. PHASE8_SIGNOFF Part 5 records
implicit_hate-7570 and a b4 control record differing only by a leading '": "'
at cosine 0.987 - two distinct ids, byte-different text, and invisible to
every check we have. For a FROZEN LLM that is harmless: it cannot memorise
between runs, and the note correctly says "no effect here".

An encoder memorises. If TRAIN contains near-twins of eval items, Arm A's
score is inflated, and Arm A is the ceiling the whole adaptability comparison
is read against. The cost table in 7.10 then sits on a soft number. This has
to run BEFORE any checkpoint is trained, because afterwards the only honest
options are to retrain or to caveat.

THREE DETECTORS, CHEAPEST FIRST.
  1. exact text match          - catches copy-paste duplication across splits
  2. normalised text match     - NFKC, casefold, whitespace collapsed, edge
                                 punctuation stripped. Catches the documented
                                 '": "' case at zero embedding cost, and every
                                 quoting/encoding variant of it.
  3. cosine >= 0.95 / 0.98     - catches paraphrase and truncation

Reported separately, because they mean different things. An exact match is a
data-preparation bug. A 0.96 cosine on two different tweets about the same
news event is not leakage at all, and the report prints pairs so that
judgement stays with a human rather than with a threshold.

LANGUAGE-POOLED, NOT SPLIT-POOLED. An eval subset is compared against every
TRAIN row of the same language, from all train splits at once, plus the KB
examples. The KB is included because Arm B trains on exactly
kb/records.jsonl where kind == "example", so a KB example that near-duplicates
an eval item leaks into Arm B specifically.

WHAT TO DO WITH THE OUTPUT. Either drop the offending TRAIN rows before
training and say so, or keep them and report the count as a bound on Arm A.
Both are defensible. Silence is not, because the defect is already documented
in our own sign-off.
"""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROCESSED = Path("data/processed")
KB_RECORDS = Path("kb/records.jsonl")
CACHE = Path("scratch/leakage_emb")
OUT = Path("experiments/leakage_report.json")
EXCL = Path("experiments/leakage_train_exclusions.json")

TRAIN_SPLITS = ["en_train", "de_train", "de_legal_train"]


def norm_text(s: str) -> str:
    """Aggressive normalisation for the cheap duplicate detector.

    NFKC folds the fullwidth / compatibility characters that survive a
    copy-paste through a spreadsheet. casefold beats lower() on German
    (strasse / STRASSE). Edge punctuation is stripped because the documented
    near-duplicate differs from its twin by exactly a leading '": "'.
    """
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\r\n\"'`«»“”‘’.,:;!?-–—…")


def load_train() -> dict:
    """lang -> list of (source, id, text). Pools every train split plus the KB."""
    pool = defaultdict(list)
    for split in TRAIN_SPLITS:
        p = PROCESSED / f"{split}.parquet"
        if not p.exists():
            print(f"  MISSING {p} - skipped")
            continue
        df = pd.read_parquet(p)
        for r in df.itertuples(index=False):
            pool[str(getattr(r, "lang", "en"))].append(
                (split, str(r.id), str(r.text)))
        print(f"  {split:16} {len(df):>7} rows")

    if KB_RECORDS.exists():
        n = 0
        for line in KB_RECORDS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") != "example":
                continue
            lang = rec.get("lang") or rec.get("meta", {}).get("lang") or "en"
            pool[str(lang)].append(("kb_example", rec["id"], str(rec["text"])))
            n += 1
        print(f"  {'kb (examples)':16} {n:>7} rows")

    for lang, rows in pool.items():
        print(f"  -> lang {lang}: {len(rows)} train-side texts")
    return pool


def eval_subsets() -> list:
    """Every eval subset on disk, found rather than hardcoded.

    The en_test_eval* names are not recorded anywhere this script can read,
    and a hardcoded list that silently omits one is exactly the failure this
    scan exists to prevent.
    """
    return sorted(p for p in PROCESSED.glob("*_eval*.parquet"))


def encode(model, texts: list, tag: str, batch: int, use_cache: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    # Cache key includes the row count so a rebuilt split invalidates it. NOT
    # a content hash: that would cost a full pass over 56k texts to decide
    # whether to skip a full pass over 56k texts.
    f = CACHE / f"{tag}_{len(texts)}.npy"
    if use_cache and f.exists():
        print(f"    cached {f.name}")
        return np.load(f)
    v = model.encode(texts, normalize_embeddings=True, batch_size=batch,
                     show_progress_bar=True)
    v = np.asarray(v, dtype=np.float32)
    if use_cache:
        np.save(f, v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2000 train rows per language; verifies the whole "
                         "path in ~1 min before committing to the full scan")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4000,
                    help="train rows per similarity chunk; lower it if the "
                         "GPU runs out of memory")
    ap.add_argument("--near", type=float, default=0.95)
    ap.add_argument("--very-near", type=float, default=0.98)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--only", default=None,
                    help="restrict the train side to ONE source: en_train, "
                         "de_train, de_legal_train or kb_example. The pooled "
                         "scan answers 'is this eval item near anything we "
                         "train on at all'. --only answers 'is it near "
                         "something THIS model trains on', which is the "
                         "question per encoder head. A 0.96 twin in the "
                         "320-record KB is invisible in the pooled scan "
                         "whenever a 56k-row split has a 0.99, so the pooled "
                         "run CANNOT clear the KB or Arm B.")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    print(f"\n{'=' * 78}\nTRAIN -> EVAL near-duplicate scan\n{'=' * 78}")
    print("\ntrain side:")
    pool = load_train()
    if args.only:
        pool = {k: [r for r in v if r[0] == args.only]
                for k, v in pool.items()}
        pool = {k: v for k, v in pool.items() if v}
        print(f"\n  --only {args.only}: train side restricted to that source.")
        if not pool:
            print("  no rows match that source name; nothing to compare.")
            return
    if args.smoke:
        pool = {k: v[:2000] for k, v in pool.items()}
        print("\n  SMOKE MODE: 2000 train rows per language. Counts below are "
              "NOT the real answer.")

    subsets = eval_subsets()
    print(f"\neval side: {len(subsets)} subsets found")
    for p in subsets:
        print(f"  {p.stem}")

    from sentence_transformers import SentenceTransformer
    import torch
    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    print(f"\nembedding model {cfg['kb']['embedding_model']}   "
          f"cuda={torch.cuda.is_available()}")
    model = SentenceTransformer(cfg["kb"]["embedding_model"])

    # Normalised-text index over the train pool, built once. This is the
    # detector that would have caught the documented 0.987 pair for free.
    norm_index = defaultdict(list)
    exact_index = defaultdict(list)
    for lang, rows in pool.items():
        for src, tid, txt in rows:
            norm_index[(lang, norm_text(txt))].append((src, tid))
            exact_index[(lang, txt)].append((src, tid))

    train_vecs = {}
    # Per-TRAIN-ROW maximum similarity against ANY eval item, accumulated
    # across every subset. The per-eval-item argmax below answers "is this
    # eval item near something we train on"; this answers the inverse, "must
    # this train row be dropped". They are not the same list - two train rows
    # can both sit above threshold against one eval item while only one is
    # that item's argmax - and the encoder filter needs the inverse.
    train_max = {}
    report = {}

    for path in subsets:
        name = path.stem
        df = pd.read_parquet(path)
        langs = sorted({str(getattr(r, "lang", "en"))
                        for r in df.itertuples(index=False)})
        print(f"\n{'-' * 78}\n{name}   {len(df)} items   langs {langs}")

        hits_exact, hits_norm = [], []
        for r in df.itertuples(index=False):
            lang, txt = str(getattr(r, "lang", "en")), str(r.text)
            for src, tid in exact_index.get((lang, txt), []):
                hits_exact.append((str(r.id), src, tid, 1.0))
            for src, tid in norm_index.get((lang, norm_text(txt)), []):
                hits_norm.append((str(r.id), src, tid, 1.0))

        best = {}
        for lang in langs:
            rows = pool.get(lang)
            if not rows:
                print(f"  lang {lang}: no train rows, skipped")
                continue
            sub = [r for r in df.itertuples(index=False)
                   if str(getattr(r, "lang", "en")) == lang]
            ev = encode(model, [str(r.text) for r in sub],
                        f"eval_{name}_{lang}", args.batch, not args.no_cache)

            if lang not in train_vecs:
                print(f"  encoding train pool for {lang} "
                      f"({len(rows)} texts)")
                train_vecs[lang] = encode(
                    model, [t for _, _, t in rows],
                    f"train_{lang}{'_smoke' if args.smoke else ''}"
                    f"{'_' + args.only if args.only else ''}",
                    args.batch, not args.no_cache)
            tv = train_vecs[lang]

            # Chunked so the similarity matrix never materialises whole:
            # 56k x 467 is fine, 56k x 12k is not.
            top = np.full(len(sub), -1.0, dtype=np.float32)
            arg = np.zeros(len(sub), dtype=np.int64)
            if lang not in train_max:
                train_max[lang] = np.full(len(tv), -1.0, dtype=np.float32)
            for s in range(0, len(tv), args.chunk):
                sim = ev @ tv[s:s + args.chunk].T
                m = sim.max(axis=1)
                a = sim.argmax(axis=1) + s
                upd = m > top
                top[upd], arg[upd] = m[upd], a[upd]
                # Same matrix, the other axis, accumulated across subsets.
                seg = train_max[lang][s:s + sim.shape[1]]
                np.maximum(seg, sim.max(axis=0), out=seg)

            for i, r in enumerate(sub):
                best[str(r.id)] = (float(top[i]), rows[arg[i]],
                                   str(r.text))

        n_near = sum(1 for v in best.values() if v[0] >= args.near)
        n_very = sum(1 for v in best.values() if v[0] >= args.very_near)
        sims = sorted(best.items(), key=lambda kv: -kv[1][0])

        print(f"  exact text matches      {len(set(h[0] for h in hits_exact))}"
              f" item(s)")
        print(f"  normalised text matches "
              f"{len(set(h[0] for h in hits_norm))} item(s)")
        print(f"  cosine >= {args.very_near}          {n_very} item(s)")
        print(f"  cosine >= {args.near}          {n_near} item(s)")
        if sims:
            print(f"  max cosine              {sims[0][1][0]:.4f}")

        shown = [kv for kv in sims if kv[1][0] >= args.near][:args.examples]
        for item_id, (sim, (src, tid, ttxt), etxt) in shown:
            print(f"\n    sim {sim:.4f}   train[{src}] {tid}")
            print(f"      eval  {item_id}: {etxt[:100]}")
            print(f"      train {tid}: {ttxt[:100]}")

        report[name] = {
            "n_items": len(df),
            "langs": langs,
            "exact_match_items": sorted(set(h[0] for h in hits_exact)),
            "normalised_match_items": sorted(set(h[0] for h in hits_norm)),
            "near_items": sorted(k for k, v in best.items()
                                 if v[0] >= args.near),
            "very_near_items": sorted(k for k, v in best.items()
                                      if v[0] >= args.very_near),
            "max_cosine": float(sims[0][1][0]) if sims else None,
            "top": [{"eval_id": k, "cosine": round(v[0], 4),
                     "train_split": v[1][0], "train_id": v[1][1]}
                    for k, v in sims[:20]],
        }

    # WRITTEN ONLY ON A FULL, UNRESTRICTED SCAN. A --smoke run sees 2000 train
    # rows and an --only run sees one source, so either would produce a
    # partial exclusion list that looks complete - and a leakage filter that
    # silently misses most of what it should catch is worse than none.
    if not args.only and not args.smoke:
        excl = {}
        for lang, mx in train_max.items():
            rows = pool[lang]
            excl[lang] = sorted(
                {rows[i][1] for i in np.where(mx >= args.near)[0]})
        EXCL.write_text(json.dumps(
            {"threshold": args.near,
             "note": "TRAIN ids with a twin at or above threshold cosine in "
                     "ANY eval subset. Consumed by make_encoder_data; the "
                     "count goes into each checkpoint's meta.json so the "
                     "filter is provable after the fact.",
             "counts": {k: len(v) for k, v in excl.items()},
             "ids": excl}, indent=2), encoding="utf-8")
        print(f"\n  wrote {EXCL}: "
              + ", ".join(f"{k} {len(v)}" for k, v in excl.items())
              + f" train rows to exclude at >= {args.near}")

    out = (OUT.with_name(f"leakage_report_{args.only}.json") if args.only
           else OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"smoke": args.smoke, "near": args.near,
         "very_near": args.very_near, "subsets": report},
        indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}\nSUMMARY")
    print(f"  {'subset':28} {'n':>5} {'exact':>6} {'norm':>6} "
          f"{'>=' + str(args.very_near):>7} {'>=' + str(args.near):>7} "
          f"{'max':>7}")
    for name, r in report.items():
        print(f"  {name:28} {r['n_items']:>5} "
              f"{len(r['exact_match_items']):>6} "
              f"{len(r['normalised_match_items']):>6} "
              f"{len(r['very_near_items']):>7} {len(r['near_items']):>7} "
              f"{(r['max_cosine'] or 0):>7.4f}")
    print(f"\n  wrote {out}")
    if args.smoke:
        print("\n  SMOKE MODE - rerun without --smoke for the real counts.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
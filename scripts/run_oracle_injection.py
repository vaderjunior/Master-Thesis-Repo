"""
scripts/run_oracle_injection.py - the oracle upper bound for SQ3.
SPENDS API CALLS. ~363 per condition (121 items x 3 votes), roughly 10 minutes.
Run: python -m scripts.run_oracle_injection --condition A --dry-run

WHAT QUESTION THIS SETTLES. SQ3 found that corrections are retrieved and used
on the items they correct but do not transfer to unseen text. Two explanations
remain and the chapter cannot choose between them from the round data:

  (a) SELECTION - the corrections that reach the prompt are not label-relevant
      enough for the model to benefit.
  (b) GROUNDING - corrections do reach the prompt, are relevant, and the model
      does not act on them.

This condition removes (a) by force: retrieval is bypassed and a correction
carrying the item's own gold label is injected directly.

  macro-F1 rises  -> selection was the bottleneck
  macro-F1 flat   -> grounding failure, and SQ3 becomes the THIRD in-system
                     reproduction of Mohammadi et al. 2025, after
                     guide-profanity-without-target (retrieved, correct,
                     decisive, obeyed ~40% of the time) and
                     def-target_group-other (in the prompt on 6 items, used
                     on 1)

THIS IS AN ORACLE, NOT A SYSTEM. Condition A chooses the injected record using
the held-out item's GOLD label, which no deployed system could do. It is an
upper bound on what perfect correction retrieval could buy, and must be
labelled as such wherever it is reported.

CONDITION B IS WHY THIS IS WORTH RUNNING RATHER THAN JUST CONDITION A. It
injects a correction whose labels do NOT intersect the item's gold, matched in
count and position. A alone confounds "a relevant correction" with "a
correction at all"; A minus B isolates label relevance. Both conditions inject
exactly one record into exactly the slot the feedback bucket occupies, so
prompt shape is identical to the round-4 feedback arm.

EVERYTHING ELSE IS HELD CONSTANT: same KB (round-4 feedback), same prompt
version, same model, same temperature, same n_votes, same held_out items. Only
the contents of the feedback slot change.

WHY IT DOES NOT GO THROUGH classify(). classify() builds its context from
Arms, which retrieves. Injection has to replace the feedback bucket after
retrieval and before the prompt is rendered, so this script drives
build_prompt / run_once / aggregate directly. It writes ItemResults in the
same shape as run_slice1 so score_by_role and the other analysis scripts read
them unchanged - but it IS a different code path, which is one more reason to
report it as an oracle rather than as an arm.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from scripts.check_sq3_coverage import PROCESSED, local_gold_sets
from src.hsrag.classify import ItemResult, text_hash
from src.hsrag.client import make_client
from src.hsrag.parse import run_once
from src.hsrag.prompt import (PromptContext, build_prompt, prompt_version,
                              taxonomy_version)
from src.hsrag.retrieve import Hit, Retriever
from src.hsrag.schema import Result
from src.hsrag.vote import aggregate

RESULTS = Path("experiments/results")
SUBSET = "en_dev_eval_sq3_types"
GOLD_COL, DIM = "hate_types", "hate_type"
RECORDS = Path("kb") / "records_sq3_fb_r4.jsonl"
CHROMA = Path("kb") / "chroma_sq3_fb_r4"
SEED = 42


def feedback_records() -> list:
    return [r for r in
            (json.loads(l) for l in
             RECORDS.read_text(encoding="utf-8").splitlines() if l.strip())
            if r["kind"] == "feedback"]


def as_hit(rec: dict) -> Hit:
    """A KB record as a retrieval Hit, so PromptContext and _render_gold treat
    it exactly as they would a retrieved one. via='oracle' keeps the injection
    visible in the stored provenance."""
    m = rec.get("meta", {})
    return Hit(id=rec["id"], kind=rec["kind"], text=rec["text"],
               meta={**m, "kind": rec["kind"], "lang": rec["lang"],
                     "source": rec["source"], "dimension": rec["dimension"],
                     "label": rec["label"]},
               score=0.0, via="oracle")


def choose(recs: list, gold: set, condition: str, rng: random.Random):
    """Pick the injected record. A: labels intersect the item's gold.
    B: they do not. Deterministic given the seed and the item order."""
    if condition == "A":
        pool = [r for r in recs if set(r["meta"]["hate_types"] or []) & gold]
    else:
        pool = [r for r in recs
                if not (set(r["meta"]["hate_types"] or []) & gold)]
    return rng.choice(pool) if pool else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["A", "B"], required=True,
                    help="A = label-matched (oracle), B = label-mismatched")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rep", type=int, default=1,
                    help="replicate index. The injected records are chosen "
                         "from a fixed seed and are IDENTICAL across "
                         "replicates, so only decoding varies - these are "
                         "true replicates of one condition, not a broader "
                         "sample over which record gets injected.")
    ap.add_argument("--dry-run", action="store_true",
                    help="MockClient, no API calls")
    args = ap.parse_args()

    cfg_all = yaml.safe_load(
        Path("config/config.yaml").read_text(encoding="utf-8"))
    api, ccfg, kb = cfg_all["api"], cfg_all["classify"], cfg_all["kb"]
    n_votes = ccfg["n_votes_dev"]
    pinned = "Qwen/Qwen3-VL-32B-Instruct"
    provider = "mock" if args.dry_run else "peasec"

    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    items = df[df["sq3_role"] == "held_out"].reset_index(drop=True)
    gold = local_gold_sets(items, GOLD_COL)

    # k_examples_feedback = 0: retrieval supplies definitions, guidelines and
    # regular examples exactly as in the round-4 run, and the feedback slot is
    # filled by injection instead. That keeps prompt shape identical to the
    # feedback arm while removing retrieval from the one bucket under test.
    retriever = Retriever(chroma_path=CHROMA, records_path=RECORDS,
                          model_name=kb["embedding_model"],
                          cfg={**cfg_all["retrieval"],
                               "k_examples_feedback": 0})
    kb_version = retriever.col.metadata.get("kb_version")
    recs = feedback_records()

    tag = "mock" if args.dry_run else "live"
    suffix = "" if args.rep == 1 else f"_rep{args.rep}"
    out_path = RESULTS / f"oracle_{args.condition}{suffix}_{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and "_manifest" not in line:
                done.add(json.loads(line)["item_id"])
        print(f"resuming: {len(done)} items already complete")

    rng = random.Random(SEED)
    todo = []
    for r in items.itertuples(index=False):
        iid = str(r.id)
        if iid in gold and iid not in done:
            pick = choose(recs, gold[iid], args.condition, rng)
            if pick is not None:
                todo.append((r, pick))

    print(f"\n{'=' * 74}\nORACLE INJECTION, condition {args.condition}")
    print(f"  {'A = label-matched (upper bound)' if args.condition == 'A' else 'B = label-mismatched (control)'}")
    print(f"  KB OVERRIDE: {CHROMA} -> kb_version {kb_version}")
    print(f"  prompt {prompt_version()}   model {pinned}   T "
          f"{api['temperature']}   n_votes {n_votes}")
    print(f"  {len(todo)} items x {n_votes} votes = {len(todo) * n_votes} calls")
    if not args.dry_run and input("proceed? [y/N] ").strip().lower() != "y":
        return

    lock = threading.Lock()
    local = threading.local()

    def client():
        if not hasattr(local, "c"):
            if provider == "mock":
                local.c = make_client("mock")
            else:
                p = api[provider]
                local.c = make_client(
                    provider, [pinned], temperature=api["temperature"],
                    timeout=api["timeout_seconds"],
                    max_tokens=api.get("max_tokens"), allow_fallback=False,
                    url_env=p["url_env"], token_env=p["token_env"])
        return local.c

    def work(job):
        row, pick = job
        iid, text, lang = str(row.id), str(row.text), str(row.lang)
        with lock:                       # retrieval is not thread-safe
            res = retriever.retrieve(text, lang)
        res.feedback = [as_hit(pick)]
        ctx = PromptContext.from_retrieval(res)
        messages = build_prompt(text, lang, ctx)

        out = ItemResult(
            item_id=iid, text_hash=text_hash(text), lang=lang, arm="rag",
            prompt_version=prompt_version(),
            taxonomy_version=taxonomy_version(), kb_version=kb_version,
            retrieval_config=dict(retriever.cfg),
            retrieved={"definitions": [h.id for h in ctx.definitions],
                       "guidelines": [h.id for h in ctx.guidelines],
                       **{name: [h.id for h in hits]
                          for name, hits in ctx.example_groups},
                       "injected": [pick["id"]]},
            n_runs=n_votes, timestamp=time.time(), workers=args.workers,
            temperature=api["temperature"])

        runs = []
        for _ in range(n_votes):
            r = run_once(client(), messages,
                         max_repairs=ccfg["max_repair_retries"])
            runs.append(r)
            out.raw_runs.append(r.raw_outputs)
            out.latencies.append(round(r.latency_s, 2))
            out.repairs += r.repairs
            out.normalisations += r.normalisations
            out.gate_normalised += int(r.gate_normalised)
            out.parse_failures += int(r.parse_failure)
            out.run_predictions.append(
                r.result.model_dump() if r.result is not None else None)

        if not args.dry_run and client().active_model != pinned:
            raise RuntimeError(f"expected {pinned}, got {client().active_model}")
        out.active_model = client().active_model

        v = aggregate(runs)
        out.result = v.result.model_dump() if v.result is not None else None
        out.uncertain, out.n_valid, out.agreement = (
            v.uncertain, v.n_valid, v.agreement)
        return out

    t0, ok = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(work, todo):
                with lock:
                    f.write(r.to_json() + "\n")
                    f.flush()
                    ok += 1
                    if ok % 20 == 0 or ok == len(todo):
                        el = time.time() - t0
                        print(f"  {ok}/{len(todo)}  {el / 60:.1f} min")

    print(f"\n  wrote {out_path}")
    # score_by_role appends _live itself, so the stem is passed without it.
    print(f"  score with: python -m scripts.score_by_role --role held_out "
          f"--runs oracle_{args.condition} sq3_r4_fb sq3_r4_ctl")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
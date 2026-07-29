"""
scripts/run_slice1.py - Slice 1: the first real no-RAG vs RAG comparison.

THREE ARMS, NOT TWO. zero_shot vs rag alone cannot separate "retrieval helped"
from "having examples at all helped". few_shot supplies the same NUMBER of
examples, statically sampled, no retrieval, no definitions, no guidelines. So:
  few_shot - zero_shot  = the value of examples
  rag      - few_shot   = the value of RETRIEVAL, which is the actual question

RESUMABLE. Results append to JSONL as they complete and completed (item, arm)
pairs are skipped on restart. A ~90 minute run on shared university
infrastructure will be interrupted; losing it whole is not acceptable.

SUCCESS IS NOT "RAG WINS". Success is a real, attributed, reproducible
comparison on identical items with a pinned model. If RAG loses, three
suspects are already documented: Finding E (retrieval supplies only hateful
evidence for non-hateful input), definition wording (Finding G), and the
German-inert hybrid.

  python -m scripts.run_slice1 --dry-run          # MockClient, no API calls
  python -m scripts.run_slice1 --n 10             # live smoke test
  python -m scripts.run_slice1 --n 150            # the real thing
  python -m scripts.run_slice1 --score-only       # rescore existing results
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor
# Scoring lives in ONE place. This script had its own copy from Phase 5, which
# silently missed the single-class fix and reported a meaningless 0.500 gate
# macro-F1 on the all-hateful targets subset.
from src.hsrag.metrics import score_all
from scripts.score_run import report
from src.hsrag.classify import Arms, classify
from src.hsrag.client import MockClient, TUDaGPTClient
from src.hsrag.retrieve import Retriever

RESULTS = Path("experiments/results")
ARMS = ["zero_shot", "few_shot", "rag"]






# --- runner ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="take run settings from a manifest; "
                                       "overrides the CLI defaults below")
    ap.add_argument("--subset", default="en_dev_eval_main")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel API calls; each worker gets its own client")
    ap.add_argument("--dry-run", action="store_true", help="MockClient, no API")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_all = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    kb, ccfg, api = cfg_all["kb"], cfg_all["classify"], cfg_all["api"]
    
    # An experiment IS its manifest. When one is given it overrides the CLI
    # defaults, so a run cannot be configured half from a file and half from
    # shell history - which is exactly what made Slice 1 hard to describe
    # after the fact.
    mf = None
    if args.manifest:
        from src.hsrag.manifest import load as load_manifest, resolve
        mf = load_manifest(args.manifest)
        args.subset, args.arms, args.workers = mf.subset, mf.arms, mf.workers
        if mf.limit:
            args.n = mf.limit
        ccfg = {**ccfg,
                "n_votes_dev": mf.n_votes,
                "max_repair_retries": mf.max_repair_retries,
                "pinned_model": mf.pinned_model or ccfg["pinned_model"]}

    df = pd.read_parquet(Path("data/processed") / f"{args.subset}.parquet").head(args.n)
    tag = "mock" if args.dry_run else "live"
    out_path = Path(args.out or RESULTS / f"slice1_{args.subset}_{tag}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "_manifest" in r:
                    continue
                done.add((r["item_id"], r["arm"]))
        print(f"resuming: {len(done)} (item, arm) pairs already complete")

    if not args.score_only:
        retriever = Retriever(chroma_path=Path(kb["chroma_path"]),
                              records_path=Path(kb["records_path"]),
                              model_name=kb["embedding_model"],
                              cfg=dict(cfg_all["retrieval"]))
        arms = Arms(retriever=retriever, records_path=Path(kb["records_path"]),
                    k_examples=cfg_all["retrieval"]["k_examples"],
                    seed=ccfg["fewshot_seed"])
        kb_version = retriever.col.metadata.get("kb_version")
        
        if mf is not None and not out_path.exists():
            # Resolved manifest as the first line of the results file, so the
            # result set describes itself even if the YAML or config.yaml is
            # edited later. Only on creation - a resumed run must not get a
            # second header.
            out_path.write_text(resolve(mf, retriever).header() + "\n",
                                encoding="utf-8")

        n_votes = ccfg["n_votes_dev"]
        # ITEM-MAJOR, not arm-major: a run that dies at 60% then covers all
        # three arms on the items it finished, instead of two complete arms
        # and nothing to compare them against.
        todo = [(row, arm) for row in df.itertuples(index=False)
                for arm in args.arms
                if (str(row.id), arm) not in done]
        calls = len(todo) * n_votes
        # MEASURED WALL-CLOCK THROUGHPUT, not per-worker latency. Dividing a
        # per-worker latency by the worker count double-counts the parallelism
        # that is already inside these numbers.
        #   sequential:  29.8 s/call
        #   4 workers:   13.2 s/call (Slice 1, 744 calls / 164 min)
        #                16.8 s/call (targets_dev, 498 calls / 140 min)
        # Throughput improves only ~1.2x from 1 to 4 workers because the
        # server caps total throughput rather than queueing, so the worker
        # count barely enters the estimate at all.
        wall_s = 30 if args.workers == 1 else 17
        est = calls * wall_s / 60
        print(f"\n{len(todo)} item-arm pairs x {n_votes} votes = {calls} calls")
        print(f"  ~{est:.0f} min with {args.workers} worker(s), "
              f"assuming parallelism scales (it may not)")

        if not args.dry_run and input("proceed? [y/N] ").strip().lower() != "y":
            return

        # Each worker gets its OWN client: active_model is mutable state, and a
        # shared client would race on exactly the stamp the attribution
        # assertion below depends on.
        local = threading.local()

        def get_client():
            if not hasattr(local, "client"):
                local.client = (
                    MockClient() if args.dry_run else
                    # allow_fallback=False: a silent substitution mid-run would
                    # make every number after it unattributable to a model.
                    TUDaGPTClient(models=[ccfg["pinned_model"]],
                                  temperature=api["temperature"],
                                  timeout=api["timeout_seconds"],
                                  allow_fallback=False))
            return local.client

        def work(job):
            row, arm = job
            try:
                return classify(
                    item_id=str(row.id), text=row.text, lang=row.lang, arm=arm,
                    client=get_client(), arms=arms, n_votes=n_votes,
                    max_repairs=ccfg["max_repair_retries"],
                    pinned_model=None if args.dry_run else ccfg["pinned_model"],
                    kb_version=kb_version, workers=args.workers)
            except RuntimeError as e:
                # A transient API failure must not end a multi-hour run.
                # Nothing is written for this pair, so a rerun retries it.
                return (row.id, arm, str(e))

        write_lock = threading.Lock()
        state = {"ok": 0, "failed": 0, "consecutive": 0}
        t0 = time.time()

        with open(out_path, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for res in ex.map(work, todo):
                    with write_lock:
                        if isinstance(res, tuple):
                            state["failed"] += 1
                            state["consecutive"] += 1
                            print(f"  FAILED {res[0]} [{res[1]}] "
                                  f"({state['consecutive']} in a row): "
                                  f"{res[2][:120]}")
                        else:
                            state["consecutive"] = 0
                            state["ok"] += 1
                            f.write(res.to_json() + "\n")
                            f.flush()      # survive an interrupt
                        n = state["ok"] + state["failed"]
                        if n % 10 == 0 or n == len(todo):
                            el = time.time() - t0
                            rate = el / max(state["ok"], 1) / n_votes * args.workers
                            print(f"  {n}/{len(todo)}  {el / 60:.1f} min elapsed, "
                                  f"~{el / n * (len(todo) - n) / 60:.1f} min left "
                                  f"({rate:.1f} s/call, {state['failed']} failed)")

        if state["failed"]:
            print(f"\n{state['failed']} pairs failed and were not written. "
                  f"Rerun the same command to retry only those.")

    # The first line is the resolved manifest, not a result.
    results = [r for r in
               (json.loads(l) for l in
                out_path.read_text(encoding="utf-8").splitlines() if l.strip())
               if "_manifest" not in r]
    scores = score_all(df, results,
                       gate_mapping=(mf.gate_mapping if mf else "both"))
    report(scores)

    json_path = out_path.with_suffix(".metrics.json")
    json_path.write_text(json.dumps(scores, indent=2, default=str),
                         encoding="utf-8")
    print(f"\nwrote {out_path}\nwrote {json_path}")

    if not args.dry_run:
        models = Counter(r["active_model"] for r in results)
        print(f"models that answered: {dict(models)}")
        assert set(models) == {ccfg["pinned_model"]}, "unattributable results"

if __name__ == "__main__":
    main()
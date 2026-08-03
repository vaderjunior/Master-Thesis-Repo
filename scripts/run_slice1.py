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
from src.hsrag.client import make_client
from src.hsrag.retrieve import Retriever

RESULTS = Path("experiments/results")
ARMS = ["zero_shot", "few_shot", "rag"]
PROCESSED = Path("data/processed")

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
        # A manifest with no limit means the whole subset, not the CLI
        # default. legal_dev_peasec silently ran 150 of 175 items because
        # --n defaults to 150 and an absent limit left it in place.
        args.n = mf.limit if mf.limit else len(
            pd.read_parquet(PROCESSED / f"{mf.subset}.parquet"))
        ccfg = {**ccfg,
                "n_votes_dev": mf.n_votes,
                "max_repair_retries": mf.max_repair_retries,
                "pinned_model": mf.pinned_model or ccfg["pinned_model"]}
        # The manifest's temperature must reach the client, or a temperature
        # ablation runs every arm at the config default and reports a null
        # result for the wrong reason.
        api = {**api, "temperature": mf.temperature or api["temperature"]}

    df = pd.read_parquet(Path("data/processed") / f"{args.subset}.parquet").head(args.n)
    tag = "mock" if args.dry_run else "live"
    # Manifest name in the filename, not just the subset. The bake-off runs
    # three manifests over en_dev_eval_sq1_tune, and a subset-only name would
    # make them share one file - the second and third would see the first's
    # results as "already complete" and silently do nothing.
    stem = f"{mf.name}_{tag}" if mf else f"slice1_{args.subset}_{tag}"
    out_path = Path(args.out or RESULTS / f"{stem}.jsonl")
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
        # A manifest may point at a PREVIOUS knowledge base, so that a KB
        # comparison changes one variable rather than two. Both paths move
        # together: Chroma serves the dense channel, the jsonl serves BM25
        # and the few_shot sample.
        chroma_dir = Path(mf.chroma_path) if (mf and mf.chroma_path) \
            else Path(kb["chroma_path"])
        records_file = Path(mf.records_path) if (mf and mf.records_path) \
            else Path(kb["records_path"])
        retriever = Retriever(chroma_path=chroma_dir,
                              records_path=records_file,
                              model_name=kb["embedding_model"],
                              cfg=dict(mf.retrieval if (mf and mf.retrieval)
                                       else cfg_all["retrieval"]))
        arms = Arms(retriever=retriever, records_path=records_file,
                    k_examples=cfg_all["retrieval"]["k_examples"],
                    seed=ccfg["fewshot_seed"])
        kb_version = retriever.col.metadata.get("kb_version")
        if mf and mf.chroma_path:
            print(f"  KB OVERRIDE: {chroma_dir} -> kb_version {kb_version}")
        
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
        # MEASURED WALL-CLOCK THROUGHPUT per provider, not per-worker latency.
        # Dividing a per-worker latency by the worker count double-counts the
        # parallelism already inside these numbers.
        #
        # tudagpt: 29.8 s/call sequential; 13.2 s/call at 4 workers (Slice 1,
        #   744 calls / 164 min) and 16.8 s/call (targets_dev, 498 / 140).
        #   The server caps total throughput rather than queueing, so workers
        #   buy only ~1.2x, and 8 workers triggers 422 on about half of all
        #   requests.
        # peasec: continuous batching over vLLM. Median latency stays flat
        #   under load (1.1 -> 2.2 s from 1 to 16 workers) while throughput
        #   scales 5.7x: 1.1 / 0.3 / 0.3 / 0.2 s/call at 1 / 4 / 8 / 16.
        #   Those were small JSON prompts with no retrieved context, so 3.0
        #   s/call is a deliberately conservative planning figure until a real
        #   classification run measures it.
        provider = ("mock" if args.dry_run
                    else (mf.provider if mf else ccfg["provider"]))
        if provider == "mock":
            wall_s = 0.01
        elif provider == "peasec":
            wall_s = 8.0 if args.workers == 1 else 3.0
        else:
            wall_s = 30.0 if args.workers == 1 else 17.0
        est = calls * wall_s / 60
        print(f"\n{len(todo)} item-arm pairs x {n_votes} votes = {calls} calls")
        print(f"  provider {provider}, {args.workers} worker(s), "
              f"~{wall_s:g} s/call measured -> ~{est:.0f} min")

        if not args.dry_run and input("proceed? [y/N] ").strip().lower() != "y":
            return

        # Each worker gets its OWN client: active_model is mutable state, and a
        # shared client would race on exactly the stamp the attribution
        # assertion below depends on.
        local = threading.local()

        def get_client():
            if not hasattr(local, "client"):
                if provider == "mock":
                    local.client = make_client("mock")
                else:
                    p = api[provider]
                    local.client = make_client(
                        provider, [ccfg["pinned_model"]],
                        temperature=api["temperature"],
                        timeout=api["timeout_seconds"],
                        max_tokens=api.get("max_tokens"),
                        # allow_fallback=False: a silent substitution mid-run
                        # would make every number after it unattributable.
                        allow_fallback=False,
                        url_env=p["url_env"], token_env=p["token_env"])
            return local.client

        def work(job):
            row, arm = job
            try:
                return classify(
                    item_id=str(row.id), text=row.text, lang=row.lang, arm=arm,
                    client=get_client(), arms=arms, n_votes=n_votes,
                    max_repairs=ccfg["max_repair_retries"],
                    pinned_model=None if args.dry_run else ccfg["pinned_model"],
                    kb_version=kb_version, workers=args.workers, temperature=api["temperature"])
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
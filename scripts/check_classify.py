"""
scripts/check_classify.py - end-to-end smoke test on MockClient. NO API CALLS.

  python -m scripts.check_classify --arm rag --n 20
  python -m scripts.check_classify --arm zero_shot --broken
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.hsrag.classify import ARMS, Arms, classify
from src.hsrag.client import MockClient
from src.hsrag.retrieve import Retriever

REQUIRED_STAMPS = ["item_id", "text_hash", "lang", "arm", "active_model",
                   "prompt_version", "n_runs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="rag", choices=list(ARMS) + ["all"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--subset", default="en_dev_eval_main")
    ap.add_argument("--broken", action="store_true",
                    help="MockClient fails its first call, to exercise repair")
    args = ap.parse_args()

    cfg_all = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    kb, ccfg = cfg_all["kb"], cfg_all["classify"]

    df = pd.read_parquet(Path("data/processed") / f"{args.subset}.parquet").head(args.n)

    retriever = Retriever(chroma_path=Path(kb["chroma_path"]),
                          records_path=Path(kb["records_path"]),
                          model_name=kb["embedding_model"],
                          cfg=dict(cfg_all["retrieval"]))
    kb_version = retriever.col.metadata.get("kb_version")

    arms = Arms(retriever=retriever, records_path=Path(kb["records_path"]),
                k_examples=cfg_all["retrieval"]["k_examples"],
                seed=ccfg["fewshot_seed"])

    for arm in (list(ARMS) if args.arm == "all" else [args.arm]):
        client = MockClient(broken=args.broken)
        results = []
        for row in df.itertuples(index=False):
            client.reset()
            results.append(classify(
                item_id=str(row.id), text=row.text, lang=row.lang, arm=arm,
                client=client, arms=arms, n_votes=ccfg["n_votes_dev"],
                max_repairs=ccfg["max_repair_retries"], kb_version=kb_version))

        missing = [f for r in results for f in REQUIRED_STAMPS
                   if getattr(r, f) in (None, "")]
        print(f"\n=== {arm} : {len(results)} items ===")
        print(f"  missing stamps      {sorted(set(missing)) or 'none'}")
        print(f"  kb_version          {results[0].kb_version}")
        print(f"  prompt_version      {results[0].prompt_version}")
        print(f"  retrieved buckets   "
              f"{ {k: len(v) for k, v in results[0].retrieved.items()} }")
        print(f"  raw runs per item   {len(results[0].raw_runs)}")
        print(f"  predictions stored  {len(results[0].run_predictions)}")
        print(f"  repairs total       {sum(r.repairs for r in results)}")
        print(f"  parse failures      {sum(r.parse_failures for r in results)}")
        print(f"  uncertain           {sum(r.uncertain for r in results)}")
        print(f"  hate=true predicted {sum(1 for r in results if r.result and r.result['hate'])}")

        out = Path("experiments/results") / f"check_classify_{arm}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(r.to_json() for r in results), encoding="utf-8")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
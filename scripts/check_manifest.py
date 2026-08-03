"""
scripts/check_manifest.py - validate and cost a manifest. No API calls.

  python -m scripts.check_manifest targets_dev
  python -m scripts.check_manifest targets_dev --seconds-per-call 25
  python -m scripts.check_manifest --all
"""

import argparse
from pathlib import Path

import yaml

from src.hsrag.manifest import (DEFAULT_SECONDS_PER_CALL, MANIFESTS, describe,
                                load, resolve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seconds-per-call", type=float,
                    default=DEFAULT_SECONDS_PER_CALL)
    args = ap.parse_args()

    names = ([p.stem for p in sorted(MANIFESTS.glob("*.yaml"))]
             if args.all else [args.name])

    # The retriever is loaded once, and only if some manifest needs a
    # kb_version stamp - it costs a model load.
    retriever = None
    if any("rag" in load(n).arms for n in names):
        cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
        from src.hsrag.retrieve import Retriever
        kb = cfg["kb"]
        # Build from the FIRST manifest that overrides the KB, if any, so the
        # stamped kb_version describes what the run will actually query rather
        # than what config.yaml happens to point at.
        first = next((load(n) for n in names if load(n).chroma_path), None)
        retriever = Retriever(
            chroma_path=Path(first.chroma_path if first else kb["chroma_path"]),
            records_path=Path(first.records_path if first else kb["records_path"]),
            model_name=kb["embedding_model"],
            cfg=dict(cfg["retrieval"]))

    total = 0
    for name in names:
        m = resolve(load(name), retriever)
        print("\n" + "=" * 72)
        print(describe(m, args.seconds_per_call))
        total += m.calls

    if len(names) > 1:
        hrs = total * args.seconds_per_call / 4 / 3600
        print("\n" + "=" * 72)
        print(f"TOTAL {total} calls  ~{hrs:.1f} h at 4 workers")


if __name__ == "__main__":
    main()
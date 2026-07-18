"""
Retrieval microscope. Shows the three buckets for a query.

    python -m scripts.debug_retrieval "some text" --lang en
    python -m scripts.debug_retrieval          # runs the 6 standard probes
"""

import argparse
from pathlib import Path

import yaml

from src.hsrag.retrieve import Retriever

config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
kb_cfg = config["kb"]
ret_cfg = config["retrieval"]

parser = argparse.ArgumentParser()
parser.add_argument("text", nargs="?", default=None)
parser.add_argument("--lang", default="en", choices=["en", "de"])
parser.add_argument("--strategy", default=None,
                    choices=["dense", "bm25", "hybrid"],
                    help="override config strategy")
args = parser.parse_args()

if args.strategy:
    ret_cfg = {**ret_cfg, "strategy": args.strategy}

if args.strategy:
    ret_cfg = {**ret_cfg, "strategy": args.strategy}

retriever = Retriever(
    chroma_path=Path(kb_cfg["chroma_path"]),
    records_path=Path(kb_cfg["records_path"]),
    model_name=kb_cfg["embedding_model"],
    cfg=ret_cfg,
)


def show(text: str, lang: str):
    result = retriever.retrieve(text, lang=lang)
    print(f"\n{'='*72}\nQUERY ({lang}): {text}\n{'='*72}")

    for bucket_name, hits in [
        ("DEFINITIONS", result.definitions),
        ("GUIDELINES", result.guidelines),
        ("EXAMPLES", result.examples),
    ]:
        print(f"\n  --- {bucket_name} ({len(hits)}) ---")
        for i, h in enumerate(hits, 1):
            if h.kind == "example":
                tag = (f"gate={h.meta.get('gate')} "
                       f"tg={h.meta.get('target_groups')}")
                if h.meta.get("illustrative_only"):
                    tag += f" [LEGAL §{h.meta.get('stgb')}]"
            else:
                tag = f"{h.meta.get('dimension') or ''}/{h.meta.get('label') or ''}"
            print(f"  [{i}] {h.score:.3f} ({h.via}) {h.id}")
            if h.meta.get("_ranks"):
                print(f"      ranks: {h.meta['_ranks']}")
            print(f"      {tag}")
            print(f"      {h.text[:88]}")


PROBES = [
    ("women are too stupid to vote", "en"),
    ("muslims are terrorists and should be banned", "en"),
    ("Ausländer raus, wir wollen euch hier nicht", "de"),
    ("you're such an idiot, get lost", "en"),
    ("he called me the n-word and I was shocked", "en"),
    ("the weather is lovely for a walk today", "en"),
]

if args.text:
    show(args.text, args.lang)
else:
    for probe, lang in PROBES:
        show(probe, lang)
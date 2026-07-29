"""
scripts/check_mmr.py - measure the redundancy MMR removes.

Runs each probe twice (use_mmr False, then True) under strategy=dense and
reports, over the returned examples:
  - number of pairs with cosine > 0.9   (near-duplicates)
  - mean pairwise cosine                (overall redundancy)

Use strategy=dense: under hybrid, MMR reorders the dense channel and RRF then
fuses, so the effect on the final bucket is indirect and hard to read.

Run:  python -m scripts.check_mmr
      python -m scripts.check_mmr --probe "women are too stupid to vote" --lang en
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.hsrag.retrieve import Retriever

THRESHOLD = 0.9

PROBES = [
    ("women are too stupid to vote", "en"),
    ("muslims are terrorists and should be banned", "en"),
    ("Ausländer raus, wir wollen euch hier nicht", "de"),
    ("you're such an idiot, get lost", "en"),
    ("he called me the n-word and I was shocked", "en"),
    ("the weather is lovely for a walk today", "en"),
]


def redundancy(model, hits):
    """(n_pairs_over_threshold, mean_pairwise_cos, max_pairwise_cos)."""
    if len(hits) < 2:
        return 0, 0.0, 0.0
    V = model.encode([h.text for h in hits], normalize_embeddings=True)
    sims = V @ V.T
    pairs = [sims[i][j] for i, j in itertools.combinations(range(len(hits)), 2)]
    n_dup = sum(1 for s in pairs if s > THRESHOLD)
    return n_dup, float(np.mean(pairs)), float(np.max(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=None)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    cfg_all = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    kb_cfg = cfg_all["kb"]
    model_name = kb_cfg["embedding_model"]

    probes = [(args.probe, args.lang)] if args.probe else PROBES
    model = SentenceTransformer(model_name)

    print(f"\nnear-duplicate threshold: cosine > {THRESHOLD}")
    print(f"{'probe':45} {'MMR':5} {'n':>2} {'dup':>4} {'mean':>6} {'max':>6}")
    print("-" * 76)

    for use_mmr in (False, True):
        cfg = dict(cfg_all["retrieval"])
        cfg["strategy"] = "dense"
        cfg["use_mmr"] = use_mmr

        r = Retriever(
            chroma_path=Path(kb_cfg["chroma_path"]),
            records_path=Path(kb_cfg["records_path"]),
            model_name=model_name,
            cfg=cfg,
        )

        for text, lang in probes:
            ex = r.retrieve(text, lang).examples
            n_dup, mean_s, max_s = redundancy(model, ex)
            label = "on" if use_mmr else "off"
            print(f"{text[:44]:45} {label:5} {len(ex):>2} {n_dup:>4} "
                  f"{mean_s:>6.3f} {max_s:>6.3f}")
        print()


if __name__ == "__main__":
    main()
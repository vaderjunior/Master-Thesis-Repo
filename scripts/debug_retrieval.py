"""
Debug retrieval: embed hand-written probes, show the top-k KB hits.
Phase 4 will make this a proper CLI; for now it's for eyeballing sanity.
"""

import json
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

from src.hsrag.kb import load_collection

config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
kb_cfg = config["kb"]

model = SentenceTransformer(kb_cfg["embedding_model"])
col = load_collection(Path(kb_cfg["chroma_path"]), kb_cfg["embedding_model"])


def unsentinel(v):
    """Reverse the KB metadata encoding for display."""
    if v == "__none__" or v == "":
        return None
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def retrieve(query: str, k: int = 5):
    q = model.encode([query], normalize_embeddings=True)
    hits = col.query(query_embeddings=q.tolist(), n_results=k)

    print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}")
    for i in range(len(hits["ids"][0])):
        meta = hits["metadatas"][0][i]
        dist = hits["distances"][0][i]
        doc = hits["documents"][0][i]

        kind = meta["kind"]
        label = unsentinel(meta.get("label"))
        lang = meta["lang"]

        # for examples, show the gold labels; for defs/guides, the dim/label
        if kind == "example":
            tag = f"gate={unsentinel(meta.get('gate'))} " \
                  f"tg={unsentinel(meta.get('target_groups'))}"
        else:
            tag = f"{meta.get('dimension','')}/{label}"

        print(f"\n  [{i+1}] {kind:10} ({lang}) dist={dist:.3f}  {tag}")
        print(f"      {doc[:90]}")


PROBES = [
    "women are too stupid to vote",                    # misogyny
    "muslims are terrorists and should be banned",      # religious hate
    "Ausländer raus, wir wollen euch hier nicht",       # German (foreigners out)
    "you're such an idiot, get lost",                   # offensive, not hate
    "he called me the n-word and I was shocked",        # quoted slur, not hate
    "the weather is lovely for a walk today",           # neutral
]

for probe in PROBES:
    retrieve(probe)
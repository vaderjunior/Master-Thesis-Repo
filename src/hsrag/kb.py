"""
Knowledge base: build a ChromaDB index from kb/records.jsonl and load it back.

DESIGN
  records.jsonl is the SOURCE OF TRUTH. The Chroma collection is a disposable
  index rebuilt from it (build() deletes and re-creates - idempotent).

THREE CHROMA GOTCHAS (all handled here):
  1. We pass our OWN BGE-M3 embeddings via embeddings=. Chroma never embeds
     for us (it would use an unpinned default model).
  2. Collection is created with cosine space, not the L2 default.
  3. Metadata must be scalar. The example gold-label lists (target_groups,
     hate_types) are JSON-stringified into metadata, parsed back on read.

VERSION HASH
  kb_version = sha256 of records.jsonl. Stored in collection metadata.
  Later: experiment manifests record it, the response cache keys on it, so a
  KB edit auto-invalidates the right cache entries. embedding_model is also
  stored, and load_collection asserts it matches - guards against the silent
  failure of querying with a different model than you ingested with.
"""

import hashlib
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "hate_kb"

# metadata keys whose values are lists and must be JSON-stringified
LIST_META_KEYS = ("target_groups", "hate_types")


def _kb_hash(records_path: Path) -> str:
    return hashlib.sha256(records_path.read_bytes()).hexdigest()[:16]

def unsentinel(value):
    """Reverse the Chroma metadata encoding.

    CONTRACT: every reader of KB metadata goes through this. Chroma can't
    store None or lists, so build() writes "__none__" for None, "" for null
    dimension/label, and JSON strings for lists. If a Hit.meta ever shows
    "__none__" or a raw JSON string, some code path bypassed this function.

    The None / [] / ["race"] three-way distinction from Phase 2 depends on it.
    """
    if value == "__none__" or value == "":
        return None
    if isinstance(value, str) and value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def unsentinel_meta(meta: dict) -> dict:
    """Apply unsentinel to every value in a Chroma metadata dict."""
    return {k: unsentinel(v) for k, v in meta.items()}

def _flatten_meta(record: dict) -> dict:
    """Turn a KB record into Chroma-safe scalar metadata.
    Lists -> JSON strings. None -> a sentinel we can distinguish on read."""
    meta = {
        "kind": record["kind"],
        "lang": record["lang"],
        "source": record["source"],
        # dimension/label are None for some kinds; store "" and treat as None
        "dimension": record["dimension"] or "",
        "label": record["label"] or "",
    }

    # merge the example-specific meta (gate, severity, illustrative_only, stgb)
    for k, v in record.get("meta", {}).items():
        if k in LIST_META_KEYS:
            # None stays None-ish via a sentinel; lists -> JSON
            meta[k] = json.dumps(v) if v is not None else "__none__"
        elif v is None:
            meta[k] = "__none__"
        else:
            meta[k] = v

    return meta


def build(records_path: Path, chroma_path: Path, model_name: str) -> None:
    """Delete + rebuild the Chroma collection from the JSONL. Idempotent."""
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(records)} records from {records_path}")

    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in records]
    print(f"Encoding {len(texts)} texts with {model_name}...")
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(chroma_path))

    # idempotent: drop the collection if it already exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet

    col = client.create_collection(
        COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",          # gotcha 2
            "embedding_model": model_name,   # for the load-time assert
            "kb_version": _kb_hash(records_path),
        },
    )

    col.add(
        ids=[r["id"] for r in records],
        embeddings=embeddings,               # gotcha 1: our own vectors
        documents=texts,
        metadatas=[_flatten_meta(r) for r in records],
    )

    print(f"Ingested {col.count()} records.")
    print(f"kb_version: {_kb_hash(records_path)}")


def load_collection(chroma_path: Path, model_name: str):
    """Load the collection and assert it was built with the same model."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_collection(COLLECTION_NAME)

    ingested_model = col.metadata.get("embedding_model")
    if ingested_model != model_name:
        raise RuntimeError(
            f"Model mismatch: collection was built with '{ingested_model}', "
            f"but you're querying with '{model_name}'. Rebuild or fix config."
        )

    return col
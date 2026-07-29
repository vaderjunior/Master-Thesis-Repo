"""
Retrieval: turn a query text into three budgeted buckets of KB knowledge.

WHY THREE BUCKETS, NOT ONE RANKED LIST
  Phase 3 found that naive dense retrieval over the mixed KB buries
  definitions and guidelines under examples. Cause: examples are raw
  social-media text, lexically and stylistically close to raw-text queries;
  definitions/guidelines are abstract descriptive language, far away in
  embedding space. They lose the nearest-neighbour race every time.

  The fix is structural, not a better ranker: the prompt has a section for
  definitions, one for guidelines, one for examples. Retrieval's job is to
  FILL those sections, not to produce a leaderboard. So we run one filtered
  query per kind with its own budget, and each kind is guaranteed its slots.

  Structured prompts need structured retrieval.
"""

from dataclasses import dataclass, field

from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.hsrag.kb import load_collection, unsentinel_meta


import json
import re
from rank_bm25 import BM25Okapi


@dataclass
class Hit:
    """One retrieved KB record."""
    id: str
    kind: str            # definition | guideline | example
    text: str
    meta: dict           # post-unsentinel: real None / real lists restored
    score: float
    via: str             # "dense" | "bm25" | "rrf" - provenance for debugging
    vec: list[float] | None = None   # dense candidate vector, MMR only.
                                     # Nulled before return; never reaches the prompt.


@dataclass
class RetrievalResult:
    """What the prompt builder (Phase 5) consumes. One bucket per prompt
    section."""
    definitions: list[Hit] = field(default_factory=list)
    guidelines: list[Hit] = field(default_factory=list)
    examples: list[Hit] = field(default_factory=list)

    def all_hits(self) -> list[Hit]:
        """Flat view, for logging and the debug CLI."""
        return self.definitions + self.guidelines + self.examples

    def __len__(self) -> int:
        return len(self.all_hits())
    
    


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Punctuation-splitting means 'n-word'
    becomes ['n','word'], which is what we want for matching.

    LIMITATION (noted, not fixed): German compounds are opaque to this.
    'Ausländerhass' will never match a query for 'Ausländer' because
    whitespace/punctuation tokenisation can't see inside compounds. Hybrid
    mitigates this by falling back to the dense channel.
    """
    return re.findall(r"\w+", text.lower())

def rrf_fuse(rank_lists: dict[str, list[Hit]], k: int = 60,
             top_n: int = 5) -> list[Hit]:
    """Reciprocal Rank Fusion over several ranked lists.

    WHY RANK, NOT SCORE: cosine sits in ~[0,1]; BM25 is unbounded and
    corpus-dependent. Averaging them is meaningless without normalisation
    gymnastics. RRF only uses POSITION, which is scale-free.

    Each list contributes 1/(k + rank) per document; sum across lists,
    re-sort. k=60 follows Cormack et al. 2009; results are insensitive to
    k in [20,100], so it's cited, not tuned.
    """
    scores: dict[str, float] = {}
    hits_by_id: dict[str, Hit] = {}
    ranks_by_id: dict[str, dict[str, int]] = {}

    for channel, hits in rank_lists.items():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            hits_by_id.setdefault(hit.id, hit)
            ranks_by_id.setdefault(hit.id, {})[channel] = rank

    fused = []
    # Secondary sort on id: single-channel hits all tie at exactly 1/(k+1),
    # so without a tie-break the output order depends on dict insertion order.
    for hit_id, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]:
        src = hits_by_id[hit_id]
        fused.append(
            Hit(
                id=src.id,
                kind=src.kind,
                text=src.text,
                # keep the source ranks visible for the debug view
                meta={**src.meta, "_ranks": ranks_by_id[hit_id]},
                score=score,
                via="rrf",
            )
        )
    return fused

def mmr_select(qvec: list[float], hits: list[Hit], lam: float = 0.7,
               top_n: int = 5) -> list[Hit]:
    """Maximal Marginal Relevance over dense candidates.

    THE PROBLEM: dense retrieval loves near-duplicates. Five example slots can
    fill with five variants of the same tweet - one slot of information, four
    wasted. Phase 4.4 output showed this concretely: probe 3 returned the same
    DeTox comment twice.

    MMR picks iteratively:  argmax  lam * sim(q, d) - (1 - lam) * max_sim(d, S)
    i.e. relevance minus redundancy against what is already chosen.
    lam = 0.7 favours relevance with a meaningful diversity penalty.

    WHY THE DENSE CHANNEL ONLY: BM25-only hits have no cached embedding, so
    running MMR after fusion would need a fresh encode of every candidate on
    every query. Running it on the dense candidate list before fusion is free -
    Chroma already has the vectors - and keeps the behaviour identical between
    strategy=dense and strategy=hybrid. The BM25 channel contributes its own
    diversity through lexical variation.

    Returns fewer than top_n if the candidate pool is thin (German). Does not pad.
    """
    import numpy as np

    if not hits:
        return []
    if len(hits) <= top_n or any(h.vec is None for h in hits):
        for h in hits:
            h.vec = None
        return hits[:top_n]

    q = np.asarray(qvec, dtype=float).ravel()
    q = q / (np.linalg.norm(q) or 1.0)

    V = np.asarray([h.vec for h in hits], dtype=float)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    rel = V @ q          # relevance to the query
    sim = V @ V.T        # pairwise redundancy

    selected = [int(np.argmax(rel))]
    while len(selected) < top_n:
        best_i, best_score = None, -np.inf
        for i in range(len(hits)):
            if i in selected:
                continue
            redundancy = max(sim[i][j] for j in selected)
            score = lam * rel[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)

    out = [hits[i] for i in selected]
    for h in out:
        h.vec = None     # drop 1024 floats per hit before anything logs this
    return out

class Retriever:
    """Holds the embedding model and Chroma collection so they load once.

    Per-kind budgeted retrieval: one filtered query per kind, each with its
    own k. This is the fix for the burial problem - definitions and
    guidelines are guaranteed their slots instead of competing with 282
    examples in a single similarity ranking.
    """

    def __init__(self, chroma_path: Path,records_path: Path, model_name: str, cfg: dict):
        self.model = SentenceTransformer(model_name)
        self.col = load_collection(chroma_path, model_name)
        self.cfg = cfg
        # --- BM25 indexes, one per bucket ---
        # Built from records.jsonl (the source of truth), not from Chroma.
        # 311 records: build cost is negligible, rebuild whenever KB changes.
        self._bm25 = {}
        self._bm25_records = {}

        records = [
            json.loads(line)
            for line in Path(records_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        buckets = {}
        for r in records:
            if r["kind"] == "example":
                key = f"example:{r['lang']}"      # examples are lang-specific
            else:
                key = r["kind"]                   # definitions, guidelines
            buckets.setdefault(key, []).append(r)

        for key, recs in buckets.items():
            self._bm25_records[key] = recs
            self._bm25[key] = BM25Okapi([tokenize(r["text"]) for r in recs])

    def _embed(self, text: str) -> list[list[float]]:
        return self.model.encode([text], normalize_embeddings=True).tolist()

    def _query_kind(self, qvec, where: dict, k: int, via: str,
                    want_vecs: bool = False) -> list[Hit]:
        """One filtered Chroma query -> list of Hits.

        want_vecs asks Chroma to return the stored embeddings too. Only MMR
        needs them, and only for the example bucket, so it is off by default -
        1024 floats per hit is not something to carry around by accident.
        """
        if k <= 0:
            return []

        include = ["metadatas", "documents", "distances"]
        if want_vecs:
            include = include + ["embeddings"]

        res = self.col.query(query_embeddings=qvec, n_results=k,
                             where=where, include=include)

        embs = res.get("embeddings") if want_vecs else None

        hits = []
        for i in range(len(res["ids"][0])):
            meta = unsentinel_meta(res["metadatas"][0][i])
            hits.append(
                Hit(
                    id=res["ids"][0][i],
                    kind=meta["kind"],
                    text=res["documents"][0][i],
                    meta=meta,
                    # Chroma returns cosine DISTANCE; convert to similarity
                    score=1.0 - res["distances"][0][i],
                    via=via,
                    vec=(list(embs[0][i]) if embs is not None else None),
                )
            )
        return hits

    def _bm25_kind(self, query: str, bucket_key: str, k: int) -> list[Hit]:
        """BM25 within one bucket."""
        if k <= 0 or bucket_key not in self._bm25:
            return []

        scores = self._bm25[bucket_key].get_scores(tokenize(query))
        recs = self._bm25_records[bucket_key]

        ranked = sorted(zip(scores, recs), key=lambda x: -x[0])[:k]

        hits = []
        for score, r in ranked:
            if score <= 0:
                continue          # no lexical overlap at all - don't pad
            meta = {
                "kind": r["kind"], "lang": r["lang"], "source": r["source"],
                "dimension": r["dimension"], "label": r["label"],
                **r.get("meta", {}),
            }
            hits.append(
                Hit(id=r["id"], kind=r["kind"], text=r["text"],
                    meta=meta, score=float(score), via="bm25")
            )
        return hits
    
    def retrieve(self, text: str, lang: str = "en") -> RetrievalResult:
        cfg = self.cfg
        strategy = cfg.get("strategy", "dense")

        if strategy == "dense":
            qvec = self._embed(text)
            ex_where = {"$and": [{"kind": "example"}, {"lang": lang}]}
            k_ex = cfg["k_examples"]

            if cfg.get("use_mmr", False):
                pool = cfg.get("mmr_pool_factor", 3) * k_ex
                cands = self._query_kind(qvec, ex_where, pool, "dense",
                                         want_vecs=True)
                examples = mmr_select(qvec, cands,
                                      cfg.get("mmr_lambda", 0.7), k_ex)
            else:
                examples = self._query_kind(qvec, ex_where, k_ex, "dense")

            return RetrievalResult(
                definitions=self._query_kind(qvec, {"kind": "definition"},
                                             cfg["k_definitions"], "dense"),
                guidelines=self._query_kind(qvec, {"kind": "guideline"},
                                            cfg["k_guidelines"], "dense"),
                examples=examples,
            )

        if strategy == "bm25":
            return RetrievalResult(
                definitions=self._bm25_kind(text, "definition",
                                            cfg["k_definitions"]),
                guidelines=self._bm25_kind(text, "guideline",
                                           cfg["k_guidelines"]),
                examples=self._bm25_kind(text, f"example:{lang}",
                                         cfg["k_examples"]),
            )
        if strategy == "hybrid":
            qvec = self._embed(text)
            rrf_k = cfg.get("rrf_k", 60)

            # Retrieve deeper than the budget from each channel, then fuse
            # down to k. Fusing only the top-k from each would throw away
            # exactly the evidence RRF needs to arbitrate.
            depth = 10

            buckets = {}
            for bucket, where, bm25_key, k_out in [
                ("definitions", {"kind": "definition"}, "definition",
                 cfg["k_definitions"]),
                ("guidelines", {"kind": "guideline"}, "guideline",
                 cfg["k_guidelines"]),
                ("examples",
                 {"$and": [{"kind": "example"}, {"lang": lang}]},
                 f"example:{lang}", cfg["k_examples"]),
            ]:
                # MMR re-ranks the dense channel before fusion. Pool is
                # mmr_pool_factor x depth, selected down to depth, so the
                # dense list handed to RRF is the same LENGTH as before and
                # only its ORDER changes - which is exactly what RRF consumes.
                if bucket == "examples" and cfg.get("use_mmr", False):
                    pool = cfg.get("mmr_pool_factor", 3) * depth
                    cands = self._query_kind(qvec, where, pool, "dense",
                                             want_vecs=True)
                    dense_hits = mmr_select(qvec, cands,
                                            cfg.get("mmr_lambda", 0.7), depth)
                else:
                    dense_hits = self._query_kind(qvec, where, depth, "dense")
                bm25_hits = self._bm25_kind(text, bm25_key, depth)
                buckets[bucket] = rrf_fuse(
                    {"dense": dense_hits, "bm25": bm25_hits},
                    k=rrf_k, top_n=k_out,
                )

            return RetrievalResult(**buckets)

        raise ValueError(f"Unknown strategy: {strategy}")
# Experiment log

Lab notebook. Newest entries at the bottom. Each entry: setup, results,
interpretation, open next steps. PoC vs formal evaluation flagged explicitly.

---

## 2026-07-16 — KB records built (Phase 3.2)

**Setup:** Generated kb/records.jsonl from taxonomy.yaml (definitions),
guidelines.yaml (11 hand-authored rules), and train splits (examples).
No API calls, no embeddings yet.

**Results:** 308 records total.
- 15 definitions, 11 guidelines, 282 examples.
- EN 175, DE 133.
- 78 DeTox legal illustrations (illustrative_only=true). NOTE: mostly §185
  (Beleidigung/insult), not §130 — p_185 fires far more often than p_130.
  Don't describe these as "§130 examples".

**KB example coverage (the German limitation, made concrete):**
- EN: full taxonomy — target_group (8), hate_type (7), severity (3), + negs.
- DE: target_group only (national_origin/race/religion/sexual_orientation),
  + negatives. No DE hate_type or severity (no German source annotates them).
  DE target_group also missing gender/disability/other (DeTox sparsity).

**Open / to check:**
- Verify target_group has 8 definitions (incl. `other`) and severity has 3
  (low/medium/high). Current def count = 15; expected 1+8+7+3 = 19. Something
  may be silently skipped — investigate.
- Guidelines are v1, mine — flagged for Markus (FOR_MARKUS #10).

**Resolved:** definition count was 15, expected 18. Cause: severity block in
taxonomy.yaml was indented at the top level, outside `dimensions:`, so both
the KB generator and check_taxonomy silently skipped it. Re-indented under
dimensions. Now 18 definitions (gate 1 + target 7 + type 7 + severity 3),
check_taxonomy sees 4 dimensions. Classic YAML indentation trap.

## 2026-07-16 — BGE-M3 cross-lingual sanity (Phase 3.3)

**Setup:** Loaded BAAI/bge-m3 (first run: ~2.3 GB download). Encoded an EN
misogyny sentence, its German translation (zero shared words), and an
unrelated EN sentence. Normalised embeddings, cosine similarity. Running on
CPU (iGPU active to save power; dedicated GPU available on restart for
heavier re-ingest work later).

**Results (cosine):**
- EN-misogyny <-> DE-misogyny : 0.887   (HIGH)
- EN-misogyny <-> EN-hiking   : 0.383   (LOW)
- DE-misogyny <-> EN-hiking   : 0.350   (LOW)

**Interpretation:** the cross-lingual misogyny pair scores far above either
unrelated pair despite no lexical overlap. Confirms the shared multilingual
embedding space works — a German query can retrieve an English concept. This
is the empirical basis for the design decision: definitions and guidelines
authored once in English, examples kept per-language (slurs/slang don't
transfer, concepts do).

**PoC**, not formal eval — one illustrative triple, not a systematic
cross-lingual retrieval benchmark.


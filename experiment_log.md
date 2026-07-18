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

## 2026-07-16 — First KB retrieval, naive dense (Phase 3.5)

**Setup:** BGE-M3 dense retrieval, top-5, over the full 311-record KB (defs +
guidelines + examples mixed). 6 hand-written probes. No filtering, no hybrid
— deliberately naive baseline.

**What worked:**
- Misogyny probe -> gender-labelled examples (top hit gender+disability).
- Religion probe -> muslim/islam examples, one tagged religion.
- German probe "Ausländer raus..." -> sensible ENGLISH hateful examples
  (deport/"not wanted here"), zero shared words. Cross-lingual transfer
  confirmed in the live KB, not just the toy demo.
- Neutral probe -> highest distances (0.42+ vs 0.25-0.30 for real hate); the
  distance gap itself is signal.

**Key finding (SQ1 motivation):** examples dominate; definitions and
guidelines almost never surface. Only 1 definition appeared across all 6
probes (and on the neutral probe); ZERO guidelines appeared. Cause: examples
are raw text, lexically close to raw-text queries; defs/guidelines are
abstract descriptive language, far in embedding space. Naive dense retrieval
over a mixed KB buries the abstract knowledge.

**Two informative misses:**
- "you're such an idiot, get lost" (offensive, NOT hate) -> retrieved hateful
  examples ("Dumb bitch is dumb"), not the profanity-without-target guideline.
  The over-sensitivity failure mode, live.
- "he called me the n-word and I was shocked" (victim recounting, NOT hate)
  -> retrieved unrelated sexual/violent examples; the quoted-slurs guideline
  that exactly covers this case NEVER surfaced.

**Implication:** motivates Phase 4 directly — per-kind metadata-filtered
retrieval (surface definitions/guidelines separately from examples), hybrid
BM25 (would catch literal "quoted"/"slur"), and per-kind retrieval budgets.
Naive dense-over-mixed-KB is insufficient on its own.

**PoC / eyeball**, not a metric.

## 2026-07-18 — Per-kind budgeted retrieval (Phase 4.2)

**Setup:** Replaced single-ranking dense retrieval with three filtered Chroma
queries, one per kind, each with its own budget (k_def=2, k_guide=2, k_ex=5).
Examples filtered by query language; definitions/guidelines not (EN-only by
design, cross-lingually reachable). Same 6 probes as Phase 3.5.

**Burial problem: structurally solved.** Every probe now returns 2 defs +
2 guidelines + 5 examples by construction, vs Phase 3 where 6 probes yielded
1 definition total and ZERO guidelines.

**Before/after on the two documented misses:**

| Probe | Expected record | Phase 3 (naive) | Phase 4.2 (budgeted) |
|---|---|---|---|
| "you're such an idiot, get lost" | guide-profanity-without-target | absent | **#1 guideline** ✅ |
| "he called me the n-word..." | guide-quoted-slurs | absent | still absent ❌ |

Miss #1 resolved by budgeting alone. Miss #2 needs the lexical channel —
dense can't bridge "victim recounting" to a guideline about quoting.
Carried into 4.3 as the BM25 test case.

**Other probes now sane:** religion probe → def-target_group-religion #1 +
guide-religion-ideas-vs-believers #1. German probe → German DeTox examples
(incl. one legal illustration, §185) with English definitions/guidelines,
confirming the asymmetric cross-lingual design works end to end.

**NEW finding — definition wording determines retrievability.**
"women are too stupid to vote" retrieved disability + inferiority
definitions, NOT gender. Cause: the gender definition text says "gender...
misogyny... trans people" and never contains the word "women"/"female".
Meanwhile "stupid" pulls def-target_group-disability strongly. So a
correctly-authored definition can still be effectively unreachable for the
most common phrasing of that hate.
Implication: definition text is not just documentation for the LLM, it is
also the retrieval key. Worth an explicit methodology point, and a candidate
for a taxonomy edit (add "women", "female" to the gender definition) — which
would itself be a clean demonstration of the adaptability claim.

**Also noted:** def-target_group-disability appears in the definitions bucket
for 4 of 6 probes — behaving as a hub in embedding space. Monitor; may need
attention if it crowds out better-matched definitions at larger k.

**Planned experiment (do NOT fix silently):** rewrite the gender definition to
include "women"/"female", re-ingest, re-run the same probe, and record whether
def-target_group-gender now surfaces. This is a clean, small demonstration of
the adaptability claim on our own workflow: one YAML edit + re-ingest, frozen
LLM, measurable retrieval change. Hold until retrieval work (Phase 4) is
finished so the before/after isn't confounded by strategy changes.

## 2026-07-18 — BM25 lexical channel (Phase 4.3)

**Setup:** BM25Okapi over KB records, built per bucket (definition,
guideline, example:en, example:de) from records.jsonl. Lowercase \w+ tokens.
Same 6 probes, strategy=bm25.

**Miss #2 ("he called me the n-word and I was shocked"):**
guide-quoted-slurs STILL absent. But guide-reclaimed-slurs surfaced at #1
(score 3.31) — the bridge is the token "word" ("...presence of the word"),
matching the query's "n-word" -> [n, word]. Lexical luck, but it did pull a
slur-related guideline that dense never reached.
So: partial resolution. The lexical channel reaches slur-territory; it just
can't bridge "n-word" -> "slur" (different strings, no shared token).

**Asymmetry found — BM25 helps some buckets and hurts others:**
- Guidelines/definitions (small pools, longer descriptive text): BM25 pulls
  relevant records dense missed.
- Examples (short raw social-media text): BM25 is notably WORSE than dense.
  Top hits for the n-word probe were unrelated ("i use korans for kindling",
  Trayvon comment) — winning on common-token overlap, not meaning.
  Also def-target_group-gender ranked #1 spuriously on stopword overlap.

**Implication:** this asymmetry is the argument for hybrid rather than
either channel alone. RRF (4.4) lets each channel contribute where it's
strong instead of picking one globally.

**Limitations noted (not fixed):**
- German compounds are opaque to whitespace tokenisation: "Ausländerhass"
  will never match a query for "Ausländer". Hybrid mitigates by falling back
  to dense.
- German query vs English definitions/guidelines: BM25 contributes nothing
  (zero lexical overlap). Hybrid degrades gracefully to dense there — by
  design, not a bug.
- BM25 hits with score <= 0 are dropped rather than padded, so a bucket can
  return fewer than k when there's no lexical overlap at all.

  ## 2026-07-18 — Dense vs BM25, all 6 probes (Phase 4.3)

**Setup:** Same 6 probes under strategy=dense and strategy=bm25,
per-kind budgets unchanged (2/2/5).

**Confirmed asymmetry — the channels are good at different buckets:**

| Bucket | Winner | Evidence |
|---|---|---|
| Guidelines | **BM25** | neutral probe: BM25 → profanity-without-target (correct); dense → reclaimed-slurs + quoted-slurs (wrong). German probe: BM25 → negation-and-support + counter-speech; dense → two generic rules. |
| Examples | **Dense** | religion probe: dense pulls consistent muslim/islam examples; BM25 drifts after the first hit. Neutral probe: BM25 returns nun/slur text matching only on "weather"/"walk" tokens. |
| Definitions | Mixed, both weak | misogyny probe: BM25 → gender (correct!), dense → disability + inferiority (wrong). But neutral probe under BM25 → severity-medium + irony (noise). |

**This is the direct argument for RRF fusion (4.4):** neither channel should
be chosen globally. Fusion lets the lexical channel carry the small wordy
buckets and the dense channel carry the short-text example bucket.

**KB-authoring finding (second instance):** guide-quoted-slurs surfaced under
dense for "the weather is lovely for a walk today" but never for
"he called me the n-word and I was shocked". Its embedding sits in an
unhelpful region. Same class of problem as the gender definition: KB record
WORDING determines retrievability, independent of retrieval strategy.
Candidate for the same deliberate rewrite-and-remeasure demo.

**BM25 sparsity confirmed:** the neutral probe returned only 1 guideline
(not 2) — no other guideline had non-zero lexical overlap. Dropping
zero-score hits rather than padding is working as intended.

**Rewrite demo, refined:** the gender definition is retrievable by BM25
(lexical "gender") but not by dense — so the rewrite-and-remeasure demo can
report the effect PER CHANNEL, not just before/after. Richer result.
guide-quoted-slurs is a second candidate with the same diagnosis (wrong
region of embedding space, wrong probe fires it). Two records with one
diagnosis = a small pattern rather than an anecdote. Run both demos together
after Phase 4 is finished.

**BM25 on the definitions bucket is near-noise.** Definitions share almost no
vocabulary with raw hateful text ("irony" ranked #1 for "women are too stupid
to vote" on the single shared token "to"). With only 18 records, near-zero
signal still produces a full ranking, and RRF weights it equally with dense.
Consider excluding BM25 from the definitions bucket, or requiring a minimum
BM25 score before a hit enters fusion. Candidate SQ1 Stage B variant.
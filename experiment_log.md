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

  ## 2026-07-18 — Dense vs BM25 vs hybrid, all 6 probes (Phase 4.3/4.4)

**Note on this entry:** an earlier version of this comparison was written from
terminal output that failed to upload and was reconstructed from memory. It was
substantially wrong and has been deleted. Everything below is transcribed from
`debug_retrieval.py` output under `--strategy dense`, `--strategy bm25` and
`--strategy hybrid`, run on kb_version 4224f81f8d35ee88.

**Setup:** 6 standard probes, per-kind budgets 2/2/5, depth=10 and rrf_k=60 for
hybrid.

### Hit table (expected record present in its bucket)

| # | Probe | Expected | dense | bm25 | hybrid |
|---|---|---|---|---|---|
| 1 | women are too stupid to vote | def-target_group-gender | miss | miss | miss |
| 2 | muslims are terrorists | def-target_group-religion | #1 | #1 | #1 |
| 3 | Ausländer raus | DE examples + defs + guides | full | **0 defs, 0 guides** | full |
| 4 | you're such an idiot | guide-profanity-without-target | #1 | #1 | #1 |
| 5 | he called me the n-word | guide-quoted-slurs | miss | miss | miss |
| 6 | the weather is lovely | few hateful examples | 3/5 hateful | 4/5 hateful | 2/5 hateful |

**Verdict: hybrid does not beat dense on this probe set.** It ties on probes 1-5
and is marginally cleaner on probe 6. n=6 is noise-level evidence and no
strategy claim is made from it. The SQ1 comparison runs on 300 dev items in
Phase 8. The value of this run is diagnostic, not comparative.

### Finding A — RRF at k=60, depth=10 is a consensus gate, not a blend

Best possible single-channel RRF score: 1/(60+1) = 0.0164.
Worst possible two-channel score: 1/(60+10) x 2 = 0.0286.

0.0286 > 0.0164, so **any** record found by both channels outranks **any**
record found by one, independent of rank. Generally this holds whenever
k > depth - 2, which is always true at our settings.

Observed:
- Probe 1 definitions: hybrid returned exactly BM25's ranking (irony,
  stereotyping). Dense's top hit was excluded for having no BM25 support.
- Probe 5 guidelines: guide-reclaimed-slurs was BM25 #1 (3.31), the only
  slur-related guideline any channel reached, and was dropped from the fused
  output in favour of two records with two-channel support.

Not a bug, and not fixed. RRF is designed to be conservative; the cost is
single-channel recall. Recorded as a limitation, with the practical
implication that a noisy channel does not merely contribute to a bucket, it
can determine it.

### Finding B — BM25 on the definitions bucket is anti-correlated with relevance

Top BM25 definition score per probe:

| Probe | Top score | Record |
|---|---|---|
| 1 misogyny | 2.318 | irony |
| 2 muslims | 2.410 | religion (correct) |
| 4 idiot | 2.637 | other |
| 5 n-word | 2.441 | gender |
| 6 neutral (the weather is lovely) | **3.497** | def-hate |

The neutral probe scores highest of all six. Cause: 18 short documents, and
BM25Okapi floors negative IDF at a small positive epsilon, so near-stopwords
still contribute; short documents additionally get a large length-normalisation
boost. An 8-token query of mostly common words matches every definition.

Combined with Finding A, BM25 controls the definitions bucket under hybrid.
Candidate SQ1 Stage B arm: `bm25_in_definitions: false`, or a minimum BM25
score before a hit may enter fusion.

### Finding C — BM25 is inert for German against EN knowledge

Probe 3 under `--strategy bm25`: 0 definitions, 0 guidelines, 5 examples. Zero
lexical overlap between a German query and English abstract text. Under hybrid
every German definition/guideline hit shows `ranks: {'dense': N}` only, scoring
0.016. Hybrid degrades to dense in the cross-lingual case, by design.

### Finding D — legal illustrations crowd the German example bucket

DE example pool is roughly 60 balanced + 78 legal illustrations, so ~57% of
retrievable German examples are `illustrative_only`.

- BM25, probe 3: 3 of 5 examples are `ex-detox-legal-*`.
- Hybrid, probe 3: slots 1 and 2 are the **same comment twice**
  (`ex-detox-1394019963506155527` and `ex-detox-legal-1394019963506155527`).
  Two of five German example slots carry one text.

Root cause is KB construction, not retrieval: `build_kb_records.py` emitted the
same DeTox comment into both the balanced sample and the legal illustrations.
Fixed at build time in 4.5a below.

### Finding E — retrieval supplies only hateful evidence for non-hateful input

Probe 4 (offensive, not hate): 5/5 retrieved examples gate=True under every
strategy. Probe 5 (victim recounting, not hate): 5/5 gate=True under every
strategy.

Structural, not incidental: 20 negatives per language cannot win a similarity
race against 282 examples. The prompt therefore receives hateful evidence when
the correct answer is "not hate" - a direct false-positive pathway into the
classifier. Candidate mitigation and SQ1 arm: reserve part of the example
budget for a `gate=False`-filtered query (`k_examples_neg`), the same per-kind
budgeting logic one level down. Not implemented in Phase 4; noted for Phase 8.

### Correction to the KB-rewording demo plan

An earlier note claimed def-target_group-gender is reachable by BM25 but not by
dense, and that the rewrite demo could therefore be reported per channel. The
real output does not support this: on the misogyny probe BM25 returns irony and
stereotyping, not gender. Gender appears in BM25 output only on the muslims
probe (#2) and the n-word probe (#1), i.e. on queries it has nothing to do
with, via stopword overlap.

The corrected diagnosis is stronger, not weaker: def-target_group-gender is
unreachable by **both** channels for the query type it exists to serve, and is
reachable by BM25 only as noise on unrelated queries. The rewrite demo stands;
the per-channel framing is dropped.

## 2026-07-18 — KB duplicate-text fix + MMR (Phase 4.5)

### Duplicate texts in the KB

Finding D of the 4.3/4.4 entry traced to KB construction, not retrieval:
`gen_examples` emits `ex-detox-<id>` and `gen_legal_illustrations` emits
`ex-detox-legal-<id>` for the same comment. Different ids, identical text, so
the id-based dedup in `gen_examples` could not see it. Retrieval consequently
spent two of five German example slots on one comment.

Fix: the legal generator now receives the balanced examples and merges
`meta.stgb` onto an existing record instead of emitting a second copy. The
surviving record keeps `illustrative_only: False` - it is a genuine scored
example that additionally carries a paragraph flag. Scarce legal annotation
retained, duplicate removed.

Guard: `check_kb_schema.py` now checks KB-wide text uniqueness, so this class
of defect cannot return silently. Confirmed: 307 records, 307 distinct texts.

- merged into existing examples: 4
- standalone legal illustrations: 74 (was 78)
- total records: 307 (was 311) = 18 definitions + 11 guidelines + 278 examples
- kb_version: 475869f9e2422969 (was 4224f81f8d35ee88)
- pytest: 7 green

**Finding D is only half resolved.** The duplicate is gone, but the
underlying imbalance is not: the German example pool is 129 records of which
74 (57%) are `illustrative_only` legal illustrations, against 55 balanced
examples. Only 4 of the 78 legal records overlapped a balanced one, so
deduplication could not shift the ratio. German retrieval therefore still
draws its five example slots from a pool that is majority-illustrative.
Candidate Phase 8 mitigations: cap the legal illustrations, or give them a
separate budget slot the way retrieval already budgets by kind. Not addressed
in Phase 4.

**Sampling note - `examples_per_label` is not linear in KB size.** The
per-label coverage counts sum to 190 for English, but the KB holds 149 unique
English examples: 41 rows were sampled under more than one label. Cause is the
richness-preferred sort, which draws rows populating the most dimensions
first, so the same rich row wins the target_groups, hate_types and severity
draws. German shows only 5 such overlaps because German rows populate fewer
dimensions. Consequence: multi-label rows are over-represented relative to
what "10 per label" suggests. Relevant to the Phase 8 ablation on
`examples_per_label`, where the knob does not map linearly onto pool size.

### MMR on the example bucket

Implemented over the dense channel only, before fusion. BM25-only hits carry
no cached embedding, so post-fusion MMR would need a fresh encode of every
candidate on every query; pre-fusion is free (Chroma already holds the
vectors) and keeps behaviour identical between strategy=dense and
strategy=hybrid. Pool = 3 x budget, selected down to budget, lambda=0.7.
Returns fewer than k on thin pools rather than padding.

Measured with `scripts/check_mmr.py` under strategy=dense over the returned
examples: pairs above cosine 0.9, and mean pairwise cosine. strategy=dense is
used deliberately - under hybrid, MMR reorders the dense channel and RRF then
fuses, so the effect on the final bucket is indirect.

| Probe | dup off | dup on | mean off | mean on | max off | max on |
|---|---|---|---|---|---|---|
| women are too stupid to vote | 0 | 0 | 0.565 | 0.476 | 0.696 | 0.576 |
| muslims are terrorists | 0 | 0 | 0.570 | 0.570 | 0.682 | 0.682 |
| Ausländer raus | 0 | 0 | 0.546 | 0.466 | 0.647 | 0.574 |
| you're such an idiot | 0 | 0 | 0.633 | 0.530 | 0.745 | 0.745 |
| he called me the n-word | 0 | 0 | 0.519 | 0.505 | 0.583 | 0.566 |
| the weather is lovely | 0 | 0 | 0.480 | 0.480 | 0.536 | 0.536 |

**No near-duplicate pairs occur, with or without MMR.** The redundancy failure
mode MMR exists to prevent does not arise in this KB at k=5: balanced
per-label sampling from TRAIN already yields a diverse example pool. This is a
property of the corpus, not a general property of RAG, and it bounds what SQ1
Stage C can be expected to show.

Mean pairwise cosine nonetheless falls on 4 of 6 probes (largest drop 0.633 ->
0.530), is unchanged on 2, and rises on none. The two unchanged probes are
exact no-ops in both mean and max: where candidates are already well spread,
the redundancy penalty never overturns the relevance ranking and MMR
reproduces plain top-k. On the profanity probe the mean falls while max stays
at 0.745 - the most-similar pair survives selection, because at lambda=0.7 a
redundant-but-highly-relevant candidate can still win. A concrete illustration
of what lambda controls.

Consistency check: the table is identical before and after the KB rebuild.
Expected - the four removed records were German and sat at dense ranks 9-10,
outside the top-5, and no English record changed.

Note: `dup=0` on the German probe is not evidence against Finding D. Under
strategy=dense the duplicate pair never entered the top-5; it surfaced only
under hybrid, through fusion with BM25.

Default kept at `use_mmr: true`: costs one larger Chroma query, degraded no
probe. n=6 settles nothing - retained as the SQ1 Stage C ablation arm for
Phase 8.

## 2026-07-21 — TUDaGPT model roster changed (Phase 5.0)

Re-verified all model slugs against the live TUDaGPT model panel before
pinning. The roster has changed substantially since Phase 0-1:
`mistral-large-3-675b-instruct-2512`, the strong-tier lead, is now offline;
`llama-3.3-70b-instruct`, `gemma-3-27b-it` and `qwen3.5-27b` differ from the
slugs recorded earlier and are also offline. The entire fast tier as
configured was unreachable.

`qwen3.5-122b-a10b` is online and remains the pinned model for all formal
runs (decision Q2).

Reproducibility note: model availability on shared university infrastructure
is not stable across the lifetime of a thesis, and slugs are not guaranteed
stable either. This is a direct threat to reproducibility for any result
attributed to a specific model, and it strengthens rather than weakens the
case for `allow_fallback=False` on formal runs plus `active_model` stamped on
every result record - a silent fallback to whatever happened to be online
would make results unattributable after the fact. Slugs re-verified
2026-07-21.

## 2026-07-21 — Phase 5.0: six decisions

Six blocking questions resolved before writing classifier code. Each is a
defensible default rather than the only option; the ones that change what the
thesis can claim go to FOR_MARKUS.md.

**Q1 — frozen evaluation subsets.** EN test is 12,041 items; at ~3 s per call
x n self-consistency votes x N arms the full split is not runnable. Frozen
seeded stratified subsets instead, built once and shared by every system from
here on, including the Phase 7 encoder baselines. Shared items are what make
cross-system comparison paired - McNemar and the paired bootstrap need
item-level pairing.

**Q2 — the frozen model is `qwen3.5-122b-a10b`, fallback off for formal runs.**
"Frozen LLM" is the premise of the adaptability claim, so a silent fallback
mid-experiment would make a result unattributable to any model. Strong tier
becomes a cost/quality ablation arm only. Confirmed live 2026-07-21:
`check_client --tier medium --no-fallback` answered from `qwen3.5-122b-a10b`.

**Q3 — temperature 1.0 everywhere; the vote is the stabiliser.** System
prediction = self-consistency vote over n runs (n=3 dev, n=5 formal).
Consistency (Krippendorff's alpha) is computed across those same raw runs. No
separate deterministic arm: alpha at T=0 is meaningless by definition, and T=0
determinism on this API is unverified. Storing every raw run also yields an
n=1 cost-vs-stability arm for free by scoring only the first run per item.

**Q4 — German scope. REVISED after 5.1: gate only, not gate + target_group.**
Original decision was gate + target_group, on the basis that those are the two
dimensions with German gold. The eval-subset diagnostics showed target_group
is unusable in practice - see the 5.1 entry. German is therefore a gate-only
side study. The classifier still outputs all four dimensions for German input
(identical schema, identical code path); scoring masks the three without gold.
-> FOR_MARKUS #11.

**Q5 — SQ3 sentinel spec.** Two instruments, both drawn from
`en_dev_eval_sq3_feedback`. Over-steering sentinel: ~100 items the system got
right on the first stable run, frozen, never corrected, re-scored after every
feedback round; metric = flips-to-wrong. Generalisation probe: for each
corrected item, its top-5 nearest unseen items by BGE-M3 similarity, scored
before and after the round; metric = whether one correction fixes a
neighbourhood or only itself. Neither can be frozen at subset-creation time -
the sentinel is defined by a run that does not exist yet - so both are
selected at run time from within the slice. Both need per-item retrieval
provenance, which is why ItemResult stores retrieved hit ids.

**Q6 — adaptability: one metric, three edit types.** Adaptability stays
defined as delta macro-F1 per KB edit with the LLM frozen. The three edit
types map to: corrected examples -> SQ3 feedback rounds (primary,
quantitative); new label -> SQ3 rounds-to-recovery, holding a hate_type out
and adding it mid-run; revised definition -> the Finding-G rewording demo
(`def-target_group-gender`, `guide-quoted-slurs`), measured at both retrieval
and F1 level. Three instances of one definition, not competing evidence
structures. kb_version attribution is what makes each falsifiable.

---

## 2026-07-21 — TUDaGPT model roster changed (Phase 5.0)

Re-verified every model slug against the live TUDaGPT panel before pinning.
The roster has changed substantially since Phase 0-1:
`mistral-large-3-675b-instruct-2512`, the former strong-tier lead, is now
offline. `llama-3.3-70b-instruct`, `gemma-3-27b-it` and `qwen3.5-27b` differ
from the slugs recorded earlier and are also offline - the entire configured
fast tier was three wrong slugs pointing at three offline models, which is why
it never worked.

`qwen3.5-122b-a10b` is online and remains the pinned model (Q2).

Reproducibility note: model availability on shared university infrastructure
is not stable across the lifetime of a thesis, and slugs are not stable
either. This is a direct threat to reproducibility for any result attributed
to a specific model, and it strengthens the case for `allow_fallback=False`
plus `active_model` stamped on every result record: a silent fallback to
whatever happened to be online would make results unattributable after the
fact.

Also noted: `qwen3.5-122b-a10b` advertises `output: ["text", "thought"]`, so
the pinned model can in principle emit reasoning traces. It has not done so in
any call to date, but the 5.4 parser strips `<think>` blocks before
`json.loads` regardless - otherwise a change in server-side defaults would
surface as an unexplained parse-failure rate.

---

## 2026-07-21 — Phase 5.1: frozen evaluation subsets

**Stratification is on (source, gate), not gate alone.** Every fine-grained
dimension comes from exactly one dataset - hate_type from implicit_hate,
severity from mhs, target_group from hatexplain/mhs/detox. A sample balanced
on gate but blind to source can starve a dimension, and which dimension gets
starved would be luck. Composition came out within 0.5 points of the parent
split on every subset.

**The English dev pool is split into three disjoint named slices** (900 items
total): `sq1_tune` (250) is tuned on, `main` (299) is reported from,
`sq3_feedback` (351) carries the feedback rounds. One dev set cannot be
tuning set, headline set and adaptability set at once - selecting a retrieval
configuration on the same items later used to measure adaptability is
circular. The partition is itself stratified, so the three slices are
comparable to each other and to the parent. Disjointness is asserted in the
build script and in pytest.

**A naturally-distributed subset cannot report the fine-grained dimensions.**
Diagnostics on the 299-item `main` slice:

| dimension | annotated | thinnest label |
|---|---|---|
| gate | 299 | - |
| target_groups | 62 | disability=4, other=4 |
| severity | 40 | low=9 |
| hate_types | 27 | incitement=2, irony=2, threatening=2 |

Macro-F1 over a 2-item label is not reportable. Growing the natural subset
enough to fix it needs roughly 7x the items and therefore 7x the API budget,
which is not affordable.

**Solution: separate label-stratified dimension subsets.** `main` reports
gate; `_targets`, `_types` and `_severity` report their dimension, drawn
label-balanced and disjoint from the slices and from each other. The
fine-grained dimensions are only ever scored on hateful items anyway, so their
evaluation set was never a natural distribution in any meaningful sense -
making the balance explicit and stated is more defensible than presenting a
27-item incidental sample as one. Every label now clears 15 (dev) / 25 (test):

| subset | items | thinnest |
|---|---|---|
| en_dev_eval_targets | 56 | religion=15 |
| en_dev_eval_types | 103 | 15 (all seven) |
| en_dev_eval_severity | 45 | 15 / 15 / 15 |
| en_test_eval_targets | 102 | disability=25 (was 1 in the natural test subset) |
| en_test_eval_types | 173 | 25 (all seven) |
| en_test_eval_severity | 75 | 25 / 25 / 25 |

56 items yielding 15+ per target label is the multilabel effect: one item
carries several labels, so filling the rarest label first fills the common
ones incidentally. Dev dimension subsets total 204 items ~= 612 calls at n=3,
roughly 30 minutes per arm.

**Reporting rule fixed now:** a label with fewer than 10 support items is
reported with its support and excluded from the macro average, never silently
averaged in.

**German target_group is empty in practice - the finding that revised Q4.**

| subset | items | target_groups annotated | positive labels |
|---|---|---|---|
| de_dev_eval | 150 | 9 | religion=1 |
| de_test_eval | 300 | 15 | **none** |

Cause: DeTox annotates targets only on hateful comments, DeTox is 11% hateful,
and most `discrim_*` annotator fractions sit below the 0.5 threshold, so the
item resolves to "annotated, no target group applies" = `[]`. Not a sampling
artefact - the whole German test split holds only about 30 items with a
positive target label, so a larger subset barely moves it. German is therefore
gate-only for scoring.

This is the `None` vs `[]` rule earning its keep. Those 15 items are not
missing data; they are annotated, and the correct answer is "no protected
group targeted". Had the two been collapsed in Phase 2, the system would now
be scored against phantom labels.

Framing: German resources for multi-label hate speech taxonomies do not exist
at usable density. That is a stated finding, not an apology, and it is the
same gap the Kums et al. 2025 group works in. GAHD is adversarial by
construction, so German gate detection on 300 items remains a hard robustness
test rather than an easy one.

---

## 2026-07-21 — Phase 5.2: prompt assembly

`build_prompt(text, lang, ctx)` is a pure function: no retrieval, no API, no
I/O beyond templates. That separation is what makes `check_prompt.py` free to
run and what keeps the SQ2 comparison clean - zero-shot, few-shot and RAG
differ only in the context passed in, never in the code path.

`prompt_version` is the template stem plus a content hash of the template file
(currently `classify_v1-c74cb7ab`), stamped on every result for the same
reason as kb_version: a reworded prompt must not be silently confused with the
old one.

**The label space is task specification and is present in every arm, including
zero-shot.** It is generated from taxonomy.yaml, so adding a label changes the
prompt, the knowledge base and the accepted output schema with no code edit.
What varies between SQ2 arms is retrieved knowledge, never the task
definition - a model that does not know the permitted values is not doing the
task at all.

**Example groups, not one example list.** Legal illustrations already have to
render differently today (text plus `flagged under StGB §{meta.stgb}`, no gold
multi-labels, explicitly introduced as background rather than label evidence),
so the multi-group shape is required regardless. Phase 8's negative-example
budget and any separate legal budget become new groups, not a rewrite.

**No scores anywhere in the prompt.** RRF scores are positional, not
similarities, and are not comparable across queries or strategies.

Two rendering issues found and fixed against real output:
- Example text is whitespace-collapsed before rendering. Social-media text
  carries embedded newlines and blank lines; a blank line inside a quoted
  example visually terminates the markdown list, and German examples are the
  longest and worst affected.
- Definitions render dimension-qualified (`target_group / gender`,
  `hate_type / threatening`). Unqualified they are ambiguous: `other` exists
  in more than one dimension, and severity definitions would render as a bare
  `- low:`.

Prompt sizes, measured:

| arm | system | user | total | approx tokens |
|---|---|---|---|---|
| EN RAG (n-word probe) | 1063 | 1650 | 2713 | ~678 |
| DE RAG (Ausländer probe, incl. 1 legal illustration) | 1063 | 2705 | 3768 | ~942 |

German is the longest case, as expected, and is nowhere near a context limit.

**Data note (deferred):** the DeTox source text carries unescaped HTML
entities - `nix-&gt;` appears verbatim in a rendered German example. Harmless
but untidy in a prompt. Fixing it means `html.unescape()` in the DeTox loader
plus a KB rebuild; deferred to a Phase 8 tidy rather than invalidating
kb_version 475869f9e2422969 mid-phase.

**Finding E caught in a rendered prompt.** The n-word probe
("he called me the n-word and I was shocked", gold: NOT hate) produces a
prompt in which every piece of retrieved evidence points toward "hate":

- 5 of 5 labelled examples are `hate=true`
- retrieved definitions are `target_group / gender` and
  `hate_type / threatening` - neither relevant
- retrieved guidelines are `threats-and-incitement` and
  `negation-and-support` - `guide-quoted-slurs` absent, as under every
  retrieval strategy

The Finding-E caution line is the only counterweight in the prompt. Whether it
is enough is measurable at Slice 1 via the gate false-positive count. Prompt
saved verbatim as a qualitative example.

## 2026-07-22 — Phase 5.7: Slice 1, first live no-RAG vs RAG comparison

**Setup:** `en_dev_eval_main`, 150 items, three arms, n=3 self-consistency
votes at T=1.0, `qwen3.5-122b-a10b` pinned with `allow_fallback=False`,
kb_version 475869f9e2422969, prompt classify_v1-c74cb7ab. 450 ItemResults,
~1350 calls. Attribution assertion passed: every result answered by the pinned
model, none unattributable.

**Three arms, not two.** zero_shot vs rag alone cannot separate "retrieval
helped" from "having examples at all helped". few_shot supplies the same
number of examples, statically sampled with a fixed seed, no retrieval, no
definitions, no guidelines. So few_shot minus zero_shot isolates examples, and
rag minus few_shot isolates retrieval.

### Results

| arm | gate macro-F1 | FP | FN | parse fail | uncertain | mean latency |
|---|---|---|---|---|---|---|
| zero_shot | 0.717 | 33 | 6 | 2.0% | 0 | 47.4 s |
| few_shot | 0.706 | 32 | 8 | 2.2% | 0 | 62.4 s |
| rag | 0.696 | 35 | 7 | 2.7% | 0 | 54.7 s |

| arm | target_group macro-F1 (n=28) |
|---|---|
| zero_shot | 0.819 |
| few_shot | 0.778 |
| rag | **0.909** |

target_group averaged over `gender` and `race` only; the other five labels
fell below MIN_SUPPORT=10 and were excluded per the 5.1 reporting rule.
hate_type (n=14) and severity (n=18) had no label clearing support, so both
report n/a. Those dimensions get their real numbers from the label-stratified
subsets, not from `main`.

### Finding 1 — the arms are tied on the gate, and separate on target_group

A 0.021 spread across three arms on n=150 is noise; no strategy claim is made
from it. The dimension-level split is the informative part:

- examples alone: **-0.041** (few_shot minus zero_shot). Static examples hurt.
- retrieval: **+0.131** (rag minus few_shot). Retrieval helps.

The pattern is coherent with the hypothesis rather than against it: no effect
on the coarse binary decision the model already knows how to make, a clear
effect on the fine-grained dimension where the specific label definitions in
this taxonomy actually matter. n=28 over two labels, so this is suggestive and
not established; `en_dev_eval_targets` (102 items, all seven labels above 25)
is what confirms or kills it.

### Finding 2 — the system over-flags, and retrieval is not the cause

False positives outnumber false negatives roughly 5 to 1 in every arm. With
~46 hateful and ~104 benign items, that is roughly 87% recall on hate against
~55% precision: about a third of all benign text is flagged.

Critically, **zero_shot has 33 FPs while using no retrieval at all**, and rag
adds only two. Finding E (retrieval supplies only hateful evidence for
non-hateful input) predicted RAG-specific over-flagging and is NOT the
explanation here. The over-flagging is a model prior that retrieval neither
causes nor corrects.

Errors by source dataset (FP rate over that source's benign items):

| source | benign | hateful | zero_shot FP | few_shot FP | rag FP |
|---|---|---|---|---|---|
| hatexplain | 23 | 10 | 14 (61%) | 14 (61%) | 13 (57%) |
| implicit_hate | 30 | 15 | 3 (10%) | 3 (10%) | 3 (10%) |
| mhs | 54 | 18 | 16 (30%) | 15 (28%) | 19 (35%) |

A 61% versus 10% spread is not uniform over-sensitivity. It is one dataset's
definition of not-hate disagreeing with the model.

### Finding 3 — the HateXplain `offensive` mapping explains part of it

Phase 2 mapped HateXplain `offensive` to gate=False on a strict reading, with
raw votes kept in `Record.raw` so the decision stays reversible
(FOR_MARKUS #4). Splitting the HateXplain items by their original class:

| gold class | n | flagged as hate (all three arms) |
|---|---|---|
| hatespeech | 10 | 100% (correct) |
| offensive | 9 | 89% |
| normal | 14 | 43% |

So the 14 HateXplain false positives are 8 `offensive` plus 6 `normal`. The
mapping accounts for 8 of the 33 total false positives, about a quarter. The
system is largely reproducing the hatespeech/offensive boundary the HateXplain
authors themselves drew; our strict mapping scores that agreement as error.

It does not account for the rest. 43% of genuinely `normal` items flagged is
real over-flagging, as is MHS's 30%.

**All three arms produce identical counts on `normal` (6/14) and near-identical
on `offensive`.** Retrieval does not move these judgements at all, which is the
second independent line of evidence that the over-flagging is a model prior.

Implication for FOR_MARKUS #4: the offensive-to-not-hate decision is not
cosmetic. It costs roughly 8 false positives per 150 items and depresses the
reported gate F1. Worth a decision rather than a default, and the numbers are
now available to make it with.

### Finding 4 — hate_type gold is single-label by construction

Mean labels per hateful item:

| dimension | gold | zero_shot | few_shot | rag |
|---|---|---|---|---|
| target_group | 1.57 | 1.48 | 1.52 | 1.48 |
| hate_type | **1.00** | 2.27 | 2.09 | 2.10 |

target_group prediction counts track gold closely. hate_type gold is exactly
1.00 because Implicit Hate assigns one type per post, while the taxonomy
declares hate_type multilabel and the prompt invites multiple values. The
model supplies about two and is penalised for the extra.

This is a taxonomy-versus-data mismatch, not a model error, and it predicts
hate_type will score poorly on `en_dev_eval_types` for reasons unrelated to
retrieval. Decision needed before that run: score top-1 against single-label
gold as primary, with the multilabel score reported alongside. -> FOR_MARKUS.

### Finding 5 — arm agreement is high

All three arms agree on the gate for 135 of 150 items (90%). Pairwise:
few_shot/zero_shot 95%, few_shot/rag 93%, rag/zero_shot 91%. The arms differ
on a small margin of items, which is consistent with the tied macro-F1s and
means arm differences on the gate will need either many more items or a
paired significance test to detect at all.

### Qualitative: retrieval working, and retrieval ignored

**Working, causally traceable.** `implicit_hate-20412`
("rt now deport them!!: illegal immigrants arrested protesting at capitol").
Gold hate. Retrieval surfaced `guide-threats-and-incitement`, whose text names
deportation explicitly, and the model's rationale cites that guideline by its
content. Guideline to retrieval to citation to correct label, visible end to
end. rag correct, zero_shot wrong.

**Ignored, and it is the failure mode already in the related work.**
`mhs-30856` ("@Insayneasylum Stupid bitch"), gold NOT hate. Retrieval surfaced
`guide-profanity-without-target`, which states precisely that insults without
a protected-characteristic target are not hate speech. The model predicted
hate and justified it by pointing at the retrieved EXAMPLES instead. This is
Mohammadi et al. 2025 (LLMs not reliably following label definitions) and
Finding E in a single item: the guideline was retrieved correctly, placed in
the prompt correctly, and lost to the examples sitting beside it. Strongest
qualitative argument so far for the `k_examples_neg` arm and for guideline
salience being a prompt-design question rather than a retrieval one.

### Honesty numbers

Parse-failure rate 2.0 / 2.2 / 2.7% of runs, concentrated in MHS (11 of 14
affected items). No item lost its prediction: n=3 votes absorb a single
failure, and `uncertain` is 0 across all 450 results. Zero gate-consistency
normalisations. Repair loop bounded at 2 retries throughout.

Note that the rate rose over the run: the first 202 results carried 1 parse
failure in 606 calls (0.16%), the full 450 carry ~2.3%. The later portion ran
at 4 concurrent workers. Not diagnosed; flagged as possibly load-correlated.

### Infrastructure findings

**422 Validation Error under concurrent load.** At 8 workers the API returned
422 on roughly half of all requests, immediately and across all datasets. At 4
workers the rate fell to 2 in 744 calls, both of which succeeded on retry. A
minimal known-good request to the same endpoint returned 200 throughout, so
the server, slug and token were never the issue. Conclusion: the API refuses
requests under concurrency rather than rate-limiting cleanly, and 4 workers is
the safe ceiling. `BadRequest` was added to the client so non-429 4xx errors
fail fast instead of being retried five times with exponential backoff.

**Throughput is capped, not latency-bound.** Sequential zero_shot ran at
29.8 s/call; 4 workers on the same arm gave 24.6 s/call, about 1.2x. Wall time
in the concurrency probe equalled the maximum per-call time rather than the
sum, so the server genuinely overlaps requests, but it apportions a fixed
throughput budget across them. Under parallel load, latency also stops
tracking prompt length: rag (2700-char prompts) averaged 54.7 s against
few_shot (2000-char) at 62.4 s. So the ~3x cost premium for RAG arms measured
sequentially does not hold under concurrency, which matters for Phase 8
budgeting.

**The run took ~164 minutes for 248 item-arm pairs and was interrupted twice**
by network drops. The resumable design (append-only JSONL, completed pairs
skipped, failures never written) meant zero recomputation. Item-major ordering
means an interrupted run leaves all three arms partially covered rather

## 2026-07-23 — Phase 6.9: confirmation run on en_dev_eval_targets

**Setup:** manifest `targets_dev` [75b5bc475d39defa], `en_dev_eval_targets`
(56 items, all seven target_group labels at support 15-25), three arms, n=3
votes at T=1.0, `qwen3.5-122b-a10b` pinned with `allow_fallback=False`,
workers=4, kb_version 475869f9e2422969, prompt classify_v1-c74cb7ab. 168
item-arm pairs, 504 calls, 140 minutes. Two pairs failed with 422 on the first
attempt and succeeded on retry. Attribution assertion passed: 168/168 answered
by the pinned model. Zero parse failures, zero repairs, zero normalisations,
zero uncertain.

**What was being tested.** Slice 1 measured rag +0.131 over few_shot on
target_group macro-F1, with a paired bootstrap in which rag won 984 of 1000
resamples. That was n=26 scorable items over two labels (`gender`, `race`);
the other five fell below MIN_SUPPORT. This subset is label-stratified so all
seven labels are averaged.

### Result: the effect does not replicate

| arm | macro-F1 | micro-F1 | exact | Hamming | gold/pred labels |
|---|---|---|---|---|---|
| zero_shot | 0.664 | 0.697 | 0.268 | 0.168 | 2.23 / 1.66 |
| few_shot | 0.671 | 0.706 | 0.304 | 0.163 | 2.23 / 1.66 |
| rag | 0.669 | **0.726** | 0.304 | **0.156** | 2.23 / 1.75 |

Paired bootstrap, B=1000, label set fixed from the full sample:

| comparison | delta | 95% CI | p | favours a/b/tie |
|---|---|---|---|---|
| few_shot vs rag | +0.001 | [-0.046, +0.047] | 1.000 | 500 / 500 / 0 |
| zero_shot vs rag | -0.005 | [-0.057, +0.049] | 0.802 | 401 / 599 / 0 |
| zero_shot vs few_shot | -0.007 | [-0.028, +0.013] | 0.514 | 255 / 741 / 4 |

**rag - few_shot falls from +0.131 to +0.001, and the bootstrap splits
exactly 500/500.** Slice 1's effect was a small-sample artefact. Restricting
this subset to the same two labels Slice 1 could average gives rag - few_shot
= +0.049, roughly a third of the original, which is the expected shape of
regression to the mean rather than a contradiction between the two runs.

This is why the confirmation run existed. Recorded as a null result per the
D7 rule.

### But the aggregate error metrics still favour rag

macro-F1 weights every LABEL equally; micro-F1 weights every label DECISION
equally. They diverge here because one label behaves unlike the rest.

Per-label F1:

| label | support | zero_shot | few_shot | rag |
|---|---|---|---|---|
| gender | 19 | 0.688 | 0.727 | **0.765** |
| national_origin | 17 | 0.629 | 0.686 | **0.706** |
| race | 25 | 0.783 | 0.756 | **0.816** |
| sexual_orientation | 16 | 0.815 | 0.815 | 0.815 |
| religion | 15 | **0.897** | 0.867 | 0.875 |
| disability | 17 | **0.733** | 0.733 | 0.710 |
| **other** | 16 | 0.105 | 0.111 | **0.000** |

rag wins three labels, loses two narrowly, ties one, and scores exactly zero
on `other`. That single label costs rag 0.016 on the macro average. Excluding
it, rag 0.778 vs few_shot 0.761 vs zero_shot 0.758.

### Finding: `other` is unlearnable as currently defined

All three arms fail on it (0.105 / 0.111 / 0.000), so this is a taxonomy
property rather than a system property. The definition reads: "The text
targets an identifiable protected group not covered by the labels above." It
contains no content words describing what it targets. Two consequences:

1. It is semantically empty for the classifier - it is defined by exclusion
   from a list rather than by any property of its own.
2. It is unretrievable. A definition with no content words has no lexical or
   embedding signal for a query to match.

This is Phase 4 Finding G in its most extreme form. `def-target_group-gender`
was unreachable for misogyny queries because its text lacked "women" or
"female"; `def-target_group-other` has no content words at all. Definition
wording is the retrieval key, and this definition has no key.

Plausible mechanism for rag scoring 0.000 specifically: per-kind budgeted
retrieval returns the two most SIMILAR definitions, and a residual category
can never be most-similar to anything, so retrieval systematically supplies
the two nearest specific labels and crowds out the residual. Testable from
the stored `retrieved` hit ids; not yet run.

Candidate remedies, all taxonomy edits rather than code changes: enumerate
example groups inside the definition text (age, veteran status, caste,
appearance), rename it, or drop it and treat unmatched groups as an empty
label set. Any of these is one YAML edit plus a re-ingest with a frozen LLM,
so this is also a natural third case for the adaptability demo.

### Finding: the flagging bias, measured from the opposite direction

This subset is 100% gold-hateful by construction (fine-grained labels exist
only on hateful items), so gate macro-F1 over both classes is undefined and
is reported as n/a with a note. Recall on the present class:

| arm | TP | FN | recall |
|---|---|---|---|
| zero_shot | 50 | 6 | 0.893 |
| few_shot | 52 | 4 | 0.929 |
| rag | **54** | **2** | **0.964** |

**rag has the FEWEST false negatives here and the MOST false positives on
`en_dev_eval_main`.** These are the same bias measured from two directions: a
subset with no benign items cannot penalise over-flagging, so the disposition
that cost rag on `main` pays on `targets`. Neither number is a property of
the system alone; both are properties of the system crossed with the class
balance of the evaluation set. Worth stating explicitly, since it determines
how any single-number comparison should be read.

The guideline instrument shows the same reversal. On Slice 1 (70% benign),
deltas on gold=not-hate items were negative: profanity-without-target -10%,
negation-and-support -6%, counter-speech -50%. Here (100% hateful), every
delta is positive:

| guideline | implies | retrieved | rag / zero_shot | delta |
|---|---|---|---|---|
| guide-threats-and-incitement | hate | 18 | 94% / 78% | **+17%** |
| guide-reclaimed-slurs | not hate | 13 | 100% / 85% | **+15%** |
| guide-negation-and-support | not hate | 15 | 100% / 93% | +7% |
| guide-profanity-without-target | not hate | 21 | 90% / 90% | +0% |

Note that the "not hate" guidelines here have a 100% contradiction rate by
construction, so their positive deltas do not indicate that rag followed
them - it indicates that rag ignored them, which happens to be correct on
this subset. That is the point: retrieval does not appear to act as evidence
the model weighs, but as a push in one direction whose sign is fixed by the
overall composition of the retrieved context, which is dominated by hateful
examples (Phase 4 Finding E).

Severity agrees independently. rag is worst on `low` (F1 0.143 vs few_shot
0.471) and has the highest mean absolute rank error (0.72 vs 0.59): it pushes
intensity upward as well as pushing the gate upward.

### The self-consistency vote buys nothing on this subset

target_group macro-F1, n=1 (first run only) versus the n=3 vote:

| arm | n=1 | n=3 | delta |
|---|---|---|---|
| zero_shot | 0.667 | 0.664 | -0.003 |
| few_shot | 0.680 | 0.671 | -0.010 |
| rag | 0.668 | 0.669 | +0.002 |

On Slice 1 the vote bought +0.016 / +0.048 / +0.033. Here it buys nothing at
three times the API cost. Both measurements stand; the Slice 1 gain was
measured over 26 items and two labels, so it carries the same fragility as
the rag effect measured on the same data. Recommendation for Phase 8: n=1 for
exploratory sweeps, n=3 or n=5 reserved for final reported runs where
inter-run consistency is itself being measured.

### Consistency

| arm | gate | target_group | hate_type | severity |
|---|---|---|---|---|
| zero_shot | 0.782 | 0.925 | 0.722 | 0.763 |
| few_shot | 0.822 | 0.908 | 0.707 | 0.844 |
| rag | **0.478** | 0.869 | 0.718 | 0.746 |

rag's gate alpha of 0.478 is a prevalence artefact, not instability. rag
predicted hate on 54 of 56 items, and Krippendorff's alpha deflates under
skewed marginals because expected disagreement collapses toward zero. Raw
inter-run agreement on the gate is comparable across arms. Reported with this
caveat rather than as an instability finding.

### McNemar on the gate

All comparisons non-significant (p = 0.500 / 0.219 / 0.625) with 2 to 6
discordant pairs. At 56 single-class items the test has no power, as
expected.

### Cost model correction

The estimator predicted 42 minutes; the run took 140. Cause: the 20 s/call
figure was already wall-clock throughput measured across four workers, and
the formula divided it by the worker count a second time. Corrected to
30 s/call sequential, 17 s/call at 4 workers, with no division by workers.
Measured wall-clock throughput to date: 29.8 s/call sequential (Slice 1),
13.2 s/call at 4 workers (Slice 1, 744 calls / 164 min), 16.8 s/call at 4
workers (this run, 498 calls / 140 min). Throughput varies by roughly 25%
between runs, presumably with server load, so estimates should be treated as
a range rather than a figure.

### Verdict

The headline effect from Slice 1 does not survive proper label coverage and a
paired bootstrap. What survives is: a consistent advantage for rag on micro-F1
and Hamming loss, a directional bias toward flagging that is now confirmed
from both sides of the class balance, and one taxonomy defect (`other`) large
enough to control the macro average by itself.
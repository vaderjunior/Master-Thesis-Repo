# Build Plan — Adaptable Multi-Label Hate Speech Classifier (RAG)

**What this is.** The end-to-end plan from where the project stands *today* to all experiments run and thesis-ready results. Work top to bottom; every step has a checkpoint. Legend: ✅ done (verified in past sessions) · 🔜 current frontier · ⚠️ risky / needs lead time.

**How this differs from the prompt you wrote.** That prompt was written for a model with zero context, so it treats three things as open that we've already closed:

1. Temperature on `/api/ai-req` is **empirically confirmed working** (you run 1.0 today). The "build a temperature test" step is deleted; self-consistency can rely on real sampling.
2. The response shape is known: `response["content"]["text"]`, fences stripped before `json.loads()`. ✅
3. The model list in the prompt is outdated. Confirmed strong model: **`mistral-large-3-675b-instruct-2512`**. Medium (Qwen 3.5 122B) and fast (Llama 3.3 70B) slugs still need Network-tab confirmation — that stays in the plan.

If you ever want the "from-zero" version of this plan for a thesis appendix, ask — but for actual work, this calibrated one is the right one.

---

## Assumptions (correct me if any are wrong)

- **A1 — Codebase home.** The active code is `TUDAGPT_API.py` + scripts in VS Code. The earlier `hsrag v0.1.0` package is *reference only* — mine it for ideas, don't merge it. One home, one git history.
- **A2 — Hardware.** ≥8 GB VRAM assumed. That fits: `multilingual-e5-base` embeddings, `bge-reranker-v2-m3` in fp16, and `roberta-base` fine-tuning at batch 16 fp16. If you only have 6 GB: batch 8, and run the reranker on CPU (fine at KB scale).
- **A3 — API budget.** TUDaGPT has no published rate limit. Be polite (≤2 req/s), retry 429/500 with backoff, and cache every response (Phase 6.2) so nothing is ever paid for twice.
- **A4 — Methodology correction (important).** True constrained decoding (Outlines / XGrammar) needs token-level logit access. A remote HTTP API like TUDaGPT cannot provide that. The honest equivalent here is **schema-validated decoding**: Pydantic validation + bounded repair-retries + a reported parse-failure rate. Use that wording in the methodology chapter, and note "true constrained decoding is possible with local models" in limitations. This is a footnote-level correction, not a redesign.
- **A5 — Sign-off dependency.** Markus is on leave until ~late July. Taxonomy and severity-band decisions need his sign-off, but they are **not blockers**: the taxonomy lives in `taxonomy.yaml`, so post-sign-off changes = edit YAML + re-ingest KB, zero code changes. Build with a draft; flag every judgement call in a "for Markus" list.
- **A6 — Dataset lead times.** Implicit Hate needs an access request; DeTox access may route through PEASEC. Fire off both requests in week 1 regardless of build order.
- **A7 — Data governance.** All hate-speech text sent to an LLM goes through TUDaGPT only (TU-approved infra). Local models are for embeddings/reranking/encoder-baseline only — they never *generate* on this data, they only encode it, which stays on your machine either way.

---

## Milestones (thin vertical slices)

| Slice | Contents | Deliberately deferred |
|---|---|---|
| **0 ✅** | API round-trip, JSON parsing, temp confirmed, 5-run consistency PoC with a hardcoded guideline, `experiment_log.md` started | everything else |
| **1 🔜** | HateXplain dev subset (~150 items), hate gate + target_group, **no-RAG vs real ChromaDB retrieval**, macro-F1 for both | hate_type, severity, German, feedback, baselines |
| **2** | + hate_type + severity, full EN dev set, few-shot baseline | German, feedback |
| **3** | + German (GAHD, DeTox), multilingual retrieval sanity check | feedback |
| **4** | + feedback write-back machinery (SQ3 plumbing) | full experiment grid |
| **5** | Full SQ1→SQ3 experiment suite, difficult-cases analysis, packaging | — |

Slice 1 = Phases 0–6 on their minimum path. Everything after is widening, not new architecture.

---

## Project folder structure (grow into it — don't refactor everything on day one)

```
hsrag/
├─ .env                     # token — gitignored ✅
├─ config/
│  ├─ config.yaml           # paths, model slugs, retrieval defaults, API settings
│  └─ taxonomy.yaml         # labels + definition texts = single source of truth
├─ data/
│  ├─ raw/                  # downloads, never edited
│  ├─ interim/              # per-dataset unified records
│  └─ processed/            # frozen splits (train/dev/test × en/de), parquet
├─ kb/
│  ├─ records.jsonl         # atomic KB units (definitions/guidelines/examples)
│  └─ chroma/               # vector store — gitignored
├─ prompts/                 # versioned prompt templates: classify_v1.txt, ...
├─ src/hsrag/
│  ├─ client.py             # TUDaGPT client + MockClient (grows out of TUDAGPT_API.py)
│  ├─ data/                 # one loader per dataset + mapping code
│  ├─ kb.py                 # build / ingest / version the knowledge base
│  ├─ retrieve.py           # dense / bm25 / hybrid / MMR / rerank
│  ├─ classify.py           # prompt assembly, schema, self-consistency vote
│  ├─ evaluate.py           # metrics, significance tests, report generation
│  └─ feedback.py           # HITL write-back
├─ experiments/
│  ├─ manifests/            # one YAML per experiment run
│  └─ results/              # response cache, results.json, figures
├─ scripts/                 # thin CLIs: run_experiment.py, debug_retrieval.py, build_kb.py
├─ tests/                   # pytest smoke tests
├─ experiment_log.md        # your lab notebook ✅ — keep the discipline
└─ requirements.txt
```

Rule of thumb: move code into `src/hsrag/` at the moment a phase touches it, not before.

---

# Phase 0 — Consolidation & hygiene (~0.5 day · mostly ✅)

**0.1 — Rotate the API token ⚠️**
Goal: eliminate the exposed token. A live token was pasted in plaintext in an earlier chat; if you haven't rotated it since, do it now in the TUDaGPT/HAWKI profile settings, update `.env`, done. Checkpoint: old token returns 401 on `GET /api/user`, new one returns 200.

**0.2 — One codebase home**
Goal: single source of truth. Create the folder skeleton above around your current scripts; `git init` if not done; confirm `.gitignore` covers `.env` ✅, `kb/chroma/`, `data/`, `experiments/results/`. Checkpoint: `git status` shows no secrets or data.

**0.3 — `config/config.yaml`**
Goal: nothing hardcoded twice. Contents: `api.base_url`, `api.model_strong: mistral-large-3-675b-instruct-2512`, placeholders for `model_medium` / `model_fast`, `api.temperature: 1.0`, paths, `retrieval:` defaults (filled in Phase 4). Load with `pyyaml`. Checkpoint: a two-line script prints the loaded config.

**0.4 — `requirements.txt` with pins**
Start: `python-dotenv`, `pyyaml`, `requests`, `pydantic>=2`, `chromadb`, `sentence-transformers`, `pandas`, `pyarrow`, `scikit-learn`, `datasets`, `rank-bm25`, `krippendorff`, `statsmodels`, `scipy`, `matplotlib`, `tenacity`, `pytest`. Pin exact versions with `pip freeze` once installed. Checkpoint: fresh venv installs cleanly.

Common errors: Windows path separators (always `pathlib.Path`); UTF-8 (open every file with `encoding="utf-8"` — hate speech data is full of emoji and umlauts).

*Thesis mapping: feeds the Reproducibility section.*

---

# Phase 1 — TUDaGPT client hardening (~1 day · ~70% ✅)

You already have: request shape (`payload` wrapper, nested `{"text": ...}` content ✅), response parsing ✅, fence stripping ✅, temperature ✅, dotenv ✅.

**1.1 — Wrap into a client class**
Goal: one `TUDaGPTClient.complete(messages: list[dict]) -> str` used by everything downstream. Move your working call into `src/hsrag/client.py`. Checkpoint: existing PoC script works when rewired through the class.

**1.2 — Error handling + backoff**
Goal: unattended experiment runs survive hiccups. Map: 401 → fail loud ("rotate token?"); 403 → fail loud ("external API disabled"); 422 → log the field detail, fail that item; 429/500 → retry with exponential backoff (use `tenacity`: `@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))`), plus a hard timeout (~120 s). Checkpoint: unit test with `responses`/monkeypatch simulating a 500-then-200 sequence passes.

**1.3 — Call logging**
Goal: cost/usage evidence for the thesis. Log per call: timestamp, model, prompt chars, response chars, latency, retry count → append to `experiments/results/api_log.jsonl`. Checkpoint: log line appears after a live call.

**1.4 — MockClient**
Goal: develop and test the whole pipeline offline, for free. Same interface, returns canned valid JSON; add a `broken=True` mode returning malformed JSON to exercise the repair-retry path (Phase 5.3). Checkpoint: full Slice-1 pipeline runs end-to-end with `--backend mock`.

**1.5 — Confirm medium/fast model slugs 🔜**
Goal: fill the two config placeholders. You know the method: browser DevTools → Network → send a message in the TUDaGPT web UI with the medium (Qwen 3.5 122B) then fast (Llama 3.3 70B) model selected → inspect the `streamAI` request payload → copy the exact `model` string. Checkpoint: a live `complete()` with each slug returns 200 and sensible text.

Common errors: 422 almost always means the `content` nesting is a plain string instead of `{"text": ...}`; silent truncation on very long prompts — log prompt length and watch for it.

*Thesis mapping: Methodology §system / infrastructure.*

---

# Phase 2 — Data loading & unified taxonomy (~4 days · ⚠️ decision-heavy)

**2.1 — Fire off access requests NOW ⚠️**
Implicit Hate Corpus: request via the authors' form (GitHub repo → data access). DeTox: confirm whether PEASEC already holds it; if it routes through Markus, queue it and check whether a lab colleague can grant access during his leave. GAHD is openly on GitHub; HateXplain, Measuring Hate Speech, HateCheck and Multilingual HateCheck are on Hugging Face. Checkpoint: requests sent, tracked in a `data/ACCESS.md` note.

**2.2 — One loader per dataset → common `Record`**
Goal: uniform in-memory shape. Define:

```python
@dataclass
class Record:
    id: str; text: str; lang: str; source: str
    gate: bool | None
    target_groups: list[str] | None
    hate_types: list[str] | None
    severity: str | None
    raw: dict
```

`None` means "this dataset does not annotate this dimension" — that distinction matters enormously later (see 2.6). One file per dataset in `src/hsrag/data/`. Checkpoint: each loader prints record count + one sample; counts match published dataset sizes.

**2.3 — `taxonomy.yaml` (single source of truth)**
Goal: the unified scheme as *data*. Four dimensions: `hate` (bool), `target_group`, `hate_type`, `severity` (ordinal). Each label gets: `name`, `definition` (2–4 sentences), optional `guideline` notes. This file later *generates* the KB definition records — write the definitions as if an annotator will read them, because the LLM will. Checkpoint: YAML validates; a script lists all labels per dimension.

**2.4 — Mapping tables (the harmonisation core) ⚠️**
Goal: every source label → unified label, or explicitly excluded. One CSV per dataset: `source_label, unified_dimension, unified_label, rule, note`. Key judgement calls to make and flag for Markus:
- HateXplain: 3-way annotator majority; `hatespeech` → gate=true; **decide** whether `offensive` → gate=false (strict) — recommended, but document it; keep the original in `raw`.
- Measuring Hate Speech: continuous Rasch `hate_speech_score` → plot the histogram first; apply the dataset card's recommended hate threshold for the gate; band the hateful region into severity levels (e.g. terciles of the hateful mass) — **document the exact cut points** in the mapping table; sign-off flag.
- Implicit Hate: `explicit`/`implicit` and the 6 implicit types map into `hate_type`.
- German sets: same procedure once access lands.
Log unmapped/excluded labels with counts. Checkpoint: mapping-coverage table renders (per dataset: % mapped, % excluded, top exclusion reasons).

**2.5 — Frozen splits**
Goal: leakage-proof evaluation. Stratify on gate (+ primary target where possible), 70/15/15 train/dev/test per language, seed 42, save parquet to `data/processed/`. **Hard rules:** KB examples come from *train only*; test never touches the KB; the SQ3 sentinel set comes from dev. Checkpoint: split sizes + per-split label histograms saved as a figure; an assert-script verifies zero text overlap between splits.

**2.6 — Label masks in evaluation (subtle, important)**
Because datasets annotate different dimensions, unified evaluation must score each dimension **only on records whose gold actually annotates it** (`None` ≠ negative). Bake this into the metrics design now (implemented in Phase 6.3) so no dataset is punished for dimensions it never labelled.

**2.7 — Datasheet stub**
One markdown file: sources, licences, sizes, mapping decisions, known biases. Fill as you go. *Thesis mapping: Methodology §data + appendix (mapping table verbatim).*

Common errors: HF auth token needed for some sets (`huggingface-cli login`); Windows console choking on emoji (`PYTHONUTF8=1`); class imbalance shocks — look at the histograms before designing experiments.

---

# Phase 3 — Knowledge base construction (~1.5 days)

**3.1 — Atomic record schema**
One JSONL line per unit in `kb/records.jsonl`:

```json
{"id": "def-tg-gender-en", "kind": "definition", "dimension": "target_group",
 "label": "gender", "lang": "en", "text": "...", "source": "taxonomy_v1",
 "version": 1, "created_by": "taxonomy"}
```

`kind` ∈ {definition, guideline, example}. Examples additionally carry the full gold label set in metadata. Small and atomic is the point: one definition per record, one rule per record, one labelled example per record — that's what makes retrieval and later feedback-upserts surgical.

**3.2 — Generate records**
Definitions + guidelines: generated straight from `taxonomy.yaml` (this is why 2.3 doubles as KB authoring). Examples: sampled from the **train split only**, balanced per label, capped (start: ≤10 per label per language) — cap is a config value. Checkpoint: record counts per kind/dimension/lang printed and sane.

**3.3 — Ingest into ChromaDB**
`scripts/build_kb.py`: read records → embed → upsert into a Chroma persistent collection with all metadata; store `embedding_model` name and a **KB version hash** (sha256 of `records.jsonl`) in collection metadata. Re-ingest = delete + rebuild (idempotent). Checkpoint: `collection.count()` == record count; a metadata-filtered query (`kind="definition", lang="en"`) returns only definitions.

Common errors: Chroma persistence path must exist and be absolute on Windows; **embedding-dimension mismatch after swapping models** — the stored model name check prevents silently mixing embeddings.

*Thesis mapping: Methodology §knowledge base.*

---

# Phase 4 — Retrieval component (~3 days)

**4.1 — Dense retrieval (the Slice-1 must-have) 🔜**
Model: `intfloat/multilingual-e5-base` via `sentence-transformers`. **Critical detail:** E5 models require prefixes — embed KB records as `"passage: {text}"` and queries as `"query: {text}"`. Omitting this silently degrades quality and is the #1 E5 gotcha. Cosine similarity, tunable `top_k`. Checkpoint: for 10 hand-written probe texts, the top-3 retrieved definitions match the expected label (a misogynistic probe retrieves the `target_group: gender` definition, etc.). Keep the probes as a pytest.

**4.2 — Sparse retrieval (BM25)**
`rank-bm25` over lowercased whitespace tokens. Known limitation: German compounds hurt BM25 — note it as a limitation, hybrid mitigates it. Checkpoint: probe test again; expect it to win on exact-slur matches, lose on paraphrases.

**4.3 — Hybrid via Reciprocal Rank Fusion**

```python
def rrf(rank_lists, k=60):
    scores = {}
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + r + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

That's the whole algorithm — rank-based, so no score-scale headaches. Checkpoint: hybrid ≥ each single method on the probe set.

**4.4 — MMR for diversity**
Maximal Marginal Relevance re-orders the candidate list to balance relevance vs redundancy: pick `argmax λ·sim(query,d) − (1−λ)·max_sim(d, already_picked)`, λ≈0.7. Prevents retrieving five near-identical examples. ~15 lines; implement over the dense embeddings you already have.

**4.5 — Optional cross-encoder re-ranking**
`BAAI/bge-reranker-v2-m3` (multilingual, fp16 on GPU): retrieve top-30, re-rank, keep top-k. Defer until the SQ1 grid needs it.

**4.6 — Retrieval debug CLI**
`scripts/debug_retrieval.py "text" --strategy hybrid --k 5` → prints hits with kind/label/score. You will use this constantly; it's also where qualitative retrieval examples for the thesis come from.

**4.7 — Everything behind config**
`retrieval: {strategy, k, use_mmr, use_reranker, embedder}` in `config.yaml` / per-experiment manifest. This block *is* the SQ1 experiment surface.

*Thesis mapping: Methodology §retrieval; SQ1 design justification.*

---

# Phase 5 — Classifier with structured output (~2.5 days · PoC ✅)

**5.1 — Prompt template (versioned file: `prompts/classify_v1.txt`)**
System: role ("You are a content-moderation annotator…"), the four dimensions with allowed values, the JSON schema, "output ONLY JSON". User message, in this order: `## Definitions` → `## Guidelines` → `## Labelled examples` → `## Text to classify`. Reasoning field **first** in the JSON so the model reasons before committing to labels. Checkpoint: rendered prompt for one dev item reads sensibly top to bottom.

**5.2 — Pydantic schema**

```python
class Result(BaseModel):
    reasoning: str
    hate: bool
    target_group: list[TargetGroup] = []
    hate_type: list[HateType] = []
    severity: Severity | None = None

    @model_validator(mode="after")
    def gate_consistency(self):
        if not self.hate:
            self.target_group, self.hate_type, self.severity = [], [], None
        return self
```

Enums generated from `taxonomy.yaml` (taxonomy-as-data pays off again). Decision, documented: gate=false with populated sub-labels is *normalised + logged* as a soft failure, not a hard parse failure — keeps the pipeline robust while still measurable.

**5.3 — Validate + repair-retry loop**
Parse → on failure, re-send with `"Your previous output failed validation: {error}. Return ONLY the corrected JSON."` → max 2 retries → then log `parse_failure` for that item. Normalise label strings before enum validation (lowercase, strip, underscores) and count normalisations. Report the parse-failure rate — it's a thesis number (this is the "schema-validated decoding" of assumption A4). Checkpoint: MockClient in `broken=True` mode triggers exactly one repair round then succeeds.

**5.4 — Self-consistency vote**
n runs at temperature 1.0 (confirmed ✅). Aggregate per label: gate = majority bool; multi-label dims = keep each value appearing in > n/2 runs; severity = median of the ordinal. Ties on the gate → mark `uncertain`, count against consistency, resolve to not-hate for metrics (conservative; document). Store *all* raw runs, not just the vote — consistency metrics need them. Default n=5 for final results, n=3 during development (A3: budget).

**5.5 — One entry point for all conditions**
`classify(text, context_blocks) -> Result` where `context_blocks` is empty (zero-shot), static examples (few-shot), or retrieval output (RAG). Baselines and RAG share every line of code except context assembly — that's what makes SQ2 a clean comparison. Checkpoint: 20 dev texts through the RAG path → 100% valid JSON after retries, votes logged.

**🏁 Slice 1 lands here** (with the minimal manifest/metrics from Phase 6): HateXplain subset, gate + target_group, no-RAG vs ChromaDB-RAG, macro-F1 both. Log it in `experiment_log.md` — and mark it as the first entry using *real retrieval*, closing the hardcoded-guidelines caveat from your PoC entries.

*Thesis mapping: Methodology §classifier.*

---

# Phase 6 — Evaluation harness (~2.5 days)

**6.1 — Run manifests**
One YAML per experiment: dataset+split, languages, backend+model, prompt version, retrieval config, KB version hash, seed, n_consistency, item limit. An experiment *is* its manifest — no unlogged knob anywhere.

**6.2 — Response cache (the cost-saver)**
Key = sha256(model, prompt version, retrieval config hash, KB version, text, sample index) → JSON file under `experiments/results/cache/`. Re-running a manifest with unchanged inputs costs zero API calls; changed KB version naturally invalidates only what changed. Checkpoint: second run of the same manifest completes in seconds with identical results.

**6.3 — Metrics module**
`scikit-learn`: per-label P/R/F1, micro/macro-F1 (macro = headline), subset accuracy, Hamming loss. Structure: **gate metrics on all items; sub-label metrics only on gold-hateful items; every dimension masked to records whose gold annotates it** (Phase 2.6). Consistency: `krippendorff` (nominal α per label across the n raw runs) + paraphrase stability on a small paraphrase-pair set (build ~25 pairs when you reach Phase 9). Calibration: ECE with **vote share as the confidence proxy** (no logprobs from the API — state the proxy explicitly in the thesis). Checkpoint: metrics validated against a tiny hand-computed 5-item example in a pytest.

**6.4 — Significance helpers**
Gate comparisons: McNemar (`statsmodels.stats.contingency_tables.mcnemar`) on the paired correct/incorrect table. Macro-F1 deltas: paired bootstrap (resample test indices, B=1000, report CI + p). Two small functions in `evaluate.py`.

**6.5 — Report generator**
Manifest in → `results.json` + a markdown block appended to `experiment_log.md` (setup / table / interpretation / next steps — your existing notebook structure, automated). Figures to `experiments/results/figures/`.

*Thesis mapping: Methodology §evaluation; every results table is born here.*

---

# Phase 7 — Baselines (~3.5 days)

**7.1 — Zero-shot and few-shot (0.5 day)**
Both are `classify()` with different `context_blocks` (5.5). Few-shot: K static examples fixed by seed, same K as the RAG top-k for fairness. Checkpoint: both run on the EN dev subset via one manifest each.

**7.2 — Fine-tuned encoder ⚠️ (the chunky part)**
English: `roberta-base`; German: `deepset/gbert-base` (optionally XLM-R once for a cross-lingual line). Simplest sound design: **one classifier per dimension** — binary head for the gate, `problem_type="multi_label_classification"` heads for target_group/hate_type, ordinal-as-classification for severity; `MultiLabelBinarizer` aligned to taxonomy order. HF `Trainer`, fp16, batch 16 (A2), 3–5 epochs, early stopping on dev macro-F1. Train on the *same* train split the KB examples come from — same information budget as RAG, which is the fair-comparison argument for SQ2. Checkpoint: dev macro-F1 clearly above the majority-class baseline and in the literature ballpark for the dataset; training curves saved.

Common errors: none of the Windows/bitsandbytes pain applies (no quantisation needed at base size); if VRAM overflows, gradient accumulation ×2 at batch 8.

*Thesis mapping: Experiments §baselines.*

---

# Phase 8 — SQ1 experiment: representation & retrieval (~3 days · ⚠️ call-volume)

A full grid is too big; run it **staged**, carrying the winner forward:

| Stage | Vary | Fixed | Configs |
|---|---|---|---|
| A | knowledge kinds: {defs, guidelines, examples, all} | dense, k=5, e5-base | 4 |
| B | search: {dense, bm25, hybrid-RRF} | best kind K* | 3 |
| C | ± reranker, ± MMR | K*, best search S* | 4 |
| D | k ∈ {3, 5, 8} | best of C | 3 |
| E | embedder: {e5-base, BAAI/bge-m3} — **re-ingest KB per embedder** | best of D | 2 |

≈16 configs × ~300 EN dev items × n=3 ≈ **14k calls** — cache (6.2) and resumability make this survivable; run overnight batches. Log: one manifest per config. Outputs: config-vs-macro-F1 table; bar chart (kinds); line plot (k sweep). Decision: freeze the winner as `config.yaml` defaults for everything downstream.

*Maps to: SQ1 directly; Methodology gets the justification, Results gets the table.*

---

# Phase 9 — SQ2 experiment: RAG vs prompting (~2.5 days)

The ladder on the frozen config: **encoder / zero-shot / few-shot / RAG**, on EN test *and* DE test separately. n=5 self-consistency, 3 repeated runs → mean ± SD. Analysis that answers SQ2:
- headline: macro-F1 per system per language;
- **per-label deltas** (does retrieval lift rare labels? — this is the interesting finding either way);
- significance: McNemar on the gate, paired bootstrap on macro-F1, RAG vs each baseline;
- consistency: Krippendorff's α per system (does grounding stabilise outputs?);
- parse-failure and normalisation rates per system.
Figures: ladder bar chart per language; per-label delta heatmap (RAG − few-shot).

*Maps to: SQ2; the core Results table of the thesis.*

---

# Phase 10 — Feedback mechanism (~2 days)

**10.1 — Append-only log as source of truth**
`feedback_log.jsonl`: `{text, gold_labels, reason, round, timestamp}`. Never edited, only appended — the KB can always be rebuilt by replay.

**10.2 — `apply_feedback()`**
Correction → KB record with `kind=example, source=feedback, round=r`. **Upsert semantics: same text overwrites — last human decision wins** (as designed). Deterministic id: `fb-{sha1(text)[:12]}`. Bump the KB version hash per round.

**10.3 — Over-steering sentinel**
Fixed set of ~100 previously-correct dev items (from Phase 9's RAG run). Re-run after each feedback round; count flips-to-wrong = the over-steering metric. Checkpoint for the whole phase: manually correct one known-wrong item → that item is now retrieved for similar inputs → its prediction flips to correct → sentinel set unchanged.

*Maps to: Methodology §feedback; the mechanism half of SQ3.*

---

# Phase 11 — SQ3 experiment: feedback rounds (~4 days · ⚠️ most novel = least charted)

**Protocol per round r (make it resumable — `state.json` per round):**

```
batch B_r  = next ~100 unseen items from the feedback pool (train-side)
preds      = classify(B_r)
errors     = preds vs gold
apply_feedback(errors[:cap])        # cap ≈ 20/round keeps the curve readable
evaluate(held_out_H)                # fixed, never fed back
evaluate(sentinel_S)                # over-steering check
```

R ≈ 6–8 rounds. **Dry-run the whole loop with MockClient first** — the machinery has the most moving parts of the project; debug it for free.

Measurements → SQ3 and the two core definitions:
- **learning curve**: macro/micro-F1 on H vs cumulative corrections; summarise as area under the learning curve;
- **adaptability** = Δmacro-F1 per KB edit, LLM frozen — read directly off the curve;
- **rounds-to-recovery**: hold one hate_type out of `taxonomy.yaml` + KB from the start; at round k add its definition + corrections; count rounds until its per-label F1 crosses a floor (taxonomy-as-data makes this a YAML edit + re-ingest);
- **generalisation probe**: for each correction, evaluate on its nearest-neighbour unseen items (retrieval reuse) — does one correction fix a neighbourhood or just itself?
- **consistency across rounds**: α on H per round (do corrections destabilise?);
- **comparators**: no-update baseline; static-few-shot refresh (corrections pasted into the prompt instead of the KB — isolates the value of *retrieval*); same-budget encoder (re-fine-tune on cumulative corrections each round);
- **noisy expert variant**: flip ~15% of correction labels; re-run; how gracefully does it degrade?

Figures: learning curves (all comparators, one plot per language); recovery curve for the new label; over-steering flip counts per round.

*Maps to: SQ3; adaptability + consistency definitions; the novelty chapter of the thesis.*

---

# Phase 12 — Difficult-cases analysis (~1.5 days)

HateCheck + Multilingual HateCheck as functional test suites: run the final RAG system, zero-shot, and the encoder once each (template data → one manifest per system) → **per-functionality pass-rate table**. Then bucket residual test-set errors: start from HateCheck functionality categories, add manual codes (sarcasm, quoting-vs-endorsing, reclaimed slurs, coded language) in a small coding sheet (`id, text, gold, pred, category, note`). Pick 6–10 qualitative examples for the Discussion. *Maps to: the difficult-cases discussion that follows SQ2/SQ3.*

---

# Phase 13 — Reproducibility & packaging (~1.5 days)

Pin and record everything: `pip freeze` lock; model slugs incl. `mistral-large-3-675b-instruct-2512`; embedder name + revision; prompt files (versioned in `prompts/`); KB version hashes per experiment; seeds; the manifests themselves (they already capture most of this). Finish the datasheet. Produce the cost/usage report from `api_log.jsonl`. Write the README run-book: **one command per experiment** (`python scripts/run_experiment.py experiments/manifests/sq2_ladder_en.yaml`). Archive `experiments/results/`. *Maps to: Reproducibility section + submission artefact.*

---

## Effort & risk summary

| Phase | Days | Risk |
|---|---|---|
| 0 Consolidation | 0.5 | — (mostly ✅) |
| 1 Client | 1 | — (mostly ✅) |
| 2 Data & taxonomy | 4 | ⚠️ access lead times, mapping judgement calls |
| 3 Knowledge base | 1.5 | — |
| 4 Retrieval | 3 | E5 prefix gotcha; otherwise smooth |
| 5 Classifier | 2.5 | enum drift → normalise + log |
| 6 Eval harness | 2.5 | ⚠️ label-mask subtlety — unit-test early |
| 7 Baselines | 3.5 | encoder training is routine but time-consuming |
| 8 SQ1 | 3 | ⚠️ API volume → cache, dev subset, n=3 |
| 9 SQ2 | 2.5 | — |
| 10 Feedback | 2 | — |
| 11 SQ3 | 4 | ⚠️ most moving parts → Mock dry-run first |
| 12 Difficult cases | 1.5 | — |
| 13 Packaging | 1.5 | — |
| **Total** | **≈33 focused days** | |

**Top risks, mitigations built in:** (1) dataset access lag → requests fired in week 1 (2.1); (2) taxonomy sign-off timing → build on a draft, YAML re-ingest after Markus returns (A5); (3) API call volume → cache + subsets + n=3 in development (6.2, A3); (4) SQ3 complexity → resumable runner + Mock dry-run (11); (5) evaluation label-masks → pytest with hand-computed values (6.3).

## Phase → thesis mapping

| Phase | Thesis section |
|---|---|
| 2 | Methodology: data & unified taxonomy; appendix: mapping tables; datasheet |
| 3–5 | Methodology: system (KB, retrieval, classifier) |
| 6 | Methodology: evaluation design |
| 7, 9 | Experiments & Results: SQ2 (baseline ladder) |
| 8 | Experiments & Results: SQ1 |
| 10–11 | Experiments & Results: SQ3; adaptability/consistency findings |
| 12 | Discussion: difficult cases |
| 13 | Reproducibility section + submission artefact |

## Master checklist

- [x] Slice 0 — API round-trip, temp confirmed, consistency PoC, lab notebook
- [ ] Phase 0 — consolidation (finish: **token rotation**, config.yaml, pins)
- [ ] Phase 1 — client hardening (finish: backoff, Mock, **medium/fast slugs**)
- [ ] Phase 2 — data & taxonomy ⚠️ *(start access requests immediately)*
- [ ] Phase 3 — knowledge base in ChromaDB
- [ ] Phase 4 — retrieval (dense first) 🔜
- [ ] Phase 5 — classifier + schema-validated output → **🏁 Slice 1**
- [ ] Phase 6 — evaluation harness
- [ ] Phase 7 — baselines
- [ ] Phase 8 — SQ1 experiment
- [ ] Phase 9 — SQ2 experiment
- [ ] Phase 10 — feedback mechanism
- [ ] Phase 11 — SQ3 experiment
- [ ] Phase 12 — difficult-cases analysis
- [ ] Phase 13 — reproducibility & packaging
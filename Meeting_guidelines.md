# Prototype walkthrough + meeting script

Part 1 is the refresher, so you know it cold.
Part 2 is the speech, written to be read aloud.

---

# PART 1: WHAT YOU BUILT

## The shape of the system

```
comment in
  -> search the knowledge base
       (label definitions + annotation guidelines + labelled examples)
  -> those go into the prompt
  -> LLM returns labels as JSON
  -> expert corrects what's wrong
  -> correction goes back into the knowledge base
  -> no retraining, anywhere
```

**Four output dimensions**, defined in `taxonomy.yaml`:

| dimension | type | labels |
|---|---|---|
| hate (gate) | binary | true / false |
| target_group | multilabel | race, religion, gender, sexual_orientation, disability, national_origin, other |
| hate_type | multilabel | explicit, incitement, inferiority, irony, stereotyping, threatening, grievance |
| severity | ordinal | low / medium / high |

Legal is defined but **not scored**. The data was too thin. BoTox may change that.

**The adaptability claim:** edit the taxonomy or the knowledge base, re-ingest, run again with the same frozen LLM, measure the change in macro-F1. That delta is the evidence.

## The knowledge base: 307 records, three kinds

| kind | count | from | written by |
|---|---|---|---|
| definitions | 18 | `taxonomy.yaml` | you |
| guidelines | 11 | `guidelines.yaml` | you, by hand |
| examples | 278 | train splits of the 5 datasets | dataset authors |

Small on purpose. Retrieval only ever puts about 9 records into a prompt, so what matters is coverage per label, not volume.

## Where the datasets come in

Five datasets, chosen so that **together** they cover the taxonomy. No single one annotates everything.

| dataset | lang | gives you |
|---|---|---|
| HateXplain | EN | gate + target_group |
| Measuring Hate Speech | EN | severity (only graded source) |
| Implicit Hate | EN | hate_type, coded/implicit cases |
| GAHD | DE | gate, adversarial |
| DeTox | DE | gate + light target_group |
| HateCheck | EN+DE | diagnostic only, never sampled from |

**Each dataset lives two lives.** The **train** split is where KB examples get sampled from. The **test** split is what you score on. Splits are frozen, seeded, stratified on gate, with a leakage check, because if a test comment ended up in the KB the system would be retrieving the answer to the question it's being graded on.

## The tech choices, and why

**BGE-M3** for embeddings. Three reasons: it's multilingual with a properly aligned shared space, so a German query can reach an English record; it's MIT-licensed and self-hosted, which matters given the DeTox governance question; and it runs on CPU so it doesn't compete for VRAM. You verified the cross-lingual property directly (EN/DE misogyny pair 0.887 cosine, unrelated pairs ~0.35). One illustrative triple, not a benchmark, but enough to justify the design decision.

**The asymmetric design that follows from it:** definitions and guidelines written once in English, no language filter. Examples kept per language, filtered. Concepts transfer across languages, surface forms don't. A German slur has no English equivalent carrying the same weight.

**ChromaDB** as the vector store, with a version hash on every build.

**TUDaGPT** as the LLM endpoint, chosen for data governance rather than convenience. Pinned to `qwen3.5-122b-a10b`, **fallback off** for formal runs. A frozen model is the premise of the adaptability claim, so a silent fallback would make results unattributable. `active_model`, `kb_version` and `prompt_version` get stamped on every result record.

**Temperature 1.0**, prediction is a self-consistency vote over n runs (3 dev, 5 formal). Consistency is Krippendorff's alpha across those same runs, which is why there's no temperature-zero arm: alpha at T=0 is meaningless by construction.

**Retrieval:** dense (BGE-M3) and BM25 as two channels, fused with RRF at k=60, taken from Cormack et al. 2009 and cited rather than tuned. Each record kind gets its own guaranteed slots rather than all competing in one ranking.

## How evaluation is set up

The English test split is 12,041 items, which isn't runnable at multiple votes across multiple arms. So: **frozen seeded subsets**, built once, shared by every system including the later encoder baselines. Shared items are what make comparisons paired, which McNemar and the paired bootstrap need.

The English dev pool splits into three disjoint slices: one to tune on, one to report from, one for the feedback rounds. One dev set can't be tuning set, headline set and adaptability set at once without going circular.

**A naturally-distributed subset can't report the fine-grained dimensions.** On a 299-item natural sample, the thinnest hate_type label had 2 items. So the fine-grained dimensions get **label-stratified subsets** instead, every label clearing 15 on dev and 25 on test. Defensible because those dimensions are only ever scored on hateful items anyway, so the set was never a natural distribution in any meaningful sense. Any label with under 10 support is reported with its support and excluded from the macro average.

---

# PART 2: WHAT YOU FOUND

**1. The burial problem.** Naive dense retrieval over the whole KB returned one definition and zero guidelines across six probes. Examples dominated completely. The cause is that the KB mixes two registers: examples are raw social media text, definitions and guidelines are formal descriptive prose, and queries are raw social media text too. The abstract records describe hate instead of being hate, so they lose the similarity race. Property of mixed-register collections, not a flaw in the embedder. Fixed by giving each kind its own guaranteed slots. **This is what makes SQ1 a real research question rather than an implementation detail.**

**2. Definition wording is a retrieval key, not just documentation.** The gender definition says "gender, misogyny, trans people" and never contains "women" or "female", so "women are too stupid to vote" cannot reach it by either channel. Meanwhile "stupid" pulls the disability definition, which says "mental disability". Left deliberately unfixed as the rewording demo: one YAML edit, frozen model, measurable change.

**3. Retrieval supplies only hateful evidence, even for non-hateful input.** Both not-hate probes got 5 of 5 hateful examples under every strategy. Structural: 20 negatives per language against 278 examples can never win a similarity race. Direct false-positive pathway, and it lines up with the known LLM over-flagging problem.

**4. Hybrid did not beat plain dense.** Tied on 5 of 6 probes. n=6 is noise, so no claim is made from it, but the arithmetic behind the fusion is worth knowing: at these settings RRF acts as a **consensus gate rather than a blend**, since anything found by both channels outranks anything found by one, regardless of rank. So a noisy channel doesn't merely contribute to a bucket, it can determine it. And BM25 on the definitions bucket is close to noise: the harmless control probe about the weather scored higher against the definitions than any hateful probe did.

**5. German fine-grained scoring isn't possible with existing data.** In the 300-item German test subset, 15 items have target_groups annotated and **zero** have a positive label. Not a sampling artifact, the whole German split has about 30. So German is gate-only for scoring. The classifier still outputs all four dimensions for German input; scoring masks the ones without gold. Stated as a finding about German resource density, not as an apology.

**6. Latency is the practical bottleneck.** RAG calls run ~93s each, and four parallel workers bought only 1.2x, which points at a server-side cap.

---

# PART 3: BOTOX, SHORT VERSION

- It **is** the Kums dataset. Same authors, same WOAH 2025 paper. Public, so the access question is closed.
- **Can't replace DeTox.** No gate or target labels, and `class_0` means "not criminally relevant", not "not hate". A comment can be hateful and legal, so mapping it onto the gate would be wrong.
- **What it unlocks:** the legal dimension. DeTox gave ~16 usable legally-labelled comments. BoTox has 1,190, ~7% multi-label, which is exactly the argument the Background chapter makes. Plus prosecutor-trained annotation guidelines, better material than the 11 rules written by hand.
- **The catch:** ~58% of BoTox is drawn from DeTox, which is already loaded and split. Cross-dataset dedup would have to be re-run or there'd be leakage between the DeTox train split and a BoTox eval set.

---

# PART 4: THE SPEECH

Read this. Pause where the paragraphs break.

---

Thanks for the time. I thought I'd walk you through the prototype I've built, share a few things I found that I think are actually interesting, and then talk about next steps. I'll keep the walkthrough quick and spend more time on the findings. I'll also send you the first draft of the methodology and experiments sections next week.

**The basic shape.** A comment comes in. The system searches a knowledge base and pulls three kinds of thing: label definitions, annotation guidelines, and labelled examples. Those go into the prompt, the model reads them, and returns the labels as JSON. Then an expert corrects anything wrong and the correction goes back into the knowledge base. No retraining anywhere.

Four output dimensions at the moment. A binary hate gate, target group and hate type as multilabel, and severity as ordinal. Legal is defined but not scored yet, and I'll come back to that.

The knowledge base is deliberately small, about 300 records. Eighteen definitions generated from my taxonomy file, eleven guidelines I wrote by hand, and the rest are examples sampled from the training splits of my datasets. The point isn't volume. Retrieval only ever puts about nine records into a prompt, so what matters is coverage per label.

**On data.** I'm using five datasets, picked so that together they cover the taxonomy, because no single one annotates everything. HateXplain gives gate and target group, Measuring Hate Speech gives severity and is the only graded source I found, Implicit Hate gives hate type, and for German it's GAHD and DeTox. HateCheck I'm holding back purely as a diagnostic suite, never sampled from.

Each dataset lives two lives. The train split is where knowledge base examples come from, the test split is what I score on. The splits are frozen and seeded, and I ran a leakage check, because if a test comment ended up in the knowledge base the system would be retrieving the answer to the question it's being graded on.

**On the retrieval stack.** Embeddings are BGE-M3. I picked it for three reasons. It's multilingual with a properly aligned shared space, so a German query can reach an English record. It's MIT licensed and self-hosted, which matters given the DeTox data governance question. And it runs on CPU, so it doesn't compete for VRAM.

I checked the cross-lingual property directly. An English misogyny sentence and its German translation, with no shared words, scored 0.887 cosine, against about 0.35 for unrelated pairs. That's one illustrative triple, not a benchmark, but it justified the design decision.

That decision is that definitions and guidelines are written once in English and aren't language filtered, while examples are kept per language and are filtered. The reasoning is that concepts transfer across languages but surface forms don't. A German slur has no English equivalent carrying the same weight.

Vector store is ChromaDB. The model is pinned to Qwen 3.5 122B on TUDaGPT with fallback explicitly switched off for formal runs. That last part matters, because a frozen model is the premise of the whole adaptability claim. If a call silently fell through to whatever else happened to be online, the result wouldn't be attributable to anything. So I stamp the active model, the knowledge base version hash and the prompt version hash onto every result record.

Temperature is 1.0 and the prediction is a self-consistency vote over several runs. Consistency gets measured as Krippendorff's alpha across those same runs, which is also why I didn't add a temperature-zero arm. Alpha at zero would be meaningless by construction.

**Now the findings, which is the part I actually want your view on.**

The first one turned SQ1 from an implementation detail into a real question. With naive dense retrieval over the whole knowledge base, across six probes I got one definition and zero guidelines. Examples dominated completely.

The reason is that the knowledge base mixes two registers. Examples are raw social media text. Definitions and guidelines are formal descriptive prose. My queries are also raw social media text, so they land next to the examples every time. The definitions aren't irrelevant, they just describe hate instead of being hate, and they lose the similarity race. That's a property of mixed-register collections rather than a flaw in the embedding model. I solved it by giving each kind its own guaranteed slots.

The second finding is my favourite. Definition wording is a retrieval key, not just documentation. My gender definition says gender, misogyny, trans people. It never contains the word women or female. So the query "women are too stupid to vote" can't reach it, by embeddings or by keyword search. And meanwhile the word stupid pulls the disability definition, because that one says mental disability.

I've deliberately left that unfixed. Rewriting the definition, re-ingesting and re-running the same probe is a clean demonstration of the adaptability claim on my own workflow. One YAML edit, frozen model, measurable change.

The third one is the one that worries me. Retrieval supplies only hateful evidence, even when the input isn't hateful. Both of my not-hate probes got five out of five hateful examples under every strategy. It's structural: there are twenty negatives per language against nearly three hundred examples, so negatives can't win a similarity race. That's a direct false-positive pathway into the classifier, and it lines up with the known result that LLMs over-flag anything mentioning a protected group.

Fourth, hybrid retrieval did not beat plain dense. It tied on five of six probes. Six probes is noise-level evidence so I'm not making a claim from it, the real comparison runs on three hundred items. But I did work out why the fusion behaves the way it does. At my settings RRF acts as a consensus gate rather than a blend: anything found by both channels outranks anything found by only one, regardless of rank. So a noisy channel doesn't just contribute to a bucket, it can determine it. And BM25 on the definitions bucket is close to noise. My harmless control probe about the weather scored higher against the definitions than any of the hateful probes did.

The fifth is a scope change. German fine-grained scoring isn't possible with the data that exists. In my three-hundred-item German test subset, fifteen items have target groups annotated and zero have a positive label. That's not a sampling artifact, the whole German split has about thirty. So German is gate-only for scoring. The classifier still outputs all four dimensions for German input, the scoring just masks the ones without gold. I'd rather state that as a finding about German resource density than pretend I can report a number.

**On BoTox, which you asked me to look at.** It's the Kums dataset, same authors, and it's public, so that access question is closed.

It can't replace DeTox. It has no gate or target labels, and its class zero means not criminally relevant rather than not hate. A comment can be hateful and perfectly legal, so mapping class zero onto my gate would be wrong.

What it does do is make the legal dimension viable. DeTox gave me about sixteen usable legally-labelled comments, which is why legal is currently defined but not scored. BoTox has just under twelve hundred, with about seven percent multi-label, which is exactly the argument my background chapter makes. It also ships prosecutor-trained annotation guidelines, which would be better knowledge base material than the eleven rules I wrote myself.

One catch. Roughly fifty-eight percent of BoTox is drawn from DeTox, which I already have loaded and split. So I'd have to re-run my cross-dataset deduplication or I'd get leakage between my DeTox train split and a BoTox evaluation set.

**Next steps.** Finishing the classifier evaluation, then the encoder baseline, then the SQ1 retrieval experiments on three hundred items, which is where the real dense-versus-hybrid comparison actually happens.

The practical bottleneck is latency. RAG calls are running about ninety seconds each, and going to four parallel workers only bought me about 1.2x, which points at a server-side cap rather than anything in my code.

And as I said, I'll send you the first draft of the methodology and experiments sections next week.

---

# PART 5: YOUR QUESTIONS

Four, general, and let him lead.

1. On German being gate-only, does that scope reduction seem right to you, or is it worth chasing more German data first?

2. On BoTox, would you turn legal into a scored fifth dimension, or keep it as background material in the knowledge base?

3. On the latency, is it worth moving to the PEASEC servers you mentioned?

4. For the methodology chapter, is there anything specific you want to see in it, or a past thesis you'd point me at as a model?
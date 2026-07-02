# Experiment Log

Running notes for thesis experiments. Append new entries at the bottom, newest last.
Each entry: what I tested, exact setup, raw results, what it means, what's next.

---

## Exp 001 — Guideline tightening reduces target_group/hate_type variance

**Date:** 2026-06-28
**Script:** `TUDAGPT_API.py`
**Model:** `mistral-large-3-675b-instruct-2512` (via HAWKI)
**Temperature:** 1.0
**Runs:** 5x per condition (same text, same prompt, looped)

### Setup

Fixed test text (no RAG retrieval yet, just hardcoded guidelines + example in prompt):
> "Go back to your own country, nobody wants you here."

Output schema: `{hate, target_group, hate_type, legal, severity}`

### Condition A — before guideline fix

Guidelines block did not define the boundary between `national_origin`, `race`, and `ethnicity`.

Results across 5 runs:

| Run | target_group | hate_type | severity |
|---|---|---|---|
| 1 | [national_origin] | [insult] | medium |
| 2 | [national_origin, race] | [insult, exclusion] | medium |
| 3 | [national_origin, race] | [insult] | medium |
| 4 | [national_origin, race] | [insult, exclusion] | medium |
| 5 | [national_origin, ethnicity] | [insult] | medium |

- `target_group` agreement: 1/5 runs matched the majority exactly (if majority = adds a second group, then 1/5 had no second group — depends how you define agreement, recompute properly later)
- `hate_type` flipped 2/5 runs (extra "exclusion" label appeared)
- `hate`, `legal`, `severity` were stable (always True / [] / medium)

### Condition B — after guideline fix

Added one line to the GUIDELINES block, directly under the target_group bullet:

> "national_origin" = citizenship/country of origin only. "race"/"ethnicity" = use only if explicitly invoked (skin color, ancestry, etc.), not implied by nationality alone.

Results across 5 runs:

| Run | target_group | hate_type | severity |
|---|---|---|---|
| 1 | [national_origin] | [insult] | medium |
| 2 | [national_origin] | [insult] | low |
| 3 | [national_origin] | [insult] | medium |
| 4 | [national_origin] | [insult] | medium |
| 5 | [national_origin] | [insult] | medium |

- `target_group`: **5/5 exact agreement** (was 1/5 majority before, now fully locked)
- `hate_type`: **5/5 exact agreement** (was 3/5, now fully locked)
- `severity`: new flip appeared, 1/5 runs gave "low" instead of "medium" — wasn't visible before because the other two fields were noisy

### Interpretation

- A single clarifying sentence in the guidelines (no model change, no retraining) fully resolved the `target_group` and `hate_type` disagreement for this text.
- This is a clean, small demonstration of the thesis's core adaptability claim: tightening the retrieval base / guidelines changes model consistency directly.
- The `severity` flip is a new, smaller inconsistency that was masked by the bigger one. Worth chasing the same way later (define low vs medium boundary more precisely) — see open questions.
- Caveat: N=5 runs, single text example, no real RAG retrieval yet (guidelines hardcoded directly into the prompt, not pulled from a knowledge base). This is a proof-of-concept on the mechanism, not yet a formal evaluation result.

### Next steps from this experiment

- [ ] Recompute agreement numbers properly (define exact metric: pairwise agreement? exact-match-to-majority? — decide and use consistently from now on)
- [ ] Chase the severity flip the same way (tighten low vs medium definition, rerun 5x)
- [ ] Move guidelines from hardcoded prompt text into actual retrieval base (ChromaDB) so this becomes a real RAG-vs-no-RAG test, not just prompt engineering
- [ ] Test a harder/more ambiguous text (sarcasm, reclaimed slur, quoting-not-endorsing) to see if the same fix pattern holds or breaks down
- [ ] Decide whether 5 runs is enough or whether to standardize on a larger N (e.g. 10) for future consistency tests

---

<!-- Add Exp 002 below this line when ready -->
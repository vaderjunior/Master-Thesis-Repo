"""
build_kb_rcords.py
Generate kb/records.jsonl from three sources:
  1. definitions - auto-generated from taxonomy.yaml
  2. guidelines  - from guidelines.yaml (hand-authored policy)
  3. examples    - sampled from TRAIN splits only, per label, per language,
                   capped, balanced with negatives
  + DeTox legal illustrations (illustrative_only=true) from train raw.

The JSONL is the SOURCE OF TRUTH. Chroma (3.4) is a disposable index rebuilt
from this file.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

TAXONOMY = Path("config/taxonomy.yaml")
GUIDELINES = Path("config/guidelines.yaml")
GUIDELINES_DE = Path("config/guidelines_de.yaml")
PROCESSED = Path("data/processed")
OUT = Path("kb/records.jsonl")

EXAMPLES_PER_LABEL = 10  # mirror config; Phase 8 ablates this

# BoTox legal examples. Higher than EXAMPLES_PER_LABEL because the legal
# dimension has only one source dataset and no cross-lingual reach: the
# English definitions and guidelines cannot help a German criminal-law
# question the way they help a target_group question. 15 per class plus
# negatives gives the German example pool real content instead of the 20
# balanced records it has now.
LEGAL_EXAMPLES_PER_LABEL = 15
LEGAL_NEGATIVES = 20


def gen_definitions() -> list[dict]:
    """One record per gate + per label of every dimension."""
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    records = []

    for dim_name, dim in tax["dimensions"].items():
        dtype = dim["type"]

        if dtype == "binary":
            # the gate definition - label is null
            records.append({
                "id": f"def-{dim_name}",
                "kind": "definition",
                "dimension": dim_name,
                "label": None,
                "lang": "en",
                "text": " ".join(dim["definition"].split()),
                "source": "taxonomy_v1",
                "meta": {},
            })

        elif dtype in ("multilabel", "ordinal"):
            labels = dim["labels"]
            # ordinal labels can be a list or a dict; normalise to dict
            if isinstance(labels, list):
                labels = {l: {} for l in labels}
            for label_name, label in labels.items():
                definition = (label or {}).get("definition", "")
                if not definition:
                    continue  # e.g. ordinal labels without their own text
                records.append({
                    "id": f"def-{dim_name}-{label_name}",
                    "kind": "definition",
                    "dimension": dim_name,
                    "label": label_name,
                    "lang": "en",
                    "text": " ".join(definition.split()),
                    "source": "taxonomy_v1",
                    "meta": {},
                })

    return records


def gen_guidelines() -> list[dict]:
    """One record per hand-authored guideline, English and German.

    German guidelines are derived from the BoTox annotation guidelines
    (Kums et al. 2025) rather than self-authored, and are kept in a separate
    file so that provenance stays visible. They are also the first German
    text in the guideline bucket: Phase 4 Finding C found BM25 inert for
    German queries because every guideline was English, so a German query had
    no lexical material to match against.
    """
    records = []
    for path, source, lang in ((GUIDELINES, "guidelines_v1", "en"),
                               (GUIDELINES_DE, "guidelines_de_botox", "de")):
        if not path.exists():
            continue
        g = yaml.safe_load(path.read_text(encoding="utf-8"))
        for rule in g["guidelines"]:
            records.append({
                "id": f"guide-{rule['id']}",
                "kind": "guideline",
                "dimension": rule.get("dimension"),
                "label": None,
                "lang": rule.get("lang", lang),
                "text": " ".join(rule["text"].split()),
                "source": source,
                "meta": {},
            })
    return records

def clean(val):
            """Normalise parquet nulls (NaN) back to None, arrays to lists."""
            if val is None:
                return None
            if isinstance(val, float) and pd.isna(val):
                return None
            if hasattr(val, "tolist"):   # numpy array -> list
                return val.tolist()
            return val

def gen_examples() -> list[dict]:
    """Sample examples from TRAIN splits only.

     The plan:

    1. Load data/processed/en_train.parquet and de_train.parquet.
       (raw is a JSON string - json.loads it when you need meta.)

    2. For each language, for each target_group / hate_type / severity label,
       collect train rows that carry that label, and sample up to
       EXAMPLES_PER_LABEL of them. Prefer rows with RICH labels (more
       dimensions populated) - sample those first, fill with simpler ones.

    3. INCLUDE NEGATIVES: also sample gate=False rows per language, with a
       bias toward offensive-but-not-hate (the valuable negatives). These
       teach the model where the line is.

    4. Build one record per sampled row:
         id    = f"ex-{source}-{origid}"   (origid: strip the 'source-' prefix
                                            from record id, or use the id)
         kind  = "example"
         dimension = None, label = None
         lang  = row.lang
         text  = row.text
         source = f"{row.source}-train"
         meta  = {"gate": ..., "target_groups": ..., "hate_types": ...,
                  "severity": ..., "illustrative_only": False}

    5. Dedup: a row might be sampled for two labels - keep one record per id.

    Return the list. Print a per-label, per-language count so you can see the
    sparse cells (some DE x label combos will be empty - that's the stated
    German limitation, print the gap).
    """
    records = []
    seen_ids = set()          # dedup: a row can be sampled for two labels
    coverage = defaultdict(int)   # (lang, label) -> count, for the gap table

    for lang in ["en", "de"]:
        path = PROCESSED / f"{lang}_train.parquet"
        df = pd.read_parquet(path)

        # richness = how many dimensions this row populates. Prefer rich rows.
        def richness(row):
            r = 0
            if row.target_groups is not None:
                r += 1
            if row.hate_types is not None:
                r += 1
            if row.severity is not None:
                r += 1
            return r


        df = df.assign(rich=[richness(r) for r in df.itertuples(index=False)])

        def add_row(row, label_key):
            """Turn one dataframe row into an example record (once)."""
            rec_id = f"ex-{row.id}"
            coverage[(lang, label_key)] += 1
            if rec_id in seen_ids:
                return
            seen_ids.add(rec_id)
            records.append({
                "id": rec_id,
                "kind": "example",
                "dimension": None,
                "label": None,
                "lang": lang,
                "text": row.text,
                "source": f"{row.source}-train",
                "meta": {
                    "gate": bool(row.gate) if row.gate is not None else None,
                    "target_groups": clean(row.target_groups),
                    "hate_types": clean(row.hate_types),
                    "severity": clean(row.severity),
                    "illustrative_only": False,
                },
            })

        # --- positive examples: per label of each multilabel/ordinal dim ---
        for col, is_list in [("target_groups", True),
                             ("hate_types", True),
                             ("severity", False)]:
            # gather the distinct labels present in this column
            # gather the distinct labels present in this column
            labels = set()
            for row in df.itertuples(index=False):
                val = getattr(row, col)
                if val is None:
                    continue
                if is_list:
                    # val may be a numpy array or list; skip non-iterables
                    if isinstance(val, (list, tuple)) or hasattr(val, "__iter__"):
                        labels.update(v for v in val if isinstance(v, str))
                else:
                    if isinstance(val, str):
                        labels.add(val)
        
            for label in sorted(labels):
                # rows carrying this label, richest first
                def has_label(row):
                    val = getattr(row, col)
                    if val is None:
                        return False
                    if is_list:
                        if not (isinstance(val, (list, tuple)) or hasattr(val, "__iter__")):
                            return False
                        return label in val
                    return val == label

                candidates = [r for r in df.itertuples(index=False) if has_label(r)]
                candidates.sort(key=lambda r: r.rich, reverse=True)
                for row in candidates[:EXAMPLES_PER_LABEL]:
                    add_row(row, f"{col}:{label}")

        # --- negatives: gate=False rows (the offensive-but-not-hate line) ---
        negatives = [r for r in df.itertuples(index=False) if r.gate is False]
        # richest-first here means rows that were annotated on other dims too
        negatives.sort(key=lambda r: r.rich, reverse=True)
        for row in negatives[:EXAMPLES_PER_LABEL * 2]:  # a few more negatives
            add_row(row, "gate:false")

    # --- print the coverage / gap table ---
    print("\n  example coverage (lang, label -> count):")
    for (lang, label), n in sorted(coverage.items()):
        print(f"    {lang}  {label:30} {n}")

    return records

def gen_legal_illustrations(existing: list[dict]) -> list[dict]:
    """DeTox train comments flagged with an StGB paragraph -> illustrative
    examples. illustrative_only=True, paragraph in meta.stgb. These ground
    the legal motivation without being a scored dimension (Option 2).

    DEDUP (added Phase 4.5): a flagged comment may ALREADY be in the KB as a
    balanced example, under a different id (ex-detox-<x> vs
    ex-detox-legal-<x>). That produced byte-identical duplicate records, and
    retrieval returned the same German comment in two of five example slots.
    When a flagged comment is already present, merge meta.stgb onto the
    existing record and emit nothing new: the scarce legal annotation is
    kept, the duplicate is not. The surviving record stays
    illustrative_only=False - it is a genuine scored example that happens to
    carry a paragraph flag.
    """
    path = PROCESSED / "de_train.parquet"
    df = pd.read_parquet(path)
    df = df[df["source"] == "detox"]

    by_id = {r["id"]: r for r in existing}

    records = []
    merged = 0
    for row in df.itertuples(index=False):
        raw = json.loads(row.raw)
        paragraphs = raw.get("legal_paragraphs", {})
        # a paragraph "fires" if > 0.5 of annotators flagged it
        flagged = [p.replace("p_", "") for p, v in paragraphs.items() if v > 0.5]
        if not flagged:
            continue
        stgb = ",".join(flagged)

        base_id = f"ex-{row.id}"
        if base_id in by_id:
            by_id[base_id]["meta"]["stgb"] = stgb
            merged += 1
            continue

        records.append({
            "id": f"ex-detox-legal-{row.id.split('-', 1)[1]}",
            "kind": "example",
            "dimension": None,
            "label": None,
            "lang": "de",
            "text": row.text,
            "source": "detox-train-legal",
            "meta": {
                "gate": bool(row.gate),
                "target_groups": clean(row.target_groups),
                "hate_types": None,
                "severity": None,
                "illustrative_only": True,
                "stgb": stgb,
            },
        })

    print(f"\n  legal: {len(records)} standalone illustrations, "
          f"{merged} merged into existing balanced examples")
    return records

def gen_botox_examples(existing: list[dict]) -> list[dict]:
    """BoTox train examples for the legal dimension.

    WHY THESE ARE REAL EXAMPLES AND NOT ILLUSTRATIONS. The DeTox legal records
    are illustrative_only: their paragraph flags were too sparse to score
    (section 130 in ~16 comments), so they ground the legal motivation without
    being evidence for a label. BoTox is prosecutor-trained and dense enough
    to score - 188 / 215 / 220 across three classes - so its items are
    labelled examples like any other.

    NOTE ON meta.gate. BoTox never annotates the hate gate, so gate is None:
    "this dataset never asked", not False. Criminal relevance is NOT a
    hate-speech gate - a section 185 insult aimed at one private individual is
    criminally relevant and not hate speech, while section 130 is both.
    _render_gold prints only non-None dimensions, so these examples show a
    legal label and nothing else, which is exactly right.

    NEGATIVES ARE INCLUDED DELIBERATELY. legal=[] means "annotated, not
    criminally relevant" - class 0 - and those are the offensive-but-not-
    criminal cases the German example pool almost entirely lacks. The
    accidental German few_shot experiment showed negatives suppress false
    positives substantially, so they are sampled explicitly rather than left
    to chance.

    Leakage is already handled: the BoTox loader drops any row whose text
    appears in a held-out split, and main() asserts the same thing over the
    whole KB.
    """
    path = PROCESSED / "de_legal_train.parquet"
    if not path.exists():
        print("\n  botox: de_legal_train.parquet not found, skipping")
        return []

    df = pd.read_parquet(path)
    seen = {r["id"] for r in existing}
    records = []
    coverage = defaultdict(int)

    def add(row, key):
        rec_id = f"ex-{row.id}"
        coverage[key] += 1
        if rec_id in seen:
            return
        seen.add(rec_id)
        records.append({
            "id": rec_id,
            "kind": "example",
            "dimension": None,
            "label": None,
            "lang": "de",
            "text": row.text,
            "source": "botox-train",
            "meta": {
                "gate": None,           # never annotated by this dataset
                "target_groups": None,
                "hate_types": None,
                "severity": None,
                "legal": clean(row.legal),
                "illustrative_only": False,
            },
        })

    rows = list(df.itertuples(index=False))

    # Rarest class first: a multi-class item drawn for a rare class often also
    # carries a common one, so filling rare classes first covers the common
    # ones incidentally and keeps the total item count down.
    #
    # The coverage count is recomputed from `records` on every iteration, not
    # once per label, because add() silently skips ids already present. An
    # earlier version counted only at the top of each label's loop and the
    # commonest class ended up with 2 of 15.
    labels = sorted({l for r in rows for l in (clean(r.legal) or [])})
    by_freq = sorted(labels,
                     key=lambda l: sum(1 for r in rows
                                       if l in (clean(r.legal) or [])))

    def covered(label):
        return sum(1 for r in records if label in (r["meta"]["legal"] or []))

    for label in by_freq:
        # Multi-class items first: they carry more information per prompt slot.
        cands = [r for r in rows if label in (clean(r.legal) or [])]
        cands.sort(key=lambda r: len(clean(r.legal) or []), reverse=True)
        for row in cands:
            if covered(label) >= LEGAL_EXAMPLES_PER_LABEL:
                break
            add(row, f"legal:{label}")

    # Negatives: class 0, annotated and not criminally relevant.
    negatives = [r for r in rows if not (clean(r.legal) or [])]
    for row in negatives[:LEGAL_NEGATIVES]:
        add(row, "legal:none")

    print("\n  botox example coverage:")
    for key, n in sorted(coverage.items()):
        print(f"    {key:30} {n}")
    print(f"  botox: {len(records)} records added")
    return records


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_records += gen_definitions()
    all_records += gen_guidelines()
    # legal generator needs the balanced examples so it can merge rather than
    # duplicate a comment that is already in the KB under a different id
    examples = gen_examples()
    all_records += examples
    all_records += gen_legal_illustrations(examples)
    # BoTox last, and given everything so far, so it cannot duplicate an id
    # already present.
    all_records += gen_botox_examples(all_records)

    # DUPLICATE IDS mean a generator ran twice, or two generators produced the
    # same record. Chroma rejects them at ingest, but only after the build has
    # already spent a minute embedding; failing here is cheaper and says why.
    # This is not hypothetical: gen_legal_illustrations was accidentally called
    # twice and produced 148 records where there should have been 74.
    dupe_ids = [i for i, n in Counter(r["id"] for r in all_records).items()
                if n > 1]
    assert not dupe_ids, (f"{len(dupe_ids)} duplicate record ids: "
                          f"{dupe_ids[:5]}")

    # NO KB RECORD MAY COME FROM A HELD-OUT SPLIT.
    # Hard rule 1: KB examples come from TRAIN only. A KB example that also
    # appears in an evaluation set turns retrieval into a lookup of the answer.
    #
    # DROPPED, NOT ASSERTED. Text overlap is a property of the source data, not
    # a code bug: DeTox emits the same comment under more than one id, so one
    # copy can be in de_train and another in de_dev while make_splits' dedup
    # sees them as distinct rows. Three such records were found the first time
    # this guard ran. Every German RAG result before that date was produced
    # with them present.
    held_out = set()
    for split in ("en_dev", "en_test", "de_dev", "de_test",
                  "de_legal_dev", "de_legal_test"):
        p = PROCESSED / f"{split}.parquet"
        if p.exists():
            held_out |= {" ".join(str(t).lower().split())
                         for t in pd.read_parquet(p)["text"]}

    leaked = [r for r in all_records
              if r["kind"] == "example"
              and " ".join(str(r["text"]).lower().split()) in held_out]
    if leaked:
        print(f"\n  LEAKAGE: dropping {len(leaked)} KB examples whose text "
              f"appears in a held-out split")
        for r in leaked:
            print(f"    {r['id']}  [{r['source']}]")
        leaked_ids = {r["id"] for r in leaked}
        all_records = [r for r in all_records if r["id"] not in leaked_ids]
    print(f"  leakage check: 0 of {len(all_records)} records appear in "
          f"{len(held_out)} held-out texts")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    by_kind = Counter(r["kind"] for r in all_records)
    by_lang = Counter(r["lang"] for r in all_records)
    print(f"Wrote {len(all_records)} records to {OUT}")
    print(f"  by kind: {dict(by_kind)}")
    print(f"  by lang: {dict(by_lang)}")
    legal = sum(1 for r in all_records if r["meta"].get("illustrative_only"))
    print(f"  legal illustrations: {legal}")
    by_source = Counter(r["source"] for r in all_records)
    print(f"  by source: {dict(by_source)}")
    de_ex = [r for r in all_records
             if r["kind"] == "example" and r["lang"] == "de"]
    de_illus = sum(1 for r in de_ex if r["meta"].get("illustrative_only"))
    print(f"  German examples: {len(de_ex)} "
          f"({de_illus} illustrative, {len(de_ex) - de_illus} labelled)")
    de_guides = sum(1 for r in all_records
                    if r["kind"] == "guideline" and r["lang"] == "de")
    print(f"  German guidelines: {de_guides} (was 0 before BoTox)")


if __name__ == "__main__":
    main()
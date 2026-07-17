"""
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
PROCESSED = Path("data/processed")
OUT = Path("kb/records.jsonl")

EXAMPLES_PER_LABEL = 10  # mirror config; Phase 8 ablates this


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
    """One record per hand-authored guideline."""
    g = yaml.safe_load(GUIDELINES.read_text(encoding="utf-8"))
    records = []
    for rule in g["guidelines"]:
        records.append({
            "id": f"guide-{rule['id']}",
            "kind": "guideline",
            "dimension": rule.get("dimension"),
            "label": None,
            "lang": "en",
            "text": " ".join(rule["text"].split()),
            "source": "guidelines_v1",
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


def gen_legal_illustrations() -> list[dict]:
    """DeTox train comments flagged with an StGB paragraph -> illustrative
    examples. illustrative_only=True, paragraph in meta.stgb. These ground
    the legal motivation without being a scored dimension (Option 2)."""
    path = PROCESSED / "de_train.parquet"
    df = pd.read_parquet(path)
    df = df[df["source"] == "detox"]

    records = []
    for row in df.itertuples(index=False):
        raw = json.loads(row.raw)
        paragraphs = raw.get("legal_paragraphs", {})
        # a paragraph "fires" if > 0.5 of annotators flagged it
        flagged = [p.replace("p_", "") for p, v in paragraphs.items() if v > 0.5]
        if not flagged:
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
                "target_groups": clean(row.target_groups),   # was passed raw
                "hate_types": None,
                "severity": None,
                "illustrative_only": True,
                "stgb": ",".join(flagged),
            },
        })
    return records


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_records += gen_definitions()
    all_records += gen_guidelines()
    all_records += gen_examples()
    all_records += gen_legal_illustrations()

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


if __name__ == "__main__":
    main()
"""
scripts/make_guideline_edit.py - the guideline-edit case study (guide 8.10).
Writes a KB variant and checks the retrieval link. Zero API calls.

  python -m scripts.make_guideline_edit --variant directive
  python -m scripts.build_kb_alt kb/records_guide_directive.jsonl kb/chroma_guide_directive
  python -m scripts.make_guideline_edit --variant directive --check

WHAT THIS TESTS, AND WHY IT IS NOT THE SAME QUESTION AS SQ3. Every correction
written in SQ3 was a labelled EXAMPLE. A guideline is the other adaptability
lever, and it is the one the Phase 7 encoder baseline structurally cannot use -
an encoder trained on the KB's example records cannot consume a rule at all.

The oracle (SQ3 section 4.3) already showed the model acts on retrieved
material, so this asks the narrower question left over: is an abstract RULE
acted on as readily as a concrete labelled example?

THE TARGET. guide-profanity-without-target is decisive - it implies gate=False
when it applies - and was measured as retrieved, correct, and obeyed only about
40% of the time on exactly the cases it addresses. That was the second
in-system reproduction of Mohammadi et al. 2025, after def-target_group-other.
On en_dev_eval_main the baseline carries 102 false positives over 207 benign
items, so an obeyed guideline has a large and measurable target. The 11 false
negatives that made the balanced-set gate test underpowered are not the
quantity here.

SALIENCE ONLY, CONTENT FIXED. Both variants keep the rule, both examples and
the exception. What changes is where the decision sits and how it is phrased.
Changing what the rule SAYS would test a different rule rather than whether the
model attends to rules at all.

  salience   decision first, imperative, same vocabulary
  directive  the same, plus the output field named explicitly

ONE RECORD, NOT A REBUILD. Rebuilding from guidelines.yaml would re-run every
generator and risks a second variable. Swapping one record's text inside
records.jsonl guarantees the variant differs in exactly one field of exactly
one record, and the script asserts that.

TWO LINKS, REPORTED SEPARATELY. Rewording changes the embedding, so it changes
retrieval as well as salience. If the reworded guideline reaches a different
set of items, a prediction-level comparison is confounded. --check measures the
retrieval link first, at no cost, and the prediction comparison is then run on
the INTERSECTION - items where the guideline is retrieved under both KBs.
Phase 6 established twice that retrievability is necessary but not sufficient;
reporting one link without the other is what makes such a result unreadable.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

KB = Path("kb")
BASE_RECORDS = KB / "records.jsonl"
BASE_CHROMA = KB / "chroma"
PROCESSED = Path("data/processed")
TARGET = "guide-profanity-without-target"
SUBSET = "en_dev_eval_main"

VARIANTS = {
    # Decision first, imperative mood, same vocabulary as the original. No
    # schema field is named, so this isolates ordering and mood.
    "salience":
        'Not hate speech: profanity, insults, or aggression with no '
        'protected-characteristic target. Generic rudeness such as "you idiot" '
        'is offensive, not hateful. It becomes hate speech only if it attacks '
        'a protected group.',
    # The same, plus the output field named explicitly. Run this one first: it
    # is the stronger manipulation, so a null here means a null for salience
    # too, and stopping there costs nothing.
    "directive":
        'Set hate=false: profanity, insults, or aggression with no '
        'protected-characteristic target. Generic rudeness such as "you idiot" '
        'is offensive, not hateful. Set hate=true only if it attacks a '
        'protected group.',
}


def load(path: Path) -> list:
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_variant(variant: str) -> Path:
    records = load(BASE_RECORDS)
    hit = [r for r in records if r["id"] == TARGET]
    assert len(hit) == 1, f"{TARGET} appears {len(hit)} times in {BASE_RECORDS}"

    old = hit[0]["text"]
    new = VARIANTS[variant]
    print(f"\n{'=' * 74}\n{TARGET}, variant '{variant}'")
    print(f"\n  OLD ({len(old.split())} words)\n    {old}")
    print(f"\n  NEW ({len(new.split())} words)\n    {new}")

    out_records = []
    changed = 0
    for r in records:
        if r["id"] == TARGET:
            r = {**r, "text": new}
            changed += 1
        out_records.append(r)

    # Exactly one field of exactly one record may differ. Anything else means
    # a second variable has entered the comparison.
    diffs = [(a["id"], k) for a, b in zip(out_records, records)
             for k in set(a) | set(b) if a.get(k) != b.get(k)]
    assert diffs == [(TARGET, "text")], f"unexpected differences: {diffs}"
    assert changed == 1

    out = KB / f"records_guide_{variant}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  wrote {out} ({len(out_records)} records, 1 changed)")
    print(f"\n  next: python -m scripts.build_kb_alt {out} "
          f"kb/chroma_guide_{variant}")
    print(f"{'=' * 74}\n")
    return out


def check(variant: str, limit: int | None) -> None:
    """The retrieval link, before any API call is spent."""
    from src.hsrag.retrieve import Retriever

    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    df = pd.read_parquet(PROCESSED / f"{SUBSET}.parquet")
    if limit:
        df = df.head(limit)

    variants = {
        "base": (BASE_CHROMA, BASE_RECORDS),
        variant: (KB / f"chroma_guide_{variant}", KB / f"records_guide_{variant}.jsonl"),
    }
    got = {}
    for name, (chroma, records) in variants.items():
        assert chroma.exists(), f"{chroma} not built yet"
        # One Retriever at a time. chromadb is SQLite-backed and several
        # PersistentClients in one process can deadlock below the Python level.
        r = Retriever(chroma_path=chroma, records_path=records,
                      model_name=cfg["kb"]["embedding_model"],
                      cfg=dict(cfg["retrieval"]))
        hits = set()
        for row in df.itertuples(index=False):
            res = r.retrieve(str(row.text), str(row.lang))
            if any(h.id == TARGET for h in res.guidelines):
                hits.add(str(row.id))
        got[name] = hits
        del r

    a, b = got["base"], got[variant]
    print(f"\n{'=' * 74}\nRETRIEVAL LINK: {TARGET} on {SUBSET} "
          f"({len(df)} items)")
    print(f"  base       retrieved on {len(a):4} items")
    print(f"  {variant:10} retrieved on {len(b):4} items")
    print(f"  both       {len(a & b):4}   base only {len(a - b):4}   "
          f"{variant} only {len(b - a):4}")

    gold = {str(r.id): r for r in df.itertuples(index=False)}
    for name, ids in (("base", a), (variant, b)):
        benign = sum(1 for i in ids if gold[i].gate is False)
        print(f"  {name:10} of those, {benign} are gold not-hate "
              f"(where the guideline applies correctly)")

    print(f"\n  The prediction comparison runs on the {len(a & b)}-item "
          f"INTERSECTION,\n  so a change in retrieval cannot be mistaken for a "
          f"change in adherence.\n  Report both links; retrievability is "
          f"necessary but not sufficient.")
    out = KB / f"guide_{variant}_retrieval.json"
    out.write_text(json.dumps(
        {"target": TARGET, "subset": SUBSET, "variant": variant,
         "base_ids": sorted(a), "variant_ids": sorted(b),
         "both_ids": sorted(a & b)}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}\n{'=' * 74}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--check", action="store_true",
                    help="measure the retrieval link (needs the index built)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.check:
        check(args.variant, args.limit)
    else:
        write_variant(args.variant)


if __name__ == "__main__":
    main()
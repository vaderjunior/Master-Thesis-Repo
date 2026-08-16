"""
Validate every line of kb/records.jsonl against the KB record schema.
Run after build_kb_records (3.2). Fails loudly on any malformed record.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RECORDS = Path("kb/records.jsonl")

# "feedback" is SQ3's correction records. A separate kind rather than another
# example, so that (a) retrieval can give it its own budget instead of letting
# one correction compete with ~320 training examples for 5 slots, and (b)
# Arms._build_fewshot, which filters on kind == "example", cannot sample a
# correction into the static few_shot control.
VALID_KINDS = {"definition", "guideline", "example", "feedback"}
REQUIRED = {"id", "kind", "dimension", "label", "lang", "text", "source", "meta"}
VALID_LANGS = {"en", "de"}


def validate_line(obj: dict, n: int) -> list[str]:
    errors = []

    missing = REQUIRED - obj.keys()
    if missing:
        errors.append(f"line {n}: missing fields {missing}")
        return errors  # can't check further without the fields

    if obj["kind"] not in VALID_KINDS:
        errors.append(f"line {n} ({obj['id']}): bad kind '{obj['kind']}'")

    if obj["lang"] not in VALID_LANGS:
        errors.append(f"line {n} ({obj['id']}): bad lang '{obj['lang']}'")

    if not obj["text"] or not str(obj["text"]).strip():
        errors.append(f"line {n} ({obj['id']}): empty text")

    if not isinstance(obj["meta"], dict):
        errors.append(f"line {n} ({obj['id']}): meta is not a dict")

    # kind-specific rules
    if obj["kind"] == "definition":
        if not obj["dimension"] or not obj["label"]:
            # gate definition is the one allowed to have label=null
            if not (obj["dimension"] == "hate" and obj["label"] is None):
                errors.append(
                    f"line {n} ({obj['id']}): definition needs dimension+label"
                )

    if obj["kind"] in ("example", "feedback"):
        m = obj["meta"]
        if "gate" not in m:
            errors.append(
                f"line {n} ({obj['id']}): {obj['kind']} meta missing 'gate'")
        if "illustrative_only" not in m:
            errors.append(
                f"line {n} ({obj['id']}): {obj['kind']} meta missing "
                f"'illustrative_only'")

    # SQ3 provenance. A correction whose round and origin item are not
    # recorded cannot be traced to the run that produced it, and the
    # adaptability claim rests on every result being attributable to an exact
    # KB state. feedback_arm distinguishes a real correction from the
    # matched-size control's randomly drawn example - they are the same shape
    # of record and only this field separates them.
    if obj["kind"] == "feedback":
        for key in ("round", "origin_item_id", "criterion", "feedback_arm"):
            if key not in obj["meta"]:
                errors.append(
                    f"line {n} ({obj['id']}): feedback meta missing '{key}'")

    return errors


def main():
    # SQ3 writes per-round files (kb/records_sq3_r{n}.jsonl) so that
    # kb/records.jsonl stays the frozen baseline. --file validates one of
    # those instead of the default.
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RECORDS))
    records_path = Path(ap.parse_args().file)

    if not records_path.exists():
        print(f"{records_path} does not exist yet (build it in 3.2)")
        sys.exit(0)

    all_errors = []
    ids = set()
    by_text = defaultdict(list)   # normalised text -> [ids]
    # DERIVED FROM VALID_KINDS, not hardcoded. This was a literal dict of
    # three kinds, so adding "feedback" left 20 records uncounted and a
    # 380-record file reported "Records: 360" - which is exactly the base KB
    # size and therefore looks correct. A new kind silently missing from a
    # counter, producing a plausible number instead of a crash, is the same
    # failure that once discarded 974 predictions.
    counts = {k: 0 for k in sorted(VALID_KINDS)}
    n_parsed = 0

    with open(records_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"line {n}: bad JSON ({e})")
                continue

            all_errors.extend(validate_line(obj, n))

            if obj.get("id") in ids:
                all_errors.append(f"line {n}: duplicate id '{obj['id']}'")
            ids.add(obj.get("id"))

            if obj.get("text"):
                key = " ".join(str(obj["text"]).lower().split())
                by_text[key].append(obj.get("id"))

            n_parsed += 1
            if obj.get("kind") in counts:
                counts[obj["kind"]] += 1

    # --- KB-wide text uniqueness (Phase 4.5) ---
    # ex-detox-<id> and ex-detox-legal-<id> were the same comment under two
    # different ids, so the id check above could not see it. Retrieval then
    # spent two of five German example slots on one text.
    dupes = {t: rec_ids for t, rec_ids in by_text.items() if len(rec_ids) > 1}
    for rec_ids in list(dupes.values())[:10]:
        all_errors.append(f"duplicate text shared by ids {rec_ids}")

    print(f"Records: {n_parsed}")
    for kind, n in counts.items():
        print(f"  {kind:12} {n}")
    print(f"  distinct texts {len(by_text)}")
    # The per-kind counts must account for every parsed record. If they do
    # not, a kind exists in the file that VALID_KINDS does not know about and
    # the table above is quietly incomplete.
    if sum(counts.values()) != n_parsed:
        all_errors.append(
            f"per-kind counts sum to {sum(counts.values())} but {n_parsed} "
            f"records were parsed - an unknown kind is present")

    if all_errors:
        print(f"\n{len(all_errors)} ERRORS:")
        for e in all_errors[:30]:
            print(f"  {e}")
        sys.exit(1)

    print("\nSchema valid.")


if __name__ == "__main__":
    main()
"""
Validate every line of kb/records.jsonl against the KB record schema.
Run after build_kb_records (3.2). Fails loudly on any malformed record.
"""

import json
import sys
from pathlib import Path

RECORDS = Path("kb/records.jsonl")

VALID_KINDS = {"definition", "guideline", "example"}
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

    if obj["kind"] == "example":
        m = obj["meta"]
        if "gate" not in m:
            errors.append(f"line {n} ({obj['id']}): example meta missing 'gate'")
        if "illustrative_only" not in m:
            errors.append(
                f"line {n} ({obj['id']}): example meta missing 'illustrative_only'"
            )

    return errors


def main():
    if not RECORDS.exists():
        print(f"{RECORDS} does not exist yet (build it in 3.2)")
        sys.exit(0)

    all_errors = []
    ids = set()
    counts = {"definition": 0, "guideline": 0, "example": 0}

    with open(RECORDS, encoding="utf-8") as f:
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

            if obj.get("kind") in counts:
                counts[obj["kind"]] += 1

    print(f"Records: {sum(counts.values())}")
    for kind, n in counts.items():
        print(f"  {kind:12} {n}")

    if all_errors:
        print(f"\n{len(all_errors)} ERRORS:")
        for e in all_errors[:30]:
            print(f"  {e}")
        sys.exit(1)

    print("\nSchema valid.")


if __name__ == "__main__":
    main()
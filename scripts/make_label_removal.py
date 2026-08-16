"""
scripts/make_label_removal.py - build the removed state for the new-label
adaptability experiment. Writes files, spends no API calls.

  python -m scripts.make_label_removal --label grievance
  python -m scripts.make_label_removal --label grievance --check

WHAT THIS EXPERIMENT IS. Decision Q6 named three adaptability edit types:
corrected examples, a revised definition, and a new label added mid-run. The
first two have now been measured and both are null - SQ3's feedback rounds, and
the `other` definition rewrite. The third has never been run, and it tests a
DIFFERENT PATHWAY: the label space is read from taxonomy.yaml at render time
and interpolated into every system prompt in every arm, so it reaches the model
without any retrieval. The mechanism that produced SQ3's null - similarity
cannot preferentially select label-relevant material - therefore does not apply.

WHY IT IS NOT "ROUNDS TO RECOVERY". Restoring a taxonomy label is one edit, not
a series, so there is no learning curve to trace. What it measures is delta
macro-F1 after a knowledge-base edit with the LLM frozen, which is the thesis's
locked definition of adaptability stated as plainly as it can be: edit one YAML
file, gain a working label, no retraining.

THE RESTORED STATE ALREADY EXISTS. Round 0 gives four rag replicates on
en_dev_eval_sq3_types under the current taxonomy, and sq3_types_sq2_r1..r3 give
three zero_shot replicates. Only the removed state needs running.

WHICH ARM IS CLEAN. zero_shot receives no examples at all, so removing a label
from the taxonomy changes exactly one thing: the label space. That is the
isolated test of the label-name pathway. rag additionally loses the label's
definition and its labelled examples, so it is three simultaneous changes -
which is what adding a label to a deployed system actually looks like, and
should be reported as a system-level result rather than a pathway test.

WHY THE EXAMPLES MUST GO for the rag arm. Removing a label from the taxonomy
does not remove KB example records carrying it. _render_gold prints an
example's gold labels into the prompt, so the prompt would display
hate_type=[grievance] while grievance is absent from the label space. The
prompt would contradict itself and no result from it would be interpretable.

WHY A TEXT-LEVEL EDIT AND NOT yaml.safe_load PLUS yaml.dump. Round-tripping the
taxonomy through PyYAML reformats the entire file - block scalars collapse,
comments vanish, key order can move - which changes many things at once when
the experiment needs exactly one. The hate_type labels are each a single
flow-style line, so removing one line is a minimal edit and the script asserts
that exactly one line matched.

THE FILE SWAP IS THE RISK. prompt.render_label_space and schema.py both read
config/taxonomy.yaml from a fixed path, so running the removed state means
temporarily replacing that file. Back it up by COPY, not by git checkout: git
normalises CRLF on checkout, which changes the content hash. The
taxonomy_version stamp added on 2026-08-11 is what makes a forgotten restore
detectable - make_comparability flags more than one non-None value across runs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

CONFIG = Path("config")
KB = Path("kb")
TAXONOMY = CONFIG / "taxonomy.yaml"
BACKUP = CONFIG / "taxonomy_backup.yaml"
RECORDS = KB / "records.jsonl"
DIM = "hate_type"


def strip_label(text: str, label: str) -> str:
    """Remove one hate_type label's line from the taxonomy text.

    hate_type labels are flow-style one-liners, so the label and its definition
    live on a single line and removing it is minimal. Asserted to match exactly
    once so a near-miss cannot silently remove the wrong thing or nothing.
    """
    lines = text.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines)
            if l.strip().startswith(f"{label}:") and "definition" in l]
    assert len(hits) == 1, (
        f"expected exactly one flow-style line for '{label}', found "
        f"{len(hits)}. If this label is written as a nested block rather than "
        f"a one-liner, edit it by hand and skip this script.")
    return "".join(l for i, l in enumerate(lines) if i != hits[0])


def build(label: str) -> None:
    original = TAXONOMY.read_text(encoding="utf-8")
    tax = yaml.safe_load(original)
    labels = list(tax["dimensions"][DIM]["labels"])
    assert label in labels, f"'{label}' is not a {DIM} label: {labels}"

    edited = strip_label(original, label)
    after = yaml.safe_load(edited)
    assert list(after["dimensions"][DIM]["labels"]) == [
        l for l in labels if l != label], "label set is not as intended"
    # Every other dimension must be untouched. A YAML indent slip inside a
    # dimension block fails SILENTLY - nothing sees it and the count is just
    # short - which is the recorded taxonomy trap.
    for d in tax["dimensions"]:
        if d == DIM:
            continue
        assert after["dimensions"][d] == tax["dimensions"][d], \
            f"dimension '{d}' changed; only {DIM} may differ"

    out_tax = CONFIG / f"taxonomy_no_{label}.yaml"
    out_tax.write_text(edited, encoding="utf-8")

    records = [json.loads(l) for l in
               RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    def_id = f"def-{DIM}-{label}"
    dropped_def = [r for r in records if r["id"] == def_id]
    dropped_ex = [r for r in records
                  if r["kind"] == "example"
                  and label in (r.get("meta", {}).get(f"{DIM}s") or [])]
    keep = [r for r in records
            if r["id"] != def_id and r not in dropped_ex]

    out_rec = KB / f"records_no_{label}.jsonl"
    with open(out_rec, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    h = hashlib.sha256(edited.encode()).hexdigest()[:8]
    print(f"\n{'=' * 74}\nLABEL REMOVAL: {DIM}/{label}")
    print(f"\n  taxonomy  {len(labels)} -> {len(labels) - 1} {DIM} labels")
    print(f"            {out_tax}")
    print(f"            taxonomy_version when swapped in: taxonomy-{h}")
    print(f"\n  KB        {len(records)} -> {len(keep)} records")
    print(f"            definition dropped: {len(dropped_def)}")
    print(f"            examples dropped:   {len(dropped_ex)}")
    for r in dropped_ex[:5]:
        print(f"              {r['id']:34} {r['meta'].get(f'{DIM}s')}")
    if len(dropped_ex) > 5:
        print(f"              ... and {len(dropped_ex) - 5} more")
    print(f"            {out_rec}")
    print(f"\n  NOTE the rag arm therefore differs from the restored state in "
          f"THREE\n  ways: the label is absent from the label space, its "
          f"definition is gone,\n  and {len(dropped_ex)} examples are gone. "
          f"zero_shot differs in ONE way, since it\n  receives no examples "
          f"at all - that is the clean pathway test.")
    print(f"\n  next:")
    print(f"    python -m scripts.build_kb_alt {out_rec} "
          f"kb/chroma_no_{label}")
    print(f"    python -m scripts.make_label_removal --label {label} --check")
    print(f"{'=' * 74}\n")


def check(label: str) -> None:
    """Verify the swapped-in state before spending calls, and report what a
    removed label does to the parts of the system that read the taxonomy."""
    from src.hsrag.prompt import render_label_space, taxonomy_version

    current = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    labels = list(current["dimensions"][DIM]["labels"])
    swapped = label not in labels

    print(f"\n{'=' * 74}\nCHECK: is the removed taxonomy currently in place?")
    print(f"  config/taxonomy.yaml {DIM} labels: {labels}")
    print(f"  '{label}' present: {not swapped}")
    print(f"  taxonomy_version:   {taxonomy_version()}")
    print(f"  backup exists:      {BACKUP.exists()}  ({BACKUP})")

    if not swapped:
        print(f"\n  The ORIGINAL taxonomy is in place. To swap:")
        print(f"    Copy-Item {TAXONOMY} {BACKUP}")
        print(f"    Copy-Item {CONFIG / f'taxonomy_no_{label}.yaml'} "
              f"{TAXONOMY} -Force")
        print(f"{'=' * 74}\n")
        return

    print(f"\n  the removed taxonomy IS in place. Rendered label space:\n")
    for line in render_label_space().splitlines():
        print(f"    {line}")

    # The Pydantic enums are generated from the taxonomy at import time, so
    # during the removed state a model that emits the removed label from
    # pretraining produces a validation failure rather than a wrong label.
    # That inflates parse failures and eats repair budget. It is informative -
    # it measures whether the taxonomy constrains output at all - but it must
    # be anticipated rather than discovered mid-run.
    from src.hsrag.schema import HateType
    vals = [e.value for e in HateType]
    print(f"\n  Pydantic HateType enum: {vals}")
    assert label not in vals, "the enum still carries the removed label"
    print(f"  -> a prediction of '{label}' now fails validation. Expect parse "
          f"failures\n     to rise; count them, they are part of the result.")

    print(f"\n  RESTORE when the runs are done:")
    print(f"    Copy-Item {BACKUP} {TAXONOMY} -Force")
    print(f"    python -m scripts.make_label_removal --label {label} --check")
    print(f"  and confirm taxonomy_version returns to its pre-swap value.")
    print(f"{'=' * 74}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check(args.label) if args.check else build(args.label)


if __name__ == "__main__":
    main()
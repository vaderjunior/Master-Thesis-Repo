"""
scripts/check_definition_budget.py - where do the definition slots actually go?
Read-only, zero API calls.

  python -m scripts.check_definition_budget --primary hate_type \
      --stems sq3_round0_r1 sq3_round0_r2 sq3_round0_r3 sq3_round0_r4

WHY THIS EXISTS. Phase 3 found the BURIAL PROBLEM: naive dense retrieval over
the mixed KB buries definitions and guidelines under examples, because examples
are raw social-media text and lexically close to a raw-text query while
definitions are abstract descriptive language. The fix was structural rather
than a better ranker - one filtered query per KIND, each with its own budget.

The same problem recurs one level down, INSIDE the definitions bucket, and was
never fixed. All 21 definitions across all 5 dimensions compete in one
similarity ranking for k_definitions slots. target_group's definitions name
groups ("Black", "Muslims", "gay") whose words appear verbatim in hate speech;
hate_type's definitions name rhetorical modes ("sarcasm", "inferiority") which
do not. So target_group wins the race regardless of which dimension is being
measured.

PHASE8_SIGNOFF section 7.11 measured it once: on en_dev_eval_sq3_types, where
every item is hate_type-scorable and NO other dimension is annotated, 64% of
definition slots went to other dimensions (target_group 532, hate_type 336 of
934). Config was frozen across all Phase 8 rounds so the deltas hold, but
absolute levels are depressed everywhere.

This script makes that measurement repeatable, per run and per subset, so the
same instrument reports the before and the after. It reads
ItemResult.retrieved["definitions"], which every run has stamped since Phase 5,
so the whole baseline costs nothing.

ALSO REPORTS DEAD RECORDS. A definition never retrieved on any item is a record
that has no effect on any number in the project. The SQ3 feedback analysis
found 14 of 60 corrections in that state; the same question has never been
asked of the definitions.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path("experiments/results")
KB_RECORDS = Path("kb/records.jsonl")


def definition_index() -> tuple[dict, dict]:
    """id -> dimension, and dimension -> [ids]. From the source of truth."""
    by_id, by_dim = {}, defaultdict(list)
    for line in KB_RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "definition":
            continue
        dim = r.get("dimension") or "(none)"
        by_id[r["id"]] = dim
        by_dim[dim].append(r["id"])
    return by_id, dict(by_dim)


def read(stem: str) -> list:
    p = RESULTS / f"{stem}_live.jsonl"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r and r.get("arm") == "rag":
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--primary", default=None,
                    help="the dimension this subset actually scores. The "
                         "share of slots reaching it is the number that "
                         "matters; without it only the raw distribution is "
                         "printed.")
    args = ap.parse_args()

    by_id, by_dim = definition_index()
    print(f"\n{'=' * 78}\nDEFINITION SLOT ALLOCATION\n{'=' * 78}")
    print(f"\nKB holds {len(by_id)} definitions across {len(by_dim)} "
          f"dimensions:")
    for dim, ids in by_dim.items():
        print(f"  {dim:16} {len(ids):>3} definition(s)")

    seen_ids = Counter()
    grand = Counter()
    grand_slots = 0

    for stem in args.stems:
        rows = read(stem)
        if not rows:
            print(f"\n{stem}: no rag rows")
            continue
        counts, slots, unknown = Counter(), 0, 0
        per_item = Counter()
        for r in rows:
            ids = (r.get("retrieved") or {}).get("definitions") or []
            per_item[len(ids)] += 1
            for i in ids:
                slots += 1
                seen_ids[i] += 1
                dim = by_id.get(i)
                if dim is None:
                    unknown += 1
                else:
                    counts[dim] += 1
        grand += counts
        grand_slots += slots

        print(f"\n{'-' * 78}\n{stem}   {len(rows)} rag items, "
              f"{slots} definition slots")
        if len(per_item) > 1:
            # A varying slot count per item is not a bug - a bucket returns
            # fewer than its budget when the candidate pool is thin, and
            # _bm25_kind refuses to pad on zero lexical overlap. Worth seeing,
            # because it changes the denominator of every share below.
            print(f"  slots per item: {dict(sorted(per_item.items()))}")
        for dim, n in counts.most_common():
            bar = "#" * int(40 * n / max(slots, 1))
            star = "  <-- PRIMARY" if dim == args.primary else ""
            print(f"  {dim:16} {n:>6}  {n / slots:6.1%}  {bar}{star}")
        if unknown:
            print(f"  {'(id not in KB)':16} {unknown:>6} - retrieved ids that "
                  f"are not definitions in the CURRENT records.jsonl, i.e. "
                  f"this run used a different KB")
        if args.primary:
            got = counts.get(args.primary, 0)
            n_items = len(rows)
            print(f"\n  {args.primary}: {got} slots over {n_items} items = "
                  f"{got / max(n_items, 1):.2f} definitions per item.")
            print(f"  share of the budget: {got / max(slots, 1):.1%} "
                  f"({1 - got / max(slots, 1):.1%} to dimensions this subset "
                  f"does not score).")
            # SHARE IS THE MISLEADING ONE AND WAS REPORTED ALONE UNTIL
            # 2026-08-16. PHASE8_SIGNOFF's headline "64% of definition slots
            # went to other dimensions" is a share, and it motivated the whole
            # 7.3 experiment. Under proportional budgets hate_type's share
            # FELL to 28.6% while its absolute coverage ROSE from 0.72 to 2.00
            # per item, because the denominator went from 2 slots to 7. The
            # per-item figure is what the model actually sees; the share is an
            # artefact of the total budget. Both are printed now, per-item
            # first.

    if len(args.stems) > 1 and grand_slots:
        print(f"\n{'-' * 78}\nPOOLED over {len(args.stems)} run(s), "
              f"{grand_slots} slots")
        for dim, n in grand.most_common():
            star = "  <-- PRIMARY" if dim == args.primary else ""
            print(f"  {dim:16} {n:>6}  {n / grand_slots:6.1%}{star}")

    # -------------------------------------------------- dead definitions
    print(f"\n{'-' * 78}\nDEAD DEFINITIONS - never retrieved on any item")
    print("  A definition that is never retrieved has no effect on any number")
    print("  in the project. It is authored, versioned, hashed into")
    print("  kb_version, and inert.\n")
    dead = [(dim, i) for i, dim in by_id.items() if i not in seen_ids]
    if not dead:
        print("  None. Every definition reached at least one prompt.")
    else:
        for dim in by_dim:
            ids = [i for d, i in dead if d == dim]
            if ids:
                print(f"  {dim:16} {len(ids)}/{len(by_dim[dim])} never "
                      f"retrieved: {', '.join(sorted(ids))}")

    print(f"\n{'-' * 78}\nMOST AND LEAST RETRIEVED")
    ranked = seen_ids.most_common()
    for i, n in ranked[:5]:
        print(f"  {n:>6}  {by_id.get(i, '?'):16} {i}")
    if len(ranked) > 10:
        print("   ...")
    for i, n in ranked[-5:]:
        print(f"  {n:>6}  {by_id.get(i, '?'):16} {i}")

    print(f"\n{'=' * 78}")
    print("HOW TO READ THIS. On a subset that scores ONE dimension, every slot")
    print("given to another dimension is a slot the measured dimension did not")
    print("get. That does not make the run wrong - the model must answer all")
    print("five dimensions for every item, so the other definitions are not")
    print("useless - but it does mean the measured dimension is being scored")
    print("under a definition budget far smaller than k_definitions suggests.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
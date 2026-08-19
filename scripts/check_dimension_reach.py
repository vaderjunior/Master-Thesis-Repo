"""
scripts/check_dimension_reach.py - which dimensions actually reached the vote,
in every stored run. Read-only, zero API calls.
Run: python -m scripts.check_dimension_reach

WHY THIS EXISTS. The project's signature failure is a dimension that is
silently absent from one code path and produces a plausible number instead of
a crash. Two instances are now on record:

  1. Adding `legal` broke four functions, none of which crashed, and 974 of
     1,350 predictions were discarded while the report looked exactly like a
     model that had never learned the task (PHASE8_SIGNOFF section 5).
  2. legal_dev_peasec (2026-08-02) scored macro 0.203 against 0.576 and 0.587
     for its two supposed replicates at the same model, prompt, KB and
     temperature. The raw per-run predictions are near-identical across all
     three (insult_defamation 286/292/285, propaganda 169/161/159,
     incitement_public_peace 173/179/174) and the gate distribution is
     unchanged, so the model behaved the same. The final voted result carried
     legal on 22 of 175 items against 144 of 175. `legal` was never aggregated.

Instance 2 was found by accident, six weeks late, while investigating
something else. This check finds it in one second.

THE INDICATOR IS `agreement`, NOT THE SCORES. aggregate() writes one entry per
dimension it voted on. A dimension missing from that dict was not voted, which
is a fact about the code path rather than about the data or the model. It
cannot be confused with a subset where the dimension simply has no gold: the
prediction exists either way, because the model is asked for every dimension
in the label space regardless of what the gold happens to annotate.

WHAT A LEGITIMATE ABSENCE LOOKS LIKE. Runs predating a dimension's
introduction cannot carry it and are not faults - they are dated, and the
report groups by date so the boundary is visible rather than alarming. What is
NOT legitimate is a run after that boundary missing a dimension its
contemporaries carry.

THIS IS ALSO THE SHAPE OF THE 7.4 CHECK. Five encoder heads is the same
problem: training data present -> checkpoint saved -> prediction emitted ->
scored -> in the report table. The per-hop assertion there should be modelled
on this, and watched to fail before it is trusted.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

RESULTS = Path("experiments/results")
TAXONOMY = Path("config/taxonomy.yaml")


def expected_dims() -> list:
    """Dimension names as the vote writes them, read from the taxonomy.

    Read rather than hardcoded: a hardcoded list is exactly what fails to
    mention a newly added dimension, which is the bug this script exists to
    catch. `hate` is the binary gate and is written under its own name.
    """
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    return list(tax["dimensions"])


def read(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "_manifest" not in r:
            out.append(r)
    return out


def main():
    dims = expected_dims()
    print(f"\n{'=' * 96}\nDIMENSION REACH AUDIT")
    print(f"expected from taxonomy.yaml: {dims}")
    print(f"{'=' * 96}")
    print(f"\n{'run':30} {'date':11} {'n':>5}  "
          f"{'voted dimensions':38} {'_dropped':>9}  missing")
    print("-" * 96)

    rows, faults, undated = [], [], []
    for path in sorted(RESULTS.glob("*_live.jsonl")):
        stem = path.stem.replace("_live", "")
        items = read(path)
        if not items:
            continue

        # PER ITEM, not the union. The union says whether a dimension was ever
        # voted in this file. It cannot say whether it was voted in EVERY item,
        # and that is the distinction that matters. run_slice1 is resumable, so
        # a run interrupted, patched and resumed writes items aggregated by two
        # different versions of vote.py into one file. legal_dev_peasec is
        # exactly that: ~15% of its items carry `legal` in agreement and ~85%
        # do not, and the union alone reports it as clean. Nothing else in the
        # project can see this - model, prompt, kb, taxonomy, temperature and
        # workers were identical throughout, because what changed was the
        # library code and nothing stamps that.
        # DENOMINATOR IS SCORED ITEMS, NOT ALL ITEMS. aggregate() returns
        # early with an empty agreement dict when every run failed to parse:
        #
        #     valid = [r.result for r in runs if getattr(r, "ok", False)]
        #     if not valid: return out          # out.agreement == {}
        #
        # That is a legitimate no-prediction, not a missing code path. Counting
        # it flagged six runs as mixed on 2026-08-14 - de_legal_kbv3_r1/r2,
        # german_dev_kbv1_newprompt, main_base_r0, main_base_r0_rep2,
        # main_sq3_ctl_r4 - every one of them short by exactly the 1 or 2
        # items its own honesty table already reports as unscored, and every
        # one short on ALL FIVE dimensions at once, which no code-version mix
        # produces. A check that cries wolf gets ignored, and this one was
        # drowning its single real finding in six false ones.
        scored = [r for r in items if r.get("result") is not None]
        n_ref = len(scored)
        per_dim = {d: sum(1 for r in scored
                          if d in (r.get("agreement") or {})) for d in dims}
        n_drop = sum(1 for r in scored
                     if "_dropped" in (r.get("agreement") or {}))
        # WHICH DIMENSIONS IS THIS RUN EXPECTED TO CARRY? An LLM run answers
        # all five in one pass, so all five are expected. An encoder arm has
        # only the heads it was trained with: the German arm has hate and
        # legal because BoTox never annotates the gate and only Implicit Hate
        # carries hate_type gold; the English arms have no legal head because
        # KB records store legal provenance as meta.stgb, a paragraph number,
        # not a taxonomy label.
        #
        # Without this, six encoder runs generate fifteen faults that are all
        # by design, and the one real fault - legal_dev_peasec's mixed code
        # versions - is buried in them. That is the third time this pattern
        # has appeared: make_comparability produced 30 kb_version false alarms
        # by reading a field two arms deliberately leave None, and this script
        # produced 6 by counting parse failures. A check that cries wolf gets
        # ignored, and an ignored check is worse than none.
        enc = next((r["encoder_meta"] for r in items if r.get("encoder_meta")),
                   None)
        by_design = []
        expected = list(dims)
        if enc:
            have = set(enc.get("heads") or {})
            by_design = [d for d in dims if d not in have]
            expected = [d for d in dims if d in have]

        voted = [d for d in expected if n_ref and per_dim[d] == n_ref]
        missing = [d for d in expected if per_dim[d] == 0]
        mixed = [d for d in expected if 0 < per_dim[d] < n_ref]
        has_dropped = bool(n_ref) and n_drop == n_ref

        ts = [r.get("timestamp") or 0 for r in items]
        when = (datetime.fromtimestamp(max(ts), timezone.utc).strftime("%Y-%m-%d")
                if max(ts) else "?")

        # Share of items carrying a non-empty prediction per dimension. A
        # dimension that is voted but empty everywhere is the same symptom
        # with a different cause, so it is worth seeing beside the key list.
        fill = {}
        for d in dims:
            n = 0
            for r in items:
                res = r.get("result")
                if not res:
                    continue
                v = res.get(d)
                if v is True or (isinstance(v, list) and v) or \
                        (isinstance(v, str) and v):
                    n += 1
            fill[d] = n / len(items)

        rows.append({"run": stem, "date": when, "n": n_ref,
                     "voted": voted, "missing": missing, "mixed": mixed,
                     "by_design": by_design,
                     "per_dim": per_dim, "dropped": has_dropped, "fill": fill})
        note = ",".join(missing) if missing else "-"
        if by_design:
            note += f"   [no head: {','.join(by_design)}]"
        if mixed:
            note += "   MIXED: " + ", ".join(
                f"{d} on {per_dim[d]}/{n_ref}" for d in mixed)
        if n_ref < len(items):
            note += f"   ({len(items) - n_ref} unscored, excluded)"
        print(f"{stem:30} {when:11} {len(items):>5}  "
              f"{','.join(voted):38} {str(has_dropped):>9}  {note}")

    # ------------------------------------------------------- the verdict
    print(f"\n{'-' * 96}\nWHEN DID EACH DIMENSION START BEING VOTED?")
    print("  A run missing a dimension BEFORE its first appearance is not a")
    print("  fault. A run missing one AFTER is.\n")
    nd = sum(1 for r in rows if r.get("by_design"))
    if nd:
        print(f"\n  {nd} run(s) declare heads they do not have; those "
              f"dimensions are excluded from the fault check below.")
    first = {}
    for d in dims:
        dated = sorted(r["date"] for r in rows if d in r["voted"])
        first[d] = dated[0] if dated else None
        n = sum(1 for r in rows if d in r["voted"])
        print(f"  {d:16} first voted {str(first[d]):12} "
              f"present in {n}/{len(rows)} runs")

    for r in rows:
        # An undated run cannot be placed relative to a dimension's
        # introduction, so it is listed separately rather than flagged. The
        # first Slice 1 run predates timestamps entirely and was reported as a
        # fault for lacking a dimension that did not exist when it ran.
        if r["date"] == "?":
            undated.append(r["run"])
            continue
        for d in r["mixed"]:
            faults.append(
                f"{r['run']} ({r['date']}) votes `{d}` on only "
                f"{r['per_dim'][d]}/{r['n']} items - the file mixes two code "
                f"versions and is attributable to no single system state")
        for d in r["missing"]:
            if first[d] and r["date"] > first[d]:
                faults.append(
                    f"{r['run']} ({r['date']}) does not vote `{d}`, which its "
                    f"contemporaries have carried since {first[d]}")

    print(f"\n{'-' * 96}\nFAULTS")
    if faults:
        print("  A dimension absent from the vote is absent from every number")
        print("  derived from that run, silently.\n")
        for f in faults:
            print(f"  - {f}")
    else:
        print("  None. Every run votes every dimension that existed when it")
        print("  ran. Note this is a statement about the CODE PATH, not about")
        print("  whether the answers were any good.")
    if undated:
        print(f"\n  UNDATED, not assessed ({len(undated)}): "
              f"{', '.join(undated)}")
        print("  These carry no timestamps and predate the stamp, so they")
        print("  cannot be placed relative to any dimension's introduction.")

    print(f"\n{'-' * 96}\nNON-EMPTY PREDICTION RATE, per dimension")
    print("  An outlier here is the second symptom: voted, but empty. Read it")
    print("  against runs on the SAME subset - a German gate subset has no")
    print("  hate_type gold and a low rate there means nothing.\n")
    print(f"  {'run':30} " + " ".join(f"{d[:9]:>10}" for d in dims))
    for r in sorted(rows, key=lambda r: r["date"]):
        print(f"  {r['run']:30} "
              + " ".join(f"{r['fill'][d]:>10.2f}" for d in dims))

    print(f"\n{'=' * 96}\n")


if __name__ == "__main__":
    main()
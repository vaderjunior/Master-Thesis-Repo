"""
scripts/make_comparability.py - generate experiments/COMPARABILITY.md from the
results files. Read-only on results, zero API calls.
Run: python -m scripts.make_comparability

WHY GENERATED AND NOT WRITTEN BY HAND. The project now has around ten KB
versions in play across SQ3's two arms and four rounds, plus the base and the
recovered kbv1. A hand-maintained table would be wrong within a week, and a
comparability table that is wrong is worse than none: it invites exactly the
mismatched comparison it exists to prevent.

WHY IT IS AN AUDIT AND NOT JUST DOCUMENTATION. The manifest header records
what was REQUESTED; every ItemResult records what actually HAPPENED. Comparing
them catches a silently ignored override, which has cost this project a full
run before - run_slice1 built its Retriever from config.yaml while the
manifest named a different chroma_path, and 27 minutes of calls produced a
duplicate of an earlier run instead of the intended KB comparison.

WHAT MAKES TWO NUMBERS COMPARABLE. Same model, same prompt_version, same
temperature. kb_version is the variable under test in the adaptability
experiments, so it is reported but not required to match; anything else
differing means the two numbers answer different questions.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("experiments/results")
OUT = Path("experiments/COMPARABILITY.md")

# Stamps that must be identical for two numbers to be compared directly.
# kb_version is deliberately absent: it is the independent variable.
KEY_STAMPS = ("active_model", "prompt_version", "temperature")
# code_version is in ALL_STAMPS and deliberately NOT in KEY_STAMPS. As an
# ALL_STAMP it gets the within-run consistency check, which is the one that
# matters: "code_version varies within the run" would have caught
# legal_dev_peasec's resume-across-a-patch on the day it happened. As a KEY
# stamp it would put every run in its own comparability group, since the
# library changes constantly and most changes are irrelevant to any given
# number.
ALL_STAMPS = KEY_STAMPS + ("kb_version", "taxonomy_version", "code_version",
                           "workers")

# Stamps that are legitimately absent on some arms and must be read only from
# the arm that carries them. classify() writes kb_version=None for zero_shot
# and few_shot deliberately: they consult no knowledge base, and stamping one
# would imply a dependency that does not exist. Reading it across all arms
# reported 30 false alarms on this script's first run and put a wrong kb value
# in the table for every multi-arm run.
ARM_SCOPED = ("kb_version",)


def read(path: Path) -> tuple[dict | None, list]:
    header, items = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "_manifest" in rec:
            m = rec["_manifest"]
            header = m if isinstance(m, dict) else rec
            continue
        items.append(rec)
    return header, items


def observed(items: list, field: str) -> list:
    """Distinct values actually stamped on the items, sorted for stability."""
    return sorted({str(i.get(field)) for i in items})


def main():
    rows, issues = [], []
    for path in sorted(RESULTS.glob("*_live.jsonl")):
        stem = path.stem.replace("_live", "")
        header, items = read(path)
        if not items:
            issues.append(f"{stem}: no results (header only)")
            continue

        rag = [i for i in items if i.get("arm") == "rag"] or items
        obs = {f: observed(rag if f in ARM_SCOPED else items, f)
               for f in ALL_STAMPS}

        # A run whose items disagree among themselves is not attributable to
        # any single system state, so it cannot be compared with anything.
        for f in ALL_STAMPS:
            if len(obs[f]) > 1:
                issues.append(f"{stem}: {f} varies within the run -> "
                              f"{obs[f]}")

        # Requested vs actual. A mismatch means an override was ignored.
        if header:
            for f in ("kb_version", "prompt_version"):
                want = str(header.get(f))
                if want != "None" and obs[f] and want != obs[f][0]:
                    issues.append(
                        f"{stem}: manifest asked for {f}={want} but items "
                        f"carry {obs[f][0]} - an override was ignored")

        arms = Counter(i["arm"] for i in items)
        ts = [i.get("timestamp") or 0 for i in items]
        when = (datetime.fromtimestamp(max(ts), timezone.utc).strftime("%Y-%m-%d")
                if max(ts) else "?")
        rows.append({
            "run": stem,
            "date": when,
            "subset": (header or {}).get("subset", "?"),
            "n": len(items),
            "arms": "+".join(sorted(arms)),
            "model": obs["active_model"][0].split("/")[-1],
            "prompt": obs["prompt_version"][0].replace("classify_v1-", ""),
            "kb": obs["kb_version"][0][:8],
            "taxonomy": (obs["taxonomy_version"][0]
                         if obs["taxonomy_version"][0] != "None" else ""),
            "T": obs["temperature"][0],
            "votes": (header or {}).get("n_votes", "?"),
        })

    # ------------------------------------------------------------- write
    lines = [
        "# Comparability ledger",
        "",
        "GENERATED by `python -m scripts.make_comparability`. Do not edit by",
        "hand: every value below is read from the stamps on the actual",
        "`ItemResult` records, not from the manifests, so it says what",
        "happened rather than what was requested.",
        "",
        "**Two numbers may be compared directly only if they share model,",
        "prompt and T.** `kb` is the independent variable in the adaptability",
        "experiments and is expected to differ. Any other mismatch means the",
        "two numbers answer different questions and needs an explicit caveat",
        "wherever they appear together.",
        "",
        f"_{len(rows)} runs, "
        f"{sum(r['n'] for r in rows)} item-arm records, generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}._",
        "",
        "| run | date | subset | n | arms | model | prompt | kb | taxonomy | "
        "T | votes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["date"], r["run"])):
        lines.append(
            f"| `{r['run']}` | {r['date']} | {r['subset']} | {r['n']} | "
            f"{r['arms']} | {r['model']} | `{r['prompt']}` | `{r['kb']}` | "
            f"{r['taxonomy'] or 'unstamped'} | {r['T']} | {r['votes']} |")

    # Comparability groups: the actual answer to "can I put these in one table"
    # taxonomy_version joins the GROUP KEY rather than the global audit. As an
    # audit it fired on every regeneration from 2026-08-14 onward - correctly,
    # since the four nolabel_* runs deliberately used the six-label taxonomy -
    # and would have gone on firing forever on a fact that is by design. A
    # check that cries wolf gets ignored, which is how the kb_version false
    # alarms got 30 runs mislabelled.
    #
    # Grouping is the right home: two runs under different label spaces are
    # genuinely not comparable, which is exactly what a group boundary means.
    # 'unstamped' (pre-2026-08-11) is its own group value rather than a fault:
    # the taxonomy did not change then, only the stamp did not exist.
    # UNSTAMPED IS TREATED AS THE CURRENT TAXONOMY, and the group label says so.
    #
    # The first version of this change made "unstamped" its own group value,
    # which split types_kbv3_r1/r2 (2026-08-11, unstamped) from r3/r4
    # (2026-08-14, stamped) - the four replicates that produced retraction 15
    # and the corrected 0.070 floor, and which are comparable by KB, prompt,
    # model and T. This file's own earlier comment predicted exactly that:
    # making the stamp a key "would split the ledger into 'before the stamp
    # existed' and 'after' even though the taxonomy did not change".
    #
    # The assumption being made, so it is visible rather than buried: every
    # unstamped run is assumed to have used the taxonomy in force at the time.
    # It is NOT universally safe - the taxonomy gained `legal` during
    # 2026-08-02, so runs before that used a four-dimension label space.
    # `check_dimension_reach` identifies those exactly (they are the runs with
    # `legal` missing from the vote), which is why the assumption is tolerable
    # here and the cross-check is named.
    CURRENT_TAX = None
    try:
        from src.hsrag.prompt import taxonomy_version
        CURRENT_TAX = taxonomy_version()
    except Exception:
        pass
    groups = defaultdict(list)
    for r in rows:
        tax = r["taxonomy"] or (CURRENT_TAX or "unstamped")
        groups[(r["model"], r["prompt"], r["T"], tax)].append(r["run"])
    lines += ["", "## Directly comparable groups", "",
              "Runs sharing model, prompt, T and taxonomy. Numbers from",
              "different groups need an explicit caveat.", "",
              "Runs predating the `taxonomy_version` stamp (before",
              "2026-08-12) carry no taxonomy value and are placed in the",
              "CURRENT taxonomy's group on the assumption that they used the",
              "taxonomy in force at the time. That assumption fails for runs",
              "predating 2026-08-02, when `legal` was added: run",
              "`check_dimension_reach` to identify them, they are the runs",
              "with `legal` absent from the vote.", ""]
    for (model, prompt, temp, tax), runs in sorted(
            groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- **{model} / `{prompt}` / T={temp} / {tax}** "
                     f"({len(runs)} runs): {', '.join(sorted(runs))}")

    # A taxonomy edit changes the label space in every prompt in every arm.
    # It is NOT added to KEY_STAMPS: every run predating 2026-08-11 has
    # taxonomy_version=None, and making it a key stamp would split the ledger
    # into "before the stamp existed" and "after" even though the taxonomy did
    # not change. Instead, more than one non-None value is flagged loudly.
    # Deliberately no global taxonomy issue any more - see the grouping
    # comment above. The nolabel_* runs are supposed to carry a different
    # taxonomy; that is the new-label experiment. They now land in their own
    # comparability group, which says the same thing without raising an alarm
    # on every future regeneration.

    lines += ["", "## Consistency audit", ""]
    if issues:
        lines.append("Each line below is a run whose stamps disagree with "
                     "themselves or with its manifest. **Do not report a "
                     "number from a flagged run until the cause is known.**")
        lines.append("")
        lines += [f"- {i}" for i in issues]
    else:
        lines.append("No run has stamps that disagree with themselves or with "
                     "its manifest.")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{len(rows)} runs, {len(groups)} comparability group(s)")
    for (model, prompt, temp, tax), runs in sorted(
            groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {model} / {prompt} / T={temp} / {tax}: {len(runs)} runs")
    if issues:
        print(f"\n{len(issues)} CONSISTENCY ISSUE(S):")
        for i in issues:
            print(f"  {i}")
    else:
        print("\nno consistency issues")
    print(f"\nwrote {OUT}\n")


if __name__ == "__main__":
    main()
"""
scripts/analyse_run.py - consistency, significance, guideline effect.
Runs on stored results. NO API CALLS.

  python -m scripts.analyse_run --manifest slice1_main
  python -m scripts.analyse_run --manifest targets_dev --bootstrap 2000
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.hsrag.analysis import (consistency, guideline_effect, mcnemar,
                                paired_bootstrap, single_run_view)
from src.hsrag.manifest import load as load_manifest
from src.hsrag.metrics import (ARMS, effective_gate, score_gate,
                               score_multilabel)

PROCESSED = Path("data/processed")
RESULTS = Path("experiments/results")


def fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="slice1_main")
    ap.add_argument("--file")
    ap.add_argument("--mapping", default="strict")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    m = load_manifest(args.manifest)
    path = Path(args.file) if args.file else \
        RESULTS / f"slice1_{m.subset}_live.jsonl"

    df = pd.read_parquet(PROCESSED / f"{m.subset}.parquet")
    if m.limit:
        df = df.head(m.limit)
    gold = {str(r.id): r for r in df.itertuples(index=False)}

    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_arm = {}
    for r in rows:
        if "_manifest" in r:
            continue
        by_arm.setdefault(r["arm"], []).append(r)
    arms = [a for a in ARMS if a in by_arm]
    assert arms, f"no recognised arms in {path}"

    tax = yaml.safe_load(Path("config/taxonomy.yaml").read_text(encoding="utf-8"))
    labels = {d: (list(tax["dimensions"][d]["labels"])
                  if isinstance(tax["dimensions"][d]["labels"], dict)
                  else list(tax["dimensions"][d]["labels"]))
              for d in ("target_group", "hate_type")}

    # ---------------------------------------------------------- consistency
    print(f"\n{'=' * 78}\nCONSISTENCY  (Krippendorff alpha across the "
          f"{m.n_votes} raw runs)\n{'=' * 78}")
    print(f"{'arm':12} {'gate':>10} {'target_group':>14} {'hate_type':>12} "
          f"{'severity':>10}")
    cons = {a: consistency(by_arm[a], labels) for a in arms}
    for a in arms:
        c = cons[a]
        print(f"{a:12} {fmt(c['gate']['alpha']):>10} "
              f"{fmt(c['target_group']['alpha']):>14} "
              f"{fmt(c['hate_type']['alpha']):>12} "
              f"{fmt(c['severity']['alpha']):>10}")
    print("\n  units: gate/severity per item; multilabel per (item, label) "
          "binary decision")
    for a in arms:
        for dim, v in cons[a].items():
            if v.get("note"):
                print(f"  {a} / {dim}: {v['note']}")

    # ---------------------------------------------------- the free n=1 arm
    print(f"\n{'=' * 78}\nCOST vs STABILITY: n=1 (first run) vs n={m.n_votes} "
          f"vote\n{'=' * 78}")
    print(f"{'arm':12} {'gate n=1':>10} {'gate n=3':>10} {'delta':>8}   "
          f"{'tg n=1':>8} {'tg n=3':>8} {'delta':>8}")
    for a in arms:
        one = single_run_view(by_arm[a])
        g1 = score_gate(one, gold, args.mapping).macro_f1
        g3 = score_gate(by_arm[a], gold, args.mapping).macro_f1
        t1 = score_multilabel(one, gold, "target_groups",
                              "target_group").macro_f1
        t3 = score_multilabel(by_arm[a], gold, "target_groups",
                              "target_group").macro_f1
        # Gate macro-F1 is None on a single-class subset (every gold item is
        # hateful), so the n=1 vs n=3 delta is undefined there.
        d1 = (g3 - g1) if (g1 is not None and g3 is not None) else None
        d2 = (t3 - t1) if (t1 is not None and t3 is not None) else None
        print(f"{a:12} {fmt(g1):>10} {fmt(g3):>10} {fmt(d1, '+.3f'):>8}   "
              f"{fmt(t1):>8} {fmt(t3):>8} "
              f"{fmt(d2, '+.3f') if d2 is not None else 'n/a':>8}")

    # --------------------------------------------------------- significance
    print(f"\n{'=' * 78}\nSIGNIFICANCE\n{'=' * 78}")

    if len(arms) < 2:
        print("  only one arm present; nothing to compare")
    else:
        print(f"\n  McNemar on the gate (mapping={args.mapping}), exact "
              f"binomial on discordant pairs")
        print(f"  {'comparison':26} {'both ok':>8} {'only A':>7} {'only B':>7} "
              f"{'both bad':>9} {'p':>8}")
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                r = mcnemar(by_arm[a], by_arm[b], gold, effective_gate,
                            args.mapping)
                print(f"  {a + ' vs ' + b:26} {r['both_correct']:8} "
                      f"{r['only_a_correct']:7} {r['only_b_correct']:7} "
                      f"{r['both_wrong']:9} {r['p_value']:8.3f}")

        print(f"\n  paired bootstrap, B={args.bootstrap}, target_group "
              f"macro-F1")
        # The label set is fixed ONCE from the full sample so every resample
        # scores the same quantity. Recomputing the MIN_SUPPORT filter per
        # draw would admit a different label set each time and the CI would
        # describe a moving target. Any arm yields the same set: support
        # counts gold positives only, so it is a property of the gold data.
        fixed = score_multilabel(by_arm[arms[0]], gold, "target_groups",
                                 "target_group").labels_averaged
        print(f"  labels held fixed across resamples: {fixed}")

        if not fixed:
            print("  no label clears MIN_SUPPORT; macro-F1 is undefined and "
                  "the bootstrap is skipped")
        else:
            def tg(rows_, gold_):
                return score_multilabel(rows_, gold_, "target_groups",
                                        "target_group",
                                        fixed_labels=fixed).macro_f1

            for i, a in enumerate(arms):
                for b in arms[i + 1:]:
                    r = paired_bootstrap(by_arm[a], by_arm[b], gold, tg,
                                         b=args.bootstrap)
                    if r.get("delta") is None:
                        print(f"  {a} vs {b}: {r.get('note')}")
                        continue
                    print(f"  {a + ' vs ' + b:26} delta {r['delta']:+.3f}  "
                          f"95% CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
                          f"p={r['p_value']:.3f}   "
                          f"favours {a}/{b}/tie = "
                          f"{r['favour_a']}/{r['favour_b']}/{r['ties']}")

    # ---------------------------------------------------- guideline effect
    ge = {}
    if "rag" in by_arm and "zero_shot" in by_arm:
        print(f"\n{'=' * 78}\nGUIDELINE EFFECT  (rag vs zero_shot on the SAME "
              f"items)\n{'=' * 78}")
        ge = guideline_effect(by_arm["rag"], by_arm["zero_shot"], gold,
                              effective_gate, args.mapping)
        print(f"  {'guideline':32} {'implies':>9} {'retr':>5} {'contra':>7}   "
              f"{'gold=hate rag/zs':>18}   {'gold=not-hate rag/zs':>22}")
        for guide, d in ge.items():
            def cell(side):
                c = d[side]
                if not c["n"]:
                    return "-"
                zs = (f"{c['zero_shot_accuracy']:.0%}"
                      if c["zero_shot_accuracy"] is not None else "-")
                return (f"{c['rag_accuracy']:.0%}/{zs} "
                        f"(n={c['n']}"
                        + (f", {c['delta']:+.0%}" if c["delta"] is not None
                           else "")
                        + ")")
            print(f"  {guide:32} "
                  f"{'hate' if d['implies_hate'] else 'not hate':>9} "
                  f"{d['retrieved_on']:5} "
                  f"{fmt(d['contradiction_rate'], '.0%'):>7}   "
                  f"{cell('gold_hate'):>18}   {cell('gold_not_hate'):>22}")
        print("\n  'contra' = share of retrievals where the gold label opposes")
        print("  what the guideline implies, i.e. it was retrieved but does")
        print("  not apply. delta = rag accuracy minus zero_shot accuracy on")
        print("  the same items; zero_shot never saw the guideline.")
    else:
        print(f"\n  guideline effect needs both a rag and a zero_shot arm; "
              f"present: {arms}")

    # Written unconditionally: consistency is computed for every run, and
    # nesting this inside the guideline block silently discarded it whenever
    # a manifest had no rag arm.
    out = path.with_suffix(".analysis.json")
    out.write_text(json.dumps(
        {"consistency": cons, "guideline_effect": ge},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
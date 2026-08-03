"""
scripts/score_run.py - score a results file. No API calls.

  python -m scripts.score_run --manifest slice1_main
  python -m scripts.score_run --file experiments/results/foo.jsonl \
                              --subset en_dev_eval_main --limit 150
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from src.hsrag.metrics import ARMS, MIN_SUPPORT, score_all
import pandas as pd

from src.hsrag.metrics import ARMS, score_all
from src.hsrag.manifest import load as load_manifest

PROCESSED = Path("data/processed")
RESULTS = Path("experiments/results")


def fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def report(scores: dict) -> None:
    arms = [a for a in ARMS if a in scores]

    for mapping in ("strict", "lenient"):
        if mapping not in scores[arms[0]]["gate"]:
            continue
        print(f"\n{'=' * 78}\nGATE  (mapping={mapping})\n{'=' * 78}")
        print(f"{'arm':12} {'macro-F1':>9} {'TP':>5} {'FP':>5} {'FN':>5} "
              f"{'TN':>5} {'uncert':>7} {'unscored':>9}")
        for a in arms:
            g = scores[a]["gate"][mapping]
            print(f"{a:12} {fmt(g['macro_f1']):>9} {g['tp']:5} {g['fp']:5} "
                  f"{g['fn']:5} {g['tn']:5} {g['n_uncertain']:7} "
                  f"{g['n_unscored']:9}")
        note = scores[arms[0]]["gate"][mapping].get("note")
        if note:
            print(f"\n  NOTE: {note}")

        print(f"\n  false-positive rate by source dataset")
        srcs = sorted(scores[arms[0]]["gate"][mapping]["per_source"])
        print(f"  {'source':16} {'benign':>7} {'hateful':>8}  " +
              "  ".join(f"{a:>14}" for a in arms))
        for src in srcs:
            base = scores[arms[0]]["gate"][mapping]["per_source"][src]
            cells = []
            for a in arms:
                s = scores[a]["gate"][mapping]["per_source"][src]
                rate = f"{s['fp_rate']:.0%}" if s["fp_rate"] is not None else "-"
                cells.append(f"{s['fp']:3} ({rate:>4}) fn={s['fn']:<2}")
            print(f"  {src:16} {base['benign']:7} {base['hateful']:8}  " +
                  "  ".join(f"{c:>14}" for c in cells))

    for dim in ("target_group", "hate_type", "legal", "severity"):
        d0 = scores[arms[0]][dim]
        if not d0["n_items"]:
            continue
        print(f"\n{'=' * 78}\n{dim.upper()}  "
              f"(n={d0['n_items']} scorable items)\n{'=' * 78}")
        print(f"  averaged over {d0['labels_averaged']}")
        if d0["labels_excluded"]:
            excl = {l: scores[arms[0]][dim]['per_label'][l]['support']
                    for l in d0["labels_excluded"]}
            print(f"  excluded (support < MIN_SUPPORT): {excl}")

        print(f"\n{'arm':12} {'macro-F1':>9} {'micro-F1':>9} {'exact':>7} "
              f"{'hamming':>8} {'gold/pred labels':>18}")
        for a in arms:
            d = scores[a][dim]
            ex = d["extra"]
            counts = (f"{ex.get('mean_gold_labels', 0):.2f} / "
                      f"{ex.get('mean_pred_labels', 0):.2f}"
                      if "mean_gold_labels" in ex else "")
            print(f"{a:12} {fmt(d['macro_f1']):>9} {fmt(d['micro_f1']):>9} "
                  f"{d['subset_accuracy']:7.3f} {d['hamming_loss']:8.3f} "
                  f"{counts:>18}")

        if dim == "hate_type":
            print(f"\n  diagnostic ceiling (gold label present in the "
                  f"predicted set):")
            for a in arms:
                e = scores[a][dim]["extra"]
                print(f"    {a:12} any-overlap {e['any_overlap_rate']:.3f}   "
                      f"gold-subset-of-pred {e['gold_subset_of_pred']:.3f}")
        if dim == "severity":
            for a in arms:
                e = scores[a][dim]["extra"]
                print(f"    {a:12} mean |rank error| "
                      f"{fmt(e['mean_abs_rank_error'], '.2f')}   "
                      f"within-one {fmt(e['within_one_accuracy'])}")

        print(f"\n  per-label F1")
        # UNION across arms, not arms[0] alone. The label set is gold labels
        # PLUS predicted ones, so an arm that hallucinates a label absent from
        # gold has it in its table while the others do not. A label missing
        # from an arm means that arm never predicted it and gold never had it,
        # which is F1 = 0, not an error.
        allx = sorted({l for a in arms for l in scores[a][dim]["per_label"]})
        print(f"  {'label':22} {'support':>8}  " +
              "  ".join(f"{a:>10}" for a in arms))
        for l in allx:
            sup = max(scores[a][dim]["per_label"].get(l, {}).get("support", 0)
                      for a in arms)
            mark = "" if sup >= MIN_SUPPORT else "  *"
            cells = "  ".join(
                f"{scores[a][dim]['per_label'].get(l, {}).get('f1', 0.0):10.3f}"
                for a in arms)
            print(f"  {l:22} {sup:8}  {cells}{mark}")

    print(f"\n{'=' * 78}\nHONESTY\n{'=' * 78}")
    print(f"{'arm':12} {'items':>6} {'runs':>6} {'parse fail':>11} "
          f"{'repairs':>8} {'norm':>6} {'uncert':>7} {'latency':>9}")
    for a in arms:
        h = scores[a]["honesty"]
        print(f"{a:12} {h['n_items']:6} {h['total_runs']:6} "
              f"{h['parse_failure_rate']:10.1%} {h['repairs']:8} "
              f"{h['normalisations']:6} {h['uncertain']:7} "
              f"{h['mean_latency_s']:8.1f}s")

    # Aggregate across ALL arms, not arms[0]: kb_version is None by design for
    # zero_shot and few_shot (they consult no knowledge base), so reading the
    # first arm alone reports an empty KB list for a run that used one.
    models = Counter()
    for a in arms:
        models.update(scores[a]["honesty"]["models"])
    kbs = sorted({v for a in arms
                  for v in scores[a]["honesty"]["kb_versions"]})
    prompts = sorted({v for a in arms
                      for v in scores[a]["honesty"]["prompt_versions"]})
    workers = sorted({w for a in arms
                      for w in scores[a]["honesty"]["workers"]})
    print(f"\n  models {dict(models)}")
    print(f"  kb {kbs or '(rag arm only; not used by this run)'}   "
          f"prompt {prompts}   workers {workers}")

    c = scores[arms[0]]["calibration"]
    if c["ece"] is not None:
        bins = ", ".join(f"conf={k}: n={v['n']}, acc={v['accuracy']:.2f}"
                         for k, v in c["bins"].items())
        print(f"  gate ECE {c['ece']:.3f} over {c['n']} items  [{bins}]")
        print(f"  note: {c['note']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--file")
    ap.add_argument("--subset")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gate-mapping", default="both",
                    choices=["strict", "lenient", "both"])
    args = ap.parse_args()

    if args.manifest:
        m = load_manifest(args.manifest)
        subset, limit = m.subset, m.limit
        path = Path(args.file) if args.file else \
            RESULTS / f"slice1_{subset}_live.jsonl"
        mapping = args.gate_mapping
    else:
        subset, limit, path = args.subset, args.limit, Path(args.file)
        mapping = args.gate_mapping

    df = pd.read_parquet(PROCESSED / f"{subset}.parquet")
    if limit:
        df = df.head(limit)

    results = [json.loads(l) for l in
               path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(results)} results from {path}")
    print(f"{len(df)} gold items from {subset}")

    scores = score_all(df, results, gate_mapping=mapping)
    report(scores)

    out = path.with_suffix(".metrics.json")
    out.write_text(json.dumps(scores, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
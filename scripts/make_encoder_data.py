"""
scripts/make_encoder_data.py - build the per-head encoder training sets.
Read-only on the frozen splits, zero API calls, no GPU.

  python -m scripts.make_encoder_data --arm full --dry-run
  python -m scripts.make_encoder_data --arm full
  python -m scripts.make_encoder_data --arm de
  python -m scripts.make_encoder_data --arm kb

WHY ONE DATASET PER HEAD, AND NOT ONE MASKED MULTI-TASK LOSS. `None` means the
source dataset never annotated this dimension; `[]` means it did and the answer
is empty. The distinction is preserved from Phase 2 through parquet, Chroma,
the prompt and the metrics precisely so that scoring can rely on it, and a
shared backbone would have to re-express it as a mask inside a loss function -
the exact place the `legal` bug lived when four functions dropped a dimension
without crashing and 974 of 1,350 predictions were discarded. Here an item
whose gold for a dimension is None simply never enters that head's file. The
masking is structural and there is nothing to get wrong at training time.

THE LEAKAGE FILTER IS NOT OPTIONAL. 18 of 467 items on en_dev_eval_sq3_types
have a >=0.95 cosine twin in en_train, and 19 of 175 on de_legal_dev_eval. A
frozen LLM cannot memorise between runs, which is why this never mattered
before; an encoder can, and Arm A is the ceiling the whole adaptability
comparison is read against. Excluded ids come from
experiments/leakage_train_exclusions.json, and the COUNT is written into this
head's meta so the filter is provable after the fact rather than trusted.

VALIDATION COMES OUT OF TRAIN, NEVER OUT OF DEV. The build guide says "early
stopping on dev macro-F1", but every eval subset is drawn from en_dev / de_dev,
which is what the encoder is scored on. Selecting checkpoints against the
scoring data would hand the encoder a fitting opportunity the LLM never had -
the LLM's equivalent discipline was en_dev_eval_sq1_tune, tuned on and never
reported from. The split here is carved from TRAIN with the project's seed 42.

WHAT THIS SCRIPT DOES NOT DO. It does not binarise. The parquet carries the
gold column as it appears in the frozen split, and the label ORDER lives in the
head's meta.json, read from taxonomy.yaml. Training reads that order and copies
it into the checkpoint, where check_encoder_reachability asserts it - by order,
not by set, because a set comparison passes on a permutation and every
per-label number would then be silently wrong.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

PROCESSED = Path("data/processed")
OUT = Path("data/encoder")
EXCL = Path("experiments/leakage_train_exclusions.json")
KB_RECORDS = Path("kb/records.jsonl")
TAXONOMY = Path("config/taxonomy.yaml")

# head -> (gold column, kind, gate_required)
#
# gate_required MIRRORS metrics.score_multilabel EXACTLY:
#
#     if pred_field != "legal" and not bool(g.gate):
#         continue
#
# The fine-grained dimensions are SCORED on gold-hateful items only, because
# including benign ones fills every label with true negatives and drives F1
# toward 1.0 without measuring anything. Training on a population the head is
# never evaluated on dilutes the positive signal for no benefit: at assembly
# time Result.gate_consistency clears the sub-labels whenever the gate head
# says not-hate, so what the sub-head predicts on benign text is discarded
# anyway.
#
# `legal` is the exception, and for the same reason it is exempt in the
# scorer: BoTox never annotates the gate, so gating on it would skip every
# item. Criminal relevance is independent of the hate gate - a section 185
# insult aimed at one private individual is criminally relevant and not hate
# speech.
HEADS = {
    "hate": ("gate", "binary", False),
    "target_group": ("target_groups", "multilabel", True),
    "hate_type": ("hate_types", "multilabel", True),
    "severity": ("severity", "ordinal", True),
    "legal": ("legal", "multilabel", False),
}

ARMS = {
    "full": {"lang": "en", "source": ["en_train"],
             "heads": ["hate", "target_group", "hate_type", "severity"]},
    "de": {"lang": "de", "source": ["de_train", "de_legal_train"],
           "heads": ["hate", "legal"]},
    # Arm B trains on the KB's EXAMPLES only - base records.jsonl, never an
    # SQ3 round KB, because those contain DEV items through the rule-1
    # carve-out and an encoder trained on them would be evaluated on data it
    # had seen.
    #
    # NO LEGAL HEAD. KB records store legal provenance as meta.stgb - a
    # paragraph number, kept for rendering "flagged under StGB §185" - and not
    # as a taxonomy label. Building one would mean inventing a
    # paragraph-to-label mapping no other part of the system uses. The gap is
    # itself a small finding about the KB's design and is reported rather than
    # patched.
    "kb": {"lang": "en", "source": ["__kb__"],
           "heads": ["hate", "target_group", "hate_type", "severity"]},
}

VAL_FRACTION = 0.1
SEED = 42
MIN_SUPPORT = 10


def clean(val):
    """Parquet round-trip: NaN -> None, numpy array -> list.

    Same helper as metrics.clean and for the same reason: the None / [] /
    ["race"] distinction is what every mask in this script depends on, and
    parquet does not preserve it natively.
    """
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def taxonomy_labels(dim: str) -> list:
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    d = tax["dimensions"][dim]
    if d["type"] == "binary":
        return ["false", "true"]
    labels = d["labels"]
    return list(labels) if isinstance(labels, dict) else list(labels)


def load_kb_examples() -> pd.DataFrame:
    """KB example records as a frame shaped like a split."""
    rows = []
    for line in KB_RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "example":
            continue
        m = r.get("meta", {})
        rows.append({
            "id": r["id"], "text": r["text"], "lang": r["lang"],
            "source": r.get("source", "kb"),
            # meta carries the same three-way convention as the splits, so
            # .get returning None for an absent key is correct rather than a
            # coincidence.
            "gate": m.get("gate"),
            "target_groups": m.get("target_groups"),
            "hate_types": m.get("hate_types"),
            "severity": m.get("severity"),
            "legal": None,
        })
    return pd.DataFrame(rows)


def load_pool(arm: str) -> pd.DataFrame:
    """Source rows for an arm, filtered to the arm's language.

    THE FILTER IS NOT COSMETIC. kb/records.jsonl is bilingual - the German
    example pool is roughly 60 balanced records plus 78 legal illustrations -
    and load_kb_examples returns all of them. Arm B is declared lang="en" and
    trains roberta-base, so without this it would be trained on German text.
    On the frozen splits the filter is a no-op, which is exactly why it has to
    be applied to every path rather than only the one that needs it: a guard
    present in one branch and absent from another is the shape of the bug that
    discarded 974 predictions.
    """
    spec = ARMS[arm]
    if spec["source"] == ["__kb__"]:
        return _by_lang(load_kb_examples(), spec["lang"], "kb examples")
    frames = []
    for name in spec["source"]:
        p = PROCESSED / f"{name}.parquet"
        if not p.exists():
            print(f"  MISSING {p}")
            continue
        df = pd.read_parquet(p)
        df["_split"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _by_lang(pd.concat(frames, ignore_index=True), spec["lang"],
                    "+".join(spec["source"]))


def _by_lang(df: pd.DataFrame, lang: str, label: str) -> pd.DataFrame:
    if df.empty or "lang" not in df.columns:
        return df
    before = len(df)
    out = df[df["lang"].astype(str) == lang].reset_index(drop=True)
    if len(out) != before:
        print(f"  language filter: {label} {before} -> {len(out)} rows "
              f"(lang == {lang!r}); {before - len(out)} dropped")
    return out


def support_of(rows: list, kind: str, labels: list) -> dict:
    c = Counter()
    for v in rows:
        if kind == "binary":
            c[str(bool(v)).lower()] += 1
        elif kind == "ordinal":
            if v is not None:
                c[str(v)] += 1
        else:
            for l in (v or []):
                c[str(l)] += 1
    return {l: c.get(l, 0) for l in labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--dry-run", action="store_true",
                    help="report every count and write nothing. Run this "
                         "first: a head that turns out unbuildable is much "
                         "cheaper to find here than after training.")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--head", default=None,
                    help="build one head only; default is every head of the "
                         "arm")
    ap.add_argument("--exclude-labels", nargs="*", default=[],
                    help="remove these labels from the head's output space "
                         "AND drop every row whose gold contains one. For "
                         "7.10: --exclude-labels grievance.")
    ap.add_argument("--tag", default=None,
                    help="write to data/encoder/{arm}_{tag}/ so a restricted "
                         "build never overwrites the full one")
    args = ap.parse_args()

    spec = ARMS[args.arm]
    print(f"\n{'=' * 78}\nENCODER DATA PREP  arm={args.arm}  "
          f"lang={spec['lang']}\n{'=' * 78}")

    # ------------------------------------------------- the leakage filter
    if not EXCL.exists():
        print(f"\nFAIL: {EXCL} missing. Run `python -m scripts.check_leakage` "
              f"(full scan, no --smoke and no --only) first. Training without "
              f"the filter would inflate Arm A, which is the ceiling the "
              f"whole adaptability comparison is read against.")
        raise SystemExit(1)
    exc = json.loads(EXCL.read_text(encoding="utf-8"))
    excluded_ids = set(exc["ids"].get(spec["lang"], []))
    print(f"\nleakage filter: {len(excluded_ids)} {spec['lang']} ids at "
          f">= {exc['threshold']} cosine against ANY eval subset")

    pool = load_pool(args.arm)
    if pool.empty:
        print("no source rows; nothing to build.")
        raise SystemExit(1)
    print(f"pool: {len(pool)} rows from {', '.join(spec['source'])}")

    # A TAGGED BUILD IS A SEPARATE ARM ON DISK. The 7.10 six-label head must
    # not overwrite the seven-label one it is compared against - they are the
    # two sides of the comparison and both have to survive.
    out_dir = OUT / (f"{args.arm}_{args.tag}" if args.tag else args.arm)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    heads_to_build = [args.head] if args.head else spec["heads"]
    for head in heads_to_build:
        gold_col, kind, gate_required = HEADS[head]
        labels = taxonomy_labels(head)
        # THE OUTPUT SPACE IS RESTRICTED HERE, NOT BY EDITING taxonomy.yaml.
        # The build guide proposed swapping in taxonomy_no_grievance.yaml, but
        # that rebuilds the HateType enum at import time for every process,
        # declares all 30 existing checkpoints stale via taxonomy_version, and
        # needs a byte-identity restore (Copy-Item, never git checkout - git
        # normalises CRLF and changes the content hash).
        #
        # Restricting the head's output space instead leaves taxonomy_version
        # honest: the taxonomy did NOT change, the head's configuration did.
        # The taxonomy file is the LLM's mechanism for a label space and the
        # output layer is the encoder's; they are analogous, which is the
        # point of 7.10, not identical.
        excl = [l for l in args.exclude_labels if l in labels]
        if args.exclude_labels and not excl:
            print(f"  --exclude-labels {args.exclude_labels} matches nothing "
                  f"in {head}'s labels {labels}; nothing removed")
        labels = [l for l in labels if l not in excl]
        print(f"\n{'-' * 78}\nhead: {head}  ({kind}, {len(labels)} labels)"
              + ("  [gold-hateful only]" if gate_required else ""))

        # --- mask: gold is not None, and gate if the scorer requires it --
        keep, dropped_none, dropped_gate, dropped_excl = [], 0, 0, 0
        stripped_excl = 0
        for r in pool.itertuples(index=False):
            v = clean(getattr(r, gold_col, None))
            if v is None:
                dropped_none += 1
                continue
            if gate_required and not bool(getattr(r, "gate", False)):
                dropped_gate += 1
                continue
            # DROP the row, do not blank it. Keeping it with empty gold would
            # train the head that these texts have NO hate_type at all - a
            # claim the data never makes, since hate_type gold is single-label
            # by construction. Dropping models "the label was never conceived
            # of", which is the counterfactual 7.10 is about: a project
            # without `grievance` would have no grievance annotations either.
            #
            # The resulting size difference is NOT a confound to be corrected.
            # It is the measurement: the count printed below is the "new
            # labelled examples required" cell of the 7.10 cost table, against
            # the LLM's zero.
            # STRIP THE LABEL, DROP ONLY WHAT BECOMES EMPTY. An earlier version
            # dropped every row whose gold contained an excluded label, which
            # also removed 26 rows carrying grievance PLUS another label -
            # costing threatening 12 of its 475 instances, and threatening is
            # one of the two labels carrying the entire SQ2 effect.
            #
            # That is the wrong counterfactual. A project that never conceived
            # of `grievance` would still have annotated those 26 texts, with
            # whatever other label applies. Stripping preserves that signal;
            # dropping invents a data gap the counterfactual does not have.
            if excl and kind != "binary":
                vv = [x for x in (v if isinstance(v, list) else [v])
                      if x not in excl]
                if not vv:
                    dropped_excl += 1
                    continue
                if len(vv) != len(v if isinstance(v, list) else [v]):
                    stripped_excl += 1
                v = vv
            keep.append((str(r.id), str(r.text), str(getattr(r, "lang", "")),
                         str(getattr(r, "source", "")),
                         getattr(r, "_split", args.arm), v))
        print(f"  mask         {len(keep)} rows with non-None `{gold_col}` "
              f"({dropped_none} masked out)"
              + (f", {dropped_gate} gold-not-hateful dropped to match the "
                 f"scored population" if gate_required else ""))
        if excl:
            print(f"  EXCLUDED     {excl} -> {len(labels)}-label output "
                  f"space, {dropped_excl} rows dropped, {stripped_excl} rows "
                  f"kept with the label stripped")
            print(f"               ^ this count is the '{','.join(excl)} "
                  f"labelled examples required' cell of the 7.10 cost table")
        if not keep:
            print(f"  UNBUILDABLE: no row in this pool annotates `{head}`.")
            summary[head] = {"buildable": False,
                             "reason": f"no non-None {gold_col} in pool"}
            continue

        # --- leakage exclusion -----------------------------------------
        before = len(keep)
        keep = [k for k in keep if k[0] not in excluded_ids]
        n_excluded = before - len(keep)
        print(f"  leakage      {n_excluded} row(s) excluded, {len(keep)} "
              f"remain")

        # --- support ----------------------------------------------------
        sup = support_of([k[5] for k in keep], kind, labels)
        thin = [l for l, n in sup.items() if n < MIN_SUPPORT]
        print(f"  support      "
              + "  ".join(f"{l}={n}" for l, n in sup.items()))
        if thin:
            # Reported, not fixed. sklearn would silently score an absent
            # label 0.0, which reads as "the model got it wrong" rather than
            # "the model was never shown it". Arm B is expected to hit this
            # on most labels and that IS the Arm B result.
            print(f"  THIN         below MIN_SUPPORT={MIN_SUPPORT}: {thin}")

        # --- validation split, carved from TRAIN ------------------------
        df = pd.DataFrame(keep, columns=["id", "text", "lang", "source",
                                         "split", gold_col])
        n_val = max(1, int(round(len(df) * args.val_fraction)))
        # Shuffled with the project seed rather than taken off the end: the
        # frozen splits are ordered by source, so a tail slice would be one
        # dataset and early stopping would select against a distribution the
        # model is not scored on.
        shuffled = df.sample(frac=1.0, random_state=args.seed)
        val = shuffled.head(n_val).reset_index(drop=True)
        train = shuffled.tail(len(df) - n_val).reset_index(drop=True)
        assert set(train["id"]) & set(val["id"]) == set(), \
            "train and val overlap - the shuffle-then-slice is wrong"
        print(f"  split        train {len(train)}  val {len(val)}  "
              f"(seed {args.seed}, {args.val_fraction:.0%} held out FROM "
              f"TRAIN, not from dev)")

        meta = {
            "head": head, "dimension": head, "kind": kind,
            "labels": labels,                       # ORDER matters
            "arm": args.arm, "lang": spec["lang"],
            "source_splits": spec["source"],
            "n_pool": len(pool), "n_masked_out": dropped_none,
            "gate_required": gate_required,
            "n_dropped_not_hateful": dropped_gate,
            "leakage_threshold": exc["threshold"],
            "leakage_excluded": n_excluded,
            "n_train": len(train), "n_val": len(val),
            "val_fraction": args.val_fraction, "seed": args.seed,
            "support_train_plus_val": sup,
            "labels_below_min_support": thin,
            "min_support": MIN_SUPPORT,
            "source_mix": dict(Counter(df["source"])),
            "labels_excluded_from_output": excl,
            "n_dropped_excluded_label": dropped_excl,
            "n_stripped_excluded_label": stripped_excl,
            "tag": args.tag,
        }
        summary[head] = {"buildable": True, **{k: meta[k] for k in
                                               ("n_train", "n_val",
                                                "leakage_excluded")},
                         "thin": thin}

        if args.dry_run:
            print("  (dry run - nothing written)")
            continue
        train.to_parquet(out_dir / f"{head}_train.parquet", index=False)
        val.to_parquet(out_dir / f"{head}_val.parquet", index=False)
        (out_dir / f"{head}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"  wrote        {out_dir / f'{head}_train.parquet'}, "
              f"{head}_val.parquet, {head}_meta.json")

    # ------------------------------------------------------------ verdict
    print(f"\n{'=' * 78}\nSUMMARY  arm={args.arm}")
    for head, s in summary.items():
        if not s.get("buildable"):
            print(f"  {head:16} UNBUILDABLE - {s['reason']}")
        else:
            print(f"  {head:16} train {s['n_train']:>6}  val {s['n_val']:>5}  "
                  f"leakage-excluded {s['leakage_excluded']:>3}"
                  + (f"  THIN {s['thin']}" if s["thin"] else ""))
    if args.arm == "kb":
        print("\n  Arm B has no `legal` head by construction: KB records "
              "carry meta.stgb, a paragraph number, not a taxonomy label.")
    print(f"\n  Next: python -m scripts.check_encoder_reachability "
          f"--arm {args.arm}")
    print("  Hop 1 should turn from NOT BUILT to ok; hops 2-8 stay NOT BUILT "
          f"until 7.6 trains a checkpoint.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
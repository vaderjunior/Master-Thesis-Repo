"""
scripts/check_encoder_reachability.py - does each encoder head reach the report?
Read-only, zero API calls, no GPU.

  python -m scripts.check_encoder_reachability --arm full
  python -m scripts.check_encoder_reachability --arm full --break-head severity

RUN IT BEFORE ANYTHING IS BUILT. Every hop reports NOT BUILT rather than
crashing, so today it is a build checklist that prints the exact contract, and
after 7.5-7.8 it is the guard. Same discipline as check_sq3_reachability, which
was run at --k-feedback 0 and confirmed to exit non-zero before round 1.

THE FAILURE THIS EXISTS TO CATCH. Adding the `legal` dimension broke four
functions, none of which crashed, and 974 of 1,350 predictions were discarded
while the report looked exactly like a model that had never learned the task.
Five encoder heads is the same shape five times over. A head that trains fine
and is dropped between prediction and scoring produces a clean, plausible,
entirely meaningless table.

It happened twice more since: `legal` was absent from vote.py's aggregator for
a whole run (check_dimension_reach found it six weeks later), and the SQ3
feedback bucket had to be wired into all three retrieval strategies because a
bucket present in one code path and absent from another is the same bug.

SEEN TO FAIL. --break-head <name> simulates that head vanishing from the
scorer. The check MUST exit non-zero naming it. A guard nobody has watched fail
is not a guard, and this is part of the procedure rather than an afterthought.

THE HOPS, each asserted separately so a failure says WHERE:
  1 train data   the head's training file exists, has rows, and every label
                 clears MIN_SUPPORT
  2 leakage      the >=0.95 near-duplicate filter was applied and its count
                 recorded. Not optional: 18 of 467 items on
                 en_dev_eval_sq3_types have a >=0.95 twin in en_train, and an
                 encoder memorises where a frozen LLM cannot
  3 checkpoint   the directory exists and carries meta.json with every stamp
  4 taxonomy     the checkpoint's taxonomy_version matches the taxonomy in
                 place NOW. The MultiLabelBinarizer alignment is valid for
                 exactly one taxonomy version, and the new-label experiment
                 deliberately trains under two
  5 labels       the checkpoint's label list matches taxonomy.yaml ORDER, not
                 just its contents. A set comparison cannot see a reordering,
                 which is how a Phase 8 retrieval check missed that RRF had
                 reordered the same records
  6 predictions  every eval item has a prediction for this head
  7 scored       the scorer returns n_items > 0 for this dimension
  8 reported     the dimension appears in the metrics file with a macro-F1 or
                 an explicit, readable reason for its absence

THE ARTEFACT CONTRACT, defined here because the guard has to assert against
something.

  data/encoder/{arm}/{head}_{split}.parquet
      columns: id, text, lang, label columns as in the frozen splits.
      An item enters a head ONLY if its gold for that dimension is not None.
      `None` means the source dataset never annotated it; `[]` means it did and
      the answer is empty. Masking by construction rather than inside a loss
      function is why the design is one model per head.

  models/encoder/{arm}/{head}/seed{N}/meta.json
      head, dimension, labels (ORDERED, from taxonomy.yaml)
      taxonomy_version, code_version, seed
      base_model, max_seq_length, threshold
      n_train, n_val, leakage_excluded
      train_seconds, epochs, batch_size
      train_seconds and n_train are not bookkeeping: they are two cells of the
      7.10 cost table, and collecting them after the fact means re-training.

  experiments/results/encoder_{arm}_seed{N}_live.jsonl
      ItemResult records, so score_all / check_dimension_reach /
      make_comparability / the paired bootstrap all work unchanged and the
      encoder-vs-LLM comparison runs through ONE scoring implementation.
      run_slice1 carried its own copy of the scoring functions for two phases
      and missed a fix; the rule is one implementation, always.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

DATA = Path("data/encoder")
MODELS = Path("models/encoder")
RESULTS = Path("experiments/results")
TAXONOMY = Path("config/taxonomy.yaml")

# Which heads belong to which arm, and what each is evaluated on. Derived from
# the mask semantics rather than chosen: GAHD annotates the gate only, BoTox
# annotates legal only, Implicit Hate is the only source with hate_type gold,
# and severity comes from the MHS hateful subset.
ARMS = {
    "full": {                      # Arm A, full supervision, English
        "lang": "en",
        "heads": {
            "hate": "en_dev_eval_main",
            "target_group": "en_dev_eval_targets",
            "hate_type": "en_dev_eval_sq3_types",
            "severity": "en_dev_eval_severity",
        },
    },
    "kb": {                        # Arm B, KB examples only
        "lang": "en",
        "heads": {
            "hate": "en_dev_eval_main",
            "target_group": "en_dev_eval_targets",
            "hate_type": "en_dev_eval_sq3_types",
            "severity": "en_dev_eval_severity",
        },
    },
    "de": {                        # German
        "lang": "de",
        "heads": {
            "hate": "de_dev_eval",
            "legal": "de_legal_dev_eval",
        },
    },
    # 7.10. A ONE-HEAD ARM, deliberately. Its gate, target_group and severity
    # heads are BORROWED from the untagged arm at the same seed, which is what
    # makes the comparison isolate the label space: the gate head is
    # byte-identical across the two arms, and it costs hate_type -0.092 on
    # this subset, so an unmatched gate would swamp the effect. The borrowed
    # heads are recorded in encoder_meta.heads_borrowed and are checked under
    # `full`, not here.
    "full_nogrievance": {
        "lang": "en",
        "heads": {"hate_type": "en_dev_eval_sq3_types"},
    },
}

REQUIRED_META = [
    "head", "dimension", "labels", "taxonomy_version", "code_version", "seed",
    "base_model", "max_seq_length", "threshold", "n_train", "n_val",
    "leakage_excluded", "train_seconds", "epochs", "batch_size",
]


def taxonomy_labels(dim: str) -> list:
    """Ordered labels for a dimension, straight from taxonomy.yaml.

    ORDER, not contents. The binarizer aligns encoder output columns to this
    order so encoder and LLM predictions index identically; a set comparison
    would pass on a permutation and every per-label number would be wrong.
    """
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    d = tax["dimensions"][dim]
    if d["type"] == "binary":
        return ["false", "true"]
    labels = d["labels"]
    return list(labels) if isinstance(labels, dict) else list(labels)


class Hops:
    """Collects per-hop results so the report says WHERE, not just whether."""

    def __init__(self, head: str):
        self.head = head
        self.rows = []

    def add(self, name: str, ok: bool | None, detail: str = ""):
        # ok=None means NOT BUILT, which is distinct from failing. Before 7.5
        # every hop is None and that is the expected state; after 7.8 a None
        # is as much a failure as a False.
        self.rows.append((name, ok, detail))

    @property
    def failed(self) -> bool:
        return any(ok is False for _, ok, _ in self.rows)

    @property
    def unbuilt(self) -> bool:
        return any(ok is None for _, ok, _ in self.rows)

    def show(self):
        print(f"\n  head: {self.head}")
        for name, ok, detail in self.rows:
            mark = "  ok " if ok else (" .. " if ok is None else "FAIL")
            print(f"    [{mark}] {name:22} {detail}")


def check_head(arm: str, head: str, subset: str, seed: int,
               broken: str | None) -> Hops:
    h = Hops(head)
    dim = head
    # want_labels is resolved after meta is read, because a tagged head's
    # output space is deliberately a subset of the taxonomy's.
    want_labels = taxonomy_labels(dim)

    # --- hop 1: training data ------------------------------------------
    train = DATA / arm / f"{head}_train.parquet"
    val = DATA / arm / f"{head}_val.parquet"
    if not train.exists():
        h.add("1 train data", None, f"missing {train}")
    else:
        try:
            import pandas as pd
            df = pd.read_parquet(train)
            h.add("1 train data", len(df) > 0,
                  f"{len(df)} rows, val "
                  f"{'present' if val.exists() else 'MISSING'}")
            if not val.exists():
                # Early stopping must not read the eval subsets: they are
                # drawn from en_dev, which is what the encoder is scored on.
                # The LLM's equivalent discipline was en_dev_eval_sq1_tune -
                # tuned on, never reported from.
                h.add("1b val split", False,
                      "no validation split; early stopping would have to read "
                      "the eval subsets, which are drawn from the same dev "
                      "pool the encoder is scored on")
        except Exception as e:
            h.add("1 train data", False, f"unreadable: {e}")

    # --- hop 2: leakage filter -----------------------------------------
    report = Path("experiments/leakage_report.json")
    meta_path = MODELS / arm / head / f"seed{seed}" / "meta.json"
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            h.add("3 checkpoint", False, f"meta.json unreadable: {e}")
    if not report.exists():
        h.add("2 leakage filter", False,
              "experiments/leakage_report.json missing - run check_leakage")
    elif meta is None:
        h.add("2 leakage filter", None, "no checkpoint meta to read it from")
    else:
        n = meta.get("leakage_excluded")
        h.add("2 leakage filter", isinstance(n, int),
              f"{n} train rows excluded at >=0.95 cosine"
              if isinstance(n, int) else
              "meta.leakage_excluded absent - the filter may not have run, "
              "and 3.9% of en_dev_eval_sq3_types has a >=0.95 twin in "
              "en_train")

    # --- hop 3: checkpoint ---------------------------------------------
    if meta is None:
        h.add("3 checkpoint", None, f"missing {meta_path}")
    else:
        missing = [k for k in REQUIRED_META if k not in meta]
        h.add("3 checkpoint", not missing,
              "all stamps present" if not missing
              else f"meta.json missing {missing}")

    # --- hop 4: taxonomy version ---------------------------------------
    if meta is None:
        h.add("4 taxonomy", None, "no checkpoint")
    else:
        from src.hsrag.prompt import taxonomy_version
        now, then = taxonomy_version(), meta.get("taxonomy_version")
        h.add("4 taxonomy", now == then,
              f"checkpoint {then} == current {now}" if now == then else
              f"checkpoint trained under {then}, taxonomy is now {now}. The "
              f"binarizer alignment is valid for exactly one version; this "
              f"checkpoint is stale and must not be scored.")

    # --- hop 5: label order --------------------------------------------
    if meta is None:
        h.add("5 label order", None, "no checkpoint")
    else:
        got = list(meta.get("labels") or [])
        # Subtract any deliberately excluded labels before comparing. Still by
        # ORDER, never by set.
        excl = meta.get("labels_excluded_from_output") or []
        want_labels = [l for l in want_labels if l not in excl]
        ok = got == want_labels
        h.add("5 label order", ok,
              (f"{len(got)} labels in taxonomy order"
               + (f", excluding {excl} by design" if excl else "")) if ok else
              f"got {got} want {want_labels}"
              + ("  (SAME SET, DIFFERENT ORDER - every per-label number would "
                 "be silently wrong)" if set(got) == set(want_labels) else ""))

    # --- hop 6: predictions emitted ------------------------------------
    res = RESULTS / f"encoder_{arm}_seed{seed}_live.jsonl"
    rows = []
    if not res.exists():
        h.add("6 predictions", None, f"missing {res}")
    else:
        for line in res.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "_manifest" not in r:
                rows.append(r)
        if broken == head:
            # THE DELIBERATE BREAK. Simulates the head vanishing between
            # prediction and scoring - the shape that discarded 974
            # predictions without crashing.
            for r in rows:
                if r.get("result"):
                    r["result"].pop(dim, None)
        n_pred = sum(1 for r in rows
                     if r.get("result") is not None and dim in r["result"])
        h.add("6 predictions", n_pred == len(rows) and n_pred > 0,
              f"{n_pred}/{len(rows)} items carry a `{dim}` prediction"
              + ("   <-- --break-head active" if broken == head else ""))

    # --- hops 7 and 8: scored, and in the report -----------------------
    if not rows:
        h.add("7 scored", None, "no predictions")
        h.add("8 reported", None, "no predictions")
        return h

    try:
        import pandas as pd
        from scripts.check_sq3_coverage import PROCESSED
        from src.hsrag.metrics import (score_gate, score_multilabel,
                                       score_severity)
        GOLD_COL = {"target_group": "target_groups",
                    "hate_type": "hate_types", "legal": "legal"}
        df = pd.read_parquet(PROCESSED / f"{subset}.parquet")
        gold = {str(r.id): r for r in df.itertuples(index=False)}
        # FILTER TO THIS HEAD'S SUBSET. The results file pools every eval
        # subset for the arm - 970 items across five for `full` - and passing
        # all of them to the scorer got the right count only because the
        # None-mask filtered them by accident: hate_type gold is None outside
        # en_dev_eval_sq3_types. That is the right answer for the wrong
        # reason, and it would not catch a genuine subset mix-up. Where the
        # stamp is absent (pre-2026-08-17 encoder runs, or an LLM results
        # file), fall back to the unfiltered rows rather than dropping
        # everything.
        stamped = [r for r in rows
                   if (r.get("encoder_meta") or {}).get("subset") == subset]
        if stamped:
            n_other = len(rows) - len(stamped)
            rows = stamped
            if n_other:
                h.add("6b subset filter", True,
                      f"{len(rows)} of {len(rows) + n_other} records are "
                      f"stamped {subset}; the rest belong to other subsets")

        if dim == "hate":
            s = score_gate(rows, gold)
            n_items, macro = s.n_scored, s.macro_f1
        elif dim == "severity":
            s = score_severity(rows, gold)
            n_items, macro = s.n_items, s.macro_f1
        else:
            s = score_multilabel(rows, gold, GOLD_COL[dim], dim)
            n_items, macro = s.n_items, s.macro_f1

        h.add("7 scored", n_items > 0,
              f"{n_items} scorable items on {subset}"
              if n_items > 0 else
              f"ZERO scorable items on {subset}. The predictions exist and "
              f"the scorer discards them - this is the 974-of-1350 shape.")
        # A None macro is not automatically a fault: a single-class gate
        # subset makes it undefined, and MIN_SUPPORT can exclude every label.
        # What it must never be is None WITHOUT a reason on the record.
        note = getattr(s, "note", None) or (
            f"no label clears MIN_SUPPORT (excluded "
            f"{getattr(s, 'labels_excluded', [])})" if macro is None else "")
        # A macro of exactly 0.000 is legitimate - Arm B's hate_type predicts
        # nothing at all, max sigmoid 0.212 against a 0.5 threshold, which IS
        # the Arm B result - but at this hop it is indistinguishable from a
        # head that trained and then failed to reach the scorer. Report the
        # predicted-positive count beside it so the two can never be confused.
        extra = ""
        if macro is not None and macro == 0.0 and dim != "hate":
            npos = sum(len(r["result"].get(dim) or []) for r in rows
                       if r.get("result")) if dim != "severity" else \
                sum(1 for r in rows
                    if r.get("result") and r["result"].get("severity"))
            extra = (f"  ({npos} predicted positives across {len(rows)} "
                     f"items - {'predicts NOTHING, which is a result' if not npos else 'predicts, and is wrong'})")
        h.add("8 reported", macro is not None or bool(note),
              f"macro-F1 {macro:.3f}{extra}" if macro is not None else
              (f"macro undefined, reason recorded: {note}" if note else
               "macro-F1 is None and NO reason is recorded - unreadable"))
    except Exception as e:
        h.add("7 scored", False, f"scoring raised: {type(e).__name__}: {e}")
        h.add("8 reported", False, "not reached")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="full", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--break-head", default=None,
                    help="simulate this head vanishing from the scorer. The "
                         "check MUST exit non-zero naming it.")
    args = ap.parse_args()

    spec = ARMS[args.arm]
    print(f"\n{'=' * 78}\nENCODER REACHABILITY  arm={args.arm} "
          f"seed={args.seed}\n{'=' * 78}")
    if args.break_head:
        print(f"\n  --break-head {args.break_head}: this run is EXPECTED to "
              f"fail naming that head.")
        if args.break_head not in spec["heads"]:
            print(f"  but {args.break_head} is not a head of arm "
                  f"{args.arm}: {sorted(spec['heads'])}")
            sys.exit(2)

    all_hops = [check_head(args.arm, head, subset, args.seed, args.break_head)
                for head, subset in spec["heads"].items()]
    for h in all_hops:
        h.show()

    failed = [h.head for h in all_hops if h.failed]
    unbuilt = [h.head for h in all_hops if h.unbuilt and not h.failed]

    print(f"\n{'-' * 78}")
    if failed:
        print(f"FAIL: {len(failed)} head(s) broken: {', '.join(failed)}")
    if args.break_head:
        # THE GUARD MUST BE WATCHED TO FAIL, and "the run exited non-zero" is
        # not the same as "the guard caught the break". Before 7.6 the break is
        # injected at hop 6, which reports NOT BUILT, so the injection is
        # invisible and the exit code comes from the missing artefacts instead.
        # That looks like it worked and proves nothing.
        # check_sq3_reachability was run at --k-feedback 0 and SEEN to exit
        # non-zero before round 1; this owes the same demonstration on a real
        # checkpoint, and says so until it gets one.
        if args.break_head in failed:
            print(f"  `{args.break_head}` is among them, so the guard works. "
                  f"This is the expected failure.")
        elif args.break_head in unbuilt:
            print(f"\n  GUARD UNVERIFIED: `{args.break_head}` is NOT BUILT, "
                  f"so removing its prediction changed nothing observable and "
                  f"the non-zero exit above comes from the missing artefacts, "
                  f"not from the guard. Re-run with --break-head after 7.6 "
                  f"trains a real checkpoint. Until then this guard has never "
                  f"been seen to fail and must not be trusted.")
            sys.exit(2)
        else:
            print(f"\n  GUARD FAILED ITS OWN TEST: `{args.break_head}` was "
                  f"removed from the scorer and the check did not notice. "
                  f"That is the one thing it exists to do.")
            sys.exit(2)
    if unbuilt:
        print(f"NOT BUILT: {', '.join(unbuilt)}")
        print("  Expected before 7.5-7.8. After training, a NOT BUILT is as "
              "much a failure as a FAIL.")
    if not failed and not unbuilt:
        print(f"PASS: all {len(all_hops)} heads reach the report table.")
        print("  This is a statement about the CODE PATH, not about whether "
              "the numbers are any good.")
    print(f"{'=' * 78}\n")
    sys.exit(1 if (failed or unbuilt) else 0)


if __name__ == "__main__":
    main()
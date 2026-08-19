"""
scripts/train_encoder.py - train one encoder head. GPU, no API calls.

  python -m scripts.train_encoder --arm full --head hate_type --seed 0 --dry-run
  python -m scripts.train_encoder --arm full --head hate_type --seed 0
  python -m scripts.train_encoder --arm de   --head legal     --seed 0

ONE MODEL PER HEAD. Not a shared backbone with a masked multi-task loss. The
`None` vs `[]` distinction - None means the source dataset never annotated this
dimension, [] means it did and the answer is empty - is preserved from Phase 2
through parquet, Chroma, the prompt and the metrics. A shared backbone would
have to re-express it as a mask inside a loss function, which is exactly where
the `legal` bug lived when four functions dropped a dimension without crashing
and 974 of 1,350 predictions were discarded. make_encoder_data already applied
the mask by construction, so there is nothing here to get wrong.

EVERY STAMP THE GUARD ASKS FOR IS WRITTEN. check_encoder_reachability asserts
on meta.json: taxonomy_version must match the taxonomy in place at scoring
time, and the label list must match taxonomy.yaml ORDER, not just its contents.
A set comparison passes on a permutation and every per-label number would then
be silently wrong - the same blindness that let a Phase 8 retrieval check miss
that RRF had reordered the same records.

train_seconds AND n_train ARE NOT BOOKKEEPING. They are two cells of the 7.10
cost table, which is the thesis's adaptability claim made concrete: one line in
taxonomy.yaml and zero training against N labelled examples and M seconds.
Collecting them after the fact means re-training.

DECISIONS, PRE-REGISTERED HERE RATHER THAN DISCOVERED LATER.

  threshold 0.5, fixed. Tuning it on anything would hand the encoder a fitting
  opportunity the LLM never had. A rate-matched secondary reading - the
  threshold at which the encoder predicts as many positives per item as the
  LLM - belongs in the analysis, not in training, because prediction
  cardinality is now a measured mechanism rather than a nuisance: retrieval
  tightens the prediction set, which helps where the model over-predicts
  (hate_type, legal, the gate) and hurts where it under-predicts
  (target_group).

  NO class weighting and NO resampling. Standard practice, and the imbalance
  is itself reportable: target_group runs race 5313 against disability 500.
  Weighting would improve the numbers and make the encoder a less honest
  baseline for what a practitioner gets by default.

  severity is trained as PLAIN 3-CLASS, ordinality unexploited. The LLM's vote
  takes an ordinal median so that low-vs-high counts more than low-vs-medium;
  this does not. Recorded as a limitation rather than fixed, because ordinal
  regression is a second modelling decision the comparison does not need.

  ARM B GETS FIXED EPOCHS AND NO EARLY STOPPING. Its validation splits are 15
  / 6 / 7 / 4 items. Early stopping on 4 items selects on noise. Twenty epochs
  fixed, recorded in meta.

  ARM B'S GATE HEAD IS 86.5% POSITIVE (128 true / 20 false) against an eval
  set that is 30.4% hateful. KB examples exist to illustrate labels, so they
  are overwhelmingly hateful. It will over-flag severely. That is the Arm B
  result - a few hundred examples cannot train five heads - and it is written
  down here so it is not later mistaken for a broken run.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA = Path("data/encoder")
MODELS = Path("models/encoder")
TAXONOMY = Path("config/taxonomy.yaml")

BASE_MODEL = {"en": "roberta-base", "de": "deepset/gbert-base"}
GOLD_COL = {"hate": "gate", "target_group": "target_groups",
            "hate_type": "hate_types", "severity": "severity",
            "legal": "legal"}


def clean(val):
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


def encode_targets(values, kind: str, labels: list):
    """Gold column -> model targets, in taxonomy label ORDER.

    The order is the contract between this file, the checkpoint's meta.json and
    the scorer. It is asserted at three points and never inferred.
    """
    idx = {l: i for i, l in enumerate(labels)}
    if kind == "binary":
        return np.array([int(bool(v)) for v in values], dtype=np.int64)
    if kind == "ordinal":
        return np.array([idx[str(v)] for v in values], dtype=np.int64)
    Y = np.zeros((len(values), len(labels)), dtype=np.float32)
    for i, v in enumerate(values):
        for l in (clean(v) or []):
            # A label outside the taxonomy is a data error, not something to
            # skip quietly. make_encoder_data reads the same taxonomy, so this
            # can only fire if the taxonomy changed between the two steps -
            # which is precisely what taxonomy_version exists to catch.
            if str(l) not in idx:
                raise ValueError(
                    f"label {l!r} is not in the taxonomy's {labels}. The "
                    f"taxonomy changed after make_encoder_data ran; rebuild "
                    f"the head's data.")
            Y[i, idx[str(l)]] = 1.0
    return Y


def macro_f1(y_true, y_pred, multilabel: bool, n_labels: int) -> float:
    """Macro-F1 for early stopping only. Deliberately NOT metrics.score_*.

    The project scorer applies MIN_SUPPORT, the gold-hateful filter and the
    None/[] mask, all of which belong to REPORTING on a frozen eval subset.
    This runs on a TRAIN-internal validation split for checkpoint selection.
    Reusing the reporting scorer here would blur the line between the number
    used to choose a model and the number used to judge it - which is the line
    en_dev_eval_sq1_tune exists to keep.
    """
    f1s = []
    for k in range(n_labels):
        if multilabel:
            t, p = y_true[:, k] > 0.5, y_pred[:, k] > 0.5
        else:
            t, p = y_true == k, y_pred == k
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        fn = int((t & ~p).sum())
        if tp + fn == 0:
            continue                       # label absent from val: undefined
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "de", "kb"])
    ap.add_argument("--head", required=True, choices=sorted(GOLD_COL))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=None,
                    help="default 5 (20 for arm kb, which has no usable "
                         "validation split)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tag", default=None,
                    help="read data/encoder/{arm}_{tag}/ and write "
                         "models/encoder/{arm}_{tag}/. For 7.10: "
                         "--tag nogrievance")
    ap.add_argument("--bf16", action="store_true",
                    help="the 3070 Ti is Ampere so bf16 works and is more "
                         "stable than fp16; fp16 is the default because the "
                         "build guide specifies it")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything, train 2 steps, write nothing. "
                         "Exercises tokenizer, binarizer, collator and loss "
                         "before any GPU time is committed.")
    args = ap.parse_args()

    import torch
    from transformers import (AutoConfig, AutoModelForSequenceClassification,
                              AutoTokenizer, BertConfig,
                              BertForSequenceClassification, BertTokenizerFast,
                              EarlyStoppingCallback, Trainer, TrainingArguments)
    from src.hsrag.classify import code_version
    from src.hsrag.prompt import taxonomy_version

    head, arm = args.head, args.arm
    # A TAGGED BUILD IS A SEPARATE ARM ON DISK, so the 7.10 six-label head can
    # never overwrite the seven-label one it is compared against.
    slug = f"{arm}_{args.tag}" if args.tag else arm
    d = DATA / slug
    meta_in = json.loads((d / f"{head}_meta.json").read_text(encoding="utf-8"))
    lang, kind = meta_in["lang"], meta_in["kind"]
    # THE OUTPUT SPACE IS A PROPERTY OF THE HEAD, NOT OF THE TAXONOMY. The
    # build guide proposed swapping taxonomy.yaml for the six-label variant;
    # restricting the head instead keeps taxonomy_version honest at
    # taxonomy-072387cd - the taxonomy did not change, this head's
    # configuration did - and avoids declaring all 30 existing checkpoints
    # stale. The taxonomy file is the LLM's mechanism for a label space and
    # the output layer is the encoder's: analogous, which is the point of
    # 7.10, not identical.
    excl = meta_in.get("labels_excluded_from_output") or []
    labels = [l for l in taxonomy_labels(head) if l not in excl]
    assert labels == meta_in["labels"], (
        f"label order changed since make_encoder_data ran: data says "
        f"{meta_in['labels']}, taxonomy minus {excl} says {labels}. "
        f"Rebuild the head.")
    multilabel = kind == "multilabel"
    base = BASE_MODEL[lang]
    epochs = args.epochs if args.epochs else (20 if arm == "kb" else 5)

    print(f"\n{'=' * 78}\nTRAIN  arm={arm} head={head} seed={args.seed}")
    print(f"  base model      {base}")
    print(f"  kind            {kind}, {len(labels)} labels in taxonomy order")
    print(f"  labels          {labels}")
    print(f"  cuda            {torch.cuda.is_available()}")
    print(f"{'=' * 78}")

    # GERMAN USES THE Bert* CLASSES EXPLICITLY, ALL THREE OF THEM.
    #
    # deepset/gbert-base was published before two conventions transformers now
    # relies on, and both Auto* paths fail on it:
    #   - config.json has no `model_type`. Transformers used to infer it from
    #     `architectures`; 5.x removed that fallback, so AutoConfig raises
    #     "Unrecognized model in deepset/gbert-base".
    #   - the repo ships vocab.txt with no tokenizer.json, so AutoTokenizer
    #     attempts a slow->fast conversion that needs sentencepiece, which is
    #     not installed.
    #
    # Naming the classes is one decision rather than two workarounds, and it
    # avoids adding a dependency to an environment where a forced torch
    # reinstall already broke fsspec once. It is also better practice here:
    # the architecture and tokenizer become recorded choices in meta.json
    # rather than inferences that a future transformers release could change
    # underneath the checkpoints.
    if lang == "de":
        ConfigCls = BertConfig
        ModelCls = BertForSequenceClassification
        tok = BertTokenizerFast.from_pretrained(base)
    else:
        ConfigCls = AutoConfig
        ModelCls = AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(base)
    tok_class = type(tok).__name__
    print(f"  config/model    {ConfigCls.__name__} / {ModelCls.__name__}")
    print(f"  tokenizer       {tok_class}")

    gold_col = GOLD_COL[head]
    frames = {}
    for split in ("train", "val"):
        df = pd.read_parquet(d / f"{head}_{split}.parquet")
        texts = [str(t) for t in df["text"]]
        enc = tok(texts, truncation=True, max_length=args.max_seq_length,
                  padding=False)
        y = encode_targets(list(df[gold_col]), kind, labels)
        n_trunc = sum(1 for ids in enc["input_ids"]
                      if len(ids) >= args.max_seq_length)
        print(f"  {split:5} {len(df):>6} rows, {n_trunc} truncated at "
              f"{args.max_seq_length} ({n_trunc / max(len(df), 1):.2%})")
        frames[split] = (enc, y, n_trunc, len(df))

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc, self.y = enc, y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            out = {k: self.enc[k][i] for k in ("input_ids", "attention_mask")}
            out["labels"] = self.y[i]
            return out

    from transformers import DataCollatorWithPadding
    collate = DataCollatorWithPadding(tok)

    cfg = ConfigCls.from_pretrained(
        base, num_labels=len(labels),
        problem_type="multi_label_classification" if multilabel
        else "single_label_classification")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ModelCls.from_pretrained(base, config=cfg)

    def compute_metrics(p):
        logits = p.predictions
        y = p.label_ids
        if multilabel:
            pred = (1 / (1 + np.exp(-logits)) >= args.threshold).astype(float)
        else:
            pred = logits.argmax(axis=-1)
        return {"macro_f1": macro_f1(y, pred, multilabel, len(labels))}

    out_dir = MODELS / slug / head / f"seed{args.seed}"
    # ARM B gets fixed epochs: its validation splits are 4 to 15 items and
    # early stopping on 4 items selects on noise.
    early = arm != "kb"
    targs = TrainingArguments(
        output_dir=str(out_dir / "_hf"),
        seed=args.seed, data_seed=args.seed,
        num_train_epochs=2 if args.dry_run else epochs,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr, warmup_ratio=0.1, weight_decay=0.01,
        fp16=not args.bf16, bf16=args.bf16,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=early,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, logging_steps=50, report_to=[],
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=DS(*frames["train"][:2]),
        eval_dataset=DS(*frames["val"][:2]),
        data_collator=collate, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        if early else [],
    )

    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    ev = trainer.evaluate()
    print(f"\n  val macro_f1    {ev.get('eval_macro_f1'):.4f}")
    print(f"  train_seconds   {train_seconds}")

    if args.dry_run:
        print(f"\n  DRY RUN - 2 steps only, nothing written. The tokenizer, "
              f"binarizer, collator, loss and metric all ran.\n{'=' * 78}\n")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    meta = {
        "head": head, "dimension": head, "kind": kind,
        "labels": labels,
        "taxonomy_version": taxonomy_version(),
        "code_version": code_version(),
        "seed": args.seed,
        "base_model": base, "tokenizer_class": tok_class,
        "tokenizer_is_fast": bool(getattr(tok, "is_fast", False)),
        "config_class": ConfigCls.__name__,
        "model_class": ModelCls.__name__,
        "transformers_version": __import__("transformers").__version__,
        # THE STAMP GAP THIS CLOSES. code_version() hashes src/hsrag/*.py and
        # this script lives in scripts/, so nothing recorded what it did.
        # warmup_ratio and weight_decay affect the result and were in the
        # script only; transformers 5.12.1 already warns that warmup_ratio
        # will be removed, so this file will very likely change mid-phase.
        # A results set spanning a code change with every stamp constant is
        # exactly the legal_dev_peasec incident - 150 items aggregated by one
        # version of vote.py and 25 by another, in one file, undetectable.
        # Hashing the script means two checkpoints trained under different
        # hyperparameters can never look identical.
        "train_script_sha": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()[:8],
        "warmup_ratio": targs.warmup_ratio,
        "weight_decay": targs.weight_decay,
        "precision": "bf16" if targs.bf16 else ("fp16" if targs.fp16
                                                else "fp32"),
        "max_seq_length": args.max_seq_length,
        "truncated_train": frames["train"][2],
        "truncated_val": frames["val"][2],
        "threshold": args.threshold,
        "n_train": frames["train"][3], "n_val": frames["val"][3],
        "leakage_excluded": meta_in["leakage_excluded"],
        "leakage_threshold": meta_in["leakage_threshold"],
        "train_seconds": train_seconds,
        "epochs": epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "early_stopping": early,
        "val_macro_f1": float(ev.get("eval_macro_f1", 0.0)),
        "class_weighting": False, "resampling": False,
        "arm": arm, "lang": lang, "tag": args.tag,
        "labels_excluded_from_output": excl,
        "n_dropped_excluded_label": meta_in.get("n_dropped_excluded_label"),
        "n_stripped_excluded_label": meta_in.get("n_stripped_excluded_label"),
        "support_train_plus_val": meta_in["support_train_plus_val"],
        "labels_below_min_support": meta_in["labels_below_min_support"],
        "note": ("severity is trained as plain 3-class; ordinality is not "
                 "exploited, unlike the LLM's ordinal-median vote"
                 if head == "severity" else ""),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")
    print(f"  wrote           {out_dir}")
    print(f"\n  Next: python -m scripts.check_encoder_reachability "
          f"--arm {arm} --seed {args.seed}")
    print(f"  Hops 3, 4 and 5 should turn ok for `{head}`.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
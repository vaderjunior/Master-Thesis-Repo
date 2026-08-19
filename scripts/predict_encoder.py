"""
scripts/predict_encoder.py - run every trained head over the eval subsets and
write ItemResult records. GPU, no API calls.

  python -m scripts.predict_encoder --arm full --seed 0 --dry-run
  python -m scripts.predict_encoder --arm full --seed 0
  python -m scripts.predict_encoder --arm de   --seed 0
  python -m scripts.predict_encoder --arm kb   --seed 0

ItemResult RECORDS, NOT A SEPARATE ENCODER FORMAT. score_all,
check_dimension_reach, make_comparability and the paired bootstrap then all
work unchanged, and the encoder-vs-LLM comparison runs through ONE scoring
implementation. run_slice1 carried its own copy of the scoring functions for
two phases and missed a fix; the rule is one implementation, always.

Fields that look like fudges and are not:
  parse_failures 0   true. An encoder cannot emit unparseable output.
  n_runs 1           true. Argmax and thresholding are deterministic; there is
                     no self-consistency vote to take.
  prompt_version None, kb_version None
                     true for Arm A. Arm B stamps the base KB hash, because
                     its training data IS kb/records.jsonl.

GATE CONSISTENCY IS APPLIED AND COUNTED, AND THAT IS A FINDING. Result's
validator clears target_group, hate_type and severity when hate is false. For
the LLM that resolves one model contradicting itself across sampled runs. Here
it resolves FIVE INDEPENDENTLY TRAINED MODELS contradicting each other, which
they have no mechanism to avoid - nothing in training tells the target_group
head what the gate head decided. gate_normalised measures how often that
happens, and it is a cost of the separate-model-per-head design that the LLM
does not pay. `legal` is exempt here as everywhere: criminal relevance is
independent of the hate gate.

WHAT A MISSING HEAD MEANS. Arm A has no `legal` head, Arm B has neither
`legal` nor anything German. A dimension with no head is written as the schema
default - [] for multilabel, None for severity - and its absence is recorded in
encoder_meta.heads. That is NOT the same as a head predicting empty, and
check_dimension_reach will correctly report the dimension as absent from those
runs, which is exactly the audit that caught legal_dev_peasec six weeks late.

RAW PROBABILITIES ARE STORED. run_predictions carries the per-label sigmoid or
softmax output, so the rate-matched threshold reading - the threshold at which
the encoder predicts as many positives per item as the LLM - costs no GPU
afterwards. Prediction cardinality is a measured mechanism in this thesis, not
a nuisance: retrieval tightens the prediction set, which helps where the model
over-predicts (hate_type, legal, the gate) and hurts where it under-predicts
(target_group). An encoder at a fixed 0.5 has its own cardinality and it must
be reportable without retraining.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA = Path("data/encoder")
MODELS = Path("models/encoder")
RESULTS = Path("experiments/results")
PROCESSED = Path("data/processed")

# arm -> eval subsets it is scored on. DEV ONLY. Test subsets are touched by
# final runs and nothing else; that rule predates this script and outranks it.
# TEST SUBSETS ARE INCLUDED, DELIBERATELY. Rule 2 says the test splits are
# touched by final runs and nothing else; the encoder baseline IS a final run,
# and adding them costs about 2.5 seconds because inference is a forward pass
# rather than 1,557 API calls. Without them the supervision-ceiling claim in
# baselines_log.md would be dev-only while the LLM's is confirmed on test.
#
# No new leakage exposure: check_leakage scanned all 15 eval subsets including
# test, and the >= 0.95 exclusion list that make_encoder_data applied covers
# every one of them. The 6 en_test_eval_types items with a TRAIN near-copy
# were already excluded from training.
ARM_SUBSETS = {
    "full": ["en_dev_eval_main", "en_dev_eval_targets",
             "en_dev_eval_types", "en_dev_eval_sq3_types",
             "en_dev_eval_severity",
             "en_test_eval", "en_test_eval_targets",
             "en_test_eval_types", "en_test_eval_severity"],
    "kb": ["en_dev_eval_main", "en_dev_eval_targets",
           "en_dev_eval_types", "en_dev_eval_sq3_types",
           "en_dev_eval_severity",
           "en_test_eval", "en_test_eval_targets",
           "en_test_eval_types", "en_test_eval_severity"],
    "de": ["de_dev_eval", "de_legal_dev_eval",
           "de_test_eval", "de_legal_test_eval"],
}
ARM_HEADS = {
    "full": ["hate", "target_group", "hate_type", "severity"],
    "kb": ["hate", "target_group", "hate_type", "severity"],
    "de": ["hate", "legal"],
}
MULTILABEL = {"target_group", "hate_type", "legal"}


def taxonomy_labels(dim: str) -> list:
    tax = yaml.safe_load(
        Path("config/taxonomy.yaml").read_text(encoding="utf-8"))
    d = tax["dimensions"][dim]
    if d["type"] == "binary":
        return ["false", "true"]
    labels = d["labels"]
    return list(labels) if isinstance(labels, dict) else list(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_HEADS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="predict with a tagged arm, e.g. --tag nogrievance. "
                         "Heads the tagged arm does not have are BORROWED "
                         "from the untagged arm at the same seed.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true",
                    help="first 8 items of the first subset, print the "
                         "assembled Result, write nothing")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, BertForSequenceClassification,
                              BertTokenizerFast)
    from src.hsrag.classify import ItemResult, code_version, text_hash
    from src.hsrag.prompt import taxonomy_version
    from src.hsrag.schema import Result

    arm, seed = args.arm, args.seed
    slug = f"{arm}_{args.tag}" if args.tag else arm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 78}\nENCODER PREDICT  arm={slug} seed={seed}  "
          f"device={dev}\n{'=' * 78}")

    # --------------------------------------------------------- load heads
    #
    # BORROWED HEADS ARE THE POINT, NOT A CONVENIENCE. A tagged arm has only
    # the head under test. Without borrowing, `hate` would default to False,
    # Result.gate_consistency would clear every hate_type prediction, and the
    # six-label arm would score 0.000 for a reason with nothing to do with the
    # label space.
    #
    # Borrowing from the SAME SEED of the untagged arm also makes the
    # comparison clean: the gate head is byte-identical across the two arms,
    # so the only thing that differs is the hate_type output space. That
    # matters concretely - the gate costs hate_type -0.092 on this subset, so
    # an unmatched gate would swamp the effect being measured.
    heads, missing, borrowed = {}, [], []
    for head in ARM_HEADS[arm]:
        d = MODELS / slug / head / f"seed{seed}"
        if not (d / "meta.json").exists() and args.tag:
            d = MODELS / arm / head / f"seed{seed}"
            if (d / "meta.json").exists():
                borrowed.append(head)
        if not (d / "meta.json").exists():
            missing.append(head)
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        # The taxonomy assertion again, here rather than only in the guard: a
        # checkpoint's MultiLabelBinarizer alignment is valid for exactly one
        # taxonomy version, and predicting under a different one would produce
        # a full, plausible, silently mislabelled results file.
        now = taxonomy_version()
        if meta["taxonomy_version"] != now:
            raise SystemExit(
                f"{head}: checkpoint trained under "
                f"{meta['taxonomy_version']}, taxonomy is now {now}. Refusing "
                f"to predict - the label alignment is stale.")
        # A TAGGED HEAD HAS A DELIBERATELY RESTRICTED OUTPUT SPACE. 7.10's
        # six-label hate_type head is missing `grievance` on purpose, and
        # meta.labels_excluded_from_output records which labels and why. The
        # comparison stays by ORDER, never by set: a set comparison passes on a
        # permutation and every per-label number would then be silently wrong,
        # which is the blindness that let a Phase 8 retrieval check miss that
        # RRF had reordered the same records.
        excl = meta.get("labels_excluded_from_output") or []
        want = [l for l in taxonomy_labels(head) if l not in excl]
        if meta["labels"] != want:
            raise SystemExit(
                f"{head}: label ORDER differs from taxonomy.yaml minus "
                f"{excl}. checkpoint {meta['labels']}, expected {want}")
        if meta["lang"] == "de":
            tok = BertTokenizerFast.from_pretrained(str(d))
            mdl = BertForSequenceClassification.from_pretrained(str(d))
        else:
            tok = AutoTokenizer.from_pretrained(str(d))
            mdl = AutoModelForSequenceClassification.from_pretrained(str(d))
        heads[head] = {"tok": tok, "model": mdl.to(dev).eval(), "meta": meta,
                       "labels": meta["labels"]}
        tagmark = ("  <- BORROWED from untagged" if head in borrowed
                   else (f"  [{len(meta['labels'])} labels"
                         + (f", excl {meta['labels_excluded_from_output']}"
                            if meta.get("labels_excluded_from_output")
                            else "") + "]"))
        print(f"  {head:14} {meta['base_model']:22} "
              f"n_train {meta['n_train']:>6}  thr {meta['threshold']}  "
              f"val_macro {meta['val_macro_f1']:.4f}{tagmark}")
    if not heads:
        raise SystemExit("no trained heads for this arm and seed")
    if missing:
        print(f"  MISSING HEADS: {missing} - written as schema defaults and "
              f"recorded in encoder_meta.heads")

    # Arm B's training data IS the knowledge base, so it carries the base KB
    # hash. Arm A trains on the frozen splits and consults no KB, so None -
    # the same rule classify() applies to zero_shot.
    kb_version = None
    if arm == "kb":
        from src.hsrag.kb import load_collection
        cfg = yaml.safe_load(
            Path("config/config.yaml").read_text(encoding="utf-8"))
        col = load_collection(Path(cfg["kb"]["chroma_path"]),
                              cfg["kb"]["embedding_model"])
        kb_version = col.metadata.get("kb_version")
        print(f"  kb_version     {kb_version}  (Arm B trains on the KB)")

    @torch.no_grad()
    def infer(head: str, texts: list) -> np.ndarray:
        h = heads[head]
        out = []
        for i in range(0, len(texts), args.batch_size):
            enc = h["tok"](texts[i:i + args.batch_size], truncation=True,
                           max_length=h["meta"]["max_seq_length"],
                           padding=True, return_tensors="pt").to(dev)
            logit = h["model"](**enc).logits
            p = (torch.sigmoid(logit) if head in MULTILABEL
                 else torch.softmax(logit, dim=-1))
            out.append(p.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    out_path = RESULTS / f"encoder_{slug}_seed{seed}_live.jsonl"
    written, t0 = 0, time.time()
    lines = []

    for subset in ARM_SUBSETS[arm]:
        p = PROCESSED / f"{subset}.parquet"
        if not p.exists():
            print(f"\n  {subset}: MISSING, skipped")
            continue
        df = pd.read_parquet(p)
        if args.dry_run:
            df = df.head(8)
        texts = [str(t) for t in df["text"]]
        ids = [str(i) for i in df["id"]]
        langs = [str(l) for l in df["lang"]]

        t_sub = time.time()
        probs = {h: infer(h, texts) for h in heads}
        infer_s = round(time.time() - t_sub, 2)
        print(f"\n  {subset:24} {len(df):>5} items  {infer_s:>6.2f}s")

        for i, item_id in enumerate(ids):
            fields, raw, agree = {}, {}, {}
            for head, h in heads.items():
                labels, thr = h["labels"], h["meta"]["threshold"]
                pv = probs[head][i]
                raw[head] = {l: round(float(x), 4) for l, x in zip(labels, pv)}
                if head == "hate":
                    fields["hate"] = bool(int(pv.argmax()))
                    agree["hate"] = float(pv.max())
                elif head == "severity":
                    fields["severity"] = labels[int(pv.argmax())]
                    agree["severity"] = float(pv.max())
                else:
                    fields[head] = [l for l, x in zip(labels, pv) if x >= thr]
                    # Confidence on a multilabel head is the margin from the
                    # threshold, not a probability: the mean distance of each
                    # label's decision from thr. Reported for symmetry with
                    # the LLM's vote share and NOT comparable to it - the LLM
                    # figure is agreement among sampled runs, this is one
                    # deterministic forward pass.
                    agree[head] = float(np.mean(np.abs(pv - thr)))
            # A head this arm does not have leaves the schema default: [] for
            # multilabel, None for severity, False for the gate. Recorded in
            # encoder_meta.heads so it is never confused with a prediction.
            res = Result(reasoning=f"encoder {arm} seed{seed}",
                         hate=fields.get("hate", False),
                         target_group=fields.get("target_group", []),
                         hate_type=fields.get("hate_type", []),
                         severity=fields.get("severity"),
                         legal=fields.get("legal", []))
            r = ItemResult(
                item_id=item_id, text_hash=text_hash(texts[i]),
                lang=langs[i], arm=f"encoder_{slug}",
                timestamp=time.time(), workers=1, temperature=None,
                active_model="+".join(
                    f"{h}:{heads[h]['meta']['base_model']}" for h in heads),
                prompt_version=None,
                taxonomy_version=taxonomy_version(),
                code_version=code_version(),
                kb_version=kb_version,
                retrieval_config=None, retrieved={},
                result=res.model_dump(), uncertain=False,
                n_valid=1, n_runs=1, agreement=agree,
                parse_failures=0, repairs=0, normalisations=0,
                gate_normalised=int(res.gate_normalised),
                encoder_meta={
                    "arm": slug, "seed": seed, "subset": subset,
                    "tag": args.tag, "heads_borrowed": borrowed,
                    "labels_excluded": {
                        h: heads[h]["meta"].get(
                            "labels_excluded_from_output") or []
                        for h in heads},
                    "heads": {h: {
                        "base_model": heads[h]["meta"]["base_model"],
                        "threshold": heads[h]["meta"]["threshold"],
                        "n_train": heads[h]["meta"]["n_train"],
                        "train_seconds": heads[h]["meta"]["train_seconds"],
                        "train_script_sha": heads[h]["meta"].get(
                            "train_script_sha"),
                    } for h in heads},
                    "heads_missing": missing,
                    "infer_seconds_subset": infer_s,
                },
                raw_runs=[], run_predictions=[raw], latencies=[],
            )
            lines.append(r.to_json())
            written += 1

        if args.dry_run:
            print(f"\n  first assembled Result:\n    "
                  + json.dumps(json.loads(lines[0])["result"],
                               ensure_ascii=False)[:400])
            print(f"    raw: "
                  + json.dumps(json.loads(lines[0])["run_predictions"][0],
                               ensure_ascii=False)[:400])
            break

    total = round(time.time() - t0, 1)
    print(f"\n{'-' * 78}")
    n_norm = sum(json.loads(l)["gate_normalised"] for l in lines)
    print(f"  gate_normalised on {n_norm}/{written} items "
          f"({n_norm / max(written, 1):.1%}) - independently trained heads "
          f"contradicting each other")
    if args.dry_run:
        print(f"\n  DRY RUN - nothing written.\n{'=' * 78}\n")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_path}: {written} records in {total}s")
    print(f"\n  Next: python -m scripts.check_encoder_reachability "
          f"--arm {arm} --seed {seed}")
    print(f"  Hops 6, 7 and 8 should turn ok. Then --break-head to watch the "
          f"guard fail for the first time.\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
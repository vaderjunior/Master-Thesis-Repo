"""
scripts/check_hatexplain_fp.py - are the false positives 'offensive' posts?

HateXplain annotates three classes: hatespeech, offensive, normal. Phase 2
mapped offensive -> gate=False on a strict reading, keeping the raw votes in
Record.raw so the decision stays reversible. Slice 1 shows a 61% false
positive rate on HateXplain benign items against 10% on Implicit Hate, so the
question is whether those errors are concentrated in the offensive class.

If they are, the model is not over-flagging: it is disagreeing with one
annotation-policy decision, and that is a finding about the label definition
rather than about the system.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

df = pd.read_parquet("data/processed/en_dev_eval_main.parquet").head(150)
hx = df[df["source"] == "hatexplain"]

# inspect the raw structure once before relying on it
print("raw keys:", list(json.loads(hx.iloc[0]["raw"]).keys()))
print("sample:", json.dumps(json.loads(hx.iloc[0]["raw"]), ensure_ascii=False)[:400])


def hx_class(raw_json: str) -> str:
    """Majority annotator label. Adapt the key name if the print above differs."""
    raw = json.loads(raw_json)
    votes = (raw.get("annotator_labels") or raw.get("labels")
             or raw.get("votes") or [])
    if isinstance(votes, str):
        votes = [votes]
    if not votes:
        return "unknown"
    return Counter(votes).most_common(1)[0][0]


hx = hx.assign(hx_class=[hx_class(r) for r in hx["raw"]])
print("\nclass distribution:", dict(Counter(hx["hx_class"])))

rows = [json.loads(l) for l in
        Path("experiments/results/slice1_en_dev_eval_main_live.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
by_arm = defaultdict(dict)
for r in rows:
    by_arm[r["arm"]][r["item_id"]] = r

print(f"\n{'class':12} {'n':>4} " + " ".join(f"{a:>12}" for a in
                                             ("zero_shot", "few_shot", "rag")))
for cls in sorted(set(hx["hx_class"])):
    sub = hx[hx["hx_class"] == cls]
    cells = []
    for arm in ("zero_shot", "few_shot", "rag"):
        pred = [by_arm[arm][str(i)]["result"] for i in sub["id"]
                if str(i) in by_arm[arm] and by_arm[arm][str(i)]["result"]]
        n_hate = sum(1 for p in pred if p["hate"])
        cells.append(f"{n_hate:3}/{len(pred):3} ({n_hate / max(len(pred), 1):3.0%})")
    print(f"{cls:12} {len(sub):4} " + " ".join(f"{c:>12}" for c in cells))

print("\n(rows are gold classes; cells are how often the system said hate)")
print("gate=False in our mapping covers BOTH 'offensive' and 'normal'")
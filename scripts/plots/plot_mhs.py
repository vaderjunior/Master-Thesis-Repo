"""
Aggregate MHS to comment level, plot the score distribution, and compute
the severity cut-points. Run this ONCE, look at the plot, then freeze the
numbers into data/mappings/mhs_thresholds.md.

The loader reads the FROZEN numbers, not these. If the loader recomputed
terciles every run, the severity bands would silently move whenever the
data changed, and nothing would be reproducible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

FIG_DIR = Path("experiments/results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

GATE_THRESHOLD = 0.5  # the dataset authors' own guidance

ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
df = ds.to_pandas()

print(f"Annotator-level rows : {len(df)}")
print(f"Unique comments      : {df['comment_id'].nunique()}")

# --- aggregate to comment level ---
# score: mean across annotators
# text: first (identical across rows for a comment_id)
# targets: any-vote (if ANY annotator saw the target, count it)
target_cols = [
    "target_race",
    "target_religion",
    "target_origin",
    "target_gender",
    "target_sexuality",
    "target_disability",
]

agg = df.groupby("comment_id").agg(
    text=("text", "first"),
    score=("hate_speech_score", "mean"),
    n_annotators=("annotator_id", "count"),
    **{c: (c, "any") for c in target_cols},
)

print(f"Aggregated comments  : {len(agg)}")
print(f"\nAnnotators per comment:\n{agg['n_annotators'].value_counts().sort_index()}")

print(f"\nScore distribution:")
print(agg["score"].describe())

# --- gate ---
hateful = agg[agg["score"] > GATE_THRESHOLD]
print(f"\nGate (score > {GATE_THRESHOLD}):")
print(f"  hateful     : {len(hateful):6}  ({len(hateful)/len(agg)*100:.1f}%)")
print(f"  not hateful : {len(agg)-len(hateful):6}")

# --- severity terciles, WITHIN the hateful mass only ---
t1, t2 = np.percentile(hateful["score"], [33.333, 66.667])
print(f"\nSeverity cut-points (terciles of hateful mass):")
print(f"  low    : {GATE_THRESHOLD} < score <= {t1:.4f}")
print(f"  medium : {t1:.4f} < score <= {t2:.4f}")
print(f"  high   : score > {t2:.4f}")

for name, lo, hi in [
    ("low", GATE_THRESHOLD, t1),
    ("medium", t1, t2),
    ("high", t2, float("inf")),
]:
    n = ((hateful["score"] > lo) & (hateful["score"] <= hi)).sum()
    print(f"  {name:7} {n:6}  ({n/len(hateful)*100:.1f}% of hateful mass)")

# --- plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.hist(agg["score"], bins=80, color="steelblue")
ax1.axvline(GATE_THRESHOLD, color="red", ls="--", label=f"gate = {GATE_THRESHOLD}")
ax1.set_title("All comments")
ax1.set_xlabel("hate_speech_score (mean over annotators)")
ax1.legend()

ax2.hist(hateful["score"], bins=60, color="indianred")
ax2.axvline(t1, color="black", ls="--", label=f"low|medium = {t1:.2f}")
ax2.axvline(t2, color="black", ls=":", label=f"medium|high = {t2:.2f}")
ax2.set_title("Hateful mass only (score > 0.5)")
ax2.set_xlabel("hate_speech_score")
ax2.legend()

plt.tight_layout()
out = FIG_DIR / "mhs_score_hist.png"
plt.savefig(out, dpi=150)
print(f"\nSaved {out}")

# --- bands---
print("\n--- 5 samples per severity band ---")
print("(check these against your taxonomy.yaml definitions:")
print(" low=insult, medium=dehumanisation, high=incitement/threat)")
for name, lo, hi in [
    ("LOW", GATE_THRESHOLD, t1),
    ("MEDIUM", t1, t2),
    ("HIGH", t2, float("inf")),
]:
    band = hateful[(hateful["score"] > lo) & (hateful["score"] <= hi)]
    print(f"\n{name} ({lo:.2f} .. {hi:.2f}):")
    for text in band["text"].sample(5, random_state=42):
        print(f"  - {text[:100]}")
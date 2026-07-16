import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True,
                    choices=["hatexplain", "mhs", "implicit_hate", "gahd", "detox"],)
args = parser.parse_args()

if args.dataset == "hatexplain":
    from src.hsrag.data.hatexplain import load_hatexplain
    records, stats = load_hatexplain()
elif args.dataset == "mhs":
    from src.hsrag.data.mhs import load_mhs
    records, stats = load_mhs()
elif args.dataset == "implicit_hate":
    from src.hsrag.data.implicit_hate import load_implicit_hate
    records, stats = load_implicit_hate()
elif args.dataset == "detox":
    from src.hsrag.data.detox import load_detox
    records, stats = load_detox()
elif args.dataset == "gahd":
    from src.hsrag.data.gahd import load_gahd
    records, stats = load_gahd()
else:
    raise NotImplementedError(f"{args.dataset} loader not written yet")

print(f"\n=== {args.dataset} ===")
print(f"Total records kept : {len(records)}")
print(f"Dropped (3-way tie): {stats['dropped_tie']}")

label_stats = {k: v for k, v in stats.items() if k.startswith("label_")}
if label_stats:
    print(f"\nSource label distribution (post-level, after aggregation):")
    for label, n in sorted(label_stats.items(), key=lambda x: -x[1]):
        print(f"  {label.removeprefix('label_'):12} {n:6}")

gate_counts = Counter(r.gate for r in records)
print(f"\nGate distribution:")
for gate, n in gate_counts.most_common():
    print(f"  {str(gate):6} {n:6}  ({n/len(records)*100:.1f}%)")

target_counts = Counter(
    t for r in records if r.target_groups for t in r.target_groups
)
print(f"\nTarget histogram (gate=True records only):")
for target, n in target_counts.most_common():
    print(f"  {target:20} {n:6}")

type_counts = Counter(
    t for r in records if r.hate_types for t in r.hate_types
)
if type_counts:
    print(f"\nHate-type histogram (gate=True records):")
    for t, n in type_counts.most_common():
        print(f"  {t:20} {n:6}")

sev_stats = {k: v for k, v in stats.items() if k.startswith("severity_")}
if sev_stats:
    print(f"\nSeverity distribution (gate=True records only):")
    for label in ["low", "medium", "high"]:
        n = sev_stats.get(f"severity_{label}", 0)
        print(f"  {label:8} {n:6}")
        
print(f"\nSample records:")
for r in records[:2]:
    print(f"\n  id      : {r.id}")
    print(f"  text    : {r.text[:80]}")
    print(f"  gate    : {r.gate}")
    print(f"  targets : {r.target_groups}")
    print(f"  raw     : {r.raw}")
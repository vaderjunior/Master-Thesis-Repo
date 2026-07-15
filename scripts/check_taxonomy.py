import sys
from pathlib import Path

import yaml

taxonomy = yaml.safe_load(
    Path("config/taxonomy.yaml").read_text(encoding="utf-8")
)

print(f"Taxonomy version: {taxonomy['version']}\n")

problems = []

for dim_name, dim in taxonomy["dimensions"].items():
    dim_type = dim["type"]
    print(f"{dim_name}  [{dim_type}]")

    if dim_type == "binary":
        # binary dimensions carry a single top-level definition
        definition = dim.get("definition", "").strip()
        if not definition:
            problems.append(f"{dim_name}: missing definition")
        print(f"    {definition[:80]}...")

    elif dim_type == "ordinal":
        # ordinal dimensions carry a flat list of labels
        labels = dim.get("labels", [])
        if not labels:
            problems.append(f"{dim_name}: no labels")
        print(f"    labels: {labels}")

    elif dim_type == "multilabel":
        labels = dim.get("labels", {})
        if not labels:
            problems.append(f"{dim_name}: no labels")
        for label_name, label in labels.items():
            definition = (label or {}).get("definition", "").strip()
            if not definition:
                problems.append(f"{dim_name}.{label_name}: missing definition")
            print(f"    {label_name:20} {definition[:60]}...")

    else:
        problems.append(f"{dim_name}: unknown type '{dim_type}'")

    print()

print(f"{len(taxonomy['dimensions'])} dimensions")

if problems:
    print("\nPROBLEMS:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)

print("All labels have definitions.")
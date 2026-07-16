"""
Coverage report across all datasets.

Loads every dataset and reports, per dataset:
  - total records, gate distribution, language
  - which dimensions it actually populates (non-None)
And builds the dimensions x datasets matrix that goes into the thesis
(Methodology, data section).

No new decisions here - this just summarises what the loaders produced.
"""

from collections import Counter

from src.hsrag.data.hatexplain import load_hatexplain
from src.hsrag.data.mhs import load_mhs
from src.hsrag.data.implicit_hate import load_implicit_hate
from src.hsrag.data.gahd import load_gahd
from src.hsrag.data.detox import load_detox

LOADERS = {
    "hatexplain": load_hatexplain,
    "mhs": load_mhs,
    "implicit_hate": load_implicit_hate,
    "gahd": load_gahd,
    "detox": load_detox,
}

DIMENSIONS = ["gate", "target_group", "hate_type", "severity"]


def dimension_populated(records, dim: str) -> bool:
    """True if ANY record has a non-None value for this dimension.
    (None = dataset never annotates it; [] or a value = it does.)"""
    attr = {
        "gate": "gate",
        "target_group": "target_groups",
        "hate_type": "hate_types",
        "severity": "severity",
    }[dim]
    return any(getattr(r, attr) is not None for r in records)


def main():
    all_records = {}

    print("Loading all datasets (this takes a minute)...\n")
    for name, loader in LOADERS.items():
        records, _ = loader()
        all_records[name] = records
        print(f"  {name:15} {len(records):6} records")

    # --- per-dataset summary ---
    print("\n\n=== PER-DATASET SUMMARY ===")
    for name, records in all_records.items():
        lang = records[0].lang
        gate_true = sum(1 for r in records if r.gate is True)
        gate_false = sum(1 for r in records if r.gate is False)
        print(f"\n{name}  (lang={lang}, {len(records)} records)")
        print(f"  gate: {gate_true} hateful / {gate_false} not "
              f"({gate_true/len(records)*100:.1f}% hateful)")
        dims = [d for d in DIMENSIONS if dimension_populated(records, d)]
        print(f"  populates: {', '.join(dims)}")

    # --- dimensions x datasets matrix ---
    print("\n\n=== COVERAGE MATRIX (dimension x dataset) ===\n")
    header = f"{'dimension':16}" + "".join(f"{n[:10]:12}" for n in LOADERS)
    print(header)
    print("-" * len(header))
    for dim in DIMENSIONS:
        row = f"{dim:16}"
        for name in LOADERS:
            mark = "yes" if dimension_populated(all_records[name], dim) else "-"
            row += f"{mark:12}"
        print(row)

    # --- language coverage per dimension ---
    print("\n\n=== LANGUAGE COVERAGE PER DIMENSION ===\n")
    for dim in DIMENSIONS:
        langs = set()
        for name, records in all_records.items():
            if dimension_populated(records, dim):
                langs.add(records[0].lang)
        print(f"  {dim:16} {sorted(langs)}")

    # --- checkpoint assertions ---
    print("\n\n=== CHECKPOINT ===")
    ok = True
    for dim in DIMENSIONS:
        en_sources = [
            n for n, r in all_records.items()
            if r[0].lang == "en" and dimension_populated(r, dim)
        ]
        if not en_sources:
            print(f"  FAIL: {dim} has no EN source")
            ok = False
        else:
            print(f"  OK  : {dim} EN source(s): {en_sources}")

    de_gate = [
        n for n, r in all_records.items()
        if r[0].lang == "de" and dimension_populated(r, "gate")
    ]
    if de_gate:
        print(f"  OK  : gate has DE source(s): {de_gate}")
    else:
        print(f"  FAIL: gate has no DE source")
        ok = False

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
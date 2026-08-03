"""
scripts/revote.py - recompute the vote from stored raw runs. NO API CALLS.

WHY THIS EXISTS: ItemResult.result is written at run time, so a bug in
aggregate() is baked into the results file and --score-only cannot see past
it. aggregate() was written before the legal dimension existed and built
Result with four fields, so legal defaulted to [] and 974 of 1350 model
predictions were silently discarded. Every raw run is stored (that is why
they are stored), so the vote can be recomputed rather than the 1350 calls
repeated.

Writes alongside the original with a .revoted.jsonl suffix; the original is
never modified, so the old and new votes remain comparable.

  python -m scripts.revote experiments/results/legal_dev_peasec_live.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.hsrag.parse import RunResult
from src.hsrag.schema import Result
from src.hsrag.vote import aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.path)
    out = Path(args.out) if args.out else src.with_suffix(".revoted.jsonl")

    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = 0
    written = []

    for line in lines:
        r = json.loads(line)
        if "_manifest" in r:
            written.append(line)
            continue

        # Rebuild RunResult objects from the stored per-run predictions. A
        # None prediction is a parse failure and must stay out of the pool,
        # exactly as it did at run time.
        runs = [RunResult(result=Result(**p) if p else None)
                for p in r["run_predictions"]]
        v = aggregate(runs)

        before = r["result"]
        after = v.result.model_dump() if v.result is not None else None
        if before != after:
            changed += 1
        r["result"] = after
        r["uncertain"] = v.uncertain
        r["n_valid"] = v.n_valid
        r["agreement"] = v.agreement
        written.append(json.dumps(r, ensure_ascii=False))

    out.write_text("\n".join(written) + "\n", encoding="utf-8")
    print(f"{len(written) - 1} results revoted, {changed} changed")
    print(f"wrote {out}")

    res = [json.loads(l) for l in written if "_manifest" not in l]
    for field in ("target_group", "hate_type", "legal"):
        c = Counter(len(r["result"][field]) if r["result"] else None
                    for r in res)
        print(f"  {field:14} labels per item: {dict(sorted(c.items(), key=str))}")


if __name__ == "__main__":
    main()
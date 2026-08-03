"""
scripts/check_concurrency.py - does parallelism actually help on this provider?

TUDaGPT capped total throughput: 4 workers bought only ~1.2x, and 8 workers
made it return 422 on roughly half of all requests. PEASEC claims continuous
batching over Nvidia MPS, which would be a genuine speedup rather than a
shared budget being divided.

EACH CALL SENDS A DIFFERENT PROMPT. An earlier version sent identical prompts
and measured server-side caching (six of eight calls returned in under a
second), which is not parallelism.

  python -m scripts.check_concurrency --provider peasec
  python -m scripts.check_concurrency --provider peasec --workers 1 4 8 16
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src.hsrag.client import make_client

# Bounded-output prompts. An earlier version asked open-ended essay questions,
# and calls that ran to max_tokens produced 80-185 s outliers that dominated
# wall-clock at low worker counts. The real workload asks for a small JSON
# object, so the probe must too.
PROMPTS = [f'Return ONLY this JSON, no prose: {{"n": {n}, "even": <true if '
           f'{n} is even else false>}}' for n in range(101, 301)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--calls", type=int, default=8,
                    help="calls per worker-count setting")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    api = cfg["api"]
    provider = args.provider or cfg["classify"]["provider"]
    p = api[provider]
    model = args.model or p["default_model"]

    def one(i):
        # Own client per worker: active_model is mutable state and a shared
        # client would race on it.
        c = make_client(provider, [model],
                        temperature=api["temperature"],
                        timeout=api["timeout_seconds"],
                        max_tokens=api.get("max_tokens"),
                        allow_fallback=False,
                        url_env=p["url_env"], token_env=p["token_env"])
        t0 = time.time()
        try:
            out = c.complete([{"role": "user", "text": PROMPTS[i % len(PROMPTS)]}])
            return time.time() - t0, len(out), None
        except Exception as e:
            return time.time() - t0, 0, f"{type(e).__name__}: {str(e)[:100]}"

    print(f"provider {provider}, model {model}, {args.calls} calls per setting\n")
    print(f"{'workers':>8} {'wall s':>8} {'s/call':>8} {'speedup':>8} "
          f"{'p50 lat':>8} {'p95 lat':>8} {'errors':>7}")

    base = None
    offset = 0
    for w in args.workers:
        idx = list(range(offset, offset + args.calls))
        offset += args.calls          # never reuse a prompt across settings

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=w) as ex:
            res = list(ex.map(one, idx))
        wall = time.time() - t0

        lat = sorted(d for d, _, e in res if e is None)
        errs = [e for _, _, e in res if e]
        per_call = wall / args.calls
        base = base or per_call
        p50 = statistics.median(lat) if lat else 0
        p95 = lat[int(len(lat) * 0.95)] if lat else 0

        print(f"{w:>8} {wall:8.1f} {per_call:8.1f} {base / per_call:7.1f}x "
              f"{p50:8.1f} {p95:8.1f} {len(errs):7}")
        for e in errs[:2]:
            print(f"           {e}")

    print("\n  s/call is WALL-CLOCK throughput, which is what run estimates "
          "need.\n  Rising per-call latency with flat throughput means the "
          "server is\n  dividing a fixed budget rather than truly "
          "parallelising.")


if __name__ == "__main__":
    main()
"""
scripts/check_concurrency.py - does parallelism actually help?

Slice 1 latency is 29.8s mean but 16.8s median, and output length does not
predict it. That signature says queue wait rather than compute, and queue wait
parallelises. But this is shared university infrastructure: it may serialise
requests anyway, or rate-limit. 16 calls tells us for the price of 8 minutes.

  python -m scripts.check_concurrency
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src.hsrag.client import TUDaGPTClient

PROMPT = [{"role": "system", "text": "Reply with a JSON object: "
                                     '{"ok": true, "n": <the number below>}'},
          {"role": "user", "text": "7"}]


def one(i, cfg, api):
    """Each worker gets its own client: active_model is mutable state and a
    shared client would race on it."""
    c = TUDaGPTClient(models=[cfg["pinned_model"]], temperature=api["temperature"],
                      timeout=api["timeout_seconds"], allow_fallback=False)
    t0 = time.time()
    try:
        c.complete(PROMPT)
        return time.time() - t0, None
    except Exception as e:
        return time.time() - t0, f"{type(e).__name__}: {str(e)[:80]}"


def main():
    cfg_all = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    cfg, api = cfg_all["classify"], cfg_all["api"]

    print("sequential x4...")
    t0 = time.time()
    seq = [one(i, cfg, api) for i in range(4)]
    seq_wall = time.time() - t0
    print(f"  wall {seq_wall:5.1f}s   per-call {[f'{d:.1f}' for d, _ in seq]}")
    for _, err in seq:
        if err:
            print(f"  ERROR {err}")

    for workers in (4, 8):
        print(f"\nparallel x{workers}...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            par = list(ex.map(lambda i: one(i, cfg, api), range(workers)))
        wall = time.time() - t0
        errs = [e for _, e in par if e]
        print(f"  wall {wall:5.1f}s   per-call {[f'{d:.1f}' for d, _ in par]}")
        print(f"  errors {len(errs)}/{workers}")
        for e in errs[:3]:
            print(f"    {e}")
        eff = (seq_wall / 4 * workers) / wall if wall else 0
        print(f"  effective speedup vs sequential: {eff:.1f}x")


if __name__ == "__main__":
    main()
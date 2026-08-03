"""
scripts/check_client.py - verify a provider works.

  python -m scripts.check_client --provider peasec --list-models
  python -m scripts.check_client --provider peasec --no-fallback
  python -m scripts.check_client --provider peasec --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import time
from pathlib import Path

import yaml

from src.hsrag.client import make_client

parser = argparse.ArgumentParser()
parser.add_argument("--provider", default=None,
                    help="tudagpt | peasec | mock; default from config")
parser.add_argument("--tier", default="medium",
                    choices=["strong", "medium", "fast"])
parser.add_argument("--model", default=None, help="override the tier list")
parser.add_argument("--no-fallback", action="store_true",
                    help="fail instead of trying the next model")
parser.add_argument("--list-models", action="store_true",
                    help="ask the server what it actually serves")
args = parser.parse_args()

cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
api = cfg["api"]
provider = args.provider or cfg["classify"]["provider"]
pcfg = api[provider]

models = [args.model] if args.model else pcfg[f"models_{args.tier}"]

client = make_client(
    provider, models,
    temperature=api["temperature"],
    timeout=api["timeout_seconds"],
    max_tokens=api.get("max_tokens"),
    allow_fallback=not args.no_fallback,
    url_env=pcfg["url_env"], token_env=pcfg["token_env"],
)

if args.list_models:
    if not hasattr(client, "list_models"):
        print(f"{provider} has no model-listing endpoint")
    else:
        print(f"models served by {provider}:")
        for m in client.list_models():
            print(f"  {m}")
    raise SystemExit(0)

print(f"provider {provider}, trying {models}")
t0 = time.time()
out = client.complete([
    {"role": "system", "text": "Reply with the single word: working"},
    {"role": "user", "text": "hello"},
])
print(f"Model said: {out.strip()!r}")
print(f"Answered by: {client.active_model}   ({time.time() - t0:.1f}s)")

# JSON discipline is the thing that actually matters for this pipeline, so a
# smoke test that only proves connectivity is not enough.
t0 = time.time()
out = client.complete([
    {"role": "system", "text": 'Return ONLY a JSON object, no prose, no code '
                               'fences: {"ok": true, "n": <the number below>}'},
    {"role": "user", "text": "7"},
])
print(f"\nJSON test ({time.time() - t0:.1f}s):\n{out}")
print("\n  raw repr:", repr(out[:200]))
print("  contains <think>:", "<think>" in out.lower())
print("  contains fences:", "```" in out)
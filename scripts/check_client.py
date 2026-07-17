import argparse
import yaml
from pathlib import Path
from src.hsrag.client import TUDaGPTClient

parser = argparse.ArgumentParser()
parser.add_argument("--tier", default="strong", choices=["strong", "medium", "fast"])
parser.add_argument("--no-fallback", action="store_true",
                    help="fail instead of trying the next model")
args = parser.parse_args()

config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))

client = TUDaGPTClient(
    models=config["api"][f"models_{args.tier}"],
    temperature=config["api"]["temperature"],
    timeout=config["api"]["timeout_seconds"],
    allow_fallback=not args.no_fallback,
)

out = client.complete([
    {"role": "system", "text": "Reply with the single word: working"},
    {"role": "user", "text": "hello"},
])

print("Model said:", out)
print("Answered by:", client.active_model)
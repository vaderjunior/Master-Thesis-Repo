import yaml
from pathlib import Path
from src.hsrag.client import TUDaGPTClient

config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
client = TUDaGPTClient(model=config["api"]["model_medium"],
                         temperature=config["api"]["temperature"])

out = client.complete([
  {"role": "system", "text": "Reply with the single word: working"},
  {"role": "user", "text": "hello"},
])
print("Model said:", out)
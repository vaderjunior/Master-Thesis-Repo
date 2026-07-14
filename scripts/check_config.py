
import yaml
from pathlib import Path

path = Path("config/config.yaml")

config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
print("Strong model:", config["api"]["model_strong"])
print("Temperature:", config["api"]["temperature"])
print("Config loaded OK")
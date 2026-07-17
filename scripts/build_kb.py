"""Build the Chroma KB index from kb/records.jsonl."""

from pathlib import Path

import yaml

from src.hsrag.kb import build

config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
kb_cfg = config["kb"]

build(
    records_path=Path(kb_cfg["records_path"]),
    chroma_path=Path(kb_cfg["chroma_path"]),
    model_name=kb_cfg["embedding_model"],
)
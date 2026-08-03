"""
scripts/build_kb_alt.py - build a Chroma index from an ALTERNATIVE records
file, into an alternative path. The primary index is never touched.

WHY: the German adaptability measurement compared kb 475869f9e2422969 against
kb c34e780cc973cfde, but the PROMPT also changed between those runs
(classify_v1-c74cb7ab -> classify_v1-8e588db7, adding the legal dimension to
the schema). So the +0.019 is "KB edit plus prompt edit", not a clean
one-variable measurement. Rebuilding the old KB lets the old records be
queried under the CURRENT prompt, changing exactly one thing.

  python -m scripts.build_kb_alt kb/records_kbv1.jsonl kb/chroma_kbv1
"""

import sys
from pathlib import Path

import yaml

from src.hsrag.kb import build

records_path = Path(sys.argv[1])
chroma_path = Path(sys.argv[2])

cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
build(records_path=records_path,
      chroma_path=chroma_path,
      model_name=cfg["kb"]["embedding_model"])
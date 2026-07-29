"""
scripts/check_prompt.py - render a full prompt and print it. ZERO API CALLS.

Closes the coverage gap: MockClient covers the response side of the pipeline,
nothing covered the prompt side, and the prompt is where every Phase 4
retrieval finding actually takes effect.

  python -m scripts.check_prompt "he called me the n-word and I was shocked"
  python -m scripts.check_prompt "Ausländer raus" --lang de --stats
  python -m scripts.check_prompt "text" --arm zero_shot
"""

import argparse
from pathlib import Path

import yaml

from src.hsrag.prompt import PromptContext, build_prompt, prompt_version
from src.hsrag.retrieve import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--arm", default="rag", choices=["rag", "zero_shot"])
    ap.add_argument("--strategy", default=None, help="override config strategy")
    ap.add_argument("--stats", action="store_true",
                    help="char counts per section (prompt length before the "
                         "API tells you the hard way)")
    args = ap.parse_args()

    cfg_all = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))

    if args.arm == "zero_shot":
        ctx = PromptContext.zero_shot()
    else:
        cfg = dict(cfg_all["retrieval"])
        if args.strategy:
            cfg["strategy"] = args.strategy
        kb = cfg_all["kb"]
        r = Retriever(chroma_path=Path(kb["chroma_path"]),
                      records_path=Path(kb["records_path"]),
                      model_name=kb["embedding_model"], cfg=cfg)
        ctx = PromptContext.from_retrieval(r.retrieve(args.text, args.lang))

    msgs = build_prompt(args.text, args.lang, ctx)

    print(f"\nprompt_version: {prompt_version()}   arm: {args.arm}\n")
    for m in msgs:
        print("=" * 72)
        print(f"[{m['role'].upper()}]")
        print("=" * 72)
        print(m["text"])
        print()

    if args.stats:
        total = sum(len(m["text"]) for m in msgs)
        print("=" * 72)
        for m in msgs:
            print(f"  {m['role']:8} {len(m['text']):6} chars")
        print(f"  {'TOTAL':8} {total:6} chars  (~{total // 4} tokens)")
        print(f"  definitions {len(ctx.definitions)}, "
              f"guidelines {len(ctx.guidelines)}, "
              f"example groups {[(n, len(h)) for n, h in ctx.example_groups]}")


if __name__ == "__main__":
    main()
"""
scripts/check_config.py - validate config/config.yaml. Zero API calls.
Run: python -m scripts.check_config

WHY IT WAS REWRITTEN. The previous version read config["api"]["model_strong"],
a key that stopped existing when the api block was restructured per provider.
It had been raising KeyError ever since and nobody noticed, because a check
script that is never run checks nothing.

THE DUPLICATE-KEY CHECK IS THE POINT. config.yaml carried two top-level
`classify:` blocks. PyYAML silently keeps the LAST one, and the last one had
no `provider` key - so a manifest-less run of run_slice1 would have raised
KeyError, while every manifest-driven run passed because
`mf.provider if mf else ccfg["provider"]` never evaluates the else branch. A
duplicate key in a config file is invisible to a normal load, so it is
detected explicitly here.
"""

import os
import sys
from collections import Counter
from pathlib import Path

import yaml

try:
    # The credentials live in .env, which client.py loads at runtime. Without
    # this the checker reports four missing variables that are in fact
    # present, and a check that cries wolf gets ignored.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CONFIG = Path("config/config.yaml")

REQUIRED = {
    "api": ["temperature", "timeout_seconds"],
    "classify": ["pinned_model", "max_repair_retries", "n_votes_dev",
                 "fewshot_seed"],
    "kb": ["records_path", "chroma_path", "embedding_model"],
    "retrieval": ["strategy", "k_definitions", "k_guidelines", "k_examples"],
}
VALID_STRATEGIES = {"dense", "bm25", "hybrid"}


class DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key.

    yaml.safe_load takes the last value and reports nothing, so a duplicated
    block looks identical to a correct file until something downstream reads a
    key the surviving block does not have.
    """


def _no_duplicates(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    if dupes:
        raise yaml.YAMLError(
            f"duplicate key(s) {dupes} at line {node.start_mark.line + 1}. "
            f"YAML keeps the LAST one silently.")
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def main():
    errors, warnings = [], []

    if not CONFIG.exists():
        print(f"{CONFIG} does not exist")
        sys.exit(1)

    raw = CONFIG.read_text(encoding="utf-8")
    try:
        cfg = yaml.load(raw, Loader=DuplicateKeyLoader)
    except yaml.YAMLError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    print(f"{CONFIG}: parses, no duplicate keys")

    for block, keys in REQUIRED.items():
        if block not in cfg:
            errors.append(f"missing block '{block}'")
            continue
        for key in keys:
            if key not in cfg[block]:
                errors.append(f"{block}.{key} missing")

    # --- providers ---
    api = cfg.get("api", {})
    providers = [k for k, v in api.items() if isinstance(v, dict)]
    print(f"\nproviders: {providers}")
    for p in providers:
        block = api[p]
        for key in ("url_env", "token_env", "default_model"):
            if key not in block:
                errors.append(f"api.{p}.{key} missing")
        for key in ("url_env", "token_env"):
            env = block.get(key)
            # Not an error: .env is gitignored and a machine may legitimately
            # hold credentials for only one provider.
            if env and not os.environ.get(env):
                warnings.append(f"api.{p}.{key} -> ${env} is not set")
        print(f"  {p:10} default_model={block.get('default_model')}")

    # --- classify ---
    cl = cfg.get("classify", {})
    print(f"\nclassify: provider={cl.get('provider')} "
          f"pinned_model={cl.get('pinned_model')} "
          f"n_votes_dev={cl.get('n_votes_dev')} T={api.get('temperature')}")
    # A pinned model that no provider serves fails only at run time, after the
    # confirmation prompt has been answered.
    served = {m for p in providers
              for key in ("default_model", "models_strong", "models_medium",
                          "models_fast")
              for m in ([api[p][key]] if isinstance(api[p].get(key), str)
                        else api[p].get(key) or [])}
    if cl.get("pinned_model") and cl["pinned_model"] not in served:
        warnings.append(
            f"classify.pinned_model '{cl['pinned_model']}' is not listed "
            f"under any provider; every manifest must override it")

    # --- retrieval ---
    r = cfg.get("retrieval", {})
    if r.get("strategy") not in VALID_STRATEGIES:
        errors.append(f"retrieval.strategy '{r.get('strategy')}' not in "
                      f"{VALID_STRATEGIES}")
    budgets = {k: v for k, v in r.items() if k.startswith("k_")}
    print(f"\nretrieval: strategy={r.get('strategy')} mmr={r.get('use_mmr')}")
    for k, v in sorted(budgets.items()):
        if not isinstance(v, int) or v < 0:
            errors.append(f"retrieval.{k} must be a non-negative int, got {v}")
        print(f"  {k:22} {v}")
    if "k_examples_feedback" not in budgets:
        warnings.append("retrieval.k_examples_feedback missing; SQ3 rounds "
                        "need it and it defaults to 0 in code")

    # --- paths ---
    kb = cfg.get("kb", {})
    print()
    for key in ("records_path", "chroma_path"):
        p = Path(kb.get(key, ""))
        print(f"  kb.{key:14} {p}  {'exists' if p.exists() else 'MISSING'}")
        if not p.exists():
            warnings.append(f"kb.{key} -> {p} does not exist")

    # --- verdict ---
    for w in warnings:
        print(f"\nWARN  {w}")
    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print(f"\nConfig valid. {len(warnings)} warning(s).")


if __name__ == "__main__":
    main()
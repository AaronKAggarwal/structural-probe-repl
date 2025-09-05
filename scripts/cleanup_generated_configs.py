#!/usr/bin/env python3
"""
Remove auto-generated configs for a given experiment group/model/language slugs.

This deletes:
  - configs/dataset/<slug>/<slug>.yaml
  - configs/embeddings/<slug>/<model_sanitized>/*.yaml
  - configs/logging/<slug>_<model_sanitized>/*.yaml
  - configs/experiment/<experiment_group>/<model_sanitized>/{dist,depth}/*.yaml

Use with care; it only targets the specific group/model/slugs provided.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def rm_tree_if_exists(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cleanup generated configs")
    parser.add_argument("--experiment_group", required=True)
    parser.add_argument("--model", required=True, help="Sanitized model name (e.g., bert-base-multilingual-cased)")
    parser.add_argument("--slugs", required=True, help="Comma-separated list of treebank slugs")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]

    # Remove dataset YAMLs
    for slug in slugs:
        ds_dir = project_root / "configs" / "dataset" / slug
        ds_yaml = ds_dir / f"{slug}.yaml"
        if ds_yaml.exists():
            ds_yaml.unlink()
        # Remove directory if empty
        try:
            ds_dir.rmdir()
        except OSError:
            pass

    # Remove embeddings
    emb_dir = project_root / "configs" / "embeddings"
    for slug in slugs:
        rm_tree_if_exists(emb_dir / slug / args.model)
        # Clean slug dir if empty
        slug_dir = emb_dir / slug
        try:
            slug_dir.rmdir()
        except OSError:
            pass

    # Remove logging
    log_dir = project_root / "configs" / "logging"
    for slug in slugs:
        rm_tree_if_exists(log_dir / f"{slug}_{args.model}")

    # Remove experiments (dist, depth)
    exp_base = project_root / "configs" / "experiment" / args.experiment_group / args.model
    rm_tree_if_exists(exp_base / "dist")
    rm_tree_if_exists(exp_base / "depth")
    # Remove model dir if empty
    try:
        exp_base.rmdir()
    except OSError:
        pass
    # Remove group dir if empty
    try:
        exp_group_dir = project_root / "configs" / "experiment" / args.experiment_group
        exp_group_dir.rmdir()
    except OSError:
        pass

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



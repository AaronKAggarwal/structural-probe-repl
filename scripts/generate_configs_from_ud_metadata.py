#!/usr/bin/env python3
"""
Generate Hydra configs for all UD treebanks listed in metadata CSV.

For each row in data/lang_stats/ud_metadata.csv (or provided CSV), this script:
  - Creates a dataset config at configs/dataset/<treebank_slug>/<treebank_slug>.yaml
    pointing to copied CoNLL-U files under data/ud/<treebank_slug>/.
  - Generates embeddings/logging/experiment configs for a chosen model and experiment group
    by invoking the same layout used in scripts/generate_configs.py.

These configs can be deleted later with scripts/cleanup_generated_configs.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List
import importlib.util


def load_create_config_files_function(scripts_dir: Path):
    """Dynamically load create_config_files from scripts/generate_configs.py."""
    target = scripts_dir / "generate_configs.py"
    spec = importlib.util.spec_from_file_location("gen_cfg", str(target))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "create_config_files"):
        raise RuntimeError("create_config_files not found in generate_configs.py")
    return getattr(mod, "create_config_files")


def write_dataset_yaml(base_dir: Path, slug: str, train: str, dev: str, test: str) -> Path:
    ds_dir = base_dir / slug
    ds_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = ds_dir / f"{slug}.yaml"
    # Use project-root relative paths
    train_rel = str(Path(train))
    dev_rel = str(Path(dev))
    test_rel = str(Path(test))
    content = f"""
name: {slug}
preload: true
paths:
  conllu_train: {train_rel}
  conllu_dev:   {dev_rel}
  conllu_test:  {test_rel}
""".lstrip()
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate dataset+experiment configs from UD metadata CSV")
    parser.add_argument("--metadata_csv", type=Path, default=Path("data/lang_stats/ud_metadata.csv"))
    parser.add_argument("--model_hf_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--model_name_sanitized", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--model_dimension", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=13)
    parser.add_argument("--experiment_group", type=str, default="baselines_auto")
    parser.add_argument("--filter_slugs", type=str, default=None, help="Comma-separated subset of treebank slugs to include")
    parser.add_argument("--local_model_dir", type=str, default=None, help="Optional local HF model dir for offline runs")
    args = parser.parse_args(argv)

    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent

    # Load generator from existing script
    create_config_files = load_create_config_files_function(scripts_dir)

    # Read metadata
    rows = list(csv.DictReader(open(args.metadata_csv, encoding="utf-8")))
    if not rows:
        print(f"No rows found in {args.metadata_csv}", file=sys.stderr)
        return 1

    subset = None
    if args.filter_slugs:
        subset = {s.strip() for s in args.filter_slugs.split(",") if s.strip()}

    generated = 0
    for row in rows:
        slug = (row.get("treebank_slug") or "").strip()
        if not slug:
            continue
        if subset and slug not in subset:
            continue

        train = row.get("train_conllu_path") or ""
        dev = row.get("dev_conllu_path") or ""
        test = row.get("test_conllu_path") or ""
        if not (train and dev and test):
            print(f"Skipping {slug}: missing split paths", file=sys.stderr)
            continue

        # 1) Dataset YAML
        ds_yaml = write_dataset_yaml(project_root / "configs" / "dataset", slug, train, dev, test)
        print(f"Wrote dataset config: {ds_yaml}")

        # 2) Embeddings/logging/experiment configs for model
        # Keep human-readable model name for directory naming; use local path only for loading
        effective_model_hf_name = args.local_model_dir or args.model_hf_name
        effective_model_name_sanitized = args.model_name_sanitized.replace('/', '-')

        create_config_files(
            model_hf_name=effective_model_hf_name,
            model_name_sanitized=effective_model_name_sanitized,
            dataset_name=slug,
            experiment_group=args.experiment_group,
            model_dimension=args.model_dimension,
            num_layers=args.num_layers,
            base_output_dir=str(project_root / "configs"),
        )
        generated += 1

    print(f"\nGenerated configs for {generated} treebank(s).")
    print("Cleanup later with scripts/cleanup_generated_configs.py --experiment_group <group> --model <sanitized> --slugs <...>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



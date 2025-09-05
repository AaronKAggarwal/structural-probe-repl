#!/usr/bin/env python3
"""
Prefetch a Hugging Face model repository to a local directory for offline use.

Example:
  poetry run python scripts/prefetch_model.py \
    --model_id bert-base-multilingual-cased \
    --local_dir /Users/aaronaggarwal/structural-probe-main/env/models/bert-base-multilingual-cased

Then run extract/train with:
  model.hf_model_name=/Users/aaronaggarwal/structural-probe-main/env/models/bert-base-multilingual-cased

To force offline:
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 poetry run python scripts/extract_embeddings.py ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch HF model to local dir for offline use")
    parser.add_argument("--model_id", required=True, help="HF model id, e.g., bert-base-multilingual-cased")
    parser.add_argument(
        "--local_dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to env/models/<sanitized model_id>",
    )
    parser.add_argument("--revision", default=None, help="Optional git revision/tag/hash to pin")
    args = parser.parse_args(argv)

    model_id = args.model_id
    sanitized = model_id.replace("/", "-")
    dest = args.local_dir or (Path("env/models") / sanitized)
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{model_id}' to: {dest}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        revision=args.revision,
        ignore_patterns=["*.msgpack*", "*.h5", "*.pt", "*.bin.index.json", "*.safetensors.index.json"],
    )
    print("Done.")
    print("Use this path as model.hf_model_name in configs or CLI overrides:")
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



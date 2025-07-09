#!/bin/bash
# This script runs a 1-epoch smoke test for all major experiment configs.
# It helps verify that all paths and Hydra compositions are correct.
# The script will exit immediately if any command fails.
set -e

echo "--- Testing Sanity Check Config ---"
poetry run python scripts/train_probe.py experiment=sanity_checks/bert-base-cased/dist/L0_tinysample

echo -e "\n--- Testing H&M Replication (Depth) Config ---"
poetry run python scripts/train_probe.py experiment=hm_replication/bert-base-cased/depth/L7 training.epochs=1 training.batch_size=2 logging.wandb.enable=false

echo -e "\n--- Testing UD Baselines (Depth) Configs ---"
poetry run python scripts/train_probe.py experiment=ud_ewt/elmo/depth/L0 training.epochs=1 training.batch_size=2 logging.wandb.enable=false
poetry run python scripts/train_probe.py experiment=ud_ewt/elmo/depth/L1 training.epochs=1 training.batch_size=2 logging.wandb.enable=false
poetry run python scripts/train_probe.py experiment=ud_ewt/elmo/depth/L2 training.epochs=1 training.batch_size=2 logging.wandb.enable=false

echo -e "\n\n All tested configurations loaded and ran for one epoch successfully!"
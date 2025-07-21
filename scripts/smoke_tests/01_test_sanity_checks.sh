#!/bin/bash
set -e
echo -e "\n\n========================================================================"
echo " SMOKE TEST SUITE: Sanity Checks"
echo "========================================================================"

poetry run python scripts/train_probe.py \
  experiment="sanity_checks/bert-base-cased/dist/L0_tinysample"

echo -e "SUCCESS: Sanity Checks PASSED."
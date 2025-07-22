#!/bin/bash
set -e
echo -e "\n\n========================================================================"
echo " SMOKE TEST SUITE: Sanity Checks"
echo "========================================================================"

# Define fast overrides for a quick, minimal run.
# Using '++' to robustly add/override keys.
FAST_OVERRIDES="training.epochs=2 training.batch_size=1 \
                ++training.eval_on_train_epoch_end=true \
                ++training.limit_train_batches=-1 \
                ++training.limit_eval_batches=-1 \
                ++training.early_stopping_metric=loss \
                ++training.patience=2 \
                logging.wandb.enable=false +logging.enable_plots=false"

# Define a runner function for consistency with other test suites.
run_test() {
  echo -e "\n--- Testing: $1 ---"
  poetry run python scripts/train_probe.py experiment="$2" $FAST_OVERRIDES
}

# Run the sanity check experiment.
run_test "Sanity Check (BERT L0 Distance on Tiny Sample)" "sanity_checks/bert-base-cased/dist/L0_tinysample"

echo -e "\nSUCCESS: Sanity Checks PASSED."
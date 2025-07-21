#!/bin/bash
set -e
echo -e "\n\n========================================================================"
echo " SMOKE TEST SUITE: Modern Baselines on UD-EWT"
echo "========================================================================"

FAST_OVERRIDES="training.epochs=1 training.batch_size=2 \
                ++training.eval_on_train_epoch_end=false \
                ++training.limit_train_batches=2 \
                ++training.limit_eval_batches=2 \
                logging.wandb.enable=false +logging.enable_plots=false"

run_test() {
  echo -e "\n--- Testing: $1 ---"
  poetry run python scripts/train_probe.py experiment="$2" $FAST_OVERRIDES
}

run_test "BERT L7 Distance (Modern)" "modern_baselines/ud_ewt/bert-base-cased/dist/L7"
run_test "ELMo L0 Distance (Modern)" "modern_baselines/ud_ewt/elmo/dist/L0"
run_test "ELMo L1 Distance (Modern)" "modern_baselines/ud_ewt/elmo/dist/L1"
run_test "ELMo L2 Distance (Modern)" "modern_baselines/ud_ewt/elmo/dist/L2"
# Add other modern baselines here as you create them

echo -e "\nSUCCESS: Modern Baselines Configs PASSED."
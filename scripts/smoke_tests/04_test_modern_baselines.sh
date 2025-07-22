#!/bin/bash
set -e
echo -e "\n\n========================================================================"
echo " SMOKE TEST SUITE: Modern Baselines on UD-EWT"
echo "========================================================================"

# Use '++' to force override if key exists, or add if it doesn't.
# This makes the script robust to different underlying training configs.
FAST_OVERRIDES="training.epochs=1 training.batch_size=2 \
                ++training.eval_on_train_epoch_end=false \
                ++training.limit_train_batches=20 \
                ++training.limit_eval_batches=4 \
                logging.wandb.enable=false +logging.enable_plots=false"

run_test() {
  echo -e "\n--- Testing: $1 ---"
  poetry run python scripts/train_probe.py experiment="$2" $FAST_OVERRIDES
}

# --- Test Distance Probes ---
run_test "Modern Baseline (BERT L7 Distance)" "modern_baselines/ud_ewt/bert-base-cased/dist/L7"
run_test "Modern Baseline (ELMo L0 Distance)" "modern_baselines/ud_ewt/elmo/dist/L0"
run_test "Modern Baseline (ELMo L1 Distance)" "modern_baselines/ud_ewt/elmo/dist/L1"
run_test "Modern Baseline (ELMo L2 Distance)" "modern_baselines/ud_ewt/elmo/dist/L2"

# --- Test Depth Probes ---
run_test "Modern Baseline (ELMo L0 Depth)" "modern_baselines/ud_ewt/elmo/depth/L0"
run_test "Modern Baseline (ELMo L1 Depth)" "modern_baselines/ud_ewt/elmo/depth/L1"
run_test "Modern Baseline (ELMo L2 Depth)" "modern_baselines/ud_ewt/elmo/depth/L2"


echo -e "\nSUCCESS: Modern Baselines Configs PASSED."
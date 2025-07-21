#!/bin/bash
set -e
echo -e "\n\n========================================================================"
echo " SMOKE TEST SUITE: H&M-Style Baselines on UD-EWT"
echo "========================================================================"

FAST_OVERRIDES="training.epochs=1 training.batch_size=2 \
                ++training.eval_on_train_epoch_end=false \
                ++training.limit_train_batches=20 \
                ++training.limit_eval_batches=4 \
                logging.wandb.enable=false +logging.enable_plots=false"

run_test() {
  echo -e "\n--- Testing: $1 ---"
  poetry run python scripts/train_probe.py experiment="$2" $FAST_OVERRIDES
}

run_test "BERT L7 Distance" "hm_style_baselines/ud_ewt/bert/dist/L7"
run_test "ELMo L0 Distance" "hm_style_baselines/ud_ewt/elmo/dist/L0"
run_test "ELMo L1 Distance" "hm_style_baselines/ud_ewt/elmo/dist/L1"
run_test "ELMo L2 Distance" "hm_style_baselines/ud_ewt/elmo/dist/L2"

echo -e "\nSUCCESS: H&M-Style Baselines Configs PASSED."
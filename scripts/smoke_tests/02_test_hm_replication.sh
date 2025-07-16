#!/bin/bash
set -e
echo -e "\n\n========================================================================"
echo " SMOKE TEST SUITE: H&M Replication on PTB"
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

run_test "H&M Replication (BERT L7 Distance)" "hm_replication/bert-base-cased/dist/L7"
run_test "H&M Replication (BERT L7 Depth)" "hm_replication/bert-base-cased/depth/L7"

echo -e "\nSUCCESS: H&M Replication Configs PASSED."
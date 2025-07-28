#!/bin/bash
#
# This script generates the full set of Hydra configuration files for the
# DISTANCE probes for both Llama-3.2-3B and Llama-3.2-3B-Instruct models.
#
# It creates:
# 1. Experiment configs for each model and layer (28 + 28 files).
# 2. Corresponding logging configs for each experiment.
#
# Run this script from the project's root directory.

set -e # Exit immediately if a command fails
echo "--- Generating Hydra DISTANCE Probe Configs for Llama 3.2 Models ---"

# --- Configuration ---
CONFIGS_ROOT="configs"
MODELS=("Llama-3.2-3B" "Llama-3.2-3B-Instruct")
LAYERS=$(seq 0 27) # Llama 3.2 3B has 28 layers (0-27)

# --- Template Content (based on your modern_baselines/ud_ewt/elmo/dist/L1) ---
# We will use variables inside this template string.
# Note: The `evaluation.metrics` is already correct in the `evaluation/default.yaml`
# for a distance probe, so we don't need to override it.

create_experiment_config() {
    local MODEL_NAME=$1
    local SANITIZED_MODEL_NAME=$(echo ${MODEL_NAME} | sed 's/\./p/g') # For filenames
    local LAYER_INDEX=$2

    local EXP_DIR="${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${MODEL_NAME}/dist"
    mkdir -p "${EXP_DIR}"
    
    local OUTPUT_FILE="${EXP_DIR}/L${LAYER_INDEX}.yaml"

    # Note: We use "128" for probe_rank as a starting point, consistent with ELMo.
    # This can be overridden on the command line if you want to test other ranks.
    cat > "${OUTPUT_FILE}" <<- EOM
# configs/experiment/new_models/ud_ewt/${MODEL_NAME}/dist/L${LAYER_INDEX}.yaml

name: new_models/ud_ewt/${MODEL_NAME}/dist/L${LAYER_INDEX}

defaults:
  - _self_
  - /dataset: ud_ewt/ud_english_ewt_full
  - /embeddings: ud_ewt/${MODEL_NAME}/L${LAYER_INDEX}
  - /probe: distance
  - /probe_rank@probe: "128"
  - /training: adam_modern_scheduler
  - /evaluation: default
  - /runtime: mps # Defaulting to parallel loader runtime
  - /logging: new_models/${SANITIZED_MODEL_NAME}_dist_L${LAYER_INDEX}

logging:
  experiment_name: "Modern_UDEWT_${MODEL_NAME}_L${LAYER_INDEX}_Dist_R128"
EOM
}

create_logging_config() {
    local MODEL_NAME=$1
    local SANITIZED_MODEL_NAME=$(echo ${MODEL_NAME} | sed 's/\./p/g')
    local TAG_MODEL_NAME=$(echo ${MODEL_NAME} | tr '[:upper:]' '[:lower:]')
    local LAYER_INDEX=$2
    
    local LOG_DIR="${CONFIGS_ROOT}/logging/new_models"
    mkdir -p "${LOG_DIR}"

    local OUTPUT_FILE="${LOG_DIR}/${SANITIZED_MODEL_NAME}_dist_L${LAYER_INDEX}.yaml"
    
    local TAG_TYPE="modern_llm"
    if [[ $MODEL_NAME == *"Instruct"* ]]; then
        TAG_TYPE="instruct_model"
    fi

    cat > "${OUTPUT_FILE}" <<- EOM
# configs/logging/new_models/${SANITIZED_MODEL_NAME}_dist_L${LAYER_INDEX}.yaml
wandb:
  tags: ["${TAG_TYPE}", "ud_ewt", "${TAG_MODEL_NAME}", "layer${LAYER_INDEX}", "distance", "r\${probe.rank}"]
EOM
}


# --- Main Loop ---
for model in "${MODELS[@]}"; do
    echo -e "\n>>> Generating configs for model: ${model} <<<"
    for layer in ${LAYERS}; do
        echo -n "." # Progress indicator
        create_experiment_config "${model}" "${layer}"
        create_logging_config "${model}" "${layer}"
    done
    echo -e "\nDone with ${model}."
done

echo -e "\n--- Config generation for ALL distance probes complete! ---"
echo "Please review the newly created files before running a sweep."
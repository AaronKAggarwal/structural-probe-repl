#!/bin/bash
#
# This script generates the full set of Hydra configuration files for the
# Llama-3.2-3B-Instruct model by using the existing base Llama-3.2-3B configs
# as templates. It handles embeddings, experiments, and logging files.
#
# Run this script from the project's root directory.

set -e # Exit immediately if a command fails
echo "--- Generating Hydra Configs for Llama-3.2-3B-Instruct ---"

# Define base paths
CONFIGS_ROOT="configs"
BASE_MODEL_NAME="Llama-3.2-3B"
INSTRUCT_MODEL_NAME="Llama-3.2-3B-Instruct"

# Check if the template directories exist
if [ ! -d "${CONFIGS_ROOT}/embeddings/ud_ewt/${BASE_MODEL_NAME}" ]; then
    echo "ERROR: Template embedding directory not found at ${CONFIGS_ROOT}/embeddings/ud_ewt/${BASE_MODEL_NAME}"
    exit 1
fi
if [ ! -d "${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${BASE_MODEL_NAME}" ]; then
    echo "ERROR: Template experiment directory not found at ${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${BASE_MODEL_NAME}"
    exit 1
fi

# --- 1. Create Embedding Configs ---
EMBEDDINGS_DIR="${CONFIGS_ROOT}/embeddings/ud_ewt/${INSTRUCT_MODEL_NAME}"
mkdir -p "${EMBEDDINGS_DIR}"
echo ">>> Creating Embedding configs in ${EMBEDDINGS_DIR}..."

# Llama 3.2 3B has 28 layers (indices 0-27)
for i in $(seq 0 27); do
    TEMPLATE_FILE="${CONFIGS_ROOT}/embeddings/ud_ewt/${BASE_MODEL_NAME}/L${i}.yaml"
    # Handling potential missing files if you haven't created all 0-31
    if [ ! -f "${TEMPLATE_FILE}" ]; then
        echo "Warning: Template for L${i} not found, creating from L0 template."
        TEMPLATE_FILE="${CONFIGS_ROOT}/embeddings/ud_ewt/${BASE_MODEL_NAME}/L0.yaml"
    fi

    OUTPUT_FILE="${EMBEDDINGS_DIR}/L${i}.yaml"
    sed "s/${BASE_MODEL_NAME}/${INSTRUCT_MODEL_NAME}/g" "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"
    # Correct the layer index in case we used L0 as a template
    sed -i '' "s/layer_index: .*/layer_index: ${i}/" "${OUTPUT_FILE}"
done
echo "Embedding configs created."

# --- 2. Create Experiment Configs (Depth) ---
EXP_DIR_DEPTH="${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${INSTRUCT_MODEL_NAME}/depth"
mkdir -p "${EXP_DIR_DEPTH}"
echo ">>> Creating Depth Probe Experiment configs in ${EXP_DIR_DEPTH}..."

for i in $(seq 0 27); do
    TEMPLATE_FILE="${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${BASE_MODEL_NAME}/depth/L${i}.yaml"
    if [ ! -f "${TEMPLATE_FILE}" ]; then
        echo "Warning: Template for L${i} not found, creating from L0 template."
        TEMPLATE_FILE="${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${BASE_MODEL_NAME}/depth/L0.yaml"
    fi

    OUTPUT_FILE="${EXP_DIR_DEPTH}/L${i}.yaml"
    # Use a different delimiter for sed since the paths contain slashes
    sed "s|${BASE_MODEL_NAME}|${INSTRUCT_MODEL_NAME}|g" "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"
    # Correct the layer index in the name, paths, etc.
    sed -i '' "s|/L[0-9]\{1,2\}|/L${i}|g" "${OUTPUT_FILE}"
    sed -i '' "s|L[0-9]\{1,2\}_Depth|L${i}_Depth|g" "${OUTPUT_FILE}"
    sed -i '' "s|depth_L[0-9]\{1,2\}|depth_L${i}|g" "${OUTPUT_FILE}"
done
echo "Depth probe experiment configs created."

# --- 3. Create Logging Configs (Depth) ---
LOGGING_DIR="${CONFIGS_ROOT}/logging/new_models"
mkdir -p "${LOGGING_DIR}"
echo ">>> Creating Logging configs in ${LOGGING_DIR}..."

for i in $(seq 0 27); do
    # It seems your logging files for llama are not in a sub-directory, let's use a template.
    # We will create a generic template here to be safe.
    LOG_FILE_CONTENT="wandb:\n  tags: [\"instruct_model\", \"ud_ewt\", \"llama-3.2-3b-instruct\", \"layer${i}\", \"depth\", \"r\${probe.rank}\"]"
    
    OUTPUT_FILE="${LOGGING_DIR}/llama-3.2-3b-instruct_depth_L${i}.yaml"
    echo -e "${LOG_FILE_CONTENT}" > "${OUTPUT_FILE}"
done
echo "Logging configs created."


# --- 4. (Optional but Recommended) Create Distance Probe Experiment Configs ---
read -p "Do you want to generate configs for DISTANCE probes as well? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    EXP_DIR_DIST="${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${INSTRUCT_MODEL_NAME}/dist"
    mkdir -p "${EXP_DIR_DIST}"
    echo ">>> Creating Distance Probe Experiment configs in ${EXP_DIR_DIST}..."

    for i in $(seq 0 27); do
        # We assume a dist folder exists for the base model, or we create it from a template
        TEMPLATE_FILE="${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${BASE_MODEL_NAME}/dist/L${i}.yaml"
        if [ ! -f "${TEMPLATE_FILE}" ]; then
            echo "Warning: No distance template for L${i}. Creating from depth template and modifying."
            TEMPLATE_FILE="${CONFIGS_ROOT}/experiment/new_models/ud_ewt/${BASE_MODEL_NAME}/depth/L0.yaml" # Use a known good file
            
            OUTPUT_FILE="${EXP_DIR_DIST}/L${i}.yaml"
            sed "s|${BASE_MODEL_NAME}|${INSTRUCT_MODEL_NAME}|g" "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"
            
            # Modify for distance probe
            sed -i '' "s|/L0|/L${i}|g" "${OUTPUT_FILE}"
            sed -i '' "s|/depth|/dist|g" "${OUTPUT_FILE}"
            sed -i '' "s|probe: depth|probe: distance|g" "${OUTPUT_FILE}"
            sed -i '' "s|metrics:.*|metrics: [\"spearmanr_hm\", \"uuas\"] # Metrics for distance probe|" "${OUTPUT_FILE}"
            sed -i '' "s|L0_Depth|L${i}_Dist|g" "${OUTPUT_FILE}"
        else
            OUTPUT_FILE="${EXP_DIR_DIST}/L${i}.yaml"
            sed "s|${BASE_MODEL_NAME}|${INSTRUCT_MODEL_NAME}|g" "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"
            sed -i '' "s|/L[0-9]\{1,2\}|/L${i}|g" "${OUTPUT_FILE}"
            sed -i '' "s|L[0-9]\{1,2\}_Dist|L${i}_Dist|g" "${OUTPUT_FILE}"
            sed -i '' "s|dist_L[0-9]\{1,2\}|dist_L${i}|g" "${OUTPUT_FILE}"
        fi
    done
    echo "Distance probe experiment configs created."
fi

echo -e "\n--- Config generation complete! ---"
echo "Please review the newly created files before running a sweep."
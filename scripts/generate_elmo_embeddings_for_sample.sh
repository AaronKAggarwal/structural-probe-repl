#!/bin/bash
set -e

SAMPLE_DATA_DIR_HOST="data_staging/my_ewt_sample_for_legacy_probe/example/data/en_ewt-ud-sample"
# Path INSIDE the container where we'll mount this data
SAMPLE_DATA_DIR_CONTAINER="/mnt/sample_data" 

# Output directory for HDF5 files (will be created on host via mount)
OUTPUT_HDF5_DIR_HOST="${SAMPLE_DATA_DIR_HOST}" # Save HDF5 in the same place as .txt and .conllu

IMAGE_NAME="probe:legacy_pt_cpu" # Our existing Docker image

echo "Ensuring Docker image ${IMAGE_NAME} is built..."
# (Assuming it's already built from previous steps)

for SPLIT in train dev test; do
    RAW_TEXT_FILE="en_ewt-ud-${SPLIT}.txt"
    HDF5_OUTPUT_FILE="en_ewt-ud-${SPLIT}.elmo-layers.hdf5" # H&M naming convention

    if [ ! -f "${SAMPLE_DATA_DIR_HOST}/${RAW_TEXT_FILE}" ]; then
        echo "ERROR: Raw text file ${SAMPLE_DATA_DIR_HOST}/${RAW_TEXT_FILE} not found. Run convert_sample_conllu_to_raw.py first."
        exit 1
    fi

    echo "Generating ELMo embeddings for ${RAW_TEXT_FILE} -> ${HDF5_OUTPUT_FILE}..."
    echo "This will take some time, especially the first time ELMo models are downloaded."

    # Run AllenNLP elmo command inside the Docker container
    # We mount the sample data directory to /mnt/sample_data inside the container
    # AllenNLP will write the output HDF5 file to this mounted directory.
    # The `allennlp elmo` command in v0.9.0 expects input file and output file paths.
    # The --all flag ensures all 3 ELMo layers are output.
    docker run --rm --platform=linux/amd64 \
        -v "$(pwd)/${SAMPLE_DATA_DIR_HOST}":"${SAMPLE_DATA_DIR_CONTAINER}" \
        -w "${SAMPLE_DATA_DIR_CONTAINER}" \
        "${IMAGE_NAME}" \
        allennlp elmo "${RAW_TEXT_FILE}" "${HDF5_OUTPUT_FILE}" --all 
        # Note: If allennlp elmo is not directly on PATH, may need `python -m allennlp.run elmo ...`
        # The base pytorch image might not have allennlp CLI on PATH.
        # Let's try with `python -m allennlp.run elmo` for robustness.
    
    # Corrected command using python -m:
    docker run --rm --platform=linux/amd64 \
        -v "$(pwd)/${SAMPLE_DATA_DIR_HOST}":"${SAMPLE_DATA_DIR_CONTAINER}" \
        -w "${SAMPLE_DATA_DIR_CONTAINER}" \
        "${IMAGE_NAME}" \
        python -m allennlp.run elmo "${RAW_TEXT_FILE}" "${HDF5_OUTPUT_FILE}" --all

    if [ -f "${OUTPUT_HDF5_DIR_HOST}/${HDF5_OUTPUT_FILE}" ]; then
        echo "Successfully generated ${OUTPUT_HDF5_DIR_HOST}/${HDF5_OUTPUT_FILE}"
    else
        echo "ERROR: Failed to generate ${OUTPUT_HDF5_DIR_HOST}/${HDF5_OUTPUT_FILE}"
        exit 1
    fi
done

echo "ELMo HDF5 embedding generation complete for all sample splits."
ls -lh "${OUTPUT_HDF5_DIR_HOST}"/*.hdf5
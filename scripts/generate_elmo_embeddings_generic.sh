#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_raw_text_file_path_on_host> <output_hdf5_file_path_on_host>"
    exit 1
fi

# These are paths relative to the project root when you call the script
RELATIVE_HOST_RAW_TEXT_FILE_PATH="$1"
RELATIVE_HOST_HDF5_OUTPUT_FILE_PATH="$2"

# Construct ABSOLUTE host paths
ABS_HOST_RAW_TEXT_FILE_PATH="$(pwd)/${RELATIVE_HOST_RAW_TEXT_FILE_PATH}"
ABS_HOST_HDF5_OUTPUT_FILE_PATH="$(pwd)/${RELATIVE_HOST_HDF5_OUTPUT_FILE_PATH}"

# Derive container paths from host paths
RAW_TEXT_FILENAME=$(basename "${ABS_HOST_RAW_TEXT_FILE_PATH}")
HDF5_OUTPUT_FILENAME=$(basename "${ABS_HOST_HDF5_OUTPUT_FILE_PATH}")

# Mount points inside container
RAW_TEXT_DIR_CONTAINER="/mnt/raw_texts"
HDF5_OUTPUT_DIR_CONTAINER="/mnt/hdf5_outputs"

CONTAINER_RAW_TEXT_FILE="${RAW_TEXT_DIR_CONTAINER}/${RAW_TEXT_FILENAME}"
CONTAINER_HDF5_OUTPUT_FILE="${HDF5_OUTPUT_DIR_CONTAINER}/${HDF5_OUTPUT_FILENAME}"

IMAGE_NAME="probe:legacy_pt_cpu"

echo "Ensuring Docker image ${IMAGE_NAME} is built..."

if [ ! -f "${ABS_HOST_RAW_TEXT_FILE_PATH}" ]; then
    echo "ERROR: Input raw text file ${ABS_HOST_RAW_TEXT_FILE_PATH} not found."
    exit 1
fi

mkdir -p "$(dirname "${ABS_HOST_HDF5_OUTPUT_FILE_PATH}")"

echo "Generating ELMo embeddings for ${ABS_HOST_RAW_TEXT_FILE_PATH} -> ${ABS_HOST_HDF5_OUTPUT_FILE_PATH}..."
echo "This may take a very long time for large files, especially the first time ELMo models are downloaded by the container."

docker run --rm --platform=linux/amd64 \
    -v "$(dirname "${ABS_HOST_RAW_TEXT_FILE_PATH}")":"${RAW_TEXT_DIR_CONTAINER}":ro \
    -v "$(dirname "${ABS_HOST_HDF5_OUTPUT_FILE_PATH}")":"${HDF5_OUTPUT_DIR_CONTAINER}":rw \
    -w "${RAW_TEXT_DIR_CONTAINER}" \
    "${IMAGE_NAME}" \
    python -m allennlp.run elmo "${RAW_TEXT_FILENAME}" "${CONTAINER_HDF5_OUTPUT_FILE}" --all

if [ -f "${ABS_HOST_HDF5_OUTPUT_FILE_PATH}" ]; then
    echo "Successfully generated ${ABS_HOST_HDF5_OUTPUT_FILE_PATH}"
else
    echo "ERROR: Failed to generate ${ABS_HOST_HDF5_OUTPUT_FILE_PATH}"
    exit 1
fi
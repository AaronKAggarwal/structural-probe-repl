#!/bin/bash
# Adapted from Hewitt & Manning's script for converting PTB to Stanford Dependencies (CoNLL-X)

# --- Configuration ---
# !!! IMPORTANT: SET THIS TO THE ROOT OF YOUR EXTRACTED PENN TREEBANK 3 DIRECTORY !!!
# This directory should contain 'parsed/mrg/wsj/...'
PTB_ROOT_DIR="/Users/aaronaggarwal/data/LDC99T42/treebank3" 

# !!! IMPORTANT: SET THIS TO THE DIRECTORY WHERE STANFORD CORENLP 3.9.2 IS UNZIPPED !!!
# This is the directory that contains all the .jar files (e.g., stanford-corenlp-3.9.2.jar)
CORENLP_HOME="/Users/aaronaggarwal/tools/stanford-corenlp-3.9.2/stanford-corenlp-full-2018-10-05"

# Output directory for processed files within your project
# Assuming this script is run from structural-probe-repl/data_processing_scripts/
PROJECT_ROOT_GUESS="$(cd "$(dirname "$0")/.." && pwd)" # Tries to guess project root
OUTPUT_BASE_DIR="${PROJECT_ROOT_GUESS}/data/ptb_stanford_dependencies_conllx"
TEMP_TREES_DIR="${OUTPUT_BASE_DIR}/temp_trees" # For intermediate concatenated .trees files

# Java memory allocation
JAVA_MX="16g" # Adjust if needed (e.g., "4g")
# --- End Configuration ---


# --- Ensure Output Directories Exist ---
mkdir -p "${TEMP_TREES_DIR}"
mkdir -p "${OUTPUT_BASE_DIR}"

# --- Check if PTB_ROOT_DIR is set and valid ---
if [ ! -d "${PTB_ROOT_DIR}/parsed/mrg/wsj/02" ]; then
    echo "ERROR: PTB_ROOT_DIR is not set correctly or PTB data not found at expected path."
    echo "Please set PTB_ROOT_DIR in this script to the root of your Penn Treebank 3 extraction."
    echo "Expected to find: ${PTB_ROOT_DIR}/parsed/mrg/wsj/02"
    exit 1
fi

# --- Check if CORENLP_HOME is set and valid ---
if [ ! -f "${CORENLP_HOME}/stanford-corenlp-3.9.2.jar" ]; then # Check for a key jar file
    echo "ERROR: CORENLP_HOME is not set correctly or CoreNLP JARs not found."
    echo "Please set CORENLP_HOME to your Stanford CoreNLP 3.9.2 directory."
    echo "Expected to find: ${CORENLP_HOME}/stanford-corenlp-3.9.2.jar"
    exit 1
fi

echo "Using PTB from: ${PTB_ROOT_DIR}"
echo "Using CoreNLP from: ${CORENLP_HOME}"
echo "Processed files will be saved in: ${OUTPUT_BASE_DIR}"
echo "Temporary .trees files in: ${TEMP_TREES_DIR}"


# Concatenate .mrg files for each split
echo "Concatenating training trees (sections 02-21)..."
# Ensure the output file is empty before appending
> "${TEMP_TREES_DIR}/ptb3-wsj-train.trees" 
for i in $(seq -w 02 21); do
        cat "${PTB_ROOT_DIR}/parsed/mrg/wsj/${i}/"*.mrg >> "${TEMP_TREES_DIR}/ptb3-wsj-train.trees"
done
echo "Done."

echo "Concatenating development trees (section 22)..."
> "${TEMP_TREES_DIR}/ptb3-wsj-dev.trees"
for i in 22; do
        cat "${PTB_ROOT_DIR}/parsed/mrg/wsj/${i}/"*.mrg >> "${TEMP_TREES_DIR}/ptb3-wsj-dev.trees"
done
echo "Done."

echo "Concatenating test trees (section 23)..."
> "${TEMP_TREES_DIR}/ptb3-wsj-test.trees"
for i in 23; do
        cat "${PTB_ROOT_DIR}/parsed/mrg/wsj/${i}/"*.mrg >> "${TEMP_TREES_DIR}/ptb3-wsj-test.trees"
done
echo "Done."

# Convert concatenated .trees files to CoNLL-X format
# IMPORTANT: This command needs to be run from CORENLP_HOME or have its classpath set correctly.
# We will cd into CORENLP_HOME for simplicity.
echo "Changing directory to ${CORENLP_HOME} for CoreNLP execution..."
ORIGINAL_PWD=$(pwd)
cd "${CORENLP_HOME}" || { echo "Failed to cd into ${CORENLP_HOME}"; exit 1; }

for split in train dev test; do
    echo "Converting ${split} split to CoNLL-X..."
    input_tree_file="${TEMP_TREES_DIR}/ptb3-wsj-${split}.trees"
    output_conllx_file="${OUTPUT_BASE_DIR}/ptb3-wsj-${split}.conllx"
    
    java "-mx${JAVA_MX}" -cp "*" edu.stanford.nlp.trees.EnglishGrammaticalStructure \
        -treeFile "${input_tree_file}" \
        -checkConnected \
        -basic \
        -keepPunct \
        -conllx > "${output_conllx_file}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully converted ${split} split to ${output_conllx_file}"
    else
        echo "ERROR: Failed to convert ${split} split. Check for Java errors."
        # Consider exiting or allowing to continue: exit 1; 
    fi
done

echo "Returning to original directory: ${ORIGINAL_PWD}"
cd "${ORIGINAL_PWD}"

echo "Conversion process complete. Output files should be in ${OUTPUT_BASE_DIR}"
echo "Temporary .trees files are in ${TEMP_TREES_DIR} (can be deleted if desired)."
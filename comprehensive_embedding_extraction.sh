#!/bin/bash

# Comprehensive Embedding Extraction for All Clean UD Languages
# Generated: $(date)
# Purpose: Extract embeddings with maximum utility from corrected, clean UD data
#
# Features:
# - All 23 clean UD languages
# - All 13 layers (0-12) extracted
# - Full output (no truncation)
# - Tokenization maps saved (for alignment analysis)
# - Input IDs saved (for detailed token analysis)
# - Uses NEW clean data with corrected Vietnamese and filtered Czech

set -e  # Exit on any error

echo "=== COMPREHENSIVE EMBEDDING EXTRACTION FOR ALL UD LANGUAGES ==="
echo "Started: $(date)"
echo ""

# Configuration
MODEL="bert-base-multilingual-cased"
BASE_COMMAND="poetry run python scripts/extract_embeddings.py"

# Array of all clean UD languages (NEW corrected data)
UD_LANGUAGES=(
    "UD_Arabic-PADT"
    "UD_Basque-BDT"
    "UD_Bulgarian-BTB"
    "UD_Chinese-GSD"
    "UD_Czech-PDTC"        # Filtered Czech data (68k train sentences)
    "UD_English-EWT"
    "UD_Finnish-TDT"
    "UD_French-GSD"
    "UD_German-GSD"
    "UD_Greek-GDT"
    "UD_Hebrew-HTB"
    "UD_Hindi-HDTB"
    "UD_Hungarian-Szeged"
    "UD_Indonesian-GSD"
    "UD_Japanese-GSD"
    "UD_Korean-GSD"
    "UD_Persian-Seraji"
    "UD_Polish-PDB"
    "UD_Russian-SynTagRus"
    "UD_Spanish-AnCora"
    "UD_Turkish-IMST"
    "UD_Urdu-UDTB"
    "UD_Vietnamese-VTB"    # Fixed Vietnamese data (1123 dev sentences)
)

# Function to run extraction for a single language
extract_language() {
    local lang=$1
    echo ""
    echo "🚀 Processing: $lang"
    echo "----------------------------------------"
    
    $BASE_COMMAND \
        dataset="${lang}/${lang}" \
        model="$MODEL" \
        job.layers_to_extract=all \
        job.save_tokenization_map=true \
        job.save_input_ids=true
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $lang"
    else
        echo "❌ Failed: $lang"
        return 1
    fi
}

# Main extraction loop
echo "Total languages to process: ${#UD_LANGUAGES[@]}"
echo "Model: $MODEL"
echo "Layers: all (0-12)"
echo "Extras: tokenization_map + input_ids"
echo "Output: full (no truncation)"
echo ""

SUCCESSFUL=0
FAILED=0
FAILED_LANGUAGES=()

for lang in "${UD_LANGUAGES[@]}"; do
    if extract_language "$lang"; then
        ((SUCCESSFUL++))
    else
        ((FAILED++))
        FAILED_LANGUAGES+=("$lang")
    fi
done

echo ""
echo "=== EXTRACTION SUMMARY ==="
echo "Completed: $(date)"
echo "Successful: $SUCCESSFUL/$((SUCCESSFUL + FAILED))"
echo "Failed: $FAILED/$((SUCCESSFUL + FAILED))"

if [ ${#FAILED_LANGUAGES[@]} -gt 0 ]; then
    echo ""
    echo "Failed languages:"
    for lang in "${FAILED_LANGUAGES[@]}"; do
        echo "  - $lang"
    done
fi

echo ""
echo "All embedding files saved to: data_staging/embeddings/"
echo "Each language will have: train.hdf5, dev.hdf5, test.hdf5"
echo "Each file contains: embeddings + tokenization_map + input_ids"
echo ""
echo "Next steps:"
echo "1. Verify embeddings: ls data_staging/embeddings/"
echo "2. Check file sizes: du -sh data_staging/embeddings/*"
echo "3. Begin training: Use generated configs in configs/dataset/"

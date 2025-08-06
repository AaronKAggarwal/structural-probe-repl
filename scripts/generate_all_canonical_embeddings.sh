#!/bin/bash
set -e
echo "--- Generating All Canonical Embeddings ---"

# --- BERT-base-cased on PTB ---
echo -e "\n>> Generating BERT-base-cased embeddings for PTB-SD (all layers)..."
poetry run python scripts/extract_embeddings.py \
  dataset=ptb_sd/ptb_sd_official \
  model=bert-base-cased \
  runtime=mps \
  job.layers_to_extract='all'

# --- BERT-base-cased on UD EWT ---
echo -e "\n>> Generating BERT-base-cased embeddings for UD EWT (all layers)..."
poetry run python scripts/extract_embeddings.py \
  dataset=ud_ewt/ud_english_ewt_full \
  model=bert-base-cased \
  runtime=mps \
  job.layers_to_extract='all'

# --- ELMo on PTB ---
echo -e "\n>> Generating ELMo embeddings for PTB-SD..."
mkdir -p data_staging/embeddings/ptb_sd_official_splits/elmo/
./scripts/generate_elmo_embeddings_generic.sh \
  data_staging/ptb_stanford_dependencies_raw_text/ptb3-wsj-train.txt \
  data_staging/embeddings/ptb_sd_official_splits/elmo/ptb_sd_official_splits_train_layers-all_align-mean.hdf5
./scripts/generate_elmo_embeddings_generic.sh \
  data_staging/ptb_stanford_dependencies_raw_text/ptb3-wsj-dev.txt \
  data_staging/embeddings/ptb_sd_official_splits/elmo/ptb_sd_official_splits_dev_layers-all_align-mean.hdf5
./scripts/generate_elmo_embeddings_generic.sh \
  data_staging/ptb_stanford_dependencies_raw_text/ptb3-wsj-test.txt \
  data_staging/embeddings/ptb_sd_official_splits/elmo/ptb_sd_official_splits_test_layers-all_align-mean.hdf5

# --- ELMo on UD EWT ---
echo -e "\n>> Generating ELMo embeddings for UD EWT..."
mkdir -p data_staging/embeddings/ud_english_ewt_full/elmo/
./scripts/generate_elmo_embeddings_generic.sh \
  data_staging/ud_ewt_official_processed/txt_files/en_ewt-ud-train.txt \
  data_staging/embeddings/ud_english_ewt_full/elmo/ud_english_ewt_full_train_layers-all_align-mean.hdf5
./scripts/generate_elmo_embeddings_generic.sh \
  data_staging/ud_ewt_official_processed/txt_files/en_ewt-ud-dev.txt \
  data_staging/embeddings/ud_english_ewt_full/elmo/ud_english_ewt_full_dev_layers-all_align-mean.hdf5
./scripts/generate_elmo_embeddings_generic.sh \
  data_staging/ud_ewt_official_processed/txt_files/en_ewt-ud-test.txt \
  data_staging/embeddings/ud_english_ewt_full/elmo/ud_english_ewt_full_test_layers-all_align-mean.hdf5

echo -e "\n--- All Canonical Embeddings Generation Complete ---"
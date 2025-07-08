# Repository Architecture

Last updated: 2025-06-17 <!-- Updated date -->

This document explains the layout of the `structural-probe-repl` project, covering the vendored original Hewitt & Manning (2019) codebase and the new, modern PyTorch implementation.

## 1. Original Hewitt & Manning Code (Vendored)

*   **Location:** `src/legacy/structural_probe/`
*   **Description:** A direct copy of the original codebase from `john-hewitt/structural-probes`. This code is PyTorch-based (~v1.0-1.3) and was used in Phase 0 to verify the original probing pipeline on sample data.
*   **Key Subdirectories & Files:**
    *   `example/config/`: YAML configurations for H&M's experiments (e.g., `prd_en_ewt-ud-sample.yaml`, PTB configs in `naacl19/`).
    *   `example/data/`: Sample EWT data and pre-trained BERT probe parameters sourced from the `whykay-01` fork.
    *   `structural-probes/`: The core Python modules of the H&M probe (`data.py`, `model.py`, `probe.py`, `run_experiment.py`, etc.).

## 2. Modern Probe Implementation & Project Scaffold

This section details the structure of the current project, which uses a modern PyTorch re-implementation for all new experiments.

*   **`configs/`**: Hydra configuration files for the modern probing framework.
    *   `config.yaml`: Main default configuration.
    *   `config_extract.yaml`: Main configuration for `scripts/extract_embeddings.py`.
    *   `dataset/`: Configs defining datasets. Now includes both `ptb_sd_official.yaml` (for H&M replication) and **`ud_english_ewt_full.yaml`** for the project's primary experiments.
    *   `embeddings/`: Configs for pre-computed embeddings. Includes configs for both PTB-based HDF5s (e.g., `bert_base_L7_ptb_sd.yaml`) and UD-based HDF5s (e.g., **`elmo_l1_ud_ewt_full.yaml`**).
    *   `experiment/`: Composable experiment configurations, including `hm_replication/` for PTB and **`ud_replication/`** (or similar) for UD baselines.
    *   `probe/`: Configs for different probe types and ranks (e.g., `distance_rank128.yaml`).
    *   `training/`: Configs for training parameters (e.g., `training_hm_replication.yaml`).
    *   `evaluation/`: Configs for evaluation settings (e.g., `eval_hm_metrics.yaml`).
    *   <!-- Added this new directory -->
    *   `extraction/`: Dedicated, single-use configs for large embedding extraction jobs (e.g., **`bert_base_cased_ud_ewt_all_layers.yaml`**).

*   **`data/`**: (Gitignored by default)
    *   Location for storing primary datasets like the **full Universal Dependencies (UD) treebanks** and the Penn Treebank (PTB).
    *   Example Subdirectory: `data/ud_english_ewt_official/` containing the `en_ewt-ud-*.conllu` files.

*   **`data_staging/`**: (Gitignored)
    *   Local staging area for downloading full datasets and storing large intermediate files.
    *   `ptb_stanford_dependencies_conllx/`: Stores the CoNLL-X files generated from PTB `.mrg` files for H&M replication.
    *   `ud_ewt_official_processed/elmo_hdf5_layers/`: Example directory for storing HDF5 files generated from UD data.
    *   `embeddings_ptb_sd/`: Stores HDF5 files of embeddings extracted from models on the PTB-SD data.

*   **`docs/`**: All project documentation. See `docs/DOC_INDEX.md` for a full map.

*   **`env/`**: Dockerfiles for containerized environments.
    *   `Dockerfile.legacy_pt_cpu`: Defines the environment for running the original H&M code.

*   **`outputs/` & `outputs_extract_embeddings/`**: (Gitignored) Default root directories where Hydra saves outputs for each run.

*   **`data_processing_scripts/`**: Shell scripts for data preparation.
    *   `ptb_to_conllx.sh`: Script to convert PTB `.mrg` constituency parses to Stanford Dependencies (CoNLL-X format). Used for H&M replication.
    *   <!-- Added these generic scripts -->
    *   `convert_conllu_to_raw_generic.py`: Generic script to convert any CoNLL-U/X file to raw text, needed for legacy ELMo embedding generation.
    *   `generate_elmo_embeddings_generic.sh`: Generic script to generate legacy-style ELMo HDF5s for any raw text file.

*   **`scripts/`**: Executable Python and shell scripts for the project.
    *   **Legacy Support:**
        *   `run_legacy_probe.sh`: Wrapper to run H&M's `run_experiment.py` or `run_demo.py` inside the legacy container.
    *   **Modern Probe Pipeline:**
        *   `extract_embeddings.py`: Main Hydra-configurable script for extracting word-aligned hidden state embeddings from Hugging Face Transformer models for any given CoNLL-X/U dataset.
        *   `train_probe.py`: Main Hydra-configurable script for training and evaluating modern structural probes.

*   **`src/`**: Source code for the project.
    *   `legacy/structural_probe/`: Contains the vendored original Hewitt & Manning codebase.
    *   `torch_probe/`: Houses the modern PyTorch (v2.x) re-implementation.
        *   `utils/`: Utility modules for the modern probe.
            *   `conllu_reader.py`: Parses CoNLL-U and CoNLL-X files, extracting tokens, heads, dependency relations, UPOS, and **XPOS tags**.
            *   `gold_labels.py`: Computes gold tree depths and pairwise distances from head indices.
            *   `embedding_loader.py`: Loads pre-computed embeddings from HDF5 files.
        *   `dataset.py`: Defines `ProbeDataset` and the `collate_probe_batch` function.
        *   `probe_models.py`: Defines `DistanceProbe` and `DepthProbe` as `nn.Module`s.
        *   `loss_functions.py`: Custom L1 loss functions aligned with H&M's methodology.
        *   `evaluate.py`: Functions for calculating metrics (Spearman, UUAS, Root Accuracy). **Crucially, uses H&M's punctuation filtering based on XPOS tags**, which works for both PTB-SD and UD English EWT.
        *   `train_utils.py`: Helpers for training (optimizer, `LRSchedulerWithOptimizerReset`, `EarlyStopper`, checkpointing).

*   **`tests/`**: Contains all tests for the project.
    *   `smoke/`: Integration tests for the full `train_probe.py` pipeline.
    *   `unit/torch_probe/`: Unit tests for individual modules in `src/torch_probe/`.

*   **Root Directory Files:**
    *   `README.md`: Main project entry point.
    *   `pyproject.toml`, `poetry.lock`: Poetry dependency and project management.
    *   `.gitignore`: Specifies intentionally untracked files.
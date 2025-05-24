# Repository Architecture

Last updated: 2025-05-07

This document explains the layout of both the original Hewitt & Manning structural probe code (vendored into this project) and the new project structure.

## 1. Original Hewitt & Manning Code (`src/legacy/structural_probe`)

This directory contains a copy of the original codebase from `john-hewitt/structural-probes`. It is PyTorch-based (approx. v1.0-1.3).

-   **README.md / LICENSE:** Original project's overview and license.
-   **doc-assets/:** Figures (PNG) used in the original paper/README.
-   **download_example.sh:** Script to fetch a small sample of the English EWT Universal Dependencies corpus and pre-trained probe parameters. **Note: The URLs in this script are currently dead (404). This script is NOT used in the current Docker build.**
-   **example/:**
    -   **config/:** YAML configuration files for experiments on various models (e.g., ELMo, BERT-base) and tasks (parse-distance `prd`, parse-depth `pad`). Includes subdirectories like `naacl19/` for paper-specific configs.
    -   **data/:** Original intended location for example datasets (e.g., `en_ewt-ud-sample/`) including `.conllu` files and pre-computed embeddings (e.g., `.elmo-layers.hdf5`). Our prepared sample data is now copied here during Docker build.
    -   **demo-bert.yaml:** An end-to-end demo configuration, likely using pre-trained probe parameters (which are also from dead links).
-   **requirements.txt:** Python package dependencies for the original code (e.g., `Cython`, `seaborn`, `PyYAML`, `numpy`, `h5py`, `tqdm`). PyTorch itself and `pytorch-pretrained-bert` were to be installed separately according to their README.
-   **scripts/:** Original data preparation utilities:
    -   `convert_conll_to_raw.py`
    -   `convert_raw_to_bert.py`
    -   `convert_raw_to_elmo.sh`
    -   `convert_splits_to_depparse.sh` (uses Stanford CoreNLP)
-   **structural-probes/:** Core probe implementation and orchestration:
    -   `probe.py`: Core structural probe logic.
    -   `model.py`, `data.py`: Model loading, data handling (CoNLLU, HDF5 embeddings). **Note: `data.py` has been locally modified to correctly handle CoNLLU MWTs for token counting.**
    -   `run_experiment.py`, `run_demo.py`: Main drivers for experiments and demos.
    -   `loss.py`, `regimen.py`, `reporter.py`, `task.py`: Training loop, loss functions, reporting, and task definitions.

## 2. Current Project Scaffold (`structural-probe-repl/`)

-   **`src/`:**
    -   **`legacy/structural_probe/`:** Contains the (slightly modified) Hewitt & Manning codebase...
    -   **`torch_probe/`:** Houses the modern PyTorch (v2.x) re-implementation of the structural probe.
    -   `utils/`: Contains utility modules for the modern probe.
    -   `conllu_reader.py`: Parses CoNLL-U files, extracts tokens, head indices, and other annotations, correctly handling multi-word tokens.
    -   `gold_labels.py`: Computes gold standard tree depths and pairwise distances from head index information.
    -   `embedding_loader.py`: Provides functions to efficiently load sentence-specific pre-computed embeddings (e.g., ELMo) from HDF5 files, allowing for layer selection.
    -   `dataset.py`: Contains the `ProbeDataset` PyTorch `Dataset` class for loading CoNLL-U parses and corresponding embeddings, and the custom `collate_probe_batch` function for padding and batching variable-length sequences.
    -   `probe_models.py` (New for MS1.2): Defines the PyTorch `nn.Module` classes for the structural probes:
    -   `DistanceProbe`: Implements the linear transformation and calculates squared L2 distances between projected embedding pairs.
    -   `DepthProbe`: Implements the linear transformation and calculates the squared L2 norm of projected embeddings.
    -   `loss_functions.py` (New for MS1.2): Provides custom L1 loss functions tailored for the probing tasks:
    -   `evaluate.py` (New for MS1.3): Contains functions for calculating evaluation metrics (Spearman correlation, UUAS, Root Accuracy), including logic for punctuation filtering to align with H&M methodology.
    -   `train_utils.py` (New for MS1.3): Provides utility functions for the training process, such as optimizer instantiation, early stopping mechanisms, and model checkpointing (saving/loading).
    -   `distance_l1_loss`: Calculates L1 loss on squared distances, correctly handling padding and considering unique pairs.
    -   `depth_l1_loss`: Calculates L1 loss on squared depths, correctly handling padding.     
    -   *(Future: Main training script will reside in `scripts/`)*
    -   **`common/`:** *(To be created)* ...
-   **`env/`:**
    -   **`Dockerfile.legacy_pt_cpu`:** Dockerfile to build an environment for running the original Hewitt & Manning code (Python 3.7, PyTorch 1.3.0+cpu, AllenNLP 0.9.0, etc.) on `linux/amd64`. Includes prepared sample data.
    -   *(Future: `Dockerfile.cuda` for modern LLM experiments on remote GPUs).*
-   **`scripts/`:**
    -   **`check_legacy_env.sh`:** Health check script for the `probe:legacy_pt_cpu` Docker container.
    -   **`run_legacy_probe.sh`:** Wrapper script to execute `run_experiment.py` from the legacy code within its Docker container.
    -   **`create_conllu_sample.py`:** Script to generate small sample CoNLLU files from full UD EWT data.
    -   **`convert_sample_conllu_to_raw.py`:** Script to convert sample CoNLLU to raw text for ELMo input.
    -   **`generate_elmo_embeddings_for_sample.sh`:** Script to generate ELMo HDF5 embeddings for the sample data using AllenNLP in Docker.
    -   *(Future: Scripts for PTB preprocessing, running new experiments, etc.)*
-   **`data_staging/`:** (Gitignored) Local staging area for downloading full datasets and preparing sample data before it's copied into Docker images or processed.
    -   `ud_ewt_full/`: For downloaded full UD EWT CoNLLU files.
    -   `my_ewt_sample_for_legacy_probe/`: Contains the prepared sample CoNLLU, TXT, and HDF5 files used by the legacy probe container.
-   **`results_staging/`:** (Gitignored) Local directory for mounting and inspecting results generated by Docker container runs.
-   **`data/`:** *(To be created/populated)* For storing primary datasets like PTB, processed versions, and generated embeddings intended for direct project use.
-   **`tests/`:** *(To be created)* For unit tests, integration tests, and smoke tests.
-   **`paper/`:** *(To be created)* For LaTeX source, figures, and bibliography for any publications.
-   **`notebooks/`:** *(To be created, optional)* For exploratory data analysis (EDA) and plotting.

## 3. Documentation (`docs/`)

This directory houses all project documentation. Key files include:
-   `README.md` (Project root): Overall project summary and entry point.
-   `ENV_SETUP.md`: Instructions for setting up native macOS (MPS) and Dockerized (legacy CPU, future CUDA) environments.
-   `DEPENDENCIES.md`: Notes on key dependencies and version constraints for both native and containerized environments.
-   `DOC_INDEX.md`: Master index of all documentation files.
-   `ARCHITECTURE.md`: *This file*, describing the project's structure.
-   `HISTORY.md`: Chronological log of build/debug milestones and resolutions.
-   `QUIRKS.md`: Lists non-obvious issues, surprises, and workarounds encountered.
-   *(Future: More detailed docs on specific components as they are developed).*
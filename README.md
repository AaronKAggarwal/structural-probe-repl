# Structural Probe Replication & Extensions

This project aims to replicate the methodology developed in **Hewitt & Manning (2019), "A Structural Probe for Finding Syntax in Word Representations,"** and subsequently extend this work to analyze syntactic encoding in modern Large Language Models (LLMs).

The original paper introduced a method for identifying syntactic structure in language model representations by finding a linear transformation under which squared L2 distances/norms encode parsing relationships.

## Project Goals

1.  **Replicate:** Faithfully reproduce the Hewitt & Manning probing methodology, initially by running their original PyTorch code in a containerized legacy environment and then by re-implementing the probe in a modern PyTorch framework.
2.  **Verify:** Validate the modern probe implementation against results from H&M's original work on the Penn Treebank (PTB) dataset.
3.  **Extend:** Apply the modern structural probe to a range of contemporary LLMs (from Hugging Face) using the **Universal Dependencies (UD)** treebanks to investigate how these newer models encode syntax.
4.  **Explore:** Investigate novel research directions based on the findings, potentially including different probe types, analysis of model training stages, or mechanistic interpretability approaches.

## Current Status (as of 2025-08-06)

*   **Phase 0: Environment Setup & Legacy Probe Replication - COMPLETE**
    *   **Native macOS Development Environment (PyTorch 2.x, MPS):** Successfully set up using Python 3.11 and Poetry. Core dependencies installed and MPS functionality verified. (See `docs/ENV_SETUP.md`)
    *   **Legacy Hewitt & Manning Probe Replication (PyTorch 1.x CPU):**
        *   Original H&M PyTorch-based code (from `john-hewitt/structural-probes`) vendored into `src/legacy/structural_probe/`.
        *   A Docker environment (`probe:legacy_pt_cpu` image) has been successfully built to run this legacy code.
        *   Original example data (CoNLLU, ELMo HDF5) and pre-trained BERT probe parameters were sourced from the `whykay-01/structural-probes` GitHub fork, as original download links were dead. This data is included in the Docker image.
        *   The legacy code successfully runs both the **ELMo training example** (on the sourced EWT sample) and the **BERT demo** (with pre-trained probes) end-to-end within the container.

*   **Phase 1: Modern PyTorch Probe Re-implementation - COMPLETE**
    *   **Core Components:** Implemented and unit-tested a full, modern probing pipeline in `src/torch_probe/`, including data utilities, PyTorch `Dataset`s, `DistanceProbe` and `DepthProbe` models, H&M-aligned loss functions, and evaluation metrics (Spearman, UUAS, Root Accuracy with punctuation filtering).
    *   **Training Pipeline:** Implemented `scripts/train_probe.py` with Hydra integration, H&M-style optimizer reset, and granular checkpointing. Smoke tests for the full pipeline pass.
    *   Details in `docs/ARCHITECTURE.md` and `docs/HISTORY.md`.

*   **Phase 2: Pipeline Validation & Dataset Pivot - COMPLETE** 

    *   **PTB-SD Replication (Methodology Validation):**
        *   Removed.
    *   **Pivot to Universal Dependencies (UD):**
        *   The project's primary dataset for extension has been shifted from PTB-SD to the **Universal Dependencies English Web Treebank (UD EWT)** to align with modern NLP standards and facilitate cross-linguistic work.
        *   All data loading and evaluation code has been verified for compatibility with the UD CoNLL-U format.
    *   **UD Baseline Establishment:**
        *   Successfully ran the validated modern pipeline to establish strong baselines on UD EWT for key H&M models:
            *   **ELMo (Distance Probe):** Layer 1 (UUAS: 0.72, Spearmanr: 0.71), Layer 2 (UUAS: 0.66, Spearmanr: 0.68), Layer 0 (UUAS: 0.32, Spearmanr: 0.28) on UD EWT.
            *   **BERT-base Layer 7 (Distance Probe):** Test UUAS of 0.800 on UD EWT (Spearmanr of 0.77).

*   **Phase 3: Systematic Probing of Modern LLMs - IN PROGRESS**
    *   Successfully extracted embeddings for all layers of `meta-llama/Llama-3.2-3B`, `meta-llama/Llama-3.2-3B-Instruct`, and `bert-base-multilingual-cased` on the UD EWT and UD HDTB datasets.
    *   Completed full depth and distance probe sweeps for all models.
    *   **Key Finding:** Instruction tuning preserves the strength of syntactic encoding in Llama-3.2-3B but shifts its location to deeper layers in the network. See `docs/KEY_RESULTS.md` for details.
*   **Next Major Phase: Analysis & Paper Drafting**
    *   Analyze and visualize the syntactic encoding patterns in Llama-3.2 models versus mBERT baselines.
    *   Begin drafting research paper based on key findings.

## Repository Structure Overview

*   **`configs/`**: Hydra configuration files for experiments. Structured into composable groups like `dataset`, `embeddings`, and `probe`. Experiment-specific files in `configs/experiment/` combine these components.
*   **`data_staging/`**: (Gitignored) Local area for downloading/preparing raw datasets.
*   **`docs/`**: All project documentation (see `docs/DOC_INDEX.md` for a full list).
*   **`env/`**: Dockerfiles for various environments (e.g., `Dockerfile.legacy_pt_cpu`).
*   **`outputs/`**: (Gitignored) Default output directory for Hydra runs.
*   **`results_staging/`**: (Gitignored) Local area for inspecting results from container runs.
*   **`scripts/`**: Executable scripts (data preparation, training, evaluation).
    *   `scripts/extract_embeddings.py`: Generic script to extract embeddings for any Hugging Face model.
    *   `scripts/train_probe.py`: Main Hydra-configurable script for training and evaluating probes.
    *   `scripts/smoke_tests/`: A suite of fast, modular scripts to validate all experiment configurations.
*   **`src/`**: Source code for the project.
    *   `src/legacy/structural_probe/`: Vendored original Hewitt & Manning codebase.
    *   `src/torch_probe/`: Modern PyTorch re-implementation of the probe and utilities.
*   **`tests/`**: Unit tests and smoke tests.
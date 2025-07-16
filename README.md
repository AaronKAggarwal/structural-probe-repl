# Structural Probe Replication & Extensions

This project aims to replicate the methodology developed in **Hewitt & Manning (2019), "A Structural Probe for Finding Syntax in Word Representations,"** and subsequently extend this work to analyze syntactic encoding in modern Large Language Models (LLMs).

The original paper introduced a method for identifying syntactic structure in language model representations by finding a linear transformation under which squared L2 distances/norms encode parsing relationships.

## Project Goals

1.  **Replicate:** Faithfully reproduce the Hewitt & Manning probing methodology, initially by running their original PyTorch code in a containerized legacy environment and then by re-implementing the probe in a modern PyTorch framework.
2.  **Verify:** Validate the modern probe implementation against results from H&M's original work on the Penn Treebank (PTB) dataset.
3.  **Extend:** Apply the modern structural probe to a range of contemporary LLMs (from Hugging Face) using the **Universal Dependencies (UD)** treebanks to investigate how these newer models encode syntax.
4.  **Explore:** Investigate novel research directions based on the findings, potentially including different probe types, analysis of model training stages, or mechanistic interpretability approaches.

## Current Status (as of 2025-07-16)

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
        *   Successfully processed the Penn Treebank (PTB) into Stanford Dependencies (CoNLL-X) using `data_processing_scripts/ptb_to_conllx.sh`.
        *   Developed `scripts/extract_embeddings.py` for extracting word-aligned embeddings from Hugging Face models.
        *   **Completed a full replication of Hewitt & Manning (2019, Table 1)** for the BERT-base Layer 7 Distance Probe on PTB-SD, achieving highly consistent UUAS and DSpr metrics. **This validates the correctness of the modern pipeline.**
    *   **Pivot to Universal Dependencies (UD):**
        *   The project's primary dataset for extension has been shifted from PTB-SD to the **Universal Dependencies English Web Treebank (UD EWT)** to align with modern NLP standards and facilitate cross-linguistic work.
        *   All data loading and evaluation code has been verified for compatibility with the UD CoNLL-U format.
    *   **UD Baseline Establishment:**
        *   Successfully ran the validated modern pipeline to establish strong baselines on UD EWT for key H&M models:
            *   **ELMo (Distance Probe):** Layer 1 (UUAS: 0.72), Layer 2 (UUAS: 0.65), Layer 0 (UUAS: 0.32).
            *   **BERT-base Layer 7 (Distance Probe):** UUAS ~0.80.

*   **Next Major Phase: Systematic Probing of Modern LLMs on Universal Dependencies**
    *   Replicate depth probe baselines for ELMo and BERT on UD EWT.
    *   Extract embeddings and run distance/depth probes for a selection of modern LLMs (e.g., Llama-3, Mistral) on UD EWT.
    *   Analyze and compare syntactic encoding across different model architectures and scales.

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

## Getting Started

1.  **Understand the Project:** Start by reading this README and then:
    *   `docs/PROJECT_OVERVIEW.md`: For the research context and goals.
    *   `docs/ROADMAP.md`: For the detailed project plan and phase descriptions.
    *   Original Paper: [Hewitt & Manning (2019), A Structural Probe for Finding Syntax in Word Representations](https://www.aclweb.org/anthology/N19-1042/)
2.  **Set up Environment:** Follow the instructions in `docs/ENV_SETUP.md` to set up either the native macOS environment or the Dockerized legacy environment.
3.  **Explore the Code:** Refer to `docs/ARCHITECTURE.md` for a guide to the codebase.
4.  **Verify Configurations:** Run the smoke test suite to ensure all experiment setups are valid before launching a full run: `./scripts/test_all_configs.sh`.
5.  **Run Experiments (Modern Probe):** See `docs/EXPERIMENT_PROTOCOL.md` for instructions on using `scripts/train_probe.py` with Hydra.
6.  **Run Legacy Examples:** See `docs/ENV_SETUP.md` (Section 2) for running the legacy probe examples within Docker.

## Key Documentation Files

*   **`docs/DOC_INDEX.md`**: Master list of all documentation.
*   **`docs/HISTORY.md`**: Chronological log of development and debugging.
*   **`docs/QUIRKS.md`**: Important notes on non-obvious issues and workarounds.

## License

This project's custom code (e.g., in `src/torch_probe/`, `scripts/`) is licensed under the [YOUR CHOSEN LICENSE, e.g., MIT License]. The vendored code in `src/legacy/structural_probe/` is subject to its original license (see `src/legacy/structural_probe/LICENSE`).

---
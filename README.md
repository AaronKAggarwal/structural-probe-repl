# Structural Probe Replication & Extensions

This project aims to replicate the methodology developed in **Hewitt & Manning (2019), "A Structural Probe for Finding Syntax in Word Representations,"** and subsequently extend this work to analyze syntactic encoding in modern Large Language Models (LLMs).

The original paper introduced a method for identifying syntactic structure in language model representations by finding a linear transformation under which squared L2 distances/norms encode parsing relationships.

## Project Goals

1.  **Replicate:** Faithfully reproduce the Hewitt & Manning probing methodology, initially by running their original PyTorch code in a containerized legacy environment and then by re-implementing the probe in a modern PyTorch framework.
2.  **Verify:** Validate the modern probe implementation against results from the legacy codebase using sample data.
3.  **Extend:** Apply the modern structural probe to a range of contemporary LLMs (from Hugging Face) using the Penn Treebank (PTB) to investigate how these newer models encode syntax.
4.  **Explore:** Investigate novel research directions based on the findings, potentially including different probe types, analysis of model training stages, or mechanistic interpretability approaches.

## Current Status (as of 2025-05-26)

*   **Phase 0: Environment Setup & Legacy Probe Replication - COMPLETE**
    *   **Native macOS Development Environment (PyTorch 2.x, MPS):** Successfully set up using Python 3.11 and Poetry. Core dependencies installed and MPS functionality verified. (See `docs/ENV_SETUP.md`)
    *   **Legacy Hewitt & Manning Probe Replication (PyTorch 1.x CPU):**
        *   Original H&M PyTorch-based code (from `john-hewitt/structural-probes`) vendored into `src/legacy/structural_probe/`.
        *   A Docker environment (`probe:legacy_pt_cpu` image) has been successfully built to run this legacy code.
        *   Original example data (CoNLLU, ELMo HDF5) and pre-trained BERT probe parameters were sourced from the `whykay-01/structural-probes` GitHub fork, as original download links were dead. This data is included in the Docker image.
        *   The legacy code successfully runs both the **ELMo training example** (on the sourced EWT sample) and the **BERT demo** (with pre-trained probes) end-to-end within the container, producing plausible metrics and visualizations.
        *   Challenges and fixes are documented in `docs/HISTORY.md` and `docs/QUIRKS.md`.

*   **Phase 1: Modern PyTorch Probe Re-implementation - COMPLETE**
    *   **Core Components:**
        *   Data utilities (`conllu_reader`, `gold_labels`, `embedding_loader`) implemented and unit-tested.
        *   PyTorch `ProbeDataset` and `collate_fn` for data loading/batching implemented and unit-tested.
        *   `DistanceProbe` and `DepthProbe` PyTorch `nn.Module`s implemented and unit-tested.
        *   L1 loss functions (aligned with H&M methodology) implemented and unit-tested.
        *   Training utilities (`get_optimizer`, `EarlyStopper`, checkpointing) and evaluation metrics (Spearman, UUAS, RootAcc with punctuation filtering) implemented and unit-tested.
    *   **Training Pipeline:**
        *   Main training script (`scripts/train_probe.py`) with Hydra integration implemented.
        *   Smoke test for the full modern training pipeline passes.
    *   **Validation:**
        *   Successfully trained and evaluated modern distance and depth probes on an ELMo sample data (using self-generated, MWT-filtered aligned HDF5s), producing plausible metrics and demonstrating functionality on MPS. Qualitative parity with Phase 0a legacy code behavior established.
    *   Details in `docs/ARCHITECTURE.md` and `docs/HISTORY.md`.

*   **Next Major Phase:** **Phase 2 - Data Preparation for Modern LLMs (PTB & Hidden State Extraction)**
    *   Acquiring and preprocessing the Penn Treebank.
    *   Selecting target modern LLMs.
    *   Developing scripts for hidden state extraction and subword-to-word alignment.
    *   Setting up a CUDA Docker environment for larger models.

## Repository Structure Overview

*   **`configs/`**: Hydra configuration files for experiments.
*   **`data_staging/`**: (Gitignored) Local area for downloading/preparing raw datasets.
*   **`docs/`**: All project documentation (see `docs/DOC_INDEX.md` for a full list).
*   **`env/`**: Dockerfiles for various environments (e.g., `Dockerfile.legacy_pt_cpu`).
*   **`outputs/`**: (Gitignored) Default output directory for Hydra runs.
*   **`results_staging/`**: (Gitignored) Local area for inspecting results from container runs.
*   **`scripts/`**: Executable scripts (data preparation, training, evaluation).
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
4.  **Run Experiments (Modern Probe):** See `docs/EXPERIMENT_PROTOCOL.md` for instructions on using `scripts/train_probe.py` with Hydra.
5.  **Run Legacy Examples:** See `docs/ENV_SETUP.md` (Section 2) for running the legacy probe examples within Docker.

## Key Documentation Files

*   **`docs/DOC_INDEX.md`**: Master list of all documentation.
*   **`docs/HISTORY.md`**: Chronological log of development and debugging.
*   **`docs/QUIRKS.md`**: Important notes on non-obvious issues and workarounds.

## License

This project's custom code (e.g., in `src/torch_probe/`, `scripts/`) is licensed under the [YOUR CHOSEN LICENSE, e.g., MIT License]. The vendored code in `src/legacy/structural_probe/` is subject to its original license (see `src/legacy/structural_probe/LICENSE`).

---
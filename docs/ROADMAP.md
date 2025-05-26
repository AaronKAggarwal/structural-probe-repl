# Project Roadmap: Replicating and Extending Structural Probes

Last updated: 2025-05-26

This document outlines the planned phases, key tasks, deliverables, and current status for the "Replicating and Extending Structural Probes for Finding Syntax in Word Representations" project.

## Overall Goal

To replicate the Hewitt & Manning (2019) structural probe methodology on a range of modern Hugging Face LLMs, analyze the findings, and then extend this work in novel directions.

## Guiding Principles

*   **Reproducibility:** All experiments and results should be reproducible through clear documentation, version-controlled code, containerization, and fixed dependencies/seeds where appropriate.
*   **Modularity:** Code and experimental setups will be designed in a modular fashion to facilitate ease of use, extension, and maintenance.
*   **Openness:** Where feasible, code and findings will be shared to contribute to the broader research community.
*   **Rigorous Documentation:** Comprehensive documentation will be maintained throughout the project lifecycle (see `docs/DOC_INDEX.md`).

---

## Phase 0: Environment Setup & Legacy Probe Replication

**Goal:** Establish foundational development environments and verify the original Hewitt & Manning (2019) probing pipeline using their legacy code and example data.

**Status: COMPLETE** (Commit SHA: `e54c18d` or your latest for Phase 1 completion)

*   **Sub-Phase 0.1: Native macOS Development Environment Setup**
    *   **Tasks:**
        1.  Install system prerequisites (Python 3.11 via Homebrew, Rust, pipx).
        2.  Install Poetry via pipx for isolated dependency management.
        3.  Initialize Poetry project (`pyproject.toml`, `poetry.lock`).
        4.  Install core native dependencies (PyTorch 2.2.x with MPS, Transformers, Datasets, NumPy<2.0, etc.).
        5.  Install `poetry-plugin-export` and generate `requirements-mps.txt`.
        6.  Verify MPS functionality for PyTorch.
        7.  Set up initial project documentation structure (`README.md`, `docs/DOC_INDEX.md`, `docs/ENV_SETUP.md`, `docs/DEPENDENCIES.md`, `docs/ARCHITECTURE.md`).
    *   **Deliverables:** Functional native macOS development environment; initial documentation.
    *   **Status: COMPLETE.**

*   **Sub-Phase 0.2: Legacy Hewitt & Manning Probe Replication (PyTorch 1.x CPU)**
    *   **Tasks:**
        1.  Vendor original H&M code (`john-hewitt/structural-probes`) into `src/legacy/structural_probe/`.
        2.  Analyze H&M code and confirm it's PyTorch-based (not TensorFlow).
        3.  Develop `env/Dockerfile.legacy_pt_cpu` (Base: `python:3.7-slim-buster`, PyTorch 1.3.0+cpu, AllenNLP 0.9.0, specific legacy dependencies like `PyYAML==3.13`, `overrides==3.1.0`).
        4.  Implement helper scripts (`scripts/check_legacy_env.sh`, `scripts/run_legacy_probe.sh`).
        5.  Identify dead download links in H&M's `download_example.sh`.
        6.  Source original H&M example data (CoNLLU, ELMo HDF5, BERT `.params`) from `whykay-01/structural-probes` GitHub fork and integrate into `src/legacy/structural_probe/example/data/`.
        7.  Update `Dockerfile.legacy_pt_cpu` to `COPY` this sourced data.
        8.  Ensure original H&M `data.py` is used (found to be consistent with sourced ELMo HDF5s).
        9.  Successfully build `probe:legacy_pt_cpu` Docker image for `linux/amd64`.
        10. Run H&M ELMo example (`prd_en_ewt-ud-sample.yaml` and `pad_en_ewt-ud-sample.yaml`) end-to-end on sourced sample data, verifying plausible metrics (UUAS, Spearman, Root Accuracy).
        11. Run H&M BERT demo (`demo-bert.yaml` with `run_demo.py`) using sourced pre-trained `.params` files, verifying visualization output.
        12. Document setup, challenges, and resolutions in `docs/HISTORY.md` and `docs/QUIRKS.md`.
    *   **Deliverables:** Working `probe:legacy_pt_cpu` Docker image; successful execution of H&M ELMo and BERT examples; understanding of legacy pipeline.
    *   **Status: COMPLETE.**

---

## Phase 1: Modern PyTorch Probe Re-implementation

**Goal:** Re-implement the Hewitt & Manning structural probe using modern PyTorch (v2.x) and best practices in the native macOS (MPS-accelerated) environment. Validate against the ELMo sample data.

**Status: COMPLETE** (Commit SHA: `e54c18d` or your latest for Phase 1 completion)

*   **Milestone 1.0: Project Setup & Basic Data Handling Utilities**
    *   **Tasks:** Implement and unit-test `src/torch_probe/utils/conllu_reader.py` (MWT-filtering) and `src/torch_probe/utils/gold_labels.py` (depths/distances).
    *   **Deliverables:** Tested utility modules.
    *   **Status: COMPLETE.**

*   **Milestone 1.1: PyTorch `Dataset` and `DataLoader` for ELMo Sample**
    *   **Tasks:** Implement and unit-test `src/torch_probe/utils/embedding_loader.py` (for HDF5), `src/torch_probe/dataset.py` (containing `ProbeDataset` and `collate_probe_batch`).
    *   **Deliverables:** Functional data loading pipeline for PyTorch.
    *   **Status: COMPLETE.**

*   **Milestone 1.2: Probe Model and Loss Function Implementation**
    *   **Tasks:** Implement and unit-test `src/torch_probe/probe_models.py` (`DistanceProbe`, `DepthProbe`) and `src/torch_probe/loss_functions.py` (L1 loss on pred_sq vs gold_nonsq, with H&M per-sentence normalization).
    *   **Deliverables:** Core probe and loss components.
    *   **Status: COMPLETE.**

*   **Milestone 1.3: Training and Evaluation Loop Implementation**
    *   **Tasks:** Implement and unit-test `src/torch_probe/train_utils.py` (optimizer, early stopping, checkpointing) and `src/torch_probe/evaluate.py` (Spearman, UUAS, RootAcc with punctuation filtering). Implement main training script `scripts/train_probe.py` with Hydra integration.
    *   **Deliverables:** Functional training script; initial Hydra configurations.
    *   **Status: COMPLETE.**

*   **Milestone 1.4: Full Validation on ELMo Sample & Qualitative Parity**
    *   **Tasks:** Run full training and evaluation of modern distance and depth probes on the ELMo sample data (using self-generated HDF5s aligned with modern MWT-filtering `conllu_reader.py`). Compare metrics qualitatively with Phase 0a legacy results.
    *   **Deliverables:** Validated modern probe implementation; comparison notes.
    *   **Status: COMPLETE.**

---

## Phase 2: Data Preparation for Modern LLMs (PTB & Hidden State Extraction)

**Goal:** Prepare the Penn Treebank (PTB) dataset and extract hidden state embeddings from target modern LLMs for this dataset.

**Status: PENDING (Current Phase)**

*   **Tasks:**
    1.  **Acquire Penn Treebank (PTB):** Secure LDC license and dataset files.
    2.  **Preprocess PTB:**
        *   Convert PTB constituency parses to dependency parses (e.g., Stanford Dependencies via CoreNLP) in CoNLL-U format.
        *   Define standard train/dev/test splits (e.g., WSJ sections).
        *   Document preprocessing steps meticulously.
    3.  **Select Initial Target Modern LLMs:** Finalize a list (e.g., Llama-3 8B, Mistral-7B, a RoBERTa variant). Document rationale.
    4.  **Develop Hidden State Extraction Script (`scripts/extract_embeddings.py`):**
        *   Utilize Hugging Face `transformers`.
        *   Handle model-specific tokenization.
        *   Implement robust subword-to-word alignment strategies (e.g., first, mean-pooling), making it configurable.
        *   Extract hidden states from specified layers ("all" or a list).
        *   Save word-aligned embeddings efficiently (e.g., HDF5, one file per model/split containing all layers, or per layer). Store metadata (model name, layer, alignment) in HDF5 attributes.
        *   Develop for MPS, plan for CUDA.
    5.  **Setup CUDA Docker Environment (`env/Dockerfile.cuda`):**
        *   Based on PyTorch CUDA image (matching native version).
        *   Include project dependencies via Poetry.
        *   Test basic CUDA functionality.
    6.  **Run Hidden State Extractions:**
        *   For smaller models on MPS.
        *   For larger models on remote GPU using the `probe:cuda` Docker image.
    7.  **Unit Tests for Extraction Utilities:** E.g., for alignment logic.
*   **Deliverables:**
    *   Processed PTB data in CoNLL-U format.
    *   Working and tested `scripts/extract_embeddings.py`.
    *   Collection of HDF5 files containing word-aligned hidden states for target LLMs on PTB.
    *   Buildable `probe:cuda` Docker image.
    *   Updated documentation (`DATA_PREP.md`, `ARCHITECTURE.md`, `HISTORY.md`).

---

## Phase 3: Probing Modern LLMs (Baseline Sweeps)

**Goal:** Apply the modern PyTorch probe (from Phase 1) to the extracted hidden states (from Phase 2) of modern LLMs using the PTB dataset.

**Status: PENDING**

*   **Tasks:**
    1.  Adapt modern probe's `ProbeDataset` to load PTB CoNLLU and the new HDF5 embedding formats (if different from ELMo HDF5 structure).
    2.  Update/Create Hydra configurations for experiments on PTB with different LLMs and layers.
    3.  For each target LLM:
        *   Train distance probes on embeddings from various layers.
        *   Train depth probes on embeddings from various layers.
    4.  Collect and systematically log performance metrics (UUAS, Spearman, Root Accuracy) per layer for each model (using W&B or local logging).
    5.  Generate "U-curve" plots (metric vs. layer) for each model and metric.
    6.  Compare findings across models and with original H&M results for BERT. Analyze how syntax is encoded.
    7.  (Optional) Implement H&M's specific "5-50 sentence length" averaging for Spearman for closer paper comparison.
*   **Deliverables:**
    *   Comprehensive set of probing results for selected modern LLMs on PTB.
    *   Visualizations (U-curves, example parse tree predictions).
    *   Initial analysis and interpretation of results.
    *   Draft sections for a potential paper (Methods, Results, Discussion).

---

## Phase 4: Extensions & Novel Research

**Goal:** Explore advanced research questions and extend the probing methodology based on findings from Phase 3.

**Status: PENDING**

*   **Potential Tasks (Select 1-3 based on interest and Phase 3 results):**
    1.  **Non-Linear Probes:** Implement and test simple MLP probes.
    2.  **Models at Different Training Stages:** Compare base pre-trained vs. SFT/RLHF aligned models.
    3.  **Mechanistic Interpretability Links:**
        *   Train SAEs on relevant layers and map probe subspace to SAE features.
        *   Perform causal tracing/activation patching on probe-identified subspaces or SAE features.
    4.  **Effect of Model Scale/Architecture:** More in-depth comparisons if not fully covered in Phase 3.
    5.  **Probing Specific Syntactic Phenomena:** Design probes or analyze probe behavior for phenomena like long-distance dependencies, agreement, argument structure.
    6.  Cross-lingual Probing (if project scope expands).
*   **Deliverables:**
    *   Results and analysis for chosen extension(s).
    *   Further draft material for paper/report.

---

## Phase 5: Paper Assembly, Finalization, and Reproducibility Package

**Goal:** Produce a high-quality research paper (or report) and ensure the project is fully reproducible by the wider community.

**Status: PENDING**

*   **Tasks:**
    1.  Write/Complete all sections of the research paper.
    2.  Generate final figures, tables, and statistical analyses.
    3.  Thoroughly review and finalize all project documentation in `docs/`.
    4.  Create or refine a `REPRODUCE.md` guide and/or `reproduce.sh` script to replicate key findings (e.g., training a specific probe on a specific model/layer).
    5.  Clean up codebase, ensure all tests pass, add more integration tests if needed.
    6.  Prepare for public release of code, configurations, and potentially some derived data (e.g., probe parameters, aggregated results) on GitHub.
*   **Deliverables:**
    *   Completed research paper draft (for submission, preprint, or internal report).
    *   Public GitHub repository with fully reproducible code and documentation.

---

This roadmap provides a comprehensive overview. Specific tasks and timelines within each phase will be refined as the project progresses.
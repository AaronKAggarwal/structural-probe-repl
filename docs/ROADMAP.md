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

## Phase 2: PTB Data Processing & Hewitt & Manning Replication Baselines

**Goal:** Prepare the Penn Treebank (PTB) dataset according to H&M's methodology and perform full replication runs for key H&M results (e.g., BERT-base Layer 7) to establish strong baselines.

**Status: COMPLETE** (as of 2025-05-28)

*   **Tasks:**
    1.  **Acquire Penn Treebank (PTB):** Secure LDC license and dataset files. *(Status: Provisional local copy obtained, official license processing underway).*
    2.  **Set up Stanford CoreNLP 3.9.2:** Download and configure the required older version of CoreNLP. *(Status: COMPLETE).*
    3.  **PTB to Stanford Dependencies Conversion:**
        *   Implement/adapt script (`data_processing_scripts/ptb_to_conllx.sh`) based on H&M's original to convert PTB `.mrg` constituency parses to Stanford Dependencies (CoNLL-X format) for standard train/dev/test splits. *(Status: COMPLETE).*
        *   Generate `ptb3-wsj-train.conllx`, `ptb3-wsj-dev.conllx`, `ptb3-wsj-test.conllx`. *(Status: COMPLETE).*
    4.  **Develop Hidden State Extraction Script (`scripts/extract_embeddings.py`):**
        *   Utilize Hugging Face `transformers` for model loading and tokenization.
        *   Implement robust subword-to-word alignment (mean pooling via `word_ids()`).
        *   Handle layer selection and save word-aligned embeddings to HDF5. *(Status: COMPLETE).*
    5.  **Extract Embeddings for H&M Baseline (BERT-base):**
        *   Run `scripts/extract_embeddings.py` for `bert-base-cased` (all layers) on the full PTB-SD train/dev/test splits. *(Status: COMPLETE).*
    6.  **Perform Full H&M Replication Run (BERT-base L7 Distance Probe):**
        *   Configure Hydra experiment using H&M parameters (`training_hm_replication.yaml`, full-rank probe, etc.).
        *   Run `scripts/train_probe.py` on full PTB-SD data with BERT-base L7 embeddings.
        *   Collect metrics (UUAS, DSpr H&M-style) and compare with H&M (2019, Table 1). *(Status: COMPLETE - Results highly consistent with H&M).*
    7.  **Unit Tests and Smoke Tests:** Ensure all tests pass after refactoring and new feature additions. *(Status: COMPLETE).*
    8.  *(PENDING/OPTIONAL)* Extract embeddings and run probes for other H&M baselines (e.g., BERT-base L7 Depth, BERT-Large L16 Distance/Depth, ELMo L1 Distance/Depth) for more comprehensive baseline data.
    9.  *(PENDING)* Setup CUDA Docker Environment (`env/Dockerfile.cuda`) for efficient processing of larger models.

*   **Deliverables:**
    *   Processed PTB data in H&M-aligned CoNLL-X Stanford Dependency format.
    *   Working and validated `scripts/extract_embeddings.py`.
    *   HDF5 files containing word-aligned embeddings for `bert-base-cased` on PTB-SD.
    *   Successful replication of BERT-base L7 Distance probe results, matching H&M figures.
    *   Fully updated documentation (`ARCHITECTURE.md`, `HISTORY.md`, `EXPERIMENT_PROTOCOL.md`).

---

## Phase 3: Systematic Probing of Modern LLMs

**Goal:** Apply the validated modern PyTorch probe to a diverse set of recent Hugging Face LLMs using the prepared PTB-SD dataset.

**Status: PENDING (Next Phase)**

*   **Tasks:**
    1.  **Finalize Target Modern LLMs:** Select a diverse set (e.g., Llama-3 8B, Mistral-7B, a RoBERTa variant, potentially a larger model if compute allows). Document rationale.
    2.  **Optimize Training Pipeline (Optional but Recommended):** Implement `eval_on_train_every_n_epochs` in `train_probe.py` to reduce epoch times.
    3.  **Extract Embeddings for Modern LLMs:** Run `scripts/extract_embeddings.py` for all selected models and layers on the PTB-SD data. This will be computationally intensive.
    4.  **Prepare Hydra Configurations:** Create `embeddings`, `probe` (adjusting rank for new model dimensions), and `experiment` configs for each modern LLM experiment.
    5.  **Run Probing Experiments:** For each target LLM/layer:
        *   Train distance probes.
        *   Train depth probes.
    6.  **Collect & Analyze Results:**
        *   Systematically log performance metrics (UUAS, DSpr H&M, RootAcc) per layer for each model via W&B.
        *   Generate "U-curve" plots (metric vs. layer) for each model.
        *   Compare findings across modern models and against the BERT/ELMo baselines. Analyze how different architectures and model scales encode syntax.
*   **Deliverables:**
    *   HDF5 embedding sets for selected modern LLMs on PTB-SD.
    *   Comprehensive set of probing results (metrics, U-curves) for these models.
    *   Initial analysis and interpretation comparing modern LLMs to earlier ones.
    *   Draft sections for research paper (Methods updates, new Results, initial Discussion).

---

## Phase 4: Extensions & Novel Research

**Goal:** Explore advanced research questions and extend the probing methodology based on findings from Phase 3.

**Status: PENDING**
*   (Existing content for Potential Tasks and Deliverables is fine)

---

## Phase 5: Paper Assembly, Finalization, and Reproducibility Package

**Goal:** Produce a high-quality research paper (or report) and ensure the project is fully reproducible.

**Status: PENDING**
*   (Existing content for Tasks and Deliverables is fine)

---

This roadmap provides a comprehensive overview. Specific tasks and timelines within each phase will be refined as the project progresses.
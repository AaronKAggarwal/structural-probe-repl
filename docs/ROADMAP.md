# Project Roadmap: Replicating and Extending Structural Probes

Last updated: 2025-06-17 <!-- Updated date -->

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

**Status: COMPLETE**

*   **Sub-Phase 0.1: Native macOS Development Environment Setup**
    *   **Status: COMPLETE.** (Details omitted for brevity, unchanged)
*   **Sub-Phase 0.2: Legacy Hewitt & Manning Probe Replication (PyTorch 1.x CPU)**
    *   **Status: COMPLETE.** (Details omitted for brevity, unchanged)

---

## Phase 1: Modern PyTorch Probe Re-implementation

**Goal:** Re-implement the Hewitt & Manning structural probe using modern PyTorch (v2.x) and best practices. Validate against sample data.

**Status: COMPLETE**

*   **Milestone 1.0: Project Setup & Basic Data Handling Utilities**
    *   **Status: COMPLETE.**
*   **Milestone 1.1: PyTorch `Dataset` and `DataLoader` for ELMo Sample**
    *   **Status: COMPLETE.**
*   **Milestone 1.2: Probe Model and Loss Function Implementation**
    *   **Status: COMPLETE.**
*   **Milestone 1.3: Training and Evaluation Loop Implementation**
    *   **Status: COMPLETE.**
*   **Milestone 1.4: Full Validation on ELMo Sample & Qualitative Parity**
    *   **Status: COMPLETE.**

---

## Phase 2: Pipeline Validation and Dataset Preparation

<!-- Updated Phase Goal -->
**Goal:** Validate the modern probing pipeline by replicating key H&M results on the Penn Treebank (PTB). Then, pivot to and prepare the Universal Dependencies (UD) English Web Treebank as the primary dataset for all future extension experiments.

<!-- Updated Phase Status -->
**Status: COMPLETE**

*   **Sub-Phase 2.1: H&M Replication on Penn Treebank (PTB-SD)**
    *   **Tasks:**
        1.  Acquire PTB and set up Stanford CoreNLP 3.9.2. *(Status: COMPLETE).*
        2.  Implement `ptb_to_conllx.sh` to convert PTB `.mrg` files to Stanford Dependencies (CoNLL-X). *(Status: COMPLETE).*
        3.  Develop generic `scripts/extract_embeddings.py` for Hugging Face models. *(Status: COMPLETE).*
        4.  Extract BERT-base embeddings for the full PTB-SD dataset. *(Status: COMPLETE).*
        5.  **Perform Full H&M Replication Run (BERT-base L7 Distance Probe):** Configure and run the modern probe on PTB-SD data. *(Status: COMPLETE - Achieved UUAS ~80.75%, successfully validating the pipeline against H&M's published results).*
    *   **Deliverables:** Validated modern probing pipeline; processed PTB-SD data; HDF5 embeddings for BERT-base on PTB-SD.
    *   **Status: COMPLETE.**

*   **Sub-Phase 2.2: Pivot to Universal Dependencies (UD)** <!-- New Sub-Phase -->
    *   **Tasks:**
        1.  **Select & Acquire Primary Dataset:** Chose UD English Web Treebank (EWT) as the primary dataset for all future experiments. *(Status: COMPLETE).*
        2.  **Generalize Data Prep Scripts:** Refactor ELMo data preparation scripts to be generic and argument-driven. *(Status: COMPLETE).*
        3.  **Extract Embeddings for UD EWT:** Run `extract_embeddings.py` and the new ELMo scripts to generate embeddings for ELMo (all layers) and BERT-base (L7) on the full UD EWT dataset. *(Status: COMPLETE).*
        4.  **Establish Baselines on UD EWT:** Configure and run distance probes for ELMo and BERT-base on UD EWT to establish new, foundational baseline scores for this dataset. *(Status: COMPLETE - Key results: ELMo L1 UUAS ~72%, ELMo L2 UUAS ~65%, ELMo L0 UUAS ~32%, BERT L7 UUAS ~80%).*
    *   **Deliverables:** Full UD English EWT dataset integrated into project workflow; HDF5 embeddings for baseline models on UD EWT; documented baseline performance metrics for UD EWT.
    *   **Status: COMPLETE.**

---

## Phase 3: Systematic Probing of Modern LLMs on Universal Dependencies

**Goal:** Apply the validated modern PyTorch probe to a diverse set of recent Hugging Face LLMs using the prepared **UD English EWT dataset**. <!-- Updated Dataset -->

**Status: PENDING (Next Phase)**

*   **Tasks:**
    1.  **Finalize Target Modern LLMs:** Select a diverse set (e.g., Llama-3 8B, Mistral-7B, a RoBERTa variant, etc.). Document rationale.
    2.  **Extract Embeddings for Modern LLMs:** Run `scripts/extract_embeddings.py` for all selected models and layers on the **UD EWT** data. This will be computationally intensive.
    3.  **Prepare Hydra Configurations:** Create `embeddings`, `probe`, and `experiment` configs for each modern LLM experiment on UD EWT.
    4.  **Run Probing Experiments:** For each target LLM/layer, train both distance and depth probes.
    5.  **Collect & Analyze Results:**
        *   Systematically log performance metrics (UUAS, DSpr H&M, RootAcc) per layer for each model.
        *   Generate "U-curve" plots (metric vs. layer) for each model.
        *   Compare findings across modern models and against the **new ELMo/BERT baselines on UD EWT**. Analyze how different architectures and model scales encode syntax.
*   **Deliverables:**
    *   HDF5 embedding sets for selected modern LLMs on **UD EWT**.
    *   Comprehensive set of probing results (metrics, U-curves) for these models on UD.
    *   Initial analysis and interpretation comparing modern LLMs to earlier ones on a consistent dataset.

---

## Phase 4: Extensions & Novel Research

**Goal:** Explore advanced research questions and extend the probing methodology based on findings from Phase 3.

**Status: PENDING**
*   *(Existing content for Potential Tasks and Deliverables is fine)*

---

## Phase 5: Paper Assembly, Finalization, and Reproducibility Package

**Goal:** Produce a high-quality research paper (or report) and ensure the project is fully reproducible.

**Status: PENDING**
*   *(Existing content for Tasks and Deliverables is fine)*

---

This roadmap provides a comprehensive overview. Specific tasks and timelines within each phase will be refined as the project progresses.
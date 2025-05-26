# Structural Probe Replication & Extensions

This repo hosts a replication of Hewitt & Manning (2019) structural probes
and subsequent extensions on modern language models.

## Current Status (as of 2025-05-07)

*   **Native macOS Development Environment (PyTorch 2.x, MPS):** Successfully set up using Python 3.11 and Poetry. Core dependencies installed and MPS functionality verified. `requirements-mps.txt` generated.
*   **Phase 0a: Legacy Hewitt & Manning Probe Replication (PyTorch 1.x CPU): COMPLETE**
    *   Original Hewitt & Manning (2019) PyTorch-based code vendored into `src/legacy/structural_probe/`.
    *   Docker environment (`probe:legacy_pt_cpu` image based on Python 3.7, PyTorch 1.3.0+cpu, AllenNLP 0.9.0) successfully built for `linux/amd64`.
    *   Helper scripts for environment check and running the legacy probe created.
    *   Sample data (subset of UD English EWT) prepared locally: CoNLLU files created, ELMo HDF5 embeddings generated using AllenNLP within the Docker container.
    *   Legacy code successfully runs an example experiment (parse-distance and parse-depth probes) end-to-end on the prepared sample data within the container, producing plausible metrics (UUAS, Spearman rho, Root Accuracy).
    *   Key issues encountered (dead download links, dependency conflicts, CoNLLU MWT handling, Docker build quirks) have been resolved and documented.
*   **Next Step:** Phase 1 - Modern PyTorch Probe Re-implementation.

## Current Status (as of 2025-05-23)
# ...
*   **Phase 0a: Legacy Hewitt & Manning Probe Replication (PyTorch 1.x CPU): COMPLETE** 
      * ... (details as before) ...
*   **Phase 1: Modern PyTorch Probe Re-implementation - IN PROGRESS**
    *   **Milestone 1.0 (Basic Data Handling Utilities): COMPLETE.** CoNLL-U reader and gold label (depth/distance) calculation utilities implemented and unit tested.
    *   **Milestone 1.1 (PyTorch Dataset & DataLoader): COMPLETE.** `ProbeDataset` and `collate_fn` implemented and unit tested for loading CoNLL-U parses and pre-computed ELMo embeddings.
*   **Next Step:** Phase 1, Milestone 1.2 - Implementation of modern probe models and loss functions.


## Current Status (as of 2025-05-26)

*   **Native macOS Development Environment (PyTorch 2.x, MPS):** COMPLETE.
*   **Phase 0a: Legacy Hewitt & Manning Probe Replication (PyTorch 1.x CPU): COMPLETE.**
    *   Original H&M code (with original `data.py`) vendored.
    *   `probe:legacy_pt_cpu` Docker image built.
    *   Example data and pre-trained BERT probe parameters sourced from `whykay-01/structural-probes` fork included in the image.
    *   Legacy code runs both ELMo training example and BERT demo successfully.
*   **Phase 1: Modern PyTorch Probe Re-implementation: COMPLETE.**
    *   Core data utilities (`conllu_reader`, `gold_labels`, `embedding_loader`) implemented and unit-tested.
    *   PyTorch `ProbeDataset` and `collate_fn` for data loading/batching implemented and unit-tested.
    *   `DistanceProbe` and `DepthProbe` PyTorch `nn.Module`s implemented and unit-tested.
    *   L1 loss functions (aligned with H&M methodology) implemented and unit-tested.
    *   Training utilities (`get_optimizer`, `EarlyStopper`, checkpointing) implemented and unit-tested.
    *   Main training script (`scripts/train_probe.py`) with Hydra integration implemented.
    *   Smoke test for the full modern training pipeline passes.
    *   Successfully trained and evaluated modern distance and depth probes on the ELMo sample data (using self-generated, MWT-filtered aligned HDF5s), producing plausible metrics and demonstrating functionality on MPS.
*   **Next Step:** Phase 2 - Data Preparation for Modern LLMs (PTB & Hidden State Extraction).
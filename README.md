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

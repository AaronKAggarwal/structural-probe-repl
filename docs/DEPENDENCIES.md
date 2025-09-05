# Dependency Notes

Last updated: 2025-07-09

This document outlines key software dependencies, their versions or version constraints, and the rationale for these choices across the different development and execution environments used in the `structural-probe-repl` project.

## 1. Native macOS Development Environment (Primary for Modern Probe)

This environment is managed by **Poetry** and is intended for local development of the modern structural probe (Phase 1 onwards), debugging, running experiments on models that fit on Apple Silicon (e.g., M3 Max with MPS), and general project tasks.

| Package / Tool           | Pin / Version Constraint      | Rationale & Notes                                                                      |
|--------------------------|-------------------------------|----------------------------------------------------------------------------------------|
| **Python**               | `^3.11` (in `pyproject.toml`) | Current PyTorch (e.g., 2.2.x) wheels are readily available. Python 3.12+ had limited PyTorch support at initial setup. |
| **Poetry**               | `~1.7.1` (or latest stable via `pipx`) | Dependency manager for the project. Installed via `pipx` for user-space isolation and reliable plugin management. |
| `poetry-plugin-export`   | `~1.6.0` (or compatible)      | Poetry plugin to export `requirements.txt`. Installed via `poetry self add`.             |
| **PyTorch (`torch`)**    | `~2.2.0` (or latest stable)   | Core deep learning framework. Version selected for stable MPS backend support on Apple Silicon. |
| `torchvision`            | (Compatible with `torch`)     | For image-related PyTorch utilities (often a companion, though not directly used by probes). |
| `torchaudio`             | (Compatible with `torch`)     | For audio-related PyTorch utilities (often a companion, though not directly used by probes). |
| **NumPy**                | `<2.0` (e.g., `~1.26.4`)      | **Critical Pin:** PyTorch versions up to at least 2.2.x are compiled against NumPy 1.x C APIs. NumPy 2.0 introduced breaking API changes. |
| **SciPy**                | (Compatible version)          | For scientific computing, used for `minimum_spanning_tree` in UUAS calculation.          |
| **Hugging Face `transformers`** | `~4.30.0` or newer (e.g., `~4.40.0`) | For loading modern LLMs and tokenizers (Phase 2 onwards). Version chosen for broad model support and compatibility with `tokenizers`. |
| Hugging Face `tokenizers`| `~0.15.0` or newer (e.g., `~0.19.1`) | Core tokenization library. Requires Rust for compilation from `sdist` on macOS `arm64`. |
| **Rust Toolchain**       | (Installed via `brew`)        | Needed to compile the `tokenizers` package from source on macOS `arm64`.                 |
| **Hydra (`hydra-core`)** | `~1.3.2` (or latest stable)   | For managing complex configurations for experiments.                                     |
| `hydra-joblib-launcher`  | (Compatible with `hydra-core`)| Optional Hydra plugin for parallelizing runs (multirun).                             |
| **H5Py (`h5py`)**        | (Compatible version)          | For reading/writing HDF5 files (used for ELMo embeddings and potentially for modern LLM embeddings). |
| **pytest**               | (Latest stable, e.g. `~8.x.x`)| Framework for writing and running unit and smoke tests. (Dev dependency)              |
| **W&B (`wandb`)**        | (Latest stable)               | Optional: For experiment tracking and visualization. (Dev dependency or main if widely used) |
| **tqdm**                 | (Latest stable)               | For progress bars in scripts.                                                          |
| **Docker Desktop**       | (Latest stable for macOS)     | For building and running `linux/amd64` containers (legacy probe, future CUDA workloads) via Rosetta 2 emulation. Rosetta must be enabled. |
| **Ruff**                 | (Latest stable, e.g., `^0.x.x`)| **Primary Linter & Formatter.** Installed via Poetry. Used for ensuring code quality and consistent style. |

_The full list of native dependencies and their exact resolved versions are managed by Poetry and stored in `poetry.lock`. A reference `requirements-mps.txt` can be generated using `poetry export`._

---
## 2. Legacy Probe Docker Environment (`probe:legacy_pt_cpu`)

This Docker container (defined in `env/Dockerfile.legacy_pt_cpu`) replicates the environment for the original Hewitt & Manning (2019) code (Phase 0a). It is built for `linux/amd64`.

**Base Image:** `python:3.7-slim-buster` (provides Python 3.7.17 on Debian Buster)

**Key Python Packages (installed via `pip` inside Dockerfile):**

| Package                  | Version Pin   | Rationale                                                                          |
|--------------------------|---------------|------------------------------------------------------------------------------------|
| `torch`                  | `1.3.0+cpu`   | Compatible with original H&M code era (~1.0), AllenNLP 0.9.0, CPU-only.            |
| `torchvision`            | `0.4.1+cpu`   | Companion to PyTorch 1.3.0, CPU-only.                                              |
| `allennlp`               | `0.9.0`       | Used by H&M for ELMo embedding generation and handling.                              |
| `numpy`                  | `1.19.5`      | Compatible with PyTorch 1.3.0 and AllenNLP 0.9.0.                                  |
| `scipy`                  | `1.5.4`       | Compatible with PyTorch 1.3.0 and AllenNLP 0.9.0.                                  |
| `PyYAML`                 | `3.13`        | Pre-dates API change in `yaml.load()`, compatible with H&M code's usage.           |
| `tqdm`                   | `4.47.0`      | Known version compatible with Python 3.7.                                          |
| `pytorch-pretrained-bert`| `0.6.2`       | Legacy library used by H&M for BERT embeddings (now superseded by `transformers`).  |
| `protobuf`               | `3.20.1`      | Pinned for general compatibility in older Python/library environments.               |
| `scikit-learn`           | `0.23.2`      | For general ML utilities, version appropriate for the era.                         |
| `nltk`                   | `3.5`         | For NLP utilities, version appropriate for the era.                                |
| `overrides`              | `3.1.0`       | Pinned for compatibility with AllenNLP 0.9.0 to resolve import/TypeErrors.         |
| `typing-extensions`      | `3.7.4`       | Pinned for compatibility with AllenNLP 0.9.0 and Python 3.7's typing system.      |
| `Cython`                 | (No pin)      | From H&M `requirements.txt`; `pip` resolves compatible version.                    |
| `seaborn`                | (No pin)      | From H&M `requirements.txt` (plotting); `pip` resolves compatible version.         |
| `h5py`                   | (No pin)      | From H&M `requirements.txt` (HDF5 handling); `pip` resolves compatible version.    |

**System Packages (installed via `apt-get` inside `Dockerfile.legacy_pt_cpu`):**
*   `build-essential`: For C/C++ compilation (e.g., for Cython if it builds extensions from source).
*   `git`: May be needed by `pip` for some packages or by AllenNLP for certain operations.
*   `wget`: General download utility (used by H&M's original `download_example.sh`).

## 3. Data Preprocessing Tools (External to Poetry Environment)

This section lists tools used for initial data preparation before the Python pipeline.

| Tool / Library        | Version Used/Targeted | Rationale & Notes                                                                 |
|-----------------------|-----------------------|-----------------------------------------------------------------------------------|
| **Java (JDK/JRE)**    | 11 or 17 (e.g., OpenJDK) | Required by Stanford CoreNLP.                                                    |
| **Stanford CoreNLP**  | 3.9.2 (Full package)  | Used to convert Penn Treebank constituency parses to Stanford Dependencies (CoNLL-X format), aligning with Hewitt & Manning (2019). Downloaded separately. |
| (Potentially `unzstd`)| (System dependent)    | May be needed to decompress LDC .tar.zst archives.                                |


## 4. CUDA Docker Environment (`probe:cuda` - For Phase 2+ Modern Probe on NVIDIA GPUs)

**Status: PLANNED**

This environment will be defined in `env/Dockerfile.cuda`.

**Key Components (Planned):**
*   **Base Image:** An official PyTorch CUDA image (e.g., `pytorch/pytorch:2.2.x-cuda12.1-cudnn8-runtime` or similar, matching the native PyTorch version used for development).
*   **Python & Project Dependencies:** Installed via Poetry (using `poetry.lock` for consistency with native environment where possible).
*   **NVIDIA Container Toolkit:** Required on the host machine to enable GPU access within Docker.

*(Specific package versions for CUDA environment will largely mirror the native environment, with PyTorch built for CUDA.)*
# Dependency Notes

Last updated: 2025-05-06

This document outlines key dependencies, version pins, and rationales for both the native macOS development environment and the containerized environments.

## 1. Native macOS Environment (for PyTorch 2.x with MPS)

This environment is managed by Poetry and is intended for local development, debugging, and running experiments on models that fit on Apple Silicon (M3 Max).

| Package / Tool           | Pin / Version                | Rationale                                                                      |
|--------------------------|------------------------------|--------------------------------------------------------------------------------|
| **Python**               | `3.11.x`                     | Current PyTorch (2.2.x) wheels available; 3.12/3.13 had limited/no support at time of setup. |
| **Poetry**               | `2.1.2` (via `pipx`)         | Dependency manager. Installed via `pipx` for user-space isolation and plugin compatibility (vs. Homebrew). |
| `poetry-plugin-export`   | `~1.8.0` or `~1.9.0`         | Provides `poetry export` for generating `requirements.txt` files. Installed via `poetry self add`. |
| **Torch**                | `2.2.0` (or latest `2.2.x`)  | Stable MPS backend support on Apple Silicon.                                   |
| **NumPy**                | `<2.0` (e.g., `~1.26.4`)      | **Critical Pin:** PyTorch 2.2.x is compiled against NumPy 1.x headers. NumPy 2.0 introduced breaking API changes. |
| **Transformers**         | `~4.40.0`                    | Hugging Face library for models. Version chosen for compatibility with `tokenizers`. |
| **Tokenizers**           | `~0.19.1`                    | Hugging Face tokenization library. Requires Rust for compilation from `sdist` on `arm64` (macOS). |
| **Rust Toolchain**       | (Installed via `brew`)       | Needed to compile `tokenizers` from source on macOS `arm64`.                     |
| **Docker Desktop**       | (Latest stable)              | For running `linux/amd64` containers (legacy probe, future CUDA workloads) via Rosetta 2 emulation. Rosetta must be enabled for x86 containers. |

_Full native dependency list is maintained in `poetry.lock` and can be exported to `requirements-mps.txt`._

## 2. Legacy Probe Container (`probe:legacy_pt_cpu`)

This Docker container aims to replicate the environment for the original Hewitt & Manning (2019) code, which is PyTorch-based (approx. v1.3). Built for `linux/amd64`.

**Base Image:** `python:3.7-slim-buster` (provides Python 3.7.17 on Debian Buster)

**Key Python Packages (installed via `pip` inside Dockerfile):**

| Package                  | Version Pin   | Rationale                                                                          |
|--------------------------|---------------|------------------------------------------------------------------------------------|
| `torch`                  | `1.3.0+cpu`   | Compatible with original code's era (H&M used ~1.0), AllenNLP 0.9.0, CPU-only.     |
| `torchvision`            | `0.4.1+cpu`   | Companion to PyTorch 1.3.0, CPU-only.                                              |
| `allennlp`               | `0.9.0`       | For ELMo embedding generation/handling as per H&M scripts/README.                  |
| `numpy`                  | `1.19.5`      | Compatible with PyTorch 1.3 and AllenNLP 0.9.0. (Note: H&M `requirements.txt` did not pin). |
| `scipy`                  | `1.5.4`       | Compatible with PyTorch 1.3 and AllenNLP 0.9.0. (Note: H&M `requirements.txt` did not pin). |
| `PyYAML`                 | `3.13`        | Pre-dates API change requiring `Loader` arg for `yaml.load()`, as used in H&M code. |
| `tqdm`                   | `4.47.0`      | Last version found to robustly support Python 3.7 during testing.                  |
| `pytorch-pretrained-bert`| `0.6.2`       | Legacy library used by H&M for BERT embeddings (now part of `transformers`).        |
| `protobuf`               | `3.20.1`      | Pinned for general compatibility in older Python/library environments.               |
| `scikit-learn`           | `0.23.2`      | General ML utilities, reasonable version for the era.                              |
| `nltk`                   | `3.5`         | NLP utilities, reasonable version for the era.                                     |
| `Cython`                 | (No pin)      | From H&M `requirements.txt`. Installed by `pip`.                                   |
| `seaborn`                | (No pin)      | From H&M `requirements.txt`, for plotting. Installed by `pip`.                     |
| `h5py`                   | (No pin)      | From H&M `requirements.txt`, for HDF5 file handling. Installed by `pip`.           |

**System Packages (installed via `apt-get` inside Dockerfile):**
*   `build-essential`: For C/C++ compilation (e.g., for Cython if it builds extensions).
*   `git`: May be needed by AllenNLP or other pip installs from source.
*   `wget`: Required by the original (but now non-functional due to dead links) `download_example.sh` script from H&M. Kept in case alternative download scripts are used.
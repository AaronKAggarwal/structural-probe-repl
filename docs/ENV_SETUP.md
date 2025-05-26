# Environment Setup

Last updated: 2025-05-26

This guide details setting up the development and execution environments for the `structural-probe-repl` project:
1.  **Native macOS Environment:** For modern PyTorch (2.x) development with MPS acceleration (primary for Phase 1 onwards).
2.  **Legacy Probe Docker Environment:** For running the original Hewitt & Manning (2019) code (Phase 0a).
3.  **CUDA Docker Environment (Future):** For large model processing and GPU-accelerated modern probe experiments (Phase 2 onwards).

---
## 1. Native macOS Environment (e.g., M3 Max with MPS)

This setup is for local development, debugging, and running experiments with PyTorch 2.x on Apple Silicon, leveraging MPS for GPU acceleration. This is the primary environment for developing the modern structural probe (Phase 1) and for running experiments with smaller modern LLMs.

**Prerequisites:**
*   An Apple Silicon Mac (M1/M2/M3 series).
*   Homebrew package manager: [https://brew.sh/](https://brew.sh/)
*   Git: Already part of macOS or installable via Homebrew (`brew install git`).

**Steps:**

```bash
# 1. Install System Prerequisites (Python 3.11, Rust, pipx)
# Python 3.11 is targeted for PyTorch 2.2 compatibility.
# Rust is required for building the `tokenizers` package from source on arm64.
# pipx is used to install Poetry in an isolated Python environment.
brew install python@3.11 rust pipx
brew link python@3.11 --force # Ensure brew's Python 3.11 is on PATH, may need --force if linked by system

# 2. Install Poetry via pipx
# This provides user-space isolation for Poetry itself and avoids conflicts.
pipx install poetry
pipx ensurepath # Ensures pipx bin directory (e.g., ~/.local/bin) is on PATH.
                # May require restarting your shell or sourcing your shell profile (e.g., source ~/.zshrc).
poetry --version   # Verify installation (e.g., Poetry (version 1.7.1) or similar)

# 3. Clone Project & Setup Python Environment with Poetry
# (Assuming you have already cloned the project: git clone <repo_url>)
cd path/to/structural-probe-repl # Navigate to your project root

# Tell Poetry to use the Homebrew-installed Python 3.11 for this project's virtual environment
poetry env use $(brew --prefix python@3.11)/bin/python3.11

# Install project dependencies (including dev dependencies like pytest) from poetry.lock (or pyproject.toml if no lock)
poetry install # Use `poetry install --no-dev` to skip dev dependencies

# Activate the virtual environment (optional, as `poetry run` handles it)
# poetry shell

# 4. Install Poetry Export Plugin & Freeze Dependencies (Optional but good for reference)
# This plugin allows exporting to requirements.txt format if needed elsewhere.
poetry self add poetry-plugin-export # Installs plugin to Poetry's own environment
poetry export --without-hashes --format=requirements.txt > requirements-mps.txt
echo "Native macOS environment dependencies (for reference) frozen to requirements-mps.txt"

# 5. Smoke-Test Native MPS Environment
echo "Running MPS smoke test..."
poetry run python - <<'PY'
import torch
import numpy
import platform
print(f"Python Version: {platform.python_version()}")
print(f"NumPy Version:  {numpy.__version__}")
print(f"PyTorch Version:{torch.__version__}")
mps_available = False
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    mps_available = True
    try:
        x = torch.tensor([1.0, 2.0]).to("mps")
        print(f"MPS Available:  {mps_available} (Device test OK: tensor moved to MPS)")
    except Exception as e:
        print(f"MPS Available:  {mps_available} (Device test FAILED to use MPS: {e})")
else:
    print(f"MPS Available:  {mps_available} (torch.backends.mps.is_available() or is_built() is False)")
PY
# Expected output (example):
# Python Version: 3.11.x
# NumPy Version:  1.26.x (or compatible <2.0)
# PyTorch Version: 2.2.x (or similar)
# MPS Available:  True (Device test OK: tensor moved to MPS)
```
See `docs/DEPENDENCIES.md` for notes on specific versions (e.g., NumPy `<2.0` for PyTorch 2.2 compatibility).

---
## 2. Legacy PyTorch Probe Environment (Docker for Phase 0a)

This Docker container (`probe:legacy_pt_cpu`) replicates the environment for the original Hewitt & Manning (2019) structural probes code. It's built for `linux/amd64` and will run under Rosetta 2 emulation on Apple Silicon Macs, or natively on a Linux AMD64 host. This environment was used to complete **Phase 0a**.

**Prerequisites for Use:**
*   Docker Desktop installed and running ([https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)).
    *   On Apple Silicon Macs, ensure "Use Rosetta for x86/amd64 emulation on Apple Silicon" is enabled in Docker Desktop settings (Preferences → Features → Beta Features or similar).
*   Logged into Docker Hub (run `docker login` once).
*   Project repository cloned, and you are in the project root directory.
*   The original Hewitt & Manning code (with their original `data.py`) is vendored into `src/legacy/structural_probe/`.
*   Example data and pre-trained BERT probe parameters (sourced from the `whykay-01/structural-probes` GitHub fork) are placed into `src/legacy/structural_probe/example/data/` and its subdirectories.

**1. Dockerfile Location:**
The environment is defined in `env/Dockerfile.legacy_pt_cpu`. Key components include:
*   Base Image: `python:3.7-slim-buster`.
*   Python: 3.7.17.
*   PyTorch: `1.3.0+cpu`.
*   AllenNLP: `0.9.0`.
*   Data from the `whykay-01` fork is `COPY`ed into the image.
*   Other specific legacy dependencies (e.g., `PyYAML==3.13`).

**2. Build the Docker Image:**
From the project root:
```bash
docker build --platform=linux/amd64 \
  -f env/Dockerfile.legacy_pt_cpu \
  -t probe:legacy_pt_cpu .
```

**3. Running Examples:**

*   **Health Check (Implicitly run by default CMD):**
    The container's `ENTRYPOINT` (`scripts/check_legacy_env.sh`) verifies key dependencies.

*   **ELMo Example (Parse Distance Probe Training - Default CMD):**
    ```bash
    # Create a local directory for results
    mkdir -p results_staging/legacy_ELMo_prd_whykay01_data

    # Run, mounting the results directory
    docker run --rm --platform=linux/amd64 \
      -v "$(pwd)/results_staging/legacy_ELMo_prd_whykay01_data":/app/structural_probe_original/example/results \
      probe:legacy_pt_cpu
    ```
    **Expected Outcome:** Health check passes. `run_experiment.py` trains an ELMo distance probe on the sample data, producing metrics (e.g., UUAS ~0.271, Spearman ~0.451) in the mounted results directory.

*   **BERT Demo (Pre-trained Probes):**
    ```bash
    # Create a local directory for results
    mkdir -p results_staging/legacy_BERT_demo_whykay01_data

    # Run, mounting results and piping input sentences
    docker run --rm --platform=linux/amd64 \
      -v "$(pwd)/results_staging/legacy_BERT_demo_whykay01_data":/app/structural_probe_original/example/results \
      -i probe:legacy_pt_cpu \
      /scripts/run_legacy_probe.sh example/demo-bert.yaml <<EOF
    The chef that went to the stores was out of food .
    This is another test sentence for the BERT probe .
    EOF
    ```
    **Expected Outcome:** `run_demo.py` executes, loads pre-trained BERT probe parameters, processes input, and saves visualizations (e.g., `.png`, `.tikz`) in the mounted results directory.

---
## 3. CUDA Docker Environment (for Phase 2+ Modern Probe on NVIDIA GPUs)

**Status: PLANNED (To be implemented in Phase 2)**

This environment will be used for:
*   Extracting hidden states from very large LLMs that may not fit/run efficiently on MPS.
*   Running training/evaluation of the modern PyTorch probe (from Phase 1) on NVIDIA GPUs for larger datasets (like full PTB) or more computationally intensive experiments.

**1. Dockerfile Location (Planned):**
`env/Dockerfile.cuda`

**2. Key Components (Planned):**
*   Base Image: An official PyTorch CUDA image (e.g., `pytorch/pytorch:2.2.x-cuda12.1-cudnn8-runtime` matching the native PyTorch version).
*   Project dependencies installed via Poetry.
*   Project source code (`src/`, `scripts/`) copied in.

**3. Build Command (Conceptual):**
```bash
# On a machine with Docker and NVIDIA GPU drivers / NVIDIA Container Toolkit
docker build -f env/Dockerfile.cuda -t probe:cuda .
```

**4. Run Command (Conceptual):**
```bash
# On a machine with Docker and NVIDIA GPU
docker run --rm --gpus all \
  -v /path/to/host/data:/data_mount \
  -v /path/to/host/configs:/configs_mount \
  -v /path/to/host/outputs:/app/outputs \
  probe:cuda \
  python scripts/train_probe.py experiment=my_ptb_experiment_cuda ...
```
Further details will be added as this environment is developed in Phase 2.
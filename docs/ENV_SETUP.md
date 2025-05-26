---

**Updated `docs/ENV_SETUP.md`**

```markdown
# Environment Setup

Last updated: 2025-05-23

This guide details setting up the development environments for the `structural-probe-repl` project:
1.  Native macOS environment for current PyTorch (MPS accelerated).
2.  Dockerized Linux environment for the legacy Hewitt & Manning probe code (CPU).

## 1. Native macOS Environment (M3 Max with MPS)

This setup is for local development, debugging, and running experiments with PyTorch 2.x on Apple Silicon, leveraging MPS for GPU acceleration.

**Prerequisites:**
*   Homebrew package manager.
*   An Apple Silicon Mac (M1/M2/M3 series).

**Steps:**

```bash
# 1. Install System Prerequisites (Python 3.11, Rust, pipx)
# Python 3.11 is targeted for PyTorch 2.2 compatibility.
# Rust is required for building the `tokenizers` package from source on arm64.
# pipx is used to install Poetry in an isolated environment.
brew install python@3.11 rust pipx
brew link python@3.11 --force # Ensure brew's Python 3.11 is on PATH, may need --force if linked by system

# 2. Install Poetry via pipx
# This provides user-space isolation and avoids conflicts with system/Homebrew Python.
pipx install poetry
pipx ensurepath # Ensures pipx bin directory is on PATH (may require shell restart/re-source)
# If not automatically added to PATH, add to your shell config (e.g., ~/.zshrc or ~/.bash_profile):
# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
poetry --version   # Verify installation (e.g., Poetry (version 2.1.2) or similar)

# 3. Clone Project & Setup Python Environment with Poetry
# (Assuming you have already cloned the project and are in its root directory)
# cd structural-probe-repl 
poetry env use $(brew --prefix python@3.11)/bin/python3.11 # Explicitly tell Poetry to use brew's Python 3.11
poetry install --no-root # Installs dependencies from pyproject.toml. --no-root if project isn't a package itself.

# 4. Install Poetry Export Plugin & Freeze Dependencies
# This plugin allows exporting to requirements.txt format.
poetry self add poetry-plugin-export
poetry export --without-hashes --format=requirements.txt > requirements-mps.txt
echo "Native macOS environment frozen to requirements-mps.txt"

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
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    mps_available = True
    # Test MPS device allocation
    try:
        x = torch.tensor([1.0, 2.0]).to("mps")
        print(f"MPS Available:  {mps_available} (Device test OK)")
    except Exception as e:
        print(f"MPS Available:  {mps_available} (Device test FAILED: {e})")
else:
    print(f"MPS Available:  {mps_available}")
PY
# Expected output (example):
# Python Version: 3.11.x
# NumPy Version:  1.26.x (or compatible <2.0)
# PyTorch Version:2.2.0 (or similar)
# MPS Available:  True (Device test OK)
```

---
## 2. Legacy PyTorch Probe Environment (Docker for Phase 0a)

This Docker container (`probe:legacy_pt_cpu`) replicates the environment for the original Hewitt & Manning (2019) structural probes code. It's built for `linux/amd64` and will run under Rosetta 2 emulation on Apple Silicon Macs, or natively on a Linux AMD64 host.

**Prerequisites:**
*   Docker Desktop installed and running.
*   Logged into Docker Hub (`docker login`).
*   The project repository cloned, and you are in the project root directory.
*   The original Hewitt & Manning code (unmodified `data.py`) vendored into `src/legacy/structural_probe/`.
*   Example data and pre-trained BERT probe parameters (sourced from the `whykay-01/structural-probes` fork) placed into `src/legacy/structural_probe/example/data/` and `src/legacy/structural_probe/example/data/en_ewt-ud-sample/`.

**1. Dockerfile:**
The environment is defined in `env/Dockerfile.legacy_pt_cpu`. Key components:
*   Base Image: `python:3.7-slim-buster` (provides Python 3.7.17).
*   PyTorch: `1.3.0+cpu`.
*   AllenNLP: `0.9.0`.
*   The data from `whykay-01` fork (CoNLLU, ELMo HDF5, BERT `.params`) is `COPY`ed into the image from `src/legacy/structural_probe/example/data/`.
*   Other dependencies as per original code's `requirements.txt` and compatibility needs (e.g., `PyYAML==3.13`).

**2. Build the Docker Image:**
```bash
docker build --platform=linux/amd64 \
  -f env/Dockerfile.legacy_pt_cpu \
  -t probe:legacy_pt_cpu .
```

**3. Run Container Health Check & Default ELMo Example:**
The container's `ENTRYPOINT` runs `scripts/check_legacy_env.sh`, and the default `CMD` executes `scripts/run_legacy_probe.sh` with the `prd_en_ewt-ud-sample.yaml` config. This uses the (presumed) original H&M example data (via `whykay-01` fork) copied into the image.
```bash
# Create a local directory for results if it doesn't exist
mkdir -p results_staging/legacy_ELMo_prd_run_whykay01_data

# Run, mounting the results directory
docker run --rm --platform=linux/amd64 \
  -v "$(pwd)/results_staging/legacy_ELMo_prd_run_whykay01_data":/app/structural_probe_original/example/results \
  probe:legacy_pt_cpu
```
**Expected Outcome:**
The health check will pass. The `run_experiment.py` script will execute end-to-end on the sample data, training an ELMo distance probe and producing metrics (e.g., UUAS ~0.271, Spearman ~0.451) in the mounted results directory.

**4. Running the BERT Demo (with pre-trained probes from `whykay-01`):**
```bash
# Create a local directory for results
mkdir -p results_staging/legacy_BERT_demo_whykay01_data

# Run, mounting the results directory and piping input sentences
docker run --rm --platform=linux/amd64 \
  -v "$(pwd)/results_staging/legacy_BERT_demo_whykay01_data":/app/structural_probe_original/example/results \
  -i probe:legacy_pt_cpu \
  /scripts/run_legacy_probe.sh example/demo-bert.yaml <<EOF
The chef that went to the stores was out of food .
This is another test sentence for the BERT probe .
EOF
```
**Expected Outcome:**
The `run_demo.py` script will execute, load the pre-trained BERT probe parameters, process the input, and save visualizations (e.g., `.png`, `.tikz`) in the mounted results directory.

**(Future sections for CUDA Docker environment, etc., can be added here later.)**
```

---

**Key Changes in `ENV_SETUP.md`:**

*   Updated "Last updated" date.
*   **Native macOS Env:**
    *   Added `--force` to `brew link python@3.11` as it's sometimes needed.
    *   Clarified `pipx ensurepath` might require shell restart.
    *   Clarified `poetry install --no-root`.
    *   Made MPS smoke test slightly more robust in printing.
*   **Legacy PyTorch Probe Environment:**
    *   Updated prerequisites to mention vendored H&M code (original `data.py`) and data from `whykay-01` fork.
    *   Dockerfile summary updated to reflect that data from `whykay-01` is `COPY`ed in.
    *   Updated "Run Container Health Check & Default ELMo Example" section to reflect the successful outcome and mention the data source.
    *   Updated "Inspecting Results / Running Custom Experiments" to "Running the BERT Demo" with the correct command and expected outcome.

This should now be a complete and accurate `ENV_SETUP.md` reflecting the successful completion of Phase 0a.
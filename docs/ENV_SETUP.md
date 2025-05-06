# Environment Setup

Last updated: 2025-05-06

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
brew link python@3.11 # Ensure brew's Python 3.11 is on PATH

# 2. Install Poetry via pipx
# This provides user-space isolation and avoids conflicts with system/Homebrew Python.
pipx install poetry
pipx ensurepath # Ensures pipx bin directory is on PATH (may require shell restart/re-source)
# Add to your shell config (e.g., ~/.zshrc or ~/.bash_profile) if not done by ensurepath:
# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
poetry --version   # Verify installation (e.g., Poetry (version 2.1.2))

# 3. Clone Project & Setup Python Environment with Poetry
git clone git@github.com:AaronKAggarwal/structural-probe-repl.git # Or your repo URL
cd structural-probe-repl
poetry env use $(brew --prefix python@3.11)/bin/python3.11 # Explicitly tell Poetry to use brew's Python 3.11
poetry install --no-root # Installs dependencies from pyproject.toml, --no-root if project isn't a package itself

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
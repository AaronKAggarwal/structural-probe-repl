# Computational Environment

This document records the exact computational environment used for reproducibility.

## Software Environment

- **Python Version**: `python --version`
- **Poetry Version**: `poetry --version`  
- **PyTorch Version**: `python -c "import torch; print(torch.__version__)"`
- **Transformers Version**: `python -c "import transformers; print(transformers.__version__)"`
- **NumPy Version**: `python -c "import numpy; print(numpy.__version__)"`
- **Pandas Version**: `python -c "import pandas; print(pandas.__version__)"`

## Hardware Environment

- **Operating System**: `uname -a`
- **CPU**: `lscpu | grep "Model name"`
- **Memory**: `free -h`
- **GPU Model**: `nvidia-smi --query-gpu=name --format=csv,noheader`
- **CUDA Version**: `nvcc --version`
- **cuDNN Version**: `python -c "import torch; print(torch.backends.cudnn.version())"`

## Model Artifacts

- **HuggingFace Model**: `bert-base-multilingual-cased`
- **Model Commit Hash**: `_______` (to be filled)
- **Tokenizer Commit Hash**: `_______` (to be filled)
- **Vocabulary Checksum**: `sha256sum vocab.txt` → `_______`

## Random Seeds

- **Global NumPy Seed**: `_______`
- **Global PyTorch Seed**: `_______`
- **Python Random Seed**: `_______`
- **Bootstrap Seeds**: Stage 5 morphology = `_______`

## Data Versions

- **UD Release**: Per-treebank versions in `docs/ANALYSIS_INVARIANTS.md`
- **Wikipedia Dumps**: 2018-10/11 dates per language in `outputs/analysis/pretrain_exposure.csv`

## Dependencies Lock

See `poetry.lock` for exact package versions and dependency resolution.

## Determinism Settings

```python
# PyTorch determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# NumPy determinism  
np.random.seed(SEED)

# Python determinism
random.seed(SEED)
```

## Non-Deterministic Operations

Document any unavoidable sources of non-determinism:
- GPU floating-point operations may have slight variance
- Parallel data loading (if used)
- Any other hardware-dependent computations

---

**Generated**: `date`  
**Commit**: `git rev-parse HEAD`

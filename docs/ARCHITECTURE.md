# Repository Architecture

This document explains the layout of both the legacy structural probe code and the new project structure.

## 1. Legacy code (`src/legacy/structural_probe`)

- **README.md / LICENSE**  
  Original paper’s overview and license.

- **doc‑assets/**  
  Figures (PNG) used in the original write‑up.

- **download_example.sh**  
  Fetches a small PTB sample for quick smoke tests.

- **example/**  
  - **config/** — YAML configs for pad/prd runs of various models (bert‑base, elmo, etc.).  
  - **demo‑bert.yaml** — an end‑to‑end demo configuration.

- **requirements.txt**  
  Exact Python package pins expected by the legacy code.

- **scripts/**  
  Data‑prep utilities:  
  - `convert_conll_to_raw.py`  
  - `convert_raw_to_bert.py`  
  - `convert_raw_to_elmo.sh`  
  - `convert_splits_to_depparse.sh`

- **structural‑probes/**  
  Core probe implementation and orchestration:  
  - `probe.py` — depth‑based structural probe.  
  - `model.py`, `data.py` — model and PTB loading.  
  - `run_experiment.py` / `run_demo.py` — experiment drivers.  
  - `loss.py`, `regimen.py`, `reporter.py`, `task.py` — training, logging, and evaluation.

## 2. New project scaffold (`src/`)

- **legacy/**  
  Contains the entire unmodified legacy codebase (see above).

- **torch_probe/**  
  _(To be created)_ For the new PyTorch re‑implementation of the structural probe.

- **common/**  
  _(To be created)_ Shared utilities (e.g. PTB parser, metrics, config loader).

## 3. Documentation (`docs/`)

- `ENV_SETUP.md`  
- `DEPENDENCIES.md`  
- `DOC_INDEX.md`  
- `ARCHITECTURE.md`  ← *this file*


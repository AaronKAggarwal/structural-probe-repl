# Environment Setup  
_Last updated: 2025‑05‑02_

## Native macOS (M3 Max, MPS)

```bash
# 1. System prerequisites
brew install python@3.11 rust pipx

# 2. Poetry via pipx (isolated from Homebrew)
pipx install poetry
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
poetry --version   # should show 2.1.2 (pipx)

# 3. Project venv (Python 3.11)
cd structural-probe-repl
poetry env use python3.11
poetry install --no-root

# 4. Export plugin & freeze
poetry self add poetry-plugin-export
poetry export --without-hashes --format=requirements.txt > requirements-mps.txt

# 5. Smoke‑test MPS
poetry run python - <<'PY'
import torch, numpy, platform
print(platform.python_version(), numpy.__version__, torch.__version__, torch.backends.mps.is_available())
PY
# Expected: 3.11.x  1.26.x  2.2.0  True

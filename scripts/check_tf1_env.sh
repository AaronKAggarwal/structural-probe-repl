#!/usr/bin/env bash
python - <<'PY'
import tensorflow as tf, numpy, torch, sys
versions = {
  "tf"   : tf.__version__,
  "numpy": numpy.__version__,
  "torch": torch.__version__,
}
print("TF:", versions["tf"], "NumPy:", versions["numpy"], "Torch:", versions["torch"])
assert versions["tf"].startswith("1.15"),  "TF 1.15.x required"
assert versions["numpy"].startswith("1.16"), "NumPy 1.16.x required"
assert versions["torch"].startswith("1.3"), "Torch 1.3.x required"
PY
exec "$@"

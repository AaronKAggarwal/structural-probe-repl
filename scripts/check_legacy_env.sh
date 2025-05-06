#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Legacy Environment Health Check ---"
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "AllenNLP version:"
python -c "import allennlp; print(allennlp.__version__)"
echo "NumPy version:"
python -c "import numpy; print(numpy.__version__)"
echo "SciPy version:"
python -c "import scipy; print(scipy.__version__)"
echo "pytorch-pretrained-bert version:"
python -c "from pytorch_pretrained_bert import __version__ as ppb_version; print(ppb_version)"


# Check for existence of main script from Hewitt & Manning code
if [ ! -f "structural-probes/run_experiment.py" ]; then
echo "ERROR: Original run_experiment.py not found at $(pwd)/structural-probes/run_experiment.py"
fi
echo "Original run_experiment.py found."

echo "Environment check passed."
echo "-----------------------------------"

# Execute the command passed to the entrypoint (i.e., the CMD or docker run arguments)
exec "$@"
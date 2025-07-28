# scripts/diagnostics/diag_15_check_model_layers.py
from transformers import AutoConfig

model_name = "meta-llama/Llama-3.2-3B"
config = AutoConfig.from_pretrained(model_name)
num_layers = config.num_hidden_layers
print(f"Model: {model_name}")
print(f"Number of hidden layers reported by config: {num_layers}")
print(f"Expected layer indices: 0 to {num_layers}") # +1 for embedding layer
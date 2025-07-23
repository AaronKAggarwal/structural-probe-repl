# scripts/diagnostics/diag_12_profile_evaluation.py
import argparse
import cProfile
import pstats
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# --- CORRECTED IMPORTS ---
# Add the project's 'src' directory to the Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Now, import directly from the modules
from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.probe_models import DistanceProbe

# This is a special setup for this diagnostic script.
# It assumes you have two versions of the evaluate.py file:
# 1. The original, serial version at `src/torch_probe/evaluate.py`
# 2. The new, parallel version at `src/torch_probe/evaluate_parallel.py`
try:
    from torch_probe.evaluate import (
        calculate_spearmanr_hm_style as calculate_spearmanr_serial,
        calculate_uuas as calculate_uuas_serial,
    )
    from torch_probe.evaluate_parallel import (
        calculate_spearmanr_hm_style as calculate_spearmanr_parallel,
        calculate_uuas as calculate_uuas_parallel,
    )
except ImportError as e:
    print("---! SETUP ERROR !---", file=sys.stderr)
    print(f"Could not import necessary modules: {e}", file=sys.stderr)
    print("Please ensure you have followed these setup steps for this diagnostic:", file=sys.stderr)
    print("1. Your original, FAST (serial) evaluation code is in 'src/torch_probe/evaluate.py'.", file=sys.stderr)
    print("2. Your new, SLOW (joblib) evaluation code is in 'src/torch_probe/evaluate_parallel.py'.", file=sys.stderr)
    sys.exit(1)
# --- END CORRECTED IMPORTS ---


def evaluate_probe_wrapper(
    probe_model, dataloader, device, probe_type, parallel: bool
):
    """A wrapper to call either the serial or parallel evaluation logic."""
    # This part is the data aggregation
    probe_model.eval()
    all_predictions_np = []
    all_gold_labels_np = []
    all_lengths_list = []
    all_gold_head_indices_list = []
    all_xpos_tags_list = []
    all_upos_tags_list = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings_b = batch["embeddings_batch"].to(device)
            labels_b_for_loss = batch["labels_batch"] # Keep labels on CPU
            lengths_b = batch["lengths_batch"]
            
            predictions_b = probe_model(embeddings_b)
            
            for i in range(predictions_b.shape[0]):
                length = lengths_b[i].item()
                all_lengths_list.append(length)
                all_gold_head_indices_list.append(batch["head_indices_batch"][i])
                all_xpos_tags_list.append(batch["xpos_tags_batch"][i])
                all_upos_tags_list.append(batch["upos_tags_batch"][i])
                
                pred = predictions_b[i, :length, :length].cpu().numpy()
                gold = labels_b_for_loss[i, :length, :length].cpu().numpy()
                all_predictions_np.append(pred)
                all_gold_labels_np.append(gold)

    # This part is the metric calculation
    if parallel:
        calculate_uuas_parallel(
            all_predictions_np, all_gold_head_indices_list, all_lengths_list,
            all_xpos_tags_list, all_upos_tags_list, "xpos"
        )
        calculate_spearmanr_parallel(
            all_predictions_np, all_gold_labels_np, all_lengths_list,
            all_xpos_tags_list, probe_type
        )
    else:
        calculate_uuas_serial(
            all_predictions_np, all_gold_head_indices_list, all_lengths_list,
            all_xpos_tags_list, all_upos_tags_list, "xpos"
        )
        calculate_spearmanr_serial(
            all_predictions_np, all_gold_labels_np, all_lengths_list,
            all_xpos_tags_list, probe_type
        )

if __name__ == "__main__":
    CONLLU_PATH = "data/ud_english_ewt_official/en_ewt-ud-dev.conllu"
    HDF5_PATH = "data_staging/embeddings/ud_english_ewt_full/elmo/ud_english_ewt_full_dev_layers-all_align-mean.hdf5"
    DEVICE = torch.device("cpu") # Metrics are CPU-bound, so let's keep everything on CPU for this test

    print("--- Diagnosis 12: Profiling Evaluation Logic ---")
    
    # Setup common objects
    dataset = ProbeDataset(
        conllu_filepath=CONLLU_PATH, hdf5_filepath=HDF5_PATH,
        embedding_layer_index=1, probe_task_type="distance", preload=True
    )
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_probe_batch)
    probe_model = DistanceProbe(dataset.embedding_dim, probe_rank=128).to(DEVICE)

    # --- Profile SERIAL version ---
    print("\n--- Profiling SERIAL Evaluation (Original Code) ---")
    profiler_serial = cProfile.Profile()
    profiler_serial.enable()
    evaluate_probe_wrapper(probe_model, dataloader, DEVICE, "distance", parallel=False)
    profiler_serial.disable()
    
    print("Top 20 functions by total time spent (tottime):")
    stats_serial = pstats.Stats(profiler_serial).sort_stats('tottime')
    stats_serial.print_stats(20)

    # --- Profile PARALLEL version ---
    print("\n--- Profiling PARALLEL Evaluation (Joblib Code) ---")
    profiler_parallel = cProfile.Profile()
    profiler_parallel.enable()
    evaluate_probe_wrapper(probe_model, dataloader, DEVICE, "distance", parallel=True)
    profiler_parallel.disable()

    print("Top 20 functions by total time spent (tottime):")
    stats_parallel = pstats.Stats(profiler_parallel).sort_stats('tottime')
    stats_parallel.print_stats(20)
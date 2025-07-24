# scripts/diagnostics/diag_05_profile_step.py
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from torch_probe.probe_models import DistanceProbe
from torch_probe.loss_functions import distance_l1_loss

def profile_training_step(
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
    probe_rank: int,
    device: torch.device,
):
    print(f"\n--- Profiling Full Step for Batch Size: {batch_size} ---")
    
    # --- Setup ---
    probe = DistanceProbe(embedding_dim, probe_rank).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.001)

    # Create dummy data on the correct device
    dummy_embeddings = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    dummy_gold_dists = torch.randn(batch_size, seq_len, seq_len, device=device)
    dummy_lengths = torch.randint(low=seq_len//2, high=seq_len+1, size=(batch_size,), device=device)
    
    # --- Profiling Run ---
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda' or device.type == 'mps':
        activities.append(ProfilerActivity.GPU if device.type == 'cuda' else ProfilerActivity.CPU) # MPS uses CPU activity profiler

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        # Use record_function to label specific parts of our code
        with record_function("forward_pass"):
            predictions = probe(dummy_embeddings)
        
        with record_function("loss_computation"):
            loss = distance_l1_loss(predictions, dummy_gold_dists, dummy_lengths)

        loss.backward()
        
        with record_function("backward_pass"):
            loss.backward()
        
        with record_function("optimizer_step"):
            optimizer.step()
            
        # For MPS, synchronize to ensure all ops are complete
        if device.type == 'mps':
            torch.mps.synchronize()

    print("\n--- Profiler Results ---")
    # Sort by self_cpu_time_total for MPS, as GPU time isn't separately reported.
    # This will still show where the expensive operations are.
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a single training step.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--probe-rank", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    profile_training_step(
        args.batch_size, args.seq_len, args.embedding_dim, args.probe_rank, torch.device(args.device)
    )
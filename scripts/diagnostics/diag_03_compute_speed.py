# scripts/diagnostics/diag_03_compute_speed.py
import argparse
import sys
import time

import torch
import torch.nn as nn

# Add src to path
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from torch_probe.probe_models import DepthProbe, DistanceProbe
# --- CHANGE: Import the actual loss functions ---
from torch_probe.loss_functions import depth_l1_loss, distance_l1_loss
# --- END CHANGE ---

def time_compute_loop(
    probe: nn.Module,
    batch: torch.Tensor,
    gold_labels: torch.Tensor, # <-- ADD gold_labels
    lengths: torch.Tensor,     # <-- ADD lengths
    loss_fn: callable,         # <-- ADD loss_fn
    iterations: int,
    device: torch.device,
):
    """Times a tight forward/backward/step loop for a given probe."""
    probe.to(device)
    batch = batch.to(device)
    gold_labels = gold_labels.to(device)
    lengths = lengths.to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.001)

    # Warm-up run
    for _ in range(10):
        optimizer.zero_grad()
        output = probe(batch)
        # --- CHANGE: Use the real loss function ---
        loss = loss_fn(output, gold_labels, lengths)
        # --- END CHANGE ---
        loss.backward()
        optimizer.step()

    if str(device) == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        output = probe(batch)
        # --- CHANGE: Use the real loss function ---
        loss = loss_fn(output, gold_labels, lengths)
        # --- END CHANGE ---
        loss.backward()
        optimizer.step()
    
    if str(device) == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    iters_per_sec = iterations / total_time
    time_per_iter_ms = (total_time / iterations) * 1000

    print(f"\n--- Results for {probe.__class__.__name__} ---")
    print(f"Total time for {iterations} steps: {total_time:.4f} seconds")
    print(f"Steps per second: {iters_per_sec:.2f}")
    print(f"Average time per step: {time_per_iter_ms:.4f} ms")
    print("------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose pure model computation speed.")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--probe-rank", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"--- Diagnosis 3: Pure Computation Speed on device '{device}' ---")

    dummy_batch = torch.randn(args.batch_size, args.seq_len, args.embedding_dim)
    dummy_lengths = torch.randint(low=args.seq_len//2, high=args.seq_len+1, size=(args.batch_size,))

    # Test Distance Probe
    dist_probe = DistanceProbe(args.embedding_dim, args.probe_rank)
    dummy_gold_dist = torch.randn(args.batch_size, args.seq_len, args.seq_len)
    time_compute_loop(dist_probe, dummy_batch, dummy_gold_dist, dummy_lengths, distance_l1_loss, args.iterations, device)

    # Test Depth Probe
    depth_probe = DepthProbe(args.embedding_dim, args.probe_rank)
    dummy_gold_depth = torch.randn(args.batch_size, args.seq_len)
    time_compute_loop(depth_probe, dummy_batch, dummy_gold_depth, dummy_lengths, depth_l1_loss, args.iterations, device)
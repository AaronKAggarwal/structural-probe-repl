# scripts/diagnostics/diag_09_seq_len_scaling.py
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from torch_probe.probe_models import DistanceProbe
from torch_probe.loss_functions import distance_l1_loss

def test_seq_len_scaling(
    seq_len: int,
    batch_size: int,
    embedding_dim: int,
    probe_rank: int,
    iterations: int,
    device: torch.device,
):
    tensor_shape = (batch_size, seq_len, seq_len, probe_rank)
    num_elements = batch_size * seq_len * seq_len * probe_rank
    size_mb = (num_elements * 4) / (1024 * 1024)

    print(f"\n--- Testing Step Time for Sequence Length: {seq_len} ---")
    print(f"Intermediate Tensor Shape: {tensor_shape}")
    print(f"Intermediate Tensor Size: {size_mb:.2f} MB")

    # --- Setup ---
    probe = DistanceProbe(embedding_dim, probe_rank).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.001)

    dummy_embeddings = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    dummy_gold_dists = torch.randn(batch_size, seq_len, seq_len, device=device)
    dummy_lengths = torch.full((batch_size,), seq_len, device=device)

    # Warm-up run
    for _ in range(10):
        optimizer.zero_grad()
        predictions = probe(dummy_embeddings)
        loss = distance_l1_loss(predictions, dummy_gold_dists, dummy_lengths)
        loss.backward()
        optimizer.step()
    
    if device.type == 'mps':
        torch.mps.synchronize()

    # --- Timed Run ---
    start_time = time.perf_counter()
    for _ in tqdm(range(iterations), desc=f"Steps (S={seq_len})"):
        optimizer.zero_grad()
        predictions = probe(dummy_embeddings)
        loss = distance_l1_loss(predictions, dummy_gold_dists, dummy_lengths)
        loss.backward()
        optimizer.step()
        if device.type == 'mps':
            torch.mps.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_step_ms = (total_time / iterations) * 1000

    print("\n--- Results ---")
    print(f"Total time for {iterations} steps: {total_time:.4f} seconds")
    print(f"Average time per step: {time_per_step_ms:.4f} ms")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose step time scaling with sequence length.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--probe-rank", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    test_seq_len_scaling(
        args.seq_len, args.batch_size, args.embedding_dim, args.probe_rank, args.iterations, torch.device(args.device)
    )
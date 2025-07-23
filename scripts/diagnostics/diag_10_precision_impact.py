# scripts/diagnostics/diag_10_precision_impact.py
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

def benchmark_precision(
    precision: torch.dtype,
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
    probe_rank: int,
    iterations: int,
    device: torch.device,
):
    print(f"\n--- Benchmarking with Precision: {str(precision)} ---")
    
    probe = DistanceProbe(embedding_dim, probe_rank).to(device).to(precision)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.001)

    dummy_embeddings = torch.randn(batch_size, seq_len, embedding_dim, device=device, dtype=precision)
    dummy_gold_dists = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=precision)
    dummy_lengths = torch.full((batch_size,), seq_len, device=device)
    
    # Warm-up
    for _ in range(10):
        optimizer.zero_grad()
        predictions = probe(dummy_embeddings)
        loss = distance_l1_loss(predictions, dummy_gold_dists, dummy_lengths)
        loss.backward()
        optimizer.step()
    
    if device.type == 'mps':
        torch.mps.synchronize()

    start_time = time.perf_counter()
    for _ in tqdm(range(iterations), desc=f"Steps ({str(precision)})"):
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

    print(f"Result: Average time per step = {time_per_step_ms:.4f} ms")
    print("---------------------------------")
    return time_per_step_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark training step with different precisions.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--probe-rank", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Benchmark Float32
    time_fp32 = benchmark_precision(
        torch.float32, args.batch_size, args.seq_len, args.embedding_dim, args.probe_rank, args.iterations, device
    )
    
    # Benchmark Float16
    time_fp16 = benchmark_precision(
        torch.float16, args.batch_size, args.seq_len, args.embedding_dim, args.probe_rank, args.iterations, device
    )
    
    print("\n--- Final Comparison ---")
    print(f"Time with float32: {time_fp32:.4f} ms/step")
    print(f"Time with float16: {time_fp16:.4f} ms/step")
    if time_fp32 > 0:
        speedup = (time_fp32 / time_fp16)
        print(f"Speedup with float16: {speedup:.2f}x")
    print("------------------------")
# scripts/diagnostics/diag_11_alternative_forward.py
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

from torch_probe.probe_models import DistanceProbe as DistanceProbeOriginal

class DistanceProbeAlternative(nn.Module):
    """
    An alternative DistanceProbe that uses the identity ||a-b||^2 = ||a||^2 - 2a^Tb + ||b||^2
    to avoid creating a massive intermediate tensor.
    """
    def __init__(self, embedding_dim: int, probe_rank: int):
        super().__init__()
        self.projection_layer = nn.Linear(embedding_dim, probe_rank, bias=False)

    def forward(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        projected_embeddings = self.projection_layer(embeddings_batch)
        
        # norms_sq shape: (batch_size, max_seq_len, 1)
        norms_sq = torch.sum(projected_embeddings.pow(2), dim=2, keepdim=True)
        
        # dot_products shape: (batch_size, max_seq_len, max_seq_len)
        dot_products = torch.bmm(projected_embeddings, projected_embeddings.transpose(1, 2))
        
        # Use broadcasting to get the pairwise squared distances
        # norms_sq.transpose(1,2) shape: (batch_size, 1, max_seq_len)
        # norms_sq shape: (batch_size, max_seq_len, 1)
        # The addition and subtraction broadcast correctly.
        squared_distances = norms_sq.transpose(1, 2) - 2 * dot_products + norms_sq
        return squared_distances

def benchmark_forward_pass(
    probe: nn.Module,
    dummy_data: torch.Tensor,
    iterations: int,
    device: torch.device,
):
    probe.to(device)
    dummy_data = dummy_data.to(device)

    # Warm-up
    for _ in range(10):
        _ = probe(dummy_data)
    if device.type == 'mps':
        torch.mps.synchronize()

    start_time = time.perf_counter()
    for _ in tqdm(range(iterations), desc=f"Forward pass ({probe.__class__.__name__})"):
        _ = probe(dummy_data)
        if device.type == 'mps':
            torch.mps.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_pass_ms = (total_time / iterations) * 1000
    
    print(f"Result: Average time per forward pass = {time_per_pass_ms:.4f} ms")
    return time_per_pass_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DistanceProbe forward pass implementations.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--probe-rank", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    dummy_embeddings = torch.randn(args.batch_size, args.seq_len, args.embedding_dim)

    print(f"--- Diagnosis 11: Alternative Forward Pass on device '{device}' ---")

    # Benchmark Original Probe
    probe_orig = DistanceProbeOriginal(args.embedding_dim, args.probe_rank)
    time_orig = benchmark_forward_pass(probe_orig, dummy_embeddings, args.iterations, device)

    # Benchmark Alternative Probe
    probe_alt = DistanceProbeAlternative(args.embedding_dim, args.probe_rank)
    time_alt = benchmark_forward_pass(probe_alt, dummy_embeddings, args.iterations, device)
    
    print("\n--- Final Comparison ---")
    print(f"Original Forward Pass Time:    {time_orig:.4f} ms/pass")
    print(f"Alternative Forward Pass Time: {time_alt:.4f} ms/pass")
    if time_alt > 0:
        speedup = time_orig / time_alt
        print(f"Speedup with Alternative: {speedup:.2f}x")
    print("------------------------")
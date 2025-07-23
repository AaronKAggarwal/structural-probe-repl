# scripts/diagnostics/diag_06_data_transfer_speed.py
import argparse
import time

import torch
from tqdm import tqdm

def test_data_transfer(
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
    iterations: int,
    device: torch.device,
):
    print(f"\n--- Testing Data Transfer Speed for Batch Size: {batch_size} ---")
    
    # Create a template tensor on CPU
    cpu_tensor = torch.randn(batch_size, seq_len, embedding_dim)

    # Warm-up
    for _ in range(10):
        _ = cpu_tensor.to(device)
    if device.type == 'mps':
        torch.mps.synchronize()

    start_time = time.perf_counter()
    for _ in tqdm(range(iterations), desc="Transferring batches"):
        tensor_on_device = cpu_tensor.to(device)
        # Crucial for accurate timing on asynchronous devices like MPS/CUDA
        if device.type == 'mps':
            torch.mps.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    transfers_per_sec = iterations / total_time
    time_per_transfer_ms = (total_time / iterations) * 1000

    print("\n--- Results ---")
    print(f"Total time for {iterations} transfers: {total_time:.4f} seconds")
    print(f"Transfers per second: {transfers_per_sec:.2f}")
    print(f"Average time per transfer: {time_per_transfer_ms:.4f} ms")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose CPU-to-Device data transfer speed.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    test_data_transfer(
        args.batch_size, args.seq_len, args.embedding_dim, args.iterations, torch.device(args.device)
    )
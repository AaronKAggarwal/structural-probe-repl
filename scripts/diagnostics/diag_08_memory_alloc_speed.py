# scripts/diagnostics/diag_08_memory_alloc_speed.py
import argparse
import time

import torch
from tqdm import tqdm

def test_memory_allocation(
    batch_size: int,
    seq_len: int,
    probe_rank: int,
    iterations: int,
    device: torch.device,
):
    tensor_shape = (batch_size, seq_len, seq_len, probe_rank)
    num_elements = batch_size * seq_len * seq_len * probe_rank
    size_mb = (num_elements * 4) / (1024 * 1024)

    print(f"\n--- Testing Memory Allocation for Batch Size: {batch_size} ---")
    print(f"Tensor Shape: {tensor_shape}")
    print(f"Tensor Size: {size_mb:.2f} MB")

    # Warm-up
    for _ in range(10):
        # Create and immediately let it go out of scope to be deallocated
        _ = torch.randn(tensor_shape, device=device)
    
    if device.type == 'mps':
        torch.mps.synchronize()

    start_time = time.perf_counter()
    for _ in tqdm(range(iterations), desc="Allocating tensors"):
        _ = torch.randn(tensor_shape, device=device)
        # Synchronize INSIDE the loop to measure each alloc/dealloc cycle accurately
        if device.type == 'mps':
            torch.mps.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    allocs_per_sec = iterations / total_time
    time_per_alloc_ms = (total_time / iterations) * 1000

    print("\n--- Results ---")
    print(f"Total time for {iterations} allocations: {total_time:.4f} seconds")
    print(f"Allocations per second: {allocs_per_sec:.2f}")
    print(f"Average time per allocation: {time_per_alloc_ms:.4f} ms")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose memory allocation overhead.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--probe-rank", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200) # Lower iterations, this can be slow
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    test_memory_allocation(
        args.batch_size, args.seq_len, args.probe_rank, args.iterations, torch.device(args.device)
    )
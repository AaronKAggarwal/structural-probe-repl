# scripts/diagnostics/diag_07_collate_speed.py
import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from torch_probe.dataset import collate_probe_batch

def test_collate_speed(
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
    iterations: int,
):
    print(f"\n--- Testing Collate Function Speed for Batch Size: {batch_size} ---")

    # Create a single dummy item, mimicking the output of ProbeDataset.__getitem__
    dummy_item = {
        "embeddings": torch.randn(seq_len, embedding_dim),
        "gold_labels": torch.randn(seq_len, seq_len),
        "tokens": ["word"] * seq_len,
        "head_indices": list(range(seq_len)),
        "upos_tags": ["X"] * seq_len,
        "xpos_tags": ["X"] * seq_len,
        "length": seq_len,
    }
    
    # Create a list of these items to simulate a batch
    batch_to_collate = [dummy_item] * batch_size

    # Warm-up
    for _ in range(10):
        _ = collate_probe_batch(batch_to_collate)

    start_time = time.perf_counter()
    for _ in tqdm(range(iterations), desc="Collating batches"):
        _ = collate_probe_batch(batch_to_collate)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    collations_per_sec = iterations / total_time
    time_per_collation_ms = (total_time / iterations) * 1000

    print("\n--- Results ---")
    print(f"Total time for {iterations} collations: {total_time:.4f} seconds")
    print(f"Collations per second: {collations_per_sec:.2f}")
    print(f"Average time per collation: {time_per_collation_ms:.4f} ms")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose collate_fn speed.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5000) # Use more iterations as this should be fast
    args = parser.parse_args()
    
    test_collate_speed(
        args.batch_size, args.seq_len, args.embedding_dim, args.iterations
    )
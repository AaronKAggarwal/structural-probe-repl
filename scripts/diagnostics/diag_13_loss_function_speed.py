# scripts/diagnostics/diag_13_loss_function_speed.py
import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import torch

# --- Add src to path ---
# This allows the script to be run from the project root `structural-probe-repl/`
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


# --- Loss Function Implementations to Compare ---

# Version 1: Hewitt & Manning's Original (Adapted to be a standalone function)
def hm_original_loss(
    predictions: torch.Tensor, label_batch: torch.Tensor, length_batch: torch.Tensor
) -> torch.Tensor:
    """Vectorized loss based on the original H&M implementation."""
    word_pair_dims = (1, 2)
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float().clamp(min=1e-9) # Clamped for safety

    if total_sents > 0:
        loss_per_sent = torch.sum(
            torch.abs(predictions_masked - labels_masked), dim=word_pair_dims
        )
        normalized_loss_per_sent = loss_per_sent / squared_lengths
        batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
        batch_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
    return batch_loss


# Version 2: Your Initial Loop-based Implementation
def your_initial_loop_loss(
    predicted_sq_distances: torch.Tensor,
    gold_distances: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Iterative, loop-based loss function."""
    batch_size, max_len, _ = predicted_sq_distances.shape
    batch_loss_sum = torch.tensor(0.0, device=predicted_sq_distances.device)
    total_valid_sents = 0

    for i in range(batch_size):
        l = lengths[i].item()
        if l < 2:
            continue
        total_valid_sents += 1
        pred_sent_dists = predicted_sq_distances[i, :l, :l]
        gold_sent_dists = gold_distances[i, :l, :l]
        current_sent_abs_diff = torch.abs(pred_sent_dists - gold_sent_dists)
        sum_abs_diff_sent_full_matrix = torch.sum(current_sent_abs_diff)
        if l > 0:
            normalized_sent_loss = sum_abs_diff_sent_full_matrix / (l * l)
            batch_loss_sum += normalized_sent_loss

    if total_valid_sents > 0:
        return batch_loss_sum / total_valid_sents
    else:
        return torch.tensor(
            0.0, device=predicted_sq_distances.device, requires_grad=True
        )


# Version 3: Your New Vectorized Implementation
def your_new_vectorized_loss(
    predicted_sq_distances: torch.Tensor,
    gold_distances: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Modern, fully vectorized loss function."""
    token_mask_1d = (
        torch.arange(predicted_sq_distances.size(1), device=lengths.device)[None, :]
        < lengths[:, None]
    )
    pair_mask_2d = token_mask_1d.unsqueeze(2) & token_mask_1d.unsqueeze(1)
    abs_diff = torch.abs(predicted_sq_distances - gold_distances)
    masked_abs_diff = abs_diff * pair_mask_2d
    loss_per_sent = masked_abs_diff.sum(dim=[1, 2])
    squared_lengths = lengths.pow(2).float().clamp(min=1e-9)
    normalized_loss_per_sent = loss_per_sent / squared_lengths
    valid_sents_mask = lengths > 0
    if valid_sents_mask.any():
        batch_loss = normalized_loss_per_sent[valid_sents_mask].mean()
    else:
        return torch.tensor(
            0.0, device=predicted_sq_distances.device, requires_grad=True
        )
    return batch_loss


# --- Benchmarking Logic ---

def time_function(
    func: Callable,
    args: tuple,
    device: torch.device,
    iterations: int,
    warmup_steps: int,
) -> float:
    """Times a function, handling warm-up and device synchronization."""
    # Warm-up
    for _ in range(warmup_steps):
        _ = func(*args)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # Timed run
    start_time = time.time()
    for _ in range(iterations):
        _ = func(*args)
    if device.type in ["mps", "cuda"]:
        torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
    end_time = time.time()

    return (end_time - start_time) / iterations * 1000  # Return average time in ms


def run_benchmark(
    batch_size: int,
    seq_len: int,
    device_str: str,
    iterations: int = 100,
    warmup: int = 10,
):
    print(f"--- Diagnosis 13: Loss Function Speed Comparison on '{device_str}' ---")
    print(
        f"Params: batch_size={batch_size}, seq_len={seq_len}, iterations={iterations}, warmup={warmup}\n"
    )

    device = torch.device(device_str)
    
    # Create dummy data directly on the target device to avoid measuring data transfer
    predictions = torch.randn(batch_size, seq_len, seq_len, device=device)
    gold_labels = torch.randn(batch_size, seq_len, seq_len, device=device)
    lengths = torch.randint(
        low=seq_len // 2, high=seq_len + 1, size=(batch_size,), device=device
    )

    loss_functions = {
        "H&M Original (Vectorized)": hm_original_loss,
        "Your Initial (Loop-based)": your_initial_loop_loss,
        "Your New (Vectorized)": your_new_vectorized_loss,
    }

    results = {}
    for name, func in loss_functions.items():
        print(f"Benchmarking '{name}'...")
        avg_time_ms = time_function(
            func, (predictions, gold_labels, lengths), device, iterations, warmup
        )
        results[name] = avg_time_ms
        print(f"  -> Average time: {avg_time_ms:.4f} ms")

    print("\n--- Final Comparison ---")
    loop_time = results["Your Initial (Loop-based)"]
    hm_time = results["H&M Original (Vectorized)"]
    new_time = results["Your New (Vectorized)"]
    
    print(f"{'Implementation':<30} | {'Time per Batch (ms)':<25} | {'Speedup vs. Loop'}")
    print("-" * 75)
    print(f"{'Your Initial (Loop-based)':<30} | {loop_time:<25.4f} | 1.0x")
    print(f"{'H&M Original (Vectorized)':<30} | {hm_time:<25.4f} | {loop_time / hm_time:.2f}x")
    print(f"{'Your New (Vectorized)':<30} | {new_time:<25.4f} | {loop_time / new_time:.2f}x")
    print("-" * 75)
    print("\nNOTE: `torch.mps.synchronize()` was used for accurate timing on MPS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark probe loss functions.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    run_benchmark(
        args.batch_size,
        args.seq_len,
        args.device,
        args.iterations,
        args.warmup,
    )
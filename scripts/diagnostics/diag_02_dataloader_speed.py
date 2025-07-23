# scripts/diagnostics/diag_02_dataloader_speed.py
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from torch_probe.dataset import ProbeDataset, collate_probe_batch

def time_dataloader_iteration(
    conllu_path: str,
    hdf5_path: str,
    batch_size: int,
    num_workers: int,
    limit: int,
    preload: bool, # This flag is the key
):
    print(f"\n--- Testing DataLoader with num_workers={num_workers}, batch_size={batch_size}, preload={preload} ---")
    try:
        start_init = time.perf_counter()
        
        # *** CORE FIX: Pass the 'preload' argument to the constructor ***
        dataset = ProbeDataset(
            conllu_filepath=conllu_path,
            hdf5_filepath=hdf5_path,
            embedding_layer_index=1,
            probe_task_type="distance",
            preload=preload  # <-- THIS IS THE FIX
        )

        init_time = time.perf_counter() - start_init
        print(f"Dataset initialization time: {init_time:.4f} seconds")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_probe_batch,
            shuffle=False,
        )
        
        num_batches = len(dataloader)
        if limit > 0 and limit < num_batches:
            num_batches = limit

        start_iter = time.perf_counter()
        
        pbar = tqdm(enumerate(dataloader), total=num_batches, desc="Iterating batches")
        for i, batch in pbar:
            if limit > 0 and i >= limit - 1:
                break
            pass # Just iterate
        
        end_iter = time.perf_counter()

        total_time = end_iter - start_iter
        total_samples = num_batches * batch_size
        batches_per_sec = num_batches / total_time
        samples_per_sec = total_samples / total_time

        print("\n--- Results ---")
        print(f"Total iteration time for {num_batches} batches: {total_time:.4f} seconds")
        print(f"Batches per second: {batches_per_sec:.2f}")
        print(f"Samples per second: {samples_per_sec:.2f}")

    except Exception as e:
        print(f"\nERROR during test: {e}", file=sys.stderr)
    finally:
        # The original ProbeDataset doesn't need this anymore, but it's safe
        if hasattr(dataset, 'close_hdf5'):
            dataset.close_hdf5()
        print("-----------------------------------------------------")


if __name__ == "__main__":
    # Required for MPS + num_workers > 0
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Diagnose DataLoader speed.")
    parser.add_argument("conllu_path", type=str, help="Path to CoNLL-U file.")
    parser.add_argument("hdf5_path", type=str, help="Path to HDF5 file.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of batches to process (0 for all).")
    
    args = parser.parse_args()

    print("--- Diagnosis 2: DataLoader Speed ---")
    
    # Test 1: Original method (on-disk, single worker)
    time_dataloader_iteration(args.conllu_path, args.hdf5_path, args.batch_size, num_workers=0, limit=args.limit, preload=False)
    
    # Test 2: Parallelized on-disk loading
    time_dataloader_iteration(args.conllu_path, args.hdf5_path, args.batch_size, num_workers=4, limit=args.limit, preload=False)

    # Test 3: Preloaded into RAM (single worker is fine here)
    time_dataloader_iteration(args.conllu_path, args.hdf5_path, args.batch_size, num_workers=0, limit=args.limit, preload=True)
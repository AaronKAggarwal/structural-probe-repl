# scripts/diagnostics/diag_01_io_speed.py
import argparse
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np

def test_io_speed(hdf5_path: str, iterations: int):
    """Tests raw HDF5 read performance."""
    print("--- Diagnosis 1: Raw Disk I/O Speed ---")
    print(f"HDF5 File: {hdf5_path}")
    print(f"Testing with {iterations} random reads...")

    if not Path(hdf5_path).exists():
        print(f"ERROR: HDF5 file not found at {hdf5_path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(hdf5_path, "r") as hf:
        keys = list(hf.keys())
        if not keys:
            print("ERROR: HDF5 file contains no keys (sentences).", file=sys.stderr)
            sys.exit(1)

        start_time = time.perf_counter()
        for _ in range(iterations):
            key = random.choice(keys)
            _ = hf[key][()]  # The actual read operation
        end_time = time.perf_counter()

    total_time = end_time - start_time
    reads_per_second = iterations / total_time
    time_per_read_ms = (total_time / iterations) * 1000

    print("\n--- Results ---")
    print(f"Total time for {iterations} reads: {total_time:.4f} seconds")
    print(f"Reads per second: {reads_per_second:.2f}")
    print(f"Average time per read: {time_per_read_ms:.4f} ms")
    print("---------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose raw HDF5 I/O speed.")
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of random reads to perform."
    )
    args = parser.parse_args()
    test_io_speed(args.hdf5_path, args.iterations)
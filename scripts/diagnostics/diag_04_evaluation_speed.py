# scripts/diagnostics/diag_04_evaluation_speed.py
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.evaluate import evaluate_probe
from torch_probe.loss_functions import distance_l1_loss, depth_l1_loss
from torch_probe.probe_models import DepthProbe, DistanceProbe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose evaluation function speed.")
    parser.add_argument("conllu_path", type=str, help="Path to CoNLL-U file for dev set.")
    parser.add_argument("hdf5_path", type=str, help="Path to HDF5 file for dev set.")
    parser.add_argument("--probe-type", type=str, choices=["distance", "depth"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"--- Diagnosis 4: Evaluation Speed on device '{device}' ---")

    dataset = ProbeDataset(
        conllu_filepath=args.conllu_path,
        hdf5_filepath=args.hdf5_path,
        embedding_layer_index=1,
        probe_task_type=args.probe_type,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_probe_batch
    )
    
    if args.probe_type == "distance":
        probe_model = DistanceProbe(dataset.embedding_dim, probe_rank=128)
        loss_fn = distance_l1_loss
    else:
        probe_model = DepthProbe(dataset.embedding_dim, probe_rank=128)
        loss_fn = depth_l1_loss
        
    probe_model.to(device)

    print(f"Evaluating {len(dataset)} sentences in {len(dataloader)} batches...")
    start_time = time.perf_counter()
    
    _ = evaluate_probe(
        probe_model=probe_model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        probe_type=args.probe_type,
        filter_by_non_punct_len=True,
        punctuation_strategy="upos",
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print("\n--- Results ---")
    print(f"Total time for evaluate_probe(): {total_time:.4f} seconds")
    print("-----------------------------------")
    
    dataset.close_hdf5()
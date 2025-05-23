# tests/smoke/test_probe_pipeline_smoke.py
import pytest
import torch
from torch.utils.data import DataLoader
import numpy as np # For np.allclose in potential value checks later
from pathlib import Path

# Assuming your project structure allows these imports when pytest runs from root
# If not, you might need conftest.py or adjustments to sys.path for smoke tests too.
from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.probe_models import DistanceProbe, DepthProbe
from torch_probe.loss_functions import distance_l1_loss, depth_l1_loss

# --- Configuration for the Smoke Test ---
# Use a small, accessible subset of your actual data for a quick test.
# These paths point to the whykay-01 data vendored into src/legacy/
# Adjust if your project root for running pytest is different or if data moves.
# For CI, this data would need to be available.
# For now, this assumes running pytest from the project root.

# Path to the root of the structural-probe-repl project
# This helps make paths more robust if tests are run from different locations.
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

CONLLU_DEV_FILE = str(PROJECT_ROOT / "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu")
HDF5_DEV_FILE = str(PROJECT_ROOT / "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.elmo-layers.hdf5")
ELMO_LAYER_INDEX = 2
EMBEDDING_DIM = 1024 # ELMo default
PROBE_RANK = 32
BATCH_SIZE = 2

# Check if data files exist, skip tests if not (useful for CI or clean checkouts)
data_files_exist = Path(CONLLU_DEV_FILE).exists() and Path(HDF5_DEV_FILE).exists()
skip_if_no_data = pytest.mark.skipif(not data_files_exist, reason="Sample data files not found for smoke test")

# --- Smoke Tests ---

@skip_if_no_data
def test_distance_probe_pipeline_smoke():
    """
    Smoke test for the distance probe:
    - Dataset loading
    - DataLoader iteration
    - Probe model forward pass
    - Loss calculation
    """
    print(f"\n--- Smoke Test: Distance Probe Pipeline ---")
    print(f"Using CoNLL-U: {CONLLU_DEV_FILE}")
    print(f"Using HDF5: {HDF5_DEV_FILE}")

    try:
        dataset = ProbeDataset(
            conllu_filepath=CONLLU_DEV_FILE,
            hdf5_filepath=HDF5_DEV_FILE,
            embedding_layer_index=ELMO_LAYER_INDEX,
            probe_task_type="distance",
            embedding_dim=EMBEDDING_DIM
        )
        assert len(dataset) > 0, "Dataset should not be empty"

        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_probe_batch)
        probe_model = DistanceProbe(embedding_dim=EMBEDDING_DIM, probe_rank=PROBE_RANK)
        
        print(f"\nDistance Probe Parameters ({sum(p.numel() for p in probe_model.parameters())} total):")
        for name, param in probe_model.named_parameters():
            if param.requires_grad:
                print(f"  {name:<30} Shape: {str(list(param.data.shape)):<15} Requires Grad: {param.requires_grad}")
                assert param.data.shape == (PROBE_RANK, EMBEDDING_DIM) # For the nn.Linear weight

        batch = next(iter(data_loader))
        embeddings_b = batch["embeddings_batch"]
        gold_labels_b = batch["labels_batch"]
        lengths_b = batch["lengths_batch"]

        print(f"\nFirst Batch Shapes (Distance):")
        print(f"  Input Embeddings: {embeddings_b.shape}")
        assert embeddings_b.ndim == 3
        assert embeddings_b.shape[0] <= BATCH_SIZE
        assert embeddings_b.shape[2] == EMBEDDING_DIM

        predicted_sq_dists = probe_model(embeddings_b)
        print(f"  Predicted Sq Distances: {predicted_sq_dists.shape}")
        assert predicted_sq_dists.ndim == 3
        assert predicted_sq_dists.shape[0] == embeddings_b.shape[0] # Batch size
        assert predicted_sq_dists.shape[1] == embeddings_b.shape[1] # Max seq len
        assert predicted_sq_dists.shape[2] == embeddings_b.shape[1] # Max seq len

        print(f"  Gold Sq Distances: {gold_labels_b.shape}")
        assert gold_labels_b.shape == predicted_sq_dists.shape
        
        print(f"  Lengths: {lengths_b.tolist()}")
        assert lengths_b.shape[0] == embeddings_b.shape[0]

        loss_val = distance_l1_loss(predicted_sq_dists, gold_labels_b, lengths_b)
        print(f"Calculated Distance L1 Loss (untrained): {loss_val.item()}")
        assert loss_val.ndim == 0, "Loss should be a scalar"
        assert not torch.isnan(loss_val), "Loss should not be NaN"
        assert not torch.isinf(loss_val), "Loss should not be Inf"

    finally:
        if 'dataset' in locals() and hasattr(dataset, 'close_hdf5'):
            dataset.close_hdf5()
    print("--- Distance Probe Pipeline Smoke Test PASSED ---")


@skip_if_no_data
def test_depth_probe_pipeline_smoke():
    """
    Smoke test for the depth probe:
    - Dataset loading
    - DataLoader iteration
    - Probe model forward pass
    - Loss calculation
    """
    print(f"\n--- Smoke Test: Depth Probe Pipeline ---")
    print(f"Using CoNLL-U: {CONLLU_DEV_FILE}")
    print(f"Using HDF5: {HDF5_DEV_FILE}")

    try:
        dataset = ProbeDataset(
            conllu_filepath=CONLLU_DEV_FILE,
            hdf5_filepath=HDF5_DEV_FILE,
            embedding_layer_index=ELMO_LAYER_INDEX,
            probe_task_type="depth",
            embedding_dim=EMBEDDING_DIM
        )
        assert len(dataset) > 0, "Dataset should not be empty"

        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_probe_batch)
        probe_model = DepthProbe(embedding_dim=EMBEDDING_DIM, probe_rank=PROBE_RANK)

        print(f"\nDepth Probe Parameters ({sum(p.numel() for p in probe_model.parameters())} total):")
        for name, param in probe_model.named_parameters():
            if param.requires_grad:
                print(f"  {name:<30} Shape: {str(list(param.data.shape)):<15} Requires Grad: {param.requires_grad}")
                assert param.data.shape == (PROBE_RANK, EMBEDDING_DIM)

        batch = next(iter(data_loader))
        embeddings_b = batch["embeddings_batch"]
        gold_labels_b = batch["labels_batch"]
        lengths_b = batch["lengths_batch"]

        print(f"\nFirst Batch Shapes (Depth):")
        print(f"  Input Embeddings: {embeddings_b.shape}")
        assert embeddings_b.ndim == 3
        assert embeddings_b.shape[0] <= BATCH_SIZE
        assert embeddings_b.shape[2] == EMBEDDING_DIM
        
        predicted_sq_depths = probe_model(embeddings_b)
        print(f"  Predicted Sq Depths: {predicted_sq_depths.shape}")
        assert predicted_sq_depths.ndim == 2
        assert predicted_sq_depths.shape[0] == embeddings_b.shape[0] # Batch size
        assert predicted_sq_depths.shape[1] == embeddings_b.shape[1] # Max seq len
        
        print(f"  Gold Sq Depths: {gold_labels_b.shape}")
        assert gold_labels_b.shape == predicted_sq_depths.shape
        
        print(f"  Lengths: {lengths_b.tolist()}")
        assert lengths_b.shape[0] == embeddings_b.shape[0]

        loss_val = depth_l1_loss(predicted_sq_depths, gold_labels_b, lengths_b)
        print(f"Calculated Depth L1 Loss (untrained): {loss_val.item()}")
        assert loss_val.ndim == 0, "Loss should be a scalar"
        assert not torch.isnan(loss_val), "Loss should not be NaN"
        assert not torch.isinf(loss_val), "Loss should not be Inf"

    finally:
        if 'dataset' in locals() and hasattr(dataset, 'close_hdf5'):
            dataset.close_hdf5()
    print("--- Depth Probe Pipeline Smoke Test PASSED ---")
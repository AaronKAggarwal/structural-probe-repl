# tests/unit/torch_probe/test_embedding_loader.py
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Import the new unified function
from src.torch_probe.utils.embedding_loader import load_embeddings_for_sentence

# --- Fixtures for creating different HDF5 formats ---

@pytest.fixture
def legacy_elmo_hdf5_file() -> str:
    """Creates a dummy HDF5 file mimicking the legacy ELMo format."""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp_f:
        filepath = tmp_f.name
    with h5py.File(filepath, "w") as hf:
        # 3 layers, as is standard for ELMo
        hf.create_dataset("0", data=np.random.rand(3, 10, 1024).astype(np.float32))
    yield filepath
    Path(filepath).unlink()

@pytest.fixture
def modern_bert_hdf5_file() -> str:
    """
    Creates a dummy HDF5 file mimicking the modern format from extract_embeddings.py,
    where only a subset of layers is stored.
    """
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp_f:
        filepath = tmp_f.name
    with h5py.File(filepath, "w") as hf:
        # Simulate extracting layers 0, 7, and 12 from a model
        layers_in_file = [0, 7, 12]
        hf.attrs['layers_extracted_indices'] = json.dumps(layers_in_file)
        
        # The data stack will have 3 layers, corresponding to the extracted ones
        hf.create_dataset("0", data=np.random.rand(3, 15, 768).astype(np.float32))
    yield filepath
    Path(filepath).unlink()

# --- Tests for the new unified loader function ---

def test_load_from_legacy_elmo_format(legacy_elmo_hdf5_file):
    """
    Tests that the loader correctly extracts layers by direct index from a legacy file.
    """
    with h5py.File(legacy_elmo_hdf5_file, "r") as hf:
        # Request layer 2 (which is at index 2 in the 3-layer stack)
        embeddings = load_embeddings_for_sentence(hf, "0", requested_layer=2)
        assert embeddings is not None
        assert embeddings.shape == (10, 1024)
        
        # Request an invalid layer
        embeddings_invalid = load_embeddings_for_sentence(hf, "0", requested_layer=3)
        assert embeddings_invalid is None

def test_load_from_modern_bert_format(modern_bert_hdf5_file):
    """
    Tests that the loader correctly uses the metadata to find the right layer
    in a modern, subset-of-layers HDF5 file.
    """
    with h5py.File(modern_bert_hdf5_file, "r") as hf:
        # Request absolute layer 7. The loader should know this is at index 1 in the stack.
        embeddings = load_embeddings_for_sentence(hf, "0", requested_layer=7)
        assert embeddings is not None
        assert embeddings.shape == (15, 768)
        
        # Request absolute layer 12. The loader should know this is at index 2.
        embeddings_l12 = load_embeddings_for_sentence(hf, "0", requested_layer=12)
        assert embeddings_l12 is not None
        assert embeddings_l12.shape == (15, 768)

        # Check that the loaded embeddings are indeed from the correct stack index
        stack = hf["0"][()]
        assert np.array_equal(embeddings, stack[1, :, :])
        assert np.array_equal(embeddings_l12, stack[2, :, :])

def test_load_from_modern_bert_format_invalid_layer(modern_bert_hdf5_file):
    """
    Tests that requesting a layer that wasn't extracted returns None.
    """
    with h5py.File(modern_bert_hdf5_file, "r") as hf:
        # Request layer 8, which was not in the original [0, 7, 12] list
        embeddings_invalid = load_embeddings_for_sentence(hf, "0", requested_layer=8)
        assert embeddings_invalid is None

def test_load_missing_sentence_key(legacy_elmo_hdf5_file):
    """Tests that a missing sentence key returns None."""
    with h5py.File(legacy_elmo_hdf5_file, "r") as hf:
        embeddings = load_embeddings_for_sentence(hf, "99", requested_layer=0)
        assert embeddings is None
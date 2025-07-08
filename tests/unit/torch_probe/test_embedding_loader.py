# tests/unit/torch_probe/test_embedding_loader.py
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.torch_probe.utils.embedding_loader import (
    load_elmo_embeddings_for_sentence,  # Adjust import
)


@pytest.fixture
def dummy_hdf5_file_fixture() -> str:
    # Create a temporary HDF5 file for testing
    # Using NamedTemporaryFile to get a filepath
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp_f:
        filepath = tmp_f.name

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("0", data=np.random.rand(3, 10, 1024).astype(np.float32))
        hf.create_dataset("1", data=np.random.rand(3, 5, 1024).astype(np.float32))
        # Malformed entry (2D instead of 3D)
        hf.create_dataset("2", data=np.random.rand(7, 1024).astype(np.float32))
        # Correct shape but not a typical ELMo layer count
        hf.create_dataset("3", data=np.random.rand(1, 8, 1024).astype(np.float32))
        # Entry that is not a numpy array
        hf.create_dataset("not_an_array", data="this is a string")

    yield filepath  # Provide the filepath to the test
    Path(filepath).unlink()  # Cleanup


def test_load_valid_sentence_and_layer(dummy_hdf5_file_fixture):
    with h5py.File(dummy_hdf5_file_fixture, "r") as hf:
        embeddings = load_elmo_embeddings_for_sentence(
            hf, "0", 2
        )  # Layer 2 (3rd layer)
        assert embeddings is not None
        assert embeddings.shape == (10, 1024)
        assert embeddings.dtype == np.float32

        embeddings_l0 = load_elmo_embeddings_for_sentence(hf, "1", 0)  # Layer 0
        assert embeddings_l0 is not None
        assert embeddings_l0.shape == (5, 1024)


def test_load_missing_sentence_key(dummy_hdf5_file_fixture):
    with h5py.File(dummy_hdf5_file_fixture, "r") as hf:
        embeddings = load_elmo_embeddings_for_sentence(
            hf, "99", 0
        )  # Key "99" doesn't exist
        assert embeddings is None


def test_load_invalid_layer_index(dummy_hdf5_file_fixture):
    with h5py.File(dummy_hdf5_file_fixture, "r") as hf:
        embeddings = load_elmo_embeddings_for_sentence(
            hf, "0", 3
        )  # Layer 3 doesn't exist (0,1,2)
        assert embeddings is None
        embeddings_neg = load_elmo_embeddings_for_sentence(
            hf, "0", -1
        )  # Negative layer
        assert embeddings_neg is None


def test_load_malformed_ndim_data(dummy_hdf5_file_fixture):
    with h5py.File(dummy_hdf5_file_fixture, "r") as hf:
        embeddings = load_elmo_embeddings_for_sentence(
            hf, "2", 0
        )  # Key "2" has 2D data
        assert embeddings is None  # Should fail ndim check


def test_load_single_layer_stack(
    dummy_hdf5_file_fixture,
):  # e.g. if ELMo only had 1 output layer
    with h5py.File(dummy_hdf5_file_fixture, "r") as hf:
        embeddings = load_elmo_embeddings_for_sentence(
            hf, "3", 0
        )  # Key "3" has shape (1, 8, 1024)
        assert embeddings is not None
        assert embeddings.shape == (8, 1024)


def test_load_non_array_data(dummy_hdf5_file_fixture):
    with h5py.File(dummy_hdf5_file_fixture, "r") as hf:
        embeddings = load_elmo_embeddings_for_sentence(hf, "not_an_array", 0)
        assert embeddings is None

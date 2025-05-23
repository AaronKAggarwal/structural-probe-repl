# tests/unit/torch_probe/test_dataset.py
import pytest
import torch
import numpy as np
import h5py
from pathlib import Path
import textwrap
import tempfile

# Assuming these are in src/torch_probe/
from src.torch_probe.dataset import ProbeDataset, collate_probe_batch # Adjust import
from torch.utils.data import DataLoader

# --- Test Fixtures ---
@pytest.fixture
def dummy_conllu_content_fixture() -> str:
    # Use textwrap.dedent for cleaner multi-line strings
    # Ensure a blank line separates sentences
    return textwrap.dedent("""\
        # sent_id = 1
        # text = This is a test.
        1	This	_	_	_	_	4	nsubj	_	_
        2	is	_	_	_	_	4	cop	_	_
        3	a	_	_	_	_	4	det	_	_
        4	test	_	_	_	_	0	root	_	_

        # sent_id = 2
        # text = Short one
        1	Short	_	_	_	_	0	root	_	_
        2	one	_	_	_	_	1	obj	_	_
        """) # Ensure it ends with a newline if _generate_sentence_lines relies on it
             # or ensure _generate_sentence_lines handles EOF correctly.
             # My _generate_sentence_lines has `if sentence_buffer: yield sentence_buffer` which handles EOF.dedent will handle leading whitespace, ensure final newline is present or add one

@pytest.fixture
def dummy_hdf5_content_fixture(tmp_path: Path) -> str:
    filepath = tmp_path / "dummy_embeddings.hdf5"
    with h5py.File(filepath, 'w') as hf:
        # Sentence 0: 4 tokens, ELMo has 3 layers, 1024 dim
        hf.create_dataset('0', data=np.random.rand(3, 4, 1024).astype(np.float32))
        # Sentence 1: 2 tokens
        hf.create_dataset('1', data=np.random.rand(3, 2, 1024).astype(np.float32))
    return str(filepath)

@pytest.fixture
def dummy_conllu_filepath_fixture(tmp_path: Path, dummy_conllu_content_fixture: str) -> str:
    filepath = tmp_path / "test.conllu"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(dummy_conllu_content_fixture)
    return str(filepath)

# --- Tests for ProbeDataset ---
def test_probe_dataset_initialization_distance(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture):
    dataset = ProbeDataset(
        conllu_filepath=dummy_conllu_filepath_fixture,
        hdf5_filepath=dummy_hdf5_content_fixture,
        embedding_layer_index=1,
        probe_task_type="distance",
        embedding_dim=1024
    )
    assert len(dataset) == 2
    dataset.close_hdf5()

def test_probe_dataset_initialization_depth(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture):
    dataset = ProbeDataset(
        conllu_filepath=dummy_conllu_filepath_fixture,
        hdf5_filepath=dummy_hdf5_content_fixture,
        embedding_layer_index=0,
        probe_task_type="depth",
        embedding_dim=1024
    )
    assert len(dataset) == 2
    dataset.close_hdf5()

def test_probe_dataset_getitem_distance(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture):
    dataset = ProbeDataset(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture, 0, "distance", 1024)
    item0 = dataset[0]
    assert isinstance(item0, dict)
    assert "embeddings" in item0 and isinstance(item0["embeddings"], torch.Tensor)
    assert "gold_labels" in item0 and isinstance(item0["gold_labels"], torch.Tensor)
    assert item0["embeddings"].shape == (4, 1024) # Sent 0 has 4 tokens
    assert item0["gold_labels"].shape == (4, 4) # Distance matrix
    assert item0["length"] == 4
    dataset.close_hdf5()

def test_probe_dataset_getitem_depth(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture):
    dataset = ProbeDataset(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture, 1, "depth", 1024)
    item1 = dataset[1]
    assert item1["embeddings"].shape == (2, 1024) # Sent 1 has 2 tokens
    assert item1["gold_labels"].shape == (2,)    # Depth list
    assert item1["length"] == 2
    dataset.close_hdf5()

def test_probe_dataset_token_count_assertion(tmp_path: Path, dummy_conllu_filepath_fixture):
    # Create HDF5 with mismatched token count for sentence 0
    hdf5_mismatch_path = tmp_path / "mismatch.hdf5"
    with h5py.File(hdf5_mismatch_path, 'w') as hf:
        hf.create_dataset('0', data=np.random.rand(3, 5, 1024).astype(np.float32)) # 5 ELMo tokens
        hf.create_dataset('1', data=np.random.rand(3, 2, 1024).astype(np.float32))
    
    dataset = ProbeDataset(dummy_conllu_filepath_fixture, str(hdf5_mismatch_path), 0, "depth", 1024)
    with pytest.raises(AssertionError, match="Token count mismatch for sentence key '0'"):
        _ = dataset[0] # Sentence 0 in CoNLLU has 4 tokens
    dataset.close_hdf5()

# --- Tests for collate_probe_batch ---
def test_collate_probe_batch_depth(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture):
    dataset = ProbeDataset(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture, 0, "depth", 1024)
    # dataset has 2 sentences, lengths 4 and 2
    items_to_batch = [dataset[0], dataset[1]]
    
    collated_batch = collate_probe_batch(items_to_batch)
    
    assert "embeddings_batch" in collated_batch
    assert collated_batch["embeddings_batch"].shape == (2, 4, 1024) # batch_size, max_len, dim
    assert "labels_batch" in collated_batch
    assert collated_batch["labels_batch"].shape == (2, 4) # batch_size, max_len
    assert "lengths_batch" in collated_batch
    assert torch.equal(collated_batch["lengths_batch"], torch.tensor([4, 2]))
    
    # Check padding value for labels
    assert collated_batch["labels_batch"][1, 2].item() == -1.0 # Padded part of shorter sentence
    assert collated_batch["labels_batch"][1, 1].item() != -1.0 # Original part
    dataset.close_hdf5()

def test_collate_probe_batch_distance(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture):
    dataset = ProbeDataset(dummy_conllu_filepath_fixture, dummy_hdf5_content_fixture, 0, "distance", 1024)
    items_to_batch = [dataset[0], dataset[1]] # lengths 4 and 2
    
    collated_batch = collate_probe_batch(items_to_batch)
    
    assert collated_batch["embeddings_batch"].shape == (2, 4, 1024)
    assert collated_batch["labels_batch"].shape == (2, 4, 4) # batch_size, max_len, max_len
    assert torch.equal(collated_batch["lengths_batch"], torch.tensor([4, 2]))

    # Check padding value for labels
    assert collated_batch["labels_batch"][1, 2, 0].item() == -1.0 # Padded part
    assert collated_batch["labels_batch"][1, 0, 2].item() == -1.0 # Padded part
    assert collated_batch["labels_batch"][1, 1, 1].item() != -1.0 # Original part (diagonal is 0)
    dataset.close_hdf5()

"""
Unit tests for ProbeDataset and collate_probe_batch in dataset.py

Tests include initialization, __getitem__, __len__, and collation,
with dummy data files (CoNLL-U and HDF5) generated via pytest's tmp_path fixture.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
import pytest
from pathlib import Path
from typing import List, Dict

from torch_probe.dataset import ProbeDataset, collate_probe_batch

# -----------------------------------------------------------------------------
# Helpers to create dummy data
# -----------------------------------------------------------------------------

def create_dummy_conllu(path: Path, sents: List[List[str]], heads: List[List[int]], upos: List[List[str]]) -> None:
    lines = []
    for i, (tokens, hs, tags) in enumerate(zip(sents, heads, upos)):
        for j, (tok, head, tag) in enumerate(zip(tokens, hs, tags), start=1):
            lines.append(f"{j}\t{tok}\t_\t{tag}\t_\t_\t{head if head != -1 else 0}\tdep\t_\t_")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def create_dummy_hdf5(path: Path, sents: List[List[str]], layers: int = 3, dim: int = 5) -> None:
    with h5py.File(path, "w") as f:
        for i, tokens in enumerate(sents):
            # Corrected shape: (layers, tokens, dim)
            data = np.random.rand(layers, len(tokens), dim).astype(np.float32) 
            f.create_dataset(str(i), data=data)

# -----------------------------------------------------------------------------
# Dataset Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def dummy_data(tmp_path: Path):
    sents = [["A", "B", "C"], ["D", "E"]]
    heads = [[2, 3, -1], [2, -1]]  # 1-indexed roots as 0
    upos = [["NOUN"] * 3, ["VERB"] * 2]

    conllu_path = tmp_path / "dummy.conllu"
    hdf5_path = tmp_path / "dummy.hdf5"
    create_dummy_conllu(conllu_path, sents, heads, upos)
    create_dummy_hdf5(hdf5_path, sents)
    return conllu_path, hdf5_path, sents, heads


def test_dataset_init_and_len(dummy_data):
    conllu, hdf5, sents, _ = dummy_data
    ds = ProbeDataset(str(conllu), str(hdf5), 1, "depth")
    assert len(ds) == len(sents)
    ds.close_hdf5()


def test_dataset_getitem_depth(dummy_data):
    conllu, hdf5, sents, heads = dummy_data
    ds = ProbeDataset(str(conllu), str(hdf5), 1, "depth")
    item = ds[0]
    assert isinstance(item, dict)
    assert item["embeddings"].shape[0] == len(sents[0])
    assert item["gold_labels"].shape == (len(sents[0]),)
    assert item["tokens"] == sents[0]
    expected = [h-1 if h >= 0 else -1 for h in heads[0]]
    assert item["head_indices"] == expected
    assert item["length"] == len(sents[0])
    ds.close_hdf5()


def test_dataset_getitem_distance(dummy_data):
    conllu, hdf5, sents, _ = dummy_data
    ds = ProbeDataset(str(conllu), str(hdf5), 0, "distance")
    item = ds[1]
    assert item["gold_labels"].shape == (len(sents[1]), len(sents[1]))
    ds.close_hdf5()


def test_dataset_embedding_mismatch_raises(tmp_path: Path):
    sents = [["A", "B"]]
    heads = [[2, -1]]
    upos = [["X"] * 2]
    conllu_path = tmp_path / "mismatch.conllu"
    hdf5_path = tmp_path / "mismatch.hdf5"
    create_dummy_conllu(conllu_path, sents, heads, upos)
    create_dummy_hdf5(hdf5_path, [["A"]])  # fewer embeddings than tokens

    with pytest.raises(AssertionError):
        ds = ProbeDataset(str(conllu_path), str(hdf5_path), 1, "depth")
        _ = ds[0]
        ds.close_hdf5()


def test_dataset_invalid_file(tmp_path: Path):
    invalid_path = tmp_path / "nonexistent.conllu"
    with pytest.raises(IOError):
        ProbeDataset(str(invalid_path), str(invalid_path), 0, "depth")


def test_dataset_out_of_bounds(dummy_data):
    conllu, hdf5, _, _ = dummy_data
    ds = ProbeDataset(str(conllu), str(hdf5), 1, "depth")
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    ds.close_hdf5()

# -----------------------------------------------------------------------------
# Collation Tests
# -----------------------------------------------------------------------------

def test_collate_depth(dummy_data):
    conllu, hdf5, _, _ = dummy_data
    ds = ProbeDataset(str(conllu), str(hdf5), 0, "depth")
    batch = [ds[0], ds[1]]
    collated = collate_probe_batch(batch)
    assert collated["embeddings_batch"].shape[0] == 2
    assert collated["labels_batch"].shape[1] == max(item["length"] for item in batch)
    assert (collated["labels_batch"] != -1).sum() == sum(len(b["gold_labels"]) for b in batch)
    ds.close_hdf5()


def test_collate_distance(dummy_data):
    conllu, hdf5, _, _ = dummy_data
    ds = ProbeDataset(str(conllu), str(hdf5), 0, "distance")
    batch = [ds[0], ds[1]]
    collated = collate_probe_batch(batch)
    assert collated["labels_batch"].ndim == 3
    assert collated["labels_batch"].shape[1] == collated["labels_batch"].shape[2]
    ds.close_hdf5()


def test_collate_batch_size_one(dummy_data):
    conllu, hdf5, _, _ = dummy_data
    ds = ProbeDataset(str(conllu), str(hdf5), 0, "depth")
    batch = [ds[1]]
    collated = collate_probe_batch(batch)
    assert collated["embeddings_batch"].shape[0] == 1
    ds.close_hdf5()


def test_collate_empty_batch():
    result = collate_probe_batch([])
    assert result["embeddings_batch"].numel() == 0
    assert result["labels_batch"].numel() == 0
    assert result["lengths_batch"].numel() == 0
    assert result["tokens_batch"] == []
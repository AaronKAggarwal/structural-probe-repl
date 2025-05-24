"""Independent unit tests for probe models and loss functions."""

from __future__ import annotations

import math
import torch
import pytest

from torch_probe.probe_models import DistanceProbe, DepthProbe
from torch_probe.loss_functions import (
    distance_l1_loss,
    depth_l1_loss,
)

# ---------------------------------------------------------------------------
# Probe model tests
# ---------------------------------------------------------------------------

def test_distance_probe_init_and_forward_shapes():
    embed_dim = 4
    rank = 2
    batch, seq_len = 3, 5
    probe = DistanceProbe(embed_dim, rank)

    # weight shape check
    assert probe.projection_layer.weight.shape == (rank, embed_dim)
    # requires_grad
    assert probe.projection_layer.weight.requires_grad is True

    x = torch.randn(batch, seq_len, embed_dim)
    out = probe(x)
    assert out.shape == (batch, seq_len, seq_len)
    assert out.dtype == torch.float32


def test_depth_probe_init_and_forward_shapes():
    embed_dim = 7
    rank = 3
    batch, seq_len = 2, 1  # sequence length 1 edge case
    probe = DepthProbe(embed_dim, rank)
    assert probe.projection_layer.weight.shape == (rank, embed_dim)

    x = torch.randn(batch, seq_len, embed_dim)
    out = probe(x)
    assert out.shape == (batch, seq_len)
    assert out.dtype == torch.float32


def test_distance_probe_invalid_input():
    probe = DistanceProbe(embedding_dim=4, probe_rank=2)
    bad_input = torch.randn(2, 3, 5)  # last dim mismatch
    with pytest.raises(ValueError):
        _ = probe(bad_input)


def test_probe_parameters_learnable():
    torch.manual_seed(0)
    probe = DepthProbe(embedding_dim=4, probe_rank=2)
    optim = torch.optim.SGD(probe.parameters(), lr=0.1)
    x = torch.randn(4, 6, 4)
    output = probe(x).sum()
    output.backward()
    # gradient non‑null
    assert probe.projection_layer.weight.grad is not None
    # take one step
    before = probe.projection_layer.weight.clone().detach()
    optim.step()
    after = probe.projection_layer.weight.clone().detach()
    assert not torch.allclose(before, after)

# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

def test_depth_l1_loss_simple():
    # batch 2, max_len 3
    pred = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    gold = torch.tensor([[1.0, 0.0, 0.0], [2.0, 4.0, 0.0]])
    lengths = torch.tensor([2, 2])  # only first two positions in each row valid

    # manual calculation
    diff = torch.tensor([0.0, 2.0, 1.0, 0.0])
    expected = diff.mean()  # 0.75
    loss = depth_l1_loss(pred, gold, lengths)
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-4)


def test_distance_l1_loss_simple():
    # One sentence length 3
    pred = torch.tensor([[[0.0, 1.0, 4.0],
                          [1.0, 0.0, 1.0],
                          [4.0, 1.0, 0.0]]])
    gold = torch.tensor([[[0.0, 1.0, 5.0],
                          [1.0, 0.0, 1.0],
                          [5.0, 1.0, 0.0]]])
    lengths = torch.tensor([3])

    # valid pairs: (0,1),(0,2),(1,2)
    expected = (0 + 1 + 0) / 3  # 0.3333
    loss = distance_l1_loss(pred, gold, lengths)
    assert math.isclose(loss.item(), expected, rel_tol=1e-4)


def test_distance_loss_zero_when_no_pairs():
    pred = torch.zeros(2, 1, 1)
    gold = torch.zeros_like(pred)
    lengths = torch.tensor([1, 1])
    loss = distance_l1_loss(pred, gold, lengths)
    assert loss.item() == 0.0


def test_depth_loss_handles_all_padding():
    pred = torch.zeros(2, 3)
    gold = torch.zeros(2, 3)
    lengths = torch.tensor([0, 0])
    loss = depth_l1_loss(pred, gold, lengths)
    assert loss.item() == 0.0


def test_distance_loss_batch_varied_lengths():
    pred = torch.zeros(2, 4, 4)
    gold = torch.zeros(2, 4, 4)
    lengths = torch.tensor([4, 2])
    # Only sentence 0 contributes 4*3/2 =6 pairs, sentence1 contributes 1 pair
    loss = distance_l1_loss(pred, gold, lengths)
    assert loss.item() == 0.0


def test_depth_loss_value_no_padding():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gold = torch.tensor([[1.5, 3.0], [2.0, 3.5]])
    lengths = torch.tensor([2, 2])
    expected = torch.abs(pred - gold).mean()
    loss = depth_l1_loss(pred, gold, lengths)
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-5)

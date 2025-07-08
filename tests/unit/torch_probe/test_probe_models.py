# tests/unit/torch_probe/test_probe_models.py
import pytest
import torch

from src.torch_probe.probe_models import DepthProbe, DistanceProbe  # Adjust import


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10])
@pytest.mark.parametrize("embedding_dim", [100, 768])
@pytest.mark.parametrize("probe_rank", [10, 32])
def test_distance_probe_forward_shape(
    batch_size, max_seq_len, embedding_dim, probe_rank
):
    probe = DistanceProbe(embedding_dim, probe_rank)
    dummy_embeddings = torch.randn(batch_size, max_seq_len, embedding_dim)

    output = probe(dummy_embeddings)

    assert output.shape == (batch_size, max_seq_len, max_seq_len)
    assert output.dtype == torch.float32
    for param in probe.parameters():
        assert param.requires_grad  # Ensure parameters are learnable


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10])
@pytest.mark.parametrize("embedding_dim", [100, 768])
@pytest.mark.parametrize("probe_rank", [10, 32])
def test_depth_probe_forward_shape(batch_size, max_seq_len, embedding_dim, probe_rank):
    probe = DepthProbe(embedding_dim, probe_rank)
    dummy_embeddings = torch.randn(batch_size, max_seq_len, embedding_dim)

    output = probe(dummy_embeddings)

    assert output.shape == (batch_size, max_seq_len)
    assert output.dtype == torch.float32
    for param in probe.parameters():
        assert param.requires_grad


def test_distance_probe_invalid_input_dim():
    probe = DistanceProbe(embedding_dim=100, probe_rank=10)
    dummy_embeddings_2d = torch.randn(4, 100)  # Missing batch or seq_len
    dummy_embeddings_wrong_emb_dim = torch.randn(4, 5, 50)

    with pytest.raises(ValueError):
        probe(dummy_embeddings_2d)
    with pytest.raises(ValueError):
        probe(dummy_embeddings_wrong_emb_dim)


def test_depth_probe_invalid_input_dim():
    probe = DepthProbe(embedding_dim=100, probe_rank=10)
    dummy_embeddings_2d = torch.randn(4, 100)
    dummy_embeddings_wrong_emb_dim = torch.randn(4, 5, 50)

    with pytest.raises(ValueError):
        probe(dummy_embeddings_2d)
    with pytest.raises(ValueError):
        probe(dummy_embeddings_wrong_emb_dim)


# It might be useful to add a test for specific values if we had a known B
# For now, shape and type are primary.

# src/torch_probe/probe_models.py
from typing import Tuple
import torch
import torch.nn as nn

class DistanceProbe(nn.Module):
    """
    Structural probe for predicting squared L2 distance between word pairs
    in a projected space.
    p(h) = Bh
    Predicted distance = ||p(h_i) - p(h_j)||^2
    """
    def __init__(self, embedding_dim: int, probe_rank: int):
        """
        Args:
            embedding_dim: Dimensionality of the input word embeddings.
            probe_rank: Dimensionality of the projected space (k).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.probe_rank = probe_rank
        # Using nn.Linear for the projection. B is the weight matrix of this layer.
        # B has shape (probe_rank, embedding_dim)
        self.projection_layer = nn.Linear(embedding_dim, probe_rank, bias=False)
        # Default initialization for nn.Linear is Kaiming uniform, which is reasonable.

    def forward(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_batch: Tensor of shape (batch_size, max_seq_len, embedding_dim).
        
        Returns:
            predicted_squared_distances: Tensor of shape (batch_size, max_seq_len, max_seq_len).
        """
        if embeddings_batch.ndim != 3 or embeddings_batch.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Input embeddings_batch expected shape (batch_size, max_seq_len, {self.embedding_dim}), "
                f"got {embeddings_batch.shape}"
            )

        # Project embeddings: B h^T
        # embeddings_batch shape: (B, S, E)
        # self.projection_layer(embeddings_batch) output shape: (B, S, K) where K is probe_rank
        projected_embeddings = self.projection_layer(embeddings_batch) # p_h, shape (B, S, K)

        # Efficiently compute all pairwise squared L2 distances
        # ||p_i - p_j||^2 = ||p_i||^2 - 2 <p_i, p_j> + ||p_j||^2
        # Let p_h be of shape (B, S, K)
        
        # norms_sq = torch.sum(projected_embeddings.pow(2), dim=2, keepdim=True) # Shape: (B, S, 1)
        # dot_products = torch.bmm(projected_embeddings, projected_embeddings.transpose(1, 2)) # Shape: (B, S, S)
        
        # # pairwise_dist_sq = norms_sq.transpose(1,2) - 2 * dot_products + norms_sq # Broadcasting issues here
        # # Need careful broadcasting for norms_sq for all pairs.

        # Simpler way with broadcasting for difference:
        p_h_expanded_i = projected_embeddings.unsqueeze(2) # Shape: (B, S, 1, K)
        p_h_expanded_j = projected_embeddings.unsqueeze(1) # Shape: (B, 1, S, K)
        
        # Difference vector for all pairs (p_i - p_j)
        diff_vectors = p_h_expanded_i - p_h_expanded_j # Shape: (B, S, S, K) via broadcasting
        
        # Squared L2 norm of difference vectors
        squared_distances = torch.sum(diff_vectors.pow(2), dim=3) # Sum over K dimension. Shape: (B, S, S)
        
        return squared_distances


class DepthProbe(nn.Module):
    """
    Structural probe for predicting squared L2 norm of a projected word embedding.
    p(h) = Bh
    Predicted depth = ||p(h)||^2
    """
    def __init__(self, embedding_dim: int, probe_rank: int):
        """
        Args:
            embedding_dim: Dimensionality of the input word embeddings.
            probe_rank: Dimensionality of the projected space (k).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.probe_rank = probe_rank
        self.projection_layer = nn.Linear(embedding_dim, probe_rank, bias=False)

    def forward(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_batch: Tensor of shape (batch_size, max_seq_len, embedding_dim).

        Returns:
            predicted_squared_depths: Tensor of shape (batch_size, max_seq_len).
        """
        if embeddings_batch.ndim != 3 or embeddings_batch.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Input embeddings_batch expected shape (batch_size, max_seq_len, {self.embedding_dim}), "
                f"got {embeddings_batch.shape}"
            )

        # Project embeddings
        projected_embeddings = self.projection_layer(embeddings_batch) # p_h, shape (B, S, K)

        # Squared L2 norm for each projected embedding
        squared_depths = torch.sum(projected_embeddings.pow(2), dim=2) # Sum over K dimension. Shape: (B, S)
        
        return squared_depths

if __name__ == '__main__':
    # Example Usage (primarily for quick checks, real tests in pytest)
    B, S, E, K = 2, 5, 10, 3 # Batch, SeqLen, EmbeddingDim, ProbeRank

    # Test DistanceProbe
    print("--- Testing DistanceProbe ---")
    dist_probe = DistanceProbe(embedding_dim=E, probe_rank=K)
    print("Probe parameters:", list(dist_probe.parameters()))
    dummy_embeddings = torch.randn(B, S, E)
    pred_sq_dist = dist_probe(dummy_embeddings)
    print("Input shape:", dummy_embeddings.shape)
    print("Predicted squared distances shape:", pred_sq_dist.shape) # Expected (B, S, S)
    assert pred_sq_dist.shape == (B, S, S)
    print("Sample output (first sentence, first row):", pred_sq_dist[0,0,:])

    # Test DepthProbe
    print("\n--- Testing DepthProbe ---")
    depth_probe = DepthProbe(embedding_dim=E, probe_rank=K)
    pred_sq_depth = depth_probe(dummy_embeddings)
    print("Input shape:", dummy_embeddings.shape)
    print("Predicted squared depths shape:", pred_sq_depth.shape) # Expected (B, S)
    assert pred_sq_depth.shape == (B, S)
    print("Sample output (first sentence):", pred_sq_depth[0,:])
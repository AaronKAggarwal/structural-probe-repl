# src/torch_probe/probe_models.py
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
        self.projection_layer = nn.Linear(embedding_dim, probe_rank, bias=False)

    def forward(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_batch: Tensor of shape (batch_size, max_seq_len, embedding_dim).

        Returns:
            predicted_squared_distances: Tensor of shape (batch_size, max_seq_len, max_seq_len).
        """
        if (
            embeddings_batch.ndim != 3
            or embeddings_batch.shape[2] != self.embedding_dim
        ):
            raise ValueError(
                f"Input embeddings_batch expected shape (batch_size, max_seq_len, {self.embedding_dim}), "
                f"got {embeddings_batch.shape}"
            )

        # Project embeddings: B h^T
        # embeddings_batch shape: (B, S, E) -> projected_embeddings shape: (B, S, K)
        projected_embeddings = self.projection_layer(embeddings_batch)

        # Use the identity ||a-b||^2 = ||a||^2 - 2a^Tb + ||b||^2 to compute distances
        # without creating large intermediate tensors.

        # norms_sq shape: (B, S, 1)
        norms_sq = torch.sum(projected_embeddings.pow(2), dim=2, keepdim=True)

        # dot_products shape: (B, S, S)
        dot_products = torch.bmm(projected_embeddings, projected_embeddings.transpose(1, 2))

        # Use broadcasting to get the pairwise squared distances.
        # norms_sq.transpose(1,2) shape: (B, 1, S)
        # norms_sq shape: (B, S, 1)
        # The addition and subtraction broadcast correctly across the matrix.
        squared_distances = norms_sq.transpose(1, 2) - 2 * dot_products + norms_sq

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
        if (
            embeddings_batch.ndim != 3
            or embeddings_batch.shape[2] != self.embedding_dim
        ):
            raise ValueError(
                f"Input embeddings_batch expected shape (batch_size, max_seq_len, {self.embedding_dim}), "
                f"got {embeddings_batch.shape}"
            )

        projected_embeddings = self.projection_layer(
            embeddings_batch
        )

        squared_depths = torch.sum(
            projected_embeddings.pow(2), dim=2
        )

        return squared_depths


# __main__ block for standalone testing.
if __name__ == "__main__":
    B, S, E, K = 2, 5, 10, 3
    print("--- Testing DistanceProbe ---")
    dist_probe = DistanceProbe(embedding_dim=E, probe_rank=K)
    print("Probe parameters:", list(dist_probe.parameters()))
    dummy_embeddings = torch.randn(B, S, E)
    pred_sq_dist = dist_probe(dummy_embeddings)
    print("Input shape:", dummy_embeddings.shape)
    print("Predicted squared distances shape:", pred_sq_dist.shape)
    assert pred_sq_dist.shape == (B, S, S)
    print("Sample output (first sentence, first row):", pred_sq_dist[0, 0, :])

    print("\n--- Testing DepthProbe ---")
    depth_probe = DepthProbe(embedding_dim=E, probe_rank=K)
    pred_sq_depth = depth_probe(dummy_embeddings)
    print("Input shape:", dummy_embeddings.shape)
    print("Predicted squared depths shape:", pred_sq_depth.shape)
    assert pred_sq_depth.shape == (B, S)
    print("Sample output (first sentence):", pred_sq_depth[0, :])
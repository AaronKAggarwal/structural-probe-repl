# src/torch_probe/loss_functions.py
from typing import Optional

import torch


def distance_l1_loss(
    predicted_sq_distances: torch.Tensor,  # Shape (batch_size, max_seq_len, max_seq_len)
    gold_distances: torch.Tensor,        # Shape (batch_size, max_seq_len, max_seq_len)
    lengths: torch.Tensor,               # Shape (batch_size,)
) -> torch.Tensor:
    """
    Calculates L1 loss between PREDICTED SQUARED L2 distances and GOLD NON-SQUARED tree distances.
    This implementation is fully vectorized to run efficiently on a GPU/MPS.
    It normalizes loss per-sentence by L^2 and then averages across the batch.
    """
    if not (
        predicted_sq_distances.ndim == 3
        and gold_distances.ndim == 3
        and predicted_sq_distances.shape == gold_distances.shape
    ):
        raise ValueError("Input tensors must be 3D and have matching shapes.")
    if lengths.ndim != 1 or lengths.shape[0] != predicted_sq_distances.shape[0]:
        raise ValueError("Lengths tensor has incompatible shape.")

    # Create a 1D mask from lengths (batch_size, max_seq_len)
    # True for valid tokens, False for padding
    token_mask_1d = torch.arange(predicted_sq_distances.size(1), device=lengths.device)[None, :] < lengths[:, None]

    # Expand to a 2D mask for pairs (batch_size, max_seq_len, max_seq_len)
    # A pair (i, j) is valid only if both token i and token j are valid.
    pair_mask_2d = token_mask_1d.unsqueeze(2) & token_mask_1d.unsqueeze(1)

    # Calculate absolute difference on the entire batch
    abs_diff = torch.abs(predicted_sq_distances - gold_distances)

    # Apply the mask to zero out losses from padded pairs
    masked_abs_diff = abs_diff * pair_mask_2d

    # Sum the loss for each sentence in the batch (sum over the two seq_len dimensions)
    loss_per_sent = masked_abs_diff.sum(dim=[1, 2])

    # Normalize by L^2, where L is the sentence length.
    # Add a small epsilon to avoid division by zero for sentences of length 0.
    squared_lengths = lengths.pow(2).float()
    squared_lengths = squared_lengths.clamp(min=1e-9) # Avoid division by zero
    
    normalized_loss_per_sent = loss_per_sent / squared_lengths

    # Average the normalized loss across the batch
    # Only consider sentences with length > 0 in the mean
    valid_sents_mask = lengths > 0
    if valid_sents_mask.any():
        batch_loss = normalized_loss_per_sent[valid_sents_mask].mean()
    else:
        # Handle case where the entire batch is empty sentences
        return torch.tensor(0.0, device=predicted_sq_distances.device, requires_grad=True)

    return batch_loss


def depth_l1_loss(
    predicted_sq_depths: torch.Tensor,  # Shape (batch_size, max_seq_len)
    gold_depths: torch.Tensor,        # Shape (batch_size, max_seq_len)
    lengths: torch.Tensor,            # Shape (batch_size,)
) -> torch.Tensor:
    """
    Calculates L1 loss between PREDICTED SQUARED L2 depths and GOLD NON-SQUARED tree depths.
    This implementation is fully vectorized to run efficiently on a GPU/MPS.
    It normalizes loss per-sentence by L and then averages across the batch.
    """
    if not (
        predicted_sq_depths.ndim == 2
        and gold_depths.ndim == 2
        and predicted_sq_depths.shape == gold_depths.shape
    ):
        raise ValueError("Input tensors must be 2D and have matching shapes.")
    if lengths.ndim != 1 or lengths.shape[0] != predicted_sq_depths.shape[0]:
        raise ValueError("Lengths tensor has incompatible shape.")

    # Create a mask for valid tokens (non-padded positions)
    token_mask = torch.arange(predicted_sq_depths.size(1), device=lengths.device)[None, :] < lengths[:, None]
    
    # Calculate absolute difference on the entire batch
    abs_diff = torch.abs(predicted_sq_depths - gold_depths)
    
    # Apply the mask to zero out losses from padded tokens
    masked_abs_diff = abs_diff * token_mask
    
    # Sum the loss for each sentence in the batch
    loss_per_sent = masked_abs_diff.sum(dim=1)
    
    # Normalize by L, the sentence length.
    # Add a small epsilon to avoid division by zero.
    lengths_float = lengths.float().clamp(min=1e-9)
    normalized_loss_per_sent = loss_per_sent / lengths_float
    
    # Average the normalized loss across the batch
    valid_sents_mask = lengths > 0
    if valid_sents_mask.any():
        batch_loss = normalized_loss_per_sent[valid_sents_mask].mean()
    else:
        return torch.tensor(0.0, device=predicted_sq_depths.device, requires_grad=True)
        
    return batch_loss
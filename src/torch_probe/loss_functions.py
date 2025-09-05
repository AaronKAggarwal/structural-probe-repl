# src/torch_probe/loss_functions.py
from typing import Optional

import torch


def distance_l1_loss(
    predicted_sq_distances: torch.Tensor,  # Shape (batch_size, max_seq_len, max_seq_len)
    gold_distances: torch.Tensor,        # Shape (batch_size, max_seq_len, max_seq_len)
    lengths: torch.Tensor,               # Shape (batch_size,)
    content_token_mask: Optional[torch.Tensor] = None,  # Shape (batch_size, max_seq_len)
) -> torch.Tensor:
    """
    Calculates L1 loss between PREDICTED SQUARED L2 distances and GOLD NON-SQUARED
    tree distances. Pair selection uses unordered off-diagonal pairs (upper triangle,
    i < j). For numerical stability and evaluation parity, both prediction and gold
    distance matrices are symmetrized before differencing.

    This implementation is fully vectorized to run efficiently on a GPU/MPS.
    It normalizes loss per-sentence by n(n-1)/2 where n is the number of content tokens
    and then averages across the batch.
    
    Args:
        content_token_mask: Optional mask indicating content tokens (non-PUNCT/SYM).
                           If None, uses all valid (non-padded) tokens.
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
    
    # Apply content mask if provided, otherwise use all valid tokens
    if content_token_mask is not None:
        if content_token_mask.shape != token_mask_1d.shape:
            raise ValueError("content_token_mask shape must match (batch_size, max_seq_len)")
        effective_token_mask = token_mask_1d & content_token_mask
    else:
        effective_token_mask = token_mask_1d

    # Expand to a 2D mask for pairs (batch_size, max_seq_len, max_seq_len)
    # A pair (i, j) is valid only if both token i and token j are valid content tokens.
    pair_mask_2d = effective_token_mask.unsqueeze(2) & effective_token_mask.unsqueeze(1)

    # Restrict to unordered off-diagonal pairs: keep only the upper triangle (i < j)
    batch_size, max_len = effective_token_mask.shape
    # Upper triangle mask (excluding diagonal)
    triu_mask = torch.triu(
        torch.ones((max_len, max_len), dtype=torch.bool, device=lengths.device),
        diagonal=1,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    pair_mask_2d = pair_mask_2d & triu_mask

    # Symmetrize predictions and gold to reduce minor numerical asymmetries
    pred_sym = 0.5 * (predicted_sq_distances + predicted_sq_distances.transpose(1, 2))
    gold_sym = 0.5 * (gold_distances + gold_distances.transpose(1, 2))

    # Calculate absolute difference on the entire batch
    abs_diff = torch.abs(pred_sym - gold_sym)

    # Apply the mask to zero out losses from padded/non-content pairs
    masked_abs_diff = abs_diff * pair_mask_2d

    # Sum the loss for each sentence in the batch (sum over the two seq_len dimensions)
    loss_per_sent = masked_abs_diff.sum(dim=[1, 2])

    # Normalize by n(n-1)/2, where n is the number of content tokens.
    # Count content tokens per sentence
    counts_dtype = predicted_sq_distances.dtype
    content_token_counts = effective_token_mask.sum(dim=1).to(dtype=counts_dtype)  # (batch_size,)
    n_pairs = content_token_counts * (content_token_counts - 1) / 2
    n_pairs = n_pairs.clamp(min=1e-9)  # Avoid division by zero
    
    normalized_loss_per_sent = loss_per_sent / n_pairs

    # Average the normalized loss across the batch
    # Only consider sentences with at least 2 content tokens
    valid_sents_mask = content_token_counts >= 2
    if valid_sents_mask.any():
        batch_loss = normalized_loss_per_sent[valid_sents_mask].mean()
    else:
        # Handle case where no sentence has enough content tokens
        return torch.tensor(0.0, device=predicted_sq_distances.device, requires_grad=True)

    return batch_loss


def depth_l1_loss(
    predicted_sq_depths: torch.Tensor,  # Shape (batch_size, max_seq_len)
    gold_depths: torch.Tensor,        # Shape (batch_size, max_seq_len)
    lengths: torch.Tensor,            # Shape (batch_size,)
    content_token_mask: Optional[torch.Tensor] = None,  # Shape (batch_size, max_seq_len)
) -> torch.Tensor:
    """
    Calculates L1 loss between PREDICTED SQUARED L2 depths and GOLD NON-SQUARED tree depths.
    This implementation is fully vectorized to run efficiently on a GPU/MPS.
    It normalizes loss per-sentence by n where n is the number of content tokens
    and then averages across the batch.
    
    Args:
        content_token_mask: Optional mask indicating content tokens (non-PUNCT/SYM).
                           If None, uses all valid (non-padded) tokens.
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
    
    # Apply content mask if provided, otherwise use all valid tokens
    if content_token_mask is not None:
        if content_token_mask.shape != token_mask.shape:
            raise ValueError("content_token_mask shape must match (batch_size, max_seq_len)")
        effective_token_mask = token_mask & content_token_mask
    else:
        effective_token_mask = token_mask
    
    # Calculate absolute difference on the entire batch
    abs_diff = torch.abs(predicted_sq_depths - gold_depths)
    
    # Apply the mask to zero out losses from padded/non-content tokens
    masked_abs_diff = abs_diff * effective_token_mask
    
    # Sum the loss for each sentence in the batch
    loss_per_sent = masked_abs_diff.sum(dim=1)
    
    # Normalize by n, the number of content tokens.
    # Add a small epsilon to avoid division by zero.
    content_token_counts = effective_token_mask.sum(dim=1).float()  # (batch_size,)
    content_token_counts = content_token_counts.clamp(min=1e-9)
    normalized_loss_per_sent = loss_per_sent / content_token_counts
    
    # Average the normalized loss across the batch
    # Only consider sentences with at least 1 content token
    valid_sents_mask = content_token_counts >= 1
    if valid_sents_mask.any():
        batch_loss = normalized_loss_per_sent[valid_sents_mask].mean()
    else:
        return torch.tensor(0.0, device=predicted_sq_depths.device, requires_grad=True)
        
    return batch_loss
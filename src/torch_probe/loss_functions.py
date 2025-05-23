# src/torch_probe/loss_functions.py
from typing import Optional
import torch

def create_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Creates a boolean mask of shape (batch_size, max_len) from a lengths tensor.
    True where elements are valid (not padding), False for padding.
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    # Create a range tensor: [[0, 1, ..., max_len-1], [0, 1, ..., max_len-1], ...]
    idx_range = torch.arange(max_len, device=lengths.device).unsqueeze(0) # Shape (1, max_len)
    # Compare with lengths: lengths.unsqueeze(1) shape (batch_size, 1)
    # Result is True where idx_range < length for that row
    mask = idx_range < lengths.unsqueeze(1)  # Shape (batch_size, max_len)
    return mask

def distance_l1_loss(
    predicted_sq_distances: torch.Tensor, # Shape (batch_size, max_seq_len, max_seq_len)
    gold_sq_distances: torch.Tensor,      # Shape (batch_size, max_seq_len, max_seq_len)
    lengths: torch.Tensor                 # Shape (batch_size,)
) -> torch.Tensor:
    """
    Calculates L1 loss between predicted and gold squared distances,
    ignoring padded elements and considering only i < j pairs.
    """
    if not (predicted_sq_distances.ndim == 3 and gold_sq_distances.ndim == 3 and \
            predicted_sq_distances.shape == gold_sq_distances.shape):
        raise ValueError("Input tensors must be 3D and have matching shapes.")
    if lengths.ndim != 1 or lengths.shape[0] != predicted_sq_distances.shape[0]:
        raise ValueError("Lengths tensor has incompatible shape.")

    batch_size, max_len, _ = predicted_sq_distances.shape
    
    # Create a mask for valid tokens in each sentence
    token_mask = create_mask_from_lengths(lengths, max_len) # Shape (batch_size, max_len)
    
    # Create a mask for valid pairs (i, j) where both i and j are valid tokens
    # and i < j (to count each pair once and ignore diagonal i=i)
    pair_mask_list = []
    for i in range(batch_size):
        l = lengths[i].item()
        # Create a mask for this sentence: True for valid (row_idx < l, col_idx < l, row_idx < col_idx)
        sent_pair_mask = torch.zeros((max_len, max_len), dtype=torch.bool, device=lengths.device)
        for r in range(l):
            for c in range(r + 1, l): # c > r ensures i < j (or r < c here)
                sent_pair_mask[r, c] = True
        pair_mask_list.append(sent_pair_mask)
    
    pair_mask = torch.stack(pair_mask_list) # Shape (batch_size, max_len, max_len)

    # Select only the valid, non-padded gold labels based on the mask
    valid_gold_sq_distances = gold_sq_distances[pair_mask]
    
    # Also select the corresponding predictions
    valid_predicted_sq_distances = predicted_sq_distances[pair_mask]

    if valid_gold_sq_distances.numel() == 0: # No valid pairs (e.g., all sentences length < 2)
        return torch.tensor(0.0, device=predicted_sq_distances.device, requires_grad=True)

    loss = torch.abs(valid_predicted_sq_distances - valid_gold_sq_distances).sum()
    num_valid_pairs = valid_gold_sq_distances.numel()
    
    return loss / num_valid_pairs


def depth_l1_loss(
    predicted_sq_depths: torch.Tensor, # Shape (batch_size, max_seq_len)
    gold_sq_depths: torch.Tensor,      # Shape (batch_size, max_seq_len)
    lengths: torch.Tensor              # Shape (batch_size,)
) -> torch.Tensor:
    """
    Calculates L1 loss between predicted and gold squared depths,
    ignoring padded elements.
    """
    if not (predicted_sq_depths.ndim == 2 and gold_sq_depths.ndim == 2 and \
            predicted_sq_depths.shape == gold_sq_depths.shape):
        raise ValueError("Input tensors must be 2D and have matching shapes.")
    if lengths.ndim != 1 or lengths.shape[0] != predicted_sq_depths.shape[0]:
        raise ValueError("Lengths tensor has incompatible shape.")

    mask = create_mask_from_lengths(lengths, predicted_sq_depths.shape[1]) # Shape (B, S)
    
    valid_gold_sq_depths = gold_sq_depths[mask]
    valid_predicted_sq_depths = predicted_sq_depths[mask]

    if valid_gold_sq_depths.numel() == 0: # No valid tokens
        return torch.tensor(0.0, device=predicted_sq_depths.device, requires_grad=True)

    loss = torch.abs(valid_predicted_sq_depths - valid_gold_sq_depths).sum()
    num_valid_tokens = valid_gold_sq_depths.numel()
    
    return loss / num_valid_tokens

if __name__ == '__main__':
    # Example Usage
    B, S = 2, 5
    # For depth
    pred_depths = torch.randn(B, S)
    gold_depths = torch.randn(B, S)
    gold_depths[0, 3:] = -1 # Padding for first sentence
    gold_depths[1, 4:] = -1 # Padding for second sentence
    lengths_depth = torch.tensor([3, 4])
    loss_d = depth_l1_loss(pred_depths, gold_depths, lengths_depth)
    print(f"Depth Loss: {loss_d.item()}")

    # For distance
    pred_dists = torch.randn(B, S, S)
    gold_dists = torch.randn(B, S, S)
    # Masking for distance is more complex (handled inside the function)
    # Here, just imagine gold_dists has -1 where pairs are invalid
    lengths_dist = torch.tensor([3, 4]) # Sentence 1 has 3 tokens, Sentence 2 has 4 tokens
    loss_dist = distance_l1_loss(pred_dists, gold_dists, lengths_dist)
    print(f"Distance Loss: {loss_dist.item()}")
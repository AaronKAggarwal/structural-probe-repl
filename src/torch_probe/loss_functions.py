# src/torch_probe/loss_functions.py
from typing import Optional
import torch

def create_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = torch.max(lengths).item() if lengths.numel() > 0 else 0
        if lengths.numel() == 0 and max_len == 0 : 
             return torch.empty((0,0), dtype=torch.bool, device=lengths.device)

    idx_range = torch.arange(max_len, device=lengths.device).unsqueeze(0) 
    mask = idx_range < lengths.unsqueeze(1)  
    return mask

def distance_l1_loss(
    predicted_sq_distances: torch.Tensor, # Shape (batch_size, max_seq_len, max_seq_len)
    gold_distances: torch.Tensor,         # Shape (batch_size, max_seq_len, max_seq_len) - NON-SQUARED
    lengths: torch.Tensor                 # Shape (batch_size,)
) -> torch.Tensor:
    """
    Calculates L1 loss between PREDICTED SQUARED L2 distances and GOLD NON-SQUARED tree distances.
    Normalizes first within sentences (by dividing by L^2, where L is actual sentence length),
    considering only i < j pairs, and then averages across the batch.
    Padded elements in gold_distances are assumed to be -1 and are handled by the pair_mask.
    """
    if not (predicted_sq_distances.ndim == 3 and gold_distances.ndim == 3 and \
            predicted_sq_distances.shape == gold_distances.shape):
        raise ValueError("Input tensors must be 3D and have matching shapes for distance_l1_loss.")
    if lengths.ndim != 1 or lengths.shape[0] != predicted_sq_distances.shape[0]:
        raise ValueError("Lengths tensor has incompatible shape for distance_l1_loss.")

    batch_size, max_len, _ = predicted_sq_distances.shape
    batch_loss_sum = torch.tensor(0.0, device=predicted_sq_distances.device)
    total_valid_sents = 0

    for i in range(batch_size):
        l = lengths[i].item()
        if l < 2: # Need at least 2 tokens to form a pair
            continue
        
        total_valid_sents += 1
        
        # Extract valid parts of the matrices for this sentence
        pred_sent_dists = predicted_sq_distances[i, :l, :l]
        gold_sent_dists = gold_distances[i, :l, :l] # These are NON-SQUARED

        # Calculate L1 loss for i < j pairs
        abs_diff_sum_sent = torch.tensor(0.0, device=predicted_sq_distances.device)
        num_pairs_sent = 0
        for r_idx in range(l):
            for c_idx in range(r_idx + 1, l):
                abs_diff_sum_sent += torch.abs(pred_sent_dists[r_idx, c_idx] - gold_sent_dists[r_idx, c_idx])
                num_pairs_sent += 1
        
        if num_pairs_sent > 0 : # Should be true if l >= 2
            # H&M paper normalizes by |s|^2 (sentence_length squared for distance)
            # H&M code normalizes PairwiseDistLoss by squared_lengths (L^2)
            # If we consider num_pairs_sent = L*(L-1)/2, then L^2 is not exactly num_pairs_sent * 2
            # However, their code uses squared_lengths for normalization.
            # Let's stick to their code's normalization for now.
            # If length is l, squared_length is l*l.
            # loss_per_sent in their code is sum of abs diffs over ALL pairs (i,j) (not just i<j, and including i=j)
            # Let's re-check H&M L1DistanceLoss.
            # `loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)`
            # `self.word_pair_dims = (1,2)`. This sums over the last two dimensions (the S,S matrix).
            # `normalized_loss_per_sent = loss_per_sent / squared_lengths`
            # This means they sum abs diffs for ALL L*L pairs and divide by L*L.
            # Padded elements are masked out first.
            
            # Re-implementing H&M style:
            # Create a mask for valid elements within this sentence (LxL)
            sent_token_mask = torch.ones((l,l), dtype=torch.bool, device=pred_sent_dists.device)
            # (Their code uses labels_1s = (label_batch != -1).float(), which handles padding before this loop)
            # So, gold_sent_dists here are already valid (not -1).
            
            current_sent_abs_diff = torch.abs(pred_sent_dists - gold_sent_dists) # LxL
            sum_abs_diff_sent_full_matrix = torch.sum(current_sent_abs_diff) # Sum over all LxL elements
            
            # Normalize by L^2 as per H&M code
            if l > 0: # Avoid division by zero for zero-length sentences (though caught by l < 2)
                normalized_sent_loss = sum_abs_diff_sent_full_matrix / (l * l)
                batch_loss_sum += normalized_sent_loss
            
    if total_valid_sents > 0:
        return batch_loss_sum / total_valid_sents
    else:
        return torch.tensor(0.0, device=predicted_sq_distances.device, requires_grad=True)


def depth_l1_loss(
    predicted_sq_depths: torch.Tensor, # Shape (batch_size, max_seq_len)
    gold_depths: torch.Tensor,         # Shape (batch_size, max_seq_len) - NON-SQUARED
    lengths: torch.Tensor              # Shape (batch_size,)
) -> torch.Tensor:
    """
    Calculates L1 loss between PREDICTED SQUARED L2 depths and GOLD NON-SQUARED tree depths.
    Normalizes first within sentences (by dividing by L, actual sentence length),
    and then averages across the batch.
    Padded elements in gold_depths are assumed to be -1.
    """
    if not (predicted_sq_depths.ndim == 2 and gold_depths.ndim == 2 and \
            predicted_sq_depths.shape == gold_depths.shape):
        raise ValueError("Input tensors must be 2D and have matching shapes for depth_l1_loss.")
    if lengths.ndim != 1 or lengths.shape[0] != predicted_sq_depths.shape[0]:
        raise ValueError("Lengths tensor has incompatible shape for depth_l1_loss.")

    batch_size, max_len = predicted_sq_depths.shape
    batch_loss_sum = torch.tensor(0.0, device=predicted_sq_depths.device)
    total_valid_sents = 0

    for i in range(batch_size):
        l = lengths[i].item()
        if l == 0:
            continue
        
        total_valid_sents +=1
        
        pred_sent_depths = predicted_sq_depths[i, :l]
        gold_sent_depths = gold_depths[i, :l] # These are NON-SQUARED

        # Mask out -1s in gold_sent_depths (if any, though our current dataset doesn't pad like this internally for gold)
        # H&M code does `labels_1s = (label_batch != -1).float()` then `labels_masked = label_batch * labels_1s`
        # This implies gold labels for padded positions are set to -1.
        # Our current `collate_fn` sets padded gold labels to -1.0.
        valid_mask_sent = gold_sent_depths != -1.0
        
        if torch.sum(valid_mask_sent).item() == 0: # All tokens in this sentence were padding
            continue

        valid_pred = pred_sent_depths[valid_mask_sent]
        valid_gold = gold_sent_depths[valid_mask_sent]
        
        sum_abs_diff_sent = torch.abs(valid_pred - valid_gold).sum()
        
        # Normalize by L (actual number of non-padded tokens in this sentence, or just l?)
        # H&M code: `normalized_loss_per_sent = loss_per_sent / length_batch.float()` -> uses original length `l`
        normalized_sent_loss = sum_abs_diff_sent / l 
        batch_loss_sum += normalized_sent_loss

    if total_valid_sents > 0:
        return batch_loss_sum / total_valid_sents
    else:
        return torch.tensor(0.0, device=predicted_sq_depths.device, requires_grad=True)
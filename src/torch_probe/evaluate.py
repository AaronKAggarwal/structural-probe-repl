# src/torch_probe/evaluate.py
from typing import List, Dict, Any, Callable, Optional
import torch
import torch.nn as nn 
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from torch.utils.data import DataLoader

# Assuming loss_functions.py might be in the same directory or a utils subdirectory
# from .loss_functions import create_mask_from_lengths # Not directly used in this version of evaluate_probe

PUNCTUATION_UPOS_TAGS = {"PUNCT"} 

# _get_valid_pairs helper is not strictly needed by the current calculate_spearmanr for distance
# but can be kept if useful for other analyses or alternative Spearman calculations.
# def _get_valid_pairs(matrix: np.ndarray, length: int, only_upper_triangle: bool = True) -> List[float]:
#     valid_elements = []
#     for r in range(length):
#         start_c = r + 1 if only_upper_triangle else 0
#         for c in range(start_c, length):
#             if r == c and only_upper_triangle: 
#                 continue
#             valid_elements.append(matrix[r, c])
#     return valid_elements

# src/torch_probe/evaluate.py

def calculate_spearmanr(
    all_predictions: List[np.ndarray], 
    all_gold_labels: List[np.ndarray], 
    all_lengths: List[int],
    probe_type: str
) -> float:
    if not all_predictions or not all_gold_labels or not all_lengths:
        return 0.0
    if not (len(all_predictions) == len(all_gold_labels) == len(all_lengths)):
        raise ValueError("Input lists (predictions, gold_labels, lengths) must have the same length.")

    sentence_level_rhos = [] 

    for i in range(len(all_predictions)):
        pred_data_sent_full = all_predictions[i] # Full array, possibly with padding
        gold_data_sent_full = all_gold_labels[i] # Full array, possibly with padding
        length = all_lengths[i]                  # True length of data in this sentence

        if probe_type == "depth":
            if length < 2: 
                continue 
            
            # Slice to actual length BEFORE passing to spearmanr
            actual_preds = pred_data_sent_full[:length]
            actual_golds = gold_data_sent_full[:length]
            
            if np.std(actual_preds) == 0 and np.std(actual_golds) == 0:
                rho_sent = 1.0 
            elif np.std(actual_preds) == 0 or np.std(actual_golds) == 0:
                rho_sent = 0.0 
            else:
                rho_sent, _ = spearmanr(actual_preds, actual_golds)
            
            if not np.isnan(rho_sent):
                sentence_level_rhos.append(rho_sent)
        
        elif probe_type == "distance":
            if length < 2: # Or length < 3 if we want per-word rows to have at least 2 elements
                continue
            
            per_word_rhos_in_sentence = []
            pred_matrix = pred_data_sent_full[:length, :length] 
            gold_matrix = gold_data_sent_full[:length, :length]

            for word_idx in range(length):
                # For word `word_idx`, its distances to other words are in pred_matrix[word_idx]
                # We need to exclude the self-distance at pred_matrix[word_idx, word_idx]
                pred_row = np.delete(pred_matrix[word_idx, :], word_idx) # Correctly gets L-1 elements
                gold_row = np.delete(gold_matrix[word_idx, :], word_idx) # Correctly gets L-1 elements

                if len(pred_row) < 2: # This means original length L must be >= 3
                    continue
                
                if np.std(pred_row) == 0 and np.std(gold_row) == 0:
                    rho_word = 1.0 
                elif np.std(pred_row) == 0 or np.std(gold_row) == 0:
                    rho_word = 0.0 
                else:
                    rho_word, _ = spearmanr(pred_row, gold_row)

                if not np.isnan(rho_word):
                    per_word_rhos_in_sentence.append(rho_word)
            
            if per_word_rhos_in_sentence: 
                sentence_level_rhos.append(np.mean(per_word_rhos_in_sentence))
        else:
            raise ValueError(f"Unknown probe_type for Spearman: {probe_type}")
            
    if not sentence_level_rhos:
        return 0.0
    
    return np.mean(sentence_level_rhos).item()


def calculate_uuas(
    all_predicted_distances: List[np.ndarray],
    all_gold_head_indices: List[List[int]],
    all_lengths: List[int],
    all_upos_tags: List[List[str]]
) -> float:
    if not all_predicted_distances or not all_gold_head_indices or \
       not all_lengths or not all_upos_tags:
        return 0.0
    if not (len(all_predicted_distances) == len(all_gold_head_indices) == \
            len(all_lengths) == len(all_upos_tags)):
        raise ValueError("Input lists for UUAS must have the same length.")

    total_sentences = len(all_predicted_distances) 
    if total_sentences == 0:
        return 0.0

    total_uuas_score = 0.0
    num_evaluable_sentences = 0

    for i in range(total_sentences):
        pred_dist_matrix_full = all_predicted_distances[i] # Already sliced to (L, L) by evaluate_probe
        gold_heads_sent_full = all_gold_head_indices[i]    # Already sliced to (L,) by evaluate_probe
        length_sent_full = all_lengths[i]                  # True length L
        upos_tags_sent = all_upos_tags[i]                  # Already sliced to (L,) by evaluate_probe

        if length_sent_full < 2:
            continue

        non_punct_original_indices = [
            idx for idx, tag in enumerate(upos_tags_sent) # No need to slice upos_tags_sent again
            if tag not in PUNCTUATION_UPOS_TAGS
        ]
        num_non_punct_tokens = len(non_punct_original_indices)

        if num_non_punct_tokens < 2: 
            continue
        
        non_punct_indices_map: Dict[int, int] = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(non_punct_original_indices)
        }

        gold_edges_non_punct = set()
        for orig_token_idx in non_punct_original_indices: 
            orig_head_idx = gold_heads_sent_full[orig_token_idx]
            if orig_head_idx != -1 and orig_head_idx in non_punct_indices_map: 
                new_token_idx = non_punct_indices_map[orig_token_idx]
                new_head_idx = non_punct_indices_map[orig_head_idx]
                gold_edges_non_punct.add(tuple(sorted((new_token_idx, new_head_idx))))
        
        if not gold_edges_non_punct:
            num_evaluable_sentences += 1 
            continue

        num_evaluable_sentences += 1 

        pred_dist_matrix_non_punct = pred_dist_matrix_full[np.ix_(non_punct_original_indices, non_punct_original_indices)]
        
        predicted_edges_non_punct = set()
        if num_non_punct_tokens >= 2: 
            try:
                # Ensure matrix is non-negative and suitable for MST if there are issues
                # Forcing symmetry for safety, though distances should be.
                # pred_dist_matrix_non_punct_symmetric = (pred_dist_matrix_non_punct + pred_dist_matrix_non_punct.T) / 2.0
                mst_sparse_matrix = minimum_spanning_tree(pred_dist_matrix_non_punct) # Using original
                rows, cols = mst_sparse_matrix.nonzero()
                for r, c in zip(rows, cols):
                    predicted_edges_non_punct.add(tuple(sorted((r, c))))
            except Exception as e:
                # print(f"Warning: MST calculation failed for non-punct sentence {i}: {e}") # Commented out
                pass 

        correct_edges = len(gold_edges_non_punct.intersection(predicted_edges_non_punct))
        sentence_uuas = correct_edges / len(gold_edges_non_punct) 
        total_uuas_score += sentence_uuas
            
    return total_uuas_score / num_evaluable_sentences if num_evaluable_sentences > 0 else 0.0


def calculate_root_accuracy(
    all_predicted_depths: List[np.ndarray],
    all_gold_head_indices: List[List[int]],
    all_lengths: List[int],
    all_upos_tags: List[List[str]]
) -> float:
    if not all_predicted_depths or not all_gold_head_indices or \
       not all_lengths or not all_upos_tags:
        return 0.0
    if not (len(all_predicted_depths) == len(all_gold_head_indices) == \
            len(all_lengths) == len(all_upos_tags)):
        raise ValueError("Input lists for Root Accuracy must have the same length.")

    total_sentences = len(all_predicted_depths)
    if total_sentences == 0:
        return 0.0

    correct_roots = 0
    num_evaluable_sentences = 0

    for i in range(total_sentences):
        current_pred_depths = all_predicted_depths[i] # Already sliced to actual length L
        current_gold_heads = all_gold_head_indices[i]   # Already sliced
        current_length = all_lengths[i]                 # True length L
        current_upos_tags = all_upos_tags[i]            # Already sliced

        if current_length == 0:
            continue

        non_punct_indices = [
            idx for idx, tag in enumerate(current_upos_tags) # No need to slice current_upos_tags again
            if tag not in PUNCTUATION_UPOS_TAGS
        ]

        if not non_punct_indices:
            continue 

        pred_depths_non_punct = current_pred_depths[non_punct_indices]
             
        min_depth_idx_in_non_punct_list = np.argmin(pred_depths_non_punct).item()
        predicted_root_original_idx = non_punct_indices[min_depth_idx_in_non_punct_list]
        
        actual_root_original_idx = -1
        for token_idx in range(current_length): 
            if current_gold_heads[token_idx] == -1: 
                if current_upos_tags[token_idx] not in PUNCTUATION_UPOS_TAGS:
                    actual_root_original_idx = token_idx
                    break 
        
        if actual_root_original_idx != -1: # If a valid non-punctuation gold root was found
            num_evaluable_sentences += 1
            if predicted_root_original_idx == actual_root_original_idx:
                correct_roots += 1
    
    return correct_roots / num_evaluable_sentences if num_evaluable_sentences > 0 else 0.0


def evaluate_probe(
    probe_model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: Callable, 
    device: torch.device, 
    probe_type: str,
) -> Dict[str, float]:
    probe_model.eval() 
    total_loss = 0.0
    num_batches = 0

    all_predictions_np: List[np.ndarray] = []       # Will store per-sentence, unpadded arrays
    all_gold_labels_np: List[np.ndarray] = []       # Will store per-sentence, unpadded, NON-SQUARED gold arrays
    all_lengths_list: List[int] = []                # Stores original length for each sentence
    all_gold_head_indices_list: List[List[int]] = [] 
    all_upos_tags_list: List[List[str]] = [] 

    with torch.no_grad():
        for batch in dataloader:
            embeddings_b = batch["embeddings_batch"].to(device)
            # labels_b are gold NON-SQUARED labels from collate_fn, padded with -1
            labels_b_for_loss = batch["labels_batch"].to(device) 
            lengths_b = batch["lengths_batch"] # Keep on CPU for slicing, pass to loss_fn which might move it
            
            # For metrics, we also need original (unpadded) gold data and other per-sentence info
            current_batch_gold_heads = batch["head_indices_batch"] 
            current_batch_upos = batch["upos_tags_batch"]       
            # The 'gold_labels' from __getitem__ are non-squared and unpadded per sentence.
            # We need to collect these *before* they are padded by collate_fn if we want them directly.
            # Alternative: collate_fn also returns the list of original unpadded gold_label tensors.
            # For now, we reconstruct unpadded gold from labels_b_for_loss for metrics.

            predictions_b = probe_model(embeddings_b) # Probe outputs SQUARED values
            
            loss = loss_fn(predictions_b, labels_b_for_loss, lengths_b.to(device)) 
            total_loss += loss.item()
            num_batches += 1

            for i in range(predictions_b.shape[0]): 
                length = lengths_b[i].item()
                all_lengths_list.append(length)
                all_gold_head_indices_list.append(current_batch_gold_heads[i]) # Already a list of ints for the sentence
                all_upos_tags_list.append(current_batch_upos[i])          # Already a list of strings

                # Predictions are SQUARED
                # Gold labels for metrics should be NON-SQUARED and unpadded
                if probe_type == "distance":
                    all_predictions_np.append(predictions_b[i, :length, :length].cpu().numpy())
                    all_gold_labels_np.append(labels_b_for_loss[i, :length, :length].cpu().numpy()) 
                elif probe_type == "depth":
                    all_predictions_np.append(predictions_b[i, :length].cpu().numpy())
                    all_gold_labels_np.append(labels_b_for_loss[i, :length].cpu().numpy()) 
    
    metrics = {"loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    if all_predictions_np:
        spearman = calculate_spearmanr(all_predictions_np, all_gold_labels_np, all_lengths_list, probe_type)
        metrics["spearmanr"] = spearman

        if probe_type == "distance":
            uuas = calculate_uuas(all_predictions_np, all_gold_head_indices_list, all_lengths_list, all_upos_tags_list)
            metrics["uuas"] = uuas
        elif probe_type == "depth":
            root_acc = calculate_root_accuracy(all_predictions_np, all_gold_head_indices_list, all_lengths_list, all_upos_tags_list)
            metrics["root_acc"] = root_acc
    else: 
        metrics["spearmanr"] = 0.0
        if probe_type == "distance": metrics["uuas"] = 0.0
        if probe_type == "depth": metrics["root_acc"] = 0.0
            
    return metrics

if __name__ == '__main__':
    # Example usage (requires dummy data and models)
    print("Evaluate.py standalone example (needs more setup for full test)")
    
    # Spearman Test (Depth)
    preds_s_depth = [np.array([0., 1., 2.]), np.array([0., 1.])]
    golds_s_depth = [np.array([0., 1., 2.]), np.array([5., 6.])]
    lengths_s_depth = [3, 2]
    spear_depth = calculate_spearmanr(preds_s_depth, golds_s_depth, lengths_s_depth, "depth")
    print(f"Spearman (depth-like, expected 1.0): {spear_depth}") 

    # Spearman Test (Distance)
    preds_s_dist = [np.array([[0,1,2],[1,0,3],[2,3,0]], dtype=np.float32)] 
    golds_s_dist = [np.array([[0,1.5,2.5],[1.5,0,3.5],[2.5,3.5,0]], dtype=np.float32)]
    lengths_s_dist = [3]
    spear_dist = calculate_spearmanr(preds_s_dist, golds_s_dist, lengths_s_dist, "distance")
    print(f"Spearman (distance-like, expected 1.0): {spear_dist}")


    # UUAS Test
    pred_dists_u = [np.array([[0, 0.5, 2.0], [0.5, 0, 0.4], [2.0, 0.4, 0]], dtype=np.float32)]
    gold_heads_u = [[-1, 0, 1]] 
    lengths_u = [3]
    upos_u: List[List[str]] = [["NOUN", "VERB", "NOUN"]]
    uuas = calculate_uuas(pred_dists_u, gold_heads_u, lengths_u, upos_u)
    print(f"UUAS (expected 1.0): {uuas}")

    # Root Accuracy Test
    pred_depths_r = [np.array([0.1, 0.5, 0.3])] 
    gold_heads_r = [[-1, 0, 0]] 
    lengths_r = [3]
    upos_r: List[List[str]] = [["NOUN", "VERB", "ADJ"]]
    root_acc = calculate_root_accuracy(pred_depths_r, gold_heads_r, lengths_r, upos_r)
    print(f"Root Accuracy (expected 1.0): {root_acc}") 
    
    pred_depths_r2 = [np.array([0.5, 0.1, 0.3])] 
    root_acc2 = calculate_root_accuracy(pred_depths_r2, gold_heads_r, lengths_r, upos_r)
    print(f"Root Accuracy (wrong root, expected 0.0): {root_acc2}")
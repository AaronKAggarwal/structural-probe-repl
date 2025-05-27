# src/torch_probe/evaluate.py
from typing import List, Dict, Any, Callable, Optional
import torch
import torch.nn as nn 
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from torch.utils.data import DataLoader
import logging # Added for logging

log = logging.getLogger(__name__)

# Punctuation XPOS tags used by Hewitt & Manning (2019)
# These are PTB-style POS tags.
H_M_PUNCTUATION_XPOS_TAGS = {"''", ",", ".", ":", "``", "-LRB-", "-RRB-"}


def calculate_spearmanr_hm_style(
    all_predictions: List[np.ndarray], 
    all_gold_labels: List[np.ndarray], 
    all_original_lengths: List[int], # Original lengths INCLUDING punctuation
    all_xpos_tags_for_sentences: List[List[str]], # Per-sentence XPOS tags
    probe_type: str,
    min_len_for_spearman_avg: int = 5,
    max_len_for_spearman_avg: int = 50
) -> tuple[float, list[float], dict[int, list[float]], dict[int, float]]:
    """
    Calculates Spearman correlation according to Hewitt & Manning (2019) style.

    Args:
        all_predictions: List of per-sentence numpy arrays of predicted SQUARED distances/depths.
                         Shape for distance: (L, L), for depth: (L,). Unpadded.
        all_gold_labels: List of per-sentence numpy arrays of gold NON-SQUARED distances/depths.
                         Shape for distance: (L, L), for depth: (L,). Unpadded.
        all_original_lengths: List of original sentence lengths (token count including punctuation).
        probe_type: "distance" or "depth".
        min_len_for_spearman_avg: Minimum sentence length (inclusive) for H&M averaging.
        max_len_for_spearman_avg: Maximum sentence length (inclusive) for H&M averaging.

    Returns:
        tuple: (
            final_macro_averaged_spearman (float),
            all_individual_scores_in_range (list[float]): For distance, these are per-word rhos. For depth, per-sentence rhos.
            scores_by_length_group (dict[int, list[float]]): Individual scores grouped by sentence length.
            avg_score_per_length_group (dict[int, float]): Average score for each length group.
        )
    """
    if not all_predictions:
        return 0.0, [], {}, {}

    # --- Step 1: Calculate individual scores (per-word rhos for distance, per-sentence rhos for depth) ---
    # Store these scores grouped by their original sentence length.
    # For distance, H&M collected all per-word spearmanrs for a given length.
    # For depth, they collected per-sentence spearmanrs for a given length.
    
    scores_by_length_group: Dict[int, List[float]] = defaultdict(list)

    for i in range(len(all_predictions)):
        pred_data_sent = all_predictions[i] # This is already unpadded to its true length L
        gold_data_sent = all_gold_labels[i] # This is also unpadded
        original_length = all_original_lengths[i]

        if probe_type == "depth":
            if original_length < 2:  # Spearman requires at least 2 data points
                continue 
            
            # For depth, prediction and gold are vectors of shape (original_length,)
            if np.std(pred_data_sent) == 0 and np.std(gold_data_sent) == 0: rho_sent = 1.0
            elif np.std(pred_data_sent) == 0 or np.std(gold_data_sent) == 0: rho_sent = 0.0
            else: rho_sent, _ = spearmanr(pred_data_sent, gold_data_sent)
            
            if not np.isnan(rho_sent):
                scores_by_length_group[original_length].append(rho_sent)
        
        elif probe_type == "distance":
            if original_length < 2: # Need at least 2 words for a distance matrix
                continue
            
            # For distance, prediction and gold are matrices of shape (original_length, original_length)
            # H&M: "For each word... computes the Spearman... between all true distances 
            #       between that word and all other words, and all predicted distances..."
            for word_idx in range(original_length):
                # Get the row for word_idx, excluding the self-distance (diagonal element)
                # The original H&M code `zip(prediction, label)` iterates through rows.
                pred_row = np.delete(pred_data_sent[word_idx, :], word_idx)
                gold_row = np.delete(gold_data_sent[word_idx, :], word_idx)

                if len(pred_row) < 2: # Each word needs at least 2 *other* words to compare distances to.
                                      # So, original_length must be >= 3 for any per-word rho.
                    continue
                
                if np.std(pred_row) == 0 and np.std(gold_row) == 0: rho_word = 1.0
                elif np.std(pred_row) == 0 or np.std(gold_row) == 0: rho_word = 0.0
                else: rho_word, _ = spearmanr(pred_row, gold_row)

                if not np.isnan(rho_word):
                    scores_by_length_group[original_length].append(rho_word)
        else:
            raise ValueError(f"Unknown probe_type for Spearman: {probe_type}")

    if not scores_by_length_group:
        return 0.0, [], {}, {}

    # --- Step 2: Calculate average score for each length group within the 5-50 range ---
    avg_score_per_length_group_in_range: Dict[int, float] = {}
    all_individual_scores_in_range: List[float] = []

    for length_val, scores_for_len in scores_by_length_group.items():
        if min_len_for_spearman_avg <= length_val <= max_len_for_spearman_avg:
            if scores_for_len: # Ensure list is not empty
                avg_score_per_length_group_in_range[length_val] = np.mean(scores_for_len)
                all_individual_scores_in_range.extend(scores_for_len)
    
    if not avg_score_per_length_group_in_range:
        # No sentences fell into the 5-50 length range after processing
        return 0.0, all_individual_scores_in_range, scores_by_length_group, {}


    # --- Step 3: Macro-average these per-length-group averages ---
    final_macro_averaged_spearman = np.mean(list(avg_score_per_length_group_in_range.values())).item() \
                                    if avg_score_per_length_group_in_range else 0.0
    
    return final_macro_averaged_spearman, all_individual_scores_in_range, scores_by_length_group, avg_score_per_length_group_in_range


def calculate_uuas(
    all_predicted_distances: List[np.ndarray], # Per-sentence, unpadded (L,L) SQUARED distances
    all_gold_head_indices: List[List[int]],    # Per-sentence, 0-indexed, root=-1
    all_lengths: List[int],                    # Original lengths (including punctuation)
    all_xpos_tags_for_sentences: List[List[str]]  # Per-sentence XPOS tags
) -> tuple[float, list[float]]:
    """Calculates Undirected Unlabeled Attachment Score, ignoring punctuation based on XPOS tags."""
    if not all_predicted_distances: return 0.0, []
    # Add input validation checks for list lengths

    all_sentence_uuas: list[float] = []
    total_correct_edges = 0
    total_gold_edges_in_evaluable_sents = 0

    for i in range(len(all_predicted_distances)):
        pred_dist_matrix_full = all_predicted_distances[i] # SQUARED distances
        gold_heads_sent = all_gold_head_indices[i]
        original_length = all_lengths[i]
        xpos_tags_sent = all_xpos_tags_for_sentences[i]

        if original_length < 2: continue

        # Identify non-punctuation tokens using their original indices
        non_punct_original_indices = [
            idx for idx, tag in enumerate(xpos_tags_sent) 
            if tag not in H_M_PUNCTUATION_XPOS_TAGS
        ]
        num_non_punct_tokens = len(non_punct_original_indices)

        if num_non_punct_tokens < 2: # Need at least two non-punctuation words to form an edge
            continue
        
        # Create a mapping from original index to new non-punctuation index
        non_punct_idx_map: Dict[int, int] = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(non_punct_original_indices)
        }

        # Build gold edges using new non-punctuation indices
        gold_edges_non_punct = set()
        for orig_token_idx in non_punct_original_indices: 
            orig_head_idx = gold_heads_sent[orig_token_idx]
            # Check if head is also a non-punctuation token and not root
            if orig_head_idx != -1 and orig_head_idx in non_punct_idx_map: 
                new_token_idx = non_punct_idx_map[orig_token_idx]
                new_head_idx = non_punct_idx_map[orig_head_idx]
                gold_edges_non_punct.add(tuple(sorted((new_token_idx, new_head_idx))))
        
        if not gold_edges_non_punct: # Sentence might consist of only a root or disconnected non-punct words
            continue # This sentence contributes no edges to evaluate against for UUAS

        # Filter predicted distance matrix to only include non-punctuation tokens
        pred_dist_matrix_non_punct = pred_dist_matrix_full[np.ix_(non_punct_original_indices, non_punct_original_indices)]
        
        # MST from predicted distances (distances are costs, lower is better)
        predicted_edges_non_punct = set()
        if num_non_punct_tokens >= 2: # MST needs at least 2 nodes
            try:
                # Ensure matrix is symmetric for MST if it might not be (though distances should be)
                # pred_dist_matrix_non_punct = (pred_dist_matrix_non_punct + pred_dist_matrix_non_punct.T) / 2.0
                mst_sparse_matrix = minimum_spanning_tree(pred_dist_matrix_non_punct)
                rows, cols = mst_sparse_matrix.nonzero()
                for r, c in zip(rows, cols):
                    predicted_edges_non_punct.add(tuple(sorted((r, c))))
            except ValueError as ve: # Catches issues like all-nan matrix if predictions are bad
                log.warning(f"MST calculation failed for sentence index {i} (non-punct len {num_non_punct_tokens}): {ve}. Pred matrix sample: {pred_dist_matrix_non_punct[:2,:2]}")
                # Sentence will have 0 correct edges for this case
            except Exception as e:
                log.error(f"Unexpected error in MST calculation for sentence {i}: {e}", exc_info=True)


        current_correct_edges = len(gold_edges_non_punct.intersection(predicted_edges_non_punct))
        sentence_uuas = current_correct_edges / len(gold_edges_non_punct) if len(gold_edges_non_punct) > 0 else 0.0
        
        all_sentence_uuas.append(sentence_uuas)
        total_correct_edges += current_correct_edges
        total_gold_edges_in_evaluable_sents += len(gold_edges_non_punct)

    mean_uuas = total_correct_edges / total_gold_edges_in_evaluable_sents if total_gold_edges_in_evaluable_sents > 0 else 0.0
    return mean_uuas, all_sentence_uuas


def calculate_root_accuracy(
    all_predicted_depths: List[np.ndarray],    # Per-sentence, unpadded (L,) SQUARED depths
    all_gold_head_indices: List[List[int]],    # Per-sentence, 0-indexed, root=-1
    all_lengths: List[int],                    # Original lengths (including punctuation)
    all_xpos_tags_for_sentences: List[List[str]]  # Per-sentence XPOS tags
) -> tuple[float, list[float]]:
    """Calculates Root Accuracy, ignoring punctuation based on XPOS tags."""
    if not all_predicted_depths: return 0.0, []
    # Add input validation checks for list lengths

    sentence_level_root_outcomes: list[float] = [] # 1.0 for correct, 0.0 for incorrect
    correct_roots = 0
    num_evaluable_sentences = 0 # Sentences with at least one non-punctuation gold root

    for i in range(len(all_predicted_depths)):
        pred_depths_full = all_predicted_depths[i] # SQUARED depths
        gold_heads_sent = all_gold_head_indices[i]
        original_length = all_lengths[i]
        xpos_tags_sent = all_xpos_tags_for_sentences[i]

        if original_length == 0: continue

        # Identify non-punctuation tokens and their original indices
        non_punct_original_indices = [
            idx for idx, tag in enumerate(xpos_tags_sent)
            if tag not in H_M_PUNCTUATION_XPOS_TAGS
        ]

        if not non_punct_original_indices: # No non-punctuation tokens in sentence
            continue 

        # Find predicted root among non-punctuation tokens
        pred_depths_non_punct = pred_depths_full[non_punct_original_indices]
        if not pred_depths_non_punct.size: # Should be caught by above, but defensive
            continue
            
        # Get the index relative to the non_punct_original_indices list
        min_depth_idx_in_non_punct_list = np.argmin(pred_depths_non_punct).item()
        # Map back to original sentence index
        predicted_root_original_idx = non_punct_original_indices[min_depth_idx_in_non_punct_list]
        
        # Find actual gold root(s) among non-punctuation tokens
        actual_gold_root_indices = [
            idx for idx in non_punct_original_indices if gold_heads_sent[idx] == -1
        ]
        
        if not actual_gold_root_indices: # No non-punctuation gold root found
            # This sentence is not considered evaluable for root accuracy by H&M logic
            # (their code implies a single root by label.index(0))
            continue 
        
        num_evaluable_sentences += 1
        # H&M implicitly assume one root; take the first if multiple (rare for non-projective SD)
        actual_root_original_idx = actual_gold_root_indices[0] 
        
        if predicted_root_original_idx == actual_root_original_idx:
            correct_roots += 1
            sentence_level_root_outcomes.append(1.0)
        else:
            sentence_level_root_outcomes.append(0.0)

    mean_accuracy = correct_roots / num_evaluable_sentences if num_evaluable_sentences > 0 else 0.0
    return mean_accuracy, sentence_level_root_outcomes


def evaluate_probe(
    probe_model: nn.Module, 
    dataloader: DataLoader, # DataLoader now yields 'xpos_tags_batch'
    loss_fn: Callable, 
    device: torch.device, 
    probe_type: str,
    # Configuration for Spearman H&M style, passed from main script's cfg
    spearman_min_len: int = 5, 
    spearman_max_len: int = 50
) -> Dict[str, Any]:
    probe_model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions_np: List[np.ndarray] = []       
    all_gold_labels_np: List[np.ndarray] = []       
    all_lengths_list: List[int] = []                
    all_gold_head_indices_list: List[List[int]] = [] 
    all_xpos_tags_list: List[List[str]] = [] # Changed from UPOS to XPOS

    with torch.no_grad():
        for batch in dataloader:
            embeddings_b = batch["embeddings_batch"].to(device)
            labels_b_for_loss = batch["labels_batch"].to(device) 
            lengths_b = batch["lengths_batch"] 
            
            current_batch_gold_heads = batch["head_indices_batch"] 
            current_batch_xpos = batch["xpos_tags_batch"] # <<< Now expecting xpos_tags_batch
            # current_batch_upos = batch["upos_tags_batch"] # Keep if needed elsewhere, but H&M metrics use XPOS

            predictions_b = probe_model(embeddings_b) # Probe outputs SQUARED values
            
            loss = loss_fn(predictions_b, labels_b_for_loss, lengths_b.to(device)) 
            total_loss += loss.item()
            num_batches += 1

            for i in range(predictions_b.shape[0]): # Iterate over sentences in batch
                length = lengths_b[i].item()
                all_lengths_list.append(length)
                all_gold_head_indices_list.append(current_batch_gold_heads[i])
                all_xpos_tags_list.append(current_batch_xpos[i]) # Store XPOS

                # Predictions are SQUARED from the probe model
                # Gold labels for metrics should be NON-SQUARED and unpadded
                if probe_type == "distance":
                    all_predictions_np.append(predictions_b[i, :length, :length].cpu().numpy())
                    all_gold_labels_np.append(labels_b_for_loss[i, :length, :length].cpu().numpy()) 
                elif probe_type == "depth":
                    all_predictions_np.append(predictions_b[i, :length].cpu().numpy())
                    all_gold_labels_np.append(labels_b_for_loss[i, :length].cpu().numpy()) 
    
    metrics: Dict[str, Any] = {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    if all_predictions_np: # Ensure there is data to evaluate
        # Use the H&M style Spearman calculation
        mean_spearman_hm, all_sent_rhos_in_range, _, _ = calculate_spearmanr_hm_style(
            all_predictions=all_predictions_np, 
            all_gold_labels=all_gold_labels_np, 
            all_original_lengths=all_lengths_list, 
            all_xpos_tags_for_sentences=all_xpos_tags_list, 
            probe_type=probe_type,  # Explicitly by keyword now
            min_len_for_spearman_avg=spearman_min_len, 
            max_len_for_spearman_avg=spearman_max_len 
        )
        metrics["spearmanr_hm"] = mean_spearman_hm # Hewitt-Manning style DSpr/NSpr
        # The `all_sent_rhos_in_range` are individual scores (per-word or per-sent) that contributed to the final avg
        # This can be logged if detailed analysis is needed, e.g. for distribution.
        metrics["spearmanr_hm_individual_scores_in_range"] = all_sent_rhos_in_range


        if probe_type == "distance":
            # UUAS calculation requires XPOS tags for punctuation filtering
            mean_uuas, per_sentence_uuas = calculate_uuas(
                all_predictions_np, all_gold_head_indices_list, all_lengths_list, all_xpos_tags_list
            )
            metrics["uuas"] = mean_uuas
            metrics["uuas_per_sentence"] = per_sentence_uuas
        elif probe_type == "depth":
            # Root Accuracy calculation requires XPOS tags for punctuation filtering
            mean_root_acc, per_sentence_root_acc = calculate_root_accuracy(
                all_predictions_np, all_gold_head_indices_list, all_lengths_list, all_xpos_tags_list
            )
            metrics["root_acc"] = mean_root_acc
            metrics["root_acc_per_sentence"] = per_sentence_root_acc
    else: # Default values if no data was evaluable
        metrics["spearmanr_hm"] = 0.0
        metrics["spearmanr_hm_individual_scores_in_range"] = []
        if probe_type == "distance": metrics["uuas"] = 0.0; metrics["uuas_per_sentence"] = []
        if probe_type == "depth": metrics["root_acc"] = 0.0; metrics["root_acc_per_sentence"] = []
            
    return metrics

if __name__ == '__main__':
    print("Evaluate.py standalone example (needs more setup for full test)")
    
    # Test H&M Style Spearman (Depth)
    preds_s_depth = [np.array([0., 1., 2., 3., 4.]), np.array([0., 1., 2., 3., 4., 5.])] # len 5, len 6
    golds_s_depth = [np.array([0., 1., 2., 3., 4.]), np.array([0., 1., 2., 3., 4., 5.])]
    lengths_s_depth = [5, 6] # Original lengths
    # Dummy XPOS, assume no punctuation for this simple test for Spearman
    xpos_s_depth = [["TAG"] * 5, ["TAG"] * 6] 
    
    mean_spear_d, indiv_d, bylen_d, avg_by_len_d = calculate_spearmanr_hm_style(
        preds_s_depth, golds_s_depth, lengths_s_depth, xpos_s_depth, "depth", min_len_for_spearman_avg=5, max_len_for_spearman_avg=10
    )
    print(f"Spearman H&M (depth, expected ~1.0): {mean_spear_d}")
    print(f"  Individual rhos contributing: {indiv_d}")
    print(f"  Avg rho per length group: {avg_by_len_d}")

    # Test H&M Style Spearman (Distance)
    # Sentence 1 (len 3): 3 words, so 3 per-word rhos.
    preds_s_dist1 = np.array([[0,1,2],[1,0,3],[2,3,0]], dtype=np.float32)
    golds_s_dist1 = np.array([[0,1,2],[1,0,3],[2,3,0]], dtype=np.float32)
    # Sentence 2 (len 5): 5 words, so 5 per-word rhos.
    preds_s_dist2 = np.random.rand(5,5).astype(np.float32); preds_s_dist2 = (preds_s_dist2 + preds_s_dist2.T)/2; np.fill_diagonal(preds_s_dist2, 0)
    golds_s_dist2 = preds_s_dist2 * 0.8 + np.random.rand(5,5)*0.2; golds_s_dist2 = (golds_s_dist2 + golds_s_dist2.T)/2; np.fill_diagonal(golds_s_dist2, 0)

    all_preds_dist = [preds_s_dist1, preds_s_dist2]
    all_golds_dist = [golds_s_dist1, golds_s_dist2]
    all_lengths_dist = [3, 5] # Original lengths
    xpos_s_dist = [["TAG"]*3, ["TAG"]*5]

    mean_spear_dist, indiv_dist, bylen_dist, avg_by_len_dist = calculate_spearmanr_hm_style(
        all_preds_dist, all_golds_dist, all_lengths_dist, xpos_s_dist, "distance", 
        min_len_for_spearman_avg=3, max_len_for_spearman_avg=5 # Adjusted range for example
    )
    print(f"Spearman H&M (distance, Sent1 expected ~1.0, Sent2 related): {mean_spear_dist}")
    print(f"  Individual per-word rhos contributing: {indiv_dist}")
    print(f"  Avg rho per length group: {avg_by_len_dist}")


    # UUAS Test with XPOS
    pred_dists_u = [np.array([[0, 0.5, 2.0, 1.8], [0.5, 0, 0.4, 1.2], [2.0, 0.4, 0, 0.9], [1.8, 1.2, 0.9, 0]], dtype=np.float32)]
    gold_heads_u = [[-1, 0, 1, 1]] # Word 3 is punct, attached to word 1 (token 'is')
    lengths_u = [4]
    xpos_u: List[List[str]] = [["NN", "VBZ", "NN", "."]] # "." is H&M punctuation
    mean_uuas, _ = calculate_uuas(pred_dists_u, gold_heads_u, lengths_u, xpos_u)
    # Expected: MST on words 0,1,2. Gold edges (0,1), (1,2). If MST is (0,1), (1,2) -> UUAS 1.0
    # Gold edges non-punct: (0,1) (NN-VBZ), (1,2) (VBZ-NN)
    # (non_punct_indices [0,1,2] -> map {0:0, 1:1, 2:2})
    # (0,1) from (idx0 head idx1) -> gold_heads_sent[0]=-1 (root)
    # (orig_idx 1, head 0) -> (new_idx 1, new_head 0) -> (0,1)
    # (orig_idx 2, head 1) -> (new_idx 2, new_head 1) -> (1,2)
    # Gold: {(0,1), (1,2)}
    # Pred: mst on [[0, 0.5, 2.0], [0.5, 0, 0.4], [2.0, 0.4, 0]] -> edges are (0,1) (cost 0.5), (1,2) (cost 0.4)
    # UUAS should be 1.0
    print(f"UUAS (mean expected 1.0 with XPOS filter): {mean_uuas}")

    # Root Accuracy Test with XPOS
    pred_depths_r = [np.array([0.1, 0.5, 0.3, 0.05])] # Punct (idx 3) has lowest predicted depth
    gold_heads_r = [[-1, 0, 0, 0]] # Root is idx 0
    lengths_r = [4]
    xpos_r: List[List[str]] = [["NN", "VBZ", "ADJ", "."]] # "." is H&M punctuation
    mean_root_acc, _ = calculate_root_accuracy(pred_depths_r, gold_heads_r, lengths_r, xpos_r)
    # Predicted root after punct filter should be token 0 (depth 0.1). Gold root is token 0. Acc = 1.0
    print(f"Root Accuracy (mean expected 1.0 with XPOS filter): {mean_root_acc}")
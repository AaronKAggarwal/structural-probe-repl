# src/torch_probe/evaluate.py
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# Punctuation tag sets
H_M_PUNCTUATION_XPOS_TAGS = {"''", ",", ".", ":", "``", "-LRB-", "-RRB-"}
UPOS_PUNCTUATION_TAGS = {"PUNCT", "SYM"}


def calculate_spearmanr_hm_style(
    all_predictions: List[np.ndarray],
    all_gold_labels: List[np.ndarray],
    all_original_lengths: List[int],
    all_xpos_tags_for_sentences: List[List[str]],
    probe_type: str,
    filter_by_non_punct_len: bool = True,
    min_len_for_spearman_avg: int = 5,
    max_len_for_spearman_avg: int = 50,
) -> tuple[float, list[float], dict[int, list[float]], dict[int, float]]:
    """
    Calculates Spearman correlation according to Hewitt & Manning (2019).

    Now includes a flag to switch between filtering by non-punctuation length (recommended)
    and original length (to exactly match the original authors' code).

    The process involves:
    1. For each sentence:
        a. For 'distance' probes: Calculate Spearman correlation for each word's predicted
           distance vector vs. its gold distance vector (to all other words).
        b. For 'depth' probes: Calculate Spearman correlation for the sentence's
           predicted depth sequence vs. its gold depth sequence.
    2. Group these individual scores (per-word for distance, per-sentence for depth)
       by the original (punctuation-inclusive) sentence length.
    3. For each sentence, determine its non-punctuation length using XPOS tags.
    4. Filter sentences based on the `filter_by_non_punct_len` flag:
       - If True: only consider those whose non-punctuation length is within the range.
       - If False: only consider those whose original length is within the range.
    5. For each original_length group that has sentences meeting the length criteria:
       Calculate the mean of all collected scores for that original_length.
    6. The final reported metric is the macro-average of these per-original-length-group means.

    Args:
        all_predictions: List of per-sentence numpy arrays of SQUARED predicted values.
        all_gold_labels: List of per-sentence numpy arrays of NON-SQUARED gold values.
        all_original_lengths: List of original sentence lengths (token count including punctuation).
        all_xpos_tags_for_sentences: List of lists, XPOS tags for each token in each sentence.
        probe_type: "distance" or "depth".
        filter_by_non_punct_len: If True, uses the count of non-punctuation tokens for the
                                 min/max length filter. If False, uses the original
                                 sentence length (including punctuation).
        min_len_for_spearman_avg: Min sentence length for H&M averaging.
        max_len_for_spearman_avg: Max sentence length for H&M averaging.

    Returns:
        tuple: (
            final_macro_averaged_spearman (float),
            all_individual_scores_in_hm_range (list[float]): Per-word (for dist) or per-sentence (for depth)
                                                              rhos from sentences that met the length criteria.
            all_scores_by_orig_length_group (dict[int, list[float]]): All individual scores, grouped by original sentence length.
            avg_score_per_orig_length_group_in_hm_range (dict[int, float]): Average score for each original_length group
                                                                           IF that group had sentences meeting length criteria.
        )
    """
    if not all_predictions:
        return 0.0, [], {}, {}

    raw_scores_and_non_punct_lengths_by_orig_length: Dict[
        int, List[Tuple[float, int]]
    ] = defaultdict(list)

    for i in range(len(all_predictions)):
        pred_data_sent = all_predictions[i]
        gold_data_sent = all_gold_labels[i]
        original_length = all_original_lengths[i]
        xpos_tags_sent = all_xpos_tags_for_sentences[i]

        non_punct_token_count = sum(
            1 for tag in xpos_tags_sent if tag not in H_M_PUNCTUATION_XPOS_TAGS
        )

        if probe_type == "depth":
            if original_length < 2:
                continue

            if np.std(pred_data_sent) < 1e-9 and np.std(gold_data_sent) < 1e-9:
                rho_sent = 1.0
            elif np.std(pred_data_sent) < 1e-9 or np.std(gold_data_sent) < 1e-9:
                rho_sent = 0.0
            else:
                rho_sent, _ = spearmanr(pred_data_sent, gold_data_sent)

            if not np.isnan(rho_sent):
                raw_scores_and_non_punct_lengths_by_orig_length[original_length].append(
                    (rho_sent, non_punct_token_count)
                )

        elif probe_type == "distance":
            if original_length < 2:
                continue

            for word_idx in range(original_length):
                pred_row = np.delete(pred_data_sent[word_idx, :], word_idx)
                gold_row = np.delete(gold_data_sent[word_idx, :], word_idx)

                if len(pred_row) < 2:
                    continue

                if np.std(pred_row) < 1e-9 and np.std(gold_row) < 1e-9:
                    rho_word = 1.0
                elif np.std(pred_row) < 1e-9 or np.std(gold_row) < 1e-9:
                    rho_word = 0.0
                else:
                    rho_word, _ = spearmanr(pred_row, gold_row)

                if not np.isnan(rho_word):
                    raw_scores_and_non_punct_lengths_by_orig_length[
                        original_length
                    ].append((rho_word, non_punct_token_count))
        else:
            raise ValueError(f"Unknown probe_type for Spearman: {probe_type}")

    if not raw_scores_and_non_punct_lengths_by_orig_length:
        return 0.0, [], {}, {}

    avg_score_per_orig_length_group_in_hm_range: Dict[int, float] = {}
    all_individual_scores_in_hm_range: List[float] = []

    for (
        orig_len,
        scores_with_non_punct_len,
    ) in raw_scores_and_non_punct_lengths_by_orig_length.items():
        if filter_by_non_punct_len:
            scores_to_average = [
                score
                for score, non_punct_len in scores_with_non_punct_len
                if min_len_for_spearman_avg <= non_punct_len <= max_len_for_spearman_avg
            ]
        else:
            if min_len_for_spearman_avg <= orig_len <= max_len_for_spearman_avg:
                scores_to_average = [score for score, _ in scores_with_non_punct_len]
            else:
                scores_to_average = []

        if scores_to_average:
            avg_score_per_orig_length_group_in_hm_range[orig_len] = np.mean(
                scores_to_average
            )
            all_individual_scores_in_hm_range.extend(scores_to_average)

    if not avg_score_per_orig_length_group_in_hm_range:
        return (
            0.0,
            all_individual_scores_in_hm_range,
            dict(raw_scores_and_non_punct_lengths_by_orig_length),
            {},
        )

    final_macro_averaged_spearman = np.mean(
        list(avg_score_per_orig_length_group_in_hm_range.values())
    ).item()

    all_scores_by_orig_length_group_debug = {
        k: [s[0] for s in v]
        for k, v in raw_scores_and_non_punct_lengths_by_orig_length.items()
    }

    return (
        final_macro_averaged_spearman,
        all_individual_scores_in_hm_range,
        all_scores_by_orig_length_group_debug,
        avg_score_per_orig_length_group_in_hm_range,
    )


def calculate_uuas(
    all_predicted_distances: List[np.ndarray],
    all_gold_head_indices: List[List[int]],
    all_lengths: List[int],
    all_xpos_tags_for_sentences: List[List[str]],
    all_upos_tags_for_sentences: List[List[str]],
    punctuation_strategy: str,
) -> tuple[float, list[float]]:
    """
    Calculates Undirected Unlabeled Attachment Score (UUAS).

    Punctuation is ignored based on the chosen strategy ('xpos' or 'upos').
    For each sentence, an MST is constructed from predicted distances (among non-punct tokens).
    Gold edges are derived from gold head indices (among non-punct tokens).
    UUAS is the ratio of correct predicted edges to total gold edges.
    The final UUAS is a micro-average over all edges in all evaluable sentences.

    Args:
        all_predicted_distances: List of (L,L) SQUARED predicted distance matrices.
        all_gold_head_indices: List of (L,) gold head indices (0-indexed, root=-1).
        all_lengths: List of original sentence lengths L (including punctuation).
        all_xpos_tags_for_sentences: List of (L,) XPOS tags for each sentence.
        all_upos_tags_for_sentences: List of (L,) UPOS tags for each sentence.
        punctuation_strategy: "xpos" or "upos", to select which tags to use for filtering.

    Returns:
        tuple: (mean_uuas_micro_avg, list_of_per_sentence_uuas)
    """
    if not all_predicted_distances:
        return 0.0, []

    per_sentence_uuas_scores: list[float] = []
    total_correct_edges_corpus = 0
    total_gold_edges_corpus = 0

    for i in range(len(all_predicted_distances)):
        pred_dist_matrix_full = all_predicted_distances[i]
        gold_heads_sent = all_gold_head_indices[i]

        if punctuation_strategy == "xpos":
            tags_to_check = all_xpos_tags_for_sentences[i]
            punct_set = H_M_PUNCTUATION_XPOS_TAGS
        elif punctuation_strategy == "upos":
            tags_to_check = all_upos_tags_for_sentences[i]
            punct_set = UPOS_PUNCTUATION_TAGS
        else:
            raise ValueError(f"Unknown punctuation_strategy: {punctuation_strategy}")

        non_punct_original_indices = [
            idx for idx, tag in enumerate(tags_to_check) if tag not in punct_set
        ]
        num_non_punct_tokens = len(non_punct_original_indices)

        if num_non_punct_tokens < 2:
            continue

        non_punct_idx_map: Dict[int, int] = {
            orig_idx: new_idx
            for new_idx, orig_idx in enumerate(non_punct_original_indices)
        }

        gold_edges_non_punct_in_new_indices = set()
        for orig_token_idx in non_punct_original_indices:
            orig_head_idx = gold_heads_sent[orig_token_idx]
            if orig_head_idx != -1 and orig_head_idx in non_punct_idx_map:
                new_token_idx = non_punct_idx_map[orig_token_idx]
                new_head_idx = non_punct_idx_map[orig_head_idx]
                gold_edges_non_punct_in_new_indices.add(
                    tuple(sorted((new_token_idx, new_head_idx)))
                )

        if not gold_edges_non_punct_in_new_indices:
            per_sentence_uuas_scores.append(0.0 if num_non_punct_tokens > 1 else 1.0)
            continue

        pred_dist_matrix_non_punct = pred_dist_matrix_full[
            np.ix_(non_punct_original_indices, non_punct_original_indices)
        ]

        predicted_edges_non_punct_in_new_indices = set()
        try:
            mst_sparse_matrix = minimum_spanning_tree(pred_dist_matrix_non_punct)
            rows, cols = mst_sparse_matrix.nonzero()
            for r_new, c_new in zip(rows, cols):
                predicted_edges_non_punct_in_new_indices.add(
                    tuple(sorted((r_new, c_new)))
                )
        except Exception as e:
            log.warning(
                f"MST calculation failed for sentence index {i}: {e}. UUAS will be 0 for this sentence."
            )

        current_correct_edges = len(
            gold_edges_non_punct_in_new_indices.intersection(
                predicted_edges_non_punct_in_new_indices
            )
        )
        num_gold_edges_this_sentence = len(gold_edges_non_punct_in_new_indices)

        sentence_uuas = (
            current_correct_edges / num_gold_edges_this_sentence
            if num_gold_edges_this_sentence > 0
            else 0.0
        )
        per_sentence_uuas_scores.append(sentence_uuas)

        total_correct_edges_corpus += current_correct_edges
        total_gold_edges_corpus += num_gold_edges_this_sentence

    mean_uuas_micro_avg = (
        total_correct_edges_corpus / total_gold_edges_corpus
        if total_gold_edges_corpus > 0
        else 0.0
    )
    return mean_uuas_micro_avg, per_sentence_uuas_scores


def calculate_root_accuracy(
    all_predicted_depths: List[np.ndarray],
    all_gold_head_indices: List[List[int]],
    all_lengths: List[int],
    all_xpos_tags_for_sentences: List[List[str]],
    all_upos_tags_for_sentences: List[List[str]],
    punctuation_strategy: str,
) -> tuple[float, list[float]]:
    """
    Calculates Root Accuracy.

    Punctuation is ignored based on the chosen strategy ('xpos' or 'upos').
    For each sentence, identifies the predicted root (shallowest non-punctuation token)
    and the gold root (non-punctuation token with head -1).
    Accuracy is the percentage of sentences where these match.

    Args:
        all_predicted_depths: List of (L,) SQUARED predicted depth arrays.
        all_gold_head_indices: List of (L,) gold head indices (0-indexed, root=-1).
        all_lengths: List of original sentence lengths L (including punctuation).
        all_xpos_tags_for_sentences: List of (L,) XPOS tags for each sentence.
        all_upos_tags_for_sentences: List of (L,) UPOS tags for each sentence.
        punctuation_strategy: "xpos" or "upos", to select which tags to use for filtering.

    Returns:
        tuple: (mean_root_accuracy, list_of_per_sentence_outcomes (1.0 or 0.0))
    """
    if not all_predicted_depths:
        return 0.0, []

    sentence_level_root_outcomes: list[float] = []
    correct_roots_count = 0
    num_evaluable_sentences_for_root_acc = 0

    for i in range(len(all_predicted_depths)):
        pred_depths_full = all_predicted_depths[i]
        gold_heads_sent = all_gold_head_indices[i]

        if punctuation_strategy == "xpos":
            tags_to_check = all_xpos_tags_for_sentences[i]
            punct_set = H_M_PUNCTUATION_XPOS_TAGS
        elif punctuation_strategy == "upos":
            tags_to_check = all_upos_tags_for_sentences[i]
            punct_set = UPOS_PUNCTUATION_TAGS
        else:
            raise ValueError(f"Unknown punctuation_strategy: {punctuation_strategy}")

        non_punct_original_indices = [
            idx for idx, tag in enumerate(tags_to_check) if tag not in punct_set
        ]

        if not non_punct_original_indices:
            continue

        pred_depths_non_punct = pred_depths_full[non_punct_original_indices]
        if not pred_depths_non_punct.size:
            continue

        min_depth_idx_in_non_punct_list = np.argmin(pred_depths_non_punct).item()
        predicted_root_original_idx = non_punct_original_indices[
            min_depth_idx_in_non_punct_list
        ]

        actual_gold_root_indices_non_punct = [
            idx for idx in non_punct_original_indices if gold_heads_sent[idx] == -1
        ]

        if not actual_gold_root_indices_non_punct:
            continue

        num_evaluable_sentences_for_root_acc += 1
        actual_root_original_idx = actual_gold_root_indices_non_punct[0]

        if predicted_root_original_idx == actual_root_original_idx:
            correct_roots_count += 1
            sentence_level_root_outcomes.append(1.0)
        else:
            sentence_level_root_outcomes.append(0.0)

    mean_accuracy = (
        correct_roots_count / num_evaluable_sentences_for_root_acc
        if num_evaluable_sentences_for_root_acc > 0
        else 0.0
    )
    return mean_accuracy, sentence_level_root_outcomes


def evaluate_probe(
    probe_model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    probe_type: str,
    filter_by_non_punct_len: bool,
    punctuation_strategy: str,
    spearman_min_len: int = 5,
    spearman_max_len: int = 50,
) -> Dict[str, Any]:
    """
    Evaluates the probe model on a given dataset.

    Args:
        probe_model: The probe model (DistanceProbe or DepthProbe).
        dataloader: DataLoader for the evaluation set.
        loss_fn: The loss function (e.g., distance_l1_loss).
        device: The torch device to use.
        probe_type: "distance" or "depth".
        filter_by_non_punct_len: Passed to Spearman calculation.
        punctuation_strategy: "xpos" or "upos" for UUAS/Root Acc punctuation handling.
        spearman_min_len: Min sentence length for H&M Spearman avg.
        spearman_max_len: Max sentence length for H&M Spearman avg.

    Returns:
        A dictionary containing aggregated metrics (loss, spearmanr_hm, uuas, root_acc)
        and lists of per-sentence scores for detailed analysis.
    """
    probe_model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions_np: List[np.ndarray] = []
    all_gold_labels_np: List[np.ndarray] = []
    all_lengths_list: List[int] = []
    all_gold_head_indices_list: List[List[int]] = []
    all_xpos_tags_list: List[List[str]] = []
    all_upos_tags_list: List[List[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings_b = batch["embeddings_batch"].to(device)
            labels_b_for_loss = batch["labels_batch"].to(device)
            lengths_b = batch["lengths_batch"]

            current_batch_gold_heads = batch["head_indices_batch"]
            current_batch_xpos = batch["xpos_tags_batch"]
            current_batch_upos = batch["upos_tags_batch"]

            predictions_b = probe_model(embeddings_b)

            loss = loss_fn(predictions_b, labels_b_for_loss, lengths_b.to(device))
            total_loss += loss.item()
            num_batches += 1

            for i in range(predictions_b.shape[0]):
                length = lengths_b[i].item()
                all_lengths_list.append(length)
                all_gold_head_indices_list.append(current_batch_gold_heads[i])
                all_xpos_tags_list.append(current_batch_xpos[i])
                all_upos_tags_list.append(current_batch_upos[i])

                if probe_type == "distance":
                    all_predictions_np.append(
                        predictions_b[i, :length, :length].cpu().numpy()
                    )
                    all_gold_labels_np.append(
                        labels_b_for_loss[i, :length, :length].cpu().numpy()
                    )
                elif probe_type == "depth":
                    all_predictions_np.append(predictions_b[i, :length].cpu().numpy())
                    all_gold_labels_np.append(
                        labels_b_for_loss[i, :length].cpu().numpy()
                    )

    metrics: Dict[str, Any] = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0
    }
    default_spear_metrics = {
        "spearmanr_hm": 0.0,
        "spearmanr_hm_individual_scores_in_range": [],
    }
    default_dist_metrics = {"uuas": 0.0, "uuas_per_sentence": []}
    default_depth_metrics = {"root_acc": 0.0, "root_acc_per_sentence": []}

    if all_predictions_np:
        mean_spear_hm, indiv_scores_range, _, _ = calculate_spearmanr_hm_style(
            all_predictions_np,
            all_gold_labels_np,
            all_lengths_list,
            all_xpos_tags_list,
            probe_type,
            filter_by_non_punct_len=filter_by_non_punct_len,
            min_len_for_spearman_avg=spearman_min_len,
            max_len_for_spearman_avg=spearman_max_len,
        )
        metrics["spearmanr_hm"] = mean_spear_hm
        metrics["spearmanr_hm_individual_scores_in_range"] = indiv_scores_range

        if probe_type == "distance":
            mean_uuas, per_sentence_uuas = calculate_uuas(
                all_predictions_np,
                all_gold_head_indices_list,
                all_lengths_list,
                all_xpos_tags_list,
                all_upos_tags_list,
                punctuation_strategy,
            )
            metrics["uuas"] = mean_uuas
            metrics["uuas_per_sentence"] = per_sentence_uuas
            metrics.update(default_depth_metrics)
        elif probe_type == "depth":
            mean_root_acc, per_sentence_root_acc = calculate_root_accuracy(
                all_predictions_np,
                all_gold_head_indices_list,
                all_lengths_list,
                all_xpos_tags_list,
                all_upos_tags_list,
                punctuation_strategy,
            )
            metrics["root_acc"] = mean_root_acc
            metrics["root_acc_per_sentence"] = per_sentence_root_acc
            metrics.update(default_dist_metrics)
    else:
        metrics.update(default_spear_metrics)
        metrics.update(default_dist_metrics)
        metrics.update(default_depth_metrics)

    return metrics


if __name__ == "__main__":
    # Updated example usage
    print("--- Evaluate.py standalone example ---")

    # --- Test H&M Style Spearman (Depth) ---
    print("\n-- Spearman H&M Style (Depth) Test --")
    preds_s_depth = [
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([2.0, 1.0, 0.0]),
    ]
    golds_s_depth = [
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([0.0, 1.0, 2.0]),
    ]
    original_lengths_s_depth = [5, 6, 3]
    xpos_s_depth = [["N"] * 5, ["N"] * 6, ["N"] * 3]

    mean_s_d_hm, _, _, _ = calculate_spearmanr_hm_style(
        preds_s_depth,
        golds_s_depth,
        original_lengths_s_depth,
        xpos_s_depth,
        "depth",
        min_len_for_spearman_avg=5,
        max_len_for_spearman_avg=50,
    )
    print(f"Spearman H&M (depth, range 5-50, expected 1.0): {mean_s_d_hm:.4f}")

    # --- Test H&M Style Spearman (Distance) ---
    print("\n-- Spearman H&M Style (Distance) Test --")
    preds_s_dist1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
    golds_s_dist1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
    np.random.seed(42)
    preds_s_dist2 = np.random.rand(5, 5).astype(np.float32)
    preds_s_dist2 = (preds_s_dist2 + preds_s_dist2.T) / 2
    np.fill_diagonal(preds_s_dist2, 0)
    golds_s_dist2 = preds_s_dist2 * 0.9 + np.random.rand(5, 5) * 0.1
    golds_s_dist2 = (golds_s_dist2 + golds_s_dist2.T) / 2
    np.fill_diagonal(golds_s_dist2, 0)
    all_preds_dist = [preds_s_dist1, preds_s_dist2]
    all_golds_dist = [golds_s_dist1, golds_s_dist2]
    all_original_lengths_dist = [3, 5]
    xpos_s_dist = [["N"] * 3, ["N"] * 5]
    mean_s_dist_hm, _, _, _ = calculate_spearmanr_hm_style(
        all_preds_dist,
        all_golds_dist,
        all_original_lengths_dist,
        xpos_s_dist,
        "distance",
        min_len_for_spearman_avg=3,
        max_len_for_spearman_avg=5,
    )
    print(f"Spearman H&M (distance, range 3-5): {mean_s_dist_hm:.4f} (expected >0.9)")

    # --- UUAS Test with XPOS/UPOS ---
    print("\n-- UUAS Test --")
    pred_dists_u = [
        np.array(
            [
                [0, 0.5, 2.0, 1.8],
                [0.5, 0, 0.4, 1.2],
                [2.0, 0.4, 0, 0.9],
                [1.8, 1.2, 0.9, 0],
            ],
            dtype=np.float32,
        )
    ]
    gold_heads_u = [[1, -1, 1, 1]]
    lengths_u = [4]
    xpos_u: List[List[str]] = [["NN", "VBZ", "NN", "."]]
    upos_u: List[List[str]] = [["NOUN", "VERB", "NOUN", "PUNCT"]]

    mean_uuas_xpos, _ = calculate_uuas(
        pred_dists_u, gold_heads_u, lengths_u, xpos_u, upos_u, "xpos"
    )
    print(f"UUAS (strategy='xpos', expected 1.0): {mean_uuas_xpos:.4f}")
    mean_uuas_upos, _ = calculate_uuas(
        pred_dists_u, gold_heads_u, lengths_u, xpos_u, upos_u, "upos"
    )
    print(f"UUAS (strategy='upos', expected 1.0): {mean_uuas_upos:.4f}")

    # --- Root Accuracy Test with XPOS/UPOS ---
    print("\n-- Root Accuracy Test --")
    pred_depths_r = [np.array([0.1, 0.5, 0.3, 0.05])]
    gold_heads_r = [[-1, 0, 0, 0]]
    lengths_r = [4]
    xpos_r: List[List[str]] = [["NN", "VBZ", "JJ", "."]]
    upos_r: List[List[str]] = [["NOUN", "VERB", "ADJ", "PUNCT"]]

    mean_root_acc_xpos, _ = calculate_root_accuracy(
        pred_depths_r, gold_heads_r, lengths_r, xpos_r, upos_r, "xpos"
    )
    print(f"Root Accuracy (strategy='xpos', expected 1.0): {mean_root_acc_xpos:.4f}")
    mean_root_acc_upos, _ = calculate_root_accuracy(
        pred_depths_r, gold_heads_r, lengths_r, xpos_r, upos_r, "upos"
    )
    print(f"Root Accuracy (strategy='upos', expected 1.0): {mean_root_acc_upos:.4f}")

    print("\n--- End of evaluate.py standalone example ---")

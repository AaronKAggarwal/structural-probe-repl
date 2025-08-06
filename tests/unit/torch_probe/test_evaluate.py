# tests/unit/torch_probe/test_evaluate.py
from typing import List  # Ensure List and Dict are imported

import numpy as np
from scipy.stats import spearmanr as scipy_spearmanr

# Import the new and updated functions from your evaluate module
from src.torch_probe.evaluate import (
    calculate_root_accuracy,  # Import H&M XPOS punct set
    calculate_spearmanr_hm_style,
    calculate_uuas,
)

# --- Tests for calculate_spearmanr_hm_style ---


def test_spearmanr_hm_depth_perfect_correlation():
    preds = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])]
    golds = [np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0])]
    original_lengths = [3, 2]
    # For H&M style, grouping is by original_lengths.
    # XPOS tags are needed for filtering sentences if non-punct length is used for 5-50 range.
    # If original_lengths are used for 5-50 range, xpos not strictly needed by this func, but good to pass.
    xpos_tags = [["N"] * 3, ["N"] * 2]  # Dummy non-punctuation XPOS

    # The function returns: final_macro_averaged_spearman, all_individual_scores, scores_by_length_group, avg_score_per_length_group
    mean_rho, _, _, _ = calculate_spearmanr_hm_style(
        preds,
        golds,
        original_lengths,
        xpos_tags,
        "depth",
        min_len_for_spearman_avg=2,
        max_len_for_spearman_avg=3,  # Adjust range for test
    )
    # Sentence 1 (len 3): rho=1.0. Sentence 2 (len 2): rho=1.0.
    # Group avg for len 3: 1.0. Group avg for len 2: 1.0.
    # Macro avg: (1.0+1.0)/2 = 1.0
    assert np.isclose(mean_rho, 1.0)


def test_spearmanr_hm_depth_perfect_anti_correlation():
    preds = [np.array([2.0, 1.0, 0.0])]
    golds = [np.array([0.0, 1.0, 2.0])]
    original_lengths = [3]
    xpos_tags = [["N"] * 3]
    mean_rho, _, _, _ = calculate_spearmanr_hm_style(
        preds,
        golds,
        original_lengths,
        xpos_tags,
        "depth",
        min_len_for_spearman_avg=3,
        max_len_for_spearman_avg=3,
    )
    assert np.isclose(mean_rho, -1.0)


def test_spearmanr_hm_depth_no_correlation():
    preds = [np.array([0.0, 1.0, 2.0])]
    golds = [np.array([1.0, 0.0, 2.0])]
    original_lengths = [3]
    xpos_tags = [["N"] * 3]
    expected_rho, _ = scipy_spearmanr([0.0, 1.0, 2.0], [1.0, 0.0, 2.0])
    mean_rho, _, _, _ = calculate_spearmanr_hm_style(
        preds,
        golds,
        original_lengths,
        xpos_tags,
        "depth",
        min_len_for_spearman_avg=3,
        max_len_for_spearman_avg=3,
    )
    assert np.isclose(mean_rho, expected_rho)


def test_spearmanr_hm_depth_length_filtering():
    # Sent 1 (len 2, non-punct len 2) -> rho = 1.0 (OUT of 3-3 range)
    # Sent 2 (len 3, non-punct len 3) -> rho = -1.0 (IN 3-3 range)
    # Sent 3 (len 4, non-punct len 4) -> rho = 0.5 (OUT of 3-3 range)
    preds = [
        np.array([0.0, 1.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    ]
    golds = [
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0, 1.0, 3.0]),
    ]
    original_lengths = [2, 3, 4]  # These are used for grouping by H&M original code
    # For filtering 5-50 range, we use non-punctuation length.
    # Let's assume all are non-punctuation for simplicity of this test.
    xpos_tags = [["N"] * 2, ["N"] * 3, ["N"] * 4]

    mean_rho, _, _, avg_by_len = calculate_spearmanr_hm_style(
        preds,
        golds,
        original_lengths,
        xpos_tags,
        "depth",
        min_len_for_spearman_avg=3,
        max_len_for_spearman_avg=3,  # Only length 3 sentences considered for final macro-avg
    )
    # Only sent2 (len 3, rho -1.0) contributes to the final macro average.
    assert np.isclose(mean_rho, -1.0)
    # Check that avg_by_len only contains data for length 3 (if non-punct length filtering is active)
    # Based on my calculate_spearmanr_hm_style, avg_by_len will contain averages for ALL lengths that had data,
    # but the final mean_rho only uses those within the 5-50 non-punct range.
    # The avg_by_len from my func contains avg per original_length group that made it past the 5-50 non-punct filter.
    assert 3 in avg_by_len
    assert np.isclose(avg_by_len[3], -1.0)
    assert 2 not in avg_by_len  # Because non-punct len 2 is outside 3-3 range
    assert 4 not in avg_by_len  # Because non-punct len 4 is outside 3-3 range


def test_spearmanr_hm_distance_simple():
    # Sentence 1: len 3. 3 words.
    # Word 0: pred_row [1,2], gold_row [1.5,2.5] -> rho=1.0
    # Word 1: pred_row [1,3], gold_row [1.5,3.5] -> rho=1.0
    # Word 2: pred_row [2,3], gold_row [2.5,3.5] -> rho=1.0
    # All per-word rhos are 1.0.
    preds_sent1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
    golds_sent1 = np.array(
        [[0, 1.5, 2.5], [1.5, 0, 3.5], [2.5, 3.5, 0]], dtype=np.float32
    )

    all_preds = [preds_sent1]
    all_golds = [golds_sent1]
    original_lengths = [3]
    xpos_tags = [["N"] * 3]

    mean_rho, indiv_scores, _, avg_by_len = calculate_spearmanr_hm_style(
        all_preds,
        all_golds,
        original_lengths,
        xpos_tags,
        "distance",
        min_len_for_spearman_avg=3,
        max_len_for_spearman_avg=3,
    )
    # Expected: length group 3 has avg of per-word rhos. All per-word rhos are 1.0. So avg is 1.0.
    # Final macro avg is 1.0.
    assert np.isclose(mean_rho, 1.0)
    assert len(indiv_scores) == 3  # 3 per-word rhos from the sentence
    assert all(np.isclose(s, 1.0) for s in indiv_scores)
    assert np.isclose(avg_by_len[3], 1.0)


def test_spearmanr_hm_empty_input():
    mean_rho, indiv, by_len, avg_by_len = calculate_spearmanr_hm_style(
        [], [], [], [], "depth"
    )
    assert mean_rho == 0.0 and not indiv and not by_len and not avg_by_len

    # Test with lengths too short for Spearman calculation or range
    # original length 1 (non-punct 1) -> skipped by probe_type == "depth" internal check (needs len >= 2)
    mean_rho_d1, _, _, _ = calculate_spearmanr_hm_style(
        [np.array([1])], [np.array([1])], [1], [["N"]], "depth", 2, 2
    )
    assert mean_rho_d1 == 0.0

    # original length 2 (non-punct 2) -> rho calc, but if range is 3-3, it's filtered out for final macro-avg
    mean_rho_d2, _, _, avg_grp_d2 = calculate_spearmanr_hm_style(
        [np.array([1, 2])],
        [np.array([1, 2])],
        [2],
        [["N", "N"]],
        "depth",
        min_len_for_spearman_avg=3,
        max_len_for_spearman_avg=3,
    )
    assert mean_rho_d2 == 0.0  # Because no length groups are IN the 3-3 range
    assert 2 not in avg_grp_d2  # avg_by_len_group should be empty or not contain key 2

    # original length 2 (non-punct 2) -> rho calc, range 2-2
    mean_rho_d3, _, _, avg_grp_d3 = calculate_spearmanr_hm_style(
        [np.array([1, 2])],
        [np.array([1, 2])],
        [2],
        [["N", "N"]],
        "depth",
        min_len_for_spearman_avg=2,
        max_len_for_spearman_avg=2,
    )
    assert np.isclose(mean_rho_d3, 1.0)
    assert np.isclose(avg_grp_d3[2], 1.0)


# --- Tests for calculate_uuas (Now require all_xpos_tags) ---
def test_uuas_perfect_match():
    pred_dists = [
        np.array([[0, 0.5, 2.0], [0.5, 0, 0.4], [2.0, 0.4, 0]], dtype=np.float32)
    ]
    gold_heads = [[-1, 0, 1]]
    lengths = [3]
    xpos_tags: List[List[str]] = [["NN", "VB", "NN"]]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "NOUN"]]  # Dummy UPOS tags
    mean_uuas, _ = calculate_uuas(
        pred_dists, gold_heads, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert np.isclose(mean_uuas, 1.0)


def test_uuas_no_match():
    pred_dists = [
        np.array([[0, 1.9, 2.0], [1.9, 0, 0.5], [2.0, 0.5, 0]], dtype=np.float32)
    ]
    gold_heads = [[-1, 0, 0]]
    lengths = [3]
    xpos_tags: List[List[str]] = [["NN", "VB", "JJ"]]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "ADJ"]]  # Dummy UPOS tags
    mean_uuas, _ = calculate_uuas(
        pred_dists, gold_heads, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert np.isclose(mean_uuas, 0.5)


def test_uuas_with_punctuation_ignored_xpos():
    pred_dists_full = [
        np.array(
            [[0, 0.5, 5, 5], [0.5, 0, 5, 5], [5, 5, 0, 0.1], [5, 5, 0.1, 0]],
            dtype=np.float32,
        )
    ]
    gold_heads_full = [[1, -1, 1, 1]]
    lengths = [4]
    xpos_tags: List[List[str]] = [["NN", "VB", ".", ","]]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "PUNCT", "PUNCT"]]  # Dummy UPOS
    mean_uuas, _ = calculate_uuas(
        pred_dists_full, gold_heads_full, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert np.isclose(mean_uuas, 1.0)


def test_uuas_all_punctuation_xpos():
    pred_dists_full = [np.array([[0, 0.1], [0.1, 0]], dtype=np.float32)]
    gold_heads_full = [[-1, 0]]
    lengths = [2]
    xpos_tags: List[List[str]] = [[".", ","]]
    upos_tags: List[List[str]] = [["PUNCT", "PUNCT"]]  # Dummy UPOS
    mean_uuas, _ = calculate_uuas(
        pred_dists_full, gold_heads_full, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert np.isclose(mean_uuas, 0.0)


def test_uuas_short_sentence_xpos():
    pred_dists = [np.array([[0]], dtype=np.float32)]
    gold_heads = [[-1]]
    lengths = [1]
    xpos_tags: List[List[str]] = [["NN"]]
    upos_tags: List[List[str]] = [["NOUN"]]  # Dummy UPOS
    mean_uuas, _ = calculate_uuas(
        pred_dists, gold_heads, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert mean_uuas == 0.0

    mean_uuas_empty, _ = calculate_uuas([], [], [], [], [], "xpos")
    assert mean_uuas_empty == 0.0


# --- Tests for calculate_root_accuracy (These also failed) ---


def test_root_accuracy_correct_xpos():
    pred_depths = [np.array([0.1, 0.5, 0.3])]
    gold_heads = [[-1, 0, 0]]
    lengths = [3]
    xpos_tags: List[List[str]] = [["NN", "VB", "JJ"]]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "ADJ"]]  # Dummy UPOS
    mean_acc, _ = calculate_root_accuracy(
        pred_depths, gold_heads, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert np.isclose(mean_acc, 1.0)


def test_root_accuracy_with_punctuation_ignored_xpos():
    pred_depths = [np.array([0.5, 0.8, 0.1, 0.9])]
    gold_heads = [[-1, 0, 0, 0]]
    lengths = [4]
    xpos_tags: List[List[str]] = [["NN", "VB", ".", "JJ"]]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "PUNCT", "ADJ"]]  # Dummy UPOS
    mean_acc, _ = calculate_root_accuracy(
        pred_depths, gold_heads, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert np.isclose(mean_acc, 1.0)


def test_root_accuracy_gold_root_is_punctuation_xpos():
    pred_depths = [np.array([0.8, 0.5, 0.1])]
    gold_heads = [[2, 2, -1]]
    lengths = [3]
    xpos_tags: List[List[str]] = [["NN", "VB", "."]]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "PUNCT"]]  # Dummy UPOS
    mean_acc, _ = calculate_root_accuracy(
        pred_depths, gold_heads, lengths, xpos_tags, upos_tags, "xpos"
    )
    assert np.isclose(mean_acc, 0.0)


def test_root_accuracy_empty_or_short_xpos():
    mean_acc_empty, _ = calculate_root_accuracy([], [], [], [], [], "xpos")
    assert mean_acc_empty == 0.0

    mean_acc_len0, _ = calculate_root_accuracy(
        [np.array([])], [[]], [0], [[]], [[]], "xpos"
    )
    assert mean_acc_len0 == 0.0

    mean_acc_len1_nonpunct, _ = calculate_root_accuracy(
        [np.array([0.1])], [[-1]], [1], [["NN"]], [["NOUN"]], "xpos"
    )
    assert np.isclose(mean_acc_len1_nonpunct, 1.0)

    mean_acc_len1_punct, _ = calculate_root_accuracy(
        [np.array([0.1])], [[-1]], [1], [[","]], [["PUNCT"]], "xpos"
    )
    assert np.isclose(mean_acc_len1_punct, 0.0)


# Optional: Add new tests for the "upos" strategy
def test_uuas_with_punctuation_ignored_upos():
    pred_dists_full = [
        np.array(
            [[0, 0.5, 5, 5], [0.5, 0, 5, 5], [5, 5, 0, 0.1], [5, 5, 0.1, 0]],
            dtype=np.float32,
        )
    ]
    gold_heads_full = [[1, -1, 1, 1]]
    lengths = [4]
    xpos_tags: List[List[str]] = [["NN", "VB", ".", "PUNCT_XPOS_EQUIVALENT"]]
    upos_tags: List[List[str]] = [
        ["NOUN", "VERB", "PUNCT", "SYM"]
    ]  # Use UPOS tags for filtering
    mean_uuas, _ = calculate_uuas(
        pred_dists_full, gold_heads_full, lengths, xpos_tags, upos_tags, "upos"
    )
    assert np.isclose(mean_uuas, 1.0)

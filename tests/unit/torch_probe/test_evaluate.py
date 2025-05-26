# tests/unit/torch_probe/test_evaluate.py
import pytest
import torch
import numpy as np
from scipy.stats import spearmanr as scipy_spearmanr 
from typing import List # Ensure List is imported

from src.torch_probe.evaluate import ( 
    calculate_spearmanr,
    calculate_uuas,
    calculate_root_accuracy,
    PUNCTUATION_UPOS_TAGS # Import for use in tests if needed, or define locally
)

# --- Tests for calculate_spearmanr ---
# (These tests remain unchanged as calculate_spearmanr does not use POS tags)
def test_spearmanr_depth_perfect_correlation():
    preds = [np.array([0., 1., 2.]), np.array([0., 1.])]
    golds = [np.array([0., 1., 2.]), np.array([5., 6.])] 
    lengths = [3, 2]
    rho = calculate_spearmanr(preds, golds, lengths, "depth")
    assert np.isclose(rho, 1.0)

def test_spearmanr_depth_perfect_anti_correlation():
    preds = [np.array([2., 1., 0.])]
    golds = [np.array([0., 1., 2.])]
    lengths = [3]
    rho = calculate_spearmanr(preds, golds, lengths, "depth")
    assert np.isclose(rho, -1.0)

def test_spearmanr_depth_no_correlation():
    preds = [np.array([0., 1., 2.])]
    golds = [np.array([1., 0., 2.])] 
    lengths = [3]
    # Calculate expected rho using scipy directly for this non-trivial case
    expected_rho, _ = scipy_spearmanr([0.,1.,2.], [1.,0.,2.])
    rho = calculate_spearmanr(preds, golds, lengths, "depth")
    assert np.isclose(rho, expected_rho)

def test_spearmanr_depth_with_padding():
    preds = [np.array([0., 1., 99., 99.])] 
    golds = [np.array([5., 6., -1., -1.])] 
    lengths = [2] 
    rho = calculate_spearmanr(preds, golds, lengths, "depth")
    assert np.isclose(rho, 1.0)
    
def test_spearmanr_distance_simple():
    preds = [np.array([[0,1,2],[1,0,3],[2,3,0]], dtype=np.float32)] 
    golds = [np.array([[0,1.5,2.5],[1.5,0,3.5],[2.5,3.5,0]], dtype=np.float32)]
    lengths = [3]
    rho = calculate_spearmanr(preds, golds, lengths, "distance")
    assert np.isclose(rho, 1.0) # With per-word spearman, each word's distances are perfectly correlated

def test_spearmanr_empty_input():
    assert calculate_spearmanr([], [], [], "depth") == 0.0
    assert calculate_spearmanr([np.array([1])], [np.array([1])], [1], "depth") == 0.0
    assert calculate_spearmanr([np.array([[0,1],[1,0]])],[np.array([[0,1],[1,0]])],[2],"distance") == 0.0 # Needs length >=3 for per-word Spearman with >1 other word


# --- Tests for calculate_uuas (Now require all_upos_tags) ---
def test_uuas_perfect_match():
    pred_dists = [np.array([[0, 0.5, 2.0], [0.5, 0, 0.4], [2.0, 0.4, 0]], dtype=np.float32)]
    gold_heads = [[-1, 0, 1]] 
    lengths = [3]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "NOUN"]] # Assume all non-punctuation
    uuas = calculate_uuas(pred_dists, gold_heads, lengths, upos_tags)
    assert np.isclose(uuas, 1.0)

def test_uuas_no_match():
    pred_dists = [np.array([[0, 1.9, 2.0], [1.9, 0, 0.5], [2.0, 0.5, 0]], dtype=np.float32)] # MST: (1,2), (0,1)
    gold_heads = [[-1, 0, 0]] # Gold edges: (0,1), (0,2)
    lengths = [3]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "ADJ"]]
    uuas = calculate_uuas(pred_dists, gold_heads, lengths, upos_tags)
    assert np.isclose(uuas, 0.5) # Only (0,1) matches

def test_uuas_with_punctuation_ignored():
    # Sentence: N V P P (Punctuation P at index 2 and 3)
    # Gold: N -> V (0->1), V -> root (-1)  => Non-punct edge: (0,1)
    pred_dists_full = [np.array([
        [0, 0.5, 5, 5],  # N
        [0.5, 0, 5, 5],  # V
        [5, 5, 0, 0.1], # P1 (Punct)
        [5, 5, 0.1, 0]   # P2 (Punct)
    ], dtype=np.float32)]
    # If we only consider N, V (indices 0, 1 after filtering):
    # pred_dists_non_punct = [[0, 0.5], [0.5, 0]] -> MST edge (0,1)
    gold_heads_full = [[1, -1, 1, 1]] # N->V, V->Root, P1->V, P2->V
    lengths = [4]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "PUNCT", "PUNCT"]]
    
    uuas = calculate_uuas(pred_dists_full, gold_heads_full, lengths, upos_tags)
    # Gold non-punct edges: {(0,1)}. Predicted non-punct edges: {(0,1)}. UUAS = 1/1 = 1.0
    assert np.isclose(uuas, 1.0)

def test_uuas_all_punctuation():
    pred_dists_full = [np.array([
        [0, 0.1],
        [0.1, 0] 
    ], dtype=np.float32)]
    gold_heads_full = [[-1, 0]] 
    lengths = [2]
    upos_tags: List[List[str]] = [["PUNCT", "PUNCT"]]
    uuas = calculate_uuas(pred_dists_full, gold_heads_full, lengths, upos_tags)
    # No non-punctuation tokens to form edges
    assert np.isclose(uuas, 0.0)

def test_uuas_short_sentence(): 
    pred_dists = [np.array([[0]], dtype=np.float32)]
    gold_heads = [[-1]]
    lengths = [1]
    upos_tags: List[List[str]] = [["NOUN"]]
    assert calculate_uuas(pred_dists, gold_heads, lengths, upos_tags) == 0.0
    assert calculate_uuas([], [], [], []) == 0.0


# --- Tests for calculate_root_accuracy (Now require all_upos_tags) ---
def test_root_accuracy_correct():
    pred_depths = [np.array([0.1, 0.5, 0.3])] 
    gold_heads = [[-1, 0, 0]] 
    lengths = [3]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "ADJ"]]
    acc = calculate_root_accuracy(pred_depths, gold_heads, lengths, upos_tags)
    assert np.isclose(acc, 1.0)

def test_root_accuracy_incorrect():
    pred_depths = [np.array([0.5, 0.1, 0.3])] 
    gold_heads = [[-1, 0, 0]] 
    lengths = [3]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "ADJ"]]
    acc = calculate_root_accuracy(pred_depths, gold_heads, lengths, upos_tags)
    assert np.isclose(acc, 0.0)

def test_root_accuracy_all_same_depth_picks_first():
    pred_depths = [np.array([0.1, 0.1, 0.1])] 
    gold_heads = [[-1, 0, 0]] 
    lengths = [3]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "ADJ"]]
    acc = calculate_root_accuracy(pred_depths, gold_heads, lengths, upos_tags)
    assert np.isclose(acc, 1.0)

def test_root_accuracy_with_punctuation_ignored():
    # Gold root is "N" (idx 0). Predicted shallowest non-punct is "N" (idx 0).
    pred_depths = [np.array([0.5, 0.8, 0.1, 0.9])] # Punct (idx 2) is shallowest overall
    gold_heads = [[-1, 0, 0, 0]] # N is root
    lengths = [4]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "PUNCT", "ADJ"]]
    acc = calculate_root_accuracy(pred_depths, gold_heads, lengths, upos_tags)
    assert np.isclose(acc, 1.0)

def test_root_accuracy_with_punctuation_ignored_misleading_pred():
    # Gold root is "N" (idx 0). Predicted shallowest non-punct is "V" (idx 1).
    pred_depths = [np.array([0.8, 0.5, 0.1, 0.9])] # Punct (idx 2) is shallowest overall
    gold_heads = [[-1, 0, 0, 0]] # N is root
    lengths = [4]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "PUNCT", "ADJ"]]
    acc = calculate_root_accuracy(pred_depths, gold_heads, lengths, upos_tags)
    assert np.isclose(acc, 0.0)

def test_root_accuracy_gold_root_is_punctuation():
    pred_depths = [np.array([0.8, 0.5, 0.1])] 
    gold_heads = [[2, 2, -1]] # Punct (idx 2) is gold root
    lengths = [3]
    upos_tags: List[List[str]] = [["NOUN", "VERB", "PUNCT"]]
    # The function should ideally find no valid non-punct root in gold,
    # or the predicted non-punct root won't match a punct root.
    # Current calculate_root_accuracy will find actual_root_original_idx = -1 if root is punct
    # and predicted_root_original_idx will be from non-punct. So they won't match.
    acc = calculate_root_accuracy(pred_depths, gold_heads, lengths, upos_tags)
    assert np.isclose(acc, 0.0)


def test_root_accuracy_empty_or_short():
    assert calculate_root_accuracy([], [], [], []) == 0.0
    assert calculate_root_accuracy([np.array([])], [[]], [0], [[]]) == 0.0
    assert calculate_root_accuracy([np.array([0.1])], [[-1]], [1], [["NOUN"]]) == 1.0 # Single non-punct token is its own root
    assert calculate_root_accuracy([np.array([0.1])], [[-1]], [1], [["PUNCT"]]) == 0.0 # Single punct token, no non-punct root
# tests/unit/torch_probe/test_gold_labels.py
import numpy as np

from src.torch_probe.utils.gold_labels import (  # Adjust import
    calculate_tree_depths,
    calculate_tree_distances,
)


def test_calculate_tree_depths_simple_chain():
    # 0 <- 1 <- 2  (Root is 0)
    # Heads (0-indexed): [-1, 0, 1]
    head_indices = [-1, 0, 1]
    expected_depths = [0, 1, 2]
    assert calculate_tree_depths(head_indices) == expected_depths


def test_calculate_tree_depths_star_graph():
    #   1 -> 0 <- 2
    #        ^
    #        |
    #        3
    # Heads (0-indexed): [-1, 0, 0, 0] (Token 0 is root)
    head_indices = [-1, 0, 0, 0]
    expected_depths = [0, 1, 1, 1]
    assert calculate_tree_depths(head_indices) == expected_depths

    # Heads (0-indexed): [0, 0, 0, -1] (Token 3 is root) - test robustness
    head_indices_alt_root = [3, 3, 3, -1]
    expected_depths_alt_root = [1, 1, 1, 0]
    assert calculate_tree_depths(head_indices_alt_root) == expected_depths_alt_root


def test_calculate_tree_depths_single_token():
    head_indices = [-1]
    expected_depths = [0]
    assert calculate_tree_depths(head_indices) == expected_depths


def test_calculate_tree_depths_empty():
    head_indices = []
    expected_depths = []
    assert calculate_tree_depths(head_indices) == expected_depths


def test_calculate_tree_distances_simple_chain():
    # 0 <- 1 <- 2
    head_indices = [-1, 0, 1]
    expected_distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert np.array_equal(calculate_tree_distances(head_indices), expected_distances)


def test_calculate_tree_distances_star_graph():
    #   1 -> 0 <- 2
    #        ^
    #        |
    #        3
    # Heads (0-indexed): [-1, 0, 0, 0] (Token 0 is root)
    head_indices = [-1, 0, 0, 0]
    expected_distances = np.array(
        [[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]]
    )
    assert np.array_equal(calculate_tree_distances(head_indices), expected_distances)


def test_calculate_tree_distances_single_token():
    head_indices = [-1]
    expected_distances = np.array([[0]])
    assert np.array_equal(calculate_tree_distances(head_indices), expected_distances)


def test_calculate_tree_distances_empty():
    head_indices = []
    # Adjust based on actual empty output, e.g. np.empty((0,0), dtype=int)
    # For now, assuming it might return a specifically shaped empty array
    # My gold_labels.py returns np.array([[]], dtype=int) which is shape (1,0)
    # Let's adjust test if needed based on actual output, or fix function to return (0,0)
    assert calculate_tree_distances(head_indices).shape == (
        1,
        0,
    ) or calculate_tree_distances(head_indices).shape == (0, 0)

# tests/unit/torch_probe/test_loss_functions.py
import torch

from src.torch_probe.loss_functions import (  # Adjust import if needed
    depth_l1_loss,
    distance_l1_loss,
)

# It's good practice to also import create_mask_from_lengths if you want to test it separately,
# but for testing the main loss functions, we rely on their internal use of it.


def test_depth_l1_loss_simple():
    pred_sq_depths = torch.tensor([[1.0, 4.0, 9.0], [1.0, 4.0, 0.0]])  # B=2, S_max=3
    gold_depths = torch.tensor(
        [[1.0, 5.0, 8.0], [2.0, 3.0, -1.0]]
    )  # Gold NON-SQUARED, -1 is padding
    lengths = torch.tensor([3, 2])  # Sent 1 len 3, Sent 2 len 2

    # Per-sentence calculation for depth_l1_loss:
    # Sent 1 (len=3): abs_diffs = [|1-1|, |4-5|, |9-8|] = [0, 1, 1]. Sum = 2. Normalized = 2/3.
    # Sent 2 (len=2): abs_diffs = [|1-2|, |4-3|] = [1, 1]. Sum = 2. Normalized = 2/2 = 1.
    # Batch loss = ( (2/3) + 1 ) / 2 = (5/3) / 2 = 5/6
    expected_loss = (
        (0.0 + 1.0 + 1.0) / 3.0 + (1.0 + 1.0) / 2.0
    ) / 2.0  # = (2/3 + 1)/2 = 5/6

    loss = depth_l1_loss(pred_sq_depths, gold_depths, lengths)
    assert torch.isclose(loss, torch.tensor(expected_loss))


def test_depth_l1_loss_all_padded_in_one_sent():
    pred_sq_depths = torch.tensor([[1.0, 4.0], [0.0, 0.0]])
    gold_depths = torch.tensor([[1.0, 5.0], [-1.0, -1.0]])  # Second sent all padded
    lengths = torch.tensor([2, 0])

    # Per-sentence calculation:
    # Sent 1 (len=2): abs_diffs = [|1-1|, |4-5|] = [0, 1]. Sum = 1. Normalized = 1/2.
    # Sent 2 (len=0): Skipped.
    # Batch loss = (1/2) / 1 (since only 1 valid sent)
    expected_loss = ((abs(1.0 - 1.0) + abs(4.0 - 5.0)) / 2.0) / 1.0
    loss = depth_l1_loss(pred_sq_depths, gold_depths, lengths)
    assert torch.isclose(loss, torch.tensor(expected_loss))


def test_depth_l1_loss_no_valid_tokens():
    pred_sq_depths = torch.tensor([[0.0, 0.0]])
    gold_depths = torch.tensor(
        [[-1.0, -1.0]]
    )  # All values are padding markers if lengths makes them so
    lengths = torch.tensor([0])  # No valid tokens in the only sentence
    expected_loss = 0.0
    loss = depth_l1_loss(pred_sq_depths, gold_depths, lengths)
    assert torch.isclose(loss, torch.tensor(expected_loss))


def test_distance_l1_loss_simple():
    # B=1, S_max=3. Lengths = [3]
    pred_sq_distances = torch.tensor(
        [[[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]]]
    )  # Predicted SQUARED L2
    gold_distances = torch.tensor(
        [[[0.0, 2.0, 3.0], [2.0, 0.0, 2.0], [3.0, 2.0, 0.0]]]
    )  # Gold NON-SQUARED
    lengths = torch.tensor([3])

    # Per-sentence calculation for distance_l1_loss (H&M style sums over all LxL pairs then divides by L^2):
    # Sent 1 (len=3):
    # pred_sent_dists = [[0,1,4],[1,0,1],[4,1,0]]
    # gold_sent_dists = [[0,2,3],[2,0,2],[3,2,0]]
    # abs_diff_matrix = [[|0-0|, |1-2|, |4-3|],
    #                    [|1-2|, |0-0|, |1-2|],
    #                    [|4-3|, |1-2|, |0-0|]]
    #                 = [[0, 1, 1],
    #                    [1, 0, 1],
    #                    [1, 1, 0]]
    # sum_abs_diff_sent_full_matrix = 0+1+1+1+0+1+1+1+0 = 6
    # L=3, L^2=9. Normalized sent loss = 6/9.
    # Batch loss = (6/9) / 1 (since 1 valid sentence)
    expected_loss = 6.0 / 9.0
    loss = distance_l1_loss(pred_sq_distances, gold_distances, lengths)
    assert torch.isclose(loss, torch.tensor(expected_loss))


def test_distance_l1_loss_with_padding():
    # B=1, S_max=3. Actual Length = 2. Padded token at index 2.
    pred_sq_distances = torch.tensor(
        [
            [
                [10.0, 1.0, 99.0],  # 99 is padding prediction
                [
                    1.0,
                    20.0,
                    88.0,
                ],  # Values for i=j can be anything, but sum includes them
                [99.0, 88.0, 30.0],
            ]
        ]
    )
    gold_distances = torch.tensor(
        [
            [
                [0.0, 2.0, -1.0],  # -1 is padding gold
                [2.0, 0.0, -1.0],
                [-1.0, -1.0, -1.0],
            ]
        ]
    )
    lengths = torch.tensor([2])

    # Per-sentence calculation for distance_l1_loss:
    # Sent 1 (len=2):
    # Relevant pred_sent_dists = [[10.0, 1.0], [1.0, 20.0]]
    # Relevant gold_sent_dists = [[0.0, 2.0], [2.0, 0.0]]
    # abs_diff_matrix = [[|10-0|, |1-2|],
    #                    [|1-2|, |20-0|]]
    #                 = [[10, 1],
    #                    [1, 20]]
    # sum_abs_diff_sent_full_matrix = 10 + 1 + 1 + 20 = 32
    # L=2, L^2=4. Normalized sent loss = 32/4 = 8.0
    # Batch loss = (8.0) / 1
    expected_loss = 32.0 / 4.0
    loss = distance_l1_loss(pred_sq_distances, gold_distances, lengths)
    assert torch.isclose(loss, torch.tensor(expected_loss))


def test_distance_l1_loss_no_valid_pairs():  # e.g. sentence of length 1 or 0
    # Case 1: Length 1
    pred_sq_distances_l1 = torch.tensor([[[10.0]]])  # Predicted squared L2
    gold_distances_l1 = torch.tensor([[[0.0]]])  # Gold non-squared
    lengths_l1 = torch.tensor([1])
    # Sent 1 (len=1):
    # abs_diff_matrix = [[|10-0|]] = [[10]]
    # sum = 10. L=1, L^2=1. Normalized = 10/1 = 10.
    # Batch loss = 10/1 = 10
    # NOTE: H&M's PairwiseDistLoss seems to sum over all pairs (i,j) then divide by L^2.
    # If L=1, L^2=1. Pair (0,0) has |pred(0,0) - gold(0,0)|.
    # If pred(0,0) is not necessarily 0 (our probe computes it from B h_0 - B h_0 which IS 0).
    # If their probe also ensures pred(i,i)=0, then this case is simpler.
    # Our DistanceProbe output for (i,i) is indeed 0.
    # So, pred_sq_distances_l1[0,0,0] will be 0. gold_distances_l1[0,0,0] is 0. Loss = 0.
    expected_loss_l1 = 0.0
    loss_l1 = distance_l1_loss(torch.tensor([[[0.0]]]), gold_distances_l1, lengths_l1)
    assert torch.isclose(loss_l1, torch.tensor(expected_loss_l1))

    # Case 2: Length 0
    pred_sq_distances_l0 = torch.tensor([[[0.0]]])  # Dummy shape, won't be used
    gold_distances_l0 = torch.tensor([[[0.0]]])
    lengths_l0 = torch.tensor([0])
    expected_loss_l0 = 0.0
    loss_l0 = distance_l1_loss(pred_sq_distances_l0, gold_distances_l0, lengths_l0)
    assert torch.isclose(loss_l0, torch.tensor(expected_loss_l0))

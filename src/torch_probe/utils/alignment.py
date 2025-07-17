# src/torch_probe/utils/alignment.py
# This file is used to align subword embeddings to word-level embeddings.
# It is used in the extract_embeddings.py script.

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import torch

log = logging.getLogger(__name__)


def robust_align_subword_embeddings(
    subword_embeddings: torch.Tensor,
    word_ids: List[Optional[int]],
    num_original_words: int,
    alignment_strategy: str = "mean",
) -> torch.Tensor:
    """
    Aligns subword embeddings to word-level embeddings using the word_ids list
    from the Hugging Face tokenizer.

    Args:
        subword_embeddings: Tensor of subword embeddings (num_subwords, dim),
                            INCLUDING embeddings for special tokens like [CLS], [SEP].
        word_ids: List mapping each subword token index to its original word index,
                  or None for special tokens.
        num_original_words: The number of words in the original sentence.
        alignment_strategy: "mean", "first".

    Returns:
        Tensor of word-level embeddings (num_original_words, dim).
    """
    if subword_embeddings.numel() == 0:
        return torch.empty((0, 0), device=subword_embeddings.device)

    word_to_subword_indices: Dict[int, List[int]] = defaultdict(list)
    for subword_idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            if 0 <= subword_idx < subword_embeddings.shape[0]:
                word_to_subword_indices[word_idx].append(subword_idx)
            else:
                log.warning(
                    f"Subword index {subword_idx} from word_ids is out of bounds "
                    f"for subword_embeddings shape {subword_embeddings.shape[0]}. Skipping."
                )

    aligned_embeddings_list: List[torch.Tensor] = []
    for i in range(num_original_words):
        subword_indices_for_word = word_to_subword_indices.get(i)
        if subword_indices_for_word:
            valid_sub_indices = [
                idx for idx in subword_indices_for_word if 0 <= idx < subword_embeddings.shape[0]
            ]
            if not valid_sub_indices:
                log.warning(f"Word {i} had no valid subword indices. Appending zero vector.")
                aligned_embeddings_list.append(
                    torch.zeros(subword_embeddings.size(1), device=subword_embeddings.device)
                )
                continue

            word_sub_embeddings = subword_embeddings[valid_sub_indices]
            if alignment_strategy == "mean":
                aligned_embeddings_list.append(torch.mean(word_sub_embeddings, dim=0))
            elif alignment_strategy == "first":
                aligned_embeddings_list.append(word_sub_embeddings[0])
            else:
                raise ValueError(f"Unknown alignment strategy: {alignment_strategy}")
        else:
            log.warning(
                f"Word at index {i} has no corresponding subwords. Appending zero vector."
            )
            aligned_embeddings_list.append(
                torch.zeros(subword_embeddings.size(1), device=subword_embeddings.device)
            )

    if not aligned_embeddings_list:
        return torch.empty((0, subword_embeddings.size(1)), device=subword_embeddings.device)

    return torch.stack(aligned_embeddings_list)
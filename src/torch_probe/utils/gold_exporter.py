from __future__ import annotations

from pathlib import Path
from typing import List

import h5py
import numpy as np

from .gold_labels import calculate_tree_depths, calculate_tree_distances


def export_gold_geometry_hdf5(
    *,
    parsed_sentences: List[dict],
    conllu_path: Path,
    output_hdf5_path: Path,
    dataset_name: str,
    split_name: str,
) -> Path:
    """
    Export gold geometry (depths and pairwise distances) for a dataset split to HDF5.

    Structure:
      - file attrs:
          dataset_name, split_name, original_conllu_file
      - groups per sentence: f"sent_{idx}"
          datasets:
            - depths: (L,) float32
            - distances: (L,L) float32
            - head_indices: (L,) int32
            - tokens: (L,) variable-length UTF-8 strings
            - xpos_tags: (L,) variable-length UTF-8 strings
            - upos_tags: (L,) variable-length UTF-8 strings

    Returns the path to the written HDF5 file.
    """

    output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_hdf5_path, "w") as hf:
        hf.attrs["dataset_name"] = dataset_name
        hf.attrs["split_name"] = split_name
        hf.attrs["original_conllu_file"] = str(conllu_path)
        hf.attrs["format"] = "gold_geometry_v1"

        for idx, sent in enumerate(parsed_sentences):
            tokens = sent["tokens"]
            head_indices = sent["head_indices"]
            xpos_tags = sent.get("xpos_tags", ["?"] * len(tokens))
            upos_tags = sent.get("upos_tags", ["?"] * len(tokens))

            depths = np.array(calculate_tree_depths(head_indices), dtype=np.float32)
            distances = np.array(
                calculate_tree_distances(head_indices), dtype=np.float32
            )

            grp = hf.create_group(f"sent_{idx}")
            grp.create_dataset("depths", data=depths, dtype=np.float32)
            grp.create_dataset("distances", data=distances, dtype=np.float32)
            grp.create_dataset(
                "head_indices", data=np.asarray(head_indices, dtype=np.int32), dtype=np.int32
            )
            grp.create_dataset("tokens", data=np.asarray(tokens, dtype=str_dtype), dtype=str_dtype)
            grp.create_dataset("xpos_tags", data=np.asarray(xpos_tags, dtype=str_dtype), dtype=str_dtype)
            grp.create_dataset("upos_tags", data=np.asarray(upos_tags, dtype=str_dtype), dtype=str_dtype)

    return output_hdf5_path


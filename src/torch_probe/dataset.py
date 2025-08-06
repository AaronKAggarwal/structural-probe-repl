# src/torch_probe/dataset.py
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # Import tqdm

from .utils.conllu_reader import SentenceData, read_conll_file
from .utils.embedding_loader import load_embeddings_for_sentence
from .utils.gold_labels import calculate_tree_depths, calculate_tree_distances


class ProbeDataset(Dataset):
    def __init__(
        self,
        conllu_filepath: str,
        hdf5_filepath: str,
        embedding_layer_index: int,
        probe_task_type: str,
        embedding_dim: Optional[int] = None,
        preload: bool = True,  # <-- Add preload parameter
    ):
        super().__init__()

        if probe_task_type not in ["distance", "depth"]:
            raise ValueError("probe_task_type must be 'distance' or 'depth'")

        self.conllu_filepath = conllu_filepath
        self.hdf5_filepath = hdf5_filepath
        self.embedding_layer_index = embedding_layer_index
        self.probe_task_type = probe_task_type
        self.embedding_dim = embedding_dim
        self.preload = preload # Store the preload choice

        self.parsed_sentences: List[SentenceData] = read_conll_file(conllu_filepath)
        if not self.parsed_sentences:
            raise ValueError(f"No sentences found or parsed from {conllu_filepath}")

        if "xpos_tags" not in self.parsed_sentences[0]:
            raise ValueError(
                f"Parsed sentence data from {conllu_filepath} is missing 'xpos_tags'."
            )

        # Infer embedding dimension if not provided
        if self.embedding_dim is None:
            if len(self.parsed_sentences) > 0:
                try:
                    with h5py.File(self.hdf5_filepath, "r") as hf:
                        first_sent_emb = load_embeddings_for_sentence(
                            hf, "0", self.embedding_layer_index
                        )
                except Exception as e:
                    raise IOError(f"Could not open HDF5 file {hdf5_filepath}: {e}")

                if first_sent_emb is not None and first_sent_emb.ndim == 2:
                    self.embedding_dim = first_sent_emb.shape[1]
                else:
                    raise ValueError(
                        f"Could not infer embedding dimension from HDF5 file {hdf5_filepath}. Please provide embedding_dim."
                    )
            else:
                raise ValueError("Cannot infer embedding dimension from empty dataset.")
        
        self.gold_labels: List[np.ndarray] = self._calculate_all_gold_labels()

        self.preloaded_data: Optional[List[Dict[str, Any]]] = None
        if self.preload:
            print(f"Pre-loading data from {Path(self.conllu_filepath).name} into RAM...")
            self.preloaded_data = []
            # Open the file once for the entire pre-loading process
            with h5py.File(self.hdf5_filepath, "r") as hdf5_file_object:
                for i in tqdm(range(len(self.parsed_sentences)), desc="Pre-loading"):
                    self.preloaded_data.append(self._get_item_data(i, hdf5_file_object))
            print("Pre-loading complete.")

    def _calculate_all_gold_labels(self) -> List[np.ndarray]:
        """Helper to compute all gold labels during initialization."""
        labels = []
        for i, sent_data in enumerate(self.parsed_sentences):
            if "head_indices" not in sent_data:
                raise ValueError(
                    f"Sentence {i} from {self.conllu_filepath} is missing 'head_indices'."
                )
            if self.probe_task_type == "distance":
                distances = calculate_tree_distances(sent_data["head_indices"])
                labels.append(distances.astype(np.float32))
            elif self.probe_task_type == "depth":
                depths = calculate_tree_depths(sent_data["head_indices"])
                labels.append(np.array(depths, dtype=np.float32))
        return labels

    def _get_item_data(self, idx: int, hdf5_file_object: h5py.File) -> Dict[str, Any]:
        """Core logic to retrieve a single item, assuming an open HDF5 file."""
        sentence_data = self.parsed_sentences[idx]
        sentence_key = str(idx)

        embeddings_np = load_embeddings_for_sentence(
            hdf5_file_object, sentence_key, self.embedding_layer_index
        )

        if embeddings_np is None:
            raise RuntimeError(
                f"Failed to load embeddings for sentence key '{sentence_key}' from {self.hdf5_filepath}."
            )

        num_tokens_conll = len(sentence_data["tokens"])
        num_tokens_hdf5 = embeddings_np.shape[0]

        if num_tokens_hdf5 != num_tokens_conll:
            raise AssertionError(
                f"Token count mismatch for sentence key '{sentence_key}' (index {idx}): "
                f"CoNLL file has {num_tokens_conll} tokens, "
                f"HDF5 file has {num_tokens_hdf5} embedding vectors. "
                f"CoNLL Tokens: {' '.join(sentence_data['tokens'])}"
            )

        embeddings_tensor = torch.from_numpy(embeddings_np).float()
        gold_labels_tensor = torch.from_numpy(self.gold_labels[idx]).float()

        return {
            "embeddings": embeddings_tensor,
            "gold_labels": gold_labels_tensor,
            "tokens": sentence_data["tokens"],
            "head_indices": sentence_data["head_indices"],
            "upos_tags": sentence_data["upos_tags"],
            "xpos_tags": sentence_data["xpos_tags"],
            "length": num_tokens_conll,
        }

    def __len__(self) -> int:
        return len(self.parsed_sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not (0 <= idx < len(self.parsed_sentences)):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self.parsed_sentences)}"
            )

        if self.preloaded_data is not None:
            return self.preloaded_data[idx]
        else:
            # On-the-fly loading (slower, but memory efficient)
            with h5py.File(self.hdf5_filepath, "r") as hdf5_file_object:
                return self._get_item_data(idx, hdf5_file_object)


    def close_hdf5(self):
        """This method is now a no-op for on-disk loading, as file handles
        are managed within __getitem__. It remains for compatibility and
        potential use in pre-loading subclasses."""
        pass

    def __del__(self):
        """No persistent file handle to close."""
        pass


def collate_probe_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {
            "embeddings_batch": torch.empty(0),
            "labels_batch": torch.empty(0),
            "lengths_batch": torch.empty(0, dtype=torch.long),
            "tokens_batch": [],
            "head_indices_batch": [],
            "upos_tags_batch": [],
            "xpos_tags_batch": [],
        }

    embeddings_list = [item["embeddings"] for item in batch]
    gold_labels_list = [item["gold_labels"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    padded_embeddings = nn.utils.rnn.pad_sequence(
        embeddings_list, batch_first=True, padding_value=0.0
    )
    max_len = padded_embeddings.shape[1]

    first_label_shape_dim = gold_labels_list[0].ndim
    if first_label_shape_dim == 1:  # Depth task
        padded_labels = torch.full((len(batch), max_len), -1.0, dtype=torch.float32)
        for i, lbl in enumerate(gold_labels_list):
            padded_labels[i, : lengths[i]] = lbl
    elif first_label_shape_dim == 2:  # Distance task
        padded_labels = torch.full(
            (len(batch), max_len, max_len), -1.0, dtype=torch.float32
        )
        for i, lbl_matrix in enumerate(gold_labels_list):
            seq_len = lengths[i]
            padded_labels[i, :seq_len, :seq_len] = lbl_matrix
    else:
        raise ValueError(f"Unsupported gold_label shape: {gold_labels_list[0].shape}")

    tokens_batch = [item["tokens"] for item in batch]
    head_indices_batch = [item["head_indices"] for item in batch]
    upos_tags_batch = [item["upos_tags"] for item in batch]
    xpos_tags_batch = [item["xpos_tags"] for item in batch]

    return {
        "embeddings_batch": padded_embeddings,
        "labels_batch": padded_labels,
        "lengths_batch": lengths,
        "tokens_batch": tokens_batch,
        "head_indices_batch": head_indices_batch,
        "upos_tags_batch": upos_tags_batch,
        "xpos_tags_batch": xpos_tags_batch,
    }

# Main block remains the same for standalone testing
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    conllu_train_path_str = str(
        project_root
        / "data/ptb_stanford_dependencies_conllx/ptb3-wsj-TINY_SAMPLE.conllx"
    )
    hdf5_train_path_str = str(
        project_root
        / "data/embeddings_sanity_check/bert-base-cased_sample_dev_layers-0_6_12_align-mean.hdf5"
    )

    if (
        not Path(conllu_train_path_str).exists()
        or not Path(hdf5_train_path_str).exists()
    ):
        print("Please ensure sample files are available to run standalone test.")
    else:
        print("Attempting to load ELMo (layer 2) for distance task from sample data...")
        try:
            distance_dataset = ProbeDataset(
                conllu_filepath=conllu_train_path_str,
                hdf5_filepath=hdf5_train_path_str,
                embedding_layer_index=0,
                probe_task_type="distance",
                embedding_dim=768,
            )
            print(
                f"Distance dataset loaded. Number of sentences: {len(distance_dataset)}"
            )

            if len(distance_dataset) > 0:
                sample_item_dist = distance_dataset[0]
                print(f"Sample item (distance task) keys: {sample_item_dist.keys()}")
                print(f"  Tokens: {sample_item_dist['tokens']}")
                print(
                    f"  XPOS Tags: {sample_item_dist['xpos_tags']}"
                )  # Check this new field
                print(f"  Embeddings shape: {sample_item_dist['embeddings'].shape}")
                print(f"  Gold labels shape: {sample_item_dist['gold_labels'].shape}")

                distance_loader = DataLoader(
                    distance_dataset,
                    batch_size=2,
                    collate_fn=collate_probe_batch,
                    shuffle=True,
                )
                for i, batch in enumerate(distance_loader):
                    print(f"\nBatch {i + 1} (distance task):")
                    print(
                        f"  Padded Embeddings shape: {batch['embeddings_batch'].shape}"
                    )
                    print(
                        f"  XPOS Tags (first sentence in batch): {batch['xpos_tags_batch'][0] if batch['xpos_tags_batch'] else 'N/A'}"
                    )
                    if i == 0:  # Print only one batch to keep output concise
                        break
            distance_dataset.close_hdf5()

        except Exception as e:
            print(f"Error during ProbeDataset example usage: {e}")
            import traceback

            traceback.print_exc()

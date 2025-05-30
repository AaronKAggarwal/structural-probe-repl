# src/torch_probe/dataset.py
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path # Added for __main__ example path checking

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import h5py

# Assuming these are in src/torch_probe/utils/
# Use the potentially renamed read_conll_file if you adopted that change
from .utils.conllu_reader import read_conll_file, SentenceData 
from .utils.gold_labels import calculate_tree_depths, calculate_tree_distances
from .utils.embedding_loader import load_elmo_embeddings_for_sentence # Assuming this stays for ELMo, might need generalization later


class ProbeDataset(Dataset):
    def __init__(self, 
                 conllu_filepath: str, 
                 hdf5_filepath: str, 
                 embedding_layer_index: int, 
                 probe_task_type: str,
                 embedding_dim: Optional[int] = None):
        super().__init__()

        if probe_task_type not in ["distance", "depth"]:
            raise ValueError("probe_task_type must be 'distance' or 'depth'")

        self.conllu_filepath = conllu_filepath
        self.hdf5_filepath = hdf5_filepath
        self.embedding_layer_index = embedding_layer_index
        self.probe_task_type = probe_task_type
        self.embedding_dim = embedding_dim

        # 1. Parse CoNLL data (CoNLL-U or CoNLL-X)
        # The read_conll_file function should now provide 'xpos_tags'
        self.parsed_sentences: List[SentenceData] = read_conll_file(conllu_filepath)
        if not self.parsed_sentences:
            raise ValueError(f"No sentences found or parsed from {conllu_filepath}")

        # Verify that xpos_tags are present (essential for H&M alignment)
        if self.parsed_sentences and 'xpos_tags' not in self.parsed_sentences[0]:
            raise ValueError(f"Parsed sentence data from {conllu_filepath} is missing 'xpos_tags'. "
                             "Ensure conllu_reader.py extracts them.")


        # 2. Open HDF5 file
        try:
            self.hdf5_file_object = h5py.File(hdf5_filepath, 'r')
        except Exception as e:
            raise IOError(f"Could not open HDF5 file {hdf5_filepath}: {e}")

        # 3. Pre-calculate gold labels
        self.gold_labels: List[np.ndarray] = []
        for i, sent_data in enumerate(self.parsed_sentences):
            if 'head_indices' not in sent_data:
                raise ValueError(f"Sentence {i} from {conllu_filepath} is missing 'head_indices'.")
            
            if self.probe_task_type == "distance":
                distances = calculate_tree_distances(sent_data['head_indices'])
                self.gold_labels.append(distances.astype(np.float32))
            elif self.probe_task_type == "depth":
                depths = calculate_tree_depths(sent_data['head_indices'])
                self.gold_labels.append(np.array(depths, dtype=np.float32))
        
        # Infer embedding dimension
        if self.embedding_dim is None:
            if len(self.parsed_sentences) > 0:
                first_sent_emb = load_elmo_embeddings_for_sentence(
                    self.hdf5_file_object, "0", self.embedding_layer_index
                )
                if first_sent_emb is not None and first_sent_emb.ndim == 2:
                    self.embedding_dim = first_sent_emb.shape[1]
                    # print(f"Inferred embedding dimension: {self.embedding_dim} from HDF5 file.") # Less verbose
                else:
                    raise ValueError(f"Could not infer embedding dimension from HDF5 file {hdf5_filepath} "
                                     f"for key '0'. Please provide embedding_dim.")
            else: # Should not happen if file parsing worked and raised error for no sentences
                 raise ValueError("Cannot infer embedding dimension from empty dataset and no embedding_dim provided.")


    def __len__(self) -> int:
        return len(self.parsed_sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not (0 <= idx < len(self.parsed_sentences)):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.parsed_sentences)}")

        sentence_data = self.parsed_sentences[idx]
        sentence_key = str(idx) 

        embeddings_np = load_elmo_embeddings_for_sentence(
            self.hdf5_file_object, 
            sentence_key, 
            self.embedding_layer_index
        )

        if embeddings_np is None:
            # This might indicate the HDF5 file is missing the key or is corrupted for this sentence
            raise RuntimeError(f"Failed to load embeddings for sentence key '{sentence_key}' "
                               f"from {self.hdf5_filepath}. HDF5 key might be missing or data corrupted.")

        num_tokens_conll = len(sentence_data['tokens'])
        num_tokens_hdf5 = embeddings_np.shape[0]
        
        if num_tokens_hdf5 != num_tokens_conll:
            raise AssertionError(
                f"Token count mismatch for sentence key '{sentence_key}' (index {idx}): "
                f"CoNLL file has {num_tokens_conll} tokens, "
                f"HDF5 file has {num_tokens_hdf5} embedding vectors. "
                f"CoNLL Tokens: {' '.join(sentence_data['tokens'])}"
            )
        
        # Ensure all required fields are present from conllu_reader
        for field in ['tokens', 'head_indices', 'upos_tags', 'xpos_tags']:
            if field not in sentence_data:
                raise ValueError(f"Sentence data for index {idx} is missing '{field}'. "
                                 f"Check output of conllu_reader.py from {self.conllu_filepath}.")

        embeddings_tensor = torch.from_numpy(embeddings_np).float()
        gold_labels_tensor = torch.from_numpy(self.gold_labels[idx]).float()
        
        return {
            "embeddings": embeddings_tensor,
            "gold_labels": gold_labels_tensor,
            "tokens": sentence_data['tokens'], 
            "head_indices": sentence_data['head_indices'], 
            "upos_tags": sentence_data['upos_tags'],
            "xpos_tags": sentence_data['xpos_tags'], # <<< ADDED XPOS_TAGS
            "length": num_tokens_conll 
        }

    def close_hdf5(self):
        """Closes the HDF5 file object if it's open."""
        # Check if the attribute exists and if it's not None AND if it has a 'close' method (duck typing)
        if hasattr(self, 'hdf5_file_object') and \
           self.hdf5_file_object is not None and \
           hasattr(self.hdf5_file_object, 'id') and \
           self.hdf5_file_object.id.valid: # h5py specific way to check if file is open
            try:
                # print(f"DEBUG: Closing HDF5 file {self.hdf5_filepath} in close_hdf5()")
                self.hdf5_file_object.close()
            except Exception as e:
                # Use logging if available, or print for standalone script
                # log.warning(f"Warning: Error closing HDF5 file {self.hdf5_filepath} during explicit close: {e}")
                print(f"Warning: Error closing HDF5 file {self.hdf5_filepath} during explicit close: {e}")
        # else:
            # print(f"DEBUG: HDF5 file {self.hdf5_filepath} already closed or not properly initialized.")
        self.hdf5_file_object = None # Set to None after closing to prevent re-closing issues

    def __del__(self):
        """Ensure HDF5 file is closed when Dataset object is garbage collected."""
        # print(f"DEBUG: ProbeDataset.__del__ called for {self.conllu_filepath}")
        self.close_hdf5()


def collate_probe_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch: # Handle empty batch case
        return {
            "embeddings_batch": torch.empty(0),
            "labels_batch": torch.empty(0),
            "lengths_batch": torch.empty(0, dtype=torch.long),
            "tokens_batch": [],
            "head_indices_batch": [],
            "upos_tags_batch": [],
            "xpos_tags_batch": [] # <<< ADDED XPOS_TAGS_BATCH
        }

    embeddings_list = [item['embeddings'] for item in batch]
    gold_labels_list = [item['gold_labels'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    padded_embeddings = nn.utils.rnn.pad_sequence(embeddings_list, batch_first=True, padding_value=0.0) 
    max_len = padded_embeddings.shape[1] 
    
    first_label_shape_dim = gold_labels_list[0].ndim
    if first_label_shape_dim == 1: # Depth task
        padded_labels = torch.full((len(batch), max_len), -1.0, dtype=torch.float32)
        for i, lbl in enumerate(gold_labels_list):
            padded_labels[i, :lengths[i]] = lbl
    elif first_label_shape_dim == 2: # Distance task
        padded_labels = torch.full((len(batch), max_len, max_len), -1.0, dtype=torch.float32)
        for i, lbl_matrix in enumerate(gold_labels_list):
            seq_len = lengths[i]
            padded_labels[i, :seq_len, :seq_len] = lbl_matrix
    else:
        raise ValueError(f"Unsupported gold_label shape: {gold_labels_list[0].shape}")

    tokens_batch = [item['tokens'] for item in batch]
    head_indices_batch = [item['head_indices'] for item in batch]
    upos_tags_batch = [item['upos_tags'] for item in batch]
    xpos_tags_batch = [item['xpos_tags'] for item in batch] # <<< ADDED XPOS_TAGS_BATCH
    
    return {
        "embeddings_batch": padded_embeddings,
        "labels_batch": padded_labels,
        "lengths_batch": lengths,
        "tokens_batch": tokens_batch, 
        "head_indices_batch": head_indices_batch, 
        "upos_tags_batch": upos_tags_batch,
        "xpos_tags_batch": xpos_tags_batch # <<< ADDED XPOS_TAGS_BATCH
    }

if __name__ == '__main__':
    # Note: This example will only work if read_conll_file in conllu_reader.py
    # has been updated to return 'xpos_tags' in its SentenceData dict.
    # Otherwise, the ProbeDataset __init__ will raise a ValueError.

    # Use pathlib for path construction
    project_root = Path(__file__).resolve().parent.parent.parent 
    conllu_train_path_str = str(project_root / "data/ptb_stanford_dependencies_conllx/ptb3-wsj-TINY_SAMPLE.conllx")
    hdf5_train_path_str = str(project_root / "data/embeddings_sanity_check/bert-base-cased_sample_dev_layers-0_6_12_align-mean.hdf5")
    
    if not Path(conllu_train_path_str).exists() or not Path(hdf5_train_path_str).exists():
        print("Please ensure CoNLL sample files from whykay-01 fork are in:")
        print(f"  CoNLL path: {Path(conllu_train_path_str).resolve()}")
        print(f"  HDF5 path:  {Path(hdf5_train_path_str).resolve()}")
        print("Skipping standalone ProbeDataset example.")
    else:
        print(f"Attempting to load ELMo (layer 2) for distance task from sample data...")
        try:
            distance_dataset = ProbeDataset(
                conllu_filepath=conllu_train_path_str,
                hdf5_filepath=hdf5_train_path_str,
                embedding_layer_index=0, 
                probe_task_type="distance",
                embedding_dim=768
            )
            print(f"Distance dataset loaded. Number of sentences: {len(distance_dataset)}")
            
            if len(distance_dataset) > 0:
                sample_item_dist = distance_dataset[0]
                print(f"Sample item (distance task) keys: {sample_item_dist.keys()}")
                print(f"  Tokens: {sample_item_dist['tokens']}")
                print(f"  XPOS Tags: {sample_item_dist['xpos_tags']}") # Check this new field
                print(f"  Embeddings shape: {sample_item_dist['embeddings'].shape}")
                print(f"  Gold labels shape: {sample_item_dist['gold_labels'].shape}")

                distance_loader = DataLoader(distance_dataset, batch_size=2, collate_fn=collate_probe_batch, shuffle=True)
                for i, batch in enumerate(distance_loader):
                    print(f"\nBatch {i+1} (distance task):")
                    print(f"  Padded Embeddings shape: {batch['embeddings_batch'].shape}")
                    print(f"  XPOS Tags (first sentence in batch): {batch['xpos_tags_batch'][0] if batch['xpos_tags_batch'] else 'N/A'}")
                    if i == 0: # Print only one batch to keep output concise
                        break 
            distance_dataset.close_hdf5() 

        except Exception as e:
            print(f"Error during ProbeDataset example usage: {e}")
            import traceback
            traceback.print_exc()
# src/torch_probe/dataset.py
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import h5py

# Assuming these are in src/torch_probe/utils/
from .utils.conllu_reader import read_conllu_file, SentenceData 
from .utils.gold_labels import calculate_tree_depths, calculate_tree_distances
from .utils.embedding_loader import load_elmo_embeddings_for_sentence


class ProbeDataset(Dataset):
    def __init__(self, 
                 conllu_filepath: str, 
                 hdf5_filepath: str, 
                 embedding_layer_index: int, 
                 probe_task_type: str,
                 embedding_dim: Optional[int] = None): # Optional: for verification or if not inferable
        super().__init__()

        if probe_task_type not in ["distance", "depth"]:
            raise ValueError("probe_task_type must be 'distance' or 'depth'")

        self.conllu_filepath = conllu_filepath
        self.hdf5_filepath = hdf5_filepath
        self.embedding_layer_index = embedding_layer_index
        self.probe_task_type = probe_task_type
        self.embedding_dim = embedding_dim # Store if provided

        # 1. Parse CoNLL-U data
        self.parsed_sentences: List[SentenceData] = read_conllu_file(conllu_filepath)
        if not self.parsed_sentences:
            raise ValueError(f"No sentences found or parsed from {conllu_filepath}")

        # 2. Open HDF5 file - keep it open for the lifetime of the Dataset object
        # This is more efficient than opening/closing for each __getitem__
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
                self.gold_labels.append(distances.astype(np.float32)) # For L1 loss consistency
            elif self.probe_task_type == "depth":
                depths = calculate_tree_depths(sent_data['head_indices'])
                self.gold_labels.append(np.array(depths, dtype=np.float32))
        
        # Infer embedding dimension from the first sentence if not provided
        if self.embedding_dim is None and len(self.parsed_sentences) > 0:
            # Try to load first sentence embedding to infer dim
            first_sent_emb = load_elmo_embeddings_for_sentence(self.hdf5_file_object, "0", self.embedding_layer_index)
            if first_sent_emb is not None and first_sent_emb.ndim == 2:
                self.embedding_dim = first_sent_emb.shape[1]
                print(f"Inferred embedding dimension: {self.embedding_dim} from HDF5 file.")
            else:
                raise ValueError(f"Could not infer embedding dimension from HDF5 file {hdf5_filepath} for key '0'. Please provide embedding_dim.")
        elif self.embedding_dim is None and len(self.parsed_sentences) == 0:
             raise ValueError("Cannot infer embedding dimension from empty dataset and no embedding_dim provided.")


    def __len__(self) -> int:
        return len(self.parsed_sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not (0 <= idx < len(self.parsed_sentences)):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.parsed_sentences)}")

        sentence_data = self.parsed_sentences[idx]
        sentence_key = str(idx) # HDF5 keys are typically strings "0", "1", ...

        embeddings_np = load_elmo_embeddings_for_sentence(
            self.hdf5_file_object, 
            sentence_key, 
            self.embedding_layer_index
        )

        if embeddings_np is None:
            raise RuntimeError(f"Failed to load embeddings for sentence key '{sentence_key}' from {self.hdf5_filepath}")

        # Critical Assertion: Token count match
        num_tokens_conllu = len(sentence_data['tokens'])
        num_tokens_elmo = embeddings_np.shape[0]
        
        if num_tokens_elmo != num_tokens_conllu:
            # This should ideally not happen if data prep (Phase 0a + whykay data) was consistent
            # Or if data.py (legacy) matches the HDF5 source.
            # For this modern probe, we expect alignment to be handled *before* this stage
            # or the HDF5 files to be perfectly aligned.
            raise AssertionError(
                f"Token count mismatch for sentence key '{sentence_key}': "
                f"CoNLL-U has {num_tokens_conllu} tokens, "
                f"ELMo HDF5 has {num_tokens_elmo} embedding vectors. "
                f"Sentence: {' '.join(sentence_data['tokens'])}"
            )
        
        # Convert to PyTorch Tensors
        embeddings_tensor = torch.from_numpy(embeddings_np).float()
        gold_labels_tensor = torch.from_numpy(self.gold_labels[idx]).float()
        
        return {
            "embeddings": embeddings_tensor,
            "gold_labels": gold_labels_tensor,
            "tokens": sentence_data['tokens'], # List of str
            "head_indices": sentence_data['head_indices'], # List of int
            "length": num_tokens_conllu # Original length
        }

    def close_hdf5(self):
        """Closes the HDF5 file object if it's open."""
        if hasattr(self, 'hdf5_file_object') and self.hdf5_file_object:
            try:
                self.hdf5_file_object.close()
                # print(f"HDF5 file {self.hdf5_filepath} closed.")
            except Exception as e:
                print(f"Error closing HDF5 file {self.hdf5_filepath}: {e}")
    
    def __del__(self):
        """Ensure HDF5 file is closed when Dataset object is garbage collected."""
        self.close_hdf5()


def collate_probe_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate_fn for ProbeDataset.
    Pads embedding sequences and gold labels.
    """

    if not batch:
        return {
            "embeddings_batch": torch.empty(0),
            "labels_batch": torch.empty(0),
            "lengths_batch": torch.empty(0, dtype=torch.long),
            "tokens_batch": [],
            "head_indices_batch": []
        }

    embeddings_list = [item['embeddings'] for item in batch]
    gold_labels_list = [item['gold_labels'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    # Pad embeddings
    # padding_value=0.0 is PyTorch default for pad_sequence with float tensors
    padded_embeddings = nn.utils.rnn.pad_sequence(embeddings_list, batch_first=True) 

    # Pad gold labels
    # Determine max_len from the already padded embeddings or original lengths
    max_len = padded_embeddings.shape[1] 
    
    # Assuming gold_labels can be 1D (depth) or 2D (distance matrix)
    first_label_shape_dim = gold_labels_list[0].ndim

    if first_label_shape_dim == 1: # Depth task (num_tokens,)
        padded_labels = torch.full((len(batch), max_len), -1.0, dtype=torch.float32) # Use -1 for padding
        for i, lbl in enumerate(gold_labels_list):
            padded_labels[i, :lengths[i]] = lbl
    elif first_label_shape_dim == 2: # Distance task (num_tokens, num_tokens)
        padded_labels = torch.full((len(batch), max_len, max_len), -1.0, dtype=torch.float32)
        for i, lbl_matrix in enumerate(gold_labels_list):
            seq_len = lengths[i]
            padded_labels[i, :seq_len, :seq_len] = lbl_matrix
    else:
        raise ValueError(f"Unsupported gold_label shape: {gold_labels_list[0].shape}")

    # Keep original tokens and head_indices as lists (not padded/tensored by default here)
    # These are useful for inspection or if specific evaluation needs them.
    tokens_batch = [item['tokens'] for item in batch]
    head_indices_batch = [item['head_indices'] for item in batch]

    return {
        "embeddings_batch": padded_embeddings,
        "labels_batch": padded_labels,
        "lengths_batch": lengths,
        "tokens_batch": tokens_batch, # For debug/analysis
        "head_indices_batch": head_indices_batch # For debug/analysis
    }

if __name__ == '__main__':
    # Example usage:
    # Needs actual CoNLL-U and HDF5 files from Phase 0a (whykay-01 data)
    # Place them in a temporary 'sample_data' directory relative to this script for this example to run
    # Or adjust paths directly.

    # Create dummy data for standalone testing if needed (as in test_embedding_loader.py)
    # For this example, we assume the data files from whykay-01 are accessible
    # via paths like `src/legacy/structural_probe/example/data/en_ewt-ud-sample/`
    
    # Adjust these paths to where your whykay-01 sample data is (vendored in src/legacy/...)
    # This example assumes you run this script from the project root.
    conllu_train_path = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-train.conllu"
    hdf5_train_path = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-train.elmo-layers.hdf5"
    
    # Check if files exist before trying to create dataset
    if not Path(conllu_train_path).exists() or not Path(hdf5_train_path).exists():
        print("Please ensure CoNLL-U and HDF5 sample files from whykay-01 are in:")
        print(f"  {Path(conllu_train_path).resolve()}")
        print(f"  {Path(hdf5_train_path).resolve()}")
        print("Skipping standalone ProbeDataset example.")
    else:
        print(f"Attempting to load ELMo (layer 2) for distance task from sample data...")
        try:
            # Embedding dim for ELMo is 1024
            distance_dataset = ProbeDataset(
                conllu_filepath=conllu_train_path,
                hdf5_filepath=hdf5_train_path,
                embedding_layer_index=2, # ELMo layer 2 (0-indexed)
                probe_task_type="distance",
                embedding_dim=1024 # Provide for ELMo
            )
            print(f"Distance dataset loaded. Number of sentences: {len(distance_dataset)}")
            
            # Test __getitem__ for the first sentence
            sample_item_dist = distance_dataset[0]
            print(f"Sample item (distance task) keys: {sample_item_dist.keys()}")
            print(f"  Embeddings shape: {sample_item_dist['embeddings'].shape}")
            print(f"  Gold labels shape: {sample_item_dist['gold_labels'].shape}")
            print(f"  Num tokens: {len(sample_item_dist['tokens'])}")

            # Test DataLoader with collation
            distance_loader = DataLoader(distance_dataset, batch_size=2, collate_fn=collate_probe_batch, shuffle=True)
            for i, batch in enumerate(distance_loader):
                print(f"\nBatch {i+1} (distance task):")
                print(f"  Padded Embeddings shape: {batch['embeddings_batch'].shape}")
                print(f"  Padded Labels shape:     {batch['labels_batch'].shape}")
                print(f"  Lengths:                 {batch['lengths_batch']}")
                if i == 1: # Print a couple of batches
                    break 
            
            distance_dataset.close_hdf5() # Important to close file

            print(f"\nAttempting to load ELMo (layer 2) for depth task from sample data...")
            depth_dataset = ProbeDataset(
                conllu_filepath=conllu_train_path,
                hdf5_filepath=hdf5_train_path,
                embedding_layer_index=2,
                probe_task_type="depth",
                embedding_dim=1024
            )
            print(f"Depth dataset loaded. Number of sentences: {len(depth_dataset)}")
            sample_item_depth = depth_dataset[0]
            print(f"Sample item (depth task) keys: {sample_item_depth.keys()}")
            print(f"  Embeddings shape: {sample_item_depth['embeddings'].shape}")
            print(f"  Gold labels shape: {sample_item_depth['gold_labels'].shape}")

            depth_loader = DataLoader(depth_dataset, batch_size=2, collate_fn=collate_probe_batch, shuffle=False)
            for i, batch in enumerate(depth_loader):
                print(f"\nBatch {i+1} (depth task):")
                print(f"  Padded Embeddings shape: {batch['embeddings_batch'].shape}")
                print(f"  Padded Labels shape:     {batch['labels_batch'].shape}")
                print(f"  Lengths:                 {batch['lengths_batch']}")
                if i == 1:
                    break
            depth_dataset.close_hdf5()

        except Exception as e:
            print(f"Error during ProbeDataset example usage: {e}")
            import traceback
            traceback.print_exc()
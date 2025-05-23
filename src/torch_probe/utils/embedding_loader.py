# src/torch_probe/utils/embedding_loader.py
from typing import Optional
import numpy as np
import h5py # Make sure h5py is in your pyproject.toml dependencies

def load_elmo_embeddings_for_sentence(
    hdf5_file_object: h5py.File, 
    sentence_key: str, 
    layer_index: int
) -> Optional[np.ndarray]:
    """
    Loads ELMo embeddings for a specific sentence and layer from an open HDF5 file object.

    Args:
        hdf5_file_object: An opened h5py.File object.
        sentence_key: The string key for the sentence (e.g., "0", "1", ...).
        layer_index: The integer index of the ELMo layer to extract (0, 1, or 2).

    Returns:
        A NumPy array of shape (num_tokens, embedding_dimension) for the specified layer,
        or None if the sentence_key is not found or layer_index is invalid.
    """
    if sentence_key not in hdf5_file_object:
        print(f"Warning: Sentence key '{sentence_key}' not found in HDF5 file.")
        return None

    feature_stack = hdf5_file_object[sentence_key][()] 
    # Expected shape: (num_elmo_layers, num_tokens, elmo_embedding_dim)
    # e.g., (3, num_tokens, 1024) for standard ELMo

    if not isinstance(feature_stack, np.ndarray):
        print(f"Warning: Data for key '{sentence_key}' is not a NumPy array.")
        return None

    if feature_stack.ndim != 3:
        print(f"Warning: Feature stack for key '{sentence_key}' has unexpected ndim {feature_stack.ndim} (expected 3). Shape: {feature_stack.shape}")
        return None
    
    num_layers_in_stack = feature_stack.shape[0]
    if not (0 <= layer_index < num_layers_in_stack):
        print(f"Warning: Invalid layer_index {layer_index} for key '{sentence_key}'. "
              f"Stack has {num_layers_in_stack} layers. Shape: {feature_stack.shape}")
        return None

    single_layer_features = feature_stack[layer_index, :, :]
    # Expected shape: (num_tokens, elmo_embedding_dim)
    return single_layer_features

if __name__ == '__main__':
    # Example Usage (requires a dummy HDF5 file to be created for testing)
    # This is better tested with pytest and a fixture.
    # To run this standalone, you'd need to:
    # 1. Create 'dummy_elmo.hdf5'
    #    with h5py.File('dummy_elmo.hdf5', 'w') as hf:
    #        hf.create_dataset('0', data=np.random.rand(3, 10, 1024)) # 3 layers, 10 tokens, 1024 dim
    #        hf.create_dataset('1', data=np.random.rand(3, 5, 1024))  # 3 layers, 5 tokens, 1024 dim
    #
    # with h5py.File('dummy_elmo.hdf5', 'r') as hf_opened:
    #     print("Testing with dummy HDF5:")
    #     emb0_l2 = load_elmo_embeddings_for_sentence(hf_opened, "0", 2)
    #     if emb0_l2 is not None:
    #         print("Sentence 0, Layer 2 shape:", emb0_l2.shape) # Expected (10, 1024)
    #
    #     emb1_l0 = load_elmo_embeddings_for_sentence(hf_opened, "1", 0)
    #     if emb1_l0 is not None:
    #         print("Sentence 1, Layer 0 shape:", emb1_l0.shape) # Expected (5, 1024)
    #
    #     emb_missing = load_elmo_embeddings_for_sentence(hf_opened, "2", 0) # Key "2" doesn't exist
    #     print("Sentence 2 (missing key) result:", emb_missing)
    #
    #     emb_bad_layer = load_elmo_embeddings_for_sentence(hf_opened, "0", 3) # Layer 3 doesn't exist
    #     print("Sentence 0 (bad layer) result:", emb_bad_layer)
    #
    # import os
    # os.remove('dummy_elmo.hdf5')
    pass
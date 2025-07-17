# scripts/compare_embeddings.py
import h5py
import numpy as np
import sys
import json # To load layers_extracted_indices attribute

def compare_hdf5_embeddings(old_file_path, new_file_path, layers_to_compare):
    print(f"Comparing HDF5 Embeddings:\n  Old File: {old_file_path}\n  New File: {new_file_path}\n")
    print(f"Comparing only these absolute layers (from original model): {layers_to_compare}\n")

    with h5py.File(old_file_path, 'r') as f_old, h5py.File(new_file_path, 'r') as f_new:
        # Check if new file has the layers_extracted_indices attribute
        new_file_extracted_layers_str = f_new.attrs.get('layers_extracted_indices')
        if new_file_extracted_layers_str is None:
            print("ERROR: New file does not have 'layers_extracted_indices' attribute. Cannot verify layer mapping.")
            return False
        
        new_file_extracted_layers = json.loads(new_file_extracted_layers_str)
        
        # Check if old file has the layers_extracted_indices attribute (it shouldn't, for this test)
        old_file_extracted_layers_str = f_old.attrs.get('layers_extracted_indices')
        if old_file_extracted_layers_str is not None:
            print("WARNING: Old file has 'layers_extracted_indices' attribute. Ensure it's a 'layers-all' type file for this comparison.")
            # For this specific comparison, we assume the old file is ALL layers (13 for BERT-base)
            # If it's not, the comparison below might pick the wrong layers.
        
        keys_old = sorted([k for k in f_old.keys() if k.isdigit()], key=int)
        keys_new = sorted([k for k in f_new.keys() if k.isdigit()], key=int)

        if len(keys_old) != len(keys_new):
            print(f"ERROR: Number of sentences (keys) differ! Old: {len(keys_old)}, New: {len(keys_new)}")
            return False

        all_layers_match = True
        num_sentences_compared = 0

        for key in keys_old: # Iterate over sentence keys
            if key not in f_new:
                print(f"ERROR: Sentence key '{key}' from old file not found in new file.")
                all_layers_match = False
                break
            
            old_sentence_stack = f_old[key][()] # (total_layers, num_tokens, dim)
            new_sentence_stack = f_new[key][()] # (extracted_layers_count, num_tokens, dim)

            # Ensure number of tokens and embedding dimension match for this sentence
            if old_sentence_stack.shape[1:] != new_sentence_stack.shape[1:] :
                print(f"ERROR: Token or dimension mismatch for sentence '{key}': Old {old_sentence_stack.shape}, New {new_sentence_stack.shape}")
                all_layers_match = False
                break

            # Compare each specified layer
            for abs_layer_idx in layers_to_compare:
                # Get data from old file (assuming it's the full 13-layer stack)
                if not (0 <= abs_layer_idx < old_sentence_stack.shape[0]):
                    print(f"ERROR: Absolute layer {abs_layer_idx} out of bounds for old file (shape: {old_sentence_stack.shape}).")
                    all_layers_match = False
                    break
                old_layer_embeddings = old_sentence_stack[abs_layer_idx, :, :]

                # Get data from new file (must map requested_layer to its index in the extracted stack)
                try:
                    new_stack_idx = new_file_extracted_layers.index(abs_layer_idx)
                except ValueError:
                    print(f"ERROR: Absolute layer {abs_layer_idx} not found in new file's extracted layers ({new_file_extracted_layers}).")
                    all_layers_match = False
                    break
                
                new_layer_embeddings = new_sentence_stack[new_stack_idx, :, :]
                
                if not np.allclose(old_layer_embeddings, new_layer_embeddings, atol=1e-6):
                    print(f"MISMATCH: Data for sentence '{key}', absolute layer {abs_layer_idx} differs!")
                    all_layers_match = False
                    # For a detailed comparison, you might want to uncomment this:
                    # diff = np.abs(old_layer_embeddings - new_layer_embeddings)
                    # print(f"  Max absolute difference: {np.max(diff)}")
                    # print(f"  Mean absolute difference: {np.mean(diff)}")
                    break # Stop at first mismatch for this sentence
            
            if not all_layers_match:
                break # Stop all comparisons if a mismatch is found
            
            num_sentences_compared += 1
            if num_sentences_compared % 1000 == 0:
                print(f"  ... Compared {num_sentences_compared} sentences so far, all layers match.")


        if all_layers_match:
            print(f"\nSUCCESS: All specified layers across all {num_sentences_compared} sentences are identical.")
            return True
        else:
            print("\nFAILURE: Mismatches found. Check detailed logs above.")
            return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: poetry run python scripts/compare_embeddings.py <path_to_old_file> <path_to_new_file>")
        sys.exit(1)
    
    # These are the layers we specified for extraction in the new script
    # and the layers we want to compare from the old 'all-layers' file.
    # Make sure this list matches what you put in job.layers_to_extract.
    LAYERS_TO_COMPARE = [0, 7, 12] 
    
    success = compare_hdf5_embeddings(sys.argv[1], sys.argv[2], LAYERS_TO_COMPARE)
    sys.exit(0 if success else 1)
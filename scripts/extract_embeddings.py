# scripts/extract_embeddings.py
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
import logging
import sys
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import json

import h5py
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer

# --- Add src to path for direct execution ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
# --- End Path Addition ---

from torch_probe.utils.conllu_reader import read_conll_file, SentenceData # Use our CoNLL reader

log = logging.getLogger(__name__)

# --- Helper Functions for Alignment ---

def get_word_token_alignment(
    original_words: List[str], 
    subword_tokens: List[str], 
    tokenizer: PreTrainedTokenizer
) -> List[List[int]]:
    """
    Aligns subword tokens to original word tokens.
    Returns a list where each inner list contains the indices of subword tokens
    corresponding to an original word.
    
    This is a basic alignment based on stripping special prefixes (like '##' for BERT, ' ' for RoBERTa/GPT)
    and re-joining subwords until they match an original word.
    More robust methods might use tokenizer.convert_ids_to_tokens and tokenizer.get_offset_mapping.
    This simplified version is for illustration and might need refinement based on tokenizer.
    """
    alignment: List[List[int]] = []
    current_word_subtokens: List[int] = []
    current_reconstructed_word = ""
    original_word_idx = 0

    # Common subword prefixes/markers
    # BERT uses '##', SentencePiece (XLNet, ALBERT, Llama, etc.) uses ' ' (space prefix for new words)
    # RoBERTa/GPT2 use 'Ġ' (space prefix)
    
    # This is a heuristic and might need adjustment based on the specific tokenizer.
    # Using tokenizer.backend_tokenizer.normalizer and pre_tokenizer might be more robust
    # or tokenizer.convert_ids_to_tokens with attention to special characters.
    
    # A more robust approach often involves using tokenizer's built-in word_ids() method if available
    # with batch_encode_plus, or by manually tracking offsets.
    # For now, let's assume a simpler re-concatenation approach for this draft.
    # THIS SECTION WILL LIKELY NEED REFINEMENT AND TESTING PER TOKENIZER.

    subword_idx = 0
    while original_word_idx < len(original_words) and subword_idx < len(subword_tokens):
        subtoken = subword_tokens[subword_idx]
        current_word_subtokens.append(subword_idx)

        # Attempt to reconstruct the word
        # This logic is highly dependent on the tokenizer type
        # BERT/WordPiece: subwords often start with '##'
        if subtoken.startswith("##"):
            current_reconstructed_word += subtoken[2:]
        # RoBERTa/GPT2/Llama with SentencePiece: subwords starting a new word might have 'Ġ' or similar
        elif current_reconstructed_word and not subtoken.startswith(("Ġ", " ", "##")) : # Continuation of a word for some tokenizers
            current_reconstructed_word += subtoken
        else: # Start of a new word (or first subtoken)
            if current_reconstructed_word: # We finished a previous word
                 # This check is too simple and will fail for many tokenizers.
                 # We should actually check if current_reconstructed_word matches original_words[original_word_idx-1]
                 # This whole alignment part is placeholder for a robust solution.
                 pass # Placeholder
            current_reconstructed_word = subtoken.lstrip("Ġ").lstrip(" ")


        # Simplified check: if reconstructed matches original, or if it's the last subword for the current original word
        # THIS IS A VERY ROUGH HEURISTIC AND NEEDS A PROPER IMPLEMENTATION
        # A better way is to use tokenizer.encode_plus with return_offsets_mapping=True
        # and then map subword indices to word indices.

        if current_reconstructed_word.lower() == original_words[original_word_idx].lower().replace(" ", ""): # Very basic check
            alignment.append(list(current_word_subtokens))
            current_word_subtokens = []
            current_reconstructed_word = ""
            original_word_idx += 1
        elif subword_idx == len(subword_tokens) - 1 and original_word_idx < len(original_words) : # Last subtoken
            if current_word_subtokens: # Assign remaining subtokens to the current word
                alignment.append(list(current_word_subtokens))
                current_word_subtokens = [] # Reset
            original_word_idx +=1 # Move to next word

        subword_idx += 1
        
    # If loop finishes but alignment isn't complete (lengths mismatch)
    # this indicates a problem with the alignment logic for this tokenizer.
    if len(alignment) != len(original_words):
        # log.warning(f"Alignment mismatch: {len(alignment)} aligned words vs {len(original_words)} original words. "
        #             f"Sentence: {' '.join(original_words)}. Subtokens: {' '.join(subword_tokens)}. "
        #             "Falling back to one-to-one or erroring might be needed.")
        # Fallback: create a dummy one-to-one if lengths are same, or error.
        # This often happens if the heuristic reconstruction fails.
        # For now, we'll let it proceed and it might cause issues downstream if not handled.
        # A robust solution using word_ids() from tokenizer output is better.
        pass

    return alignment


def align_subword_embeddings(
    subword_embeddings: torch.Tensor, # (num_subwords, hidden_dim)
    alignment: List[List[int]],       # List of lists of subword indices for each word
    strategy: str = "mean"
) -> torch.Tensor:
    """
    Aligns subword embeddings to word embeddings based on the alignment map.
    """
    if not alignment: # No words or alignment failed
        return torch.empty((0, subword_embeddings.size(1)), device=subword_embeddings.device)

    word_embeddings_list: List[torch.Tensor] = []
    for word_subtoken_indices in alignment:
        if not word_subtoken_indices: # Should not happen with good alignment
            # Add a zero vector or handle error
            log.debug(f"Empty subtoken list for a word in alignment. Subword_embeddings shape: {subword_embeddings.shape}")
            # If subword_embeddings could be empty (e.g. for empty input string)
            if subword_embeddings.numel() == 0:
                 word_embeddings_list.append(torch.empty((0,0), device=subword_embeddings.device)) # Special case for completely empty
                 continue

            word_embeddings_list.append(torch.zeros(subword_embeddings.size(1), device=subword_embeddings.device))
            continue

        # Ensure indices are within bounds
        valid_indices = [idx for idx in word_subtoken_indices if 0 <= idx < subword_embeddings.size(0)]
        if not valid_indices:
            log.warning(f"Invalid subtoken indices for a word: {word_subtoken_indices} "
                        f"when subword_embeddings shape is {subword_embeddings.shape}. Appending zero vector.")
            word_embeddings_list.append(torch.zeros(subword_embeddings.size(1), device=subword_embeddings.device))
            continue

        current_word_subword_embeddings = subword_embeddings[valid_indices]

        if strategy == "mean":
            word_emb = torch.mean(current_word_subword_embeddings, dim=0)
        elif strategy == "first":
            word_emb = current_word_subword_embeddings[0]
        # Add other strategies like "last" or "sum" if needed
        else:
            raise ValueError(f"Unknown alignment strategy: {strategy}")
        word_embeddings_list.append(word_emb)
    
    if not word_embeddings_list: # If all words somehow resulted in no embeddings
         return torch.empty((0, subword_embeddings.size(1)), device=subword_embeddings.device)

    return torch.stack(word_embeddings_list)


def get_word_ids_from_tokenizer(
    original_words: List[str],
    tokenized_output: Any, # Output from tokenizer (e.g. BatchEncoding)
    tokenizer: PreTrainedTokenizer
) -> List[Optional[int]]:
    """
    Uses tokenizer's built-in word_ids() method or offset_mapping to get word assignments for subword tokens.
    This is a more robust way to get alignments.
    """
    word_ids = tokenized_output.word_ids()

    # The word_ids list will have one entry per subword token.
    # It maps each subword token to the index of the word it belongs to in the original sentence.
    # Special tokens (CLS, SEP) will have word_id None.
    # We need to adjust for this if CLS/SEP were added *around* the sentence tokens.
    
    # Example: original_words = ["Hello", "world"]
    # Tokenizer might give: [CLS] Hel ##lo world [SEP]
    # word_ids might be:  [None,  0,   0,  1,    None]

    # We only care about subword tokens that correspond to actual words, not CLS/SEP for alignment purposes here.
    # This function returns the raw word_ids list, filtering of None can be done by caller if needed.
    return word_ids


def robust_align_subword_embeddings(
    subword_embeddings: torch.Tensor, # (num_subword_tokens_with_special, hidden_dim)
    word_ids: List[Optional[int]],    # Output from tokenized_output.word_ids()
    num_original_words: int,
    alignment_strategy: str = "mean"
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
    if subword_embeddings.numel() == 0: # Handle empty input string resulting in no embeddings
        return torch.empty((0,0), device=subword_embeddings.device)
        
    word_to_subword_indices: Dict[int, List[int]] = defaultdict(list)
    for subword_idx, word_idx in enumerate(word_ids):
        if word_idx is not None: # Skip special tokens like [CLS], [SEP]
            # Ensure subword_idx is valid for subword_embeddings tensor
            if 0 <= subword_idx < subword_embeddings.shape[0]:
                 word_to_subword_indices[word_idx].append(subword_idx)
            else:
                log.warning(f"Subword index {subword_idx} from word_ids is out of bounds "
                            f"for subword_embeddings shape {subword_embeddings.shape[0]}. Skipping.")


    aligned_embeddings_list: List[torch.Tensor] = []
    for i in range(num_original_words):
        subword_indices_for_word = word_to_subword_indices.get(i)
        if subword_indices_for_word:
            # Ensure indices are valid again before indexing (defensive)
            valid_sub_indices = [idx for idx in subword_indices_for_word if 0 <= idx < subword_embeddings.shape[0]]
            if not valid_sub_indices:
                log.warning(f"Word {i} had no valid subword indices after bounds check. Appending zero vector.")
                aligned_embeddings_list.append(torch.zeros(subword_embeddings.size(1), device=subword_embeddings.device))
                continue

            word_sub_embeddings = subword_embeddings[valid_sub_indices]
            if alignment_strategy == "mean":
                aligned_embeddings_list.append(torch.mean(word_sub_embeddings, dim=0))
            elif alignment_strategy == "first":
                aligned_embeddings_list.append(word_sub_embeddings[0])
            # Add other strategies if needed
            else:
                raise ValueError(f"Unknown alignment strategy: {alignment_strategy}")
        else:
            # This word was not represented by any subword (e.g., it was empty or tokenized away entirely)
            # Or it was a special token that got filtered.
            # For robustness, append a zero vector.
            log.warning(f"Word at index {i} (0-indexed) has no corresponding subwords in word_ids. "
                        f"Appending zero vector. Original num_words: {num_original_words}. "
                        f"Word_ids length: {len(word_ids)}. Subword_embeddings shape: {subword_embeddings.shape}")
            aligned_embeddings_list.append(torch.zeros(subword_embeddings.size(1), device=subword_embeddings.device))

    if not aligned_embeddings_list: # All words resulted in no embeddings
        return torch.empty((0, subword_embeddings.size(1)), device=subword_embeddings.device)
        
    return torch.stack(aligned_embeddings_list)


# --- Main Extraction Logic ---

@hydra.main(config_path="../configs", config_name="config_extract", version_base="1.3") # Use a dedicated config
def extract_embeddings_main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    # Output dir for this script's run will be managed by Hydra as usual
    output_dir = Path(hydra_cfg.runtime.output_dir) # This is where HDF5 will be saved.
    original_cwd = Path(hydra_cfg.runtime.cwd)
    log.info(f"Hydra output directory for this extraction run: {output_dir}")
    log.info(f"Original CWD: {original_cwd}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device Setup ---
    if cfg.runtime.device == "auto":
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps"); log.info("MPS device selected.")
        else: device = torch.device("cpu"); log.info("MPS/CUDA not available. Using CPU.")
    else: device = torch.device(cfg.runtime.device)
    log.info(f"Using device: {device}")

    # --- Load Tokenizer and Model ---
    log.info(f"Loading tokenizer and model: {cfg.model.hf_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_name)
        model = AutoModel.from_pretrained(cfg.model.hf_model_name, output_hidden_states=True)
        model.to(device)
        model.eval()
    except Exception as e:
        log.error(f"Failed to load model or tokenizer '{cfg.model.hf_model_name}': {e}", exc_info=True)
        raise

    # Determine which layers to extract
    # Hugging Face models typically output initial embeddings + all hidden layer outputs
    # For a model with N layers, hidden_states tuple has N+1 elements.
    # Layer 0 is initial embeddings, Layer 1 is output of 1st layer, ..., Layer N is output of Nth layer.
    num_model_layers = model.config.num_hidden_layers + 1 # embeddings + N layers
    
    layers_to_extract_config = cfg.model.get("layers_to_extract", "all")
    if isinstance(layers_to_extract_config, str) and layers_to_extract_config.lower() == "all":
        layers_to_extract_indices = list(range(num_model_layers))
        log.info(f"Extracting all {num_model_layers} layers (0 to {num_model_layers-1}).")
    elif isinstance(layers_to_extract_config, (list, ListConfig)): # ListConfig from OmegaConf
        layers_to_extract_indices = [int(l) for l in layers_to_extract_config]
        # Validate layer indices
        if not all(0 <= l < num_model_layers for l in layers_to_extract_indices):
            raise ValueError(f"Invalid layer index in {layers_to_extract_indices}. "
                             f"Model {cfg.model.hf_model_name} has {num_model_layers} (0-{num_model_layers-1}).")
        log.info(f"Extracting specified layers: {layers_to_extract_indices}")
    else:
        raise ValueError(f"Invalid format for layers_to_extract: {layers_to_extract_config}. "
                         "Should be 'all' or a list of integers.")
    
    if not layers_to_extract_indices:
        raise ValueError("No layers specified for extraction.")

    # --- Process each CoNLL input file ---
    for split_name, conll_rel_path in cfg.input_conll_files.items(): # e.g., train: "path/to/train.conllx"
        if conll_rel_path is None:
            log.info(f"Skipping {split_name} as its CoNLL path is null in config.")
            continue

        conll_abs_path = Path(original_cwd) / conll_rel_path # Resolve relative to original CWD
        if not conll_abs_path.exists():
            log.error(f"Input CoNLL file for split '{split_name}' not found: {conll_abs_path}")
            continue
        
        log.info(f"Processing {split_name} split from: {conll_abs_path}")
        parsed_sentences: List[SentenceData] = read_conll_file(str(conll_abs_path))
        log.info(f"Read {len(parsed_sentences)} sentences for {split_name} split.")

        # --- Prepare HDF5 Output File ---
        # Naming convention suggestion: model_name_sanitized-split_name-layers_info.hdf5
        model_name_sanitized = cfg.model.hf_model_name.replace("/", "_")
        layers_str = "all" if len(layers_to_extract_indices) == num_model_layers else "_".join(map(str, layers_to_extract_indices))
        
        # Output path now uses the base_output_path from config, resolved from original_cwd
        base_output_path_for_hdf5 = Path(original_cwd) / cfg.output_hdf5.base_output_path
        base_output_path_for_hdf5.mkdir(parents=True, exist_ok=True) # Ensure base path exists

        hdf5_filename = f"{model_name_sanitized}_{split_name}_layers-{layers_str}_align-{cfg.model.alignment_strategy}.hdf5"
        hdf5_output_path = base_output_path_for_hdf5 / hdf5_filename
        log.info(f"Output HDF5 file for {split_name}: {hdf5_output_path}")

        with h5py.File(hdf5_output_path, 'w') as hdf5_file:
            hdf5_file.attrs['model_name'] = cfg.model.hf_model_name
            hdf5_file.attrs['alignment_strategy'] = cfg.model.alignment_strategy
            hdf5_file.attrs['layers_extracted_indices'] = json.dumps(layers_to_extract_indices) # Store as JSON string
            hdf5_file.attrs['original_conll_file'] = str(conll_abs_path)
            
            total_sentences_processed = 0
            for sent_idx, sentence_data in enumerate(tqdm(parsed_sentences, desc=f"Extracting for {split_name}")):
                original_words: List[str] = sentence_data['tokens']
                if not original_words: # Skip empty sentences
                    log.warning(f"Skipping empty sentence at index {sent_idx} in {split_name}.")
                    continue

                # Tokenize using Hugging Face tokenizer
                # Using is_split_into_words=True because PTB is already word-tokenized.
                # We want the tokenizer to then perform subword tokenization on these words.
                tokenized_output = tokenizer(
                    original_words, 
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=False, # No padding needed for single sentences
                    truncation=True, # Truncate if sentence > model max length
                    max_length=tokenizer.model_max_length
                )
                input_ids = tokenized_output["input_ids"].to(device)
                attention_mask = tokenized_output["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    hidden_states_tuple: Tuple[torch.Tensor, ...] = outputs.hidden_states
                
                # hidden_states_tuple includes initial embeddings + output of each layer
                # (batch_size=1, num_subwords, hidden_dim)
                
                # Select, stack, and process specified layers
                selected_layer_embeddings_for_sent: List[torch.Tensor] = []
                for layer_idx in layers_to_extract_indices:
                    if 0 <= layer_idx < len(hidden_states_tuple):
                        # Squeeze batch dim, move to CPU for numpy conversion & storage
                        subword_embeddings_one_layer = hidden_states_tuple[layer_idx].squeeze(0).cpu() 
                        selected_layer_embeddings_for_sent.append(subword_embeddings_one_layer)
                    else:
                        log.error(f"Layer index {layer_idx} out of range for model {cfg.model.hf_model_name}. "
                                  f"Available layers: 0 to {len(hidden_states_tuple)-1}. Skipping this layer for sentence {sent_idx}.")
                
                if not selected_layer_embeddings_for_sent:
                    log.warning(f"No valid layers extracted for sentence {sent_idx}. Skipping sentence.")
                    continue
                
                # Stack layers: (num_extracted_layers, num_subwords, hidden_dim)
                stacked_subword_embeddings = torch.stack(selected_layer_embeddings_for_sent, dim=0)

                # Get word_ids for robust alignment
                # word_ids() returns a list, one entry per subword token from input_ids[0]
                # It maps subword to original word index, or None for special tokens.
                word_ids_for_sent = tokenized_output.word_ids(batch_index=0) 

                # Align each layer
                aligned_word_embeddings_all_layers_list: List[torch.Tensor] = []
                for layer_idx_in_stack in range(stacked_subword_embeddings.shape[0]):
                    current_layer_subword_embs = stacked_subword_embeddings[layer_idx_in_stack, :, :]
                    
                    aligned_layer_word_embs = robust_align_subword_embeddings(
                        current_layer_subword_embs, # (num_subwords_incl_special, hidden_dim)
                        word_ids_for_sent,          # Includes None for special tokens
                        len(original_words),
                        alignment_strategy=cfg.model.alignment_strategy
                    )
                    
                    # Sanity check alignment length
                    if aligned_layer_word_embs.shape[0] != len(original_words):
                        log.error(f"CRITICAL ALIGNMENT ERROR for sentence {sent_idx} (split {split_name}), layer index {layers_to_extract_indices[layer_idx_in_stack]}: "
                                  f"Aligned word count ({aligned_layer_word_embs.shape[0]}) "
                                  f"does not match original word count ({len(original_words)}). "
                                  f"Original words: {' '.join(original_words)}. "
                                  f"Subword tokens: {tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())}. "
                                  f"Word IDs: {word_ids_for_sent}. "
                                  "Skipping sentence in HDF5, this indicates a bug in alignment or tokenization understanding.")
                        # To prevent saving mismatched data, we skip this sentence entirely for this run
                        # Or one could fill with NaNs, but skipping is safer.
                        aligned_word_embeddings_all_layers_list = [] # Clear list to break outer loop
                        break 
                    
                    aligned_word_embeddings_all_layers_list.append(aligned_layer_word_embs)

                if not aligned_word_embeddings_all_layers_list: # If alignment failed for any layer
                    continue # Skip to next sentence

                # Stack to: (num_extracted_layers, num_original_words, hidden_dim)
                final_word_embeddings_for_sentence = torch.stack(aligned_word_embeddings_all_layers_list, dim=0)
                # Save to HDF5
                # This structure matches what load_elmo_embeddings_for_sentence expects if 
                # embedding_layer_index for ProbeDataset refers to the index in this saved stack.
                hdf5_file.create_dataset(str(sent_idx), data=final_word_embeddings_for_sentence.numpy())
                total_sentences_processed +=1

            log.info(f"Finished processing {split_name}. Saved {total_sentences_processed}/{len(parsed_sentences)} sentences to {hdf5_output_path}.")

    log.info("All specified CoNLL files processed.")


if __name__ == "__main__":
    # Setup basic logging for standalone script execution
    logging.basicConfig(
        stream=sys.stdout, 
        level=logging.INFO, 
        format='%(asctime)s [%(name)s:%(levelname)s] %(message)s'
    )
    # Call the Hydra main function
    extract_embeddings_main()
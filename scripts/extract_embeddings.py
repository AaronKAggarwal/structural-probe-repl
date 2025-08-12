# scripts/extract_embeddings.py
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from torch_probe.utils.conllu_reader import SentenceData, read_conll_file
from torch_probe.utils.alignment import robust_align_subword_embeddings

log = logging.getLogger(__name__)


# --- Main Extraction Logic ---

@hydra.main(config_path="../configs", config_name="config_extract", version_base="1.3")
def extract_embeddings_main(cfg: DictConfig) -> None:
    """
    Extracts hidden state embeddings from a Hugging Face model for a given dataset.
    This script is driven by the main configuration system and is typically invoked
    by selecting an experiment that sets `job=extract_embeddings`.
    """
    # --- Configuration and Path Setup ---
    hydra_cfg = HydraConfig.get()
    original_cwd = Path(hydra_cfg.runtime.cwd)

    # Check if the correct job is running
    if cfg.job.name != "extract_embeddings":
        log.error("This script is for embedding extraction. Please run with 'job=extract_embeddings'.")
        sys.exit(1)
        
    log.info("--- Starting Embedding Extraction Job ---")
    log.info(f"Hydra output directory for this run: {hydra_cfg.runtime.output_dir}")

    # --- Device Setup ---
    if cfg.runtime.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.runtime.device)
    log.info(f"Using device: {device}")

    # --- Load Tokenizer and Model ---
    model_name = cfg.model.hf_model_name
    log.info(f"Loading tokenizer and model: {model_name}")
    try:
        tokenizer_kwargs = {}
        # GPT-2 style tokenizers require add_prefix_space=True for pre-tokenized input
        if "gpt2" in model_name.lower():
            tokenizer_kwargs["add_prefix_space"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        model.to(device)
        model.eval()
    except Exception as e:
        log.error(f"Failed to load model or tokenizer '{model_name}': {e}", exc_info=True)
        raise

    # --- Determine Layers to Extract ---
    num_model_layers = model.config.num_hidden_layers + 1  # +1 for initial embeddings
    layers_to_extract_config = cfg.job.get("layers_to_extract", "all")

    if isinstance(layers_to_extract_config, str) and layers_to_extract_config.lower() == "all":
        layers_to_extract_indices = list(range(num_model_layers))
        log.info(f"Extracting all {num_model_layers} layers (0 to {num_model_layers - 1}).")
    elif isinstance(layers_to_extract_config, (list, ListConfig)):
        layers_to_extract_indices = [int(l) for l in layers_to_extract_config]
        if not all(0 <= l < num_model_layers for l in layers_to_extract_indices):
            raise ValueError(
                f"Invalid layer index in {layers_to_extract_indices}. "
                f"Model '{model_name}' has {num_model_layers} layers (0-{num_model_layers - 1})."
            )
        log.info(f"Extracting specified layers: {layers_to_extract_indices}")
    else:
        raise ValueError(f"Invalid format for job.layers_to_extract: {layers_to_extract_config}")

    if not layers_to_extract_indices:
        raise ValueError("No layers specified for extraction.")

    # --- Process each CoNLL input file ---
    for split_name, conll_rel_path in cfg.dataset.paths.items():
        if conll_rel_path is None:
            log.info(f"Skipping '{split_name}' split as its path is null in config.")
            continue

        conll_abs_path = Path(original_cwd) / conll_rel_path
        if not conll_abs_path.exists():
            log.error(f"Input CoNLL file for split '{split_name}' not found: {conll_abs_path}")
            continue

        log.info(f"Processing '{split_name}' split from: {conll_abs_path}")
        parsed_sentences: List[SentenceData] = read_conll_file(str(conll_abs_path))
        log.info(f"Read {len(parsed_sentences)} sentences for '{split_name}' split.")

        # --- Prepare HDF5 Output File ---
        layers_str = "all" if len(layers_to_extract_indices) == num_model_layers else "_".join(map(str, layers_to_extract_indices))
        alignment_strategy = cfg.embeddings.get("alignment_strategy", "mean") # Use a default
        
        # Output path is now determined by the 'job' config
        base_output_path = Path(original_cwd) / cfg.job.output_hdf5_path
        base_output_path.mkdir(parents=True, exist_ok=True)
        
        hdf5_filename = f"{cfg.dataset.name}_{split_name}_layers-{layers_str}_align-{alignment_strategy}.hdf5"
        hdf5_output_path = base_output_path / hdf5_filename
        log.info(f"Output HDF5 file for '{split_name}': {hdf5_output_path}")

        with h5py.File(hdf5_output_path, "w") as hdf5_file:
            hdf5_file.attrs["model_name"] = model_name
            hdf5_file.attrs["alignment_strategy"] = alignment_strategy
            hdf5_file.attrs["layers_extracted_indices"] = json.dumps(layers_to_extract_indices)
            hdf5_file.attrs["original_conll_file"] = str(conll_abs_path)

            # Optional, backward-compatible tokenization metadata containers
            save_tok_map = bool(cfg.job.get("save_tokenization_map", False))
            save_input_ids = bool(cfg.job.get("save_input_ids", False))
            word_ids_group = hdf5_file.create_group("word_ids") if save_tok_map else None
            input_ids_group = hdf5_file.create_group("input_ids") if save_input_ids else None

            total_sentences_processed = 0
            for sent_idx, sentence_data in enumerate(tqdm(parsed_sentences, desc=f"Extracting for {split_name}")):
                original_words: List[str] = sentence_data["tokens"]
                if not original_words:
                    continue

                tokenized_output = tokenizer(original_words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
                input_ids = tokenized_output["input_ids"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids)
                    hidden_states_tuple: Tuple[torch.Tensor, ...] = outputs.hidden_states

                selected_layer_embeddings = [hidden_states_tuple[i] for i in layers_to_extract_indices]
                stacked_subword_embeddings = torch.stack(selected_layer_embeddings, dim=0).squeeze(1).cpu()

                word_ids_for_sent = tokenized_output.word_ids(batch_index=0)
                
                final_word_embeddings_for_sentence = robust_align_subword_embeddings(
                    stacked_subword_embeddings.permute(1, 0, 2), # (num_subwords, num_layers, dim)
                    word_ids_for_sent,
                    len(original_words),
                    alignment_strategy=alignment_strategy,
                ).permute(1, 0, 2) # Back to (num_layers, num_words, dim)

                if final_word_embeddings_for_sentence.shape[1] != len(original_words):
                    log.error(f"CRITICAL ALIGNMENT ERROR for sentence {sent_idx}...")
                    continue
                
                hdf5_file.create_dataset(str(sent_idx), data=final_word_embeddings_for_sentence.numpy())

                # Persist tokenization artifacts if requested
                if save_tok_map and word_ids_group is not None:
                    # map None -> -1 for specials
                    import numpy as np
                    word_ids_np = np.array(
                        [wi if wi is not None else -1 for wi in word_ids_for_sent],
                        dtype=np.int32,
                    )
                    word_ids_group.create_dataset(str(sent_idx), data=word_ids_np)
                if save_input_ids and input_ids_group is not None:
                    input_ids_np = tokenized_output["input_ids"].squeeze(0).cpu().numpy()
                    input_ids_group.create_dataset(str(sent_idx), data=input_ids_np)
                total_sentences_processed += 1

            log.info(f"Finished processing '{split_name}'. Saved {total_sentences_processed}/{len(parsed_sentences)} sentences to {hdf5_output_path}.")

    log.info("All specified CoNLL files processed.")


if __name__ == "__main__":
    # Setup basic logging for standalone script execution
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
    )
    # Call the Hydra main function
    extract_embeddings_main()
#!/usr/bin/env python3
"""
Comprehensive validation script for extracted embeddings.
Checks file structure, cross-references with CoNLL-U, validates tokenization, and inspects embedding quality.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from torch_probe.utils.conllu_reader import read_conll_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def validate_file_structure(hdf5_dir: Path, expected_splits: List[str]) -> Dict[str, bool]:
    """Check that all expected HDF5 files exist."""
    results = {}
    for split in expected_splits:
        pattern = f"*_conllu_{split}_layers-all_align-mean.hdf5"
        files = list(hdf5_dir.glob(pattern))
        if len(files) == 1:
            results[f"{split}_file_exists"] = True
            results[f"{split}_file_path"] = str(files[0])
        else:
            results[f"{split}_file_exists"] = False
            log.error(f"Expected 1 file for {split}, found {len(files)}: {files}")
    return results


def validate_hdf5_structure(hdf5_path: Path, expected_layers: int = 13) -> Dict[str, any]:
    """Validate internal HDF5 structure."""
    results = {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check main datasets (sentence embeddings)
            sentence_keys = [k for k in f.keys() if k.isdigit()]
            results['num_sentences'] = len(sentence_keys)
            results['sentence_indices'] = sorted([int(k) for k in sentence_keys])
            
            if sentence_keys:
                # Check first sentence structure
                first_sent = f[sentence_keys[0]]
                results['embedding_shape'] = first_sent.shape
                results['has_correct_layers'] = first_sent.shape[0] == expected_layers
                results['embedding_dim'] = first_sent.shape[2] if len(first_sent.shape) == 3 else None
                
                # Check for auxiliary data
                results['has_word_ids'] = 'word_ids' in f
                results['has_input_ids'] = 'input_ids' in f
                
                if 'word_ids' in f:
                    word_ids_keys = list(f['word_ids'].keys())
                    results['word_ids_count'] = len(word_ids_keys)
                    results['word_ids_match_sentences'] = set(word_ids_keys) == set(sentence_keys)
                
    except Exception as e:
        results['error'] = str(e)
        log.error(f"Error reading {hdf5_path}: {e}")
    
    return results


def cross_reference_conllu(hdf5_path: Path, conllu_path: Path, max_check: int = 100) -> Dict[str, any]:
    """Cross-reference HDF5 embeddings with CoNLL-U sentences."""
    results = {}
    
    try:
        # Read CoNLL-U
        parsed_sentences = read_conll_file(str(conllu_path))
        results['conllu_sentence_count'] = len(parsed_sentences)
        
        # Check HDF5
        with h5py.File(hdf5_path, 'r') as f:
            sentence_keys = sorted([int(k) for k in f.keys() if k.isdigit()])
            results['hdf5_sentence_count'] = len(sentence_keys)
            
            mismatches = []
            skipped_indices = []
            
            # Check subset of sentences
            check_indices = sentence_keys[:max_check]
            for sent_idx in check_indices:
                if sent_idx >= len(parsed_sentences):
                    mismatches.append(f"HDF5 index {sent_idx} exceeds CoNLL-U length {len(parsed_sentences)}")
                    continue
                
                conllu_tokens = parsed_sentences[sent_idx]['tokens']
                hdf5_embeddings = f[str(sent_idx)]
                
                # Check token count match
                hdf5_num_words = hdf5_embeddings.shape[1]
                conllu_num_words = len(conllu_tokens)
                
                if hdf5_num_words != conllu_num_words:
                    mismatches.append(f"Sentence {sent_idx}: HDF5 has {hdf5_num_words} embeddings, CoNLL-U has {conllu_num_words} tokens")
            
            # Check for gaps in sentence indices (indicating skipped sentences)
            if sentence_keys:
                max_idx = max(sentence_keys)
                expected_indices = set(range(max_idx + 1))
                actual_indices = set(sentence_keys)
                skipped_indices = sorted(expected_indices - actual_indices)
            
            results['token_count_mismatches'] = mismatches
            results['skipped_sentence_indices'] = skipped_indices
            results['num_skipped_sentences'] = len(skipped_indices)
            
    except Exception as e:
        results['error'] = str(e)
        log.error(f"Error cross-referencing {hdf5_path} with {conllu_path}: {e}")
    
    return results


def validate_tokenization(hdf5_path: Path, conllu_path: Path, tokenizer, max_check: int = 50) -> Dict[str, any]:
    """Validate tokenization artifacts and check for truncation."""
    results = {}
    
    try:
        parsed_sentences = read_conll_file(str(conllu_path))
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'word_ids' not in f:
                results['error'] = "No word_ids found in HDF5"
                return results
            
            sentence_keys = sorted([int(k) for k in f.keys() if k.isdigit()])
            check_indices = sentence_keys[:max_check]
            
            truncation_issues = []
            tokenization_stats = []
            
            for sent_idx in check_indices:
                if sent_idx >= len(parsed_sentences):
                    continue
                
                conllu_tokens = parsed_sentences[sent_idx]['tokens']
                word_ids = f['word_ids'][str(sent_idx)][:]
                
                # Check if any words are missing from tokenization
                word_ids_clean = [w for w in word_ids if w >= 0]  # Remove special tokens (-1)
                max_word_id = max(word_ids_clean) if word_ids_clean else -1
                
                if max_word_id + 1 < len(conllu_tokens):
                    truncation_issues.append(f"Sentence {sent_idx}: tokenization covers {max_word_id + 1} words, but CoNLL-U has {len(conllu_tokens)}")
                
                # Calculate fragmentation ratio
                num_subwords = len([w for w in word_ids if w >= 0])
                num_words = len(conllu_tokens)
                fragmentation_ratio = num_subwords / num_words if num_words > 0 else 0
                tokenization_stats.append(fragmentation_ratio)
            
            results['truncation_issues'] = truncation_issues
            results['avg_fragmentation_ratio'] = np.mean(tokenization_stats) if tokenization_stats else 0
            results['fragmentation_ratios'] = tokenization_stats[:10]  # Sample
            
    except Exception as e:
        results['error'] = str(e)
        log.error(f"Error validating tokenization for {hdf5_path}: {e}")
    
    return results


def validate_embedding_quality(hdf5_path: Path, max_check: int = 20) -> Dict[str, any]:
    """Check embedding values for quality issues."""
    results = {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            sentence_keys = sorted([int(k) for k in f.keys() if k.isdigit()])
            check_indices = sentence_keys[:max_check]
            
            all_values = []
            layer_stats = {i: [] for i in range(13)}
            
            for sent_idx in check_indices:
                embeddings = f[str(sent_idx)][:]  # Shape: [layers, words, dim]
                
                # Check for NaN/inf
                if np.any(np.isnan(embeddings)):
                    results['has_nan'] = True
                if np.any(np.isinf(embeddings)):
                    results['has_inf'] = True
                
                # Collect statistics
                all_values.extend(embeddings.flatten())
                
                for layer in range(embeddings.shape[0]):
                    layer_data = embeddings[layer]  # [words, dim]
                    layer_stats[layer].extend(layer_data.flatten())
            
            # Overall statistics
            all_values = np.array(all_values)
            results['value_range'] = [float(np.min(all_values)), float(np.max(all_values))]
            results['value_mean'] = float(np.mean(all_values))
            results['value_std'] = float(np.std(all_values))
            results['has_zero_variance'] = np.std(all_values) < 1e-8
            
            # Layer-wise differences
            layer_means = [np.mean(layer_stats[i]) for i in range(13)]
            results['layer_means'] = [float(x) for x in layer_means]
            results['layer_differences'] = abs(layer_means[0] - layer_means[12]) > 0.01
            
            # Check if all values are zeros
            results['all_zeros'] = np.all(all_values == 0)
            
    except Exception as e:
        results['error'] = str(e)
        log.error(f"Error validating embedding quality for {hdf5_path}: {e}")
    
    return results


def validate_treebank(treebank_slug: str, base_dir: Path, tokenizer_path: str) -> Dict[str, any]:
    """Comprehensive validation of a single treebank's embeddings."""
    log.info(f"Validating treebank: {treebank_slug}")
    
    results = {'treebank_slug': treebank_slug}
    
    # Paths
    hdf5_dir = base_dir / "data_staging/embeddings" / treebank_slug / "bert-base-multilingual-cased"
    conllu_dir = base_dir / "data/ud" / treebank_slug
    
    results['hdf5_dir'] = str(hdf5_dir)
    results['conllu_dir'] = str(conllu_dir)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        results['tokenizer_max_length'] = tokenizer.model_max_length
    except Exception as e:
        results['tokenizer_error'] = str(e)
        return results
    
    # 1. File structure validation
    file_results = validate_file_structure(hdf5_dir, ['train', 'dev', 'test'])
    results.update(file_results)
    
    # 2. Detailed validation for each split
    splits = ['train', 'dev', 'test']
    for split in splits:
        if not file_results.get(f'{split}_file_exists', False):
            continue
            
        hdf5_path = Path(file_results[f'{split}_file_path'])
        conllu_path = conllu_dir / f"{split}.conllu"
        
        if not conllu_path.exists():
            results[f'{split}_conllu_missing'] = True
            continue
        
        # HDF5 structure
        h5_results = validate_hdf5_structure(hdf5_path)
        for k, v in h5_results.items():
            results[f'{split}_{k}'] = v
        
        # Cross-reference with CoNLL-U
        cross_ref_results = cross_reference_conllu(hdf5_path, conllu_path)
        for k, v in cross_ref_results.items():
            results[f'{split}_crossref_{k}'] = v
        
        # Tokenization validation
        tok_results = validate_tokenization(hdf5_path, conllu_path, tokenizer)
        for k, v in tok_results.items():
            results[f'{split}_tokenization_{k}'] = v
        
        # Embedding quality
        qual_results = validate_embedding_quality(hdf5_path)
        for k, v in qual_results.items():
            results[f'{split}_quality_{k}'] = v
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate extracted embeddings")
    parser.add_argument("--treebank_slug", type=str, required=True, help="Treebank slug to validate")
    parser.add_argument("--base_dir", type=Path, default=".", help="Base project directory")
    parser.add_argument("--tokenizer_path", type=str, default="env/models/bert-base-multilingual-cased", help="Path to tokenizer")
    
    args = parser.parse_args()
    
    # Make tokenizer path absolute if relative
    if not Path(args.tokenizer_path).is_absolute():
        args.tokenizer_path = str(args.base_dir / args.tokenizer_path)
    
    results = validate_treebank(args.treebank_slug, args.base_dir, args.tokenizer_path)
    
    # Print summary
    print(f"\n=== VALIDATION SUMMARY: {args.treebank_slug} ===")
    
    # File existence
    for split in ['train', 'dev', 'test']:
        exists = results.get(f'{split}_file_exists', False)
        print(f"{split.upper()} file exists: {exists}")
        
        if exists:
            num_sent = results.get(f'{split}_num_sentences', 'Unknown')
            conllu_count = results.get(f'{split}_crossref_conllu_sentence_count', 'Unknown')
            skipped = results.get(f'{split}_crossref_num_skipped_sentences', 0)
            print(f"  - HDF5 sentences: {num_sent}")
            print(f"  - CoNLL-U sentences: {conllu_count}")
            print(f"  - Skipped due to length: {skipped}")
            
            # Embedding shape
            shape = results.get(f'{split}_embedding_shape', 'Unknown')
            print(f"  - Embedding shape: {shape}")
            
            # Quality checks
            has_nan = results.get(f'{split}_quality_has_nan', False)
            has_inf = results.get(f'{split}_quality_has_inf', False)
            all_zeros = results.get(f'{split}_quality_all_zeros', False)
            value_range = results.get(f'{split}_quality_value_range', 'Unknown')
            
            print(f"  - Has NaN: {has_nan}")
            print(f"  - Has Inf: {has_inf}")
            print(f"  - All zeros: {all_zeros}")
            print(f"  - Value range: {value_range}")
            
            # Fragmentation
            frag_ratio = results.get(f'{split}_tokenization_avg_fragmentation_ratio', 'Unknown')
            print(f"  - Avg fragmentation ratio: {frag_ratio}")
    
    # Export full results
    import json
    output_file = args.base_dir / f"validation_results_{args.treebank_slug.replace('-', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()

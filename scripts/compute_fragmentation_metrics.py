#!/usr/bin/env python3
"""
Compute tokenization fragmentation metrics from existing HDF5 embeddings.

Uses word_ids from HDF5 files combined with UPOS tags from CoNLL-U files
to compute both content-only and overall fragmentation ratios.

Primary metric: content-only fragmentation (UPOS ∉ {PUNCT,SYM})
Secondary metric: overall fragmentation (all words)

Outputs:
- outputs/analysis/fragmentation_stats/UD_XX/{dev,test}_fragmentation_stats.jsonl
"""

from __future__ import annotations

import json
import h5py
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

REPO_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = REPO_ROOT / "data_staging" / "embeddings"

# Add src to path for imports
sys.path.insert(0, str(REPO_ROOT / "src"))
from torch_probe.utils.conllu_reader import read_conll_file

@dataclass
class FragmentationStats:
    """Fragmentation statistics for a single sentence."""
    sent_id: str
    # Overall metrics (all words)
    num_words_overall: int
    num_subwords_overall: int
    fragmentation_ratio_overall: float
    # Content-only metrics (UPOS ∉ {PUNCT,SYM})
    num_words_content: int
    num_subwords_content: int
    fragmentation_ratio_content: float

def compute_content_mask(upos_tags: List[str]) -> List[bool]:
    """Return mask for content words (UPOS ∉ {PUNCT,SYM})."""
    return [tag not in {"PUNCT", "SYM"} for tag in upos_tags]

def compute_fragmentation_from_word_ids(word_ids: np.ndarray, upos_tags: List[str]) -> Dict[str, float]:
    """Compute both overall and content-only fragmentation metrics."""
    # word_ids: array where -1 = special token, >= 0 = word index
    
    # Overall fragmentation (all words)
    valid_subword_mask = word_ids >= 0
    num_subwords_overall = np.sum(valid_subword_mask)
    
    valid_word_ids = word_ids[valid_subword_mask]
    num_words_overall = len(np.unique(valid_word_ids)) if len(valid_word_ids) > 0 else 0
    
    fragmentation_ratio_overall = num_subwords_overall / num_words_overall if num_words_overall > 0 else 0.0
    
    # Content-only fragmentation (UPOS ∉ {PUNCT,SYM})
    content_mask = compute_content_mask(upos_tags)
    
    # Create mask for subwords that belong to content words
    content_subword_mask = np.zeros_like(word_ids, dtype=bool)
    for i, word_id in enumerate(word_ids):
        if word_id >= 0 and word_id < len(content_mask) and content_mask[word_id]:
            content_subword_mask[i] = True
    
    num_subwords_content = np.sum(content_subword_mask)
    
    # Count unique content words
    content_word_ids = word_ids[content_subword_mask]
    num_words_content = len(np.unique(content_word_ids)) if len(content_word_ids) > 0 else 0
    
    fragmentation_ratio_content = num_subwords_content / num_words_content if num_words_content > 0 else 0.0
    
    return {
        "num_words_overall": num_words_overall,
        "num_subwords_overall": num_subwords_overall,
        "fragmentation_ratio_overall": fragmentation_ratio_overall,
        "num_words_content": num_words_content,
        "num_subwords_content": num_subwords_content,
        "fragmentation_ratio_content": fragmentation_ratio_content
    }

def process_language_split(language_slug: str, split: str) -> List[FragmentationStats]:
    """Process one language/split and extract fragmentation stats."""
    stats = []
    
    # Find HDF5 file
    hdf5_files = find_hdf5_files(language_slug)
    if split not in hdf5_files:
        print(f"Warning: No HDF5 file found for {language_slug}/{split}")
        return stats
    
    hdf5_path = hdf5_files[split]
    
    # Find CoNLL-U file
    conllu_path = REPO_ROOT / "data" / "ud" / language_slug / f"{split}.conllu"
    if not conllu_path.exists():
        print(f"Warning: CoNLL-U file not found: {conllu_path}")
        return stats
    
    print(f"Processing {language_slug}/{split}")
    
    try:
        # Load CoNLL-U data for UPOS tags
        parsed_sentences = read_conll_file(str(conllu_path))
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'word_ids' not in f:
                print(f"  Warning: No word_ids found in {hdf5_path}")
                return stats
            
            # Get sentence keys from word_ids group (more robust)
            sentence_keys = sorted([int(k) for k in f['word_ids'].keys() if k.isdigit()])
            
            for sent_idx in sentence_keys:
                if sent_idx >= len(parsed_sentences):
                    continue
                
                sent_key = str(sent_idx)
                word_ids = f['word_ids'][sent_key][:]
                upos_tags = parsed_sentences[sent_idx]['upos_tags']
                
                # Guard for tag/word count mismatch
                valid = word_ids >= 0
                if valid.any():
                    max_word_id = int(word_ids[valid].max())
                    if max_word_id + 1 > len(upos_tags):
                        print(f"  Warning: Sentence {sent_idx} has {max_word_id + 1} words but only {len(upos_tags)} UPOS tags. Skipping.")
                        continue
                
                frag_metrics = compute_fragmentation_from_word_ids(word_ids, upos_tags)
                
                stat = FragmentationStats(
                    sent_id=f"sent_{sent_idx}",
                    num_words_overall=int(frag_metrics["num_words_overall"]),
                    num_subwords_overall=int(frag_metrics["num_subwords_overall"]),
                    fragmentation_ratio_overall=float(frag_metrics["fragmentation_ratio_overall"]),
                    num_words_content=int(frag_metrics["num_words_content"]),
                    num_subwords_content=int(frag_metrics["num_subwords_content"]),
                    fragmentation_ratio_content=float(frag_metrics["fragmentation_ratio_content"])
                )
                stats.append(stat)
        
        print(f"  Processed {len(stats)} sentences")
        return stats
        
    except Exception as e:
        print(f"Error processing {language_slug}/{split}: {e}")
        return []

def save_fragmentation_stats(stats: List[FragmentationStats], output_path: Path) -> None:
    """Save fragmentation stats to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for stat in stats:
            json.dump({
                "sent_id": stat.sent_id,
                # Overall metrics
                "num_words_overall": stat.num_words_overall,
                "num_subwords_overall": stat.num_subwords_overall,
                "fragmentation_ratio_overall": stat.fragmentation_ratio_overall,
                # Content-only metrics (primary)
                "num_words_content": stat.num_words_content,
                "num_subwords_content": stat.num_subwords_content,
                "fragmentation_ratio_content": stat.fragmentation_ratio_content
            }, f)
            f.write('\n')

def find_hdf5_files(language_slug: str) -> Dict[str, Path]:
    """Find HDF5 files for a language."""
    lang_dir = EMBEDDINGS_DIR / language_slug / "bert-base-multilingual-cased"
    
    files = {}
    if lang_dir.exists():
        for split in ["dev", "test"]:
            # Look for the mean-aligned HDF5 file
            pattern = f"{language_slug}_conllu_{split}_layers-all_align-mean.hdf5"
            hdf5_path = lang_dir / pattern
            if hdf5_path.exists():
                files[split] = hdf5_path
    
    return files

def get_available_languages() -> List[str]:
    """Get list of available languages with embeddings."""
    if not EMBEDDINGS_DIR.exists():
        return []
    
    languages = []
    for lang_dir in EMBEDDINGS_DIR.iterdir():
        if lang_dir.is_dir() and lang_dir.name.startswith("UD_"):
            bert_dir = lang_dir / "bert-base-multilingual-cased"
            if bert_dir.exists():
                # Check if at least one HDF5 file exists
                hdf5_files = list(bert_dir.glob("*.hdf5"))
                if hdf5_files:
                    languages.append(lang_dir.name)
    
    return sorted(languages)

def main():
    """Main entry point."""
    languages = get_available_languages()
    
    if not languages:
        print("No UD languages with embeddings found")
        return
    
    print(f"Found {len(languages)} UD languages with embeddings")
    
    total_processed = 0
    
    # Process each language and split
    for language_slug in languages:
        for split in ["dev", "test"]:
            stats = process_language_split(language_slug, split)
            
            if stats:
                output_path = REPO_ROOT / "outputs" / "analysis" / "fragmentation_stats" / language_slug / f"{split}_fragmentation_stats.jsonl"
                save_fragmentation_stats(stats, output_path)
                print(f"  Saved to {output_path}")
                total_processed += len(stats)
    
    print(f"\nFragmentation computation complete! Processed {total_processed} sentences total.")

if __name__ == "__main__":
    main()

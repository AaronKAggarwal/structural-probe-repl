#!/usr/bin/env python3
"""
Quick tokenization validation for languages with suspected issues.
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from torch_probe.utils.conllu_reader import read_conll_file

def analyze_tokenization(language_slug, sample_size=100):
    """Analyze tokenization patterns for potential issues."""
    
    # Load tokenizer
    model_path = "/Users/aaronaggarwal/structural-probe-main/env/models/bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Load CoNLL-U data
    conllu_path = f"/Users/aaronaggarwal/structural-probe-main/data/ud/{language_slug}/train.conllu"
    if not Path(conllu_path).exists():
        print(f"❌ CoNLL-U not found: {conllu_path}")
        return
    
    parsed_sentences = read_conll_file(conllu_path)
    if not parsed_sentences:
        print(f"❌ No sentences parsed from {conllu_path}")
        return
    
    # Sample sentences for analysis
    sample_sentences = parsed_sentences[:sample_size]
    
    print(f"=== {language_slug} TOKENIZATION ANALYSIS ===")
    print(f"Analyzing {len(sample_sentences)} sentences")
    
    # Track tokenization statistics
    fragmentation_ratios = []
    avg_token_lengths = []
    subword_counts = []
    
    problematic_examples = []
    
    for i, sent_data in enumerate(sample_sentences):
        original_words = sent_data["tokens"]
        if not original_words:
            continue
            
        # Tokenize
        tokenized = tokenizer(original_words, is_split_into_words=True, truncation=False)
        subword_tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
        
        # Remove special tokens for analysis
        content_subwords = [t for t in subword_tokens if t not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]]
        
        # Calculate fragmentation ratio
        fragmentation_ratio = len(content_subwords) / len(original_words) if original_words else 0
        fragmentation_ratios.append(fragmentation_ratio)
        
        # Average token length in characters
        avg_char_len = np.mean([len(word) for word in original_words]) if original_words else 0
        avg_token_lengths.append(avg_char_len)
        
        subword_counts.append(len(content_subwords))
        
        # Flag problematic cases
        if fragmentation_ratio > 2.0 or len(content_subwords) > 100:
            problematic_examples.append({
                'sent_idx': i,
                'original_words': original_words[:10],  # First 10 words
                'subwords': content_subwords[:20],      # First 20 subwords  
                'fragmentation': fragmentation_ratio,
                'word_count': len(original_words),
                'subword_count': len(content_subwords)
            })
    
    # Print statistics
    print(f"\n📊 STATISTICS:")
    print(f"  Avg fragmentation ratio: {np.mean(fragmentation_ratios):.2f}")
    print(f"  Median fragmentation ratio: {np.median(fragmentation_ratios):.2f}")
    print(f"  95th percentile fragmentation: {np.percentile(fragmentation_ratios, 95):.2f}")
    print(f"  Avg character length per word: {np.mean(avg_token_lengths):.1f}")
    print(f"  Avg subwords per sentence: {np.mean(subword_counts):.1f}")
    
    # Show problematic examples
    if problematic_examples:
        print(f"\n⚠️  PROBLEMATIC EXAMPLES ({len(problematic_examples)} found):")
        for ex in problematic_examples[:3]:  # Show first 3
            print(f"  Sentence {ex['sent_idx']}: {ex['word_count']} words → {ex['subword_count']} subwords (ratio: {ex['fragmentation']:.1f})")
            print(f"    Words: {' '.join(ex['original_words'])}")
            print(f"    Subwords: {' '.join(ex['subwords'])}")
            print()
    
    print()

if __name__ == "__main__":
    # Analyze the problematic languages
    problem_languages = [
        "UD_Vietnamese-VTB-UD_Vietnamese-VTB",
        "UD_Chinese-GSD-UD_Chinese-GSD", 
        "UD_Turkish-IMST-UD_Turkish-IMST",
        "UD_Hungarian-Szeged-UD_Hungarian-Szeged",
        "UD_English-EWT-UD_English-EWT",  # Baseline
    ]
    
    for lang in problem_languages:
        try:
            analyze_tokenization(lang, sample_size=50)
        except Exception as e:
            print(f"❌ Error analyzing {lang}: {e}")
            print()

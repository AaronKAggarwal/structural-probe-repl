#!/usr/bin/env python3
"""
Compute 80th percentile sentence length statistics by language and overall.

This script computes the 80th percentile of content_len (number of non-punctuation/non-symbol tokens)
for each language in the test split, and also provides an overall 80th percentile across all languages.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import csv

# Configuration
REPO_ROOT = Path(__file__).parent.parent
STATS_DIR = REPO_ROOT / "outputs" / "analysis" / "sentence_stats"

def load_sentence_stats(language_slug: str, split: str = "test") -> List[Dict]:
    """Load sentence stats for a language/split."""
    stats_path = STATS_DIR / language_slug / f"{split}_content_stats.jsonl"
    
    if not stats_path.exists():
        print(f"Warning: {stats_path} not found")
        return []
    
    stats = []
    with open(stats_path, 'r') as f:
        for line in f:
            stats.append(json.loads(line.strip()))
    
    return stats

def get_available_languages() -> List[str]:
    """Get all available language directories."""
    languages = []
    for lang_dir in STATS_DIR.iterdir():
        if lang_dir.is_dir() and lang_dir.name.startswith("UD_"):
            test_file = lang_dir / "test_content_stats.jsonl"
            if test_file.exists():
                languages.append(lang_dir.name)
    return sorted(languages)

def compute_percentile_stats(language_slug: str, split: str = "test") -> Dict:
    """Compute length percentile statistics for one language."""
    stats = load_sentence_stats(language_slug, split)
    
    if not stats:
        return {
            "language": language_slug,
            "num_sentences": 0,
            "80th_percentile_content_len": None,
            "mean_content_len": None,
            "median_content_len": None,
            "90th_percentile_content_len": None,
            "95th_percentile_content_len": None,
            "max_content_len": None
        }
    
    # Extract content lengths
    content_lens = [s["content_len"] for s in stats]
    
    return {
        "language": language_slug,
        "num_sentences": len(content_lens),
        "80th_percentile_content_len": np.percentile(content_lens, 80),
        "mean_content_len": np.mean(content_lens),
        "median_content_len": np.median(content_lens),
        "90th_percentile_content_len": np.percentile(content_lens, 90),
        "95th_percentile_content_len": np.percentile(content_lens, 95),
        "max_content_len": np.max(content_lens)
    }

def main():
    """Compute 80th percentile statistics for all languages."""
    print("Computing 80th percentile sentence length statistics...")
    print("=" * 60)
    
    # Get available languages
    languages = get_available_languages()
    print(f"Found {len(languages)} languages with test data")
    print()
    
    # Compute per-language statistics
    all_results = []
    all_content_lens = []  # For overall statistics
    
    for language_slug in languages:
        result = compute_percentile_stats(language_slug)
        all_results.append(result)
        
        # Collect all content lengths for overall stats
        stats = load_sentence_stats(language_slug)
        content_lens = [s["content_len"] for s in stats]
        all_content_lens.extend(content_lens)
        
        print(f"{language_slug:25} | "
              f"Sentences: {result['num_sentences']:5} | "
              f"80th %%ile: {result['80th_percentile_content_len']:6.1f} | "
              f"Mean: {result['mean_content_len']:6.1f} | "
              f"Median: {result['median_content_len']:6.1f}")
    
    print()
    print("=" * 60)
    
    # Compute overall statistics
    overall_stats = {
        "language": "OVERALL",
        "num_sentences": len(all_content_lens),
        "80th_percentile_content_len": np.percentile(all_content_lens, 80),
        "mean_content_len": np.mean(all_content_lens),
        "median_content_len": np.median(all_content_lens),
        "90th_percentile_content_len": np.percentile(all_content_lens, 90),
        "95th_percentile_content_len": np.percentile(all_content_lens, 95),
        "max_content_len": np.max(all_content_lens)
    }
    
    print(f"OVERALL STATISTICS:")
    print(f"  Total sentences: {overall_stats['num_sentences']:,}")
    print(f"  80th percentile content length: {overall_stats['80th_percentile_content_len']:.1f}")
    print(f"  Mean content length: {overall_stats['mean_content_len']:.1f}")
    print(f"  Median content length: {overall_stats['median_content_len']:.1f}")
    print(f"  90th percentile content length: {overall_stats['90th_percentile_content_len']:.1f}")
    print(f"  95th percentile content length: {overall_stats['95th_percentile_content_len']:.1f}")
    print(f"  Maximum content length: {overall_stats['max_content_len']:.0f}")
    print()
    
    # Add overall stats and sort by 80th percentile
    all_results.append(overall_stats)
    
    # Sort by 80th percentile (excluding overall row)
    lang_results = [r for r in all_results if r['language'] != 'OVERALL']
    lang_results.sort(key=lambda x: x['80th_percentile_content_len'])
    
    # Add overall back at the end
    sorted_results = lang_results + [overall_stats]
    
    # Save to CSV
    output_path = REPO_ROOT / "outputs" / "analysis" / "80th_percentile_content_lengths.csv"
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['language', 'num_sentences', '80th_percentile_content_len', 
                     'mean_content_len', 'median_content_len', '90th_percentile_content_len',
                     '95th_percentile_content_len', 'max_content_len']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_results:
            writer.writerow(result)
    
    print(f"Detailed results saved to: {output_path}")
    
    print("\nSUMMARY TABLE:")
    print("Language                 | 80th %ile | Mean   | Median | Sentences")
    print("-" * 65)
    for result in sorted_results:
        lang_display = result['language'][:24] if result['language'] != 'OVERALL' else 'OVERALL'
        print(f"{lang_display:24} | {result['80th_percentile_content_len']:8.1f} | "
              f"{result['mean_content_len']:6.1f} | {result['median_content_len']:6.1f} | "
              f"{result['num_sentences']:8,}")

if __name__ == "__main__":
    main()

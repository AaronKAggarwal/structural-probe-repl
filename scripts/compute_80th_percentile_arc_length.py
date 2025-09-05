#!/usr/bin/env python3
"""
Compute 80th percentile arc length statistics by language and overall.

This script computes the 80th percentile of mean_arc_len (mean dependency arc length
in content-only trees) for each language in the test split, and also provides an 
overall 80th percentile across all languages.

Arc length is the mean |rank_I(h) - rank_I(d)| on content tree, where I = content tokens
(non-PUNCT/SYM). Only sentences with num_content_arcs_used >= 1 are included.
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

def compute_arc_percentile_stats(language_slug: str, split: str = "test") -> Dict:
    """Compute arc length percentile statistics for one language."""
    stats = load_sentence_stats(language_slug, split)
    
    if not stats:
        return {
            "language": language_slug,
            "num_sentences_total": 0,
            "num_sentences_with_arcs": 0,
            "arc_coverage": 0.0,
            "80th_percentile_arc_len": None,
            "mean_arc_len": None,
            "median_arc_len": None,
            "90th_percentile_arc_len": None,
            "95th_percentile_arc_len": None,
            "max_arc_len": None
        }
    
    # Extract arc lengths for sentences with valid arcs
    # Only include sentences where num_content_arcs_used >= 1 and mean_arc_len is not None
    valid_arc_sentences = [
        s for s in stats 
        if s.get('num_content_arcs_used', 0) >= 1 and s.get('mean_arc_len') is not None
    ]
    
    if not valid_arc_sentences:
        return {
            "language": language_slug,
            "num_sentences_total": len(stats),
            "num_sentences_with_arcs": 0,
            "arc_coverage": 0.0,
            "80th_percentile_arc_len": None,
            "mean_arc_len": None,
            "median_arc_len": None,
            "90th_percentile_arc_len": None,
            "95th_percentile_arc_len": None,
            "max_arc_len": None
        }
    
    arc_lens = [s["mean_arc_len"] for s in valid_arc_sentences]
    coverage = len(valid_arc_sentences) / len(stats) if stats else 0.0
    
    return {
        "language": language_slug,
        "num_sentences_total": len(stats),
        "num_sentences_with_arcs": len(valid_arc_sentences),
        "arc_coverage": coverage,
        "80th_percentile_arc_len": np.percentile(arc_lens, 80),
        "mean_arc_len": np.mean(arc_lens),
        "median_arc_len": np.median(arc_lens),
        "90th_percentile_arc_len": np.percentile(arc_lens, 90),
        "95th_percentile_arc_len": np.percentile(arc_lens, 95),
        "max_arc_len": np.max(arc_lens)
    }

def main():
    """Compute 80th percentile arc length statistics for all languages."""
    print("Computing 80th percentile arc length statistics...")
    print("=" * 70)
    
    # Get available languages
    languages = get_available_languages()
    print(f"Found {len(languages)} languages with test data")
    print()
    
    # Compute per-language statistics
    all_results = []
    all_arc_lens = []  # For overall statistics
    
    for language_slug in languages:
        result = compute_arc_percentile_stats(language_slug)
        all_results.append(result)
        
        # Collect all arc lengths for overall stats
        stats = load_sentence_stats(language_slug)
        valid_arc_sentences = [
            s for s in stats 
            if s.get('num_content_arcs_used', 0) >= 1 and s.get('mean_arc_len') is not None
        ]
        arc_lens = [s["mean_arc_len"] for s in valid_arc_sentences]
        all_arc_lens.extend(arc_lens)
        
        if result['num_sentences_with_arcs'] > 0:
            print(f"{language_slug:25} | "
                  f"Arcs: {result['num_sentences_with_arcs']:5} "
                  f"({result['arc_coverage']:5.1%}) | "
                  f"80th %%ile: {result['80th_percentile_arc_len']:6.2f} | "
                  f"Mean: {result['mean_arc_len']:6.2f} | "
                  f"Median: {result['median_arc_len']:6.2f}")
        else:
            print(f"{language_slug:25} | No valid arc data")
    
    print()
    print("=" * 70)
    
    # Compute overall statistics
    if all_arc_lens:
        overall_stats = {
            "language": "OVERALL",
            "num_sentences_total": sum(r['num_sentences_total'] for r in all_results),
            "num_sentences_with_arcs": len(all_arc_lens),
            "arc_coverage": len(all_arc_lens) / sum(r['num_sentences_total'] for r in all_results),
            "80th_percentile_arc_len": np.percentile(all_arc_lens, 80),
            "mean_arc_len": np.mean(all_arc_lens),
            "median_arc_len": np.median(all_arc_lens),
            "90th_percentile_arc_len": np.percentile(all_arc_lens, 90),
            "95th_percentile_arc_len": np.percentile(all_arc_lens, 95),
            "max_arc_len": np.max(all_arc_lens)
        }
        
        print(f"OVERALL STATISTICS:")
        print(f"  Total sentences: {overall_stats['num_sentences_total']:,}")
        print(f"  Sentences with arcs: {overall_stats['num_sentences_with_arcs']:,} ({overall_stats['arc_coverage']:.1%})")
        print(f"  80th percentile arc length: {overall_stats['80th_percentile_arc_len']:.2f}")
        print(f"  Mean arc length: {overall_stats['mean_arc_len']:.2f}")
        print(f"  Median arc length: {overall_stats['median_arc_len']:.2f}")
        print(f"  90th percentile arc length: {overall_stats['90th_percentile_arc_len']:.2f}")
        print(f"  95th percentile arc length: {overall_stats['95th_percentile_arc_len']:.2f}")
        print(f"  Maximum arc length: {overall_stats['max_arc_len']:.2f}")
        print()
    else:
        overall_stats = {
            "language": "OVERALL",
            "num_sentences_total": sum(r['num_sentences_total'] for r in all_results),
            "num_sentences_with_arcs": 0,
            "arc_coverage": 0.0,
            "80th_percentile_arc_len": None,
            "mean_arc_len": None,
            "median_arc_len": None,
            "90th_percentile_arc_len": None,
            "95th_percentile_arc_len": None,
            "max_arc_len": None
        }
        print("No valid arc data found across all languages!")
    
    # Add overall stats and sort by 80th percentile
    all_results.append(overall_stats)
    
    # Sort by 80th percentile (excluding overall row and None values)
    lang_results = [r for r in all_results if r['language'] != 'OVERALL' and r['80th_percentile_arc_len'] is not None]
    lang_results.sort(key=lambda x: x['80th_percentile_arc_len'])
    
    # Add languages with no arc data at the end, then overall
    no_arc_results = [r for r in all_results if r['language'] != 'OVERALL' and r['80th_percentile_arc_len'] is None]
    sorted_results = lang_results + no_arc_results + [overall_stats]
    
    # Save to CSV
    output_path = REPO_ROOT / "outputs" / "analysis" / "80th_percentile_arc_lengths.csv"
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['language', 'num_sentences_total', 'num_sentences_with_arcs', 'arc_coverage',
                     '80th_percentile_arc_len', 'mean_arc_len', 'median_arc_len', 
                     '90th_percentile_arc_len', '95th_percentile_arc_len', 'max_arc_len']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_results:
            writer.writerow(result)
    
    print(f"Detailed results saved to: {output_path}")
    
    print("\nSUMMARY TABLE:")
    print("Language                 | Arc Sent | Coverage | 80th %ile | Mean   | Median")
    print("-" * 75)
    for result in sorted_results:
        lang_display = result['language'][:24] if result['language'] != 'OVERALL' else 'OVERALL'
        if result['80th_percentile_arc_len'] is not None:
            print(f"{lang_display:24} | {result['num_sentences_with_arcs']:8,} | "
                  f"{result['arc_coverage']:7.1%} | {result['80th_percentile_arc_len']:8.2f} | "
                  f"{result['mean_arc_len']:6.2f} | {result['median_arc_len']:6.2f}")
        else:
            print(f"{lang_display:24} | {result['num_sentences_with_arcs']:8,} | "
                  f"{result['arc_coverage']:7.1%} |      N/A |    N/A |    N/A")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge sentence-level covariates into the master per-layer results table.

Aggregates canonical sentence stats to per-language covariates:
- mean_content_len_test, median_content_len_test  
- mean_arc_length_test

Then merges into outputs/analysis/master_results_per_layer.csv
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent
STATS_DIR = REPO_ROOT / "outputs" / "analysis" / "sentence_stats"
MASTER_PATH = REPO_ROOT / "outputs" / "analysis" / "master_results_per_layer.csv"

def load_sentence_stats(language_slug: str, split: str) -> List[Dict]:
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

def compute_language_aggregates(language_slug: str) -> Dict[str, Optional[float]]:
    """Compute aggregated covariates for one language from test sentences."""
    test_stats = load_sentence_stats(language_slug, "test")
    
    if not test_stats:
        return {
            "mean_content_len_test": None,
            "median_content_len_test": None, 
            "mean_arc_length_test": None,
            "mean_tree_height_test": None,
            "mean_orig_len_test": None,
            "mean_content_ratio_test": None,
            "mean_num_arcs_test": None
        }
    
    # Extract metrics
    content_lens = [s["content_len"] for s in test_stats]
    arc_lens = [s["mean_arc_len"] for s in test_stats if s["mean_arc_len"] is not None]
    tree_heights = [s.get("tree_height") for s in test_stats if s.get("tree_height") is not None]
    orig_lens = [s.get("orig_len_incl_punct") for s in test_stats if s.get("orig_len_incl_punct") is not None]
    content_ratios = [s.get("content_ratio") for s in test_stats if s.get("content_ratio") is not None]
    num_arcs = [s.get("num_content_arcs_used") for s in test_stats if s.get("num_content_arcs_used") is not None]
    
    # Compute aggregates
    mean_content_len = np.mean(content_lens) if content_lens else None
    median_content_len = np.median(content_lens) if content_lens else None
    mean_arc_len = np.mean(arc_lens) if arc_lens else None
    mean_tree_height = np.mean(tree_heights) if tree_heights else None
    mean_orig_len = np.mean(orig_lens) if orig_lens else None
    mean_content_ratio = np.mean(content_ratios) if content_ratios else None
    mean_num_arcs = np.mean(num_arcs) if num_arcs else None
    
    return {
        "mean_content_len_test": mean_content_len,
        "median_content_len_test": median_content_len,
        "mean_arc_length_test": mean_arc_len,
        "mean_tree_height_test": mean_tree_height,
        "mean_orig_len_test": mean_orig_len,
        "mean_content_ratio_test": mean_content_ratio,
        "mean_num_arcs_test": mean_num_arcs
    }

def get_available_languages() -> List[str]:
    """Get languages with computed sentence stats."""
    if not STATS_DIR.exists():
        return []
    
    return [
        d.name for d in STATS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("UD_")
    ]

def main():
    """Main entry point."""
    
    # Load existing master table
    if not MASTER_PATH.exists():
        print(f"Error: {MASTER_PATH} not found. Run build_master_per_layer.py first.")
        return
    
    print(f"Loading master table from {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Get available languages
    available_languages = get_available_languages()
    print(f"Found sentence stats for {len(available_languages)} languages")
    
    # Compute aggregates for each language
    language_covariates = {}
    for lang in available_languages:
        print(f"Computing aggregates for {lang}")
        covariates = compute_language_aggregates(lang)
        language_covariates[lang] = covariates
        
        # Print summary
        print(f"  mean_content_len_test: {covariates['mean_content_len_test']:.2f}" if covariates['mean_content_len_test'] else "  mean_content_len_test: None")
        print(f"  median_content_len_test: {covariates['median_content_len_test']:.2f}" if covariates['median_content_len_test'] else "  median_content_len_test: None")
        print(f"  mean_arc_length_test: {covariates['mean_arc_length_test']:.3f}" if covariates['mean_arc_length_test'] else "  mean_arc_length_test: None")
        print(f"  mean_tree_height_test: {covariates['mean_tree_height_test']:.2f}" if covariates['mean_tree_height_test'] else "  mean_tree_height_test: None")
        print(f"  mean_orig_len_test: {covariates['mean_orig_len_test']:.2f}" if covariates['mean_orig_len_test'] else "  mean_orig_len_test: None")
        print(f"  mean_content_ratio_test: {covariates['mean_content_ratio_test']:.3f}" if covariates['mean_content_ratio_test'] else "  mean_content_ratio_test: None")
        print(f"  mean_num_arcs_test: {covariates['mean_num_arcs_test']:.1f}" if covariates['mean_num_arcs_test'] else "  mean_num_arcs_test: None")
    
    # Add covariate columns to dataframe
    df["mean_content_len_test"] = df["language_slug"].map(
        lambda lang: language_covariates.get(lang, {}).get("mean_content_len_test")
    )
    df["median_content_len_test"] = df["language_slug"].map(
        lambda lang: language_covariates.get(lang, {}).get("median_content_len_test")
    )
    df["mean_arc_length_test"] = df["language_slug"].map(
        lambda lang: language_covariates.get(lang, {}).get("mean_arc_length_test")
    )
    df["mean_tree_height_test"] = df["language_slug"].map(
        lambda lang: language_covariates.get(lang, {}).get("mean_tree_height_test")
    )
    df["mean_orig_len_test"] = df["language_slug"].map(
        lambda lang: language_covariates.get(lang, {}).get("mean_orig_len_test")
    )
    df["mean_content_ratio_test"] = df["language_slug"].map(
        lambda lang: language_covariates.get(lang, {}).get("mean_content_ratio_test")
    )
    df["mean_num_arcs_test"] = df["language_slug"].map(
        lambda lang: language_covariates.get(lang, {}).get("mean_num_arcs_test")
    )
    
    # Check for missing data
    missing_langs = set(df["language_slug"]) - set(available_languages)
    if missing_langs:
        print(f"\nWarning: Missing sentence stats for languages: {sorted(missing_langs)}")
    
    # Save updated table
    output_path = MASTER_PATH
    df.to_csv(output_path, index=False)
    print(f"\nSaved updated master table to {output_path}")
    print(f"Added columns: mean_content_len_test, median_content_len_test, mean_arc_length_test, mean_tree_height_test, mean_orig_len_test, mean_content_ratio_test, mean_num_arcs_test")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Languages with sentence stats: {len(available_languages)}")
    print(f"  Rows with non-null mean_content_len_test: {df['mean_content_len_test'].notna().sum()}")
    print(f"  Rows with non-null mean_arc_length_test: {df['mean_arc_length_test'].notna().sum()}")
    print(f"  Rows with non-null mean_content_ratio_test: {df['mean_content_ratio_test'].notna().sum()}")

if __name__ == "__main__":
    main()

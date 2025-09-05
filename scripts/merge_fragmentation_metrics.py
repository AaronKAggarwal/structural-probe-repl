#!/usr/bin/env python3
"""
Merge fragmentation metrics into the master per-layer results table.

Aggregates fragmentation stats to per-language metrics:
- fragmentation_ratio_content_mean (primary metric)
- fragmentation_ratio_overall_mean (secondary/diagnostic)

Then merges into outputs/analysis/master_results_per_layer.csv
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent
FRAG_STATS_DIR = REPO_ROOT / "outputs" / "analysis" / "fragmentation_stats"
MASTER_PATH = REPO_ROOT / "outputs" / "analysis" / "master_results_per_layer.csv"

def load_fragmentation_stats(language_slug: str, split: str) -> List[Dict]:
    """Load fragmentation stats for a language/split."""
    stats_path = FRAG_STATS_DIR / language_slug / f"{split}_fragmentation_stats.jsonl"
    
    if not stats_path.exists():
        print(f"Warning: {stats_path} not found")
        return []
    
    stats = []
    with open(stats_path, 'r') as f:
        for line in f:
            stats.append(json.loads(line.strip()))
    
    return stats

def compute_language_fragmentation_aggregates(language_slug: str) -> Dict[str, Optional[float]]:
    """Compute aggregated fragmentation metrics for one language from test sentences."""
    test_stats = load_fragmentation_stats(language_slug, "test")
    
    if not test_stats:
        return {
            "fragmentation_ratio_content_mean": None,
            "fragmentation_ratio_overall_mean": None
        }
    
    # Extract fragmentation ratios
    content_ratios = [s["fragmentation_ratio_content"] for s in test_stats if s["fragmentation_ratio_content"] is not None]
    overall_ratios = [s["fragmentation_ratio_overall"] for s in test_stats if s["fragmentation_ratio_overall"] is not None]
    
    # Compute means
    mean_content_frag = np.mean(content_ratios) if content_ratios else None
    mean_overall_frag = np.mean(overall_ratios) if overall_ratios else None
    
    return {
        "fragmentation_ratio_content_mean": mean_content_frag,
        "fragmentation_ratio_overall_mean": mean_overall_frag
    }

def get_available_languages() -> List[str]:
    """Get languages with computed fragmentation stats."""
    if not FRAG_STATS_DIR.exists():
        return []
    
    return [
        d.name for d in FRAG_STATS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("UD_")
    ]

def main():
    """Main entry point."""
    
    # Load existing master table
    if not MASTER_PATH.exists():
        print(f"Error: {MASTER_PATH} not found. Run previous stages first.")
        return
    
    print(f"Loading master table from {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Get available languages
    available_languages = get_available_languages()
    print(f"Found fragmentation stats for {len(available_languages)} languages")
    
    # Compute aggregates for each language
    language_fragmentation = {}
    for lang in available_languages:
        print(f"Computing fragmentation aggregates for {lang}")
        frag_metrics = compute_language_fragmentation_aggregates(lang)
        language_fragmentation[lang] = frag_metrics
        
        # Print summary
        content_frag = frag_metrics['fragmentation_ratio_content_mean']
        overall_frag = frag_metrics['fragmentation_ratio_overall_mean']
        
        print(f"  fragmentation_ratio_content_mean: {content_frag:.3f}" if content_frag else "  fragmentation_ratio_content_mean: None")
        print(f"  fragmentation_ratio_overall_mean: {overall_frag:.3f}" if overall_frag else "  fragmentation_ratio_overall_mean: None")
    
    # Add fragmentation columns to dataframe
    df["fragmentation_ratio_content_mean"] = df["language_slug"].map(
        lambda lang: language_fragmentation.get(lang, {}).get("fragmentation_ratio_content_mean")
    )
    df["fragmentation_ratio_overall_mean"] = df["language_slug"].map(
        lambda lang: language_fragmentation.get(lang, {}).get("fragmentation_ratio_overall_mean")
    )
    
    # Check for missing data
    missing_langs = set(df["language_slug"]) - set(available_languages)
    if missing_langs:
        print(f"\nWarning: Missing fragmentation stats for languages: {sorted(missing_langs)}")
    
    # Save updated table
    output_path = MASTER_PATH
    df.to_csv(output_path, index=False)
    print(f"\nSaved updated master table to {output_path}")
    print(f"Added columns: fragmentation_ratio_content_mean (primary), fragmentation_ratio_overall_mean (diagnostic)")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Languages with fragmentation stats: {len(available_languages)}")
    print(f"  Rows with non-null content fragmentation: {df['fragmentation_ratio_content_mean'].notna().sum()}")
    print(f"  Rows with non-null overall fragmentation: {df['fragmentation_ratio_overall_mean'].notna().sum()}")
    
    # Show range of fragmentation values
    if df['fragmentation_ratio_content_mean'].notna().any():
        lang_stats = df.drop_duplicates('language_slug')[['language_slug', 'fragmentation_ratio_content_mean', 'fragmentation_ratio_overall_mean']]
        valid_stats = lang_stats.dropna()
        
        print(f"\nFragmentation ranges across languages:")
        print(f"  Content fragmentation: {valid_stats['fragmentation_ratio_content_mean'].min():.3f} - {valid_stats['fragmentation_ratio_content_mean'].max():.3f}")
        print(f"  Overall fragmentation: {valid_stats['fragmentation_ratio_overall_mean'].min():.3f} - {valid_stats['fragmentation_ratio_overall_mean'].max():.3f}")
        
        print(f"\nTop 5 most fragmented (content-only):")
        print(valid_stats.nlargest(5, 'fragmentation_ratio_content_mean')[['language_slug', 'fragmentation_ratio_content_mean']])
        
        print(f"\nTop 5 least fragmented (content-only):")
        print(valid_stats.nsmallest(5, 'fragmentation_ratio_content_mean')[['language_slug', 'fragmentation_ratio_content_mean']])

if __name__ == "__main__":
    main()

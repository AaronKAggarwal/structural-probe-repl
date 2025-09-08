#!/usr/bin/env python3
"""
Compute aggregated fragmentation metrics for each language.

Aggregates fragmentation stats to per-language metrics:
- fragmentation_ratio_content_mean (primary metric)
- fragmentation_ratio_overall_mean (secondary/diagnostic)

Outputs fragmentation_metrics.csv for later joining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
FRAG_STATS_DIR = REPO_ROOT / "outputs" / "analysis" / "fragmentation_stats"

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
    print("Computing aggregated fragmentation metrics...")
    print("-" * 60)
    # Get available languages
    available_languages = get_available_languages()
    print(f"Found fragmentation stats for {len(available_languages)} languages")
    # Compute aggregates for each language
    fragmentation_data = []
    for lang in available_languages:
        print(f"Computing fragmentation aggregates for {lang}...")
        frag_metrics = compute_language_fragmentation_aggregates(lang)
        # Add language_slug to the fragmentation dict
        frag_metrics["language_slug"] = lang
        fragmentation_data.append(frag_metrics)
        # Print summary
        content_frag = frag_metrics['fragmentation_ratio_content_mean']
        overall_frag = frag_metrics['fragmentation_ratio_overall_mean']
        print(f"  fragmentation_ratio_content_mean: {content_frag:.3f}" if content_frag else "  fragmentation_ratio_content_mean: None")
        print(f"  fragmentation_ratio_overall_mean: {overall_frag:.3f}" if overall_frag else "  fragmentation_ratio_overall_mean: None")
    # Create DataFrame
    df = pd.DataFrame(fragmentation_data)
    # Reorder columns to put language_slug first
    cols = ['language_slug'] + [col for col in df.columns if col != 'language_slug']
    df = df[cols]
    # Save to CSV
    output_path = Path(__file__).parent / "fragmentation_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Done: Saved fragmentation metrics to {output_path}")
    print(f"Done: Shape: {df.shape}")
    print(f"Done: Columns: {list(df.columns)}")
    # Summary statistics
    print("Summary:")
    print(f"  Languages processed: {len(df)}")
    print(f"  Non-null fragmentation_ratio_content_mean: {df['fragmentation_ratio_content_mean'].notna().sum()}")
    print(f"  Non-null fragmentation_ratio_overall_mean: {df['fragmentation_ratio_overall_mean'].notna().sum()}")
    # Show range of fragmentation values
    if df['fragmentation_ratio_content_mean'].notna().any():
        valid_stats = df.dropna()
        print("Fragmentation ranges across languages:")
        print(f"  Content fragmentation: {valid_stats['fragmentation_ratio_content_mean'].min():.3f} - {valid_stats['fragmentation_ratio_content_mean'].max():.3f}")
        print(f"  Overall fragmentation: {valid_stats['fragmentation_ratio_overall_mean'].min():.3f} - {valid_stats['fragmentation_ratio_overall_mean'].max():.3f}")
        print("Top 5 most fragmented (content-only):")
        print(valid_stats.nlargest(5, 'fragmentation_ratio_content_mean')[['language_slug', 'fragmentation_ratio_content_mean']])
        print("Top 5 least fragmented (content-only):")
        print(valid_stats.nsmallest(5, 'fragmentation_ratio_content_mean')[['language_slug', 'fragmentation_ratio_content_mean']])
    print("Done: Ready for joining into analysis tables")

if __name__ == "__main__":
    main()

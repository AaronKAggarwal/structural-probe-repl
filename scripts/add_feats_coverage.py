#!/usr/bin/env python3
"""
Add FEATS coverage metrics to morphological complexity data.

Computes percentage of content tokens with FEATS ≠ "_" for train/dev/test splits
and adds coverage bands for scientific interpretation.
"""

from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

UD_DATA_ROOT = Path(__file__).parent.parent / "data" / "ud"

def compute_feats_coverage(language_slug: str, split: str) -> float:
    """Compute FEATS coverage for one language/split."""
    language_dir = UD_DATA_ROOT / language_slug
    split_file = language_dir / f"{split}.conllu"
    
    if not split_file.exists():
        return 0.0
    
    content_tokens = 0
    non_empty_feats = 0
    
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line and not line.startswith('#'):
                fields = line.split('\t')
                if len(fields) >= 10:
                    token_id = fields[0]
                    upos = fields[3]
                    feats = fields[5]
                    
                    # Skip MWT ranges and empty nodes
                    if '-' in token_id or '.' in token_id:
                        continue
                    
                    # Content tokens only (exclude PUNCT, SYM)
                    if upos not in {'PUNCT', 'SYM'}:
                        content_tokens += 1
                        if feats != '_':
                            non_empty_feats += 1
    
    return (non_empty_feats / content_tokens * 100) if content_tokens > 0 else 0.0

def assign_coverage_band(coverage_pct: float) -> str:
    """Assign coverage band based on percentage."""
    if coverage_pct >= 10.0:
        return "Adequate"
    elif coverage_pct >= 1.0:
        return "Sparse"
    else:
        return "Absent"

def main():
    """Add FEATS coverage to morphological complexity data."""
    print("Adding FEATS Coverage to Morphological Complexity Data")
    print("=" * 60)
    
    # Load existing morph complexity data
    morph_file = Path("outputs/analysis/morph_complexity.csv")
    if not morph_file.exists():
        print(f"❌ Morphological complexity file not found: {morph_file}")
        return
    
    df = pd.read_csv(morph_file)
    print(f"Loaded {len(df)} languages from morph_complexity.csv")
    
    # Compute coverage for all languages
    coverage_data = []
    
    for _, row in df.iterrows():
        language_slug = row['language_slug']
        print(f"Processing {language_slug}...")
        
        # Compute coverage for train/dev/test
        coverage_train = compute_feats_coverage(language_slug, 'train')
        coverage_dev = compute_feats_coverage(language_slug, 'dev')
        coverage_test = compute_feats_coverage(language_slug, 'test')
        
        # Assign band based on train coverage (primary)
        coverage_band = assign_coverage_band(coverage_train)
        
        coverage_data.append({
            'language_slug': language_slug,
            'feats_coverage_train': coverage_train,
            'feats_coverage_dev': coverage_dev,
            'feats_coverage_test': coverage_test,
            'feats_coverage_band': coverage_band
        })
        
        print(f"  Train: {coverage_train:.1f}%, Dev: {coverage_dev:.1f}%, Test: {coverage_test:.1f}% → {coverage_band}")
    
    # Create coverage DataFrame
    coverage_df = pd.DataFrame(coverage_data)
    
    # Merge with existing morphological complexity data
    df_updated = pd.merge(df, coverage_df, on='language_slug', how='left')
    
    # Save updated file
    df_updated.to_csv(morph_file, index=False)
    print(f"\n✓ Updated morphological complexity file: {morph_file}")
    print(f"  Added columns: feats_coverage_train, feats_coverage_dev, feats_coverage_test, feats_coverage_band")
    
    # Summary statistics
    print(f"\n📊 FEATS Coverage Summary:")
    print(f"   Shape: {df_updated.shape}")
    
    # Coverage band distribution
    band_counts = coverage_df['feats_coverage_band'].value_counts()
    print(f"\n   Coverage Band Distribution:")
    for band in ['Adequate', 'Sparse', 'Absent']:
        count = band_counts.get(band, 0)
        pct = count / len(coverage_df) * 100
        print(f"     {band:8}: {count:2d} languages ({pct:4.1f}%)")
    
    # Show languages by band
    print(f"\n   Languages by Coverage Band:")
    for band in ['Adequate', 'Sparse', 'Absent']:
        band_langs = coverage_df[coverage_df['feats_coverage_band'] == band]['language_slug'].tolist()
        if band_langs:
            print(f"     {band}:")
            for lang in sorted(band_langs):
                coverage = coverage_df[coverage_df['language_slug'] == lang]['feats_coverage_train'].iloc[0]
                lang_short = lang.replace('UD_', '').replace('-GSD', '').replace('-BDT', '').replace('-BTB', '').replace('-PDTC', '').replace('-EWT', '').replace('-TDT', '').replace('-GDT', '').replace('-HTB', '').replace('-HDTB', '').replace('-Szeged', '').replace('-IMST', '').replace('-PDB', '').replace('-SynTagRus', '').replace('-AnCora', '').replace('-UDTB', '').replace('-VTB', '').replace('-Seraji', '').replace('-PADT', '')
                print(f"       {lang_short:12} ({coverage:4.1f}%)")
    
    print(f"\n✓ Ready for modeling plan and integration")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test coverage orthogonality of the v1.2_base morphological complexity metric."""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def main():
    # Load detailed results
    df = pd.read_csv('morph_complexity_core.csv')
    
    # Coverage orthogonality test
    complex_pc1 = df['complex_pc1']
    marking_rate = df['overall_marking_rate']
    
    # Correlations
    pearson_r, pearson_p = pearsonr(complex_pc1, marking_rate)
    spearman_rho, spearman_p = spearmanr(complex_pc1, marking_rate)
    
    print('=== COVERAGE ORTHOGONALITY TEST ===')
    print(f'Languages tested: {len(df)}')
    print(f'')
    print(f'complex_pc1 vs overall_marking_rate:')
    print(f'  Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})')
    print(f'  Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.4f})')
    print(f'')
    print(f'Target: |r| ≤ 0.2 for good orthogonality')
    print(f'Status: {"✅ PASS" if abs(pearson_r) <= 0.2 else "❌ FAIL"}')
    print(f'')
    
    # Before/after residualization comparison
    if all(col in df.columns for col in ['z_mean_feats', 'z_entropy_mm', 'z_richness']):
        print('=== BEFORE RESIDUALIZATION ===')
        for metric in ['z_mean_feats', 'z_entropy_mm', 'z_richness']:
            r_before, _ = pearsonr(df[metric], marking_rate)
            print(f'{metric:>15} vs marking_rate: r = {r_before:>7.4f}')
        
        print(f'')
        print('=== AFTER RESIDUALIZATION ===')
        for metric in ['resid_z_mean_feats', 'resid_z_entropy_mm', 'resid_z_richness']:
            if metric in df.columns:
                r_after, _ = pearsonr(df[metric], marking_rate)
                print(f'{metric:>20} vs marking_rate: r = {r_after:>7.4f}')
    
    # Show top/bottom languages by marking rate for context
    print(f'\n=== MARKING RATE CONTEXT ===')
    df_sorted = df.sort_values('overall_marking_rate')
    print('Lowest marking rates:')
    for _, row in df_sorted.head(3).iterrows():
        print(f'  {row["language_slug"]:20} rate={row["overall_marking_rate"]:.3f} complex={row["complex_pc1"]:6.3f}')
    print('Highest marking rates:')
    for _, row in df_sorted.tail(3).iterrows():
        print(f'  {row["language_slug"]:20} rate={row["overall_marking_rate"]:.3f} complex={row["complex_pc1"]:6.3f}')

if __name__ == "__main__":
    main()

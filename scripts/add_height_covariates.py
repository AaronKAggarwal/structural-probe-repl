#!/usr/bin/env python3
"""
Add residualized and normalized tree height covariates to master table.

Addresses high correlation (r=0.951) between tree height and sentence length
by creating orthogonal shape measures:

1. height_residual: height - f_hat(length) using linear regression
2. height_normalized_log: height / log2(n_content + 1) 
3. height_normalized_linear: height / n_content

These provide "shape-given-length" measures suitable for regression without VIF inflation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
MASTER_PATH = REPO_ROOT / "outputs" / "analysis" / "master_results_per_layer.csv"

def compute_height_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residualized tree height controlling for sentence length."""
    # Get unique language data for regression
    lang_data = df.drop_duplicates('language_slug')[
        ['language_slug', 'mean_tree_height_test', 'mean_content_len_test']
    ].copy()
    
    # Remove any rows with missing data
    lang_data = lang_data.dropna()
    
    print(f"Computing height residuals from {len(lang_data)} languages")
    
    # Fit linear regression: height ~ length
    X = lang_data[['mean_content_len_test']].values
    y = lang_data['mean_tree_height_test'].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Compute predicted heights
    y_pred = reg.predict(X)
    
    # Compute residuals
    residuals = y - y_pred
    
    # Create mapping from language to residual
    lang_data['height_residual'] = residuals
    lang_to_residual = dict(zip(lang_data['language_slug'], lang_data['height_residual']))
    
    # Add residuals to full dataframe
    df['height_residual'] = df['language_slug'].map(lang_to_residual)
    
    # Report regression stats
    r_squared = reg.score(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    print(f"Height ~ Length regression:")
    print(f"  R² = {r_squared:.3f}")
    print(f"  height = {slope:.3f} * length + {intercept:.3f}")
    print(f"  Residual std = {np.std(residuals):.3f}")
    
    return df

def compute_normalized_heights(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized tree height metrics."""
    
    # Log-normalized height (balanced tree baseline)
    df['height_normalized_log'] = df['mean_tree_height_test'] / np.log2(df['mean_content_len_test'] + 1)
    
    # Linear-normalized height (chain baseline)  
    df['height_normalized_linear'] = df['mean_tree_height_test'] / df['mean_content_len_test']
    
    return df

def analyze_correlations(df: pd.DataFrame) -> None:
    """Analyze correlations between height measures and length."""
    
    # Get unique language data
    lang_data = df.drop_duplicates('language_slug')[[
        'language_slug', 'mean_content_len_test', 'mean_tree_height_test',
        'height_residual', 'height_normalized_log', 'height_normalized_linear'
    ]].dropna()
    
    print(f"\nCorrelations with sentence length (n={len(lang_data)} languages):")
    
    metrics = [
        ('Original height', 'mean_tree_height_test'),
        ('Height residual', 'height_residual'), 
        ('Height/log2(n+1)', 'height_normalized_log'),
        ('Height/n', 'height_normalized_linear')
    ]
    
    for name, col in metrics:
        r = lang_data['mean_content_len_test'].corr(lang_data[col])
        print(f"  {name:20s}: r = {r:6.3f}")
    
    print(f"\nHeight measure statistics:")
    for name, col in metrics:
        values = lang_data[col]
        print(f"  {name:20s}: {values.mean():.3f} ± {values.std():.3f} (range: {values.min():.3f} - {values.max():.3f})")

def main():
    """Main entry point."""
    
    # Load master table
    if not MASTER_PATH.exists():
        print(f"Error: {MASTER_PATH} not found")
        return
    
    print(f"Loading master table from {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Check for required columns
    required_cols = ['mean_tree_height_test', 'mean_content_len_test']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Compute residualized height
    df = compute_height_residuals(df)
    
    # Compute normalized heights
    df = compute_normalized_heights(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Save updated table
    df.to_csv(MASTER_PATH, index=False)
    print(f"\nSaved updated master table with height covariates")
    print(f"Added columns: height_residual, height_normalized_log, height_normalized_linear")
    
    # Show final column count
    print(f"Master table now has {len(df.columns)} columns:")
    new_cols = ['height_residual', 'height_normalized_log', 'height_normalized_linear']
    for col in new_cols:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} non-null values")

if __name__ == "__main__":
    main()

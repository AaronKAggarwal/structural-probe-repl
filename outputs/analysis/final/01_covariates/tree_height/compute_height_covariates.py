#!/usr/bin/env python3
"""
Compute residualized and normalized tree height covariates.

Addresses high correlation (r=0.951) between tree height and sentence length
by creating orthogonal shape measures:

1. height_residual: height - f_hat(length)
2. height_normalized_log: height / log2(n_content + 1)
3. height_normalized_linear: height / n_content

These provide "shape-given-length" measures suitable for regression without VIF inflation.
Outputs height_covariates.csv for later joining.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
SENTENCE_AGGREGATES_PATH = REPO_ROOT / "outputs" / "analysis" / "final" / "01_covariates" / "sentence_aggregates" / "sentence_aggregates.csv"

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

    print("Height ~ Length regression:")
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

    print("\nHeight measure statistics:")
    for name, col in metrics:
        values = lang_data[col]
        print(f"  {name:20s}: {values.mean():.3f} \u00B1 {values.std():.3f} (range: {values.min():.3f} - {values.max():.3f})")

def main():
    """Main entry point."""
    print("Computing tree height covariates...")
    print("-" * 60)

    # Load sentence aggregates
    if not SENTENCE_AGGREGATES_PATH.exists():
        print(f"Error: {SENTENCE_AGGREGATES_PATH} not found")
        print("Run compute_sentence_aggregates.py first")
        return

    print(f"Loading sentence aggregates from {SENTENCE_AGGREGATES_PATH}")
    df = pd.read_csv(SENTENCE_AGGREGATES_PATH)
    print(f"Loaded {len(df)} languages with {len(df.columns)} columns")

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

    # Create output with just the height covariates + language_slug
    height_cols = ['language_slug', 'height_residual', 'height_normalized_log', 'height_normalized_linear']
    height_df = df[height_cols].copy()

    # Save height covariates
    output_path = Path(__file__).parent / "height_covariates.csv"
    height_df.to_csv(output_path, index=False)

    print(f"\nDone: Saved height covariates to {output_path}")
    print(f"Done: Shape: {height_df.shape}")
    print(f"Done: Columns: {list(height_df.columns)}")

    # Show final stats
    print("\nHeight covariate statistics:")
    for col in ['height_residual', 'height_normalized_log', 'height_normalized_linear']:
        non_null = height_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(height_df)} non-null values")

    print("\nDone: Ready for joining into analysis tables")

if __name__ == "__main__":
    main()

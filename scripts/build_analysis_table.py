#!/usr/bin/env python3
"""
Build comprehensive analysis table by joining all covariates.

Creates outputs/analysis/analysis_table_per_layer.csv by left-joining:
- Master results (immutable metrics-only ledger)
- Stage 2: Fragmentation metrics
- Stage 3: Tree shape covariates  
- Stage 4A: UD dataset statistics
- Stage 4B: Pretraining exposure (when ready)
- Stage 5: Morphological complexity (when ready)

Keeps master_results_per_layer.csv pristine as immutable ledger.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"

def load_master_table() -> pd.DataFrame:
    """Load the immutable master results table."""
    master_file = ANALYSIS_DIR / "master_results_per_layer.csv"
    print(f"Loading master table: {master_file}")
    
    if not master_file.exists():
        raise FileNotFoundError(f"Master table not found: {master_file}")
    
    df = pd.read_csv(master_file)
    print(f"  Shape: {df.shape}")
    print(f"  Languages: {df['language_slug'].nunique()}")
    print(f"  Layers: {sorted(df['layer'].unique())}")
    print(f"  Probes: {sorted(df['probe'].unique())}")
    
    return df

def load_covariate_tables() -> Dict[str, pd.DataFrame]:
    """Load all available covariate tables."""
    covariate_files = {
        "ud_stats": ANALYSIS_DIR / "ud_stats_derived.csv",
        "pretrain_exposure": ANALYSIS_DIR / "pretrain_exposure.csv",
        "morph_complexity": ANALYSIS_DIR / "morph_complexity.csv",
        # Note: fragmentation and tree shape are already in master table
        # We'll add future stages here as they become available
    }
    
    covariates = {}
    
    for name, file_path in covariate_files.items():
        if file_path.exists():
            print(f"Loading {name}: {file_path}")
            df = pd.read_csv(file_path)
            print(f"  Shape: {df.shape}")
            covariates[name] = df
        else:
            print(f"⚠ {name} not found: {file_path}")
    
    return covariates

def build_analysis_table(master_df: pd.DataFrame, covariates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build comprehensive analysis table by joining all covariates."""
    print("\nBuilding analysis table...")
    
    # Start with master table
    analysis_df = master_df.copy()
    print(f"Starting with master table: {analysis_df.shape}")
    
    # Track join statistics
    join_stats = []
    
    # Join UD statistics (Stage 4A)
    if "ud_stats" in covariates:
        ud_df = covariates["ud_stats"]
        
        # Check for column collisions
        ud_columns = [col for col in ud_df.columns if col != 'language_slug']
        existing_cols = [col for col in ud_columns if col in analysis_df.columns]
        if existing_cols:
            print(f"  Column collisions with UD stats: {existing_cols}")
            # Keep master table columns, skip colliding UD columns
            ud_columns = [col for col in ud_columns if col not in existing_cols]
            ud_df = ud_df[['language_slug'] + ud_columns]
        
        print(f"  Joining UD statistics ({len(ud_columns)} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, ud_df, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs in UD columns
        nan_counts = {col: analysis_df[col].isnull().sum() for col in ud_columns}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'UD Statistics (4A)',
            'columns_added': len(ud_columns),
            'rows_before': before_rows,
            'rows_after': after_rows,
            'total_nans': total_nans,
            'success': total_nans == 0
        })
        
        if total_nans > 0:
            print(f"    ⚠ {total_nans} NaN values introduced")
            for col, nan_count in nan_counts.items():
                if nan_count > 0:
                    print(f"      {col}: {nan_count} NaNs")
        else:
            print(f"    ✓ All {len(analysis_df)} rows populated")
    
    # Join pretraining exposure (Stage 4B)
    if "pretrain_exposure" in covariates:
        exposure_df = covariates["pretrain_exposure"]
        
        # Select key columns for modeling
        exposure_columns = ['wiki_size_log2_mb', 'size_mb', 'chosen_date']
        exposure_df_subset = exposure_df[['language_slug'] + exposure_columns].copy()
        exposure_df_subset['exposure_source'] = 'wikidump_2018'
        
        print(f"  Joining pretraining exposure ({len(exposure_columns)+1} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, exposure_df_subset, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs in exposure columns
        exposure_all_cols = exposure_columns + ['exposure_source']
        nan_counts = {col: analysis_df[col].isnull().sum() for col in exposure_all_cols}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'Pretraining Exposure (4B)',
            'columns_added': len(exposure_all_cols),
            'rows_before': before_rows,
            'rows_after': after_rows,
            'total_nans': total_nans,
            'success': total_nans == 0
        })
        
        if total_nans > 0:
            print(f"    ⚠ {total_nans} NaN values introduced")
            for col, nan_count in nan_counts.items():
                if nan_count > 0:
                    print(f"      {col}: {nan_count} NaNs")
        else:
            print(f"    ✓ All {len(analysis_df)} rows populated")
    
    # Join morphological complexity (Stage 5)
    if "morph_complexity" in covariates:
        morph_df = covariates["morph_complexity"]
        
        # Select key columns for modeling
        morph_columns = ['complexity_pc1', 'feats_coverage_train', 'feats_coverage_band']
        morph_df_subset = morph_df[['language_slug'] + morph_columns].copy()
        
        print(f"  Joining morphological complexity ({len(morph_columns)} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, morph_df_subset, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs in morph columns
        nan_counts = {col: analysis_df[col].isnull().sum() for col in morph_columns}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'Morphological Complexity (5)',
            'columns_added': len(morph_columns),
            'rows_before': before_rows,
            'rows_after': after_rows,
            'total_nans': total_nans,
            'success': total_nans == 0
        })
        
        if total_nans > 0:
            print(f"    ⚠ {total_nans} NaN values introduced")
            for col, nan_count in nan_counts.items():
                if nan_count > 0:
                    print(f"      {col}: {nan_count} NaNs")
        else:
            print(f"    ✓ All {len(analysis_df)} rows populated")
    
    # Add future covariate joins here (Stage 6+, etc.)
    
    return analysis_df, join_stats

def validate_analysis_table(analysis_df: pd.DataFrame, join_stats: List[Dict]) -> None:
    """Validate the final analysis table."""
    print("\nValidation:")
    
    # Check expected row count
    expected_rows = 23 * 6 * 2  # languages × layers × probes
    actual_rows = len(analysis_df)
    
    print(f"  Expected rows: {expected_rows}")
    print(f"  Actual rows: {actual_rows}")
    
    if actual_rows == expected_rows:
        print("  ✓ Row count matches expectation")
    else:
        print("  ⚠ Row count mismatch")
    
    # Check for any NaN values
    total_nans = analysis_df.isnull().sum().sum()
    if total_nans == 0:
        print("  ✓ No NaN values found")
    else:
        print(f"  ⚠ {total_nans} NaN values found")
        nan_cols = analysis_df.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        for col, count in nan_cols.items():
            print(f"    {col}: {count} NaNs")
    
    # Check language coverage
    unique_languages = analysis_df['language_slug'].nunique()
    print(f"  Languages: {unique_languages}/23")
    
    # Check layer coverage  
    unique_layers = sorted(analysis_df['layer'].unique())
    print(f"  Layers: {unique_layers}")
    
    # Summary of joins
    print("\nJoin Summary:")
    for stat in join_stats:
        status = "✓" if stat['success'] else "⚠"
        print(f"  {status} {stat['stage']}: +{stat['columns_added']} columns, {stat['total_nans']} NaNs")

def create_l7_slice(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Create L7-only slice for paper analysis."""
    print("\nCreating L7 slice for paper analysis...")
    
    l7_df = analysis_df[analysis_df['layer'] == 'L7'].copy()
    print(f"  L7 rows: {len(l7_df)}")
    print(f"  Expected: 46 (23 languages × 2 probes)")
    
    if len(l7_df) == 46:
        print("  ✓ L7 slice has expected row count")
    else:
        print("  ⚠ L7 slice row count mismatch")
    
    return l7_df

def main():
    """Main entry point."""
    print("Building comprehensive analysis table...")
    print("Keeping master_results_per_layer.csv as immutable ledger")
    print()

    # Load master table
    master_df = load_master_table()
    
    # Load covariates
    print()
    covariates = load_covariate_tables()
    
    # Build analysis table
    print()
    analysis_df, join_stats = build_analysis_table(master_df, covariates)
    
    # Validate
    validate_analysis_table(analysis_df, join_stats)
    
    # Save analysis table
    analysis_file = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    print(f"\nSaving analysis table: {analysis_file}")
    analysis_df.to_csv(analysis_file, index=False)
    print(f"  Shape: {analysis_df.shape}")
    print(f"  Columns: {len(analysis_df.columns)}")
    
    # Create L7 slice
    l7_df = create_l7_slice(analysis_df)
    
    # Save L7 slice
    l7_file = ANALYSIS_DIR / "analysis_table_L7.csv"
    print(f"\nSaving L7 slice: {l7_file}")
    l7_df.to_csv(l7_file, index=False)
    print(f"  Shape: {l7_df.shape}")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS TABLES READY")
    print("="*60)
    print(f"Immutable ledger: master_results_per_layer.csv ({master_df.shape})")
    print(f"Analysis table:   analysis_table_per_layer.csv ({analysis_df.shape})")  
    print(f"Paper slice:      analysis_table_L7.csv ({l7_df.shape})")
    print()
    print("Ready for Stage 4B: pretraining exposure")

if __name__ == "__main__":
    main()

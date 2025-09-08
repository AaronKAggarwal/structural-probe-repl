#!/usr/bin/env python3
"""
Build comprehensive analysis table by joining all covariates.

Creates outputs/analysis/final/03_tables/analysis_table_per_layer.csv by left-joining:
- Master results (immutable metrics-only ledger) from 00_master/
- Covariates from 01_covariates/: sentence aggregates, tree height, UD stats, pretraining exposure, fragmentation
- Key predictors: complexity_pc1, complexity_pc1_adequate_only, feats_coverage_train, feats_coverage_band

Keeps master_results_per_layer.csv pristine as immutable ledger.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
FINAL_DIR = Path(__file__).parent.parent  # outputs/analysis/final/
MASTER_DIR = FINAL_DIR / "00_master"
COVARIATES_DIR = FINAL_DIR / "01_covariates"
PREDICTORS_DIR = FINAL_DIR / "02_predictors"
PREDICTORS_NEW_DIR = FINAL_DIR / "02_predictors_new"
TABLES_DIR = FINAL_DIR / "03_tables"

def load_master_table() -> pd.DataFrame:
    """Load the immutable master results table."""
    master_file = MASTER_DIR / "master_results_per_layer.csv"
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
        # Covariates (confounding factors)
        "sentence_aggregates": COVARIATES_DIR / "sentence_aggregates" / "sentence_aggregates.csv",
        "tree_height": COVARIATES_DIR / "tree_height" / "height_covariates.csv",
        "ud_stats": COVARIATES_DIR / "ud_stats" / "ud_stats_derived.csv",
        "pretrain_exposure": COVARIATES_DIR / "pretrain_exposure" / "pretrain_exposure.csv",
        "fragmentation": COVARIATES_DIR / "fragmentation" / "fragmentation_metrics.csv",
        # Predictors (key morphological complexity columns only)
        "morph_complexity": PREDICTORS_DIR / "morph_complexity" / "morph_complexity.csv",
        # New predictors (coverage-orthogonal metric; optional presence)
        "morph_complexity_new": PREDICTORS_NEW_DIR / "morph_complexity" / "main" / "final" / "output.csv",
        "morph_complexity_new_core": PREDICTORS_NEW_DIR / "morph_complexity" / "main" / "final" / "morph_complexity_core.csv",
        # Language families (phylogenetic predictors)
        "language_families": PREDICTORS_DIR / "language_families" / "glottolog_map.csv",
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
    
    # Join language families FIRST (phylogenetic predictors - fundamental language properties)
    if "language_families" in covariates:
        families_df = covariates["language_families"]
        
        # Select family columns (glottocode, family_top)
        family_columns = ['glottocode', 'family_top']
        
        print(f"  Joining language families ({len(family_columns)} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, families_df, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for missing family info
        missing_glotto = analysis_df['glottocode'].isnull().sum()
        missing_family = analysis_df['family_top'].isnull().sum()
        
        if missing_glotto > 0 or missing_family > 0:
            missing_langs = analysis_df[analysis_df['glottocode'].isnull()]['language_slug'].unique()
            print(f"    ⚠ Missing family info for {len(missing_langs)} languages: {list(missing_langs)}")
        
        join_stats.append({
            'stage': 'Language Families (PHYLOGENETIC PREDICTOR)',
            'columns_added': len(family_columns),
            'rows_before': before_rows,
            'rows_after': after_rows,
            'total_nans': missing_glotto + missing_family,
            'success': (missing_glotto + missing_family) == 0
        })
        
        # Family distribution summary
        family_dist = analysis_df['family_top'].value_counts()
        print(f"    Family distribution: {len(family_dist)} families")
        for family, count in family_dist.head(3).items():
            langs = analysis_df[analysis_df['family_top'] == family]['language_slug'].nunique()
            print(f"      {family}: {langs} languages ({count} rows)")
    
    # Join sentence aggregates (Stage 1: sentence-level covariates)
    if "sentence_aggregates" in covariates:
        sent_df = covariates["sentence_aggregates"]
        
        # Select all sentence aggregate columns except language_slug
        sent_columns = [col for col in sent_df.columns if col != 'language_slug']
        
        print(f"  Joining sentence aggregates ({len(sent_columns)} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, sent_df, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs
        nan_counts = {col: analysis_df[col].isnull().sum() for col in sent_columns}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'Sentence Aggregates (1A)',
            'columns_added': len(sent_columns),
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
    
    # Join tree height covariates (Stage 1B: derived height measures)
    if "tree_height" in covariates:
        height_df = covariates["tree_height"]
        
        # Select all height covariate columns except language_slug
        height_columns = [col for col in height_df.columns if col != 'language_slug']
        
        print(f"  Joining tree height covariates ({len(height_columns)} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, height_df, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs
        nan_counts = {col: analysis_df[col].isnull().sum() for col in height_columns}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'Tree Height Covariates (1B)',
            'columns_added': len(height_columns),
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
    
    # Join fragmentation metrics (Stage 1C: tokenization confound)
    if "fragmentation" in covariates:
        frag_df = covariates["fragmentation"]
        
        # Select all fragmentation columns except language_slug
        frag_columns = [col for col in frag_df.columns if col != 'language_slug']
        
        print(f"  Joining fragmentation metrics ({len(frag_columns)} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, frag_df, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs
        nan_counts = {col: analysis_df[col].isnull().sum() for col in frag_columns}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'Fragmentation Metrics (1C)',
            'columns_added': len(frag_columns),
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
    
    # Join UD statistics (Stage 2A: dataset characteristics)
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
            'stage': 'UD Statistics (2A)',
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
    
    # Join pretraining exposure (Stage 2B: exposure confound)
    if "pretrain_exposure" in covariates:
        exposure_df = covariates["pretrain_exposure"]
        
        # Select key columns for modeling (rename for compatibility)
        exposure_df_subset = exposure_df[['language_slug', 'wiki_size_log2_mb', 'size_mb', 'chosen_date']].copy()
        exposure_df_subset.rename(columns={'wiki_size_log2_mb': 'pretrain_exposure_log2mb'}, inplace=True)
        exposure_df_subset['exposure_source'] = 'wikidump_2018'
        
        exposure_columns = ['pretrain_exposure_log2mb', 'size_mb', 'chosen_date']
        
        print(f"  Joining pretraining exposure ({len(exposure_columns)+1} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, exposure_df_subset, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs in exposure columns
        exposure_all_cols = exposure_columns + ['exposure_source']
        nan_counts = {col: analysis_df[col].isnull().sum() for col in exposure_all_cols}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'Pretraining Exposure (2B)',
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
            print(f"    ✓ All {len(analysis_df)} rows populated")    # Join key morphological complexity columns (PRIMARY PREDICTOR)
    if "morph_complexity" in covariates:
        morph_df = covariates["morph_complexity"]
        
        # Select only the three key columns for modeling
        morph_columns = ['complexity_pc1', 'complexity_pc1_adequate_only', 'feats_coverage_train', 'feats_coverage_band']
        morph_df_subset = morph_df[['language_slug'] + morph_columns].copy()
        
        # Fix feats_coverage_train scaling (convert from percentage to fraction for compatibility)
        if 'feats_coverage_train' in morph_df_subset.columns:
            morph_df_subset['feats_coverage_train'] = morph_df_subset['feats_coverage_train'] / 100.0
        
        print(f"  Joining key morphological complexity ({len(morph_columns)} columns)...")
        before_rows = len(analysis_df)
        analysis_df = pd.merge(analysis_df, morph_df_subset, on='language_slug', how='left')
        after_rows = len(analysis_df)
        
        # Check for NaNs in morph columns
        nan_counts = {col: analysis_df[col].isnull().sum() for col in morph_columns}
        total_nans = sum(nan_counts.values())
        
        join_stats.append({
            'stage': 'Key Morphological Complexity (PRIMARY PREDICTOR)',
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

    # Join new morphological complexity (coverage-orthogonal; ADDITIVE, no behavior change)
    if "morph_complexity_new" in covariates:
        try:
            new_morph_df = covariates["morph_complexity_new"][['language_slug', 'complex_pc1']].copy()
            new_morph_df.rename(columns={'complex_pc1': 'new_complex_pc1'}, inplace=True)
            print(f"  Joining NEW morphological complexity (1 column)...")
            before_rows = len(analysis_df)
            analysis_df = pd.merge(analysis_df, new_morph_df, on='language_slug', how='left')
            after_rows = len(analysis_df)

            nan_count = analysis_df['new_complex_pc1'].isnull().sum()
            join_stats.append({
                'stage': 'New Morphological Complexity (ADDITIVE)',
                'columns_added': 1,
                'rows_before': before_rows,
                'rows_after': after_rows,
                'total_nans': int(nan_count),
                'success': True
            })
        except Exception as e:
            print(f"    ⚠ Failed to join new morphological complexity: {e}")

    # Join new morph core diagnostics (overall_marking_rate) if available
    if "morph_complexity_new_core" in covariates:
        try:
            new_core_df = covariates["morph_complexity_new_core"][['language_slug', 'overall_marking_rate']].copy()
            new_core_df.rename(columns={'overall_marking_rate': 'new_overall_marking_rate'}, inplace=True)
            print(f"  Joining NEW morph diagnostics (overall_marking_rate)...")
            before_rows = len(analysis_df)
            analysis_df = pd.merge(analysis_df, new_core_df, on='language_slug', how='left')
            after_rows = len(analysis_df)

            nan_count = analysis_df['new_overall_marking_rate'].isnull().sum()
            join_stats.append({
                'stage': 'New Morph Diagnostics (ADDITIVE)',
                'columns_added': 1,
                'rows_before': before_rows,
                'rows_after': after_rows,
                'total_nans': int(nan_count),
                'success': True
            })
        except Exception as e:
            print(f"    ⚠ Failed to join new morph diagnostics: {e}")
    
    # Reorder columns to match desired sequence
    desired_order = [
        # Base columns (1-13)
        'language_slug', 'glottocode', 'family_top', 'probe', 'layer', 'is_headline_layer', 'loss', 
        'spearman_hm', 'spearman_content', 'uuas', 'root_acc', 'n_dev_sent', 'n_test_sent',
        # Sentence-level covariates (14-20)
        'mean_content_len_test', 'median_content_len_test', 'mean_arc_length_test', 
        'mean_orig_len_test', 'mean_content_ratio_test', 'mean_num_arcs_test',
        # Fragmentation (21-22)
        'fragmentation_ratio_content_mean', 'fragmentation_ratio_overall_mean',
        # Tree height (23-26)
        'mean_tree_height_test', 'height_residual', 'height_normalized_log', 'height_normalized_linear',
        # UD dataset statistics (27-37)
        'n_train_sent', 'n_train_tokens_content', 'n_test_tokens_content', 'n_deprel_types', 'n_upos_types',
        'log_n_train_sent', 'log_n_test_sent', 'log_n_train_tokens_content', 'log_n_test_tokens_content',
        'ud_release', 'conllu_checksum_train', 'conllu_checksum_test',
        # Pretraining exposure (38-41)
        'pretrain_exposure_log2mb', 'size_mb', 'chosen_date', 'exposure_source',
        # Morphological predictors (42-44)
        'complexity_pc1', 'complexity_pc1_adequate_only', 'feats_coverage_train', 'feats_coverage_band',
        # New metric (ADDITIVE; placed after legacy predictors)
        'new_complex_pc1', 'new_overall_marking_rate'
    ]
    
    # Verify all columns are present and reorder
    available_cols = set(analysis_df.columns)
    ordered_cols = [col for col in desired_order if col in available_cols]
    missing_cols = available_cols - set(ordered_cols)
    
    if missing_cols:
        print(f"  ⚠ Unexpected columns found: {sorted(missing_cols)}")
        # Add any missing columns at the end
        ordered_cols.extend(sorted(missing_cols))
    
    analysis_df = analysis_df[ordered_cols]
    print(f"  ✓ Columns reordered to desired sequence ({len(ordered_cols)} total)")
    
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
    analysis_file = TABLES_DIR / "analysis_table_per_layer.csv"
    print(f"\nSaving analysis table: {analysis_file}")
    analysis_df.to_csv(analysis_file, index=False)
    print(f"  Shape: {analysis_df.shape}")
    print(f"  Columns: {len(analysis_df.columns)}")
    
    # Create L7 slice
    l7_df = create_l7_slice(analysis_df)
    
    # Save L7 slice
    l7_file = TABLES_DIR / "analysis_table_L7.csv"
    print(f"\nSaving L7 slice: {l7_file}")
    l7_df.to_csv(l7_file, index=False)
    print(f"  Shape: {l7_df.shape}")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS TABLES READY")
    print("="*60)
    print(f"Immutable ledger: 00_master/master_results_per_layer.csv ({master_df.shape})")
    print(f"Analysis table:   03_tables/analysis_table_per_layer.csv ({analysis_df.shape})")  
    print(f"Paper slice:      03_tables/analysis_table_L7.csv ({l7_df.shape})")
    print()
    print("Ready for modeling and visualization!")

if __name__ == "__main__":
    main()

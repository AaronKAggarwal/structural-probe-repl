#!/usr/bin/env python3
"""
Stage 6: Assemble per-layer analysis tables and L7 slice.

Creates tidy, analysis-ready tables joining probe metrics with all covariates.
Follows detailed specification for systematic joins, guards, and QC.
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
CHECKS_DIR = ANALYSIS_DIR / "checks"
SCHEMA_DIR = ANALYSIS_DIR / "schema"

# Create directories
CHECKS_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=REPO_ROOT)
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def validate_inputs() -> Dict[str, pd.DataFrame]:
    """Validate all input files exist and load them."""
    print("🔍 Validating input files...")
    
    required_files = {
        "master_results": ANALYSIS_DIR / "master_results_per_layer.csv",
        "fragmentation": ANALYSIS_DIR / "fragmentation_stats.csv", 
        "tree_shape": ANALYSIS_DIR / "tree_shape_stats.csv",
        "ud_stats": ANALYSIS_DIR / "ud_stats_derived.csv",
        "pretrain_exposure": ANALYSIS_DIR / "pretrain_exposure.csv",
        "morph_complexity": ANALYSIS_DIR / "morph_complexity.csv"
    }
    
    inputs = {}
    
    for name, file_path in required_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            inputs[name] = df
            print(f"  ✓ {name}: {df.shape}")
        else:
            raise FileNotFoundError(f"Required input missing: {file_path}")
    
    # Validate master table structure
    master = inputs["master_results"]
    expected_rows = 23 * 2 * 6  # languages × probes × layers
    if len(master) != expected_rows:
        print(f"  ⚠ Master table has {len(master)} rows, expected {expected_rows}")
    
    # Check unique combinations
    key_cols = ['language_slug', 'probe', 'layer']
    if not master[key_cols].duplicated().any():
        print(f"  ✓ Master table has unique (language_slug, probe, layer) combinations")
    else:
        duplicates = master[master[key_cols].duplicated()]
        print(f"  ⚠ Master table has {len(duplicates)} duplicate keys")
    
    return inputs

def execute_join_plan(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Execute systematic join plan on language_slug."""
    print("🔗 Executing join plan...")
    
    # Start from master results (276 rows)
    df = inputs["master_results"].copy()
    print(f"  Starting with master: {df.shape}")
    
    # Add derived columns
    df['is_headline_layer'] = (df['layer'] == 'L7')
    df['layer_index'] = df['layer'].str.replace('L', '').astype(int)
    print(f"  ✓ Added is_headline_layer and layer_index")
    
    # Join fragmentation (test)
    frag_df = inputs["fragmentation"]
    frag_cols = ['fragmentation_ratio_content_mean', 'fragmentation_ratio_overall_mean']
    frag_subset = frag_df[['language_slug'] + frag_cols]
    
    df = pd.merge(df, frag_subset, on='language_slug', how='left')
    print(f"  ✓ Joined fragmentation: {df.shape}")
    
    # Join tree shape (test)
    tree_df = inputs["tree_shape"]
    tree_cols = ['mean_content_len_test', 'median_content_len_test', 'mean_arc_length_test', 'mean_tree_height_test']
    
    # Check for optional height covariates
    optional_height_cols = ['height_residual', 'height_normalized_log', 'height_normalized_linear']
    for col in optional_height_cols:
        if col in tree_df.columns:
            tree_cols.append(col)
    
    tree_subset = tree_df[['language_slug'] + tree_cols]
    df = pd.merge(df, tree_subset, on='language_slug', how='left')
    print(f"  ✓ Joined tree shape: {df.shape}")
    
    # Join UD sizes/inventories
    ud_df = inputs["ud_stats"]
    ud_cols = ['n_train_sent', 'n_test_sent', 'n_train_tokens_content', 'n_test_tokens_content',
               'log_n_train_sent', 'log_n_test_sent', 'n_deprel_types', 'n_upos_types']
    ud_subset = ud_df[['language_slug'] + ud_cols]
    
    df = pd.merge(df, ud_subset, on='language_slug', how='left')
    print(f"  ✓ Joined UD statistics: {df.shape}")
    
    # Join pretraining exposure
    exposure_df = inputs["pretrain_exposure"]
    exposure_cols = ['wiki_code', 'wiki_size_log2_mb', 'chosen_date', 'source']
    
    # Check which columns exist
    available_exposure_cols = [col for col in exposure_cols if col in exposure_df.columns]
    exposure_subset = exposure_df[['language_slug'] + available_exposure_cols]
    
    # Rename for clarity
    exposure_subset = exposure_subset.rename(columns={'wiki_size_log2_mb': 'pretrain_exposure_log2mb'})
    
    df = pd.merge(df, exposure_subset, on='language_slug', how='left')
    print(f"  ✓ Joined pretraining exposure: {df.shape}")
    
    # Join morphology
    morph_df = inputs["morph_complexity"]
    morph_cols = ['complexity_pc1', 'feats_per_token_mean', 'feats_bundle_entropy_bits', 
                  'feats_bundles_per_10k', 'feats_coverage_train', 'feats_coverage_band']
    
    # Check which columns exist
    available_morph_cols = [col for col in morph_cols if col in morph_df.columns]
    morph_subset = morph_df[['language_slug'] + available_morph_cols]
    
    df = pd.merge(df, morph_subset, on='language_slug', how='left')
    print(f"  ✓ Joined morphology: {df.shape}")
    
    print(f"  Final table: {df.shape}")
    return df

def apply_data_hygiene_guards(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply data hygiene guards and range checks."""
    print("🧹 Applying data hygiene guards...")
    
    issues = []
    
    # 1. Uniqueness check
    key_cols = ['language_slug', 'probe', 'layer']
    if df[key_cols].duplicated().any():
        dup_count = df[key_cols].duplicated().sum()
        issues.append(f"Non-unique keys: {dup_count} duplicates")
    
    # 2. Row count check
    expected_rows = 276
    if len(df) != expected_rows:
        issues.append(f"Row count: {len(df)} != {expected_rows}")
    
    # 3. Range checks
    range_checks = {
        'uuas': (0, 1),
        'root_acc': (0, 1),
        'spearman_hm': (0, 1),
        'spearman_content': (0, 1),
        'fragmentation_ratio_content_mean': (1.0, 10.0),
        'mean_arc_length_test': (1.0, 20.0),
        'mean_tree_height_test': (1.0, 50.0),
        'pretrain_exposure_log2mb': (6.0, 15.0),
        'feats_coverage_train': (0.0, 1.0)
    }
    
    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns:
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                n_out = out_of_range.sum()
                actual_min = df[col].min()
                actual_max = df[col].max()
                issues.append(f"{col}: {n_out} values out of range [{min_val}, {max_val}], actual range [{actual_min:.3f}, {actual_max:.3f}]")
    
    # 4. Missingness check
    critical_cols = ['language_slug', 'probe', 'layer', 'uuas', 'root_acc', 
                     'complexity_pc1', 'feats_coverage_band']
    
    for col in critical_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                issues.append(f"{col}: {missing_count} missing values")
    
    # 5. Special checks
    # Check fragmentation ratio >= 1.0
    if 'fragmentation_ratio_content_mean' in df.columns:
        suspicious = df['fragmentation_ratio_content_mean'] < 1.0
        if suspicious.any():
            issues.append(f"Suspicious fragmentation: {suspicious.sum()} values < 1.0")
    
    # Report findings
    if issues:
        print("  ⚠ Issues found:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  ✓ All data hygiene checks passed")
    
    # Create summary
    hygiene_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'issues': issues,
        'unique_languages': df['language_slug'].nunique(),
        'unique_probes': df['probe'].nunique(),
        'unique_layers': df['layer'].nunique()
    }
    
    return df, hygiene_report

def create_slices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create L7 and L7-Adequate slices."""
    print("✂️ Creating analysis slices...")
    
    slices = {}
    
    # L7 slice (all languages)
    l7_df = df[df['layer'] == 'L7'].copy()
    slices['L7'] = l7_df
    print(f"  ✓ L7 slice: {l7_df.shape}")
    
    # L7 Adequate-coverage slice (primary modeling)
    if 'feats_coverage_band' in df.columns:
        l7_adequate = l7_df[l7_df['feats_coverage_band'] == 'Adequate'].copy()
        slices['L7_adequate'] = l7_adequate
        print(f"  ✓ L7 Adequate slice: {l7_adequate.shape}")
        
        # Report coverage distribution
        if 'feats_coverage_band' in l7_df.columns:
            coverage_dist = l7_df['feats_coverage_band'].value_counts()
            print(f"    Coverage distribution: {dict(coverage_dist)}")
    else:
        print("  ⚠ feats_coverage_band not found, skipping Adequate slice")
    
    return slices

def generate_qc_report(df: pd.DataFrame, slices: Dict[str, pd.DataFrame], hygiene_report: Dict) -> Dict:
    """Generate comprehensive QC report."""
    print("📊 Generating QC report...")
    
    # Basic counts
    qc_report = {
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_commit(),
        'ud_release': 'UD_2.x',
        'mbert_model': 'bert-base-multilingual-cased',
        
        'row_counts': {
            'per_layer_total': len(df),
            'l7_total': len(slices.get('L7', [])),
            'l7_adequate': len(slices.get('L7_adequate', []))
        },
        
        'row_counts_by_probe_layer': {},
        'missingness_report': {},
        'range_report': {},
        'hygiene_issues': hygiene_report['issues'],
        'sanity_correlations': {}
    }
    
    # Row counts per probe/layer
    probe_layer_counts = df.groupby(['probe', 'layer']).size().to_dict()
    qc_report['row_counts_by_probe_layer'] = {f"{p}_{l}": count for (p, l), count in probe_layer_counts.items()}
    
    # Missingness report
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            qc_report['missingness_report'][col] = missing_count
    
    # Range report for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not df[col].isnull().all():
            qc_report['range_report'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    # Sanity correlations on L7 slice
    if 'L7' in slices:
        l7_df = slices['L7']
        
        # complexity_pc1 vs feats_per_token_mean
        if all(col in l7_df.columns for col in ['complexity_pc1', 'feats_per_token_mean']):
            corr1 = l7_df['complexity_pc1'].corr(l7_df['feats_per_token_mean'])
            qc_report['sanity_correlations']['complexity_pc1_vs_feats_per_token'] = float(corr1)
        
        # feats_coverage_train vs complexity_pc1  
        if all(col in l7_df.columns for col in ['feats_coverage_train', 'complexity_pc1']):
            corr2 = l7_df['feats_coverage_train'].corr(l7_df['complexity_pc1'])
            qc_report['sanity_correlations']['coverage_vs_complexity_pc1'] = float(corr2)
    
    # FEATS coverage bands
    if 'feats_coverage_band' in df.columns:
        coverage_bands = df['feats_coverage_band'].value_counts().to_dict()
        qc_report['feats_coverage_bands'] = coverage_bands
        
        # List languages by band
        band_languages = {}
        for band in ['Adequate', 'Sparse', 'Absent']:
            band_langs = df[df['feats_coverage_band'] == band]['language_slug'].unique().tolist()
            if band_langs:
                band_languages[band] = sorted(band_langs)
        qc_report['languages_by_coverage_band'] = band_languages
    
    return qc_report

def create_schema(df: pd.DataFrame) -> Dict:
    """Create schema documentation."""
    print("📋 Creating schema documentation...")
    
    schema = {
        'table_name': 'analysis_table_per_layer',
        'description': 'Tidy analysis table joining probe metrics with all covariates',
        'created_utc': datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_commit(),
        'shape': list(df.shape),
        
        'columns': {},
        'key_columns': ['language_slug', 'probe', 'layer'],
        'data_sources': {
            'probe_metrics': 'master_results_per_layer.csv',
            'fragmentation': 'fragmentation_stats.csv',
            'tree_shape': 'tree_shape_stats.csv', 
            'ud_stats': 'ud_stats_derived.csv',
            'pretrain_exposure': 'pretrain_exposure.csv',
            'morph_complexity': 'morph_complexity.csv'
        }
    }
    
    # Document columns
    column_descriptions = {
        # Identifiers
        'language_slug': 'UD treebank identifier',
        'probe': 'Probe type (dist/depth)',
        'layer': 'Transformer layer (L5-L10)',
        'layer_index': 'Numeric layer index (5-10)',
        'is_headline_layer': 'True if layer == L7',
        
        # Probe metrics
        'uuas': 'Unlabeled undirected attachment score (distance probe)',
        'root_acc': 'Root accuracy (depth probe)',
        'spearman_hm': 'Spearman correlation (head-modifier)',
        'spearman_content': 'Spearman correlation (content-only)',
        'loss': 'Probe training loss',
        
        # Sample sizes
        'n_dev_sent': 'Dev set sentence count',
        'n_test_sent': 'Test set sentence count',
        
        # Fragmentation
        'fragmentation_ratio_content_mean': 'Subwords per content word (primary)',
        'fragmentation_ratio_overall_mean': 'Subwords per word overall (diagnostic)',
        
        # Tree shape
        'mean_content_len_test': 'Mean content tokens per sentence',
        'median_content_len_test': 'Median content tokens per sentence',
        'mean_arc_length_test': 'Mean dependency arc length',
        'mean_tree_height_test': 'Mean dependency tree height',
        
        # UD statistics
        'n_train_sent': 'Training sentences',
        'n_test_sent': 'Test sentences',
        'log_n_train_sent': 'Log training sentences',
        'log_n_test_sent': 'Log test sentences',
        'n_deprel_types': 'DEPREL inventory size',
        'n_upos_types': 'UPOS inventory size',
        
        # Pretraining exposure
        'wiki_code': 'Wikipedia language code',
        'pretrain_exposure_log2mb': 'Log2 Wikipedia dump size (MB)',
        'chosen_date': 'Wikipedia dump date',
        'source': 'Dump source (wikimedia/archive_org)',
        
        # Morphology
        'complexity_pc1': 'Morphological complexity (PC1)',
        'feats_per_token_mean': 'Mean FEATS per token',
        'feats_bundle_entropy_bits': 'FEATS bundle entropy',
        'feats_bundles_per_10k': 'FEATS bundle types per 10k',
        'feats_coverage_train': 'FEATS coverage (0-1)',
        'feats_coverage_band': 'Coverage band (Adequate/Sparse/Absent)'
    }
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        description = column_descriptions.get(col, 'No description available')
        
        schema['columns'][col] = {
            'dtype': dtype,
            'description': description,
            'null_count': int(df[col].isnull().sum())
        }
    
    return schema

def main():
    """Main Stage 6 execution."""
    print("STAGE 6: ASSEMBLE ANALYSIS TABLES")
    print("=" * 50)
    print("Objective: Create tidy analysis-ready tables with all covariates")
    print()
    
    # 1. Validate inputs
    inputs = validate_inputs()
    
    # 2. Execute join plan  
    df = execute_join_plan(inputs)
    
    # 3. Apply data hygiene
    df, hygiene_report = apply_data_hygiene_guards(df)
    
    # 4. Create slices
    slices = create_slices(df)
    
    # 5. Generate QC and schema
    qc_report = generate_qc_report(df, slices, hygiene_report)
    schema = create_schema(df)
    
    # 6. Save outputs
    print("\n💾 Saving outputs...")
    
    # Main analysis table
    df.to_csv(ANALYSIS_DIR / "analysis_table_per_layer.csv", index=False)
    print(f"  ✓ analysis_table_per_layer.csv: {df.shape}")
    
    # L7 slice
    if 'L7' in slices:
        slices['L7'].to_csv(ANALYSIS_DIR / "analysis_table_L7.csv", index=False)
        print(f"  ✓ analysis_table_L7.csv: {slices['L7'].shape}")
    
    # L7 Adequate slice
    if 'L7_adequate' in slices:
        slices['L7_adequate'].to_csv(ANALYSIS_DIR / "analysis_table_L7_adequate.csv", index=False)
        print(f"  ✓ analysis_table_L7_adequate.csv: {slices['L7_adequate'].shape}")
    
    # QC report
    with open(CHECKS_DIR / "analysis_table_qc.json", 'w') as f:
        json.dump(qc_report, f, indent=2)
    print(f"  ✓ QC report: {CHECKS_DIR / 'analysis_table_qc.json'}")
    
    # Schema
    with open(SCHEMA_DIR / "analysis_table_schema.json", 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"  ✓ Schema: {SCHEMA_DIR / 'analysis_table_schema.json'}")
    
    # 7. Final validation
    print("\n✅ ACCEPTANCE CRITERIA VALIDATION")
    print("-" * 40)
    
    # File existence and shapes
    files_ok = True
    expected_files = [
        (ANALYSIS_DIR / "analysis_table_per_layer.csv", (276, "≥20")),
        (ANALYSIS_DIR / "analysis_table_L7.csv", (46, "≥20")),
    ]
    
    if 'L7_adequate' in slices:
        n_adequate = len(slices['L7_adequate'])
        expected_files.append((ANALYSIS_DIR / "analysis_table_L7_adequate.csv", (n_adequate, "≥20")))
    
    for file_path, expected_shape in expected_files:
        if file_path.exists():
            actual_shape = pd.read_csv(file_path).shape
            shape_ok = (actual_shape[0] == expected_shape[0] if isinstance(expected_shape[0], int) 
                       else True) and actual_shape[1] >= 20
            status = "✓" if shape_ok else "⚠"
            print(f"  {status} {file_path.name}: {actual_shape}")
        else:
            print(f"  ✗ {file_path.name}: MISSING")
            files_ok = False
    
    # QC checks
    issues_count = len(hygiene_report['issues'])
    qc_ok = issues_count == 0
    print(f"  {'✓' if qc_ok else '⚠'} Data hygiene: {issues_count} issues")
    
    # Critical correlations
    correlations = qc_report.get('sanity_correlations', {})
    corr_ok = True
    if 'complexity_pc1_vs_feats_per_token' in correlations:
        corr_val = correlations['complexity_pc1_vs_feats_per_token']
        corr_ok = corr_ok and corr_val > 0.8
        print(f"  {'✓' if corr_val > 0.8 else '⚠'} Complexity-FEATS correlation: {corr_val:.3f}")
    
    # Summary
    all_ok = files_ok and qc_ok and corr_ok
    print(f"\n🎯 STAGE 6 STATUS: {'COMPLETE ✓' if all_ok else 'ISSUES DETECTED ⚠'}")
    
    if all_ok:
        print("\nReady for downstream stages:")
        print("  • Stage 7/8: Matched evaluation (append to analysis tables)")
        print("  • Stage 9: Length-specific metrics (append to analysis tables)")
        print("  • Stage 11: Figures (read from analysis_table_L7.csv)")
        print("  • Stage 12: Primary models (use analysis_table_L7_adequate.csv)")

if __name__ == "__main__":
    main()

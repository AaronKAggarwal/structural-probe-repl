#!/usr/bin/env python3
"""
Stage 6 Validation: Check existing analysis tables against Stage 6 requirements.

The analysis tables were already built in previous stages. This script validates
they meet Stage 6 specifications and adds any missing Stage 6-specific features.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess

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

def validate_existing_tables():
    """Validate existing analysis tables meet Stage 6 requirements."""
    print("🔍 Validating existing analysis tables...")
    
    # Load existing tables
    per_layer_file = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    l7_file = ANALYSIS_DIR / "analysis_table_L7.csv"
    
    if not per_layer_file.exists():
        raise FileNotFoundError(f"analysis_table_per_layer.csv not found")
    if not l7_file.exists():
        raise FileNotFoundError(f"analysis_table_L7.csv not found")
    
    df_per_layer = pd.read_csv(per_layer_file)
    df_l7 = pd.read_csv(l7_file)
    
    print(f"  ✓ Per-layer table: {df_per_layer.shape}")
    print(f"  ✓ L7 table: {df_l7.shape}")
    
    # Check required Stage 6 features
    stage6_requirements = {
        'layer_index': 'Numeric layer index (5-10)',
        'is_headline_layer': 'True if layer == L7',
        'pretrain_exposure_log2mb': 'Renamed WikiSize for clarity'
    }
    
    missing_features = []
    for feature, description in stage6_requirements.items():
        if feature not in df_per_layer.columns:
            missing_features.append((feature, description))
    
    # Add missing Stage 6 features
    if missing_features:
        print(f"  📝 Adding {len(missing_features)} missing Stage 6 features...")
        
        # Add layer_index
        if 'layer_index' not in df_per_layer.columns:
            df_per_layer['layer_index'] = df_per_layer['layer'].str.replace('L', '').astype(int)
            print(f"    ✓ Added layer_index")
        
        # Add is_headline_layer
        if 'is_headline_layer' not in df_per_layer.columns:
            df_per_layer['is_headline_layer'] = (df_per_layer['layer'] == 'L7')
            print(f"    ✓ Added is_headline_layer")
        
        # Rename wiki_size_log2_mb to pretrain_exposure_log2mb for clarity
        if 'pretrain_exposure_log2mb' not in df_per_layer.columns and 'wiki_size_log2_mb' in df_per_layer.columns:
            df_per_layer['pretrain_exposure_log2mb'] = df_per_layer['wiki_size_log2_mb']
            print(f"    ✓ Added pretrain_exposure_log2mb")
        
        # Update L7 table with same features
        if 'layer_index' not in df_l7.columns:
            df_l7['layer_index'] = df_l7['layer'].str.replace('L', '').astype(int)
        if 'is_headline_layer' not in df_l7.columns:
            df_l7['is_headline_layer'] = (df_l7['layer'] == 'L7')
        if 'pretrain_exposure_log2mb' not in df_l7.columns and 'wiki_size_log2_mb' in df_l7.columns:
            df_l7['pretrain_exposure_log2mb'] = df_l7['wiki_size_log2_mb']
        
        # Save updated tables
        df_per_layer.to_csv(per_layer_file, index=False)
        df_l7.to_csv(l7_file, index=False)
        print(f"    ✓ Updated analysis tables")
    
    return df_per_layer, df_l7

def create_l7_adequate_slice(df_l7):
    """Create L7 Adequate-coverage slice for primary modeling."""
    print("✂️ Creating L7 Adequate slice...")
    
    if 'feats_coverage_band' not in df_l7.columns:
        print("  ⚠ feats_coverage_band not found, cannot create Adequate slice")
        return None
    
    l7_adequate = df_l7[df_l7['feats_coverage_band'] == 'Adequate'].copy()
    
    # Save L7 Adequate slice
    adequate_file = ANALYSIS_DIR / "analysis_table_L7_adequate.csv"
    l7_adequate.to_csv(adequate_file, index=False)
    
    print(f"  ✓ L7 Adequate slice: {l7_adequate.shape}")
    
    # Report coverage distribution
    coverage_dist = df_l7['feats_coverage_band'].value_counts()
    print(f"  📊 Coverage distribution: {dict(coverage_dist)}")
    
    return l7_adequate

def validate_stage6_requirements(df_per_layer, df_l7, df_l7_adequate):
    """Validate Stage 6 acceptance criteria."""
    print("✅ Validating Stage 6 acceptance criteria...")
    
    issues = []
    
    # 1. Uniqueness check
    key_cols = ['language_slug', 'probe', 'layer']
    if df_per_layer[key_cols].duplicated().any():
        dup_count = df_per_layer[key_cols].duplicated().sum()
        issues.append(f"Non-unique keys: {dup_count} duplicates")
    
    # 2. Row count check
    expected_rows = 276
    if len(df_per_layer) != expected_rows:
        issues.append(f"Row count: {len(df_per_layer)} != {expected_rows}")
    
    # 3. L7 slice check
    expected_l7_rows = 46
    if len(df_l7) != expected_l7_rows:
        issues.append(f"L7 row count: {len(df_l7)} != {expected_l7_rows}")
    
    # 4. Range checks for critical metrics
    range_checks = {
        'uuas': (0, 1),
        'root_acc': (0, 1),
        'spearman_content': (0, 1),
        'fragmentation_ratio_content_mean': (1.0, 10.0),
        'pretrain_exposure_log2mb': (6.0, 15.0),
        'feats_coverage_train': (0.0, 1.0)
    }
    
    for col, (min_val, max_val) in range_checks.items():
        if col in df_per_layer.columns:
            out_of_range = (df_per_layer[col] < min_val) | (df_per_layer[col] > max_val)
            if out_of_range.any():
                n_out = out_of_range.sum()
                issues.append(f"{col}: {n_out} values out of range [{min_val}, {max_val}]")
    
    # 5. Critical column presence
    required_cols = [
        'language_slug', 'probe', 'layer', 'layer_index', 'is_headline_layer',
        'uuas', 'root_acc', 'complexity_pc1', 'feats_coverage_band',
        'fragmentation_ratio_content_mean', 'pretrain_exposure_log2mb'
    ]
    
    missing_cols = [col for col in required_cols if col not in df_per_layer.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # 6. Coverage band validation
    if 'feats_coverage_band' in df_per_layer.columns:
        valid_bands = {'Adequate', 'Sparse', 'Absent'}
        invalid_bands = set(df_per_layer['feats_coverage_band'].unique()) - valid_bands
        if invalid_bands:
            issues.append(f"Invalid coverage bands: {invalid_bands}")
    
    return issues

def generate_comprehensive_qc(df_per_layer, df_l7, df_l7_adequate, validation_issues):
    """Generate comprehensive QC report."""
    print("📊 Generating comprehensive QC report...")
    
    qc_report = {
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_commit(),
        'stage': 'Stage 6 - Analysis Tables Assembly',
        'ud_release': 'UD_2.x',
        'mbert_model': 'bert-base-multilingual-cased',
        
        'table_shapes': {
            'per_layer': list(df_per_layer.shape),
            'l7_all': list(df_l7.shape),
            'l7_adequate': list(df_l7_adequate.shape) if df_l7_adequate is not None else None
        },
        
        'validation_issues': validation_issues,
        'row_counts_by_probe_layer': {},
        'coverage_analysis': {},
        'sanity_correlations': {},
        'column_summary': {}
    }
    
    # Row counts per probe/layer (should all be 23)
    probe_layer_counts = df_per_layer.groupby(['probe', 'layer']).size()
    qc_report['row_counts_by_probe_layer'] = {f"{probe}_{layer}": count for (probe, layer), count in probe_layer_counts.items()}
    
    # Coverage analysis
    if 'feats_coverage_band' in df_per_layer.columns:
        coverage_dist = df_per_layer['feats_coverage_band'].value_counts()
        qc_report['coverage_analysis']['band_distribution'] = coverage_dist.to_dict()
        
        # Languages by band
        languages_by_band = {}
        for band in ['Adequate', 'Sparse', 'Absent']:
            band_langs = df_per_layer[df_per_layer['feats_coverage_band'] == band]['language_slug'].unique()
            languages_by_band[band] = sorted(band_langs.tolist())
        qc_report['coverage_analysis']['languages_by_band'] = languages_by_band
    
    # Sanity correlations on L7
    correlations = {}
    
    # complexity_pc1 vs feats_per_token_mean (should be strong +)
    if all(col in df_l7.columns for col in ['complexity_pc1', 'feats_per_token_mean']):
        corr1 = df_l7['complexity_pc1'].corr(df_l7['feats_per_token_mean'])
        correlations['complexity_vs_feats_per_token'] = float(corr1)
    
    # feats_coverage vs complexity_pc1 (should be weak/moderate +)
    if all(col in df_l7.columns for col in ['feats_coverage_train', 'complexity_pc1']):
        corr2 = df_l7['feats_coverage_train'].corr(df_l7['complexity_pc1'])
        correlations['coverage_vs_complexity'] = float(corr2)
    
    # UUAS vs fragmentation (should be negative)
    if all(col in df_l7.columns for col in ['uuas', 'fragmentation_ratio_content_mean']):
        corr3 = df_l7['uuas'].corr(df_l7['fragmentation_ratio_content_mean'])
        correlations['uuas_vs_fragmentation'] = float(corr3)
    
    qc_report['sanity_correlations'] = correlations
    
    # Column summary
    numeric_cols = df_per_layer.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not df_per_layer[col].isnull().all():
            qc_report['column_summary'][col] = {
                'min': float(df_per_layer[col].min()),
                'max': float(df_per_layer[col].max()),
                'mean': float(df_per_layer[col].mean()),
                'missing_count': int(df_per_layer[col].isnull().sum())
            }
    
    return qc_report

def create_schema_documentation(df_per_layer):
    """Create comprehensive schema documentation."""
    print("📋 Creating schema documentation...")
    
    schema = {
        'table_name': 'analysis_table_per_layer',
        'description': 'Comprehensive analysis table joining probe metrics with all covariates',
        'created_utc': datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_commit(),
        'stage': 'Stage 6 - Analysis Tables Assembly',
        'shape': list(df_per_layer.shape),
        
        'key_columns': ['language_slug', 'probe', 'layer'],
        'headline_layer': 'L7',
        'coverage_policy': 'All languages in per-layer; L7-Adequate subset for primary modeling',
        
        'data_sources': {
            'probe_metrics': 'Stages 1-3: master_results_per_layer.csv',
            'fragmentation': 'Stage 2: sentence-level fragmentation ratios',
            'tree_shape': 'Stage 3: content-only length, arc length, height',
            'ud_statistics': 'Stage 4A: sentence/token counts, UPOS/DEPREL inventories',
            'pretraining_exposure': 'Stage 4B: Wikipedia dump sizes (2018)',
            'morphology': 'Stage 5: PC1 complexity + FEATS coverage'
        },
        
        'columns': {},
        'coverage_bands': {
            'Adequate': '≥10% FEATS coverage (primary analysis)',
            'Sparse': '1-10% FEATS coverage',
            'Absent': '<1% FEATS coverage (excluded from primary)'
        }
    }
    
    # Column documentation
    column_docs = {
        # Identifiers
        'language_slug': 'UD treebank identifier (e.g., UD_English-EWT)',
        'probe': 'Structural probe type (dist=distance, depth=depth)',
        'layer': 'Transformer layer (L5, L6, L7, L8, L9, L10)',
        'layer_index': 'Numeric layer index (5, 6, 7, 8, 9, 10)',
        'is_headline_layer': 'Boolean: True if layer == L7 (headline layer)',
        
        # Probe metrics (dependent variables)
        'uuas': 'Unlabeled Undirected Attachment Score (distance probe)',
        'root_acc': 'Root identification accuracy (depth probe)',
        'spearman_hm': 'Spearman correlation head-modifier pairs',
        'spearman_content': 'Spearman correlation content-only tokens',
        'loss': 'Probe training loss',
        
        # Sample sizes
        'n_dev_sent': 'Development set sentence count',
        'n_test_sent': 'Test set sentence count',
        
        # Fragmentation (Stage 2)
        'fragmentation_ratio_content_mean': 'Subwords per content word (primary)',
        'fragmentation_ratio_overall_mean': 'Subwords per word overall (diagnostic)',
        
        # Tree shape (Stage 3)
        'mean_content_len_test': 'Mean content tokens per sentence (test)',
        'median_content_len_test': 'Median content tokens per sentence (test)',
        'mean_arc_length_test': 'Mean dependency arc length (test)',
        'mean_tree_height_test': 'Mean dependency tree height (test)',
        'height_residual': 'Tree height residualized on sentence length',
        'height_normalized_log': 'Tree height / log₂(n_content + 1)',
        'height_normalized_linear': 'Tree height / n_content',
        
        # UD statistics (Stage 4A)
        'n_train_sent': 'Training sentences',
        'n_test_sent': 'Test sentences', 
        'n_train_tokens_content': 'Training content tokens',
        'n_test_tokens_content': 'Test content tokens',
        'log_n_train_sent': 'Log training sentences',
        'log_n_test_sent': 'Log test sentences',
        'n_deprel_types': 'DEPREL type inventory size',
        'n_upos_types': 'UPOS type inventory size',
        
        # Pretraining exposure (Stage 4B)
        'wiki_code': 'Wikipedia language code',
        'pretrain_exposure_log2mb': 'Wikipedia dump size log₂(MB) - 2018',
        'wiki_size_log2_mb': 'Alias for pretrain_exposure_log2mb',
        'chosen_date': 'Wikipedia dump date used',
        'exposure_source': 'Dump source (wikimedia/archive_org)',
        
        # Morphology (Stage 5)
        'complexity_pc1': 'Morphological complexity PC1 (train split)',
        'feats_per_token_mean': 'Mean FEATS attributes per content token',
        'feats_bundle_entropy_bits': 'Shannon entropy of FEATS bundles',
        'feats_bundles_per_10k': 'Unique FEATS bundle types per 10k tokens',
        'feats_coverage_train': 'Fraction of content tokens with FEATS ≠ "_"',
        'feats_coverage_band': 'Coverage band: Adequate/Sparse/Absent'
    }
    
    for col in df_per_layer.columns:
        dtype = str(df_per_layer[col].dtype)
        description = column_docs.get(col, 'No description available')
        
        schema['columns'][col] = {
            'dtype': dtype,
            'description': description,
            'null_count': int(df_per_layer[col].isnull().sum()),
            'unique_count': int(df_per_layer[col].nunique()) if df_per_layer[col].dtype == 'object' else None
        }
    
    return schema

def main():
    """Main Stage 6 validation and completion."""
    print("STAGE 6: ANALYSIS TABLES VALIDATION & COMPLETION")
    print("=" * 60)
    print("Note: Analysis tables already exist from previous stages")
    print("Validating against Stage 6 requirements and adding missing features")
    print()
    
    # 1. Validate and enhance existing tables
    df_per_layer, df_l7 = validate_existing_tables()
    
    # 2. Create L7 Adequate slice
    df_l7_adequate = create_l7_adequate_slice(df_l7)
    
    # 3. Validate Stage 6 requirements
    validation_issues = validate_stage6_requirements(df_per_layer, df_l7, df_l7_adequate)
    
    # 4. Generate comprehensive QC
    qc_report = generate_comprehensive_qc(df_per_layer, df_l7, df_l7_adequate, validation_issues)
    
    # 5. Create schema documentation
    schema = create_schema_documentation(df_per_layer)
    
    # 6. Save QC and schema
    with open(CHECKS_DIR / "analysis_table_qc.json", 'w') as f:
        json.dump(qc_report, f, indent=2)
    print(f"  ✓ QC report: {CHECKS_DIR / 'analysis_table_qc.json'}")
    
    with open(SCHEMA_DIR / "analysis_table_schema.json", 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"  ✓ Schema: {SCHEMA_DIR / 'analysis_table_schema.json'}")
    
    # 7. Final acceptance criteria validation
    print("\n✅ STAGE 6 ACCEPTANCE CRITERIA")
    print("-" * 40)
    
    # File shapes
    print(f"  ✓ analysis_table_per_layer.csv: {df_per_layer.shape}")
    print(f"  ✓ analysis_table_L7.csv: {df_l7.shape}")
    if df_l7_adequate is not None:
        print(f"  ✓ analysis_table_L7_adequate.csv: {df_l7_adequate.shape}")
    
    # Validation issues
    if validation_issues:
        print(f"  ⚠ Validation issues: {len(validation_issues)}")
        for issue in validation_issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ All validation checks passed")
    
    # Key correlations
    correlations = qc_report.get('sanity_correlations', {})
    if 'complexity_vs_feats_per_token' in correlations:
        corr_val = correlations['complexity_vs_feats_per_token']
        print(f"  {'✓' if corr_val > 0.8 else '⚠'} Complexity-FEATS correlation: {corr_val:.3f}")
    
    # Coverage analysis
    if 'coverage_analysis' in qc_report:
        coverage_data = qc_report['coverage_analysis']
        if 'band_distribution' in coverage_data:
            adequate_count = coverage_data['band_distribution'].get('Adequate', 0)
            print(f"  ✓ Adequate-coverage languages: {adequate_count}")
    
    # Final status
    all_good = len(validation_issues) == 0
    print(f"\n🎯 STAGE 6 STATUS: {'COMPLETE ✓' if all_good else 'ISSUES DETECTED ⚠'}")
    
    if all_good:
        print("\n📊 ANALYSIS TABLES READY FOR DOWNSTREAM STAGES:")
        print("  • Stage 7/8: Matched evaluation (append columns)")
        print("  • Stage 9: Length-specific metrics (append columns)")
        print("  • Stage 11: Figures (use analysis_table_L7.csv)")
        print("  • Stage 12: Primary models (use analysis_table_L7_adequate.csv)")
        print("  • Stage 13: Mediation analysis (complexity_pc1 → fragmentation → performance)")

if __name__ == "__main__":
    main()

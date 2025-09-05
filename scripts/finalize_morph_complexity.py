#!/usr/bin/env python3
"""
Finalize morphological complexity Stage 5 with provenance, QC, and exports.

Generates PCA provenance JSON, QC panel, face-validity plots, and tidy export.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures" / "morph"
FIGURE_DATA_DIR = REPO_ROOT / "outputs" / "analysis" / "figure_data"

# Create directories
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)

def freeze_pca_provenance():
    """Create PCA provenance JSON with exact reproduction details."""
    print("🔒 Freezing PCA provenance...")
    
    # Load morphological complexity data
    df = pd.read_csv(OUTPUT_DIR / "morph_complexity.csv")
    
    # Recreate PCA for provenance capture
    feature_matrix = df[['feats_per_token_mean', 'feats_bundle_entropy_bits', 'feats_bundles_per_10k']].values
    
    # Z-score normalization
    scaler = StandardScaler()
    z_features = scaler.fit_transform(feature_matrix)
    
    # PCA
    pca = PCA(n_components=3)
    pca_scores = pca.fit_transform(z_features)
    
    # Sign alignment check
    pc1_raw = pca_scores[:, 0]
    corr_with_feats = np.corrcoef(pc1_raw, z_features[:, 0])[0, 1]
    sign_flipped = corr_with_feats < 0
    
    if sign_flipped:
        pc1_raw = -pc1_raw
        pca.components_[0] = -pca.components_[0]
    
    # Create provenance record
    provenance = {
        "script_name": "compute_morphological_complexity.py",
        "commit_hash": "cd33433c718c7621a11d71c3f43904e35c3dae98",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "n_languages": len(df),
        
        "data_invariants": {
            "split_used": "train",
            "content_only": True,
            "upos_excluded": ["PUNCT", "SYM"],
            "mwt_excluded": True,
            "empty_nodes_excluded": True,
            "entropy_base": 2,
            "bundle_canonicalization": "sorted_keys_and_values",
            "rates_denominator": "content_tokens"
        },
        
        "features": {
            "feats_per_token_mean": "Average number of FEATS keys per content token",
            "feats_bundle_entropy_bits": "Shannon entropy (base 2) of canonical FEATS bundles", 
            "feats_bundles_per_10k": "Unique FEATS bundle types per 10,000 content tokens"
        },
        
        "pca_details": {
            "n_components": 3,
            "sign_alignment": "PC1 positively correlated with feats_per_token_mean",
            "sign_flipped": bool(sign_flipped),
            "scaler_mean": [float(x) for x in scaler.mean_],
            "scaler_scale": [float(x) for x in scaler.scale_],
            "pca_components": [[float(x) for x in row] for row in pca.components_],
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
            "total_explained_variance": float(np.sum(pca.explained_variance_ratio_))
        },
        
        "coverage_bands": {
            "adequate_threshold": 10.0,
            "sparse_threshold": 1.0,
            "adequate_languages": len(df[df['feats_coverage_band'] == 'Adequate']),
            "sparse_languages": len(df[df['feats_coverage_band'] == 'Sparse']),
            "absent_languages": len(df[df['feats_coverage_band'] == 'Absent'])
        },
        
        "bootstrap_details": {
            "bootstrap_B": 1000,
            "bootstrap_seed": 42,
            "ci_percentiles": [2.5, 97.5]
        }
    }
    
    # Save provenance
    provenance_file = OUTPUT_DIR / "morph_complexity_provenance.json"
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"  ✓ Saved provenance: {provenance_file}")
    return provenance

def generate_qc_panel(df):
    """Generate quality control panel."""
    print("📊 Generating QC panel...")
    
    # Basic statistics
    morph_features = ['feats_per_token_mean', 'feats_bundle_entropy_bits', 'feats_bundles_per_10k', 'complexity_pc1']
    
    qc_stats = {}
    for feature in morph_features:
        qc_stats[feature] = {
            'mean': float(df[feature].mean()),
            'std': float(df[feature].std()),
            'min': float(df[feature].min()),
            'max': float(df[feature].max()),
            'median': float(df[feature].median())
        }
    
    # Correlation matrix
    corr_matrix = df[morph_features].corr()
    qc_stats['correlation_matrix'] = corr_matrix.to_dict()
    
    # Coverage statistics
    coverage_dist = df['feats_coverage_band'].value_counts().to_dict()
    qc_stats['coverage_distribution'] = coverage_dist
    
    # Coverage by language
    absent_languages = df[df['feats_coverage_band'] == 'Absent']['language_slug'].tolist()
    sparse_languages = df[df['feats_coverage_band'] == 'Sparse']['language_slug'].tolist()
    
    qc_stats['absent_languages'] = absent_languages
    qc_stats['sparse_languages'] = sparse_languages
    
    # Language extremes
    top3_complexity = df.nlargest(3, 'complexity_pc1')[['language_slug', 'complexity_pc1', 'feats_coverage_band']].to_dict('records')
    bottom3_complexity = df.nsmallest(3, 'complexity_pc1')[['language_slug', 'complexity_pc1', 'feats_coverage_band']].to_dict('records')
    
    qc_stats['top3_complexity'] = top3_complexity
    qc_stats['bottom3_complexity'] = bottom3_complexity
    
    # Save QC panel
    qc_file = OUTPUT_DIR / "morph_complexity_qc.json"
    with open(qc_file, 'w') as f:
        json.dump(qc_stats, f, indent=2)
    
    print(f"  ✓ Saved QC panel: {qc_file}")
    return qc_stats

def create_face_validity_plots(df):
    """Create face-validity plots with coverage bands."""
    print("📈 Creating face-validity plots...")
    
    # Set up plotting
    plt.style.use('default')
    
    # Color mapping for coverage bands
    band_colors = {'Adequate': '#2E8B57', 'Sparse': '#FF8C00', 'Absent': '#DC143C'}
    
    # Plot 1: PC1 by language (bar chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by PC1 for better visualization
    df_sorted = df.sort_values('complexity_pc1', ascending=True)
    
    # Create bar plot
    bars = ax.bar(range(len(df_sorted)), df_sorted['complexity_pc1'], 
                  color=[band_colors[band] for band in df_sorted['feats_coverage_band']])
    
    # Customize
    ax.set_xlabel('Language')
    ax.set_ylabel('Morphological Complexity (PC1)')
    ax.set_title('Morphological Complexity by Language\n(Colored by FEATS Coverage Band)')
    
    # Language labels (abbreviated)
    lang_labels = [lang.replace('UD_', '').split('-')[0] for lang in df_sorted['language_slug']]
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(lang_labels, rotation=45, ha='right')
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in band_colors.values()]
    ax.legend(handles, band_colors.keys(), title='FEATS Coverage', loc='upper left')
    
    # Annotate extremes
    top_idx = df_sorted['complexity_pc1'].idxmax()
    bottom_idx = df_sorted['complexity_pc1'].idxmin()
    
    for idx in [top_idx, bottom_idx]:
        lang_pos = df_sorted.index.get_loc(idx)
        pc1_val = df_sorted.loc[idx, 'complexity_pc1']
        lang_name = df_sorted.loc[idx, 'language_slug'].replace('UD_', '').split('-')[0]
        ax.annotate(f'{lang_name}\\n{pc1_val:.2f}', 
                   xy=(lang_pos, pc1_val), 
                   xytext=(5, 10 if pc1_val > 0 else -15), 
                   textcoords='offset points',
                   fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'morph_pc1_by_language.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save underlying data
    plot_data = df_sorted[['language_slug', 'complexity_pc1', 'feats_coverage_band']].copy()
    plot_data.to_csv(FIGURE_DATA_DIR / 'morph_pc1_by_language.csv', index=False)
    
    # Plot 2: PC1 vs FEATS per token
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for band, color in band_colors.items():
        band_data = df[df['feats_coverage_band'] == band]
        marker = 'o' if band == 'Adequate' else '^' if band == 'Sparse' else 's'
        ax.scatter(band_data['feats_per_token_mean'], band_data['complexity_pc1'], 
                  c=color, label=band, alpha=0.7, s=80, marker=marker)
    
    # Correlation line
    x = df['feats_per_token_mean']
    y = df['complexity_pc1']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), 'k--', alpha=0.5)
    
    # Correlation coefficient
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Mean FEATS per Token')
    ax.set_ylabel('Morphological Complexity (PC1)')
    ax.set_title('PC1 vs FEATS per Token\n(Should be strongly positive)')
    ax.legend(title='FEATS Coverage')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'morph_pc1_vs_feats_per_token.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    plot_data = df[['language_slug', 'feats_per_token_mean', 'complexity_pc1', 'feats_coverage_band']].copy()
    plot_data.to_csv(FIGURE_DATA_DIR / 'morph_pc1_vs_feats_per_token.csv', index=False)
    
    # Plot 3: PC1 vs Coverage
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for band, color in band_colors.items():
        band_data = df[df['feats_coverage_band'] == band]
        marker = 'o' if band == 'Adequate' else '^' if band == 'Sparse' else 's'
        ax.scatter(band_data['feats_coverage_train'], band_data['complexity_pc1'], 
                  c=color, label=band, alpha=0.7, s=80, marker=marker)
    
    ax.set_xlabel('FEATS Coverage (% train)')
    ax.set_ylabel('Morphological Complexity (PC1)')
    ax.set_title('PC1 vs FEATS Coverage\n(Absent languages cluster at floor)')
    ax.legend(title='Coverage Band')
    ax.grid(True, alpha=0.3)
    
    # Annotate absent languages
    absent_data = df[df['feats_coverage_band'] == 'Absent']
    for _, row in absent_data.iterrows():
        lang_name = row['language_slug'].replace('UD_', '').split('-')[0]
        ax.annotate(lang_name, 
                   xy=(row['feats_coverage_train'], row['complexity_pc1']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'morph_pc1_vs_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    plot_data = df[['language_slug', 'feats_coverage_train', 'complexity_pc1', 'feats_coverage_band']].copy()
    plot_data.to_csv(FIGURE_DATA_DIR / 'morph_pc1_vs_coverage.csv', index=False)
    
    print(f"  ✓ Saved plots: {FIGURE_DIR}")
    print(f"  ✓ Saved plot data: {FIGURE_DATA_DIR}")

def compute_robustness_snapshot(df):
    """Compute PC1 (Adequate-only) for robustness check."""
    print("🔄 Computing robustness snapshot...")
    
    # Filter to adequate coverage only
    adequate_df = df[df['feats_coverage_band'] == 'Adequate'].copy()
    
    if len(adequate_df) < 3:
        print("  ⚠ Insufficient adequate-coverage languages for PCA")
        return
    
    # Recompute PCA on adequate-only
    feature_matrix = adequate_df[['feats_per_token_mean', 'feats_bundle_entropy_bits', 'feats_bundles_per_10k']].values
    
    # Z-score normalization
    scaler = StandardScaler()
    z_features = scaler.fit_transform(feature_matrix)
    
    # PCA
    pca = PCA(n_components=min(3, len(adequate_df)))
    pca_scores = pca.fit_transform(z_features)
    
    # Sign alignment
    pc1_adequate = pca_scores[:, 0]
    corr_with_feats = np.corrcoef(pc1_adequate, z_features[:, 0])[0, 1]
    if corr_with_feats < 0:
        pc1_adequate = -pc1_adequate
    
    # Store adequate-only PC1
    adequate_df = adequate_df.copy()
    adequate_df['complexity_pc1_adequate_only'] = pc1_adequate
    
    # Compute stability correlation
    pc1_all = adequate_df['complexity_pc1'].values
    stability_corr = np.corrcoef(pc1_all, pc1_adequate)[0, 1]
    
    print(f"  ✓ Adequate-only PC1 computed for {len(adequate_df)} languages")
    print(f"  ✓ Stability correlation: {stability_corr:.3f}")
    
    # Add to main dataframe
    df = df.merge(adequate_df[['language_slug', 'complexity_pc1_adequate_only']], 
                  on='language_slug', how='left')
    
    return df, stability_corr

def create_morphology_ready_export(df):
    """Create single tidy export for downstream stages."""
    print("📤 Creating morphology_ready.csv export...")
    
    # Select key columns for downstream use
    export_columns = [
        'language_slug',
        'complexity_pc1', 
        'feats_coverage_train',
        'feats_coverage_band',
        'feats_per_token_mean',
        'feats_bundle_entropy_bits', 
        'feats_bundles_per_10k',
        'n_tokens_train_content'
    ]
    
    export_df = df[export_columns].copy()
    
    # Sort by complexity for readability
    export_df = export_df.sort_values('complexity_pc1', ascending=False)
    
    # Save
    export_file = OUTPUT_DIR / "morphology_ready.csv"
    export_df.to_csv(export_file, index=False)
    
    print(f"  ✓ Saved morphology_ready.csv: {export_file}")
    print(f"  ✓ Shape: {export_df.shape}")
    print(f"  ✓ Ready for Stage 6/7/9 integration")
    
    return export_df

def main():
    """Main finalization routine."""
    print("STAGE 5 FINALIZATION: Morphological Complexity")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(OUTPUT_DIR / "morph_complexity.csv")
    print(f"Loaded morphological complexity data: {df.shape}")
    
    # A. Freeze PCA provenance  
    provenance = freeze_pca_provenance()
    
    # B. Generate QC panel
    qc_stats = generate_qc_panel(df)
    
    # Create face-validity plots
    create_face_validity_plots(df)
    
    # Robustness snapshot
    df_updated, stability_corr = compute_robustness_snapshot(df)
    if df_updated is not None:
        df = df_updated
    
    # C. Create tidy export
    export_df = create_morphology_ready_export(df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("STAGE 5 COMPLETION SUMMARY")
    print("=" * 60)
    print(f"✓ PCA provenance frozen with {len(df)} languages")
    print(f"✓ QC panel generated with correlation matrix")
    print(f"✓ Face-validity plots created (3 plots)")
    print(f"✓ Robustness PC1 computed (stability r = {stability_corr:.3f})")
    print(f"✓ Morphology ready export: {len(export_df)} languages")
    print()
    print("Ready for downstream stages:")
    print("  • Stage 6/7/9: Use morphology_ready.csv")
    print("  • Stage 11: Plots with coverage bands")
    print("  • Stage 12: Primary (Adequate) + Secondary (All) models")

if __name__ == "__main__":
    main()

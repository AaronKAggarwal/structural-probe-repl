#!/usr/bin/env python3
"""
Compute morphological complexity features from UD FEATS.

Stage 5: Following detailed specification for content-only, train-split analysis
with PCA-based complexity score (PC1) and bootstrap uncertainty estimation.

Outputs: morph_complexity.csv with features, z-scores, PCA results, and provenance.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter
import re
import time
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

REPO_ROOT = Path(__file__).parent.parent
UD_DATA_ROOT = REPO_ROOT / "data" / "ud"

@dataclass
class MorphFeatures:
    """Morphological complexity features for one language."""
    language_slug: str
    n_tokens_train_content: int
    n_sent_train: int
    
    # Raw features (train, content-only)
    feats_per_token_mean: float
    feats_bundle_entropy_bits: float
    feats_bundles_per_10k: float
    
    # Bootstrap CIs (optional)
    feats_per_token_ci_low: Optional[float] = None
    feats_per_token_ci_high: Optional[float] = None
    bundle_entropy_ci_low: Optional[float] = None
    bundle_entropy_ci_high: Optional[float] = None
    bundles_per_10k_ci_low: Optional[float] = None
    bundles_per_10k_ci_high: Optional[float] = None

def load_ud_languages() -> List[str]:
    """Load UD languages from analysis table."""
    analysis_file = REPO_ROOT / "outputs" / "analysis" / "analysis_table_L7.csv"
    df = pd.read_csv(analysis_file)
    return sorted(df['language_slug'].unique())

def parse_conllu_sentences(file_path: Path) -> List[Dict]:
    """Parse CoNLL-U file into sentence structures."""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # End of sentence
                if current_sentence:
                    sentences.append({'tokens': current_sentence})
                    current_sentence = []
            elif line.startswith('#'):
                # Skip comments
                continue
            else:
                # Parse token line
                fields = line.split('\t')
                if len(fields) >= 10:
                    token_id = fields[0]
                    
                    # Skip MWT ranges (e.g., "3-4") and empty nodes (e.g., "1.1")
                    if '-' in token_id or '.' in token_id:
                        continue
                    
                    token = {
                        'id': token_id,
                        'form': fields[1],
                        'upos': fields[3],
                        'feats': fields[5] if fields[5] != '_' else '',
                    }
                    current_sentence.append(token)
    
    # Handle final sentence
    if current_sentence:
        sentences.append({'tokens': current_sentence})
    
    return sentences

def canonicalize_feats_bundle(feats: str) -> str:
    """
    Canonicalize FEATS bundle string.
    
    - Sort keys alphabetically
    - For multi-valued keys, sort values and join with commas
    - Join key-value pairs with | 
    - Use _ when empty
    """
    if not feats or feats == '_':
        return '_'
    
    # Parse FEATS into key-value pairs
    pairs = []
    for pair in feats.split('|'):
        if '=' in pair:
            key, value = pair.split('=', 1)
            # Sort multi-values within each key
            if ',' in value:
                sorted_values = ','.join(sorted(value.split(',')))
                pairs.append(f"{key}={sorted_values}")
            else:
                pairs.append(f"{key}={value}")
    
    # Sort pairs by key and join
    if pairs:
        return '|'.join(sorted(pairs))
    else:
        return '_'

def extract_content_tokens(sentences: List[Dict]) -> List[Dict]:
    """Extract content-only tokens (drop PUNCT, SYM)."""
    content_tokens = []
    
    for sentence in sentences:
        for token in sentence['tokens']:
            upos = token['upos']
            if upos not in {'PUNCT', 'SYM'}:
                content_tokens.append(token)
    
    return content_tokens

def compute_morphological_features(content_tokens: List[Dict]) -> Tuple[float, float, float]:
    """
    Compute the three primary morphological complexity features.
    
    Returns: (feats_per_token_mean, bundle_entropy_bits, bundles_per_10k)
    """
    n_tokens = len(content_tokens)
    if n_tokens == 0:
        return 0.0, 0.0, 0.0
    
    # 1. Mean FEATS per token
    feat_counts = []
    bundles = []
    
    for token in content_tokens:
        feats = token['feats']
        canonical_bundle = canonicalize_feats_bundle(feats)
        bundles.append(canonical_bundle)
        
        # Count feature keys (ignore empty bundles)
        if canonical_bundle == '_':
            feat_counts.append(0)
        else:
            # Count number of key=value pairs
            n_features = len(canonical_bundle.split('|'))
            feat_counts.append(n_features)
    
    feats_per_token_mean = np.mean(feat_counts)
    
    # 2. Bundle entropy (bits)
    bundle_counter = Counter(bundles)
    total_bundles = len(bundles)
    
    bundle_entropy_bits = 0.0
    for bundle, count in bundle_counter.items():
        p = count / total_bundles
        if p > 0:
            bundle_entropy_bits -= p * math.log2(p)
    
    # 3. Bundle type density (per 10k)
    unique_bundles = len(bundle_counter)
    bundles_per_10k = (unique_bundles / n_tokens) * 10000
    
    return feats_per_token_mean, bundle_entropy_bits, bundles_per_10k

def bootstrap_features(sentences: List[Dict], n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap sentence-level resampling for uncertainty estimation.
    
    Returns: Dict with CI bounds for each feature
    """
    np.random.seed(seed)
    
    bootstrap_results = {
        'feats_per_token_mean': [],
        'bundle_entropy_bits': [],
        'bundles_per_10k': []
    }
    
    n_sentences = len(sentences)
    
    for _ in range(n_bootstrap):
        # Resample sentences with replacement
        resampled_indices = np.random.choice(n_sentences, size=n_sentences, replace=True)
        resampled_sentences = [sentences[i] for i in resampled_indices]
        
        # Extract content tokens and compute features
        content_tokens = extract_content_tokens(resampled_sentences)
        if len(content_tokens) > 0:
            feats_mean, entropy, density = compute_morphological_features(content_tokens)
            bootstrap_results['feats_per_token_mean'].append(feats_mean)
            bootstrap_results['bundle_entropy_bits'].append(entropy)
            bootstrap_results['bundles_per_10k'].append(density)
    
    # Compute 95% CIs
    ci_results = {}
    for feature, values in bootstrap_results.items():
        if values:
            ci_low = np.percentile(values, 2.5)
            ci_high = np.percentile(values, 97.5)
            ci_results[feature] = (ci_low, ci_high)
        else:
            ci_results[feature] = (0.0, 0.0)
    
    return ci_results

def compute_language_complexity(language_slug: str, use_bootstrap: bool = True) -> Optional[MorphFeatures]:
    """Compute morphological complexity for one language."""
    print(f"Processing {language_slug}...")
    
    # Find train file
    language_dir = UD_DATA_ROOT / language_slug
    train_file = language_dir / "train.conllu"
    
    if not train_file.exists():
        print(f"  ⚠ Train file not found: {train_file}")
        return None
    
    # Parse sentences
    try:
        sentences = parse_conllu_sentences(train_file)
        print(f"  Parsed {len(sentences)} sentences")
        
        if len(sentences) == 0:
            print(f"  ⚠ No sentences found")
            return None
        
        # Extract content tokens
        content_tokens = extract_content_tokens(sentences)
        n_content_tokens = len(content_tokens)
        
        print(f"  Content tokens: {n_content_tokens}")
        
        if n_content_tokens == 0:
            print(f"  ⚠ No content tokens found")
            return None
        
        # Compute primary features
        feats_mean, entropy, density = compute_morphological_features(content_tokens)
        
        print(f"  Features: mean={feats_mean:.3f}, entropy={entropy:.3f}, density={density:.1f}")
        
        # Bootstrap CIs (optional)
        ci_results = None
        if use_bootstrap and len(sentences) >= 10:  # Only bootstrap if sufficient data
            print(f"  Computing bootstrap CIs...")
            ci_results = bootstrap_features(sentences)
        
        # Create result
        result = MorphFeatures(
            language_slug=language_slug,
            n_tokens_train_content=n_content_tokens,
            n_sent_train=len(sentences),
            feats_per_token_mean=feats_mean,
            feats_bundle_entropy_bits=entropy,
            feats_bundles_per_10k=density
        )
        
        # Add CIs if computed
        if ci_results:
            result.feats_per_token_ci_low, result.feats_per_token_ci_high = ci_results['feats_per_token_mean']
            result.bundle_entropy_ci_low, result.bundle_entropy_ci_high = ci_results['bundle_entropy_bits']
            result.bundles_per_10k_ci_low, result.bundles_per_10k_ci_high = ci_results['bundles_per_10k']
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error processing {language_slug}: {e}")
        return None

def run_pca_analysis(features_list: List[MorphFeatures]) -> Dict:
    """
    Run PCA on z-scored morphological features.
    
    Returns: PCA results with loadings, explained variance, and PC1 scores
    """
    print("\nRunning PCA analysis...")
    
    # Prepare feature matrix
    feature_matrix = []
    language_slugs = []
    
    for features in features_list:
        feature_matrix.append([
            features.feats_per_token_mean,
            features.feats_bundle_entropy_bits,
            features.feats_bundles_per_10k
        ])
        language_slugs.append(features.language_slug)
    
    feature_matrix = np.array(feature_matrix)
    feature_names = ['feats_per_token_mean', 'bundle_entropy_bits', 'bundles_per_10k']
    
    # Z-score normalization
    scaler = StandardScaler()
    z_features = scaler.fit_transform(feature_matrix)
    
    print(f"Z-scored features for {len(features_list)} languages")
    
    # PCA
    pca = PCA(n_components=3)
    pca_scores = pca.fit_transform(z_features)
    
    # Extract PC1
    pc1_raw = pca_scores[:, 0]
    
    # Sign alignment: ensure PC1 correlates positively with feats_per_token_mean
    corr_with_feats = np.corrcoef(pc1_raw, z_features[:, 0])[0, 1]
    if corr_with_feats < 0:
        pc1_raw = -pc1_raw
        pca.components_[0] = -pca.components_[0]
        print("  Sign-flipped PC1 for positive correlation with feats_per_token")
    
    # Results
    results = {
        'language_slugs': language_slugs,
        'z_features': z_features,
        'pc1_scores': pc1_raw,
        'loadings': pca.components_[0],  # PC1 loadings
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }
    
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"  Total variance explained: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # Show loadings
    print("  PC1 loadings:")
    for i, (name, loading) in enumerate(zip(feature_names, pca.components_[0])):
        print(f"    {name}: {loading:.3f}")
    
    return results

def main():
    """Main entry point."""
    print("Stage 5: Computing Morphological Complexity")
    print("=" * 50)
    print("Target: Train split, content-only tokens, FEATS analysis")
    print()
    
    # Load languages
    languages = load_ud_languages()
    print(f"Processing {len(languages)} UD languages...")
    print()
    
    # Compute features for each language
    features_list = []
    
    for language_slug in languages:
        features = compute_language_complexity(language_slug, use_bootstrap=True)
        if features:
            features_list.append(features)
        print()
    
    print(f"Successfully processed {len(features_list)}/{len(languages)} languages")
    
    if len(features_list) < 3:
        print("✗ Insufficient data for PCA (need ≥3 languages)")
        return
    
    # Run PCA
    pca_results = run_pca_analysis(features_list)
    
    # Build output DataFrame
    output_data = []
    
    for i, features in enumerate(features_list):
        lang_slug = features.language_slug
        
        # Find indices in PCA results
        pca_idx = pca_results['language_slugs'].index(lang_slug)
        
        row = {
            # Identity & counts
            'language_slug': lang_slug,
            'n_tokens_train_content': features.n_tokens_train_content,
            'n_sent_train': features.n_sent_train,
            
            # Raw features
            'feats_per_token_mean': features.feats_per_token_mean,
            'feats_bundle_entropy_bits': features.feats_bundle_entropy_bits,
            'feats_bundles_per_10k': features.feats_bundles_per_10k,
            
            # Bootstrap CIs (if available)
            'feats_per_token_ci_low': features.feats_per_token_ci_low,
            'feats_per_token_ci_high': features.feats_per_token_ci_high,
            'bundle_entropy_ci_low': features.bundle_entropy_ci_low,
            'bundle_entropy_ci_high': features.bundle_entropy_ci_high,
            'bundles_per_10k_ci_low': features.bundles_per_10k_ci_low,
            'bundles_per_10k_ci_high': features.bundles_per_10k_ci_high,
            
            # Z-scored features
            'z_feats_per_token_mean': pca_results['z_features'][pca_idx, 0],
            'z_feats_bundle_entropy_bits': pca_results['z_features'][pca_idx, 1],
            'z_feats_bundles_per_10k': pca_results['z_features'][pca_idx, 2],
            
            # PCA results
            'complexity_pc1': pca_results['pc1_scores'][pca_idx],
            'complexity_pc1_raw': pca_results['pc1_scores'][pca_idx],  # Same after sign alignment
            'pca_loading_feats_per_token': pca_results['loadings'][0],
            'pca_loading_bundle_entropy': pca_results['loadings'][1],
            'pca_loading_bundles_per_10k': pca_results['loadings'][2],
            'pca_explained_var_pc1': pca_results['explained_variance_ratio'][0],
            'pca_explained_var_total': np.sum(pca_results['explained_variance_ratio']),
            
            # Provenance
            'split_used': 'train',
            'content_only': True,
            'entropy_base': 2,
            'bundle_canon': 'sorted_keys_and_values',
            'bootstrap_B': 1000,
        }
        
        output_data.append(row)
    
    # Save results
    output_df = pd.DataFrame(output_data)
    output_file = REPO_ROOT / "outputs" / "analysis" / "morph_complexity.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved morphological complexity: {output_file}")
    print(f"   Shape: {output_df.shape}")
    
    # Summary statistics
    print("\nComplexity Summary:")
    print(f"  PC1 range: {output_df['complexity_pc1'].min():.3f} to {output_df['complexity_pc1'].max():.3f}")
    print(f"  PC1 mean: {output_df['complexity_pc1'].mean():.3f}")
    print(f"  PC1 std: {output_df['complexity_pc1'].std():.3f}")
    
    # Top/bottom complexity languages
    top5 = output_df.nlargest(5, 'complexity_pc1')[['language_slug', 'complexity_pc1']]
    bottom5 = output_df.nsmallest(5, 'complexity_pc1')[['language_slug', 'complexity_pc1']]
    
    print("\nMost morphologically complex:")
    for _, row in top5.iterrows():
        print(f"  {row['language_slug']:20}: {row['complexity_pc1']:6.3f}")
    
    print("\nLeast morphologically complex:")
    for _, row in bottom5.iterrows():
        print(f"  {row['language_slug']:20}: {row['complexity_pc1']:6.3f}")
    
    print("\n✓ Stage 5 complete - ready for integration into analysis tables")

if __name__ == "__main__":
    main()

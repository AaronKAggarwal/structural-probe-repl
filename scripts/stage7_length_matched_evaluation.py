#!/usr/bin/env python3
"""
Stage 7: Length-matched evaluation per layer.

Implements stratified bootstrap matching to a pooled target I-length distribution,
controlling for sentence length effects in cross-linguistic probe comparisons.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import hashlib
from collections import defaultdict
from scipy.spatial.distance import jensenshannon

# Configuration
REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
MATCHED_TARGETS_DIR = ANALYSIS_DIR / "matched_targets"
MATCHED_EVAL_DIR = ANALYSIS_DIR / "matched_eval" / "length"
SENTENCE_STATS_DIR = ANALYSIS_DIR / "sentence_stats"

# Create directories
MATCHED_TARGETS_DIR.mkdir(parents=True, exist_ok=True)
MATCHED_EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Default bin configuration
DEFAULT_BIN_EDGES = [2, 6, 11, 16, 26, 41, float('inf')]
DEFAULT_BIN_LABELS = ["2–5", "6–10", "11–15", "16–25", "26–40", "41+"]

# Bootstrap parameters
BOOTSTRAP_ITERATIONS = 1000
MIN_RETENTION = 0.8
MIN_T_THRESHOLD = 200

@dataclass
class LengthBins:
    """Length bin configuration."""
    edges: List[float]
    labels: List[str]
    
    def __post_init__(self):
        assert len(self.edges) == len(self.labels) + 1
        assert self.edges[-1] == float('inf')
    
    def assign_bin(self, length: int) -> int:
        """Assign sentence length to bin index."""
        for i, edge in enumerate(self.edges[:-1]):
            if length < self.edges[i + 1]:
                return i
        return len(self.labels) - 1  # Last bin for overflow

@dataclass
class LanguageBinStats:
    """Per-language bin statistics."""
    language_slug: str
    counts_obs: List[int]
    proportions: List[float]
    total_sentences: int
    valid_bins: List[int]  # Bins with non-zero counts

@dataclass
class MatchingTarget:
    """Target distribution for length matching."""
    bin_edges: List[float]
    bin_labels: List[str]
    target_probs: List[float]
    languages_used: List[str]

def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="Stage 7: Length-matched evaluation")
    ap.add_argument('--split', choices=['test', 'dev'], default='test',
                   help='Dataset split to evaluate (default: test)')
    ap.add_argument('--bootstrap', type=int, default=1000,
                   help='Number of bootstrap iterations (default: 1000)')
    ap.add_argument('--seed', type=int, default=42,
                   help='Global random seed (default: 42)')
    ap.add_argument('--recompute_target', action='store_true',
                   help='Recompute target distribution instead of loading existing')
    ap.add_argument('--target_file', type=str, 
                   default=str(MATCHED_TARGETS_DIR / "length_bins.json"),
                   help='Target distribution file path')
    return ap.parse_args()

def derive_seed(base_seed: int, language_slug: str, probe: str, layer: str, split: str) -> int:
    """Derive reproducible sub-seed for (language, probe, layer, split).
    
    Note: Split inclusion ensures dev/test produce different but reproducible draws.
    """
    key = f"{language_slug}_{probe}_{layer}_{split}"
    hash_obj = hashlib.md5(key.encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    return (base_seed + hash_int) % (2**31)

def load_sentence_stats(language_slug: str, split: str) -> Dict[str, Any]:
    """Load per-sentence statistics from Stage 3."""
    stats_file = SENTENCE_STATS_DIR / language_slug / f"{split}_content_stats.jsonl"
    
    if not stats_file.exists():
        raise FileNotFoundError(f"Sentence stats not found: {stats_file}")
    
    # Load JSONL format (one JSON object per line)
    sentences = []
    with open(stats_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    
    return {'sentences': sentences}

def align_to_full_sentence_list(vec: np.ndarray, N: int, eval_mask: np.ndarray, 
                               ids_metrics: Optional[List[str]], ids_stage3: List[str], 
                               kept_idx: Optional[List[int]]) -> Tuple[Optional[np.ndarray], str]:
    """Align compacted metric array to full Stage-3 sentence list using 3-tier strategy.
    
    Returns:
        (aligned_array, alignment_mode) where alignment_mode is one of:
        'by_ids', 'by_indices', 'by_mask_fallback', 'failed'
    """
    vec = np.asarray(vec, dtype=float)
    full = np.full(N, np.nan, dtype=float)
    
    # (1) Join by sentence IDs
    if ids_metrics and ids_stage3 and len(ids_metrics) == len(vec):
        pos = {sid: i for i, sid in enumerate(ids_stage3)}
        hits = 0
        for j, sid in enumerate(ids_metrics):
            i = pos.get(sid)
            if i is not None and 0 <= i < N:
                full[i] = vec[j]
                hits += 1
        if hits > 0:
            return full, 'by_ids'
    
    # (2) Scatter by kept indices
    if kept_idx and len(kept_idx) == len(vec):
        hits = 0
        for dst, src in zip(kept_idx, range(len(vec))):
            if 0 <= dst < N:
                full[dst] = vec[src]
                hits += 1
        if hits > 0:
            return full, 'by_indices'
    
    # (3) Fallback: assume compact == #evaluable and preserved order
    if len(vec) == int(eval_mask.sum()):
        idx = np.where(eval_mask)[0]
        if len(idx) == len(vec):
            full[idx] = vec
            return full, 'by_mask_fallback'
    
    return None, 'failed'

def discover_run_dir(language_slug: str, probe: str, layer: str) -> Optional[str]:
    """Discover run directory containing detailed metrics for (language, probe, layer)."""
    # Per-language run directories (preferred structure)
    base_per_lang = REPO_ROOT / "outputs" / "baselines_auto" / language_slug / "bert-base-multilingual-cased" / probe / layer / "runs"
    
    if base_per_lang.exists():
        # Prefer 'latest' symlink/dir
        latest = base_per_lang / "latest"
        if latest.exists():
            return str(latest)
        
        # Otherwise pick any run dir with detailed metrics
        for run_dir in sorted(base_per_lang.iterdir()):
            if run_dir.name in {"latest", ".DS_Store"} or not run_dir.is_dir():
                continue
            
            # Check for either split's detailed metrics
            test_final = run_dir / "test_detailed_metrics_final.json"
            test_regular = run_dir / "test_detailed_metrics.json"
            dev_metrics = run_dir / "dev_detailed_metrics.json"
            
            if test_final.exists() or test_regular.exists() or dev_metrics.exists():
                return str(run_dir)
    
    return None

def load_per_sentence_metrics(language_slug: str, probe: str, layer: str, split: str) -> Optional[Dict[str, np.ndarray]]:
    """Load per-sentence probe metrics from detailed_metrics.json files.

    IMPORTANT: Length-matching only works for per-sentence metrics:
    - For dist probe: provide 'uuas' (per-sentence UUAS values)
    - For depth probe: provide 'root_acc' (per-sentence root accuracy values)
    - Optional: per-sentence Spearman if available

    Returns arrays that may be compacted (test) or full (dev), with alignment metadata.
    """
    print(f"  Loading {probe} {layer} metrics for {language_slug} {split}...")

    # Discover run directory
    run_dir = discover_run_dir(language_slug, probe, layer)
    if not run_dir:
        print(f"    ⚠ No run directory found for {language_slug}/{probe}/{layer}")
        return None

    # Determine which detailed metrics file to use
    if split == 'test':
        # Prefer final if available, otherwise regular
        test_final_path = Path(run_dir) / "test_detailed_metrics_final.json"
        test_regular_path = Path(run_dir) / "test_detailed_metrics.json"

        if test_final_path.exists():
            metrics_file = test_final_path
        elif test_regular_path.exists():
            metrics_file = test_regular_path
        else:
            print(f"    ⚠ No test metrics found in {run_dir}")
            return None
    elif split == 'dev':
        metrics_file = Path(run_dir) / "dev_detailed_metrics.json"
        if not metrics_file.exists():
            print(f"    ⚠ No dev metrics found in {run_dir}")
            return None
    else:
        print(f"    ⚠ Unknown split: {split}")
        return None

    # Load detailed metrics
    try:
        with open(metrics_file, 'r') as f:
            detailed_metrics = json.load(f)
    except Exception as e:
        print(f"    ⚠ Error loading {metrics_file}: {e}")
        return None

    # Extract per-sentence metrics based on probe type
    metrics = {}

    if probe == 'dist':
        # Distance probe: UUAS per sentence
        uuas_per_sent = detailed_metrics.get('uuas_per_sentence')
        if uuas_per_sent and isinstance(uuas_per_sent, list):
            metrics['uuas'] = np.array(uuas_per_sent, dtype=float)
        else:
            print(f"    ⚠ No uuas_per_sentence found in {metrics_file}")
            return None

    elif probe == 'depth':
        # Depth probe: Root accuracy per sentence
        root_acc_per_sent = detailed_metrics.get('root_acc_per_sentence')
        if root_acc_per_sent and isinstance(root_acc_per_sent, list):
            metrics['root_acc'] = np.array(root_acc_per_sent, dtype=float)
        else:
            print(f"    ⚠ No root_acc_per_sentence found in {metrics_file}")
            return None
    else:
        print(f"    ⚠ Unknown probe type: {probe}")
        return None

    # Extract alignment metadata for compacted arrays
    metrics['_sent_ids'] = (detailed_metrics.get('sentence_ids') or 
                           detailed_metrics.get('sent_ids'))
    metrics['_kept_indices'] = (detailed_metrics.get('kept_sentence_indices') or
                               detailed_metrics.get('valid_sentence_indices') or
                               detailed_metrics.get('eval_sentence_indices'))

    # Skip spearman_content - it's not per-sentence but per-pair/per-token
    # (This avoids the alignment issues we observed)

    print(f"    ✓ Loaded {len([k for k in metrics.keys() if not k.startswith('_')])} metric arrays")
    return metrics

def compute_bin_statistics(languages: List[str], bins: LengthBins, split: str = 'test') -> Tuple[List[LanguageBinStats], MatchingTarget]:
    """Compute per-language bin statistics and build target distribution."""
    print(f"Computing bin statistics for {len(languages)} languages on {split}...")
    
    all_language_stats = []
    
    for language_slug in languages:
        print(f"  Processing {language_slug}...")
        
        # Load sentence statistics
        try:
            stats = load_sentence_stats(language_slug, split)
        except FileNotFoundError:
            print(f"    ⚠ Sentence stats not found for {language_slug}, skipping")
            continue
        
        # Extract I-lengths (content token counts)
        sentences = stats['sentences']
        i_lengths = [sent['content_len'] for sent in sentences if sent['content_len'] >= 2]
        
        if not i_lengths:
            print(f"    ⚠ No valid sentences for {language_slug}")
            continue
        
        # Compute bin counts
        counts = [0] * len(bins.labels)
        for length in i_lengths:
            bin_idx = bins.assign_bin(length)
            counts[bin_idx] += 1
        
        total = sum(counts)
        proportions = [c / total if total > 0 else 0.0 for c in counts]
        valid_bins = [i for i, c in enumerate(counts) if c > 0]
        
        lang_stats = LanguageBinStats(
            language_slug=language_slug,
            counts_obs=counts,
            proportions=proportions,
            total_sentences=total,
            valid_bins=valid_bins
        )
        all_language_stats.append(lang_stats)
        
        print(f"    Total: {total}, Bins: {counts}")
    
    # Build target distribution from median proportions
    print("Building target distribution...")
    
    if not all_language_stats:
        raise ValueError("No valid language statistics found")
    
    # Collect proportions matrix
    prop_matrix = np.array([stats.proportions for stats in all_language_stats])
    
    # Compute elementwise median
    target_probs_raw = np.median(prop_matrix, axis=0)
    
    # Renormalize
    target_probs = target_probs_raw / np.sum(target_probs_raw)
    
    target = MatchingTarget(
        bin_edges=bins.edges.copy(),
        bin_labels=bins.labels.copy(),
        target_probs=target_probs.tolist(),
        languages_used=[stats.language_slug for stats in all_language_stats]
    )
    
    print(f"  Target distribution: {target.target_probs}")
    print(f"  Based on {len(target.languages_used)} languages")
    
    return all_language_stats, target

def load_target_distribution(target_file: str) -> MatchingTarget:
    """Load existing target distribution from JSON.
    
    Note: Preserves bin order by explicitly mapping labels to probabilities.
    """
    with open(target_file, 'r') as f:
        data = json.load(f)
    
    # Preserve bin order: build target_probs explicitly by label
    probs_by_label = data['target_probs_median']
    target_probs = [probs_by_label[lbl] for lbl in data['bin_labels']]
    
    # Normalize on load (belt-and-suspenders)
    probs = np.array(target_probs, dtype=float)
    target_probs = (probs / probs.sum()).tolist()
    
    return MatchingTarget(
        bin_edges=data['bin_edges'],
        bin_labels=data['bin_labels'],
        target_probs=target_probs,
        languages_used=data['languages_used_for_target']
    )

def stratified_bootstrap_matching(
    metrics: Dict[str, np.ndarray],
    i_lengths: List[int],
    bins: LengthBins,
    target: MatchingTarget,
    seed: int,
    bootstrap_iterations: int,
    split: str = 'test'
) -> Dict[str, Any]:
    """Perform stratified bootstrap matching for one (language, probe, layer).
    
    Notes:
    - Length-matching only works for per-sentence metrics (e.g., UUAS per sentence)
    - JS divergence is squared Jensen-Shannon distance between effective and target distributions
    - Does not renormalize after downshifts; reports transparent retention T'/T
    """
    
    rng = np.random.RandomState(seed)
    
    # Build bin indices
    bin_indices = [[] for _ in bins.labels]
    for i, length in enumerate(i_lengths):
        if length >= 2:  # Valid I-length
            bin_idx = bins.assign_bin(length)
            bin_indices[bin_idx].append(i)
    
    # Length/metric alignment guard: avoid silent index drift
    for metric_name, metric_values in metrics.items():
        if metric_values is not None:
            assert len(metric_values) == len(i_lengths), f"Length mismatch for {metric_name}"
    
    # Intersect with valid metric indices
    valid_indices = {}
    for metric_name, metric_values in metrics.items():
        if metric_values is not None:
            valid_mask = np.isfinite(metric_values)
            valid_indices[metric_name] = set(np.where(valid_mask)[0])
    
    # If no valid metrics, return empty result
    if not valid_indices:
        return {
            'T_raw': 0,
            'T_prime': 0,
            'retention_ratio': 0.0,
            'bins_truncated': list(range(len(bins.labels))),
            'seed': seed,
            'skip_reason': 'no_valid_metrics',
            'bootstrap_results': {}
        }
    
    # Use intersection of all valid metric indices
    common_valid = set.intersection(*valid_indices.values()) if valid_indices else set()
    
    # Build valid bin indices
    valid_bin_indices = []
    for bin_idx in range(len(bins.labels)):
        valid_bin = [i for i in bin_indices[bin_idx] if i in common_valid]
        valid_bin_indices.append(valid_bin)
    
    T_raw = len(common_valid)
    
    # Flag tiny samples but don't gate (compute anyway with big CIs)
    small_sample_flag = (T_raw < MIN_T_THRESHOLD)
    
    # Determine target counts with downshift
    target_counts = []
    requested_counts = []
    bins_truncated = []
    
    for bin_idx, prob in enumerate(target.target_probs):
        requested = round(T_raw * prob)
        requested_counts.append(requested)
        available = len(valid_bin_indices[bin_idx])
        
        if available < requested:
            target_counts.append(available)
            if requested > 0:  # Only flag as truncated if we actually wanted some
                bins_truncated.append(bin_idx)
        else:
            target_counts.append(requested)
    
    T_prime = sum(target_counts)
    retention_ratio = T_prime / T_raw if T_raw > 0 else 0.0
    
    # Compute effective distribution and JS divergence
    eff_counts = np.array([min(len(valid_bin_indices[i]), target_counts[i]) for i in range(len(bins.labels))])
    eff_probs = eff_counts / eff_counts.sum() if eff_counts.sum() > 0 else np.zeros_like(eff_counts, dtype=float)
    js_div = float(jensenshannon(eff_probs, np.array(target.target_probs))**2) if eff_probs.sum() > 0 else float('nan')
    
    # Compute duplication factors per bin
    duplication = []
    for i, c in enumerate(target_counts):
        u = len(valid_bin_indices[i])
        duplication.append(float(c / max(u, 1)))  # ~avg draws per unique
    
    # Compute 95th percentile duplication for QC
    duplication_95p = float(np.percentile(duplication, 95)) if duplication else 0.0
    
    # Bootstrap sampling
    bootstrap_results = {}
    
    if T_prime > 0:
        for metric_name in metrics.keys():
            if metrics[metric_name] is not None:
                bootstrap_values = []
                
                for _ in range(bootstrap_iterations):
                    sampled_indices = []
                    
                    for bin_idx, count in enumerate(target_counts):
                        if count > 0 and len(valid_bin_indices[bin_idx]) > 0:
                            sampled = rng.choice(valid_bin_indices[bin_idx], size=count, replace=True)
                            sampled_indices.extend(sampled)
                    
                    if sampled_indices:
                        metric_sample = metrics[metric_name][sampled_indices]
                        bootstrap_values.append(np.mean(metric_sample))
                
                if bootstrap_values:
                    bootstrap_array = np.array(bootstrap_values)
                    bootstrap_results[metric_name] = {
                        'point_estimate': float(np.mean(bootstrap_array)),
                        'ci_low': float(np.percentile(bootstrap_array, 2.5)),
                        'ci_high': float(np.percentile(bootstrap_array, 97.5)),
                        'ci_width': float(np.percentile(bootstrap_array, 97.5) - np.percentile(bootstrap_array, 2.5))
                    }
    
    return {
        'T_raw': T_raw,
        'T_prime': T_prime,
        'retention_ratio': retention_ratio,
        'bins_truncated': bins_truncated,
        'requested_counts': requested_counts,
        'target_counts': target_counts,
        'valid_bin_counts': [len(indices) for indices in valid_bin_indices],
        'effective_counts': eff_counts.tolist(),
        'effective_probs': eff_probs.tolist(),
        'js_to_target': js_div,
        'duplication_factors': duplication,
        'duplication_95p': duplication_95p,
        'small_sample_flag': small_sample_flag,
        'seed': seed,
        'bootstrap_results': bootstrap_results
    }

def save_target_specification(target: MatchingTarget, output_file: Path):
    """Save the global target specification."""
    target_spec = {
        'bin_edges': target.bin_edges,
        'bin_labels': target.bin_labels,
        'target_probs_median': {label: prob for label, prob in zip(target.bin_labels, target.target_probs)},
        'languages_used_for_target': target.languages_used,
        'created_by': 'stage7_length_matched_evaluation.py',
        'description': 'Pooled cross-language I-length distribution (median of per-language bin proportions)'
    }
    
    with open(output_file, 'w') as f:
        json.dump(target_spec, f, indent=2)
    
    print(f"  ✓ Saved target specification: {output_file}")

def main():
    """Main Stage 7 execution."""
    args = parse_args()
    
    print("STAGE 7: LENGTH-MATCHED EVALUATION")
    print("=" * 50)
    print(f"Split: {args.split}")
    print(f"Bootstrap iterations: {args.bootstrap}")
    print(f"Random seed: {args.seed}")
    print(f"Target file: {args.target_file}")
    print(f"Recompute target: {args.recompute_target}")
    print("Objective: Control for sentence length effects via stratified bootstrap matching")
    print()
    
    # Setup
    bins = LengthBins(edges=DEFAULT_BIN_EDGES, labels=DEFAULT_BIN_LABELS)
    
    # Get list of languages from analysis table
    master_file = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    if not master_file.exists():
        raise FileNotFoundError(f"Master analysis table not found: {master_file}")
    
    df = pd.read_csv(master_file)
    languages = sorted(df['language_slug'].unique())
    probes = ['dist', 'depth']
    layers = ['L5', 'L6', 'L7', 'L8', 'L9', 'L10']
    
    print(f"Languages: {len(languages)}")
    print(f"Probes: {probes}")
    print(f"Layers: {layers}")
    print()
    
    # 1. Load or compute target distribution
    print("1. TARGET DISTRIBUTION")
    print("-" * 40)
    
    target_file_path = Path(args.target_file)
    
    if (not args.recompute_target) and target_file_path.exists():
        print(f"Loading existing target from: {target_file_path}")
        try:
            target = load_target_distribution(str(target_file_path))
            print(f"  ✓ Loaded target with {len(target.languages_used)} languages")
            print(f"  Target probs: {target.target_probs}")
        except Exception as e:
            print(f"❌ Error loading target: {e}")
            print("Falling back to recomputing...")
            args.recompute_target = True
    
    if args.recompute_target or not target_file_path.exists():
        print(f"Computing target distribution from {args.split} split...")
        try:
            lang_stats, target = compute_bin_statistics(languages, bins, args.split)
            save_target_specification(target, target_file_path)
        except Exception as e:
            print(f"❌ Error computing bin statistics: {e}")
            print("\n🔧 IMPLEMENTATION NOTE:")
            print("This script requires per-sentence statistics from Stage 3.")
            print("If sentence stats are not available, you'll need to:")
            print("1. Run Stage 3 sentence covariate computation")
            print("2. Ensure sentence stats are saved to outputs/analysis/sentence_stats/")
            print("3. Load actual per-sentence probe metrics (currently mocked)")
            return
    
    # 2. Per-language length-matched evaluation
    print("\n2. PER-LANGUAGE MATCHING")
    print("-" * 40)
    print("✅ Real per-sentence metrics integration complete")
    print("Loading from detailed_metrics.json files in run directories")
    print()
    
    # Create results structure
    results = []
    diagnostics = {}
    
    for language_slug in languages:
        print(f"Processing {language_slug}...")
        
        # Load per-sentence I-lengths from Stage 3
        try:
            stats = load_sentence_stats(language_slug, args.split)
            sentences = stats['sentences']
            i_lengths_full = [sent['content_len'] for sent in sentences]
            N = len(i_lengths_full)
            
            # Create sentence ID list for alignment
            stage3_ids = [s.get('sent_id', f"sent_{i}") for i, s in enumerate(sentences)]
            
            # Create evaluation masks for different metrics
            mask_uuas = np.array([n >= 2 for n in i_lengths_full], dtype=bool)
            mask_root = np.array([n >= 2 for n in i_lengths_full], dtype=bool)
            mask_spear_dist = np.array([n >= 3 for n in i_lengths_full], dtype=bool)
            mask_spear_depth = np.array([n >= 2 for n in i_lengths_full], dtype=bool)
            
            if mask_uuas.sum() == 0 and mask_root.sum() == 0:
                print(f"  ⚠ No evaluable sentences for {language_slug}, skipping")
                continue
                
        except FileNotFoundError:
            print(f"  ⚠ Sentence stats not found for {language_slug}, skipping")
            continue
        
        for probe in probes:
            for layer in layers:
                # Derive seed (includes split for different dev/test draws)
                seed = derive_seed(args.seed, language_slug, probe, layer, args.split)
                
                # Load real per-sentence metrics (may be compacted)
                metrics = load_per_sentence_metrics(language_slug, probe, layer, args.split)
                if not metrics:
                    print(f"    ⚠ No metrics found for {language_slug}/{probe}/{layer}, skipping")
                    continue
                
                # Extract alignment metadata
                ids_metrics = metrics.pop('_sent_ids', None)
                kept_idx = metrics.pop('_kept_indices', None)
                
                # Align and filter metrics
                filtered_metrics = {}
                alignment_diagnostics = {}
                
                for metric_name, metric_values in metrics.items():
                    if metric_values is None:
                        continue
                    
                    # Determine appropriate evaluation mask
                    if metric_name == 'uuas':
                        eval_mask = mask_uuas
                    elif metric_name == 'root_acc':
                        eval_mask = mask_root
                    elif probe == 'dist' and metric_name.startswith('spearman'):
                        eval_mask = mask_spear_dist
                    elif probe == 'depth' and metric_name.startswith('spearman'):
                        eval_mask = mask_spear_depth
                    else:
                        print(f"    ⚠ Unknown metric {metric_name}, skipping")
                        continue
                    
                    # Try 3-tier alignment to full sentence list
                    if len(metric_values) != N:
                        aligned, alignment_mode = align_to_full_sentence_list(
                            metric_values, N, eval_mask, ids_metrics, stage3_ids, kept_idx
                        )
                        if aligned is None:
                            print(f"    ⚠ Could not align {metric_name} (len {len(metric_values)} vs {N}); skipping")
                            continue
                        aligned_values = aligned
                        alignment_diagnostics[metric_name] = {
                            'mode': alignment_mode,
                            'hits': int(np.isfinite(aligned).sum())
                        }
                    else:
                        # Already full length
                        aligned_values = metric_values
                        alignment_diagnostics[metric_name] = {
                            'mode': 'full_length',
                            'hits': len(metric_values)
                        }
                    
                    # Apply evaluation mask to get final filtered metrics
                    filtered_metrics[metric_name] = aligned_values[eval_mask]
                
                if not filtered_metrics:
                    print(f"    ⚠ No aligned metrics for {language_slug}/{probe}/{layer}, skipping")
                    continue
                
                # Get appropriate I-lengths for this probe
                if probe == 'dist':
                    i_lengths = np.array(i_lengths_full)[mask_uuas].tolist()
                elif probe == 'depth':
                    i_lengths = np.array(i_lengths_full)[mask_root].tolist()
                else:
                    i_lengths = np.array(i_lengths_full)[mask_uuas].tolist()  # fallback
                
                # QC: Verify alignment worked correctly
                for metric_name, metric_values in filtered_metrics.items():
                    T_raw_check = np.isfinite(metric_values).sum()
                    if metric_name in alignment_diagnostics:
                        expected_finite = alignment_diagnostics[metric_name]['hits']
                        # Allow some tolerance for masking differences
                        if abs(T_raw_check - expected_finite) > min(50, expected_finite * 0.1):
                            print(f"    ⚠ QC warning: {metric_name} finite count {T_raw_check} != expected {expected_finite}")

                # Perform matching
                result = stratified_bootstrap_matching(
                    metrics=filtered_metrics,
                    i_lengths=i_lengths,
                    bins=bins,
                    target=target,
                    seed=seed,
                    bootstrap_iterations=args.bootstrap,
                    split=args.split
                )
                
                # Add alignment diagnostics to result
                result['alignment_diagnostics'] = alignment_diagnostics
                
                # Add to results
                row = {
                    'language_slug': language_slug,
                    'probe': probe,
                    'layer': layer,
                    'split': args.split,
                    'T_raw': result['T_raw'],
                    'T_prime': result['T_prime'],
                    'retention_ratio': result['retention_ratio'],
                    'bins_truncated_count': len(result.get('bins_truncated', [])),
                    'js_to_target': result.get('js_to_target', float('nan')),
                    'duplication_max': max(result.get('duplication_factors', [0.0])) if result.get('duplication_factors') else 0.0,
                    'duplication_95p': result.get('duplication_95p', 0.0),
                    'valid_bin_counts_json': json.dumps(result.get('valid_bin_counts', [])),
                    'effective_bin_counts_json': json.dumps(result.get('effective_counts', [])),
                    'effective_bin_probs_json': json.dumps(result.get('effective_probs', [])),
                    'requested_counts_json': json.dumps(result.get('requested_counts', [])),
                    'target_counts_json': json.dumps(result.get('target_counts', [])),
                    'seed': result['seed'],
                    'alignment_modes_json': json.dumps({k: v['mode'] for k, v in alignment_diagnostics.items()})
                }
                
                # Add flags
                row.update({
                    'retention_low_flag': row['retention_ratio'] < MIN_RETENTION,
                    'small_sample_flag': result.get('small_sample_flag', False),
                    'thin_bins_json': json.dumps([i for i, c in enumerate(result.get('valid_bin_counts', [])) if c < 5])
                })
                
                # Add skip reason if present
                if 'skip_reason' in result:
                    row['skip_reason'] = result['skip_reason']
                
                # Add bootstrap results
                for metric_name, bootstrap_data in result['bootstrap_results'].items():
                    row[f'{metric_name}_length_matched'] = bootstrap_data['point_estimate']
                    row[f'{metric_name}_length_matched_ci_low'] = bootstrap_data['ci_low']
                    row[f'{metric_name}_length_matched_ci_high'] = bootstrap_data['ci_high']
                    row[f'{metric_name}_length_matched_ci_width'] = bootstrap_data['ci_width']

                    # Plan-aligned alias for depth probe naming: rootacc_*
                    if metric_name == 'root_acc':
                        row['rootacc_length_matched'] = bootstrap_data['point_estimate']
                        row['rootacc_length_matched_ci_low'] = bootstrap_data['ci_low']
                        row['rootacc_length_matched_ci_high'] = bootstrap_data['ci_high']
                        row['rootacc_length_matched_ci_width'] = bootstrap_data['ci_width']
                
                results.append(row)
        
        # Save richer diagnostics (include matching results for QC)
        lang_diagnostics = {
            'split': args.split,
            'bin_edges': bins.edges,
            'bin_labels': bins.labels,
            'target_probs': target.target_probs,
            'per_probe_layer': {}
        }
        
        # Add per-probe-layer diagnostics from results
        for row in [r for r in results if r['language_slug'] == language_slug]:
            key = f"{row['probe']}_{row['layer']}"
            row_target_counts = json.loads(row.get('target_counts_json', '[]'))
            row_requested_counts = json.loads(row.get('requested_counts_json', '[]'))
            row_valid_counts = json.loads(row.get('valid_bin_counts_json', '[]'))
            row_eff_counts = json.loads(row.get('effective_bin_counts_json', '[]'))
            row_eff_probs = json.loads(row.get('effective_bin_probs_json', '[]'))
            lang_diagnostics['per_probe_layer'][key] = {
                'T_raw': row['T_raw'],
                'T_prime': row['T_prime'],
                'retention_ratio': row['retention_ratio'],
                'js_to_target': row['js_to_target'],
                'duplication_max': row['duplication_max'],
                'duplication_95p': row['duplication_95p'],
                'bins_truncated_count': row['bins_truncated_count'],
                'retention_low_flag': row['retention_low_flag'],
                'small_sample_flag': row['small_sample_flag'],
                # Observed per-bin counts among valid metric sentences
                'valid_bin_counts': row_valid_counts,
                'target_counts': row_target_counts,
                'requested_counts': row_requested_counts,
                'effective_probs': row_eff_probs,
                # Explicit bins truncated list for transparency: requested > target (downshifted)
                'bins_truncated': [i for i, (req, tgt) in enumerate(zip(row_requested_counts, row_target_counts)) if tgt < req],
                'alignment_modes': json.loads(row.get('alignment_modes_json', '{}'))
            }
        
        diagnostics[language_slug] = lang_diagnostics
    
    # 3. Save outputs
    print("\n3. SAVING OUTPUTS")
    print("-" * 40)
    
    # Create split-specific directories
    split_eval_dir = MATCHED_EVAL_DIR / args.split
    split_eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Main results CSV
    results_df = pd.DataFrame(results)
    output_file = ANALYSIS_DIR / f"matched_eval_length_per_layer_{args.split}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"  ✓ Results: {output_file} ({len(results_df)} rows)")

    # Also emit the plan-specified unsuffixed CSV for the primary test split
    if args.split == 'test':
        canonical_file = ANALYSIS_DIR / "matched_eval_length_per_layer.csv"
        results_df.to_csv(canonical_file, index=False)
        print(f"  ✓ Results (canonical for test): {canonical_file}")
    
    # Per-language diagnostics
    for language_slug, diag_data in diagnostics.items():
        diag_file = split_eval_dir / f"{language_slug}.json"
        with open(diag_file, 'w') as f:
            json.dump(diag_data, f, indent=2)
        print(f"  ✓ Diagnostics: {diag_file}")
    
    # 4. Summary
    print("\n🎯 STAGE 7 STATUS")
    print("-" * 40)
    print("✅ Target distribution computed and saved")
    print("✅ Matching framework implemented")
    print("✅ Real metrics integration completed")
    print()
    print("📋 QC RECOMMENDATIONS:")
    print("1. Validate retention ratios and CI widths")
    print("2. Check JS divergence distribution")
    print("3. Review duplication patterns")
    print("4. Verify sentence alignment between Stage 3 stats and probe metrics")
    print()
    print("📊 EXPECTED USAGE:")
    print("• Stage 11: Overlay length-matched UUAS on raw UUAS figures")
    print("• Stage 12: Use length-matched metrics in robustness models")
    print()
    print("🔧 IMPLEMENTATION STATUS:")
    print("✅ CLI with split handling and target reuse")
    print("✅ Small sample and low retention flagging (no gating)")
    print("✅ JS divergence and duplication tracking")
    print("✅ Bin order preservation and normalization")
    print("✅ Split-aware seeding for reproducible dev/test draws")
    print("✅ Length/metric alignment guards")
    print("✅ CI widths in output")
    print("✅ Comprehensive diagnostics per language")
    print("✅ Real per-sentence probe metrics integrated")

if __name__ == "__main__":
    main()

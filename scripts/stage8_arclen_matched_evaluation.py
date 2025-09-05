#!/usr/bin/env python3
"""
Stage 8: Arc-length–matched evaluation per layer.

Controls for tree structural difficulty by matching per-sentence mean arc length
to a pooled (quantile-binned) target distribution across languages.

Outputs length-matched UUAS (distance) and RootAcc (depth) with 95% CIs,
plus diagnostics analogous to Stage 7.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
from scipy.spatial.distance import jensenshannon


# Configuration
REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
MATCHED_TARGETS_DIR = ANALYSIS_DIR / "matched_targets"
MATCHED_EVAL_DIR = ANALYSIS_DIR / "matched_eval" / "arclen"
SENTENCE_STATS_DIR = ANALYSIS_DIR / "sentence_stats"

# Create directories
MATCHED_TARGETS_DIR.mkdir(parents=True, exist_ok=True)
MATCHED_EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Bootstrap parameters
BOOTSTRAP_ITERATIONS = 1000
MIN_RETENTION = 0.8
MIN_T_THRESHOLD = 200


@dataclass
class ArcBins:
    edges: List[float]
    labels: List[str]

    def __post_init__(self):
        assert len(self.edges) == len(self.labels) + 1

    def assign_bin(self, value: float) -> int:
        # Right-open bins [e_i, e_{i+1}) except last which is closed
        for i in range(len(self.labels)):
            lo, hi = self.edges[i], self.edges[i + 1]
            if i < len(self.labels) - 1:
                if value >= lo and value < hi:
                    return i
            else:
                if value >= lo and value <= hi:
                    return i
        return len(self.labels) - 1


@dataclass
class MatchingTarget:
    bin_edges: List[float]
    bin_labels: List[str]
    target_probs: List[float]
    languages_used: List[str]


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 8: Arc-length–matched evaluation")
    ap.add_argument('--split', choices=['test', 'dev'], default='test',
                    help='Dataset split to evaluate (default: test)')
    ap.add_argument('--bootstrap', type=int, default=1000,
                    help='Number of bootstrap iterations (default: 1000)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Global random seed (default: 42)')
    ap.add_argument('--recompute_target', action='store_true',
                    help='Recompute target distribution instead of loading existing')
    ap.add_argument('--target_file', type=str,
                    default=str(MATCHED_TARGETS_DIR / "arclen_bins.json"),
                    help='Target distribution file path')
    return ap.parse_args()


def derive_seed(base_seed: int, language_slug: str, probe: str, layer: str, split: str) -> int:
    key = f"arc_{language_slug}_{probe}_{layer}_{split}"
    hash_int = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return (base_seed + hash_int) % (2**31)


def load_sentence_stats(language_slug: str, split: str) -> Dict[str, Any]:
    stats_file = SENTENCE_STATS_DIR / language_slug / f"{split}_content_stats.jsonl"
    if not stats_file.exists():
        raise FileNotFoundError(f"Sentence stats not found: {stats_file}")
    sentences = []
    with open(stats_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    return {'sentences': sentences}


def discover_run_dir(language_slug: str, probe: str, layer: str) -> Optional[str]:
    base_per_lang = REPO_ROOT / "outputs" / "baselines_auto" / language_slug / "bert-base-multilingual-cased" / probe / layer / "runs"
    if base_per_lang.exists():
        latest = base_per_lang / "latest"
        if latest.exists():
            return str(latest)
        for run_dir in sorted(base_per_lang.iterdir()):
            if run_dir.name in {"latest", ".DS_Store"} or not run_dir.is_dir():
                continue
            test_final = run_dir / "test_detailed_metrics_final.json"
            test_regular = run_dir / "test_detailed_metrics.json"
            dev_metrics = run_dir / "dev_detailed_metrics.json"
            if test_final.exists() or test_regular.exists() or dev_metrics.exists():
                return str(run_dir)
    return None


def load_per_sentence_metrics(language_slug: str, probe: str, layer: str, split: str) -> Optional[Dict[str, np.ndarray]]:
    run_dir = discover_run_dir(language_slug, probe, layer)
    if not run_dir:
        return None
    if split == 'test':
        test_final_path = Path(run_dir) / "test_detailed_metrics_final.json"
        test_regular_path = Path(run_dir) / "test_detailed_metrics.json"
        if test_final_path.exists():
            metrics_file = test_final_path
        elif test_regular_path.exists():
            metrics_file = test_regular_path
        else:
            return None
    else:
        metrics_file = Path(run_dir) / "dev_detailed_metrics.json"
        if not metrics_file.exists():
            return None

    try:
        with open(metrics_file, 'r') as f:
            detailed_metrics = json.load(f)
    except Exception:
        return None

    metrics: Dict[str, np.ndarray] = {}
    if probe == 'dist':
        uuas_per_sent = detailed_metrics.get('uuas_per_sentence')
        if uuas_per_sent and isinstance(uuas_per_sent, list):
            metrics['uuas'] = np.array(uuas_per_sent, dtype=float)
        else:
            return None
    elif probe == 'depth':
        root_acc_per_sent = detailed_metrics.get('root_acc_per_sentence')
        if root_acc_per_sent and isinstance(root_acc_per_sent, list):
            metrics['root_acc'] = np.array(root_acc_per_sent, dtype=float)
        else:
            return None
    else:
        return None

    metrics['_sent_ids'] = (detailed_metrics.get('sentence_ids') or detailed_metrics.get('sent_ids'))
    metrics['_kept_indices'] = (detailed_metrics.get('kept_sentence_indices') or
                                detailed_metrics.get('valid_sentence_indices') or
                                detailed_metrics.get('eval_sentence_indices'))
    return metrics


def align_to_full_sentence_list(vec: np.ndarray, N: int, eval_mask: np.ndarray,
                                ids_metrics: Optional[List[str]], ids_stage3: List[str],
                                kept_idx: Optional[List[int]]) -> Tuple[Optional[np.ndarray], str]:
    vec = np.asarray(vec, dtype=float)
    full = np.full(N, np.nan, dtype=float)
    if ids_metrics and ids_stage3 and len(ids_metrics) == len(vec):
        pos = {sid: i for i, sid in enumerate(ids_stage3)}
        hits = 0
        for j, sid in enumerate(ids_metrics):
            i = pos.get(sid)
            if i is not None and 0 <= i < N:
                full[i] = vec[j]; hits += 1
        if hits > 0:
            return full, 'by_ids'
    if kept_idx and len(kept_idx) == len(vec):
        hits = 0
        for dst, src in zip(kept_idx, range(len(vec))):
            if 0 <= dst < N:
                full[dst] = vec[src]; hits += 1
        if hits > 0:
            return full, 'by_indices'
    if len(vec) == int(eval_mask.sum()):
        idx = np.where(eval_mask)[0]
        if len(idx) == len(vec):
            full[idx] = vec
            return full, 'by_mask_fallback'
    return None, 'failed'


def compute_pooled_arclen_bins(languages: List[str], split: str = 'test', K: int = 5) -> Tuple[ArcBins, Dict[str, Any]]:
    # Pool valid arc lengths across languages
    pooled = []
    langs_used = []
    for lang in languages:
        try:
            stats = load_sentence_stats(lang, split)
        except FileNotFoundError:
            continue
        sentences = stats['sentences']
        valid = [s for s in sentences if (s.get('num_content_arcs_used', 0) or 0) >= 1 and isinstance(s.get('mean_arc_len', None), (int, float))]
        if not valid:
            continue
        pooled.extend([float(s['mean_arc_len']) for s in valid])
        langs_used.append(lang)

    if not pooled:
        raise ValueError("No valid arc lengths available across languages")

    pooled = np.array(pooled, dtype=float)

    # Compute quantile edges; handle ties by nudging or reducing K
    labels = [f"Q{i+1}" for i in range(K)]
    q = np.linspace(0.0, 1.0, K + 1)
    edges = np.quantile(pooled, q).astype(float).tolist()

    def nudge_edges(vals: List[float]) -> List[float]:
        # Ensure strictly increasing by minimal epsilon where possible
        eps = 1e-9
        out = vals[:]
        for i in range(1, len(out)):
            if out[i] <= out[i-1]:
                out[i] = np.nextafter(out[i-1], float('inf'))
        return out

    edges_nudged = nudge_edges(edges)
    # If still not strictly increasing (pathological), reduce K to 4
    if not all(edges_nudged[i] < edges_nudged[i+1] for i in range(len(edges_nudged)-1)):
        K2 = 4
        labels = [f"Q{i+1}" for i in range(K2)]
        q = np.linspace(0.0, 1.0, K2 + 1)
        edges_nudged = np.quantile(pooled, q).astype(float).tolist()
        edges_nudged = nudge_edges(edges_nudged)

    # Realized pooled probabilities
    counts = [0]* (len(edges_nudged)-1)
    for v in pooled:
        # Last bin closed
        for i in range(len(counts)):
            lo, hi = edges_nudged[i], edges_nudged[i+1]
            if (i < len(counts)-1 and v >= lo and v < hi) or (i == len(counts)-1 and v >= lo and v <= hi):
                counts[i] += 1; break
    probs = (np.array(counts, dtype=float) / max(1, sum(counts))).tolist()

    bins = ArcBins(edges=edges_nudged, labels=labels)
    meta = {
        'bin_edges': bins.edges,
        'bin_labels': bins.labels,
        'realized_probs': probs,
        'languages_used_for_target': langs_used,
    }
    return bins, meta


def load_target_distribution(target_file: str) -> MatchingTarget:
    with open(target_file, 'r') as f:
        data = json.load(f)
    probs = np.array(data['realized_probs'], dtype=float)
    probs = (probs / probs.sum()).tolist()
    return MatchingTarget(
        bin_edges=data['bin_edges'],
        bin_labels=data['bin_labels'],
        target_probs=probs,
        languages_used=data.get('languages_used_for_target', [])
    )


def stratified_bootstrap_matching(metrics: Dict[str, np.ndarray], arclens: List[float],
                                  bins: ArcBins, target: MatchingTarget,
                                  seed: int, bootstrap_iterations: int,
                                  split: str = 'test') -> Dict[str, Any]:
    rng = np.random.RandomState(seed)

    # Build bin indices
    bin_indices = [[] for _ in bins.labels]
    for i, a in enumerate(arclens):
        b = bins.assign_bin(float(a))
        bin_indices[b].append(i)

    # Intersection of valid metrics
    valid_indices = {}
    for metric_name, metric_values in metrics.items():
        if metric_values is not None:
            valid_mask = np.isfinite(metric_values)
            valid_indices[metric_name] = set(np.where(valid_mask)[0])
    if not valid_indices:
        return {
            'T_raw': 0, 'T_prime': 0, 'retention_ratio': 0.0,
            'bins_truncated': list(range(len(bins.labels))), 'seed': seed,
            'skip_reason': 'no_valid_metrics', 'bootstrap_results': {}
        }

    common_valid = set.intersection(*valid_indices.values()) if valid_indices else set()
    # Build valid bin indices
    valid_bin_indices = []
    for bi in range(len(bins.labels)):
        valid_bin_indices.append([i for i in bin_indices[bi] if i in common_valid])

    T_raw = len(common_valid)
    small_sample_flag = (T_raw < MIN_T_THRESHOLD)

    target_counts = []
    requested_counts = []
    bins_truncated = []
    for bi, prob in enumerate(target.target_probs):
        requested = round(T_raw * prob)
        requested_counts.append(requested)
        available = len(valid_bin_indices[bi])
        if available < requested:
            target_counts.append(available)
            if requested > 0:
                bins_truncated.append(bi)
        else:
            target_counts.append(requested)

    T_prime = sum(target_counts)
    retention_ratio = T_prime / T_raw if T_raw > 0 else 0.0

    eff_counts = np.array([min(len(valid_bin_indices[i]), target_counts[i]) for i in range(len(bins.labels))])
    eff_probs = (eff_counts / eff_counts.sum()) if eff_counts.sum() > 0 else np.zeros_like(eff_counts, dtype=float)
    js_div = float(jensenshannon(eff_probs, np.array(target.target_probs))**2) if eff_probs.sum() > 0 else float('nan')

    duplication = []
    for i, c in enumerate(target_counts):
        u = len(valid_bin_indices[i])
        duplication.append(float(c / max(u, 1)))
    duplication_95p = float(np.percentile(duplication, 95)) if duplication else 0.0

    bootstrap_results = {}
    if T_prime > 0:
        for metric_name, metric_values in metrics.items():
            if metric_values is None:
                continue
            vals = []
            for _ in range(bootstrap_iterations):
                sampled_indices = []
                for bi, count in enumerate(target_counts):
                    if count > 0 and len(valid_bin_indices[bi]) > 0:
                        sampled = rng.choice(valid_bin_indices[bi], size=count, replace=True)
                        sampled_indices.extend(sampled)
                if sampled_indices:
                    metric_sample = metric_values[sampled_indices]
                    vals.append(np.mean(metric_sample))
            if vals:
                arr = np.array(vals)
                bootstrap_results[metric_name] = {
                    'point_estimate': float(np.mean(arr)),
                    'ci_low': float(np.percentile(arr, 2.5)),
                    'ci_high': float(np.percentile(arr, 97.5)),
                    'ci_width': float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5)),
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
        'duplication_95p': duplication_95p,
        'small_sample_flag': small_sample_flag,
        'seed': seed,
        'bootstrap_results': bootstrap_results,
    }


def save_target_specification(target_meta: Dict[str, Any], output_file: Path, split: str):
    target_spec = {
        'bin_edges': target_meta['bin_edges'],
        'bin_labels': target_meta['bin_labels'],
        'realized_probs': target_meta['realized_probs'],
        'languages_used_for_target': target_meta['languages_used_for_target'],
        'created_by': 'stage8_arclen_matched_evaluation.py',
        'description': 'Pooled cross-language mean arc-length distribution (quantile bins)',
        'split_for_target': split,
    }
    with open(output_file, 'w') as f:
        json.dump(target_spec, f, indent=2)


def main():
    args = parse_args()

    print("STAGE 8: ARC-LENGTH–MATCHED EVALUATION")
    print("=" * 50)
    print(f"Split: {args.split}")
    print(f"Bootstrap iterations: {args.bootstrap}")
    print(f"Random seed: {args.seed}")
    print(f"Target file: {args.target_file}")
    print(f"Recompute target: {args.recompute_target}")
    print("Objective: Control for structural difficulty via arc-length matching")
    print()

    # Get list of languages from analysis table
    master_file = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    if not master_file.exists():
        raise FileNotFoundError(f"Master analysis table not found: {master_file}")
    df = pd.read_csv(master_file)
    languages = sorted(df['language_slug'].unique())
    probes = ['dist', 'depth']
    layers = ['L5', 'L6', 'L7', 'L8', 'L9', 'L10']

    # 1. Build or load target bins
    target_file_path = Path(args.target_file)
    if (not args.recompute_target) and target_file_path.exists():
        print(f"Loading existing target from: {target_file_path}")
        target = load_target_distribution(str(target_file_path))
        print(f"  ✓ Loaded target with {len(target.languages_used)} languages")
        print(f"  Target probs: {target.target_probs}")
        bins = ArcBins(edges=target.bin_edges, labels=target.bin_labels)
    else:
        print("Computing target bins from test split...")
        bins_obj, meta = compute_pooled_arclen_bins(languages, split='test', K=5)
        save_target_specification(meta, target_file_path, split='test')
        bins = bins_obj
        target = load_target_distribution(str(target_file_path))

    # 2. Matching per language/probe/layer
    print("\n2. PER-LANGUAGE ARC MATCHING")
    print("-" * 40)
    results = []
    diagnostics: Dict[str, Any] = {}

    for lang in languages:
        print(f"Processing {lang}...")
        # Load sentence stats
        try:
            stats = load_sentence_stats(lang, args.split)
        except FileNotFoundError:
            print(f"  ⚠ Sentence stats not found for {lang}, skipping")
            continue
        sentences = stats['sentences']
        N = len(sentences)
        # Robust casting: None or non-numeric -> NaN; num_content_arcs_used -> int with 0 fallback
        arclen_all: List[float] = []
        n_arcs: List[int] = []
        for s in sentences:
            v = s.get('mean_arc_len', None)
            if isinstance(v, (int, float)):
                arclen_all.append(float(v))
            else:
                arclen_all.append(np.nan)
            na = s.get('num_content_arcs_used', 0)
            try:
                n_arcs.append(int(na if na is not None else 0))
            except Exception:
                n_arcs.append(0)
        valid_arc_mask = np.array([na >= 1 and np.isfinite(al) for na, al in zip(n_arcs, arclen_all)], dtype=bool)

        # Sentence IDs (Stage 3 order) for alignment when possible
        stage3_ids = [s.get('sent_id', f"sent_{i}") for i, s in enumerate(sentences)]

        for probe in probes:
            for layer in layers:
                seed = derive_seed(args.seed, lang, probe, layer, args.split)
                metrics = load_per_sentence_metrics(lang, probe, layer, args.split)
                if not metrics:
                    print(f"    ⚠ No metrics found for {lang}/{probe}/{layer}, skipping")
                    continue
                ids_metrics = metrics.pop('_sent_ids', None)
                kept_idx = metrics.pop('_kept_indices', None)

                filtered_metrics: Dict[str, np.ndarray] = {}
                alignment_diagnostics: Dict[str, Any] = {}

                # Only one metric per probe
                for metric_name, metric_values in metrics.items():
                    if metric_values is None:
                        continue
                    # Align metric to full sentence list if needed
                    if len(metric_values) != N:
                        aligned, mode = align_to_full_sentence_list(metric_values, N, valid_arc_mask,
                                                                     ids_metrics, stage3_ids, kept_idx)
                        if aligned is None:
                            print(f"    ⚠ Could not align {metric_name} (len {len(metric_values)} vs {N}); skipping")
                            continue
                        aligned_values = aligned
                    else:
                        aligned_values = metric_values
                        mode = 'full_length'
                    alignment_diagnostics[metric_name] = {'mode': mode, 'hits': int(np.isfinite(aligned_values).sum())}
                    # Apply eligibility + finite
                    filtered_metrics[metric_name] = aligned_values[valid_arc_mask]

                if not filtered_metrics:
                    print(f"    ⚠ No aligned metrics for {lang}/{probe}/{layer}, skipping")
                    continue

                # Bin assignment for valid sentences
                arclen_valid = np.array(arclen_all)[valid_arc_mask].astype(float)

                result = stratified_bootstrap_matching(
                    metrics=filtered_metrics,
                    arclens=arclen_valid.tolist(),
                    bins=bins,
                    target=target,
                    seed=seed,
                    bootstrap_iterations=args.bootstrap,
                    split=args.split
                )
                result['alignment_diagnostics'] = alignment_diagnostics

                row = {
                    'language_slug': lang,
                    'probe': probe,
                    'layer': layer,
                    'split': args.split,
                    'T_raw': result['T_raw'],
                    'T_prime': result['T_prime'],
                    'retention_ratio': result['retention_ratio'],
                    'bins_truncated_count': len(result.get('bins_truncated', [])),
                    'js_to_target': result.get('js_to_target', float('nan')),
                    'duplication_95p': result.get('duplication_95p', 0.0),
                    'valid_bin_counts_json': json.dumps(result.get('valid_bin_counts', [])),
                    'requested_counts_json': json.dumps(result.get('requested_counts', [])),
                    'target_counts_json': json.dumps(result.get('target_counts', [])),
                    'effective_bin_counts_json': json.dumps(result.get('effective_counts', [])),
                    'effective_bin_probs_json': json.dumps(result.get('effective_probs', [])),
                    'seed': result['seed'],
                    'alignment_modes_json': json.dumps({k: v['mode'] for k, v in alignment_diagnostics.items()})
                }

                # Flags and additional audits
                row.update({
                    'retention_low_flag': row['retention_ratio'] < MIN_RETENTION,
                    'small_sample_flag': result.get('small_sample_flag', False),
                    'thin_bins_json': json.dumps([i for i, c in enumerate(result.get('valid_bin_counts', [])) if c < 5]),
                })

                for metric_name, bootstrap_data in result['bootstrap_results'].items():
                    row[f'{metric_name}_arclen_matched'] = bootstrap_data['point_estimate']
                    row[f'{metric_name}_arclen_matched_ci_low'] = bootstrap_data['ci_low']
                    row[f'{metric_name}_arclen_matched_ci_high'] = bootstrap_data['ci_high']
                    row[f'{metric_name}_arclen_matched_ci_width'] = bootstrap_data['ci_width']

                results.append(row)

        # Per-language diagnostics
        lang_diagnostics = {
            'split': args.split,
            'bin_edges': bins.edges,
            'bin_labels': bins.labels,
            'target_probs': target.target_probs,
            'per_probe_layer': {}
        }
        for row in [r for r in results if r['language_slug'] == lang]:
            key = f"{row['probe']}_{row['layer']}"
            lang_diagnostics['per_probe_layer'][key] = {
                'T_raw': row['T_raw'],
                'T_prime': row['T_prime'],
                'retention_ratio': row['retention_ratio'],
                'js_to_target': row['js_to_target'],
                'duplication_95p': row['duplication_95p'],
                'bins_truncated_count': row['bins_truncated_count'],
                # Derive explicit bins_truncated from requested vs target counts
                'bins_truncated': (
                    [i for i, (req, tgt) in enumerate(zip(
                        json.loads(row.get('requested_counts_json', '[]')),
                        json.loads(row.get('target_counts_json', '[]'))
                    )) if tgt < req] if row.get('requested_counts_json') and row.get('target_counts_json') else []
                ),
                'retention_low_flag': row.get('retention_low_flag', False),
                'small_sample_flag': row.get('small_sample_flag', False),
                'valid_bin_counts': json.loads(row.get('valid_bin_counts_json', '[]')),
                'requested_counts': json.loads(row.get('requested_counts_json', '[]')),
                'target_counts': json.loads(row.get('target_counts_json', '[]')),
                'effective_probs': json.loads(row.get('effective_bin_probs_json', '[]')),
                'alignment_modes': json.loads(row.get('alignment_modes_json', '{}')),
            }
        diagnostics[lang] = lang_diagnostics

    # 3. Save outputs
    print("\n3. SAVING OUTPUTS")
    print("-" * 40)
    split_eval_dir = MATCHED_EVAL_DIR / args.split
    split_eval_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_file = ANALYSIS_DIR / f"matched_eval_arclen_per_layer_{args.split}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"  ✓ Results: {output_file} ({len(results_df)} rows)")

    # Canonical copy for test split (align with Stage 7 convention)
    if args.split == 'test':
        canonical_file = ANALYSIS_DIR / "matched_eval_arclen_per_layer.csv"
        results_df.to_csv(canonical_file, index=False)
        print(f"  ✓ Results (canonical for test): {canonical_file}")

    for language_slug, diag_data in diagnostics.items():
        diag_file = split_eval_dir / f"{language_slug}.json"
        with open(diag_file, 'w') as f:
            json.dump(diag_data, f, indent=2)
        print(f"  ✓ Diagnostics: {diag_file}")

    print("\n🎯 STAGE 8 STATUS")
    print("-" * 40)
    print("✅ Target distribution ready and saved")
    print("✅ Arc-length matching framework implemented")
    print("✅ Results and diagnostics written")


if __name__ == "__main__":
    main()



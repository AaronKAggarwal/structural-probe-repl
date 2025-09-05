#!/usr/bin/env python3
"""
Stage 9: Anchored evaluation per layer (UUAS@L<=20 and UUAS@A<=3).

Purpose: Report probe performance on uniformly "easy-ish" subsets across languages
using two simple global anchors close to pooled p80 without any reweighting:
 - L<=20: sentences with 2..len_anchor content tokens
 - A<=3:  sentences with >=1 content arc and mean content-arc length <= arclen_anchor

Outputs two CSVs (len-anchored and arc-anchored) with point estimates and 95% CIs,
coverage diagnostics, and a tiny provenance JSON with the anchors and p80 justification.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Configuration
REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
SENTENCE_STATS_DIR = ANALYSIS_DIR / "sentence_stats"
CHECKS_DIR = ANALYSIS_DIR / "checks"

# Defaults
DEFAULT_LEN_ANCHOR = 20
DEFAULT_ARCLEN_ANCHOR = 3.0
BOOTSTRAP_ITERATIONS = 1000
SMALL_SAMPLE_THRESHOLD = 200
LOW_COVERAGE_THRESHOLD = 0.5


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 9: Anchored evaluation")
    ap.add_argument('--split', choices=['test', 'dev'], default='test',
                    help='Dataset split to evaluate (default: test)')
    ap.add_argument('--bootstrap', type=int, default=BOOTSTRAP_ITERATIONS,
                    help='Number of bootstrap iterations (default: 1000)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Global random seed (default: 42)')
    ap.add_argument('--len_anchor', type=int, default=DEFAULT_LEN_ANCHOR,
                    help='Content length anchor L (evaluate sentences with 2..L tokens)')
    ap.add_argument('--arclen_anchor', type=float, default=DEFAULT_ARCLEN_ANCHOR,
                    help='Mean arc-length anchor A (evaluate sentences with >=1 arc and mean_arc_len<=A)')
    return ap.parse_args()


def derive_seed(base_seed: int, language_slug: str, probe: str, layer: str, split: str, anchor_tag: str) -> int:
    key = f"anchor_{anchor_tag}_{language_slug}_{probe}_{layer}_{split}"
    hash_int = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return (base_seed + hash_int) % (2**31)


def load_sentence_stats(language_slug: str, split: str) -> Dict[str, Any]:
    stats_file = SENTENCE_STATS_DIR / language_slug / f"{split}_content_stats.jsonl"
    if not stats_file.exists():
        raise FileNotFoundError(f"Sentence stats not found: {stats_file}")
    sentences: List[Dict[str, Any]] = []
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


def load_per_sentence_metrics(language_slug: str, probe: str, layer: str, split: str) -> Optional[Dict[str, Any]]:
    run_dir = discover_run_dir(language_slug, probe, layer)
    if not run_dir:
        return None
    if split == 'test':
        test_final_path = Path(run_dir) / "test_detailed_metrics_final.json"
        test_regular_path = Path(run_dir) / "test_detailed_metrics.json"
        metrics_file = test_final_path if test_final_path.exists() else test_regular_path
        if not metrics_file.exists():
            return None
    else:
        metrics_file = Path(run_dir) / "dev_detailed_metrics.json"
        if not metrics_file.exists():
            return None

    try:
        with open(metrics_file, 'r') as f:
            detailed = json.load(f)
    except Exception:
        return None

    out: Dict[str, Any] = {}
    if probe == 'dist':
        # Prefer full-length if present, else compact
        full = detailed.get('uuas_per_sentence_full')
        comp = detailed.get('uuas_per_sentence')
        src = full if (isinstance(full, list) and len(full) > 0) else comp
        if src and isinstance(src, list):
            out['metric'] = np.array(src, dtype=float)
            out['metric_name'] = 'uuas'
        else:
            return None
    elif probe == 'depth':
        # Prefer full-length if present, else compact
        full = detailed.get('root_acc_per_sentence_full')
        comp = detailed.get('root_acc_per_sentence')
        src = full if (isinstance(full, list) and len(full) > 0) else comp
        if src and isinstance(src, list):
            out['metric'] = np.array(src, dtype=float)
            out['metric_name'] = 'root_acc'
        else:
            return None
    else:
        return None

    out['_sent_ids'] = (detailed.get('sentence_ids') or detailed.get('sent_ids'))
    out['_kept_indices'] = (detailed.get('kept_sentence_indices') or
                            detailed.get('valid_sentence_indices') or
                            detailed.get('eval_sentence_indices'))
    return out


def align_to_full_sentence_list(vec: np.ndarray, N: int, eval_mask: np.ndarray,
                                ids_metrics: Optional[List[str]], ids_stage3: List[str],
                                kept_idx: Optional[List[int]]) -> Tuple[Optional[np.ndarray], str]:
    vec = np.asarray(vec, dtype=float)
    full = np.full(N, np.nan, dtype=float)
    # 1) By sentence ids
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
    # 2) By kept indices
    if kept_idx and len(kept_idx) == len(vec):
        hits = 0
        for dst, src in zip(kept_idx, range(len(vec))):
            if 0 <= dst < N:
                full[dst] = vec[src]
                hits += 1
        if hits > 0:
            return full, 'by_indices'
    # 3) Fallback by mask length
    if len(vec) == int(eval_mask.sum()):
        idx = np.where(eval_mask)[0]
        if len(idx) == len(vec):
            full[idx] = vec
            return full, 'by_mask_fallback'
    return None, 'failed'


def compute_anchor_justification(languages: List[str], split: str) -> Dict[str, Any]:
    # Compute pooled p80s for content_len (>=2) and mean_arc_len (>=1 arc)
    content_lens: List[int] = []
    arc_lens: List[float] = []
    langs_used_len: List[str] = []
    langs_used_arc: List[str] = []
    for lang in languages:
        try:
            stats = load_sentence_stats(lang, split)
        except FileNotFoundError:
            continue
        sentences = stats['sentences']
        # Content length
        vals_len = [int(s.get('content_len', 0)) for s in sentences if int(s.get('content_len', 0)) >= 2]
        if vals_len:
            content_lens.extend(vals_len)
            langs_used_len.append(lang)
        # Arc length
        vals_arc = []
        for s in sentences:
            na = int(s.get('num_content_arcs_used', 0) or 0)
            ma = s.get('mean_arc_len', None)
            if na >= 1 and isinstance(ma, (int, float)):
                vals_arc.append(float(ma))
        if vals_arc:
            arc_lens.extend(vals_arc)
            langs_used_arc.append(lang)
    out = {
        'pooled_p80_content_len': float(np.quantile(np.array(content_lens, dtype=float), 0.8)) if content_lens else None,
        'pooled_p80_mean_arclen': float(np.quantile(np.array(arc_lens, dtype=float), 0.8)) if arc_lens else None,
        'languages_used_len': langs_used_len,
        'languages_used_arc': langs_used_arc,
    }
    return out


def bootstrap_mean(values: np.ndarray, rng: np.random.RandomState, B: int) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {'point_estimate': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'ci_width': float('nan')}
    pe = float(np.mean(values))
    vals = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        vals.append(float(np.mean(values[idx])))
    arr = np.array(vals, dtype=float)
    return {
        'point_estimate': pe,
        'ci_low': float(np.percentile(arr, 2.5)),
        'ci_high': float(np.percentile(arr, 97.5)),
        'ci_width': float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5)),
    }


def main():
    args = parse_args()

    print("STAGE 9: ANCHORED EVALUATION")
    print("=" * 50)
    print(f"Split: {args.split}")
    print(f"Bootstrap iterations: {args.bootstrap}")
    print(f"Random seed: {args.seed}")
    print(f"Len anchor (L): <= {args.len_anchor}")
    print(f"Arc anchor (A): <= {args.arclen_anchor}")
    print()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)

    master_file = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    if not master_file.exists():
        raise FileNotFoundError(f"Master analysis table not found: {master_file}")
    df_master = pd.read_csv(master_file)
    languages = sorted(df_master['language_slug'].unique())
    probes = ['dist', 'depth']
    layers = ['L5', 'L6', 'L7', 'L8', 'L9', 'L10']

    # Provenance: compute pooled p80s for justification
    justification = compute_anchor_justification(languages, args.split)
    anchors_json = {
        'len_anchor': int(args.len_anchor),
        'arclen_anchor': float(args.arclen_anchor),
        'justification': {
            'pooled_p80_content_len': justification.get('pooled_p80_content_len'),
            'pooled_p80_mean_arclen': justification.get('pooled_p80_mean_arclen'),
        },
        'created_by': 'stage9_anchored_eval.py'
    }
    with open(ANALYSIS_DIR / 'anchors_used.json', 'w') as f:
        json.dump(anchors_json, f, indent=2)

    # Containers
    rows_len: List[Dict[str, Any]] = []
    rows_arc: List[Dict[str, Any]] = []

    # Iterate languages
    for lang in languages:
        print(f"Processing {lang}...")
        try:
            stats = load_sentence_stats(lang, args.split)
        except FileNotFoundError:
            print(f"  ⚠ Sentence stats not found for {lang}, skipping")
            continue
        sentences = stats['sentences']
        N = len(sentences)
        # Stage 3 order sentence ids
        stage3_ids = [s.get('sent_id', f"sent_{i}") for i, s in enumerate(sentences)]
        # Content length arrays
        content_len = np.array([int(s.get('content_len', 0) or 0) for s in sentences], dtype=int)
        # Arc stats
        num_arcs = np.array([int(s.get('num_content_arcs_used', 0) or 0) for s in sentences], dtype=int)
        mean_arclen = np.empty(N, dtype=float)
        for i, s in enumerate(sentences):
            ma_key = 'mean_arc_len' if ('mean_arc_len' in s) else ('mean_content_arc_len' if ('mean_content_arc_len' in s) else None)
            v = s.get(ma_key, None) if ma_key is not None else None
            mean_arclen[i] = float(v) if isinstance(v, (int, float)) else np.nan

        # Eligibility masks (sentence-level)
        mask_len_anchor = (content_len >= 2) & (content_len <= int(args.len_anchor))
        mask_arclen_anchor = (num_arcs >= 1) & np.isfinite(mean_arclen) & (mean_arclen <= float(args.arclen_anchor))
        # Base evaluability mask for fallback alignment (avoid anchor-union)
        base_eval_mask = (content_len >= 2)

        for probe in probes:
            for layer in layers:
                metrics = load_per_sentence_metrics(lang, probe, layer, args.split)
                if not metrics:
                    print(f"  ⚠ No metrics for {lang}/{probe}/{layer}, skipping")
                    continue
                vec = metrics['metric']
                metric_name = metrics['metric_name']
                ids_metrics = metrics.get('_sent_ids', None)
                kept_idx = metrics.get('_kept_indices', None)

                # Align to full sentence list
                if len(vec) != N:
                    aligned, mode = align_to_full_sentence_list(vec, N, base_eval_mask, ids_metrics, stage3_ids, kept_idx)
                    if aligned is None:
                        print(f"  ⚠ Could not align metric (len {len(vec)} vs {N}) for {lang}/{probe}/{layer}, skipping")
                        continue
                    full_values = aligned
                else:
                    full_values = vec.astype(float)
                    mode = 'full_length'

                # Base finite mask
                finite_mask = np.isfinite(full_values)
                T_base = int(finite_mask.sum())

                # --- L anchor ---
                seed_L = derive_seed(args.seed, lang, probe, layer, args.split, anchor_tag='L')
                rng_L = np.random.RandomState(seed_L)
                elig_L_mask = mask_len_anchor & finite_mask
                T_eligible_L = int(elig_L_mask.sum())
                if T_eligible_L > 0:
                    elig_vals_L = full_values[elig_L_mask]
                    stats_L = bootstrap_mean(elig_vals_L, rng_L, args.bootstrap)
                    rows_len.append({
                        'language_slug': lang,
                        'probe': probe,
                        'layer': layer,
                        'split': args.split,
                        'anchor_type': 'L',
                        'anchor_value': int(args.len_anchor),
                        'point_estimate': stats_L['point_estimate'],
                        'ci_low': stats_L['ci_low'],
                        'ci_high': stats_L['ci_high'],
                        'ci_width': stats_L['ci_width'],
                        'N_total': N,
                        'T_base': T_base,
                        'T_eligible': T_eligible_L,
                        'coverage_all': float(T_eligible_L / max(N, 1)),
                        'coverage_base': float(T_eligible_L / max(T_base, 1)),
                        'small_sample_flag': bool(T_eligible_L < SMALL_SAMPLE_THRESHOLD),
                        'low_coverage_flag': bool((T_eligible_L / max(N, 1)) < LOW_COVERAGE_THRESHOLD),
                        'seed': seed_L,
                        'alignment_mode': mode,
                        'metric': metric_name,
                        'probe_metric': metric_name,
                    })
                else:
                    rows_len.append({
                        'language_slug': lang,
                        'probe': probe,
                        'layer': layer,
                        'split': args.split,
                        'anchor_type': 'L',
                        'anchor_value': int(args.len_anchor),
                        'point_estimate': np.nan,
                        'ci_low': np.nan,
                        'ci_high': np.nan,
                        'ci_width': np.nan,
                        'N_total': N,
                        'T_base': T_base,
                        'T_eligible': 0,
                        'coverage_all': float(0.0),
                        'coverage_base': float(0.0),
                        'small_sample_flag': True,
                        'low_coverage_flag': True,
                        'seed': seed_L,
                        'alignment_mode': mode,
                        'metric': metric_name,
                        'probe_metric': metric_name,
                        'skip_reason': 'no_eligible',
                    })

                # --- A anchor ---
                seed_A = derive_seed(args.seed, lang, probe, layer, args.split, anchor_tag='A')
                rng_A = np.random.RandomState(seed_A)
                elig_A_mask = mask_arclen_anchor & finite_mask
                T_eligible_A = int(elig_A_mask.sum())
                if T_eligible_A > 0:
                    elig_vals_A = full_values[elig_A_mask]
                    stats_A = bootstrap_mean(elig_vals_A, rng_A, args.bootstrap)
                    rows_arc.append({
                        'language_slug': lang,
                        'probe': probe,
                        'layer': layer,
                        'split': args.split,
                        'anchor_type': 'A',
                        'anchor_value': float(args.arclen_anchor),
                        'point_estimate': stats_A['point_estimate'],
                        'ci_low': stats_A['ci_low'],
                        'ci_high': stats_A['ci_high'],
                        'ci_width': stats_A['ci_width'],
                        'N_total': N,
                        'T_base': T_base,
                        'T_eligible': T_eligible_A,
                        'coverage_all': float(T_eligible_A / max(N, 1)),
                        'coverage_base': float(T_eligible_A / max(T_base, 1)),
                        'small_sample_flag': bool(T_eligible_A < SMALL_SAMPLE_THRESHOLD),
                        'low_coverage_flag': bool((T_eligible_A / max(N, 1)) < LOW_COVERAGE_THRESHOLD),
                        'seed': seed_A,
                        'alignment_mode': mode,
                        'metric': metric_name,
                        'probe_metric': metric_name,
                    })
                else:
                    rows_arc.append({
                        'language_slug': lang,
                        'probe': probe,
                        'layer': layer,
                        'split': args.split,
                        'anchor_type': 'A',
                        'anchor_value': float(args.arclen_anchor),
                        'point_estimate': np.nan,
                        'ci_low': np.nan,
                        'ci_high': np.nan,
                        'ci_width': np.nan,
                        'N_total': N,
                        'T_base': T_base,
                        'T_eligible': 0,
                        'coverage_all': float(0.0),
                        'coverage_base': float(0.0),
                        'small_sample_flag': True,
                        'low_coverage_flag': True,
                        'seed': seed_A,
                        'alignment_mode': mode,
                        'metric': metric_name,
                        'probe_metric': metric_name,
                        'skip_reason': 'no_eligible',
                    })

    # Save outputs
    out_len = ANALYSIS_DIR / f"uuas_at_len_anchor_per_layer_{args.split}.csv"
    out_arc = ANALYSIS_DIR / f"uuas_at_arclen_anchor_per_layer_{args.split}.csv"
    df_len = pd.DataFrame(rows_len)
    df_arc = pd.DataFrame(rows_arc)
    df_len.to_csv(out_len, index=False)
    df_arc.to_csv(out_arc, index=False)
    print(f"\n✓ Wrote: {out_len} ({len(df_len)} rows)")
    print(f"✓ Wrote: {out_arc} ({len(df_arc)} rows)")

    if args.split == 'test':
        df_len.to_csv(ANALYSIS_DIR / "uuas_at_len_anchor_per_layer.csv", index=False)
        df_arc.to_csv(ANALYSIS_DIR / "uuas_at_arclen_anchor_per_layer.csv", index=False)
        print("✓ Wrote canonical copies for test split")

    # QC summaries
    def qc_summary(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        out: Dict[str, Any] = {
            'rows': int(len(df)),
            'coverage_all_quantiles': {str(k): float(v) for k, v in df['coverage_all'].quantile([0,.05,.25,.5,.75,.95,1]).to_dict().items()},
            'coverage_base_quantiles': {str(k): float(v) for k, v in df['coverage_base'].quantile([0,.05,.25,.5,.75,.95,1]).to_dict().items()},
            'small_sample_count': int(df['small_sample_flag'].sum()),
            'low_coverage_count': int(df['low_coverage_flag'].sum()),
        }
        if 'ci_width' in df.columns and df['ci_width'].notna().any():
            by_probe = {}
            for probe in ['dist','depth']:
                sub = df[df['probe']==probe]
                if not sub.empty and sub['ci_width'].notna().any():
                    by_probe[probe] = {
                        'ci_width_quantiles': {str(k): float(v) for k, v in sub['ci_width'].quantile([0,.25,.5,.75,.95,1]).to_dict().items()}
                    }
            out['by_probe'] = by_probe
        return out

    qc_len = qc_summary(df_len)
    qc_arc = qc_summary(df_arc)
    (CHECKS_DIR / f"stage9_qc_len_{args.split}.json").write_text(json.dumps(qc_len, indent=2))
    (CHECKS_DIR / f"stage9_qc_arclen_{args.split}.json").write_text(json.dumps(qc_arc, indent=2))
    print("\nQC summaries written to outputs/analysis/checks/")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Select a syntax band (contiguous set of layers) on DEV only via paired bootstraps.

Research-mode behavior (no fallbacks):
- Primary metrics: distance → UUAS; depth → RootAcc (content-only, per-sentence lists)
- For each language and layer, bootstrap B resamples from per-sentence metrics
- Equivalence rule (default): P(|gap| <= Δ) ≥ 0.95, where gap = mean(best) - mean(L)
- Aggregate per-layer coverage r_k = fraction of languages where equivalent
- Select the largest contiguous block with r_k ≥ τ; tiebreak by higher macro mean
- Save selection and detailed provenance to outputs/training_logs/syntax_band_selection.json

Discovery of per-language, per-layer dev detailed metrics:
- Reads from per-language run directories:
  outputs/baselines_auto/UD_*/bert-base-multilingual-cased/{dist,depth}/L{5..10}/runs/{run_id}/dev_detailed_metrics.json
- Language slug is the UD directory name (e.g., UD_English-EWT)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from statistics import mean, stdev
import math
from typing import Dict, List, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_ROOT = os.path.join(REPO_ROOT, 'outputs')
BASE_AUTO_ROOT = os.path.join(OUTPUT_ROOT, 'baselines_auto', 'bert-base-multilingual-cased')
BASE_PER_LANG_ROOT = os.path.join(OUTPUT_ROOT, 'baselines_auto')
TRAIN_LOG = 'train_probe.log'
METRICS_SUMMARY = 'metrics_summary.json'
DEV_DETAILED_TPL = 'dev_detailed_metrics_epoch{epoch}.json'
ALL_RESULTS_CSV = os.path.join(OUTPUT_ROOT, 'training_logs', 'all_results.csv')
SELECTION_OUT = os.path.join(OUTPUT_ROOT, 'training_logs', 'syntax_band_selection.json')


LAYER_ORDER = ['L5', 'L6', 'L7', 'L8', 'L9', 'L10']
DEFAULT_CANDIDATE_WINDOW: List[str] = []  # Unused in research mode (no fallback)


@dataclass
class LangLayerMetrics:
    language: str
    layer: str
    per_sentence: List[float]


def parse_language_from_log(log_path: str) -> str | None:
    try:
        with open(log_path, 'r') as f:
            txt = f.read()
        # Match 'dataset:\n  name: XYZ' or 'Dataset:\n     Name              : XYZ'
        m = re.search(r"dataset:\s*\n\s*name:\s*([^\n]+)", txt, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"Name\s*:\s*([^\n]+)", txt)
        if m:
            return m.group(1).strip()
    except Exception:
        return None
    return None


def discover_dev_per_sentence(probe: str) -> Tuple[Dict[str, Dict[str, LangLayerMetrics]], Dict[str, Dict[str, str]]]:
    """Return mapping language_slug -> layer -> metrics with per-sentence lists, and run paths used.

    probe: 'dist' or 'depth'
    """
    assert probe in {'dist', 'depth'}

    # First try per-language run directories (preferred, robust discovery)
    lang_to_layer: Dict[str, Dict[str, LangLayerMetrics]] = defaultdict(dict)
    run_paths: Dict[str, Dict[str, str]] = defaultdict(dict)
    try:
        if os.path.isdir(BASE_PER_LANG_ROOT):
            for lang_dir in sorted(glob(os.path.join(BASE_PER_LANG_ROOT, 'UD_*'))):
                language_slug = os.path.basename(lang_dir)
                model_root = os.path.join(lang_dir, 'bert-base-multilingual-cased', probe)
                if not os.path.isdir(model_root):
                    continue
                for layer in LAYER_ORDER:
                    layer_dir = os.path.join(model_root, layer)
                    runs_dir = os.path.join(layer_dir, 'runs')
                    if not os.path.isdir(runs_dir):
                        continue
                    # Prefer 'latest' if present, otherwise use any run with dev_detailed_metrics.json
                    candidate_dirs = []
                    latest_dir = os.path.join(runs_dir, 'latest')
                    if os.path.exists(latest_dir):
                        try:
                            candidate_dirs.append(os.path.realpath(latest_dir))
                        except Exception:
                            candidate_dirs.append(latest_dir)
                    for rd in sorted(glob(os.path.join(runs_dir, '*'))):
                        bn = os.path.basename(rd)
                        if bn in {'.DS_Store', 'latest'}:
                            continue
                        if os.path.isdir(rd):
                            candidate_dirs.append(rd)
                    chosen_run = None
                    for rd in candidate_dirs:
                        if os.path.exists(os.path.join(rd, 'dev_detailed_metrics.json')):
                            chosen_run = rd
                            break
                    if not chosen_run:
                        continue
                    try:
                        with open(os.path.join(chosen_run, 'dev_detailed_metrics.json'), 'r') as f:
                            dd = json.load(f)
                        if probe == 'dist':
                            per_sent = dd.get('uuas_per_sentence', [])
                        else:
                            per_sent = dd.get('root_acc_per_sentence', [])
                        if not per_sent:
                            continue
                        lang_to_layer[language_slug][layer] = LangLayerMetrics(
                            language=language_slug, layer=layer, per_sentence=per_sent
                        )
                        run_paths[language_slug][layer] = chosen_run
                    except Exception:
                        continue
    except Exception:
        # If any unexpected error, fall back to legacy discovery below
        lang_to_layer = defaultdict(dict)

    # If per-language discovery yielded too few languages, fall back to legacy layout
    if len(lang_to_layer) < 5:
        root = os.path.join(BASE_AUTO_ROOT, probe)
        for layer in LAYER_ORDER:
            layer_dir = os.path.join(root, layer)
            if not os.path.isdir(layer_dir):
                continue
            # Identify language via train log
            lang = parse_language_from_log(os.path.join(layer_dir, TRAIN_LOG))
            if not lang:
                continue
            # Read best epoch from metrics_summary.json
            summary_path = os.path.join(layer_dir, METRICS_SUMMARY)
            if not os.path.exists(summary_path):
                continue
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                best_epoch = summary.get('best_model_epoch_completed')
                if not isinstance(best_epoch, int):
                    detailed_jsons = glob(os.path.join(layer_dir, 'dev_detailed_metrics_epoch*.json'))
                    epochs = []
                    for p in detailed_jsons:
                        m = re.search(r"epoch(\d+)\.json$", p)
                        if m:
                            epochs.append(int(m.group(1)))
                    best_epoch = max(epochs) if epochs else None
            except Exception:
                best_epoch = None
            if best_epoch is None:
                continue
            dev_detail_path = os.path.join(layer_dir, DEV_DETAILED_TPL.format(epoch=best_epoch))
            if not os.path.exists(dev_detail_path):
                continue
            try:
                with open(dev_detail_path, 'r') as f:
                    dd = json.load(f)
                if probe == 'dist':
                    per_sent = dd.get('uuas_per_sentence', [])
                else:
                    per_sent = dd.get('root_acc_per_sentence', [])
                if not per_sent:
                    continue
                lang_to_layer[lang][layer] = LangLayerMetrics(language=lang, layer=layer, per_sentence=per_sent)
                run_paths.setdefault(lang, {})[layer] = layer_dir
            except Exception:
                continue

    return lang_to_layer, run_paths


def paired_bootstrap_equivalence(
    lang_to_layer: Dict[str, Dict[str, LangLayerMetrics]],
    delta: float,
    b: int = 1000,
    seed: int = 13,
    use_abs_equivalence: bool = True,
    min_T: int = 30,
    delta_scheme: str = 'absolute',
) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, bool],
    Dict[str, Dict[str, bool]], Dict[str, bool], Dict[str, bool], int
]:
    """
    Returns:
      - coverage per layer (fraction of available languages equivalent to best)
      - macro mean per layer (mean across available languages)
      - best layer rate (fraction over languages used)
      - truncation_flag_by_language
      - equivalent_layers_by_language
      - skipped_min_T_by_language
      - dropped_nonfinite_by_language
      - num_languages_used (after gating)
    """
    rng = random.Random(seed)
    coverage: Dict[str, float] = {L: 0.0 for L in LAYER_ORDER}
    macro_means: Dict[str, float] = {L: 0.0 for L in LAYER_ORDER}
    best_layer_counts: Dict[str, int] = {L: 0 for L in LAYER_ORDER}
    avail_counts: Dict[str, int] = {L: 0 for L in LAYER_ORDER}
    truncation_by_lang: Dict[str, bool] = {}
    skipped_minT_by_lang: Dict[str, bool] = {}
    dropped_nonfinite_by_lang: Dict[str, bool] = {}
    eq_flags_by_lang: Dict[str, Dict[str, bool]] = {}

    languages = [lang for lang, d in lang_to_layer.items() if any(L in d for L in LAYER_ORDER)]
    num_used = 0

    for lang in languages:
        layer_map = lang_to_layer[lang]
        avail_layers = [L for L in LAYER_ORDER if L in layer_map]
        if not avail_layers:
            continue
        # Validate per-sentence finiteness
        finite_ok = True
        for L in avail_layers:
            for v in layer_map[L].per_sentence:
                try:
                    fv = float(v)
                except Exception:
                    finite_ok = False
                    break
                if not math.isfinite(fv):
                    finite_ok = False
                    break
            if not finite_ok:
                break
        if not finite_ok:
            dropped_nonfinite_by_lang[lang] = True
            continue

        lengths = [len(layer_map[L].per_sentence) for L in avail_layers]
        T = min(lengths)
        truncation_by_lang[lang] = not all(Ln == lengths[0] for Ln in lengths)
        if T < min_T:
            skipped_minT_by_lang[lang] = True
            continue

        arrays = {L: layer_map[L].per_sentence[:T] for L in avail_layers}
        per_layer_mean = {L: (sum(arrays[L]) / T if T > 0 else 0.0) for L in avail_layers}
        best_layer = max(per_layer_mean.items(), key=lambda kv: kv[1])[0]
        best_layer_counts[best_layer] += 1

        # Effective delta
        if delta_scheme == 'relative':
            try:
                sd_val = stdev(list(per_layer_mean.values())) if len(per_layer_mean) > 1 else 0.0
            except Exception:
                sd_val = 0.0
            delta_eff = float(delta) * float(sd_val)
        else:
            delta_eff = float(delta)

        eq_prob: Dict[str, float] = {}
        for L in avail_layers:
            avail_counts[L] += 1
            if L == best_layer:
                eq_prob[L] = 1.0
                continue
            count_le = 0
            for _ in range(b):
                idxs = [rng.randrange(T) for _ in range(T)]
                mean_best = sum(arrays[best_layer][i] for i in idxs) / T
                mean_L = sum(arrays[L][i] for i in idxs) / T
                gap = mean_best - mean_L
                ok = (abs(gap) <= delta_eff) if use_abs_equivalence else (gap <= delta_eff)
                if ok:
                    count_le += 1
            eq_prob[L] = count_le / b

        eq_flags_by_lang[lang] = {}
        for L in avail_layers:
            if eq_prob.get(L, 0.0) >= 0.95:
                coverage[L] += 1.0
                eq_flags_by_lang[lang][L] = True
            else:
                eq_flags_by_lang[lang][L] = False
            macro_means[L] += per_layer_mean[L]

        num_used += 1

    for L in LAYER_ORDER:
        denom = max(avail_counts[L], 1)
        coverage[L] = coverage[L] / denom
        macro_means[L] = macro_means[L] / denom if macro_means[L] != 0 else 0.0

    best_layer_rate = {L: (best_layer_counts[L] / num_used) if num_used > 0 else 0.0 for L in LAYER_ORDER}
    return (
        coverage,
        macro_means,
        best_layer_rate,
        truncation_by_lang,
        eq_flags_by_lang,
        skipped_minT_by_lang,
        dropped_nonfinite_by_lang,
        num_used,
    )


def fallback_from_all_results(probe: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Fallback removed in research mode; keep stub for compatibility if imported elsewhere
    return {L: 0.0 for L in LAYER_ORDER}, {L: 0.0 for L in LAYER_ORDER}


def _equivalence_flags_for_language(
    layer_map: Dict[str, LangLayerMetrics],
    delta: float,
    b: int,
    seed: int,
    use_abs_equivalence: bool,
) -> Tuple[Dict[str, bool], Dict[str, float], str, bool]:
    rng = random.Random(seed)
    avail_layers = [L for L in LAYER_ORDER if L in layer_map]
    if not avail_layers:
        return {}, {}, '', False
    per_layer_mean = {L: mean(layer_map[L].per_sentence) for L in avail_layers}
    best_layer = max(per_layer_mean.items(), key=lambda kv: kv[1])[0]
    lengths = [len(layer_map[L].per_sentence) for L in avail_layers]
    T = min(lengths)
    trunc = not all(Ln == lengths[0] for Ln in lengths)
    if T <= 1:
        # Degenerate; mark only best as equivalent
        flags = {L: (L == best_layer) for L in avail_layers}
        return flags, per_layer_mean, best_layer, trunc
    arrays = {L: layer_map[L].per_sentence[:T] for L in avail_layers}
    flags: Dict[str, bool] = {}
    for L in avail_layers:
        if L == best_layer:
            flags[L] = True
            continue
        count_ok = 0
        for _ in range(b):
            idxs = [rng.randrange(T) for _ in range(T)]
            mean_best = sum(arrays[best_layer][i] for i in idxs) / T
            mean_L = sum(arrays[L][i] for i in idxs) / T
            gap = mean_best - mean_L
            ok = (abs(gap) <= delta) if use_abs_equivalence else (gap <= delta)
            if ok:
                count_ok += 1
        flags[L] = (count_ok / b) >= 0.95
    return flags, per_layer_mean, best_layer, trunc


def select_band_from_coverage(coverage: Dict[str, float], macro_means: Dict[str, float], tau: float) -> List[str]:
    # Layers that meet coverage threshold
    eligible = [L for L in LAYER_ORDER if coverage.get(L, 0.0) >= tau]
    if not eligible:
        return []
    # Find contiguous blocks
    blocks: List[List[str]] = []
    current: List[str] = []
    for L in LAYER_ORDER:
        if L in eligible:
            current.append(L)
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)

    # Choose the largest block; tiebreak by higher sum(macro_means)
    blocks.sort(key=lambda b: (len(b), sum(macro_means.get(L, 0.0) for L in b)), reverse=True)
    return blocks[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', choices=['dist', 'depth', 'both'], required=True)
    ap.add_argument('--delta', type=float, default=0.02, help='Equivalence margin when using absolute scheme')
    ap.add_argument('--tau', type=float, default=0.60, help='Coverage threshold across languages')
    ap.add_argument('--bootstrap', type=int, default=1000, help='Number of bootstrap resamples')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--equivalence', choices=['abs', 'one-sided'], default='abs', help='Equivalence rule: abs uses P(|gap|<=delta)')
    ap.add_argument('--delta_scheme', choices=['absolute', 'relative'], default='absolute', help='Delta scheme; absolute or relative (k*SD across layers)')
    ap.add_argument('--min_T', type=int, default=30, help='Minimum paired sample size T per language')
    ap.add_argument('--mode', choices=['single', 'joint_and'], default='single', help='Band selection mode')
    args = ap.parse_args()

    if args.mode == 'single' and args.probe in {'dist', 'depth'}:
        lang_to_layer, run_paths = discover_dev_per_sentence(args.probe)
        (
            coverage,
            macro_means,
            best_rate,
            trunc_by_lang,
            eq_flags_by_lang,
            skipped_minT_by_lang,
            dropped_nonfinite_by_lang,
            num_used,
        ) = paired_bootstrap_equivalence(
            lang_to_layer,
            delta=args.delta,
            b=args.bootstrap,
            seed=args.seed,
            use_abs_equivalence=(args.equivalence == 'abs'),
            min_T=args.min_T,
            delta_scheme=args.delta_scheme,
        )

        band_layers = select_band_from_coverage(coverage, macro_means, args.tau)

        languages_considered = sorted(lang_to_layer.keys())
        layer_availability = {lang: sorted([L for L in LAYER_ORDER if L in layer_map]) for lang, layer_map in lang_to_layer.items()}
        result = {
            'mode': 'single',
            'probe': args.probe,
            'delta': args.delta,
            'tau': args.tau,
            'bootstrap': args.bootstrap,
            'seed': args.seed,
            'equivalence_rule': 'P(|gap| <= delta) >= 0.95' if args.equivalence == 'abs' else 'P(gap <= delta) >= 0.95',
            'delta_scheme': (f"absolute_{args.delta}" if args.delta_scheme == 'absolute' else f"relative_{args.delta}xSD"),
            'min_T': args.min_T,
            'layers_order': LAYER_ORDER,
            'coverage_by_layer': coverage,
            'macro_mean_by_layer': macro_means,
            'best_layer_rate': best_rate,
            'selected_band': band_layers,
            'num_languages_used': num_used,
            'languages_considered': languages_considered,
            'layer_availability_by_language': layer_availability,
            'truncation_flag_by_language': trunc_by_lang,
            'skipped_min_T_by_language': skipped_minT_by_lang,
            'dropped_nonfinite_by_language': dropped_nonfinite_by_lang,
            'num_languages_truncated': sum(1 for v in trunc_by_lang.values() if v),
            'num_languages_skipped_min_T': len(skipped_minT_by_lang),
            'num_languages_dropped_nonfinite': len(dropped_nonfinite_by_lang),
            'equivalent_layers_by_language': eq_flags_by_lang,
            'run_paths_by_language_layer': run_paths,
        }
    else:
        # Joint AND rule across dist and depth
        dist_map, dist_paths = discover_dev_per_sentence('dist')
        depth_map, depth_paths = discover_dev_per_sentence('depth')
        languages = sorted(set(dist_map.keys()) & set(depth_map.keys()))
        use_abs = (args.equivalence == 'abs')

        # Aggregates
        joint_cov: Dict[str, float] = {L: 0.0 for L in LAYER_ORDER}
        dist_cov: Dict[str, float] = {L: 0.0 for L in LAYER_ORDER}
        depth_cov: Dict[str, float] = {L: 0.0 for L in LAYER_ORDER}
        dist_macro: Dict[str, float] = {L: 0.0 for L in LAYER_ORDER}
        depth_macro: Dict[str, float] = {L: 0.0 for L in LAYER_ORDER}
        dist_best_counts: Dict[str, int] = {L: 0 for L in LAYER_ORDER}
        depth_best_counts: Dict[str, int] = {L: 0 for L in LAYER_ORDER}
        trunc_by_lang: Dict[str, bool] = {}
        layer_availability: Dict[str, List[str]] = {}

        used_languages: List[str] = []
        num_used = 0
        skipped_minT_by_lang: Dict[str, bool] = {}
        dropped_nonfinite_by_lang: Dict[str, bool] = {}
        for lang in languages:
            dist_layer_map = dist_map.get(lang, {})
            depth_layer_map = depth_map.get(lang, {})
            # per-language availability as intersection
            avail = [L for L in LAYER_ORDER if (L in dist_layer_map and L in depth_layer_map)]
            if not avail:
                continue
            layer_availability[lang] = avail

            dist_flags, dist_means, dist_best, trunc_d = _equivalence_flags_for_language(
                dist_layer_map, args.delta, args.bootstrap, args.seed, use_abs
            )
            depth_flags, depth_means, depth_best, trunc_p = _equivalence_flags_for_language(
                depth_layer_map, args.delta, args.bootstrap, args.seed, use_abs
            )
            trunc_by_lang[lang] = trunc_d or trunc_p
            # Gate: finiteness and min_T for joint path
            # Re-check finiteness explicitly across probes
            finite_ok = True
            for L in avail:
                for v in dist_layer_map[L].per_sentence:
                    try:
                        fv = float(v)
                    except Exception:
                        finite_ok = False
                        break
                    if not math.isfinite(fv):
                        finite_ok = False
                        break
                if not finite_ok:
                    break
                for v in depth_layer_map[L].per_sentence:
                    try:
                        fv = float(v)
                    except Exception:
                        finite_ok = False
                        break
                    if not math.isfinite(fv):
                        finite_ok = False
                        break
                if not finite_ok:
                    break
            if not finite_ok:
                dropped_nonfinite_by_lang[lang] = True
                continue
            T_dist = min(len(dist_layer_map[L].per_sentence) for L in avail)
            T_depth = min(len(depth_layer_map[L].per_sentence) for L in avail)
            if T_dist < args.min_T or T_depth < args.min_T:
                skipped_minT_by_lang[lang] = True
                continue

            # accumulate macro means
            for L in avail:
                dist_macro[L] += dist_means.get(L, 0.0)
                depth_macro[L] += depth_means.get(L, 0.0)

            if dist_best:
                dist_best_counts[dist_best] += 1
            if depth_best:
                depth_best_counts[depth_best] += 1

            # coverage
            for L in avail:
                if dist_flags.get(L, False):
                    dist_cov[L] += 1.0
                if depth_flags.get(L, False):
                    depth_cov[L] += 1.0
                if dist_flags.get(L, False) and depth_flags.get(L, False):
                    joint_cov[L] += 1.0

            used_languages.append(lang)
            num_used += 1

        for L in LAYER_ORDER:
            # Normalize by availability among used languages only
            dden = sum(1 for lang in used_languages if L in dist_map.get(lang, {})) or 1
            pden = sum(1 for lang in used_languages if L in depth_map.get(lang, {})) or 1
            jden = sum(1 for lang in used_languages if (L in dist_map.get(lang, {}) and L in depth_map.get(lang, {}))) or 1
            dist_cov[L] = dist_cov[L] / dden
            depth_cov[L] = depth_cov[L] / pden
            joint_cov[L] = joint_cov[L] / jden
            dist_macro[L] = dist_macro[L] / dden if dist_macro[L] != 0 else 0.0
            depth_macro[L] = depth_macro[L] / pden if depth_macro[L] != 0 else 0.0

        band_layers = select_band_from_coverage(joint_cov, {L: (dist_macro[L] + depth_macro[L]) / 2.0 for L in LAYER_ORDER}, args.tau)

        result = {
            'mode': 'joint_and',
            'probe': 'both',
            'delta': args.delta,
            'tau': args.tau,
            'bootstrap': args.bootstrap,
            'seed': args.seed,
            'equivalence_rule': 'P(|gap| <= delta) >= 0.95' if args.equivalence == 'abs' else 'P(gap <= delta) >= 0.95',
            'delta_scheme': f"absolute_{args.delta}",
            'layers_order': LAYER_ORDER,
            'coverage_by_layer': joint_cov,
            'coverage_by_layer_joint': joint_cov,
            'probe_coverage_by_layer': {'dist': dist_cov, 'depth': depth_cov},
            'macro_mean_by_layer': {'dist': dist_macro, 'depth': depth_macro},
            'best_layer_rate': {'dist': {L: (dist_best_counts[L] / float(num_used)) if num_used else 0.0 for L in LAYER_ORDER}, 'depth': {L: (depth_best_counts[L] / float(num_used)) if num_used else 0.0 for L in LAYER_ORDER}},
            'selected_band': band_layers,
            'num_languages_used': int(num_used),
            'languages_considered': languages,
            'layer_availability_by_language': layer_availability,
            'truncation_flag_by_language': trunc_by_lang,
            'skipped_min_T_by_language': skipped_minT_by_lang,
            'dropped_nonfinite_by_language': dropped_nonfinite_by_lang,
            'num_languages_truncated': sum(1 for v in trunc_by_lang.values() if v),
            'num_languages_skipped_min_T': len(skipped_minT_by_lang),
            'num_languages_dropped_nonfinite': len(dropped_nonfinite_by_lang),
            'probe_equivalent_layers_by_language': {
                'dist': {lang: {L: True for L in layer_availability.get(lang, []) if L in dist_map.get(lang, {}) and dist_flags.get(L, False)} for lang in used_languages},
                'depth': {lang: {L: True for L in layer_availability.get(lang, []) if L in depth_map.get(lang, {}) and depth_flags.get(L, False)} for lang in used_languages},
            },
            'run_paths_by_language_layer': {'dist': dist_paths, 'depth': depth_paths},
        }

    os.makedirs(os.path.dirname(SELECTION_OUT), exist_ok=True)
    with open(SELECTION_OUT, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()



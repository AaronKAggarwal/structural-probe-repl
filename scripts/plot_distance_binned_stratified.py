#!/usr/bin/env python3
"""
Stratified binned plots for distance (UUAS) at L7, test split.

Two plot types per language:
- UUAS vs sentence length, stratified by arc-length quintiles (Stage-8 pooled bins)
- UUAS vs mean arc length, stratified by sentence-length bands (default [2–10, 11–20, 21+])

Per-plot:
- Points: bin means with 95% bootstrap CIs; x = in-bin median x
- Lines: dashed Theil–Sen per stratum on raw per-sentence pairs
- Caption/subtitle: stratified median slope across strata with bootstrap CI

Outputs:
- outputs/figures/exploratory/binned_stratified/L7/uuas_vs_length_by_arclen/<LANG>.png
- outputs/figures/exploratory/binned_stratified/L7/uuas_vs_arclen_by_length/<LANG>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from plot_style import apply_style, savefig, smart_ylim

REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
SENTENCE_STATS_DIR = ANALYSIS_DIR / "sentence_stats"
MATCHED_TARGETS_DIR = ANALYSIS_DIR / "matched_targets"
FIG_ROOT = REPO_ROOT / "outputs" / "figures" / "exploratory" / "binned_stratified" / "L7"

# Deterministic bootstrap/fit seed
RNG_SEED = 42


def ensure_dirs() -> None:
    (FIG_ROOT / "uuas_vs_length_by_arclen").mkdir(parents=True, exist_ok=True)
    (FIG_ROOT / "uuas_vs_arclen_by_length").mkdir(parents=True, exist_ok=True)


def load_sentence_stats(language_slug: str, split: str) -> Dict[str, Any]:
    stats_file = SENTENCE_STATS_DIR / language_slug / f"{split}_content_stats.jsonl"
    if not stats_file.exists():
        raise FileNotFoundError(f"Sentence stats not found: {stats_file}")
    sentences: List[Dict[str, Any]] = []
    with open(stats_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    return {"sentences": sentences}


def discover_run_dir(language_slug: str, probe: str, layer: str) -> Optional[str]:
    base_per_lang = (
        REPO_ROOT
        / "outputs"
        / "baselines_auto"
        / language_slug
        / "bert-base-multilingual-cased"
        / probe
        / layer
        / "runs"
    )
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


def load_uuas_per_sentence(language_slug: str, layer: str, split: str) -> Optional[Dict[str, Any]]:
    run_dir = discover_run_dir(language_slug, "dist", layer)
    if not run_dir:
        return None
    run_path = Path(run_dir)
    if split == "test":
        test_final = run_path / "test_detailed_metrics_final.json"
        test_regular = run_path / "test_detailed_metrics.json"
        metrics_file = test_final if test_final.exists() else test_regular
    else:
        metrics_file = run_path / "dev_detailed_metrics.json"
    if not metrics_file.exists():
        return None
    detailed = json.loads(Path(metrics_file).read_text())
    full = detailed.get("uuas_per_sentence_full")
    comp = detailed.get("uuas_per_sentence")
    src = full if (isinstance(full, list) and len(full) > 0) else comp
    if not (src and isinstance(src, list)):
        return None
    out = {
        "values": np.array(src, dtype=float),
        "ids": detailed.get("sentence_ids") or detailed.get("sent_ids"),
        "kept": detailed.get("kept_sentence_indices")
        or detailed.get("valid_sentence_indices")
        or detailed.get("eval_sentence_indices"),
    }
    return out


def align_to_full(
    v: np.ndarray,
    N: int,
    kept: Optional[List[int]],
    ids_metrics: Optional[List[str]],
    ids_stage3: List[str],
    eval_mask_fallback: np.ndarray,
) -> Optional[np.ndarray]:
    # (a) by ids
    if ids_metrics and len(ids_metrics) == len(v):
        full = np.full(N, np.nan, dtype=float)
        pos = {sid: i for i, sid in enumerate(ids_stage3)}
        for j, sid in enumerate(ids_metrics):
            i = pos.get(sid)
            if i is not None and 0 <= i < N:
                full[i] = v[j]
        if np.isfinite(full).any():
            return full
    # (b) by kept indices
    if kept and len(kept) == len(v):
        full = np.full(N, np.nan, dtype=float)
        for src, dst in enumerate(kept):
            if 0 <= dst < N:
                full[dst] = v[src]
        return full
    # (c) already full
    if len(v) == N:
        return v.astype(float)
    # (d) by mask fallback
    if eval_mask_fallback is not None and len(v) == int(np.asarray(eval_mask_fallback).sum()):
        full = np.full(N, np.nan, dtype=float)
        full[np.where(eval_mask_fallback)[0]] = v
        return full
    return None


def bootstrap_mean_ci(y: np.ndarray, B: int = 1000, rng: Optional[np.random.RandomState] = None) -> Tuple[float, float, float]:
    r = rng or np.random.RandomState(RNG_SEED)
    if y.size == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(B):
        idx = r.randint(0, y.size, size=y.size)
        boots.append(np.mean(y[idx]))
    arr = np.array(boots, dtype=float)
    return float(np.mean(y)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def strictly_increasing(a: np.ndarray) -> bool:
    """Check if an array has strictly increasing values."""
    if a is None or len(a) < 2:
        return False
    diffs = np.diff(a.astype(float))
    return bool(np.all(diffs > 0))


def pooled_arclen_bins() -> Tuple[np.ndarray, List[str]]:
    spec_path = MATCHED_TARGETS_DIR / "arclen_bins.json"
    if not spec_path.exists():
        return np.array([]), []
    spec = json.loads(spec_path.read_text())
    edges = np.array(spec.get("bin_edges", []), dtype=float)
    labels = spec.get("bin_labels", ["Q1", "Q2", "Q3", "Q4", "Q5"])  # default fallback
    return edges, labels


def stratified_plot_length(language_slug: str, split: str, layer: str = "L7", arc_strata_n: int = 3) -> None:
    print(f"\n=== STRATIFIED PLOT LENGTH: {language_slug} ===")
    stats = load_sentence_stats(language_slug, split)
    sentences = stats["sentences"]
    N = len(sentences)
    print(f"Total sentences: {N}")
    if N == 0:
        return

    x_len = np.array([float(s.get("content_len", np.nan)) for s in sentences], dtype=float)
    # arc-length with fallback
    ma_vals: List[float] = []
    for s in sentences:
        ma_key = "mean_arc_len" if ("mean_arc_len" in s) else ("mean_content_arc_len" if "mean_content_arc_len" in s else None)
        v = s.get(ma_key) if ma_key else None
        ma_vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
    x_arc = np.array(ma_vals, dtype=float)
    num_arcs = np.array([int(s.get("num_content_arcs_used", 0)) for s in sentences], dtype=int)

    ypack = load_uuas_per_sentence(language_slug, layer, split)
    if not ypack:
        return
    y = ypack["values"]
    kept = ypack["kept"]
    ids_metrics = ypack["ids"]
    ids_stage3 = [s.get("sent_id", f"sent_{i}") for i, s in enumerate(sentences)]
    eval_mask_fallback = (x_len >= 2)
    y_full = align_to_full(y, N, kept, ids_metrics, ids_stage3, eval_mask_fallback)
    if y_full is None or len(y_full) != N:
        return
    # Clamp UUAS to [0,1] defensively
    y_full = np.clip(y_full, 0.0, 1.0)

    # x-axis bins (length)
    len_edges = np.array([2, 6, 11, 16, 26, 41, np.inf], dtype=float)
    len_labels = ["2–5", "6–10", "11–15", "16–25", "26–40", "41+"]

    # strata by arc-length quintiles (Stage-8 pooled bins)
    arc_edges, arc_labels = pooled_arclen_bins()
    if arc_edges.size == 0:
        # fallback: per-language quantiles
        vals = x_arc[np.isfinite(x_arc) & (num_arcs >= 1)]
        if vals.size < 50:
            return
        arc_edges = np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(float)
        arc_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    # Degeneracy guard on pooled/fallback edges
    if not strictly_increasing(arc_edges):
        vals = x_arc[np.isfinite(x_arc) & (num_arcs >= 1)]
        qs = np.quantile(vals, [0.0, 1.0/3.0, 2.0/3.0, 1.0]).astype(float)
        if strictly_increasing(qs):
            arc_edges = qs
            arc_labels = ["T1", "T2", "T3"]
        else:
            return

    # Merge to 3 strata if requested: Low(Q1–Q2), Mid(Q3), High(Q4–Q5)
    if arc_strata_n == 3 and arc_edges.size >= 6:
        merged_edges = np.array([arc_edges[0], arc_edges[2], arc_edges[3], arc_edges[5]], dtype=float)
        merged_labels = ["Low (Q1–Q2)", "Mid (Q3)", "High (Q4–Q5)"]
        arc_edges, arc_labels = merged_edges, merged_labels

    print(f"Arc stratification: {len(arc_labels)} strata")
    print(f"Arc edges: {arc_edges}")
    print(f"Arc labels: {arc_labels}")

    plt.figure(figsize=(8.0, 5.0))  # More vertical space
    rng = np.random.RandomState(RNG_SEED)
    slopes: List[float] = []
    # Consistent colors per stratum
    colors = plt.cm.tab10(np.linspace(0, 1, len(arc_labels)))
    
    # Collect all y-values for smart scaling
    all_ys = []
    all_annotations = []

    for i in range(len(arc_labels)):
        print(f"\n--- Stratum {i}: {arc_labels[i]} ---")
        lo_a, hi_a = arc_edges[i], arc_edges[i + 1]
        print(f"Arc range: [{lo_a:.3f}, {hi_a:.3f}]")
        
        stratum_mask = (
            np.isfinite(x_arc)
            & np.isfinite(y_full)
            & (num_arcs >= 1)
            & (x_arc >= lo_a)
            & (x_arc < hi_a if i < len(arc_labels) - 1 else x_arc <= hi_a)
        )
        
        stratum_count = int(stratum_mask.sum())
        print(f"Sentences in arc stratum: {stratum_count}")

        # Fit mask in stratum
        F = stratum_mask & np.isfinite(x_len) & (x_len >= 2)
        n_valid = int(np.sum(F))
        n_unique_x = int(np.unique(x_len[F]).shape[0])
        
        print(f"After fit filters: n_valid={n_valid}, n_unique_x={n_unique_x}")
        
        if n_valid > 0:
            x_fit_vals = x_len[F]
            y_fit_vals = y_full[F]
            print(f"Length range in stratum: [{x_fit_vals.min():.1f}, {x_fit_vals.max():.1f}]")
            print(f"UUAS range in stratum: [{y_fit_vals.min():.3f}, {y_fit_vals.max():.3f}]")
            unique_lengths = np.unique(x_fit_vals)
            print(f"Unique length values: {len(unique_lengths)} (first 10: {unique_lengths[:10]})")
            # Check for perfect scores
            perfect_count = np.sum(y_fit_vals >= 0.999)
            print(f"Perfect UUAS scores (>=0.999): {perfect_count}/{n_valid} ({100*perfect_count/n_valid:.1f}%)")
            # Check distribution
            uuas_quantiles = np.quantile(y_fit_vals, [0, 0.25, 0.5, 0.75, 1.0])
            print(f"UUAS quantiles: {uuas_quantiles}")
        else:
            print("No valid data points in stratum")

        # Bin means (macro, unweighted)
        xs, ys, los, his, ns = [], [], [], [], []
        for j in range(len(len_labels)):
            lo_l, hi_l = len_edges[j], len_edges[j + 1]
            B = F & (x_len >= lo_l) & (x_len < hi_l)
            y_bin = y_full[B]
            if y_bin.size >= 20:
                mean_y, ci_lo, ci_hi = bootstrap_mean_ci(y_bin, B=1000, rng=rng)
                x_median = float(np.median(x_len[B]))
                xs.append(x_median)
                ys.append(mean_y)
                los.append(ci_lo)
                his.append(ci_hi)
                ns.append(int(y_bin.size))
                print(f"  Bin {j} ({len_labels[j]}): x_median={x_median:.1f}, y_mean={mean_y:.3f}, n={y_bin.size}")
        
        print(f"Plotted bins for stratum {i}: {len(xs)} bins with y-range [{min(ys):.3f}, {max(ys):.3f}]" if xs else f"No bins plotted for stratum {i}")
        if xs:
            # Always include in legend - remove the len(xs) >= 3 filter
            plt.errorbar(
                xs,
                ys,
                yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                fmt="o-",
                capsize=2,
                label=f"{arc_labels[i]}",
                color=colors[i]
            )
            # Collect y-values for smart scaling
            all_ys.extend(ys)
            all_ys.extend(los)  # Include CI bounds
            all_ys.extend(his)
            
            # Annotate bin support at top of error bar, colored by stratum
            for xm, ym, lo_y, hi_y, n in zip(xs, ys, los, his, ns):
                try:
                    ann = plt.annotate(
                        f"n={n}",
                        (xm, hi_y),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        fontsize=7,
                        alpha=0.8,
                        color=colors[i],
                    )
                    all_annotations.append(ann)
                except Exception:
                    pass

        # Theil–Sen per stratum
        print(f"QC check: n_valid >= 200? {n_valid >= 200}, n_unique_x >= 10? {n_unique_x >= 10}")
        qc_pass = n_valid >= 200 and n_unique_x >= 10
        print(f"QC result: {'PASS' if qc_pass else 'FAIL'}")
        
        if qc_pass:
            try:
                print("Attempting Theil-Sen fit...")
                slope, intercept, _, _ = theilslopes(y_full[F], x_len[F], alpha=0.95)
                slopes.append(float(slope))
                x_fit = np.linspace(np.nanmin(x_len[F]), np.nanmax(x_len[F]), 100)
                y_fit = slope * x_fit + intercept
                print(f"Theil-Sen SUCCESS: slope={slope:.6f}, intercept={intercept:.6f}")
                print(f"Line range: x=[{x_fit.min():.1f}, {x_fit.max():.1f}], y=[{y_fit.min():.3f}, {y_fit.max():.3f}]")
                # Always plot the Theil–Sen line without extra annotations
                plt.plot(x_fit, y_fit, linestyle="--", alpha=0.6, color=colors[i])
                
                print(f"Line plotted for stratum {i}")
            except Exception as e:
                print(f"Theil-Sen FAILED: {e}")
        else:
            print("Theil-Sen fit skipped due to QC failure")

    # Caption: stratified median slope + bootstrap CI
    subtitle = ""
    if slopes:
        B = 2000
        rng2 = np.random.RandomState(RNG_SEED)
        boots = []
        arr = np.array(slopes, dtype=float)
        for _ in range(B):
            idx = rng2.randint(0, arr.size, size=arr.size)
            boots.append(np.median(arr[idx]))
        boots = np.array(boots, dtype=float)
        med = float(np.median(arr))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        subtitle = f"Stratified median slope={med:.4f} [{lo:.4f}, {hi:.4f}]"

    # Reference at L=20
    try:
        plt.axvline(20, color="gray", linestyle=":", alpha=0.25)
    except Exception:
        pass

    # Apply smart scaling; allow labels to overlap plot content
    smart_ylim(all_ys, domain_min=0.6, domain_max=1.0, padding=0.08)
    try:
        yl = plt.gca().get_ylim()
        plt.ylim(0.6, yl[1])
    except Exception:
        pass
    
    plt.title(f"{language_slug} — L7 UUAS vs length (by arc-length)")
    if subtitle:
        plt.suptitle(subtitle, y=0.96, fontsize=9)
    plt.xlabel("Content length (tokens)")
    plt.ylabel("UUAS (per sentence)")
    plt.legend(title="Arc-length stratum", loc="best", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    out = FIG_ROOT / "uuas_vs_length_by_arclen" / f"{language_slug}.png"
    savefig(plt.gcf(), out)
    plt.close()
    # Save CSV alongside PNG
    try:
        records = []
        for (xm, ym, lo_y, hi_y, n_val) in zip(xs, ys, los, his, ns):
            records.append({
                "language_slug": language_slug,
                "stratum": arc_labels[i],
                "x_median": xm,
                "y_mean": ym,
                "ci_lo": lo_y,
                "ci_hi": hi_y,
                "n_bin": n_val,
            })
        if records:
            pd.DataFrame(records).to_csv(out.with_suffix('.csv'), index=False)
    except Exception:
        pass


def stratified_plot_arclen(language_slug: str, split: str, layer: str = "L7") -> None:
    print(f"\n=== STRATIFIED PLOT ARCLEN: {language_slug} ===")
    stats = load_sentence_stats(language_slug, split)
    sentences = stats["sentences"]
    N = len(sentences)
    print(f"Total sentences: {N}")
    if N == 0:
        return

    x_len = np.array([float(s.get("content_len", np.nan)) for s in sentences], dtype=float)
    # arc-length with fallback
    ma_vals: List[float] = []
    for s in sentences:
        ma_key = "mean_arc_len" if ("mean_arc_len" in s) else ("mean_content_arc_len" if "mean_content_arc_len" in s else None)
        v = s.get(ma_key) if ma_key else None
        ma_vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
    x_arc = np.array(ma_vals, dtype=float)
    num_arcs = np.array([int(s.get("num_content_arcs_used", 0)) for s in sentences], dtype=int)

    ypack = load_uuas_per_sentence(language_slug, layer, split)
    if not ypack:
        return
    y = ypack["values"]
    kept = ypack["kept"]
    ids_metrics = ypack["ids"]
    ids_stage3 = [s.get("sent_id", f"sent_{i}") for i, s in enumerate(sentences)]
    eval_mask_fallback = (x_len >= 2)
    y_full = align_to_full(y, N, kept, ids_metrics, ids_stage3, eval_mask_fallback)
    if y_full is None or len(y_full) != N:
        return
    y_full = np.clip(y_full, 0.0, 1.0)

    # x-axis: arc-length bins (pooled or fallback quantiles)
    arc_edges, arc_labels = pooled_arclen_bins()
    if arc_edges.size == 0:
        vals = x_arc[np.isfinite(x_arc) & (num_arcs >= 1)]
        if vals.size < 50:
            return
        arc_edges = np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(float)
        arc_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    # strata: sentence length bands (default 3 bands to keep Ns healthy)
    len_edges = np.array([2, 11, 21, np.inf], dtype=float)
    len_labels = ["2–10", "11–20", "21+"]

    plt.figure(figsize=(8.0, 5.0))  # More vertical space
    rng = np.random.RandomState(RNG_SEED)
    slopes: List[float] = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(len_labels)))
    
    # Collect all y-values for smart scaling
    all_ys = []
    all_annotations = []

    for i in range(len(len_labels)):
        lo_l, hi_l = len_edges[i], len_edges[i + 1]
        stratum_mask = (
            np.isfinite(x_len)
            & np.isfinite(y_full)
            & (x_len >= lo_l)
            & (x_len < hi_l)
        )

        # Fit mask in stratum (arc-length eligibility)
        F = stratum_mask & np.isfinite(x_arc) & (num_arcs >= 1) & (x_arc > 0)
        n_valid = int(np.sum(F))
        n_unique_x = int(np.unique(x_arc[F]).shape[0])

        # Bin means on x_arc
        xs, ys, los, his, ns = [], [], [], [], []
        for j in range(len(arc_labels)):
            lo_a, hi_a = arc_edges[j], arc_edges[j + 1]
            B = F & (x_arc >= lo_a) & (x_arc < hi_a if j < len(arc_labels) - 1 else x_arc <= hi_a)
            y_bin = y_full[B]
            if y_bin.size >= 20:
                mean_y, ci_lo, ci_hi = bootstrap_mean_ci(y_bin, B=1000, rng=rng)
                x_median = float(np.median(x_arc[B]))
                xs.append(x_median)
                ys.append(mean_y)
                los.append(ci_lo)
                his.append(ci_hi)
                ns.append(int(y_bin.size))
        if xs:
            # Always include in legend - remove the len(xs) >= 3 filter
            plt.errorbar(
                xs,
                ys,
                yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                fmt="o-",
                capsize=2,
                label=f"{len_labels[i]}",
                color=colors[i]
            )
            # Collect y-values for smart scaling
            all_ys.extend(ys)
            all_ys.extend(los)  # Include CI bounds
            all_ys.extend(his)
            
            # Annotate bin support with overlap prevention
            # Annotate bin support at top of error bar, colored by stratum
            for xm, ym, lo_y, hi_y, n in zip(xs, ys, los, his, ns):
                try:
                    ann = plt.annotate(
                        f"n={n}",
                        (xm, hi_y),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        fontsize=7,
                        alpha=0.8,
                        color=colors[i],
                    )
                    all_annotations.append(ann)
                except Exception:
                    pass

        # Theil–Sen per stratum
        if n_valid >= 200 and n_unique_x >= 10:
            try:
                slope, intercept, _, _ = theilslopes(y_full[F], x_arc[F], alpha=0.95)
                slopes.append(float(slope))
                x_fit = np.linspace(np.nanmin(x_arc[F]), np.nanmax(x_arc[F]), 100)
                plt.plot(x_fit, slope * x_fit + intercept, linestyle="--", alpha=0.6, color=colors[i])
            except Exception:
                pass

    # Caption: stratified median slope + bootstrap CI
    subtitle = ""
    if slopes:
        B = 2000
        rng2 = np.random.RandomState(RNG_SEED)
        boots = []
        arr = np.array(slopes, dtype=float)
        for _ in range(B):
            idx = rng2.randint(0, arr.size, size=arr.size)
            boots.append(np.median(arr[idx]))
        boots = np.array(boots, dtype=float)
        med = float(np.median(arr))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        subtitle = f"Stratified median slope={med:.3f} [{lo:.3f}, {hi:.3f}]"

    # Apply smart scaling; allow labels to overlap plot content
    smart_ylim(all_ys, domain_min=0.6, domain_max=1.0, padding=0.08)
    try:
        yl = plt.gca().get_ylim()
        plt.ylim(0.6, yl[1])
    except Exception:
        pass
    
    plt.title(f"{language_slug} — L7 UUAS vs arc length (by sentence length)")
    if subtitle:
        plt.suptitle(subtitle, y=0.96, fontsize=9)
    plt.xlabel("Mean arc length (content-only)")
    plt.ylabel("UUAS (per sentence)")
    plt.legend(title="Length band", loc="best", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    out = FIG_ROOT / "uuas_vs_arclen_by_length" / f"{language_slug}.png"
    savefig(plt.gcf(), out)
    plt.close()
    try:
        records = []
        for (xm, ym, lo_y, hi_y, n_val) in zip(xs, ys, los, his, ns):
            records.append({
                "language_slug": language_slug,
                "stratum": len_labels[i],
                "x_median": xm,
                "y_mean": ym,
                "ci_lo": lo_y,
                "ci_hi": hi_y,
                "n_bin": n_val,
            })
        if records:
            pd.DataFrame(records).to_csv(out.with_suffix('.csv'), index=False)
    except Exception:
        pass


def get_languages() -> List[str]:
    per_layer = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    if not per_layer.exists():
        raise FileNotFoundError(f"Missing {per_layer}")
    df = pd.read_csv(per_layer)
    return sorted(df["language_slug"].unique().tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--arc_strata", type=int, default=3, choices=[3, 5], help="Number of arc-length strata (3 merges Q1+Q2 and Q4+Q5)")
    args = ap.parse_args()

    apply_style("talk")
    ensure_dirs()
    languages = get_languages()
    print(f"Stratified binned plots for {len(languages)} languages…")
    for lang in languages:
        try:
            stratified_plot_length(lang, args.split, arc_strata_n=args.arc_strata)
            stratified_plot_arclen(lang, args.split)
        except Exception as e:
            print(f"  ⚠ {lang}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()



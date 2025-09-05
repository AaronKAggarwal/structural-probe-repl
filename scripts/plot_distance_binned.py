#!/usr/bin/env python3
"""
Per-language binned distance plots at L7 with 95% bootstrap CIs.

Two predictors at test split:
- Sentence length (content_len), using Stage-7 default bins [2–5, 6–10, 11–15, 16–25, 26–40, 41+]
- Mean arc length (content-only), using Stage-8 pooled quantile bins from outputs/analysis/matched_targets/arclen_bins.json

Outputs (PNG):
- outputs/figures/exploratory/scatters/L7/length_binned/<LANG>.png
- outputs/figures/exploratory/scatters/L7/arclen_binned/<LANG>.png

Approach:
- Align per-sentence UUAS (distance probe) to Stage-3 sentence list via IDs → kept_indices → full → mask fallback
- Bin x, compute bin mean UUAS with 95% bootstrap CI, annotate bin sizes
- Overlay Theil–Sen line and report slope in subtitle
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

from plot_style import apply_style, savefig


REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
SENTENCE_STATS_DIR = ANALYSIS_DIR / "sentence_stats"
MATCHED_TARGETS_DIR = ANALYSIS_DIR / "matched_targets"
FIG_BASE = REPO_ROOT / "outputs" / "figures" / "exploratory" / "scatters" / "L7"


def ensure_dirs() -> None:
    (FIG_BASE / "length_binned").mkdir(parents=True, exist_ok=True)
    (FIG_BASE / "arclen_binned").mkdir(parents=True, exist_ok=True)


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
    r = rng or np.random.RandomState(42)
    if y.size == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(B):
        idx = r.randint(0, y.size, size=y.size)
        boots.append(np.mean(y[idx]))
    arr = np.array(boots, dtype=float)
    return float(np.mean(y)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def plot_binned_length(language_slug: str, split: str, layer: str = "L7") -> None:
    stats = load_sentence_stats(language_slug, split)
    sentences = stats["sentences"]
    N = len(sentences)
    if N == 0:
        return
    content_len = np.array([float(s.get("content_len", np.nan)) for s in sentences], dtype=float)
    ypack = load_uuas_per_sentence(language_slug, layer, split)
    if not ypack:
        return
    y = ypack["values"]
    kept = ypack["kept"]
    ids_metrics = ypack["ids"]
    ids_stage3 = [s.get("sent_id", f"sent_{i}") for i, s in enumerate(sentences)]
    eval_mask_fallback = (content_len >= 2)
    y_full = align_to_full(y, N, kept, ids_metrics, ids_stage3, eval_mask_fallback)
    if y_full is None or len(y_full) != N:
        return

    # Stage-7 default bins
    bin_edges = np.array([2, 6, 11, 16, 26, 41, np.inf], dtype=float)
    bin_labels = ["2–5", "6–10", "11–15", "16–25", "26–40", "41+"]

    # Compute bin means + CIs
    xs, ys, los, his, ns = [], [], [], [], []
    rng = np.random.RandomState(42)
    for i in range(len(bin_labels)):
        lo_edge, hi_edge = bin_edges[i], bin_edges[i + 1]
        mask = (np.isfinite(content_len) & np.isfinite(y_full) & (content_len >= lo_edge) & (content_len < hi_edge))
        y_bin = y_full[mask]
        if y_bin.size >= 20:
            mean_y, ci_lo, ci_hi = bootstrap_mean_ci(y_bin, B=1000, rng=rng)
            xs.append((lo_edge + (hi_edge if np.isfinite(hi_edge) else lo_edge + 5)) / 2.0)
            ys.append(mean_y); los.append(ci_lo); his.append(ci_hi); ns.append(int(y_bin.size))

    if len(xs) == 0:
        return

    plt.figure(figsize=(5.0, 3.4))
    plt.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)], fmt="o-", capsize=2)
    # Theil–Sen overlay
    try:
        mask_fit = np.isfinite(content_len) & np.isfinite(y_full) & (content_len >= 2)
        slope, intercept, _, _ = theilslopes(y_full[mask_fit], content_len[mask_fit], alpha=0.95)
        x_fit = np.linspace(min(xs), max(xs), 100)
        plt.plot(x_fit, slope * x_fit + intercept, linestyle="--", alpha=0.8, label=f"Theil–Sen slope={slope:.4f}")
        plt.legend(loc="best", framealpha=0.9)
    except Exception:
        pass
    plt.title(f"{language_slug} – dist L7 vs length (binned)")
    plt.xlabel("Content length (tokens)")
    plt.ylabel("UUAS (per sentence)")
    plt.ylim(0, 1)
    plt.tight_layout()
    out = FIG_BASE / "length_binned" / f"{language_slug}.png"
    savefig(plt.gcf(), out)
    plt.close()


def plot_binned_arclen(language_slug: str, split: str, layer: str = "L7") -> None:
    stats = load_sentence_stats(language_slug, split)
    sentences = stats["sentences"]
    N = len(sentences)
    if N == 0:
        return
    # Arc-length with fallback
    ma_vals: List[float] = []
    for s in sentences:
        ma_key = "mean_arc_len" if ("mean_arc_len" in s) else ("mean_content_arc_len" if "mean_content_arc_len" in s else None)
        v = s.get(ma_key) if ma_key else None
        ma_vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
    mean_arclen = np.array(ma_vals, dtype=float)
    num_arcs = np.array([int(s.get("num_content_arcs_used", 0)) for s in sentences], dtype=int)

    ypack = load_uuas_per_sentence(language_slug, layer, split)
    if not ypack:
        return
    y = ypack["values"]
    kept = ypack["kept"]
    ids_metrics = ypack["ids"]
    ids_stage3 = [s.get("sent_id", f"sent_{i}") for i, s in enumerate(sentences)]
    eval_mask_fallback = (np.array([float(s.get("content_len", np.nan)) for s in sentences]) >= 2)
    y_full = align_to_full(y, N, kept, ids_metrics, ids_stage3, eval_mask_fallback)
    if y_full is None or len(y_full) != N:
        return

    # Load pooled quantile bins for arclen
    target_file = MATCHED_TARGETS_DIR / "arclen_bins.json"
    if not target_file.exists():
        # fallback: simple quantiles per language
        vals = mean_arclen[np.isfinite(mean_arclen) & (num_arcs >= 1)]
        if vals.size < 50:
            return
        q = np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bin_edges = q
        labels = [f"Q{i+1}" for i in range(5)]
    else:
        spec = json.loads(target_file.read_text())
        bin_edges = np.array(spec.get("bin_edges", []), dtype=float)
        labels = spec.get("bin_labels", ["Q1", "Q2", "Q3", "Q4", "Q5"])  # default

    xs, ys, los, his, ns = [], [], [], [], []
    rng = np.random.RandomState(42)
    for i in range(len(labels)):
        lo_edge, hi_edge = bin_edges[i], bin_edges[i + 1]
        mask = (
            np.isfinite(mean_arclen)
            & np.isfinite(y_full)
            & (num_arcs >= 1)
            & (mean_arclen >= lo_edge)
            & (mean_arclen < hi_edge if i < len(labels) - 1 else mean_arclen <= hi_edge)
        )
        y_bin = y_full[mask]
        x_bin = mean_arclen[mask]
        if y_bin.size >= 20:
            mean_y, ci_lo, ci_hi = bootstrap_mean_ci(y_bin, B=1000, rng=rng)
            xs.append(float(np.nanmean(x_bin)))
            ys.append(mean_y); los.append(ci_lo); his.append(ci_hi); ns.append(int(y_bin.size))

    if len(xs) == 0:
        return

    plt.figure(figsize=(5.0, 3.4))
    plt.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)], fmt="o-", capsize=2)
    # Theil–Sen overlay
    try:
        mask_fit = np.isfinite(mean_arclen) & np.isfinite(y_full) & (num_arcs >= 1) & (mean_arclen > 0)
        slope, intercept, _, _ = theilslopes(y_full[mask_fit], mean_arclen[mask_fit], alpha=0.95)
        x_fit = np.linspace(min(xs), max(xs), 100)
        plt.plot(x_fit, slope * x_fit + intercept, linestyle="--", alpha=0.8, label=f"Theil–Sen slope={slope:.3f}")
        plt.legend(loc="best", framealpha=0.9)
    except Exception:
        pass
    plt.title(f"{language_slug} – dist L7 vs arc length (binned)")
    plt.xlabel("Mean arc length (content-only)")
    plt.ylabel("UUAS (per sentence)")
    plt.ylim(0, 1)
    plt.tight_layout()
    out = FIG_BASE / "arclen_binned" / f"{language_slug}.png"
    savefig(plt.gcf(), out)
    plt.close()


def get_languages() -> List[str]:
    per_layer = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    if not per_layer.exists():
        raise FileNotFoundError(f"Missing {per_layer}")
    df = pd.read_csv(per_layer)
    return sorted(df["language_slug"].unique().tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    args = ap.parse_args()

    apply_style("talk")
    ensure_dirs()
    languages = get_languages()
    print(f"Plotting binned curves for {len(languages)} languages…")
    for lang in languages:
        try:
            plot_binned_length(lang, args.split)
            plot_binned_arclen(lang, args.split)
        except Exception as e:
            print(f"  ⚠ {lang}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Create per-language scatter plots for distance probe performance vs predictors at L7.

Two predictors:
- Sentence length (content_len)
- Mean arc length (mean_arc_len with fallback to mean_content_arc_len)

Saves PNGs under:
  outputs/figures/exploratory/scatters/L7/length/<LANG>.png
  outputs/figures/exploratory/scatters/L7/arclen/<LANG>.png

Uses Stage 3 sentence stats for predictors and aligns per-sentence UUAS using the
same robust alignment strategy (IDs → kept_indices → full → mask fallback).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_style, savefig

REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
SENTENCE_STATS_DIR = ANALYSIS_DIR / "sentence_stats"
FIG_BASE = REPO_ROOT / "outputs" / "figures" / "exploratory" / "scatters" / "L7"


def ensure_dirs() -> None:
    (FIG_BASE / "length").mkdir(parents=True, exist_ok=True)
    (FIG_BASE / "arclen").mkdir(parents=True, exist_ok=True)


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


def plot_language(language_slug: str, split: str, layer: str = "L7") -> None:
    # Load predictors
    stats = load_sentence_stats(language_slug, split)
    sentences = stats["sentences"]
    N = len(sentences)
    if N == 0:
        return

    content_len = np.array([float(s.get("content_len", np.nan)) for s in sentences], dtype=float)
    ma_vals: List[float] = []
    for s in sentences:
        ma_key = "mean_arc_len" if ("mean_arc_len" in s) else ("mean_content_arc_len" if "mean_content_arc_len" in s else None)
        v = s.get(ma_key) if ma_key else None
        ma_vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
    mean_arclen = np.array(ma_vals, dtype=float)
    num_arcs = np.array([int(s.get("num_content_arcs_used", 0)) for s in sentences], dtype=int)

    # Load uuas
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

    # Length scatter
    mask_len = np.isfinite(content_len) & np.isfinite(y_full) & (content_len >= 2)
    if np.count_nonzero(mask_len) >= 20:
        plt.figure(figsize=(5.0, 3.4))
        plt.scatter(content_len[mask_len], y_full[mask_len], s=6, alpha=0.6)
        plt.xlabel("Content length (tokens)")
        plt.ylabel("UUAS (per sentence)")
        plt.title(f"{language_slug} – dist L7 vs length")
        plt.ylim(0, 1)
        plt.tight_layout()
        out = FIG_BASE / "length" / f"{language_slug}.png"
        savefig(plt.gcf(), out)
        plt.close()

    # Arc-length scatter
    x_arc = mean_arclen.copy()
    x_arc[num_arcs < 1] = np.nan
    mask_arc = np.isfinite(x_arc) & np.isfinite(y_full) & (x_arc > 0)
    if np.count_nonzero(mask_arc) >= 20:
        plt.figure(figsize=(5.0, 3.4))
        plt.scatter(x_arc[mask_arc], y_full[mask_arc], s=6, alpha=0.6)
        plt.xlabel("Mean arc length (content-only)")
        plt.ylabel("UUAS (per sentence)")
        plt.title(f"{language_slug} – dist L7 vs arc length")
        plt.ylim(0, 1)
        plt.tight_layout()
        out = FIG_BASE / "arclen" / f"{language_slug}.png"
        savefig(plt.gcf(), out)
        plt.close()


def get_languages() -> List[str]:
    per_layer = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    if not per_layer.exists():
        raise FileNotFoundError(f"Missing {per_layer}")
    import pandas as pd
    df = pd.read_csv(per_layer)
    return sorted(df["language_slug"].unique().tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    args = ap.parse_args()

    apply_style("talk")
    ensure_dirs()
    languages = get_languages()
    print(f"Plotting {len(languages)} languages…")
    for lang in languages:
        try:
            plot_language(lang, args.split)
        except Exception as e:
            print(f"  ⚠ {lang}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()



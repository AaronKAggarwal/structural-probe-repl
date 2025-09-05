#!/usr/bin/env python3
"""
Compute per-sentence Theil–Sen trends for probe performance vs sentence covariates.

Variant implemented: RAW (no matching). Stage 7/8/9 variants can be added later.

For each (language, probe, layer) and each predictor (length, arc-length):
- Align per-sentence y (uuas or root_acc) to Stage 3 sentence list
- Build x from sentence_stats (content_len or mean_arc_len)
- Apply QC filters
- Fit theilslopes(y, x, alpha=0.95)
- Compute Kendall's tau and p-value
- Write a tidy CSV per-layer with all language rows

Outputs:
- outputs/analysis/trends/theilsen_per_layer_raw.csv
- outputs/analysis/trends/theilsen_summary_by_layer.csv (median slopes + bootstrap CI)

Notes:
- Depth is binary; Theil–Sen provides a robust linear trend in success probability vs x.
- Uses alignment logic similar to Stage 7 scripts via kept indices (compact arrays) where present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes


# Repository paths
REPO_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
TRENDS_DIR = ANALYSIS_DIR / "trends"
SENTENCE_STATS_DIR = ANALYSIS_DIR / "sentence_stats"


LANGUAGES: Optional[List[str]] = None  # If None, auto-discover from analysis table
PROBES = ["dist", "depth"]
LAYERS = [f"L{i}" for i in range(5, 11)]  # L5..L10

# QC constants
MIN_N_VALID = 200
MIN_N_UNIQUE_X = 10


@dataclass
class TrendResult:
    language_slug: str
    probe: str
    layer: str
    predictor: str  # "length" or "arclen"
    split: str
    variant: str
    T_total: int
    T_evaluable: int
    coverage_valid: float
    n_valid: int
    n_unique_x: int
    x_min: float
    x_max: float
    slope: float
    slope_ci_low: float
    slope_ci_high: float
    intercept: float
    kendall_tau: float
    kendall_p: float
    tail_winsorized: bool
    alignment_mode: str


def ensure_dirs() -> None:
    TRENDS_DIR.mkdir(parents=True, exist_ok=True)


def load_languages_from_analysis_table() -> List[str]:
    per_layer = ANALYSIS_DIR / "analysis_table_per_layer.csv"
    if not per_layer.exists():
        raise FileNotFoundError(f"Missing {per_layer}")
    df = pd.read_csv(per_layer)
    langs = sorted(df["language_slug"].unique().tolist())
    return langs


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


def load_per_sentence_metrics(language_slug: str, probe: str, layer: str, split: str) -> Optional[Dict[str, Any]]:
    """Load per-sentence probe metrics and alignment metadata.

    dist → 'uuas_per_sentence' → metrics['uuas']
    depth → 'root_acc_per_sentence' → metrics['root_acc']
    """
    run_dir = discover_run_dir(language_slug, probe, layer)
    if not run_dir:
        return None
    run_path = Path(run_dir)

    if split == "test":
        test_final = run_path / "test_detailed_metrics_final.json"
        test_regular = run_path / "test_detailed_metrics.json"
        metrics_file = test_final if test_final.exists() else test_regular
    elif split == "dev":
        metrics_file = run_path / "dev_detailed_metrics.json"
    else:
        return None

    if not metrics_file or not metrics_file.exists():
        return None

    try:
        with open(metrics_file, "r") as f:
            detailed = json.load(f)
    except Exception:
        return None

    metrics: Dict[str, Any] = {}
    if probe == "dist":
        full = detailed.get("uuas_per_sentence_full")
        comp = detailed.get("uuas_per_sentence")
        src = full if (isinstance(full, list) and len(full) > 0) else comp
        if src and isinstance(src, list):
            metrics["uuas"] = np.array(src, dtype=float)
        else:
            return None
    elif probe == "depth":
        full = detailed.get("root_acc_per_sentence_full")
        comp = detailed.get("root_acc_per_sentence")
        src = full if (isinstance(full, list) and len(full) > 0) else comp
        if src and isinstance(src, list):
            metrics["root_acc"] = np.array(src, dtype=float)
        else:
            return None
    else:
        return None

    # Alignment metadata for compact arrays
    metrics["_sent_ids"] = detailed.get("sentence_ids") or detailed.get("sent_ids")
    metrics["_kept_indices"] = (
        detailed.get("kept_sentence_indices")
        or detailed.get("valid_sentence_indices")
        or detailed.get("eval_sentence_indices")
    )
    return metrics


def align_metric_to_full(
    metric_values: np.ndarray,
    num_sentences: int,
    kept_indices: Optional[List[int]],
    ids_metrics: Optional[List[str]],
    ids_stage3: List[str],
    eval_mask_fallback: np.ndarray,
) -> Tuple[Optional[np.ndarray], str]:
    """Align compact/full metric vectors to the Stage 3 sentence list with robust fallbacks.

    Priority: (a) by_ids → (b) kept_indices → (c) full_length → (d) mask_fallback → (e) unknown
    """
    v = np.asarray(metric_values, dtype=float)
    # (a) by ids
    if ids_metrics and len(ids_metrics) == len(v):
        full = np.full(num_sentences, np.nan, dtype=float)
        pos = {sid: i for i, sid in enumerate(ids_stage3)}
        hits = 0
        for j, sid in enumerate(ids_metrics):
            i = pos.get(sid)
            if i is not None and 0 <= i < num_sentences:
                full[i] = v[j]
                hits += 1
        if hits > 0:
            return full, "by_ids"
    # (b) by kept indices
    if kept_indices and len(kept_indices) == len(v):
        full = np.full(num_sentences, np.nan, dtype=float)
        for src, dst in enumerate(kept_indices):
            if 0 <= dst < num_sentences:
                full[dst] = v[src]
        return full, "kept_indices"
    # (c) already full
    if len(v) == num_sentences:
        return v.astype(float), "full_length"
    # (d) length-match fallback (rare)
    if eval_mask_fallback is not None and len(v) == int(np.asarray(eval_mask_fallback).sum()):
        full = np.full(num_sentences, np.nan, dtype=float)
        full[np.where(eval_mask_fallback)[0]] = v
        return full, "by_mask_fallback"
    # (e) unknown → return None to force caller to skip
    return None, "unknown"


def winsorize(values: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    lo = np.nanpercentile(values, low_pct)
    hi = np.nanpercentile(values, high_pct)
    return np.clip(values, lo, hi)


def compute_theilsen_for_predictor(
    x: np.ndarray,
    y: np.ndarray,
    predictor: str,
    enable_winsor: bool,
    winsor_low: float,
    winsor_high: float,
) -> Optional[Tuple[float, float, float, float, float, float, int, int, float, float, bool]]:
    """Compute Theil–Sen slope/intercept + Kendall's tau for given x,y with QC.

    Returns: (slope, lo, hi, intercept, tau, p, n_valid, n_unique_x, x_min, x_max, winsor_flag)
    """
    # Filter finite
    mask = np.isfinite(x) & np.isfinite(y)
    x_f = x[mask]
    y_f = y[mask]

    if predictor == "length":
        # Enforce content_len >= 2
        mask2 = x_f >= 2
    elif predictor == "arclen":
        # mean_arc_len must be strictly positive (already finite)
        mask2 = x_f > 0
    else:
        mask2 = np.ones_like(x_f, dtype=bool)

    x_f = x_f[mask2]
    y_f = y_f[mask2]

    n_valid = int(x_f.shape[0])
    if n_valid < MIN_N_VALID:
        return None

    # Unique x values QC
    unique_x = np.unique(x_f[~np.isnan(x_f)])
    n_unique_x = int(unique_x.shape[0])
    if n_unique_x < MIN_N_UNIQUE_X:
        return None

    # Optional winsorization
    winsor_flag = False
    if enable_winsor:
        x_f = winsorize(x_f, winsor_low, winsor_high)
        winsor_flag = True

    x_min = float(np.nanmin(x_f))
    x_max = float(np.nanmax(x_f))

    try:
        slope, intercept, lo, hi = theilslopes(y_f, x_f, alpha=0.95)
    except Exception:
        return None

    try:
        tau, p = kendalltau(x_f, y_f)
    except Exception:
        tau, p = (np.nan, np.nan)

    return slope, lo, hi, intercept, tau, p, n_valid, n_unique_x, x_min, x_max, winsor_flag


def compute_raw_trends(
    languages: List[str],
    split: str,
    enable_winsor: bool,
    winsor_low: float,
    winsor_high: float,
) -> Tuple[pd.DataFrame, List[TrendResult]]:
    rows: List[TrendResult] = []

    for language_slug in languages:
        try:
            stats = load_sentence_stats(language_slug, split)
        except FileNotFoundError:
            continue
        sentences = stats["sentences"]
        N = len(sentences)
        if N == 0:
            continue

        # Build predictors
        content_len = np.array([float(s.get("content_len", np.nan)) for s in sentences], dtype=float)
        # Arc-length key fallback
        mean_arclen_list: List[float] = []
        for s in sentences:
            ma_key = (
                "mean_arc_len"
                if ("mean_arc_len" in s)
                else ("mean_content_arc_len" if "mean_content_arc_len" in s else None)
            )
            v = s.get(ma_key) if ma_key else None
            mean_arclen_list.append(float(v) if isinstance(v, (int, float)) else np.nan)
        mean_arclen = np.array(mean_arclen_list, dtype=float)
        num_arcs = np.array([int(s.get("num_content_arcs_used", 0)) for s in sentences], dtype=int)

        # Apply arclen eligibility (at least one content arc) by masking x later

        for probe in PROBES:
            for layer in LAYERS:
                metrics = load_per_sentence_metrics(language_slug, probe, layer, split)
                if not metrics:
                    continue

                if probe == "dist":
                    metric_key = "uuas"
                else:
                    metric_key = "root_acc"

                y_compact = metrics.get(metric_key)
                kept_idx = metrics.get("_kept_indices")
                ids_metrics = metrics.get("_sent_ids")

                if y_compact is None:
                    continue

                # Stage 3 IDs (for robust ID-based alignment)
                stage3_ids = [s.get("sent_id", f"sent_{i}") for i, s in enumerate(sentences)]
                # Safe base evaluability mask
                eval_mask_fallback = (content_len >= 2)
                y_full, mode = align_metric_to_full(
                    y_compact, N, kept_idx, ids_metrics, stage3_ids, eval_mask_fallback
                )

                # Guard: skip non-N alignments or unknown alignment
                if y_full is None or len(y_full) != N:
                    # print(f"  ⚠ Could not align ({language_slug}/{probe}/{layer}), skipping (mode={mode})")
                    continue

                # Predictor: length
                res_len = compute_theilsen_for_predictor(
                    x=content_len.copy(),
                    y=y_full.copy(),
                    predictor="length",
                    enable_winsor=enable_winsor,
                    winsor_low=winsor_low,
                    winsor_high=winsor_high,
                )
                if res_len is not None:
                    slope, lo, hi, intercept, tau, p, n_valid, n_unique_x, x_min, x_max, wz = res_len
                    rows.append(
                        TrendResult(
                            language_slug=language_slug,
                            probe=probe,
                            layer=layer,
                            predictor="length",
                            split=split,
                            variant="raw",
                            T_total=N,
                            T_evaluable=int(np.isfinite(y_full).sum()),
                            coverage_valid=(float(n_valid) / float(N) if N > 0 else float("nan")),
                            n_valid=n_valid,
                            n_unique_x=n_unique_x,
                            x_min=x_min,
                            x_max=x_max,
                            slope=slope,
                            slope_ci_low=lo,
                            slope_ci_high=hi,
                            intercept=intercept,
                            kendall_tau=tau,
                            kendall_p=p,
                            tail_winsorized=wz,
                            alignment_mode=mode,
                        )
                    )

                # Predictor: arc length (mask invalid x to NaN where num_arcs < 1)
                x_arc = mean_arclen.copy()
                x_arc[num_arcs < 1] = np.nan
                res_arc = compute_theilsen_for_predictor(
                    x=x_arc,
                    y=y_full.copy(),
                    predictor="arclen",
                    enable_winsor=enable_winsor,
                    winsor_low=winsor_low,
                    winsor_high=winsor_high,
                )
                if res_arc is not None:
                    slope, lo, hi, intercept, tau, p, n_valid, n_unique_x, x_min, x_max, wz = res_arc
                    rows.append(
                        TrendResult(
                            language_slug=language_slug,
                            probe=probe,
                            layer=layer,
                            predictor="arclen",
                            split=split,
                            variant="raw",
                            T_total=N,
                            T_evaluable=int(np.isfinite(y_full).sum()),
                            coverage_valid=(float(n_valid) / float(N) if N > 0 else float("nan")),
                            n_valid=n_valid,
                            n_unique_x=n_unique_x,
                            x_min=x_min,
                            x_max=x_max,
                            slope=slope,
                            slope_ci_low=lo,
                            slope_ci_high=hi,
                            intercept=intercept,
                            kendall_tau=tau,
                            kendall_p=p,
                            tail_winsorized=wz,
                            alignment_mode=mode,
                        )
                    )

    # Convert to DataFrame
    df = pd.DataFrame([r.__dict__ for r in rows])
    return df, rows


def summarize_across_languages(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Compute median slope + bootstrap CI across languages per (probe, layer, predictor)."""
    if df.empty:
        return df

    rng = np.random.RandomState(seed)
    groups = df.groupby(["probe", "layer", "predictor"], as_index=False)
    summary_rows: List[Dict[str, Any]] = []

    for (probe, layer, predictor), g in groups:
        # Use one slope per language (if multiple rows exist, keep all; bootstrap by language index)
        langs = g["language_slug"].unique().tolist()
        if not langs:
            continue
        # Collect per-language slopes (median per language if duplicates)
        per_lang = g.groupby("language_slug")["slope"].median().reset_index()
        base_median = float(per_lang["slope"].median())

        # Bootstrap across languages
        B = 2000
        boot_vals = []
        if len(per_lang) >= 3:
            arr = per_lang["slope"].values.astype(float)
            for _ in range(B):
                idx = rng.randint(0, len(arr), size=len(arr))
                boot_vals.append(np.median(arr[idx]))
        boot_lo = float(np.percentile(boot_vals, 2.5)) if boot_vals else np.nan
        boot_hi = float(np.percentile(boot_vals, 97.5)) if boot_vals else np.nan

        # Share negative slopes
        share_neg = float((per_lang["slope"] < 0).mean())

        summary_rows.append(
            {
                "probe": probe,
                "layer": layer,
                "predictor": predictor,
                "median_slope": base_median,
                "median_slope_ci_low": boot_lo,
                "median_slope_ci_high": boot_hi,
                "n_languages": int(len(per_lang)),
                "share_negative": share_neg,
            }
        )

    return pd.DataFrame(summary_rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)
        print(f"  ✓ Wrote {path} ({df.shape[0]} rows)")
    else:
        print(f"  ⚠ No rows to write for {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"], help="Data split")
    ap.add_argument("--winsorize", action="store_true", help="Enable x winsorization")
    ap.add_argument("--winsor_low", type=float, default=1.0, help="Lower pct for winsorization")
    ap.add_argument("--winsor_high", type=float, default=99.0, help="Upper pct for winsorization")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrapping")
    ap.add_argument("--out_tag", type=str, default="", help="Suffix tag for output filenames (e.g., 'seed42')")
    ap.add_argument("--dry_run", action="store_true", help="Do not write files; just report planned outputs")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs when not using out_tag")
    return ap.parse_args()


def main() -> None:
    print("THEIL–SEN TRENDS (RAW)")
    print("=" * 50)
    args = parse_args()
    ensure_dirs()

    # Languages
    languages = LANGUAGES or load_languages_from_analysis_table()
    print(f"Languages: {len(languages)}")
    print(f"Probes: {PROBES}")
    print(f"Layers: {LAYERS}")
    print(f"Split: {args.split}")
    print(f"Seed: {args.seed}")
    print(f"Out tag: {args.out_tag or '(none)'}")
    print()

    # Compute raw trends
    df, _ = compute_raw_trends(
        languages=languages,
        split=args.split,
        enable_winsor=args.winsorize,
        winsor_low=args.winsor_low,
        winsor_high=args.winsor_high,
    )

    # Determine output paths with optional tag
    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    per_layer_path = TRENDS_DIR / f"theilsen_per_layer_raw{tag_suffix}.csv"
    summary_path = TRENDS_DIR / f"theilsen_summary_by_layer{tag_suffix}.csv"

    if args.dry_run:
        print(f"[DRY-RUN] Would write per-layer: {per_layer_path} ({df.shape[0]} rows)")
        summary_df = summarize_across_languages(df, seed=args.seed)
        print(f"[DRY-RUN] Would write summary: {summary_path} ({summary_df.shape[0]} rows)")
    else:
        # Overwrite protection if no tag
        if not args.force and not args.out_tag:
            for p in [per_layer_path, summary_path]:
                if p.exists():
                    raise FileExistsError(f"Refusing to overwrite existing file without --force or --out_tag: {p}")
        # Write per-layer details
        write_csv(df, per_layer_path)
        # Write across-language summary
        summary_df = summarize_across_languages(df, seed=args.seed)
        write_csv(summary_df, summary_path)

    # Provenance
    prov = {
        "variant": "raw",
        "split": args.split,
        "winsorize": bool(args.winsorize),
        "winsor_low": float(args.winsor_low),
        "winsor_high": float(args.winsor_high),
        "languages": languages,
        "probes": PROBES,
        "layers": LAYERS,
        "qc": {"min_n_valid": MIN_N_VALID, "min_n_unique_x": MIN_N_UNIQUE_X},
    }
    prov.update({"seed": int(args.seed), "out_tag": args.out_tag})
    prov_path = TRENDS_DIR / f"provenance{tag_suffix}.json"
    if args.dry_run:
        print(f"[DRY-RUN] Would write provenance: {prov_path}")
    else:
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()



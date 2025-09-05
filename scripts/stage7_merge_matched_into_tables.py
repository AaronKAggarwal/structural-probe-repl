#!/usr/bin/env python3
"""
Merge Stage 7 length-matched metrics into Stage 6 analysis tables and produce
summary findings (deltas, rank stability, layer curves).

Inputs:
  - outputs/analysis/analysis_table_per_layer.csv (raw + covariates)
  - outputs/analysis/matched_eval_length_per_layer.csv (Stage 7 test)
  - outputs/analysis/analysis_table_L7.csv (raw L7 slice)
  - outputs/analysis/analysis_table_L7_adequate.csv (raw L7 adequate subset)

Outputs:
  - outputs/analysis/analysis_table_per_layer_with_lenmatch.csv
  - outputs/analysis/analysis_table_L7_with_lenmatch.csv
  - outputs/analysis/analysis_table_L7_adequate_with_lenmatch.csv
  - outputs/analysis/checks/stage7_merge_qc.json
  - outputs/analysis/stage7_findings/rankcorr_raw_vs_matched_by_layer_probe.csv
  - outputs/analysis/stage7_findings/delta_L7_by_language.csv
  - outputs/analysis/stage7_findings/figure_data/layer_curves_dist.csv
  - outputs/analysis/stage7_findings/figure_data/layer_curves_depth.csv
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
FINDINGS_DIR = ANALYSIS_DIR / "stage7_findings"
FIG_DIR = FINDINGS_DIR / "figure_data"
CHECKS_DIR = ANALYSIS_DIR / "checks"


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_layer = pd.read_csv(ANALYSIS_DIR / "analysis_table_per_layer.csv")
    matched = pd.read_csv(ANALYSIS_DIR / "matched_eval_length_per_layer.csv")
    l7 = pd.read_csv(ANALYSIS_DIR / "analysis_table_L7.csv")
    l7_adequate = pd.read_csv(ANALYSIS_DIR / "analysis_table_L7_adequate.csv")
    return per_layer, matched, l7, l7_adequate


def select_matched_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize depth naming (some runs may include both rootacc_* and root_acc_*)
    df = df.copy()
    if "rootacc_length_matched" not in df.columns and "root_acc_length_matched" in df.columns:
        df["rootacc_length_matched"] = df["root_acc_length_matched"]
    if "rootacc_length_matched_ci_low" not in df.columns and "root_acc_length_matched_ci_low" in df.columns:
        df["rootacc_length_matched_ci_low"] = df["root_acc_length_matched_ci_low"]
    if "rootacc_length_matched_ci_high" not in df.columns and "root_acc_length_matched_ci_high" in df.columns:
        df["rootacc_length_matched_ci_high"] = df["root_acc_length_matched_ci_high"]

    keep = [
        "language_slug",
        "probe",
        "layer",
        "uuas_length_matched",
        "uuas_length_matched_ci_low",
        "uuas_length_matched_ci_high",
        "rootacc_length_matched",
        "rootacc_length_matched_ci_low",
        "rootacc_length_matched_ci_high",
        "retention_ratio",
        "js_to_target",
        "bins_truncated_count",
        "duplication_95p",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()
    df = df.rename(
        columns={
            "retention_ratio": "lenmatch_retention",
            "js_to_target": "lenmatch_js_to_target",
            "bins_truncated_count": "lenmatch_bins_truncated_count",
            "duplication_95p": "lenmatch_duplication_95p",
        }
    )
    return df


def merge_lenmatched(per_layer: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    matched_sel = select_matched_columns(matched)
    merged = per_layer.merge(matched_sel, on=["language_slug", "probe", "layer"], how="left", validate="one_to_one")
    # Deltas (probe-specific)
    merged["delta_uuas_lenmatch"] = np.where(
        merged["probe"] == "dist",
        merged["uuas_length_matched"] - merged.get("uuas", np.nan),
        np.nan,
    )
    merged["delta_rootacc_lenmatch"] = np.where(
        merged["probe"] == "depth",
        merged["rootacc_length_matched"] - merged.get("root_acc", np.nan),
        np.nan,
    )
    return merged


def compute_rankcorr(per_layer_with: pd.DataFrame) -> pd.DataFrame:
    rows = []
    layers = ["L5", "L6", "L7", "L8", "L9", "L10"]
    for probe in ["dist", "depth"]:
        for layer in layers:
            sub = per_layer_with[(per_layer_with["probe"] == probe) & (per_layer_with["layer"] == layer)].copy()
            if probe == "dist":
                x = sub["uuas"]
                y = sub["uuas_length_matched"]
            else:
                x = sub["root_acc"]
                y = sub["rootacc_length_matched"]
            m = pd.DataFrame({"x": x, "y": y}).dropna()
            n = len(m)
            rho = np.nan
            pval = np.nan
            if n >= 2:
                rho, pval = spearmanr(m["x"], m["y"])  # type: ignore
            rows.append({
                "probe": probe,
                "layer": layer,
                "n_languages": n,
                "spearman_rho": rho,
                "spearman_pvalue": pval,
            })
    return pd.DataFrame(rows)


def build_layer_curves(per_layer_with: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # dist
    dist = per_layer_with[per_layer_with["probe"] == "dist"][
        [
            "language_slug",
            "layer",
            "uuas",
            "uuas_length_matched",
            "uuas_length_matched_ci_low",
            "uuas_length_matched_ci_high",
        ]
    ].rename(columns={
        "uuas": "raw",
        "uuas_length_matched": "matched",
        "uuas_length_matched_ci_low": "matched_ci_low",
        "uuas_length_matched_ci_high": "matched_ci_high",
    })

    depth = per_layer_with[per_layer_with["probe"] == "depth"][
        [
            "language_slug",
            "layer",
            "root_acc",
            "rootacc_length_matched",
            "rootacc_length_matched_ci_low",
            "rootacc_length_matched_ci_high",
        ]
    ].rename(columns={
        "root_acc": "raw",
        "rootacc_length_matched": "matched",
        "rootacc_length_matched_ci_low": "matched_ci_low",
        "rootacc_length_matched_ci_high": "matched_ci_high",
    })
    return dist, depth


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    per_layer, matched, l7, l7_adequate = load_inputs()

    per_layer_with = merge_lenmatched(per_layer, matched)

    # Write merged tables
    out_per_layer = ANALYSIS_DIR / "analysis_table_per_layer_with_lenmatch.csv"
    per_layer_with.to_csv(out_per_layer, index=False)

    l7_with = per_layer_with[per_layer_with["layer"] == "L7"].copy()
    out_l7 = ANALYSIS_DIR / "analysis_table_L7_with_lenmatch.csv"
    l7_with.to_csv(out_l7, index=False)

    # Merge onto adequate L7 list (to preserve same subset)
    l7_adequate_keys = l7_adequate[["language_slug", "probe", "layer"]].drop_duplicates()
    l7_adequate_with = l7_with.merge(l7_adequate_keys, on=["language_slug", "probe", "layer"], how="inner")
    out_l7_adequate = ANALYSIS_DIR / "analysis_table_L7_adequate_with_lenmatch.csv"
    l7_adequate_with.to_csv(out_l7_adequate, index=False)

    # Findings outputs
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rankcorr = compute_rankcorr(per_layer_with)
    rankcorr.to_csv(FINDINGS_DIR / "rankcorr_raw_vs_matched_by_layer_probe.csv", index=False)

    # L7 deltas
    delta_cols = [
        "language_slug",
        "probe",
        "uuas",
        "uuas_length_matched",
        "uuas_length_matched_ci_low",
        "uuas_length_matched_ci_high",
        "delta_uuas_lenmatch",
        "root_acc",
        "rootacc_length_matched",
        "rootacc_length_matched_ci_low",
        "rootacc_length_matched_ci_high",
        "delta_rootacc_lenmatch",
        "lenmatch_retention",
        "lenmatch_js_to_target",
    ]
    l7_deltas = l7_with[delta_cols].copy()
    l7_deltas.to_csv(FINDINGS_DIR / "delta_L7_by_language.csv", index=False)

    # Layer curves
    dist_curves, depth_curves = build_layer_curves(per_layer_with)
    dist_curves.to_csv(FIG_DIR / "layer_curves_dist.csv", index=False)
    depth_curves.to_csv(FIG_DIR / "layer_curves_depth.csv", index=False)

    # QC summary
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    qc = {}
    qc["row_count_after_merge"] = int(len(per_layer_with))
    qc["unique_keys"] = int(per_layer_with[["language_slug", "probe", "layer"]].drop_duplicates().shape[0])
    # Range checks
    def in01(s: pd.Series) -> bool:
        t = s.dropna()
        return bool(((t >= 0) & (t <= 1)).all())

    qc["uuas_lenmatch_in_01"] = in01(per_layer_with.get("uuas_length_matched", pd.Series(dtype=float)))
    qc["rootacc_lenmatch_in_01"] = in01(per_layer_with.get("rootacc_length_matched", pd.Series(dtype=float)))
    # CI order
    def ci_ok(low: str, mid: str, high: str) -> bool:
        if low not in per_layer_with or high not in per_layer_with or mid not in per_layer_with:
            return True
        df = per_layer_with[[low, mid, high]].dropna()
        return bool((df[low] <= df[mid]).all() and (df[mid] <= df[high]).all())

    qc["uuas_ci_ok"] = ci_ok("uuas_length_matched_ci_low", "uuas_length_matched", "uuas_length_matched_ci_high")
    qc["rootacc_ci_ok"] = ci_ok("rootacc_length_matched_ci_low", "rootacc_length_matched", "rootacc_length_matched_ci_high")
    qc["lenmatch_retention_in_01"] = in01(per_layer_with.get("lenmatch_retention", pd.Series(dtype=float)))
    qc["lenmatch_bins_truncated_count_range"] = bool(
        per_layer_with.get("lenmatch_bins_truncated_count", pd.Series(dtype=float)).dropna().between(0, 6).all()
    )
    qc["delta_uuas_range"] = bool(
        per_layer_with.get("delta_uuas_lenmatch", pd.Series(dtype=float)).dropna().between(-1, 1).all()
    )
    qc["delta_rootacc_range"] = bool(
        per_layer_with.get("delta_rootacc_lenmatch", pd.Series(dtype=float)).dropna().between(-1, 1).all()
    )

    # File hashes
    qc["sha256_matched_eval_csv"] = sha256_file(ANALYSIS_DIR / "matched_eval_length_per_layer.csv")
    qc["sha256_target_bins_json"] = sha256_file(ANALYSIS_DIR / "matched_targets" / "length_bins.json")

    (CHECKS_DIR / "stage7_merge_qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()



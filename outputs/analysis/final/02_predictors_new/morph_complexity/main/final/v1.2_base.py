#!/usr/bin/env python3
"""
Compute a coverage-orthogonal morphological complexity index from UD FEATS
using a restricted pool and fixed-size, stratified sampling.

Configuration (defaults in code):
- Excludes: Chinese/Japanese/Korean/Vietnamese treebanks (explicit list)
- Pool: CORE_POS = {NOUN, VERB, ADJ}
- Target mix: equal thirds (n // 3 for each; remainder distributed by availability)
- Sample size per language: N = 1000
- Replicates: R = 400 (independent; seeds derived deterministically)
- Entropy gate: H_UPOS(all marked content) ≥ 1.5 bits

Outputs (written to the same directory as this script):
- output.csv: main results per language (language_slug, complex_pc1)
- morph_complexity_core.csv: detailed results per language (metrics, CIs, z-scores, PC1)
- fail_list.csv: languages that failed gates with reasons
- metadata.yaml: run configuration and environment info
- pca_params.json: PCA parameters (means, scales, components, explained variance)

Notes:
- Content tokens exclude PUNCT and SYM
- Marked tokens have FEATS != '_'
- Canonicalization: sort keys; sort multi-values; join with |
- Metrics per replicate: mean FEATS/token, Miller–Madow bundle entropy, bundle richness
- Diagnostics per language: volatile tag shares (NUM/AUX/PART/X) over all marked tokens;
  per-key presence rates on sampled tokens (averaged over replicates)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ------------------------- 
# Configuration
# -------------------------

REPO_ROOT = Path(__file__).resolve().parents[7]  # Adjusted for correct path depth
OUTPUT_DIR = Path(__file__).resolve().parent

def find_ud_data_root():
    """Find the UD data directory."""
    candidates = [
        REPO_ROOT / "data" / "tmp_ud_clones",
        REPO_ROOT / "data" / "ud-treebanks-v2.8",
        REPO_ROOT / "data" / "ud",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Fallback search
    ud_dirs = list(REPO_ROOT.glob("**/UD_*"))
    if ud_dirs:
        return ud_dirs[0].parent
    
    raise FileNotFoundError(f"No UD data found. Searched: {candidates}")

UD_DATA_ROOT = find_ud_data_root()

# Treebanks to exclude upfront (explicit set; extend if needed)
EXCLUDE_LANGS = {
    "UD_Chinese-GSD",
    "UD_Japanese-GSD", 
    "UD_Korean-GSD",
    "UD_Vietnamese-VTB",
}

CORE_POS = {"NOUN", "VERB", "ADJ"}
VOLATILE_POS = ["NUM", "AUX", "PART", "X"]
N = 1000
R = 400
MASTER_SEED = 42
H_UPOS_ALL_THRESHOLD = 1.5  # bits

# FEATS keys to summarize (diagnostics)
CORE_KEYS = [
    "Case", "Number", "Gender", "Person", "Tense", "Mood", "Aspect", 
    "Voice", "Polarity", "Degree", "Definite"
]


# -------------------------
# Helpers
# -------------------------

def canonicalize_feats_bundle(feats: str) -> str:
    """Canonicalize FEATS bundle: sort keys and multi-values."""
    if not feats or feats == "_":
        return "_"
    
    pairs: List[str] = []
    for pair in feats.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            if "," in v:
                v = ",".join(sorted(v.split(",")))
            pairs.append(f"{k}={v}")
    
    return "|".join(sorted(pairs)) if pairs else "_"


def parse_marked_tokens(train_file: Path) -> Tuple[List[Tuple[str, str]], Counter, Counter, int, int, Counter]:
    """Parse marked content tokens.
    
    Returns:
    - items: list of (upos, canonical_bundle) for CORE_POS tokens only
    - core_counts: counts per CORE_POS among marked
    - all_upos_counts: counts per UPOS among marked (all UPOS)
    - total_marked: number of marked content tokens
    - total_core_marked: number of marked CORE_POS tokens
    - volatile_counts_all: counts for volatile POS among marked (NUM/AUX/PART/X)
    """
    items: List[Tuple[str, str]] = []
    core_counts: Counter = Counter()
    all_upos_counts: Counter = Counter()
    volatile_counts_all: Counter = Counter()
    total_marked = 0
    total_core_marked = 0
    
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            
            tid = fields[0]
            if "-" in tid or "." in tid:
                continue
            
            upos = fields[3]
            feats = fields[5]
            
            # Skip punctuation (content-only policy)
            if upos in {"PUNCT", "SYM"}:
                continue
            
            # Skip unmarked tokens
            if feats == "_":
                continue
            
            # Skip foreign tokens (prevents contamination from foreign language insertions)
            if "Foreign=Yes" in feats:
                continue
            
            total_marked += 1
            all_upos_counts[upos] += 1
            
            if upos in VOLATILE_POS:
                volatile_counts_all[upos] += 1
            
            if upos in CORE_POS:
                canon = canonicalize_feats_bundle(feats)
                items.append((upos, canon))
                core_counts[upos] += 1
                total_core_marked += 1
    
    return items, core_counts, all_upos_counts, total_marked, total_core_marked, volatile_counts_all


def entropy_from_counts(counts: Counter) -> float:
    """Compute Shannon entropy from counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log(p, 2)
    
    return H


def derive_replicate_seed(master: int, lang: str, r: int) -> int:
    """Derive deterministic seed for replicate (stable across processes)."""
    key = f"{master}|{lang}|{r}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    seed32 = int(h[:16], 16) % (2**31 - 1)
    return int(seed32)


def sample_indices_stratified(items: List[Tuple[str, str]], rng: np.random.Generator, n_total: int) -> List[int]:
    """Sample indices with stratified sampling across CORE_POS."""
    # Equal thirds per CORE_POS; remainder to POS with largest availability
    base_quota = n_total // 3
    remainder = n_total - base_quota * 3
    
    idx_by_pos: Dict[str, List[int]] = {pos: [] for pos in CORE_POS}
    for i, (upos, _) in enumerate(items):
        if upos in CORE_POS:
            idx_by_pos[upos].append(i)
    
    quotas = {pos: base_quota for pos in CORE_POS}
    if remainder > 0:
        avail_sorted = sorted(CORE_POS, key=lambda p: len(idx_by_pos[p]), reverse=True)
        for k in range(remainder):
            quotas[avail_sorted[k % len(avail_sorted)]] += 1
    
    chosen: List[int] = []
    for pos, need in quotas.items():
        pool = idx_by_pos[pos]
        take = min(need, len(pool))
        
        if take < need:
            raise RuntimeError(f"Insufficient pool in {pos}: need {need}, have {len(pool)}")
        
        sel = rng.choice(len(pool), size=take, replace=False)
        chosen.extend([pool[j] for j in sel])
    
    return chosen


def compute_metrics(sampled_items: List[Tuple[str, str]]) -> Tuple[float, float, int, Dict[str, float]]:
    """Compute metrics for a sample: mean FEATS/token, bundle entropy (MM), bundle richness, per-key presence rates."""
    feat_counts = []
    bundles = []
    key_presence = Counter()
    
    for _, bundle in sampled_items:
        if bundle == "_":
            feat_counts.append(0)
        else:
            parts = bundle.split("|")
            feat_counts.append(len(parts))
            
            # Count key presence
            for kv in parts:
                if "=" in kv:
                    k = kv.split("=", 1)[0]
                    if k in CORE_KEYS:
                        key_presence[k] += 1
        
        bundles.append(bundle)
    
    n = len(sampled_items)
    mean_feats = float(np.mean(feat_counts)) if n > 0 else 0.0
    
    # Miller-Madow entropy correction
    counter = Counter(bundles)
    probs = [c / n for c in counter.values()] if n > 0 else []
    H_mle = -sum(p * math.log(p, 2) for p in probs if p > 0)
    K = len(counter)
    mm_correction = (K - 1) / (2 * n * math.log(2)) if n > 0 else 0.0
    H_mm = H_mle + mm_correction
    
    richness = K
    
    # Presence rates per key
    key_rates = {k: (key_presence.get(k, 0) / n if n > 0 else 0.0) for k in CORE_KEYS}
    
    return mean_feats, H_mm, richness, key_rates


def aggregate_replicates(values: List[float]) -> Dict[str, float]:
    """Aggregate replicate values with mean and confidence intervals."""
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "ci_low": float(np.percentile(arr, 2.5)) if arr.size else 0.0,
        "ci_high": float(np.percentile(arr, 97.5)) if arr.size else 0.0,
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def compute_overall_marking_rate(train_file: Path) -> float:
    """Compute overall marking rate from train file for this language."""
    total_content = 0
    total_marked = 0
    
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            
            tid = fields[0]
            if "-" in tid or "." in tid:
                continue
            
            upos = fields[3]
            feats = fields[5]
            
            # Skip punctuation (content-only policy)
            if upos in {"PUNCT", "SYM"}:
                continue
            
            # Skip foreign tokens for consistency with main parsing
            if feats != "_" and "Foreign=Yes" in feats:
                continue
            
            total_content += 1
            if feats != "_":
                total_marked += 1
    
    return total_marked / total_content if total_content > 0 else 0.0


def compute_coverage_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Compute coverage controls for residualization."""
    controls = df.copy()
    
    # Coverage_norm: normalize total marked tokens by total content tokens
    # Approximate total content tokens as total_marked / overall_marking_rate (if available)
    if 'total_marked' in controls.columns:
        # Estimate total content tokens
        total_content_est = controls['total_marked'] / np.maximum(controls.get('overall_marking_rate', 0.5), 0.01)
        controls['coverage_norm'] = controls['total_marked'] / total_content_est.median()  # Normalize by median
    else:
        controls['coverage_norm'] = 1.0  # Default if not available
    
    # MWT_share: Multi-word token share (approximate using volatile NUM/AUX)
    volatile_cols = [f'volatile_share_{pos}' for pos in VOLATILE_POS if f'volatile_share_{pos}' in controls.columns]
    if volatile_cols:
        controls['mwt_share'] = controls[volatile_cols].sum(axis=1)
    else:
        controls['mwt_share'] = 0.0
    
    # X/PROPN share: Non-standard token share
    x_propn_cols = ['volatile_share_X'] if 'volatile_share_X' in controls.columns else []
    if x_propn_cols:
        controls['x_propn_share'] = controls[x_propn_cols].sum(axis=1)
    else:
        controls['x_propn_share'] = 0.0
    
    return controls


def residualize_metrics(df: pd.DataFrame, metric_cols: List[str], control_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Residualize metrics on coverage controls before PCA."""
    residualized = df.copy()
    residualization_info = {}
    
    for metric_col in metric_cols:
        if metric_col not in df.columns:
            continue
            
        # Available controls
        available_controls = [col for col in control_cols if col in df.columns]
        if not available_controls:
            # No controls available, use original metric
            residualized[f'resid_{metric_col}'] = df[metric_col]
            continue
        
        # Clean data
        use_cols = [metric_col] + available_controls
        clean_data = df[use_cols].dropna()
        
        if len(clean_data) < len(available_controls) + 2:
            # Insufficient data for regression
            residualized[f'resid_{metric_col}'] = df[metric_col]
            continue
        
        # Fit regression: metric ~ controls
        X = clean_data[available_controls].values
        y = clean_data[metric_col].values
        
        reg = LinearRegression().fit(X, y)
        
        # Predict for all data (using available controls)
        X_all = df[available_controls].fillna(0).values  # Fill NA with 0 for prediction
        y_pred_all = reg.predict(X_all)
        
        # Compute residuals
        residuals = df[metric_col] - y_pred_all
        residualized[f'resid_{metric_col}'] = residuals
        
        # Store residualization info
        residualization_info[metric_col] = {
            'controls_used': available_controls,
            'r_squared': float(reg.score(X, y)),
            'n_used': len(clean_data),
            'coefficients': {ctrl: float(coef) for ctrl, coef in zip(available_controls, reg.coef_)},
            'intercept': float(reg.intercept_)
        }
    
    return residualized, residualization_info


def main() -> None:
    """Main computation function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    languages = sorted([p.name for p in UD_DATA_ROOT.iterdir() if p.is_dir()])
    
    rows: List[Dict[str, object]] = []
    fail_rows: List[Dict[str, object]] = []
    
    # Pre-collect volatile shares across marked tokens
    lang_to_volatile_share: Dict[str, Dict[str, float]] = {}
    
    for lang in languages:
        if lang in EXCLUDE_LANGS or any(x in lang for x in ["Japanese", "Korean", "Vietnamese"]):
            fail_rows.append({"language_slug": lang, "status": "EXCLUDED", "reasons": "pre-excluded"})
            continue
        
        # Look for train file with various naming patterns
        train_candidates = [
            UD_DATA_ROOT / lang / "train.conllu",
            *list((UD_DATA_ROOT / lang).glob("*-ud-train.conllu")),
            *list((UD_DATA_ROOT / lang).glob("*train*.conllu")),
        ]
        
        train = None
        for candidate in train_candidates:
            if candidate.exists():
                train = candidate
                break
        
        if train is None:
            fail_rows.append({"language_slug": lang, "status": "FAIL", "reasons": "no train.conllu"})
            continue
        
        try:
            items, core_counts, all_upos_counts, total_marked, total_core_marked, volatile_counts = parse_marked_tokens(train)
            # Compute overall marking rate for this language
            overall_marking_rate = compute_overall_marking_rate(train)
        except Exception as e:
            fail_rows.append({"language_slug": lang, "status": "FAIL", "reasons": f"parse error: {e}"})
            continue
        
        # Volatile shares (global among marked)
        total_marked_all = sum(all_upos_counts.values())
        volatile_share = {pos: (volatile_counts.get(pos, 0) / total_marked_all if total_marked_all > 0 else 0.0) for pos in VOLATILE_POS}
        lang_to_volatile_share[lang] = volatile_share
        
        # Gates
        reasons: List[str] = []
        
        if total_marked < N:
            reasons.append(f"marked_total<{N} ({total_marked})")
        
        base_quota = N // 3
        for pos in CORE_POS:
            if core_counts[pos] < base_quota:
                reasons.append(f"{pos}<{base_quota} ({core_counts[pos]})")
        
        H_all = entropy_from_counts(all_upos_counts)
        if total_marked_all == 0:
            reasons.append("all_upos_total=0")
        elif H_all < H_UPOS_ALL_THRESHOLD:
            reasons.append(f"H_upos_all<{H_UPOS_ALL_THRESHOLD:.1f} ({H_all:.2f})")
        
        if reasons:
            fail_rows.append({
                "language_slug": lang,
                "status": "FAIL",
                "reasons": "; ".join(reasons),
                "total_marked": total_marked,
                "core_marked": total_core_marked,
                "counts_NOUN": core_counts.get("NOUN", 0),
                "counts_VERB": core_counts.get("VERB", 0),
                "counts_ADJ": core_counts.get("ADJ", 0),
                "H_upos_all": H_all,
            })
            continue
        
        # Replicates
        mean_feats_vals: List[float] = []
        entropy_vals: List[float] = []
        richness_vals: List[float] = []
        key_rate_accum = defaultdict(list)  # key -> [rates per replicate]
        
        # Precompute quotas for audit
        base_quota = N // 3
        remainder = N - base_quota * 3
        idx_by_pos = {pos: [] for pos in CORE_POS}
        for i, (upos, _) in enumerate(items):
            if upos in CORE_POS:
                idx_by_pos[upos].append(i)
        quotas = {pos: base_quota for pos in CORE_POS}
        if remainder > 0:
            avail_sorted = sorted(CORE_POS, key=lambda p: len(idx_by_pos[p]), reverse=True)
            for k in range(remainder):
                quotas[avail_sorted[k % len(avail_sorted)]] += 1
        
        for r in range(R):
            seed = derive_replicate_seed(MASTER_SEED, lang, r)
            rng = np.random.default_rng(seed)
            
            idxs = sample_indices_stratified(items, rng, N)
            sampled = [items[i] for i in idxs]
            
            m, h, k, key_rates = compute_metrics(sampled)
            mean_feats_vals.append(m)
            entropy_vals.append(h)
            richness_vals.append(k)
            
            for kkey, rate in key_rates.items():
                key_rate_accum[kkey].append(rate)
        
        agg_mean_feats = aggregate_replicates(mean_feats_vals)
        agg_entropy = aggregate_replicates(entropy_vals)
        agg_richness = aggregate_replicates(richness_vals)
        
        mean_key_rates = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in key_rate_accum.items()}
        
        rows.append({
            "language_slug": lang,
            "total_marked": total_marked,
            "core_marked": total_core_marked,
            "counts_NOUN": core_counts.get("NOUN", 0),
            "counts_VERB": core_counts.get("VERB", 0),
            "counts_ADJ": core_counts.get("ADJ", 0),
            "H_upos_all": H_all,
            "overall_marking_rate": overall_marking_rate,
            # Quotas drawn (audit)
            "sampled_NOUN": quotas.get("NOUN", 0),
            "sampled_VERB": quotas.get("VERB", 0),
            "sampled_ADJ": quotas.get("ADJ", 0),
            # Metrics (replicate aggregates)
            "mean_feats": agg_mean_feats["mean"],
            "mean_feats_ci_low": agg_mean_feats["ci_low"],
            "mean_feats_ci_high": agg_mean_feats["ci_high"],
            "entropy_mm": agg_entropy["mean"],
            "entropy_mm_ci_low": agg_entropy["ci_low"],
            "entropy_mm_ci_high": agg_entropy["ci_high"],
            "richness": agg_richness["mean"],
            "richness_ci_low": agg_richness["ci_low"],
            "richness_ci_high": agg_richness["ci_high"],
            # Diagnostics: volatile shares
            **{f"volatile_share_{pos}": volatile_share[pos] for pos in VOLATILE_POS},
            # Diagnostics: per-key presence rates (averaged)
            **{f"key_rate_{k}": mean_key_rates.get(k, 0.0) for k in CORE_KEYS},
        })
    
    # Build DataFrame
    df = pd.DataFrame(rows)
    df_fail = pd.DataFrame(fail_rows)
    
    if not df.empty:
        # Z-score metrics across languages (using replicate means)
        z_means = {}
        z_stds = {}
        for col in ["mean_feats", "entropy_mm", "richness"]:
            mu = df[col].mean()
            sigma = df[col].std(ddof=0)
            z_means[col] = float(mu)
            z_stds[col] = float(sigma)
            df[f"z_{col}"] = (df[col] - mu) / (sigma if sigma > 0 else 1.0)
        
        # Residualize z-scored metrics directly on overall_marking_rate BEFORE PCA
        metric_cols = ["z_mean_feats", "z_entropy_mm", "z_richness"]
        control_cols = ["overall_marking_rate"]  # Direct residualization on the actual coverage measure
        
        df, residualization_info = residualize_metrics(df, metric_cols, control_cols)
        
        # PCA on residualized metrics (3 -> 1)
        resid_cols = [f"resid_{col}" for col in metric_cols]
        Z_resid = df[resid_cols].to_numpy()
        
        pca = PCA(n_components=3, random_state=MASTER_SEED)
        scores = pca.fit_transform(Z_resid)
        pc1 = scores[:, 0]
        
        # Align sign so that PC1 correlates positively with residualized z_mean_feats
        corr = np.corrcoef(pc1, Z_resid[:, 0])[0, 1]
        if corr < 0:
            pc1 = -pc1
            pca.components_[0] = -pca.components_[0]
        
        df["complex_pc1"] = pc1
        df["pca_explained_var_pc1"] = pca.explained_variance_ratio_[0]
        df["pca_explained_var_total"] = pca.explained_variance_ratio_.sum()
        df["pca_loading_resid_mean_feats"] = pca.components_[0, 0]
        df["pca_loading_resid_entropy_mm"] = pca.components_[0, 1]
        df["pca_loading_resid_richness"] = pca.components_[0, 2]
        
        # Save PCA params and residualization info
        pca_params = {
            "features_original": ["z_mean_feats", "z_entropy_mm", "z_richness"],
            "features_residualized": resid_cols,
            "coverage_controls": control_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
            "pc1_aligned_with_resid_mean_feats": True,
            "z_means": z_means,
            "z_stds": z_stds,
            "residualization_info": residualization_info,
        }
        with open(OUTPUT_DIR / "pca_params.json", "w") as f:
            json.dump(pca_params, f, indent=2)
        
        # Save separate residualization report
        with open(OUTPUT_DIR / "residualization_report.json", "w") as f:
            json.dump(residualization_info, f, indent=2)
    
    # Save outputs
    if not df.empty:
        # Main output in standard format
        output_df = df[["language_slug", "complex_pc1"]].copy()
        output_df.to_csv(OUTPUT_DIR / "output.csv", index=False)
        
        # Detailed output
        df.to_csv(OUTPUT_DIR / "morph_complexity_core.csv", index=False)
    
    if not df_fail.empty:
        df_fail.to_csv(OUTPUT_DIR / "fail_list.csv", index=False)
    
    # Metadata
    metadata = {
        "name": "v1.2 Core POS Stratified Morphological Complexity (Residualized + Foreign Excluded)",
        "version": "1.2_foreign_excluded",
        "description": "Coverage-orthogonal complexity index using stratified sampling from NOUN/VERB/ADJ with pre-PCA residualization and Foreign=Yes token exclusion",
        "timestamp": start_ts,
        "config": {
            "excluded": sorted(list(EXCLUDE_LANGS)),
            "core_pos": sorted(list(CORE_POS)),
            "volatile_pos": VOLATILE_POS,
            "sample_size": N,
            "replicates": R,
            "master_seed": MASTER_SEED,
            "entropy_gate_bits_all_upos": H_UPOS_ALL_THRESHOLD,
        },
        "paths": {
            "ud_data_root": str(UD_DATA_ROOT),
            "output_dir": str(OUTPUT_DIR),
            "results_csv": str(OUTPUT_DIR / "output.csv"),
            "detailed_csv": str(OUTPUT_DIR / "morph_complexity_core.csv"),
            "fail_list_csv": str(OUTPUT_DIR / "fail_list.csv"),
            "pca_params": str(OUTPUT_DIR / "pca_params.json"),
        },
        "results": {
            "languages_processed": int(df.shape[0]) if not df.empty else 0,
            "languages_failed": int(df_fail.shape[0]) if not df_fail.empty else 0,
        }
    }
    
    with open(OUTPUT_DIR / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, indent=2)
    
    print("Computed v1.2 Core POS Stratified (Residualized on Marking Rate) complexity metric")
    print("Main results: {}".format(OUTPUT_DIR / 'output.csv'))
    print("Detailed results: {}".format(OUTPUT_DIR / 'morph_complexity_core.csv'))
    if not df_fail.empty:
        print("Failed languages: {}".format(OUTPUT_DIR / 'fail_list.csv'))
    print("Metadata: {}".format(OUTPUT_DIR / 'metadata.yaml'))


if __name__ == "__main__":
    main()

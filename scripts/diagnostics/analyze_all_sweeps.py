# scripts/analyze_llama_sweeps.py
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Plotting Style Configuration ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "serif"
# Use a palette that provides good color distinction for two models + their error bands
PALETTE = {"Llama-3.2-3B": sns.color_palette("viridis", 4)[0], 
           "Llama-3.2-3B-Instruct": sns.color_palette("viridis", 4)[2]}


def find_and_parse_llama_results(results_base_dir: Path) -> pd.DataFrame:
    """
    Finds Llama 3.2 base and instruct results for depth and distance probes and parses them.
    """
    all_results_data = []
    
    # Use rglob to find all summary files recursively within the specified directory
    result_files = list(results_base_dir.rglob("**/metrics_summary.json"))

    if not result_files:
        raise FileNotFoundError(f"No 'metrics_summary.json' files found under '{results_base_dir}'.")

    print(f"Found {len(result_files)} total result files. Filtering for Llama 3.2 models...")

    # Regex to capture model name, probe type, and layer from the path.
    # It handles both 'llama-3.2-3b' and 'Llama-3.2-3B' style directory names.
    pattern = re.compile(
        r"/(Llama-3\.2-3B-Instruct|llama-3\.2-3b-instruct|Llama-3\.2-3B|llama-3\.2-3b)/(depth|dist)/L(\d+)/metrics_summary\.json"
    )

    for result_file in result_files:
        match = pattern.search(str(result_file))
        if not match:
            continue
        
        model_name_raw, probe_type, layer_str = match.groups()
        layer = int(layer_str)

        # Standardize model names for consistent plotting labels
        if "instruct" in model_name_raw.lower():
            model_name = "Llama-3.2-3B-Instruct"
        else:
            model_name = "Llama-3.2-3B"
            
        with open(result_file, 'r') as f:
            metrics = json.load(f)
        
        all_results_data.append({
            "model": model_name,
            "probe_type": probe_type,
            "layer": layer,
            "dspr_nspr": metrics.get("test_spearmanr_hm", np.nan),
            "uuas_root_acc": metrics.get("test_uuas", np.nan) if probe_type == 'dist' else metrics.get("test_root_acc", np.nan),
        })

    if not all_results_data:
        raise ValueError("No valid Llama 3.2 results could be parsed from the found files.")

    df = pd.DataFrame(all_results_data)
    df = df.sort_values(by=["model", "probe_type", "layer"]).reset_index(drop=True)
    return df

def plot_comparison(df: pd.DataFrame, probe_type: str, metric: str, ylabel: str, title: str, output_path: Path):
    """Creates and saves a plot comparing the base and instruct models for a single metric."""
    plt.figure(figsize=(16, 9))
    
    plot_df = df[df['probe_type'] == probe_type].copy()
    
    if plot_df.empty:
        print(f"No data to plot for probe type '{probe_type}'. Skipping plot.")
        return

    ax = sns.lineplot(data=plot_df, x='layer', y=metric, hue='model', marker='o', 
                      markersize=8, linewidth=2.5, palette=PALETTE)

    # Annotate the peak for each model
    for model_name in plot_df['model'].unique():
        model_df = plot_df[plot_df['model'] == model_name]
        if not model_df.empty and not model_df[metric].isnull().all():
            best_layer_idx = model_df[metric].idxmax()
            best_layer_data = model_df.loc[best_layer_idx]
            best_layer = int(best_layer_data['layer'])
            best_value = best_layer_data[metric]
            
            ax.annotate(
                f"{model_name}\nPeak: {best_value:.3f}",
                xy=(best_layer, best_value),
                xytext=(10, -30), textcoords='offset points',
                fontsize=14,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8)
            )

    ax.set_title(title, fontsize=22, pad=20, weight='bold')
    ax.set_xlabel("Model Layer Index", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(title='Model', fontsize=14, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def main(results_dir: str):
    """Main function to orchestrate the analysis."""
    project_root = Path(__file__).resolve().parents[2]
    results_base_path = project_root / results_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = project_root / "analysis" / f"llama_base_vs_instruct_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis outputs will be saved to: {output_dir}")
    
    try:
        results_df = find_and_parse_llama_results(results_base_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n--- Aggregated Data ---")
    print(results_df.to_string())

    print("\n--- Generating Plots ---")
    plot_comparison(
        df=results_df,
        probe_type="depth",
        metric="dspr_nspr",
        ylabel="NSpr (Spearman Correlation)",
        title="Syntactic Depth (NSpr) vs. Layer\nLlama-3.2-3B vs. Instruct",
        output_path=output_dir / "comparison_depth_nspr.png"
    )
    
    plot_comparison(
        df=results_df,
        probe_type="depth",
        metric="uuas_root_acc",
        ylabel="Root Accuracy",
        title="Syntactic Depth (Root Accuracy) vs. Layer\nLlama-3.2-3B vs. Instruct",
        output_path=output_dir / "comparison_depth_root_acc.png"
    )
    
    plot_comparison(
        df=results_df,
        probe_type="dist",
        metric="dspr_nspr",
        ylabel="DSpr (Spearman Correlation)",
        title="Syntactic Distance (DSpr) vs. Layer\nLlama-3.2-3B vs. Instruct",
        output_path=output_dir / "comparison_distance_dspr.png"
    )
    
    plot_comparison(
        df=results_df,
        probe_type="dist",
        metric="uuas_root_acc",
        ylabel="UUAS (Undirected Unlabeled Attachment Score)",
        title="Syntactic Distance (UUAS) vs. Layer\nLlama-3.2-3B vs. Instruct",
        output_path=output_dir / "comparison_distance_uuas.png"
    )
    
    print(f"\n--- Analysis complete! Find plots and data in {output_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and compare Llama 3.2 Base vs. Instruct probe sweeps.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="multirun/final",
        help="Path to the directory containing the Llama 3.2 sweep folders, relative to project root."
    )
    args = parser.parse_args()
    
    main(args.results_dir)
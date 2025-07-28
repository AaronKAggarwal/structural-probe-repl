# scripts/analyze_llama_comparison.py
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Add src to path if needed ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

PALETTE = {"Llama-3.2-3B": sns.color_palette("viridis", 4)[0], 
           "Llama-3.2-3B-Instruct": sns.color_palette("viridis", 4)[2]}

# --- Plotting Style Configuration ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "serif"
# Define distinct colors and styles for clarity on the dual-axis plots
BASE_MODEL_COLOR = "#1f77b4"  # Muted Blue
INSTRUCT_MODEL_COLOR = "#ff7f0e" # Safety Orange

def find_and_parse_llama_results(results_base_dir: Path) -> pd.DataFrame:
    """
    Finds all Llama 3.2 base and instruct results for depth and distance probes and parses them.
    """
    all_results_data = []
    
    result_files = list(results_base_dir.rglob("**/metrics_summary.json"))

    if not result_files:
        raise FileNotFoundError(f"No 'metrics_summary.json' files found under '{results_base_dir}'.")

    print(f"Found {len(result_files)} total result files. Filtering for Llama 3.2 models...")

    pattern = re.compile(
        r"/(Llama-3\.2-3B-Instruct|llama-3\.2-3b-instruct|Llama-3\.2-3B|llama-3\.2-3b)/(depth|dist)/L(\d+)/metrics_summary\.json"
    )

    for result_file in result_files:
        match = pattern.search(str(result_file))
        if not match:
            continue
        
        model_name_raw, probe_type, layer_str = match.groups()
        layer = int(layer_str)

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

def plot_dual_axis_comparison(
    df: pd.DataFrame, 
    probe_type: str, 
    metric1_col: str, metric2_col: str, 
    ylabel1: str, ylabel2: str, 
    title: str, 
    output_path: Path
):
    """
    Creates and saves a dual-axis plot comparing Base and Instruct models.
    
    Args:
        df: DataFrame containing all aggregated results.
        probe_type: 'dist' or 'depth'.
        metric1_col: Column name for the left y-axis (main metric).
        metric2_col: Column name for the right y-axis (Spearman).
        ylabel1: Label for the left y-axis.
        ylabel2: Label for the right y-axis.
        title: Main title for the plot.
        output_path: Path to save the generated plot.
    """
    
    plot_df = df[df['probe_type'] == probe_type].copy()
    if plot_df.empty:
        print(f"No data to plot for probe type '{probe_type}'. Skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(16, 9))

    # Plot Main Metric (UUAS or Root Acc) on the left axis
    sns.lineplot(data=plot_df, x='layer', y=metric1_col, hue='model', 
                 ax=ax1, marker='o', linestyle='-', palette=PALETTE,
                 markersize=8, linewidth=2.5)
    ax1.set_xlabel("Model Layer Index", fontsize=16)
    ax1.set_ylabel(ylabel1, fontsize=16)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    
    # Create the second axis
    ax2 = ax1.twinx()
    
    # Plot Spearman Metric on the right axis
    sns.lineplot(data=plot_df, x='layer', y=metric2_col, hue='model', 
                 ax=ax2, marker='s', linestyle='--', palette=PALETTE,
                 markersize=8, linewidth=2.5)
    ax2.set_ylabel(ylabel2, fontsize=16)
    ax2.tick_params(axis='y', labelsize=14)

    # Unify the legends
    h1, l1 = ax1.get_legend_handles_labels()
    # Manually adjust labels for clarity
    l1 = [f"{label} ({ylabel1})" for label in l1]
    h2, l2 = ax2.get_legend_handles_labels()
    l2 = [f"{label} ({ylabel2})" for label in l2]
    
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    fig.legend(h1 + h2, l1 + l2, loc='lower right', bbox_to_anchor=(0.9, 0.15), fontsize=14, title="Model & Metric")

    fig.suptitle(title, fontsize=22, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for title and legend
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def save_comprehensive_summary(df: pd.DataFrame, output_dir: Path):
    """
    Analyzes the full DataFrame to find peak performance for each model/probe
    and saves the results to a comprehensive text file.
    """
    summary_parts = []
    
    # --- Analyze Depth Probes ---
    depth_df = df[df['probe_type'] == 'depth']
    if not depth_df.empty:
        summary_parts.append("# Analysis Summary: Depth Probes (Llama-3.2-3B vs Instruct)")
        summary_parts.append("-" * 60)
        
        for model in ['Llama-3.2-3B', 'Llama-3.2-3B-Instruct']:
            model_df = depth_df[depth_df['model'] == model]
            if not model_df.empty:
                best_nspr = model_df.loc[model_df['dspr_nspr'].idxmax()]
                best_root_acc = model_df.loc[model_df['uuas_root_acc'].idxmax()]
                
                summary_parts.append(f"\n## {model}:")
                summary_parts.append(f"- Peak NSpr (Spearman): {best_nspr['dspr_nspr']:.4f} at Layer {int(best_nspr['layer'])}")
                summary_parts.append(f"- Peak Root Accuracy:   {best_root_acc['uuas_root_acc']:.4f} at Layer {int(best_root_acc['layer'])}")

    # --- Analyze Distance Probes ---
    dist_df = df[df['probe_type'] == 'dist']
    if not dist_df.empty:
        summary_parts.append("\n\n# Analysis Summary: Distance Probes (Llama-3.2-3B vs Instruct)")
        summary_parts.append("-" * 60)

        for model in ['Llama-3.2-3B', 'Llama-3.2-3B-Instruct']:
            model_df = dist_df[dist_df['model'] == model]
            if not model_df.empty:
                best_dspr = model_df.loc[model_df['dspr_nspr'].idxmax()]
                best_uuas = model_df.loc[model_df['uuas_root_acc'].idxmax()]
                
                summary_parts.append(f"\n## {model}:")
                summary_parts.append(f"- Peak DSpr (Spearman): {best_dspr['dspr_nspr']:.4f} at Layer {int(best_dspr['layer'])}")
                summary_parts.append(f"- Peak UUAS:            {best_uuas['uuas_root_acc']:.4f} at Layer {int(best_uuas['layer'])}")

    # --- Full Data Table ---
    summary_parts.append("\n\n# Full Aggregated Data")
    summary_parts.append("-" * 60)
    summary_parts.append(df.to_string())

    # --- Write to file ---
    summary_text = "\n".join(summary_parts)
    summary_path = output_dir / "comprehensive_analysis_summary.txt"
    summary_path.write_text(summary_text)
    print(f"Saved comprehensive text summary: {summary_path}")


def main(results_dir: str):
    """Main function to orchestrate the analysis."""
    project_root = Path(__file__).resolve().parents[2]
    results_base_path = project_root / results_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = project_root / "analysis" / f"llama_comparison_{timestamp}"
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
    plot_dual_axis_comparison(
        df=results_df,
        probe_type="dist",
        metric1_col="uuas_root_acc", ylabel1="UUAS",
        metric2_col="dspr_nspr", ylabel2="DSpr (Spearman)",
        title="Syntactic Distance Probing: Llama-3.2-3B vs. Instruct",
        output_path=output_dir / "comparison_distance_dual_axis.png"
    )
    
    plot_dual_axis_comparison(
        df=results_df,
        probe_type="depth",
        metric1_col="uuas_root_acc", ylabel1="Root Accuracy",
        metric2_col="dspr_nspr", ylabel2="NSpr (Spearman)",
        title="Syntactic Depth Probing: Llama-3.2-3B vs. Instruct",
        output_path=output_dir / "comparison_depth_dual_axis.png"
    )
    
    # --- ADD THIS CALL TO THE NEW SUMMARY FUNCTION ---
    print("\n--- Saving Summaries ---")
    save_comprehensive_summary(results_df, output_dir)
    
    print(f"\n--- Analysis complete! Find plots and summary in {output_dir} ---")


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
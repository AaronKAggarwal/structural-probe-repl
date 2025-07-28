# scripts/diagnostics/diag_14_analyze_llama_depth_sweep.py

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

# --- Add src to path if needed (though not strictly necessary for this script) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# --- Plotting Style Configuration ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "serif"
PALETTE = "plasma" # A visually appealing sequential color palette

def find_and_parse_results(multirun_path: Path) -> pd.DataFrame:
    """
    Finds all Llama 3.2 depth probe results and parses them into a DataFrame.
    """
    results_data = []
    # This glob pattern is specific to your folder structure
    glob_pattern = "train_probe_2025-07-24_16-15-55/new_models/ud_ewt/llama-3.2-3b/depth/L*/metrics_summary.json"
    
    result_files = sorted(list(multirun_path.glob(glob_pattern)))

    if not result_files:
        raise FileNotFoundError(f"No result files found in '{multirun_path}' matching the pattern '{glob_pattern}'. "
                                "Please check the multirun path.")

    print(f"Found {len(result_files)} result files to analyze.")

    for result_file in result_files:
        # Extract the layer number from the path using a regular expression
        match = re.search(r"/L(\d+)/", str(result_file))
        if not match:
            print(f"Warning: Could not extract layer number from path: {result_file}. Skipping.")
            continue
        
        layer = int(match.group(1))
        
        with open(result_file, 'r') as f:
            metrics = json.load(f)
        
        results_data.append({
            "layer": layer,
            "nspr": metrics.get("test_spearmanr_hm", np.nan),
            "root_acc": metrics.get("test_root_acc", np.nan),
            "dev_loss": metrics.get("best_model_metric_value_on_dev", np.nan),
        })

    if not results_data:
        raise ValueError("No valid results could be parsed from the found files.")

    df = pd.DataFrame(results_data)
    df = df.sort_values(by="layer").reset_index(drop=True)
    return df

def plot_metric_vs_layer(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: Path):
    """Creates and saves a high-quality plot for a single metric vs. layer."""
    plt.figure(figsize=(14, 8))
    
    # Find the best layer for annotation
    best_layer_idx = df[metric_col].idxmax()
    best_layer_data = df.loc[best_layer_idx]
    best_layer = int(best_layer_data['layer'])
    best_value = best_layer_data[metric_col]

    ax = sns.lineplot(data=df, x="layer", y=metric_col, marker="o", color=sns.color_palette(PALETTE)[2], markersize=8, linewidth=2.5)
    
    # Annotate the best point
    ax.annotate(
        f"Peak at Layer {best_layer}\nValue: {best_value:.4f}",
        xy=(best_layer, best_value),
        xytext=(best_layer + 1, best_value * 0.98),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8)
    )

    ax.set_title(title, fontsize=20, pad=20, weight='bold')
    ax.set_xlabel("Model Layer Index", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticks(np.arange(0, df['layer'].max() + 1, 2)) # Ticks every 2 layers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_combined_metrics(df: pd.DataFrame, output_path: Path):
    """Creates and saves a dual-axis plot comparing NSpr and Root Accuracy."""
    fig, ax1 = plt.subplots(figsize=(16, 9))

    color1 = sns.color_palette(PALETTE)[1]
    color2 = sns.color_palette(PALETTE)[3]

    # Plot NSpr on the primary y-axis
    sns.lineplot(data=df, x="layer", y="nspr", ax=ax1, color=color1, marker='o', label="NSpr (Spearman)")
    ax1.set_xlabel("Model Layer Index", fontsize=16)
    ax1.set_ylabel("NSpr (Spearman Correlation)", color=color1, fontsize=16)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_xticks(np.arange(0, df['layer'].max() + 1, 2))
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Create a secondary y-axis for Root Accuracy
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="layer", y="root_acc", ax=ax2, color=color2, marker='s', linestyle='--', label="Root Accuracy")
    ax2.set_ylabel("Root Accuracy", color=color2, fontsize=16)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)
    ax2.grid(False) # Avoid overlapping grids

    # Unify legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=14)
    ax2.get_legend().remove()

    fig.suptitle("Syntactic Depth Probing of Llama-3.2-3B on UD EWT", fontsize=22, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def save_summary_files(df: pd.DataFrame, best_nspr_layer: dict, best_root_acc_layer: dict, output_dir: Path):
    """Saves the results in CSV, Markdown, and a human-readable text summary."""
    # Save CSV
    csv_path = output_dir / "llama3.2-3b_depth_probe_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV summary: {csv_path}")

    # Save Markdown
    md_path = output_dir / "llama3.2-3b_depth_probe_summary.md"
    df.to_markdown(md_path, index=False, floatfmt=".4f")
    print(f"Saved Markdown summary: {md_path}")

    # Create and save text summary
    summary_text = f"""
    # Analysis Summary: Depth Probe on Llama-3.2-3B
    
    This analysis covers a full sweep of {len(df)} layers.
    
    ## Best Performing Layers
    
    **Peak NSpr (Spearman Correlation):**
    - Layer:      {int(best_nspr_layer['layer'])}
    - NSpr Score: {best_nspr_layer['nspr']:.4f}
    
    **Peak Root Accuracy:**
    - Layer:      {int(best_root_acc_layer['layer'])}
    - Root Acc:   {best_root_acc_layer['root_acc']:.4f}
    
    ## Full Data
    
    {df.to_string()}
    """
    summary_path = output_dir / "analysis_summary.txt"
    summary_path.write_text(dedent(summary_text))
    print(f"Saved text summary: {summary_path}")


def main(multirun_dir: str):
    """Main function to orchestrate the analysis."""
    multirun_path = PROJECT_ROOT / multirun_dir

    # Create a dedicated output directory for this analysis run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = PROJECT_ROOT / "analysis" / f"llama3.2-3b_depth_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis outputs will be saved to: {output_dir}")
    
    # 1. Find and parse all relevant results into a DataFrame
    try:
        results_df = find_and_parse_results(multirun_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Generate and save plots
    plot_metric_vs_layer(
        df=results_df,
        metric_col="nspr",
        title="NSpr (Spearman) vs. Layer for Llama-3.2-3B Depth Probe",
        ylabel="Spearman Correlation (NSpr)",
        output_path=output_dir / "nspr_vs_layer.png"
    )
    
    plot_metric_vs_layer(
        df=results_df,
        metric_col="root_acc",
        title="Root Accuracy vs. Layer for Llama-3.2-3B Depth Probe",
        ylabel="Root Accuracy",
        output_path=output_dir / "root_acc_vs_layer.png"
    )
    
    plot_combined_metrics(
        df=results_df,
        output_path=output_dir / "combined_metrics_vs_layer.png"
    )

    # 3. Find best layers and save summary files
    best_nspr_layer = results_df.loc[results_df['nspr'].idxmax()].to_dict()
    best_root_acc_layer = results_df.loc[results_df['root_acc'].idxmax()].to_dict()
    
    save_summary_files(results_df, best_nspr_layer, best_root_acc_layer, output_dir)
    
    print("\n--- Analysis complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Llama 3.2 depth probe sweep results.")
    parser.add_argument(
        "--multirun-dir",
        type=str,
        default="multirun",
        help="Path to the Hydra multirun directory, relative to project root."
    )
    args = parser.parse_args()
    
    main(args.multirun_dir)
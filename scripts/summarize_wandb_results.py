import pandas as pd
from pathlib import Path
import argparse
import sys
import numpy as np # For np.isclose and np.nan

def analyze_wandb_csv(csv_filepaths: list[Path]):
    """
    Analyzes one or more WandB export CSVs to find and display the best runs
    within distinct groups (Model, Dataset, Probe Type).

    Args:
        csv_filepaths: A list of paths to the input CSV files.
    """
    all_dfs = []
    for filepath in csv_filepaths:
        if not filepath.exists():
            print(f"Error: CSV file not found at {filepath}", file=sys.stderr)
            continue
        try:
            df_single = pd.read_csv(filepath)
            all_dfs.append(df_single)
            print(f"Loaded {filepath} with {len(df_single)} rows.")
        except Exception as e:
            print(f"Error reading CSV file {filepath}: {e}", file=sys.stderr)
            continue

    if not all_dfs:
        print("No valid CSV files were loaded. Exiting.", file=sys.stderr)
        return

    # Concatenate all DataFrames into one
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal runs loaded: {len(df)}")

    # --- Preprocessing ---
    # Ensure relevant columns exist
    # Using a comprehensive list of all potential columns used in display_order
    # to catch missing columns early
    all_possible_display_cols = [
        "Name", "Runtime", "embeddings.source_model_name", "dataset.name",
        "final_test/loss", "final_test/spearmanr_hm", "final_test/uuas",
        "final_test/root_acc", "experiment.probe.type", "experiment.probe.rank",
        "Tags", "Notes", "Group"
    ]
    
    missing_any_col = False
    for col in all_possible_display_cols:
        if col not in df.columns:
            # For columns not strictly 'required' by every run type (e.g. Tags, Notes, Group)
            # just add them with NaN to avoid KeyError downstream
            if col in ["Tags", "Notes", "Group"]:
                df[col] = np.nan 
            else:
                print(f"Error: Missing critical column '{col}' in combined CSV data.", file=sys.stderr)
                missing_any_col = True

    if missing_any_col:
        print("Cannot proceed due to missing critical columns. Please check input CSVs.", file=sys.stderr)
        return


    # Convert metric columns to numeric, coercing errors to NaN
    numeric_cols = [
        "final_test/loss",
        "final_test/spearmanr_hm",
        "final_test/uuas",
        "final_test/root_acc",
        "Runtime"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Deduplicate runs: Keep the best run if multiple entries for the same logical experiment exist
    # A logical run is uniquely identified by: Model, Dataset, Probe Type, and Layer Name
    deduplication_subset = ['embeddings.source_model_name', 'dataset.name', 'experiment.probe.type', 'Name']
    
    # Sort by spearman_hm (descending) and loss (ascending) to get the "best" duplicate first
    # This ensures that if multiple entries exist for the same run, the one with better performance is kept.
    df.sort_values(by=["final_test/spearmanr_hm", "final_test/loss"], ascending=[False, True], inplace=True)
    df.drop_duplicates(subset=deduplication_subset, keep='first', inplace=True)
    print(f"Total unique runs after deduplication: {len(df)}")

    # Drop rows where the primary metric for analysis is NaN (e.g., genuinely failed runs)
    df.dropna(subset=["final_test/spearmanr_hm"], inplace=True) 

    if df.empty:
        print("No valid data rows found after preprocessing and deduplication. Exiting.", file=sys.stderr)
        return

    # --- Helper to print a single run's details nicely ---
    def print_run_details(run_series: pd.Series, label_prefix: str):
        # Define a consistent order and friendly names for display
        display_order = [
            ("Name", "Layer"),
            ("embeddings.source_model_name", "Model"),
            ("dataset.name", "Dataset"),
            ("experiment.probe.type", "Probe Type"),
            ("experiment.probe.rank", "Probe Rank"),
            ("Runtime", "Runtime (s)"),
            ("final_test/loss", "Loss"),
            ("final_test/spearmanr_hm", "Spearman (NSpr/DSpr)"), # Will refine in print logic
            ("final_test/uuas", "UUAS"),
            ("final_test/root_acc", "Root Acc"),
            ("Tags", "Tags"),
            ("Notes", "Notes"),
            ("Group", "Group")
        ]

        print(f"  --- {label_prefix} ---")
        for col_name, display_name in display_order:
            value = run_series.get(col_name) # Use .get() for safety
            
            # Special handling for numerical metrics and empty/NaN values
            formatted_value = "N/A" # Default for empty/NaN
            if col_name in numeric_cols:
                if pd.notna(value):
                    if col_name == "Runtime":
                        formatted_value = f"{int(value)}"
                    else:
                        formatted_value = f"{value:.4f}"
            else: # For string/object columns
                formatted_value = str(value).strip()
                if not formatted_value or formatted_value == "nan": 
                    formatted_value = "N/A"
            
            # Refine Spearman label based on probe type
            if col_name == "final_test/spearmanr_hm":
                if run_series["experiment.probe.type"] == "depth":
                    display_name = "Spearman (NSpr)"
                elif run_series["experiment.probe.type"] == "distance":
                    display_name = "Spearman (DSpr)"
            
            # Explicitly mark non-applicable metrics
            if (col_name == "final_test/uuas" and run_series["experiment.probe.type"] == "depth") or \
               (col_name == "final_test/root_acc" and run_series["experiment.probe.type"] == "distance"):
                formatted_value = "N/A (Not Applicable)"
            
            print(f"    {display_name:<20}: {formatted_value}")
        print("  ---------------------------------")


    # --- Group the DataFrame and process each group ---
    # The grouping keys define a unique "experiment type" (e.g., Llama-3.2-3B Depth on EWT)
    grouped_by_experiment_type = df.groupby(['embeddings.source_model_name', 'dataset.name', 'experiment.probe.type'])

    for (model_name, dataset_name, probe_type), group_df in grouped_by_experiment_type:
        
        # Determine probe-specific metric columns and labels
        secondary_metric_col = None
        secondary_metric_display_name = ""
        if probe_type == "depth":
            secondary_metric_col = "final_test/root_acc"
            secondary_metric_display_name = "Root Acc"
        elif probe_type == "distance":
            secondary_metric_col = "final_test/uuas"
            secondary_metric_display_name = "UUAS"
        else:
            print(f"Warning: Unknown probe type '{probe_type}' for group {model_name}, {dataset_name}. Skipping group.", file=sys.stderr)
            continue

        print(f"\n{'='*80}")
        print(f"======== ANALYZING: {model_name.upper()} - {probe_type.upper()} PROBES on {dataset_name.upper()} ========")
        print(f"{'='*80}")

        if group_df.empty:
            print(f"--- No valid runs found for this group. ---")
            continue

        # --- Find best by Spearman ---
        # Note: df.idxmax() will return the first max if there are ties
        best_spearman_idx = group_df["final_test/spearmanr_hm"].idxmax()
        best_spearman_run = group_df.loc[best_spearman_idx]
        print(f"\n--- Best Run by Spearman Correlation ---")
        print_run_details(best_spearman_run, "Highest Spearman")
        
        # --- Find best by Secondary Metric (UUAS or Root Acc) ---
        # Filter for runs that actually have a non-NaN value for the secondary metric
        group_df_secondary_valid = group_df.dropna(subset=[secondary_metric_col])
        
        if not group_df_secondary_valid.empty:
            best_secondary_idx = group_df_secondary_valid[secondary_metric_col].idxmax()
            best_secondary_run = group_df_secondary_valid.loc[best_secondary_idx]

            # Check if the run with the highest secondary metric is the same as the highest Spearman run
            # Use index comparison as it's robust after deduplication
            if best_spearman_idx == best_secondary_idx:
                print(f"  (The run with the highest {secondary_metric_display_name} is the same as the highest Spearman run for this group.)")
            else:
                print(f"\n--- Best Run by {secondary_metric_display_name} ---")
                print_run_details(best_secondary_run, f"Highest {secondary_metric_display_name}")
        else:
            print(f"\n--- No valid runs with {secondary_metric_display_name} found for this group. ---")

    print(f"\n{'='*80}")
    print(f"======== ANALYSIS COMPLETE ========")
    print(f"{'='*80}")

# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze WandB CSV export for best structural probe runs."
    )
    # Using nargs='+' to accept one or more file paths
    parser.add_argument(
        "csv_filepaths",
        type=Path,
        nargs='+', # This is the key change: '+' means one or more arguments
        help="Paths to one or more WandB CSV export files (e.g., wandb_export_*.csv).",
    )
    args = parser.parse_args()

    analyze_wandb_csv(args.csv_filepaths)
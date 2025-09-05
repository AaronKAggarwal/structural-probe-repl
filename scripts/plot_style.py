import matplotlib as mpl
import seaborn as sns

def apply_style(context: str = "talk") -> None:
    sns.set_theme(style="whitegrid", context=context)
    mpl.rcParams.update({
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "axes.titleweight": "semibold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

def get_palette() -> dict:
    return {
        "dist": "#1f77b4",
        "depth": "#ff7f0e",
        "raw": "#4c78a8",
        "lenmatch": "#72b7b2",
        "arcmatch": "#e39c37",
        "anchor": "#a0a0a0",
    }

def smart_ylim(data_values, domain_min=None, domain_max=None, padding=0.05):
    """
    Auto-scale y-axis based on data range with intelligent padding.
    
    Args:
        data_values: List/array of y-values to base scaling on
        domain_min: Optional floor (e.g., 0.5 for UUAS)
        domain_max: Optional ceiling (e.g., 1.0 for UUAS)
        padding: Fraction of range to add as padding
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if not data_values or len(data_values) == 0:
        if domain_min is not None and domain_max is not None:
            plt.ylim(domain_min, domain_max)
        return
    
    data_min, data_max = np.min(data_values), np.max(data_values)
    data_range = data_max - data_min
    
    # Apply domain constraints
    y_min = max(data_min - padding * data_range, domain_min) if domain_min is not None else data_min - padding * data_range
    y_max = min(data_max + padding * data_range, domain_max) if domain_max is not None else data_max + padding * data_range
    
    # Ensure minimum useful range
    if y_max - y_min < 0.1:
        center = (y_max + y_min) / 2
        y_min, y_max = center - 0.05, center + 0.05
        if domain_min is not None:
            y_min = max(y_min, domain_min)
        if domain_max is not None:
            y_max = min(y_max, domain_max)
    
    plt.ylim(y_min, y_max)

def adjust_text_overlap(annotations, method="repel"):
    """
    Adjust text annotations to avoid overlap.
    
    Args:
        annotations: List of matplotlib annotation objects
        method: "repel" for simple vertical spreading
    """
    import matplotlib.pyplot as plt
    
    if method == "repel" and len(annotations) > 1:
        # Simple vertical repulsion - space out y-coordinates
        y_positions = []
        for ann in annotations:
            x, y = ann.xy
            y_positions.append(y)
        
        # Sort by y position and spread if too close
        sorted_indices = sorted(range(len(y_positions)), key=lambda i: y_positions[i])
        min_gap = 0.02  # minimum gap in data coordinates
        
        for i in range(1, len(sorted_indices)):
            prev_idx = sorted_indices[i-1] 
            curr_idx = sorted_indices[i]
            if y_positions[curr_idx] - y_positions[prev_idx] < min_gap:
                y_positions[curr_idx] = y_positions[prev_idx] + min_gap
        
        # Update annotation positions
        for i, ann in enumerate(annotations):
            ann.xy = (ann.xy[0], y_positions[i])

def savefig(fig, path: str) -> None:
    path = str(path)
    fig.savefig(path)
    if path.endswith(".png"):
        fig.savefig(path.replace(".png", ".svg"))

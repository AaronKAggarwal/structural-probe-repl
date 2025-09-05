# Plotting guidelines (pretty and consistent)

This document standardizes figure style, palettes, layout, and file naming so all paper-ready plots are consistent and accessible.

## Global style
- Backend: matplotlib + seaborn.
- Theme: seaborn "whitegrid"; light y-grid only; white background.
- Fonts: DejaVu Sans (default) or system sans; math text default.
- Font sizes (approx): title 16, axes 14, tick 12, legend 12.
- Line widths: lines 2.0; grid 0.6 (dotted); axes spine 0.8.
- Figure sizes (inches):
  - Single column: 3.6 × 2.8 (tight physical size for 300 dpi)
  - 1.5 column: 5.0 × 3.4
  - Double column: 7.2 × 4.2
- DPI: 300 (PNG) for paper, and SVG for vector.
- Margins: use tight_layout(); avoid clipping labels.
- Export: both .png and .svg; white facecolor; include metadata (commit/date) when practical.

## Color and encodings
Use colorblind-safe palettes and consistent encodings:

- Probes:
  - Distance (dist): #1f77b4 (blue)
  - Depth (depth): #ff7f0e (orange)
- Conditions within a probe:
  - Raw: solid line/bar, filled marker o
  - Length-matched (Stage 7): dashed line, square s
  - Arc-length-matched (Stage 8): dash-dot line, diamond D
  - Length-anchored (2≤L≤20): solid thinner line, triangle ^
- Uncertainty: 95% CIs as error bars (capsize=2) or bands (alpha≈0.18)
- Low-coverage (length-anchored 2≤L≤20): dashed outline or footnote mark (see below)

## Axes and scales
- Metrics (UUAS/RootAcc) bounded in [0, 1]; clamp axes to [0, 1] unless justified.
- Use consistent y-limits across comparable plots.
- Label axes with metric and unit: e.g., "UUAS (content-only)", "Root Accuracy (content-only)".
- Sort categorical x (languages) by the plotted value (descending) where appropriate.

## File structure and names
- Write figures under outputs/figures/ with subfolders:
  - main/ (paper figures), robustness/, layer_curves/, mediation/.
- Naming: lowercase, snake-case, minimal but descriptive.
  - Examples: L7_bars_dist_raw_vs_lenmatch.png, layer_curves_UD_German-GSD_depth.svg,
    L7_anchors_dist_bars_with_ci.svg.

## Stage-specific notes
- Stage 7 (length-matched):
  - Overlay raw bars with matched dots + CI at L7; annotate delta in tooltip or table.
  - Rank-stability plots: use Spearman ρ; scatter raw vs matched with y=x reference.
- Stage 8 (arc-length-matched):
  - Same encodings as Stage 7; optional truncation histograms per bin index.
- Length-anchored (2≤L≤20):
  - For length range 2≤L≤20, flag languages with coverage_all < 0.5 using dashed outlines around markers/bars or legend footnote.
  - Show coverage labels optionally under bars (small grey font).

## Accessibility and readability
- Use colorblind-safe palette; do not rely on color alone—vary marker shape/linestyle.
- Keep legends outside plotting area when possible; small framealpha=0.9.
- Ensure text contrast; avoid thin fonts; maintain ≥8 pt when scaled.
- Add alt-text in the paper repository (not required here) describing the takeaway.

## Data sources (by stage)
- Raw tables: outputs/analysis/analysis_table_per_layer.csv (L7 slice in outputs/analysis/analysis_table_L7.csv).
- Stage 7: outputs/analysis/matched_eval_length_per_layer.csv (+ merged outputs/analysis/analysis_table_per_layer_with_lenmatch.csv).
- Stage 8: outputs/analysis/matched_eval_arclen_per_layer.csv.
- Length-anchored: outputs/analysis/uuas_at_len_anchor_per_layer.csv, outputs/analysis/uuas_at_arclen_anchor_per_layer.csv, low-coverage lists in outputs/analysis/final/05_anchors/.

## Reusable style helper
Use this snippet or scripts/plot_style.py to apply consistent style:

```python
# scripts usage: from scripts.plot_style import apply_style, get_palette, savefig
import matplotlib as mpl, seaborn as sns

def apply_style(context="talk"):
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

def get_palette():
    return {
        "dist": "#1f77b4",
        "depth": "#ff7f0e",
        "raw": "#4c78a8",
        "lenmatch": "#72b7b2",
        "arcmatch": "#e39c37",
        "anchor": "#a0a0a0",
    }

def savefig(fig, path):
    path = str(path)
    fig.savefig(path)
    if path.endswith(".png"):
        fig.savefig(path.replace(".png", ".svg"))
```

## Example: L7 bars with overlays
Outline for a per-language L7 bar chart with Stage 7 and 9 overlays:

```python
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plot_style import apply_style, get_palette, savefig

apply_style()
pal = get_palette()

base = pd.read_csv("outputs/analysis/analysis_table_L7.csv")
lenmatch = pd.read_csv("outputs/analysis/analysis_table_L7_with_lenmatch.csv")
anchors = pd.read_csv("outputs/analysis/uuas_at_len_anchor_per_layer.csv")

# Example for distance probe only
df = base[base.probe=="dist"][ ["language_slug","uuas"] ].merge(
    lenmatch[["language_slug","uuas_length_matched","uuas_length_matched_ci_low","uuas_length_matched_ci_high"]],
    on="language_slug", how="left"
)
anc = anchors[(anchors.probe=="dist") & (anchors.layer=="L7")][["language_slug","point_estimate","ci_low","ci_high"]]
df = df.merge(anc.rename(columns={"point_estimate":"uuas_at_L20","ci_low":"uuas_L20_ci_low","ci_high":"uuas_L20_ci_high"}), on="language_slug", how="left")

# Sort by raw UUAS
df = df.sort_values("uuas", ascending=False)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.bar(df.language_slug, df.uuas, color=pal["dist"], alpha=0.9, label="Raw UUAS")
# Overlays: length-matched dots + CI
ax.errorbar(df.language_slug, df.uuas_length_matched,
            yerr=[df.uuas_length_matched - df.uuas_length_matched_ci_low,
                  df.uuas_length_matched_ci_high - df.uuas_length_matched],
            fmt='s', color=pal["lenmatch"], ms=5, capsize=2, label="Length-matched")
# Length-anchored (2≤L≤20)
ax.errorbar(df.language_slug, df.uuas_at_L20,
            yerr=[df.uuas_at_L20 - df.uuas_L20_ci_low,
                  df.uuas_L20_ci_high - df.uuas_at_L20],
            fmt='^', color=pal["anchor"], ms=5, capsize=2, label="2≤L≤20")

ax.set_ylim(0, 1)
ax.set_ylabel("UUAS (content-only)")
ax.set_xlabel("")
ax.set_title("L7 Distance: Raw vs Length-Matched vs 2≤L≤20")
ax.tick_params(axis='x', rotation=75)
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
savefig(fig, "outputs/figures/main/L7_bars_dist_raw_len_anchor.png")
```

## QA checklist (before saving)
- Axes consistent and bounded [0,1]; labels complete and units clear.
- CI rendering correct; no clipping after tight_layout.
- Colors/markers/linestyles match the encodings above.
- Language labels readable (rotate or shorten as needed).
- Legends outside data where possible and not blocking marks.
- Filenames and paths follow the convention.

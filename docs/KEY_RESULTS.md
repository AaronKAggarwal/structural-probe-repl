# Key Research Findings & Observations

Last updated: August 21, 2025

This document serves as a high-level summary of the key quantitative results and qualitative hypotheses generated during the project. It is intended as a quick reference and a guide for the evolving research narrative.

---

## Foundational Baselines on UD Treebanks

Before probing modern LLMs, we established performance baselines on the **Universal Dependencies** datasets using our validated pipeline. These scores serve as the primary point of comparison for new models.

### ELMo (Modern Baseline Training on UD EWT)

*(Results from previous phase, kept for context)*

| Probe Type | Metric     | Layer 0 | Layer 1 (Peak) | Layer 2 |
| :--------- | :--------- | :------ | :------------- | :------ |
| **Depth**  | NSpr       | 0.731   | **0.827**      | 0.789   |
|            | Root Acc   | 0.582   | **0.825**      | 0.793   |
| **Distance**| DSpr       | 0.315   | **0.712**      | 0.678   |
|            | UUAS       | 0.320   | **0.724**      | 0.658   |

### `bert-base-multilingual-cased` (Modern Baseline Training)

**On English (UD EWT):**

| Probe Type | Metric | Peak Score (Layer) |
| :--------- | :----- | :------------------- |
| **Depth**  | NSpr   | **0.8623** @ L7      |
|            | Root Acc| **0.8759** @ L7      |
| **Distance**| DSpr   | Still training     |
|            | UUAS   | Still training      |

**On Hindi (UD HDTB):**

| Probe Type | Metric | Peak Score (Layer) |
| :--------- | :----- | :------------------- |
| **Depth**  | NSpr   | **0.8665** @ L8      |
|            | Root Acc| **0.9186** @ L7      |
| **Distance**| DSpr   | **0.8040** @ L7      |
|            | UUAS   | **0.8124** @ L7      |

*   **Observation:** The multilingual BERT model shows strong, consistent encoding of syntax in its middle layers (L7-L8) for both English and Hindi, aligning with previous literature on BERT-style architectures. The performance on Hindi is notably high.

---

## Stage 7 — Length-matched evaluation (all 23 languages)

Goal: remove sentence-length as a confound by matching every language to a pooled target distribution of content-token lengths via stratified bootstrap (B=1000), then recompute metrics (UUAS for distance; RootAcc for depth) with CIs.

What we ran
- Pooled target (content lengths): median of per-language bin proportions, renormalised.
- Layers: L5–L10; Probes: distance, depth; Split: test (paper); dev (diagnostics).
- Outputs (test): `outputs/analysis/matched_eval_length_per_layer.csv` (23×2×6=276 rows) with CIs and QC.

QC summary (test)
- **Retention** (T′/T): weighted ≈ 0.845; median ≈ 0.810; overall drop ≈ 15.5%.
- **JS distance to target**: median ≈ 0.0147 (small); duplication mostly 1.0.
- **Truncation**: mostly 3–4 bins per row, concentrated in mid-length bins (11–25).

Headline findings (paper-facing)
- **Cross-language gaps persist after matching.** Raw→matched rank stability is extremely high across layers:
  - Distance: ρ ≈ 0.95–0.97 (L7: ρ=0.967)
  - Depth: ρ ≈ 0.98–0.99 (L7: ρ=0.992)
- **Magnitude of length-driven effects (L7 deltas, matched − raw):**
  - Distance (UUAS): mean +0.027, median +0.0247, all positive (23/23). Top corrections: Arabic +0.077, Urdu +0.043, Spanish +0.043, Greek +0.033, Japanese +0.032.
  - Depth (RootAcc): mean −0.0036, median −0.0011; 7 positive / 16 negative; 13 languages |Δ|<0.005. Largest decreases: Turkish −0.051, English −0.017, Korean −0.017; largest increases: Urdu +0.017, Chinese +0.016.
- **Layer effects are robust.** High rank ρ per layer indicates mid-layer peak (incl. L7) is not an artifact of length.
- **Metric sensitivity.** Depth CIs are wider (median ≈ 0.064 vs distance ≈ 0.020), but broad conclusions hold.

Interpretation
- Matching shows that **length is not the main driver** of cross-language probe differences. After correcting for length composition, **rankings and profiles persist**, especially for depth. Distance had mild length inflation in some languages (now corrected).

Artifacts for figures/modeling
- Merged tables with matched metrics and deltas:
  - `outputs/analysis/analysis_table_per_layer_with_lenmatch.csv`
  - `..._L7_with_lenmatch.csv`, `..._L7_adequate_with_lenmatch.csv`
- Findings:
  - Rank stability: `outputs/analysis/stage7_findings/rankcorr_raw_vs_matched_by_layer_probe.csv`
  - L7 deltas per language: `outputs/analysis/stage7_findings/delta_L7_by_language.csv`
  - Layer curves (raw vs matched): `outputs/analysis/stage7_findings/figure_data/layer_curves_{dist,depth}.csv`
- QC: `outputs/analysis/checks/stage7_qc_summary.json`, `outputs/analysis/checks/stage7_merge_qc.json`

Methodological note
- Depth detailed metrics for four languages (CZ/EN/DE/ES) were regenerated into `runs/regen_check` to add alignment helpers (`kept_sentence_indices`, optional full-length arrays). Original runs are preserved; Stage 7 uses the regenerated outputs for these languages.

---

## Stage 8 — Arc-length–matched evaluation (all 23 languages)

Goal: remove structural-difficulty confounding by matching per-sentence mean content-only arc length to a pooled cross-language quantile target, then recompute UUAS/RootAcc with 95% CIs.

QC summary (test)
- Retention (T′/T): see `outputs/analysis/checks/stage8_qc_summary.json` (similar to Stage 7; median ≈ 0.8 expected)
- JS-to-target small; duplication ~1.0; 276 rows produced.

Artifacts
- Matched CSV: `outputs/analysis/matched_eval_arclen_per_layer.csv` (+ split specific)
- Diagnostics: `outputs/analysis/matched_eval/arclen/test/<LANG>.json`
- Target spec: `outputs/analysis/matched_targets/arclen_bins.json`

Interpretation
- Together with Stage 7 (length-matched), arc-length matching shows cross-language differences persist after controlling for sentence length and tree structural difficulty. Residual gaps can thus be analyzed against morphology, tokenization fragmentation, and exposure in later stages.

---

## Stage 9 — Anchored evaluation (UUAS@L≤20 and UUAS@A≤3)

- **Goal**: Provide simple, reproducible snapshots on uniformly “easy-ish” subsets without reweighting by anchoring on two global thresholds close to pooled p80: L≤20 (content tokens) and A≤3 (mean content arc length).

- **What we compute**: Per language × probe × layer (L5–L10): mean per-sentence score over eligible sentences with bootstrap 95% CIs; coverage_all = T_eligible/N_total; coverage_base = T_eligible/T_base.

- **QC summary (test)**
  - **Coverage_all (median)**: L≤20 = 0.722 (p05 0.381, p95 0.930; 48 low-coverage rows); A≤3 = 0.844 (p05 0.432, p95 0.955; 24 low-coverage rows). Small-sample count = 0 for both.
  - **Median CI width**: L≤20 → dist 0.0236, depth 0.0545; A≤3 → dist 0.0212, depth 0.0561.
  - **Rank stability vs raw (Spearman, by layer)**: L≤20 → dist 0.922, depth 0.950; A≤3 → dist 0.952, depth 0.984 (medians).

- **Headline findings**
  - **Distance improves under anchors.** L7 deltas (anchor − raw): L≤20 mean +0.0499 (median +0.0477); A≤3 mean +0.0427 (median +0.0370). Languages with structurally harder raw distributions (e.g., Arabic, Turkish, Urdu, Korean, Spanish) gain the most.
  - **Depth is largely stable.** L7 deltas: L≤20 mean +0.0024 (median −0.0006); A≤3 mean +0.0035 (median +0.0034). Changes are modest and symmetric.
  - **Rankings persist.** High Spearman ρ across layers confirms that cross-language ordering is robust to these descriptive anchors.

- **Visualization note (limitations)**
  - For L≤20, explicitly flag low-coverage languages (coverage_all<0.5) using dashed outlines or footnotes. See `low_coverage_L_anchor_by_language.csv` for the list.

- **Artifacts**
  - Anchored CSVs: `outputs/analysis/uuas_at_len_anchor_per_layer.csv`, `outputs/analysis/uuas_at_arclen_anchor_per_layer.csv`
  - Findings: `outputs/analysis/stage9_findings/coverage_ci_summaries.json`, `rankcorr_raw_vs_anchors_by_layer_probe.csv`, `delta_L7_by_language_anchors.csv`, `low_coverage_L_anchor_rows.csv`, `low_coverage_L_anchor_by_language.csv`
  - Master additions for paper join: `outputs/analysis/master_additions_stage9_test.csv` (adds `uuas_at_L20`, `uuas_at_A3` with CIs and coverage; depth counterparts included)
  - Provenance: `outputs/analysis/anchors_used.json` (split-aware p80 justification)
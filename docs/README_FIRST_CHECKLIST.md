# READ ME FIRST: Reproducibility & Analysis Protocol Checklist

This document captures all critical decisions, environment details, and analysis protocols that must be locked before proceeding to ensure reproducibility and prevent post-hoc biases.

---

## 🔒 **Reproduction & Provenance (Must Lock)**

### **Model & Tokenizer Artifacts**
- [ ] **HuggingFace model commit hash**: `bert-base-multilingual-cased` @ `_______`
- [ ] **Tokenizer commit hash**: WordPiece tokenizer @ `_______`
- [ ] **Vocabulary checksum**: `vocab.txt` SHA256 = `_______`
- [ ] **UD release version**: Record exact version per treebank (e.g., UD_2.10, UD_2.11)

### **Random Seeds & Determinism**
- [ ] **Global random seeds**: NumPy, PyTorch, Python `random` module
- [ ] **Determinism flags**: `torch.backends.cudnn.deterministic = True`
- [ ] **Non-deterministic operations**: Document any unavoidable sources (e.g., GPU reductions)
- [ ] **Bootstrap seeds**: Stage 5 morphology bootstraps, future resampling

### **Environment Freeze**
- [ ] **Poetry lock**: `poetry.lock` committed and up-to-date
- [ ] **Hardware**: GPU model, CUDA version, cuDNN version
- [ ] **OS**: Operating system version
- [ ] **Python**: Exact Python version
- [ ] **Environment file**: Create `docs/ENVIRONMENT.md` with full details

### **Parameter Locks**
- [ ] **Band selection**: Δ, τ, min_T, bootstrap B, equivalence rule
- [ ] **UUAS@L20**: minimum length threshold, robust regressor choice
- [ ] **Evaluation histograms**: Save target histograms for length/arc-length matching (not just matched scores)

---

## 📊 **Data Hygiene & Scope (Frozen Decisions)**

### **Language Selection**
- [ ] **One-treebank-per-language policy**: Document rationale + alternates considered
  - Example: English EWT vs GUM → chose EWT because _______
- [ ] **Language metadata**: Create `data/language_metadata.csv` with:
  - `language_slug, language, iso639, family, subfamily, script`
  - Families: WALS-ish groupings
  - Scripts: Latin/Cyrillic/Arabic/Han/Hangul/Hebrew/etc.

### **Coverage & Units Standards**
- [ ] **FEATS coverage policy**: Adequate ≥ 0.10; Sparse ∈ [0.01, 0.10); Absent < 0.01
- [ ] **Units are fractions**: 0.0–1.0, not percentages
- [ ] **Wikipedia exposure**: Compressed multistream bytes; if sharded, sum parts
- [ ] **Date provenance**: Keep chosen date per wiki (frozen in `ANALYSIS_INVARIANTS.md`)

### **Content-Only Consistency**
- [ ] **UPOS filtering**: `UPOS ∉ {"PUNCT", "SYM"}` throughout
- [ ] **Gold tree operations**: Drop punctuation nodes, collapse punctuation heads
- [ ] **Metrics alignment**: Both token masks and tree operations use same policy

---

## 🎯 **Analysis Protocol (Pre-Registered)**

### **Primary Outcomes**
- [ ] **Distance probe**: L7 UUAS@L20 (content-only)
- [ ] **Depth probe**: L7 RootAcc (or Depth Spearman if specified)
- [ ] **Content-only**: All metrics computed on non-punctuation tokens

### **Covariates (Exact Transforms)**
- [ ] **Log sizes**: `log_n_train_sent`, `log_n_test_sent`
- [ ] **Height variants**: `height_residual`, `height_normalized_log`, `height_normalized_linear`
- [ ] **Fragmentation**: `fragmentation_ratio_content_mean` (primary)
- [ ] **Arc length**: `mean_arc_length_test` (content-only)
- [ ] **Pretraining exposure**: `pretrain_exposure_log2mb`
- [ ] **Morphology**: `complexity_pc1` (PC1 from z-scored FEATS features)

### **Interactions (Pre-Registered vs Exploratory)**
- [ ] **Pre-registered interactions**:
  - `complexity_pc1 × fragmentation_ratio_content_mean`
  - `complexity_pc1 × feats_coverage_band` (if using all 23 languages)
- [ ] **Exploratory interactions**: Document any additions as post-hoc

### **Random Effects Structure**
- [ ] **Primary**: Random intercepts by `family`
- [ ] **Secondary**: Add `script` if not collinear with family
- [ ] **Fallback plan**: If convergence fails, document fallback to fixed effects
- [ ] **Sample size constraint**: n≈20 languages → prefer bootstrap CIs over complex random slopes

---

## 🔬 **Robustness & Sensitivity (Planned Queue)**

### **Layer Locality**
- [ ] **L6/L8 replication**: Repeat mixed-effects; report coefficient stability vs L7
- [ ] **Band sensitivity**: Compare L7-only vs joint L5-L10 vs distance-band L6-L8

### **Methodological Robustness**
- [ ] **Tokenization aggregation**: Mean vs first-piece spot-check
- [ ] **XLM-R replication**: SentencePiece subset for 4-6 representative languages
- [ ] **Leave-one-out**: Leave-one-family-out and leave-one-language-out analyses
- [ ] **Budget-matched evaluation**: Equal test tokens per language

### **Negative Controls**
- [ ] **Random baseline**: Random head assignment for distance probe
- [ ] **Permuted labels**: Shuffle gold trees to ensure probes aren't overfitting noise
- [ ] **Structureless signals**: Verify probes require actual syntactic structure

---

## 📋 **QC Artifacts (Must Print to Files)**

### **Unit Sanity Checks**
- [ ] **Metric ranges**: `uuas, root_acc, spearman_* ∈ [0,1]`
- [ ] **Fragmentation**: `fragmentation_ratio_content_mean ≥ 1.0`
- [ ] **Exposure**: `pretrain_exposure_log2mb ∈ [~7, ~14]`
- [ ] **Coverage**: `feats_coverage_train ∈ [0,1]` (fractions)

### **Matched Evaluation Diagnostics**
- [ ] **Retention rates**: Per-language T'/T for length and arc-length matching
- [ ] **Bin hit-rates**: Distribution of sentences across histogram bins
- [ ] **Target histograms**: Save and version the target distributions used

### **UUAS@L20 Diagnostics**
- [ ] **Length distribution**: Number of unique lengths per language
- [ ] **Regression diagnostics**: Slope/intercept distributions, R² values
- [ ] **Outlier flags**: Languages with unusual length-performance relationships

### **Statistical Assumptions**
- [ ] **Multicollinearity**: VIF table for all fixed effects
- [ ] **Residual checks**: Normality, homoscedasticity, influential points
- [ ] **Bootstrap CIs**: Record confidence intervals for key effects
- [ ] **Coverage scatter**: FEATS coverage vs PC1 to highlight annotation extremes (Ja/Ko/Vi)

---

## 📊 **Reporting Hygiene (Standardization)**

### **Figure Standards**
- [ ] **Style sheet**: Fonts, DPI, color palette locked for consistency (see `docs/PLOTTING_GUIDELINES.md` and `scripts/plot_style.py`)
- [ ] **Axis ranges**: Consistent scales across similar plots for comparability
- [ ] **File formats**: Vector formats (PDF/SVG) for publication, PNG for quick viewing

### **Table Standards**
- [ ] **Schema documentation**: Column glossary with units
- [ ] **Units reminder**: Entropy (bits), exposure (log₂ MB), arc length (tokens), height (levels)
- [ ] **Significant digits**: Consistent precision across similar metrics

### **Interpretation Warnings**
- [ ] **PC1 caveat**: "Morphological complexity = UD-annotated FEATS richness"
- [ ] **Adequate-only**: "Primary causal interpretation uses Adequate-coverage languages"
- [ ] **Annotation artifacts**: Ja/Ko/Vi low coverage reflects UD annotation sparsity

---

## ⚖️ **Compliance & Sharing**

### **Licensing**
- [ ] **UD data**: CC-BY-SA compliance documented
- [ ] **Wikipedia**: Note using only dump sizes (no redistribution)
- [ ] **Model**: HuggingFace license terms acknowledged

### **Release Planning**
- [ ] **Public artifacts**: Which CSVs/figures will be shared
- [ ] **Internal artifacts**: Which raw per-sentence files (Parquet) stay internal
- [ ] **Reproducibility package**: Scripts, configs, and documentation for replication

---

## ⚠️ **Common Traps to Avoid**

### **Unit Consistency**
- [ ] **Percentage drift**: Always use fractions (0-1) for coverage metrics
- [ ] **Language identifiers**: Single mapping table for slug ↔ ISO codes
- [ ] **Content-only policy**: Same UPOS filtering for trees AND metrics

### **Statistical Modeling**
- [ ] **Small sample size**: n≈20 languages → bootstrap over complex random effects
- [ ] **Post-hoc fishing**: Mark exploratory analyses clearly
- [ ] **Overinterpretation**: PC1 reflects annotation, not inherent morphology

### **Data Pipeline**
- [ ] **File path consistency**: Standardize on `data/UD_<Lang-Treebank>/` throughout
- [ ] **JSON serialization**: Cast numpy types to Python types before saving
- [ ] **Version control**: Lock all intermediate artifacts with checksums

---

## 🎯 **Stage 7+ Preparation Checklist**

- [ ] **Analysis tables ready**: `outputs/analysis/analysis_table_L7.csv` and `outputs/analysis/analysis_table_L7_adequate.csv`
- [ ] **Matched evaluation**: Length-matched and arc-length–matched scores appended/merged
- [ ] **Anchored evaluation**: L≤20 and A≤3 anchored CSVs present (test, 276 rows each)
- [ ] **UUAS@L20 (regression)**: (Optional/Planned) Compute via Theil–Sen if needed for robustness
- [ ] **Mixed-effects setup**: Family/script metadata joined
- [ ] **Figure pipeline**: Automated plot generation with locked aesthetics
- [ ] **Mediation analysis**: Complexity → fragmentation → performance pathway

### **Stage 7/8/9 Artifacts Presence (Must Exist)**
- [ ] Stage 7 canonical CSV: `outputs/analysis/matched_eval_length_per_layer.csv` (rows = 276)
- [ ] Stage 8 canonical CSV: `outputs/analysis/matched_eval_arclen_per_layer.csv` (rows = 276)
- [ ] Stage 9 anchored CSVs: `outputs/analysis/uuas_at_len_anchor_per_layer.csv`, `uuas_at_arclen_anchor_per_layer.csv` (rows = 276 each)
- [ ] Stage 9 additions: `outputs/analysis/master_additions_stage9_test.csv` (contains `uuas_at_L20/A3` and depth counterparts + coverage)
- [ ] Findings tables:
  - `outputs/analysis/stage7_findings/` (rankcorr, L7 deltas, layer curves)
  - `outputs/analysis/stage8_findings/` (rankcorr, L7 deltas, truncation)
  - `outputs/analysis/stage9_findings/` (coverage/CI summaries, rankcorr, L7 deltas, low-coverage list)
- [ ] QC JSONs:
  - `outputs/analysis/checks/stage7_qc_summary.json`, `stage7_merge_qc.json`
  - `outputs/analysis/checks/stage8_qc_summary.json`
  - `outputs/analysis/checks/stage9_qc_len_test.json`, `stage9_qc_arclen_test.json`
- [ ] Targets/anchors specs: `outputs/analysis/matched_targets/length_bins.json`, `matched_targets/arclen_bins.json`, `anchors_used.json`
- [ ] Outputs map present: `outputs/analysis/README.md`
- [ ] Plotting guidelines present: `docs/PLOTTING_GUIDELINES.md`; helpers: `scripts/plot_style.py`
- [ ] Low-coverage L≤20 list present: `outputs/analysis/stage9_findings/low_coverage_L_anchor_by_language.csv` (use dashed outlines/footnote in plots)

### **Outputs Hygiene**
- [ ] `outputs/analysis/` is clean (canonical analysis only); run artifacts archived under `outputs/archive/`
- [ ] Dated run dumps moved to `outputs/archive/dates/`; orphan anchors moved to `outputs/archive/anchors/`

---

## 📝 **Documentation Status**

- [ ] `docs/ANALYSIS_INVARIANTS.md` - Locked decisions and provenance
- [ ] `docs/MODELING_PLAN.md` - Primary/secondary analysis plans  
- [ ] `docs/PIPELINE_STAGES_1_6.md` - Complete record of Stages 1-6
- [ ] `docs/ENVIRONMENT.md` - Full computational environment details
- [ ] `data/language_metadata.csv` - Language family/script mapping
- [ ] This checklist completed and reviewed before Stage 7

---

**🚀 Ready to proceed when all checkboxes are complete and all artifacts are version-controlled!**

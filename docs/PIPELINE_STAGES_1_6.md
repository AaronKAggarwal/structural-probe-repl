## Structural Probing Pipeline: Stages 1–6 (Complete Record)

This document is a methodical, end-to-end record of everything completed in Stages 1–6 of the project, including inputs, outputs, scripts, policies, quirks, validations, and acceptance criteria. It is intended to serve as a single place to remind ourselves of what was done, where each artifact lives, and any important nuances.

### Scope and Global Invariants (applies to Stages 1–6)

- **Model**: `bert-base-multilingual-cased` (mBERT)
- **Languages**: 23 UD treebanks
  - `ar, eu, bg, zh, cs, en, fi, fr, de, el, he, hi, hu, id, ja, ko, fa, pl, ru, es, tr, ur, vi`
  - UD slugs: e.g., `UD_English-EWT`, `UD_Turkish-IMST`, etc.
- **Layers**: `L5`–`L10` (focus band for syntax based on literature)
- **Headline layer**: `L7` (no averaging across layers)
- **Subword-to-word alignment**: mean pooling throughout
- **Content-only policy**: UPOS-based filtering; content tokens are those with `UPOS ∉ {"PUNCT", "SYM"}`
- **Gold tree operations**: When computing content-only metrics (e.g., arc lengths), drop punctuation nodes and collapse punctuation heads upwards to the nearest content head, then remap heads to content indices; arcs to/through punctuation are ignored in arc-length.
- **UD data**: CoNLL-U format files under `data/UD_<Lang-Treebank>/{train,dev,test}.conllu`
- **Provenance**: Decisions and locked choices captured in `docs/ANALYSIS_INVARIANTS.md` (e.g., Stage 4B, Stage 5).
- **Random Seeds**: Stage 5 bootstraps and any future random sampling record seeds for reproducibility.

---

## Stage 1 — Build Master Per-Layer Results

Consolidate per-language × probe × layer metrics into a single immutable ledger.

- **Script**: `scripts/build_master_per_layer.py`
- **Inputs**:
  - Per-run results for distance (`dist`) and depth (`depth`) probes across layers `L5`–`L10`.
  - Layer-invariant sentence counts duplicated per layer.
  - Primary metrics: `uuas`, `root_acc`, `spearman_hm`, `spearman_content`, `loss`, `n_dev_sent`, `n_test_sent`.
- **Output**:
  - `outputs/analysis/master_results_per_layer.csv`
    - Shape: 23 languages × 2 probes × 6 layers = 276 rows
    - Key columns: `language_slug, probe, layer, loss, spearman_hm, spearman_content, uuas, root_acc, n_dev_sent, n_test_sent`
- **Quirks & Fixes**:
  - `n_test_sent` was initially empty due to looking only for `test_detailed_metrics_final.json` in a legacy path. Fixed to prefer `test_detailed_metrics_final.json` and fallback to `test_detailed_metrics.json` within each per-language run directory.
  - Avoided confusion by moving legacy `outputs/baselines_auto/bert-base-multilinual-case` to backups.
  - Layer values are strings like `"L7"` (not integers). Filtering by `layer == "L7"` is required.
- **Acceptance**:
  - 276 unique `(language_slug, probe, layer)` rows with populated metrics.

---

## Stage 2 — Fragmentation (Tokenizer Burden)

Compute subwords-per-word fragmentation, aligned with content-only policy.

- **Scripts**: `scripts/compute_fragmentation_metrics.py`, `scripts/merge_fragmentation_metrics.py`
- **Inputs**:
  - HDF5 files with sentence-level `word_ids` (subword → word index mapping)
  - UD UPOS tags from CoNLL-U to define content mask
- **Method**:
  - Primary: `fragmentation_ratio_content_mean` = (#subwords on content words) / (#content words)
  - Diagnostic: `fragmentation_ratio_overall_mean` = (#subwords overall) / (#words overall)
  - Content-only mask uses UPOS filtering consistent with probes (exclude `PUNCT`, `SYM`).
- **Outputs**:
  - `outputs/analysis/fragmentation_stats.csv` (language-level aggregates for test)
  - Columns integrated later into analysis table: `fragmentation_ratio_content_mean`, `fragmentation_ratio_overall_mean`
- **Robustness Fixes**:
  - Iterate HDF5 under `f['word_ids'].keys()` rather than `f.keys()` to avoid missing/misaligned sentences.
  - Guard against mismatches between word count implied by `word_ids` and UPOS tag count; skip inconsistent sentences.
  - JSON serialization fix: cast numpy `int64` to Python `int/float` when emitting summaries.
- **Acceptance**:
  - Plausible cross-language differences; content-only ≥ overall typically; no values < 1.0 (flagged as suspicious during QC because each word must have ≥1 subword, so <1.0 implies a counting/mapping bug).

---

## Stage 3 — Tree Shape Covariates (Length, Arc Length, Height)

Compute sentence-level structural covariates from gold trees, then aggregate to language-level.

- **Scripts**: `scripts/compute_sentence_covariates.py`, `scripts/merge_sentence_covariates.py`, `scripts/add_height_covariates.py`
- **Inputs**:
  - CoNLL-U gold trees under `data/UD_*/*.{train,dev,test}.conllu`
  - UPOS tags for content filtering
- **Sentence-level stats (canonical, content-only)**:
  - `content_len` = number of content tokens per sentence
  - `mean_arc_len` = mean |rank_I(h) − rank_I(d)| over non-root dependents on content-only collapsed tree
  - `num_content_arcs_used` = count of arcs contributing to `mean_arc_len`
  - `tree_height` = max root-to-leaf depth on content-only tree
  - `orig_len_incl_punct`, `content_ratio = content_len / orig_len`
- **Aggregation (test split)**:
  - `mean_content_len_test`, `median_content_len_test`
  - `mean_arc_length_test`
  - `mean_tree_height_test`
  - Residualized and normalized heights to mitigate collinearity with length:
    - `height_residual` (height ~ length residual)
    - `height_normalized_log` (height / log₂(n_content+1))
    - `height_normalized_linear` (height / n_content)
- **Outputs**:
  - Canonical per-sentence stats stored under `outputs/analysis/sentence_stats/` (for reuse)
  - Aggregates: `outputs/analysis/tree_shape_stats.csv`
- **Quirks & Fixes**:
  - Mass balance check initially failed due to filtering layer `7` (int) vs `"L7"` (str); corrected to `"L7"` and confirmed near-1 correlation between `n_test_tokens_content` and `mean_content_len_test × n_test_sent`.
  - CoNLL-U path pattern corrected from `*-ud-train.conllu` to `train.conllu`.
  - Height strongly correlated with length (r≈0.951); treat height as size-driven; use residuals/normalized variants for modeling.
- **Acceptance**:
  - Distributions match expectations; height vs arc length r≈0.51; per-language patterns interpretable.

---

## Stage 4A — UD Dataset Statistics

Compute UD corpus sizes and type inventories per language.

- **Script**: `scripts/compute_ud_stats_derived.py`
- **Method**:
  - Count sentences and content tokens for train/test
  - Inventory sizes for UPOS and base DEPREL (strip subtypes)
  - Log-transformed counts for modeling
- **Output**:
  - `outputs/analysis/ud_stats_derived.csv`
  - Columns: `n_train_sent, n_test_sent, n_train_tokens_content, n_test_tokens_content, n_deprel_types, n_upos_types, log_n_train_sent, log_n_test_sent`
- **QC**:
  - Root presence confirmed; mass balance cross-checks pass with Stage 3 aggregates.

---

## Stage 4B — Pretraining Exposure (WikiSize)

Compute log₂ of compressed MB for Wikipedia `pages-articles-multistream.xml.bz2` dumps near Oct–Nov 2018.

- **Scripts**: `scripts/review_wiki_targets.py`, `scripts/compute_wiki_exposure.py`
- **Method**:
  - Try `https://dumps.wikimedia.org/{code}wiki/20181001/dumpstatus.json` (and nearby months)
  - If not available, fallback to Internet Archive `archive.org/metadata/{code}wiki-YYYYMMDD`
  - Use the monolithic multistream `.bz2` file size (if the multistream file is sharded, sum `*-pages-articles-multistream*.bz2` sizes; otherwise use the monolith)
  - Compute `size_mb = bytes / 2^20`, `wiki_size_log2_mb = log₂(size_mb)`
- **Output**:
  - `outputs/analysis/pretrain_exposure.csv`
  - Columns: `language_slug, wiki_code, chosen_date, size_bytes, size_mb, wiki_size_log2_mb, source`
- **Quirks & Findings**:
  - Wikimedia dumps retain only recent snapshots; Internet Archive used for 2018.
  - Found 23/23 languages; dates cluster at 2018-10-01, with a few at 2018-11-01.
  - Values in expected range (~7–14 log₂ MB).
  - Documented in `docs/ANALYSIS_INVARIANTS.md` with per-language provenance.

---

## Stage 5 — Morphological Complexity (MorphPC1)

Compute a PCA-based morphological complexity index and FEATS coverage.

- **Scripts**: `scripts/compute_morphological_complexity.py`, `scripts/investigate_feats_coverage.py`, `scripts/add_feats_coverage.py`, `scripts/finalize_morph_complexity.py`
- **Method**:
  - Train split, content-only tokens (UPOS-based)
  - Features per language:
    - `feats_per_token_mean` (mean FEATS attributes per token)
    - `feats_bundle_entropy_bits` (Shannon entropy of canonicalized FEATS bundles, log₂)
    - `feats_bundles_per_10k` (bundle type density per 10k tokens)
  - PCA on z-scored features; PC1 sign-aligned to be positively correlated with `feats_per_token_mean`.
  - FEATS coverage: % of content tokens with FEATS ≠ `_` (train/dev/test) and coverage bands:
    - `Adequate` ≥ 0.10, `Sparse` ∈ [0.01, 0.10), `Absent` < 0.01 (fractions, not %)
  - Robustness: recompute PC1 on Adequate-only languages; record stability.
- **Outputs**:
  - `outputs/analysis/morph_complexity.csv`
    - Columns include: `language_slug, complexity_pc1, feats_per_token_mean, feats_bundle_entropy_bits, feats_bundles_per_10k, feats_coverage_train, feats_coverage_dev, feats_coverage_test, feats_coverage_band, complexity_pc1_adequate_only`
  - `outputs/analysis/morphology_ready.csv` (slim export for downstream)
  - QC/provenance:
    - `outputs/analysis/morph_complexity_qc.json`
    - `outputs/analysis/morph_complexity_provenance.json` (casts numpy bool/float to Python types)
    - Figures and CSVs under `outputs/analysis/figure_data/` (face-validity plots)
- **Quirks & Fixes**:
  - FEATS coverage was originally saved as percentages (0–100). Stage 6 standardizes to fractions (0–1) in analysis tables.
  - Absent coverage languages: `UD_Japanese-GSD (~0.6%)`, `UD_Korean-GSD (~0.8%)`, `UD_Vietnamese-VTB (~0.0%)` — reflect annotation sparsity in those UD treebanks, not true lack of morphology.
  - Morphological complexity = UD-annotated FEATS richness; for languages with sparse/absent FEATS (e.g., Ja/Ko/Vi), PC1 reflects annotation policy rather than inherent morphology.
  - `complexity_pc1_adequate_only` was computed but not persisted; fixed by saving the updated DataFrame and updating `morphology_ready.csv`.
  - Stability correlation `r ≈ 0.993` between all-23 PC1 and adequate-only PC1.
  - PC1 is computed on all 23 (annotation-expressed complexity). Primary modeling uses L7 Adequate languages; all-23 analyses include feats_coverage_train as a covariate.
- **Acceptance**:
  - Morphology features and PC1 computed for all languages; coverage bands assigned; provenance frozen.

---

## Stage 6 — Assemble Analysis Tables (Per-layer + L7 Slices)

Create a single tidy analysis table joining master probe metrics with all covariates, plus L7 slices.

- **Scripts**: `scripts/stage6_assemble_analysis_tables.py`, `scripts/stage6_validate_existing_tables.py`
- **Join Plan** (left joins on `language_slug`):
  1. Start from `outputs/analysis/master_results_per_layer.csv` (276 rows)
  2. Add derived columns:
     - `is_headline_layer = (layer == "L7")`
     - `layer_index ∈ {5..10}` parsed from `layer`
  3. Join fragmentation (test): `fragmentation_ratio_content_mean`, `fragmentation_ratio_overall_mean`
  4. Join tree shape (test): `mean_content_len_test, median_content_len_test, mean_arc_length_test, mean_tree_height_test`, optional height variants (`height_residual`, `height_normalized_log`, `height_normalized_linear`)
  5. Join UD stats: `n_train_sent, n_test_sent, n_train_tokens_content, n_test_tokens_content, log_n_train_sent, log_n_test_sent, n_deprel_types, n_upos_types`
  6. Join pretraining exposure: `wiki_code, wiki_size_log2_mb` (also retained `chosen_date, source`); in the final table an alias `pretrain_exposure_log2mb` is provided.
  7. Join morphology: `complexity_pc1`, raw features, coverage fields (`feats_coverage_train`, `feats_coverage_band`).
- **Outputs**:
  - `outputs/analysis/analysis_table_per_layer.csv` (276 × 44)
  - `outputs/analysis/analysis_table_L7.csv` (46 × 44)
  - `outputs/analysis/analysis_table_L7_adequate.csv` (40 × 44) — primary modeling subset (20 languages × 2 probes)
  - QC & schema:
    - `outputs/analysis/checks/analysis_table_qc.json`
    - `outputs/analysis/schema/analysis_table_schema.json`
- **QC & Guards**:
  - Uniqueness check on `(language_slug, probe, layer)` → 276 unique
  - Row counts: 276 total; 46 for L7; L7-Adequate has 40 rows (20 languages × 2 probes)
  - Range asserts: `uuas, root_acc, spearman_* ∈ [0,1]`; `fragmentation_ratio_content_mean ≥ 1.0`; `mean_arc_length_test ≥ 1`; `pretrain_exposure_log2mb ∈ [~7,~14]`; `feats_coverage_train ∈ [0,1]` (fractions—Stage 6 converts from % if needed)
  - Missingness: no missing critical covariates
  - Sanity correlations on L7 slice: strong positive between `complexity_pc1` and `feats_per_token_mean`; positive between `feats_coverage_train` and `complexity_pc1`
- **Quirks & Fixes**:
  - Initial QC flagged `feats_coverage_train` out of `[0,1]` because Stage 5 stored coverage as percentages. Stage 6 standardizes to fractions by dividing by 100 when assembling/validating.
  - Exposure column labeled consistently as `pretrain_exposure_log2mb` (alias retained for `wiki_size_log2_mb`).
- **Acceptance**:
  - Stage 6 finalized with all validation checks passing; analysis tables ready for Stages 7–13.

---

## Key Artifacts (Quick Index)

- Stage 1: `outputs/analysis/master_results_per_layer.csv`
- Stage 2: `outputs/analysis/fragmentation_stats.csv`
- Stage 3: `outputs/analysis/tree_shape_stats.csv`, `outputs/analysis/sentence_stats/`
- Stage 4A: `outputs/analysis/ud_stats_derived.csv`
- Stage 4B: `outputs/analysis/pretrain_exposure.csv`
- Stage 5: `outputs/analysis/morph_complexity.csv`, `outputs/analysis/morph_complexity_qc.json`, `outputs/analysis/morph_complexity_provenance.json`, `outputs/analysis/morphology_ready.csv`, `outputs/analysis/figure_data/`
- Stage 6: `outputs/analysis/analysis_table_per_layer.csv`, `outputs/analysis/analysis_table_L7.csv`, `outputs/analysis/analysis_table_L7_adequate.csv`, `outputs/analysis/checks/analysis_table_qc.json`, `outputs/analysis/schema/analysis_table_schema.json`

---

## Known Decisions and Their Rationale

- No layer averaging; headline layer is `L7` to align with literature and maintain interpretability.
- Content-only policy across all covariates to match probe evaluation.
- Pretraining exposure uses compressed dump size log₂(MB) of `pages-articles-multistream.xml.bz2` (2018), consistent with prior work.
- Morphological complexity is annotation-expressed richness (UD-FEATS), computed on train split, content-only tokens.
- Primary modeling uses Adequate-coverage languages; all-language analyses include coverage terms.

---

## Units Cheat-Sheet

- **Entropy**: bits (log₂)
- **Exposure**: log₂(MB) of compressed Wikipedia dumps
- **Fragmentation**: subwords per word ratio (≥1.0)
- **Arc length**: content token distance (≥1.0)
- **Tree height**: dependency levels from root (≥1.0)
- **Coverage**: fractions 0.0–1.0 (Stage 6 converts from % if needed)

---

## Optional Future Additions

- **Language family/script mapping**: Consider adding `data/language_metadata.csv` with columns `language_slug, language, iso639, family, subfamily, script` for Stage 12 random effects.

---

## Reproduction Notes (Minimal)

- Run per script using Poetry, e.g.: `poetry run python3 scripts/<script_name>.py`
- Stage 6 validator may enrich existing analysis tables with derived columns and will emit QC + schema artifacts.
- See `docs/ANALYSIS_INVARIANTS.md` and `docs/MODELING_PLAN.md` for locked constructs and modeling stance.



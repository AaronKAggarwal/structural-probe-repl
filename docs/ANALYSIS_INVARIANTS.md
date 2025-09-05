Analysis invariants and provenance

Scope lock timestamp (UTC): 2025-08-19T10:02:37Z
Repo branch/commit: feat/preflight-checks @ cd33433

Invariants

- Model: bert-base-multilingual-cased (mBERT), local weights
- Languages: 23 UD treebanks (see outputs/baselines_auto/UD_*)
- Layers considered: L5–L10 only
- Subword→word alignment: mean pooling throughout
- Probes: distance (UUAS) and depth (RootAcc), both content-only evaluation
- Headline layer for all primary results: L7 (pre-registered based on literature and an independent dev-only equivalence analysis)
- No layer averaging in primary results: no band means, no AUC; per-layer analyses only

Provenance and confirmations

- Completed runs are present under outputs/baselines_auto/UD_*/bert-base-multilingual-cased/{dist,depth}/L{5..10}/runs/* with dev_detailed_metrics.json and metrics_summary.json
- Per-layer analysis table at outputs/analysis/analysis_table_per_layer.csv (276 rows: 23×2×6)
- Dev-only band selection artifacts exist (outputs/training_logs/syntax_band_selection*.json) but are not used for primary aggregation; L7 is pre-registered as headline

Depth detailed metrics alignment (Stage 7 readiness)

- For four languages (UD_Czech-PDTC, UD_English-EWT, UD_German-GSD, UD_Spanish-AnCora), depth detailed metrics were regenerated into separate run directories `runs/regen_check` per layer (L5–L10) to include alignment metadata:
  - `kept_sentence_indices` (indices into 0..N-1 for compact arrays)
  - `root_acc_per_sentence_full` (optional, length N with NaNs for filtered sentences)
- Original runs remain intact for provenance; Stage 7 consumption prefers the `regen_check` outputs for these languages.
- Test Stage 7 now has full coverage (276 rows); dev Stage 7 is diagnostics-only and may skip some depth rows where dev compact arrays remain under-specified.

Evaluation policies

- Content-only tokens: PUNCT/SYM excluded for Spearman and UUAS/RootAcc where applicable
- Test split used for paper numbers; dev split used for diagnostics only
- Randomness: bootstrap and selection seeds recorded in output JSONs where applicable

## Stage 4B: Pretraining Exposure (WikiSize)

Script: `scripts/compute_wiki_exposure.py` @ cd33433c718c7621a11d71c3f43904e35c3dae98

Method: Exposure = log₂(compressed MB) of pages-articles-multistream.xml.bz2 from 2018-10/11; source = Internet Archive.

Coverage: 23/23 languages found via archive.org

Per-language data:
- UD_Arabic-PADT: ar, 20181101, 801447180 bytes, 764.32 MB, WikiSize=9.578
- UD_Basque-BDT: eu, 20181001, 180725819 bytes, 172.35 MB, WikiSize=7.429  
- UD_Bulgarian-BTB: bg, 20181001, 335431255 bytes, 319.93 MB, WikiSize=8.321
- UD_Chinese-GSD: zh, 20181001, 1764674582 bytes, 1682.94 MB, WikiSize=10.717
- UD_Czech-PDTC: cs, 20181001, 771119067 bytes, 735.41 MB, WikiSize=9.522
- UD_English-EWT: en, 20181101, 16512687556 bytes, 15747.69 MB, WikiSize=13.943
- UD_Finnish-TDT: fi, 20181001, 670699920 bytes, 639.56 MB, WikiSize=9.321
- UD_French-GSD: fr, 20181001, 4308342684 bytes, 4108.76 MB, WikiSize=12.004
- UD_German-GSD: de, 20181001, 5190092667 bytes, 4949.67 MB, WikiSize=12.273
- UD_Greek-GDT: el, 20181001, 326205511 bytes, 311.11 MB, WikiSize=8.281
- UD_Hebrew-HTB: he, 20181001, 578094175 bytes, 551.29 MB, WikiSize=9.107
- UD_Hindi-HDTB: hi, 20181001, 138305361 bytes, 131.89 MB, WikiSize=7.043
- UD_Hungarian-Szeged: hu, 20181001, 825542552 bytes, 787.32 MB, WikiSize=9.621
- UD_Indonesian-GSD: id, 20181001, 527054766 bytes, 502.64 MB, WikiSize=8.973
- UD_Japanese-GSD: ja, 20181001, 2864356232 bytes, 2731.71 MB, WikiSize=11.416
- UD_Korean-GSD: ko, 20181001, 626417189 bytes, 597.40 MB, WikiSize=9.223
- UD_Persian-Seraji: fa, 20181001, 701926428 bytes, 669.36 MB, WikiSize=9.387
- UD_Polish-PDB: pl, 20181001, 1845836269 bytes, 1760.32 MB, WikiSize=10.782
- UD_Russian-SynTagRus: ru, 20181001, 3643734198 bytes, 3474.93 MB, WikiSize=11.763
- UD_Spanish-AnCora: es, 20181001, 3044849039 bytes, 2903.76 MB, WikiSize=11.504
- UD_Turkish-IMST: tr, 20181001, 509119659 bytes, 485.53 MB, WikiSize=8.923
- UD_Urdu-UDTB: ur, 20181001, 128810583 bytes, 122.84 MB, WikiSize=6.941
- UD_Vietnamese-VTB: vi, 20181001, 626165158 bytes, 597.16 MB, WikiSize=9.222

## Stage 5: Morphological Complexity (MorphPC1)

Script: `scripts/compute_morphological_complexity.py` @ cd33433c718c7621a11d71c3f43904e35c3dae98

Construct: MorphPC1 (UD-FEATS) = annotation-expressed morphological richness from content-only train tokens

Method: PC1 from PCA on z-scored features: feats_per_token_mean, bundle_entropy_bits, bundles_per_10k

FEATS coverage bands (train split): Adequate ≥10% (20 langs), Sparse 1-10% (0 langs), Absent <1% (3 langs)

Absent languages: Japanese (0.6%), Korean (0.8%), Vietnamese (0.0%) - reflects UD annotation sparsity, not linguistic morphology

Primary modeling: Adequate-coverage languages only (n=20). Secondary: All 23 with coverage interaction term.

MorphPC1 (UD-FEATS) = first PC of z-scored {FEATS/token, FEATS-bundle entropy, bundle types per 10k}, computed on train, content-only; PC1 sign aligned with FEATS/token. FEATS coverage (% non-empty FEATS on content tokens) recorded and banded; primary causal reads use Adequate-coverage languages; all-23 analyses include coverage terms. Stability correlation (PC1 all vs adequate-only): r=0.993.

## Stage 6: Assemble per-layer analysis tables and L7 slice

- Inputs: master per-layer metrics and covariates from prior stages (fragmentation, tree-shape, UD size, exposure, morphology).
- Output tables:
  - `outputs/analysis/analysis_table_per_layer.csv` (23×2×6 rows)
  - `outputs/analysis/analysis_table_L7.csv` (L7-only slice)
- Policy:
  - Join keys: (language_slug, probe, layer). Do not overwrite raw ledger; emit joined tables separately.
  - Include language family labels (for random effects in modeling).
  - L7 is the pre-registered headline layer; no band means/AUC in primary results.

## Stage 7: Length-matched evaluation policy

- Goal: control sentence-length confounding via stratified bootstrap matching to a pooled cross-language target over I-length bins.
- Bins (content-only length): default edges [2–5, 6–10, 11–15, 16–25, 26–40, 41+]. Record any bin collapses if sparsity.
- Target distribution: elementwise median of per-language bin proportions (test split), renormalized; saved to `outputs/analysis/matched_targets/length_bins.json`.
- Eligibility and alignment: use per-sentence content lengths; align metrics via IDs → kept indices → mask fallback (Stage 7/8 logic).
- Bootstrap: B=1000 with fixed seed per (language, probe, layer). Point = bootstrap mean; 95% CIs from percentiles.
- Retention and flags: report T_raw, T′, retention_ratio; `retention_low_flag` (<0.8), `small_sample_flag`; record truncated bins and JS divergence to target.
- Outputs:
  - `outputs/analysis/matched_eval_length_per_layer.csv` (test canonical)
  - Diagnostics per language: `outputs/analysis/matched_eval/length/<LANG>.json`
  - Merge step adds matched columns/deltas to `analysis_table_per_layer_with_lenmatch.csv`.

## Stage 9: Anchored evaluation policy

- Anchors: L≤20 (content tokens) and A≤3 (mean content arc length), chosen near pooled p80 thresholds for interpretability (see `outputs/analysis/anchors_used.json`).
- Eligibility: L≤20 requires content_len∈[2,20]; A≤3 requires ≥1 content arc and mean_arc_len≤3. Per-sentence metrics aligned via IDs/indices/full-length (Stage 7/8 logic).
- Estimates: point estimate = sample mean over eligible sentences; uncertainty via bootstrap 95% CI (B=1000).
- Coverage reported and used for limitations: coverage_all=T_eligible/N_total and coverage_base=T_eligible/T_base; low-coverage flagged at coverage_all<0.5 (visualized via dashed outlines/footnotes in figures).
- No reweighting or matching in Stage 9; descriptive snapshots complementary to Stages 7–8.

Changes require explicit update here and regeneration of downstream analysis tables





# Morphological Complexity Modeling Plan

## Pre-registered Analysis Strategy

**Primary Analysis (Conservative)**:
- **Scope**: Adequate FEATS coverage languages only (n=20)
- **Rationale**: Scientific validity requires reliable morphological measurement
- **Languages**: All except Japanese (0.6%), Korean (0.8%), Vietnamese (0.0%)
- **Models**: L7 mixed-effects with MorphPC1 as fixed effect

**Secondary Analysis (Robustness)**:
- **Scope**: All 23 languages
- **Model**: Include main effects + MorphPC1 × feats_coverage_band interaction
- **Purpose**: Demonstrate slope exists where FEATS is used, collapses where absent
- **Expected**: Significant positive effect for Adequate band, near-zero for Absent band

## FEATS Coverage Bands (Pre-registered)

**Adequate (≥10% coverage)**: 20 languages
- Arabic (83.3%), Basque (76.1%), Bulgarian (74.7%), Chinese (13.0%), Czech (89.5%)
- English (77.7%), Finnish (85.0%), French (69.4%), German (77.4%), Greek (75.6%)
- Hebrew (66.2%), Hindi (91.4%), Hungarian (84.9%), Indonesian (46.3%), Persian (71.3%)
- Polish (91.0%), Russian (78.8%), Spanish (63.8%), Turkish (74.2%), Urdu (88.2%)

**Sparse (1-10% coverage)**: 0 languages

**Absent (<1% coverage)**: 3 languages  
- Japanese (0.6%), Korean (0.8%), Vietnamese (0.0%)

## Visual Strategy

**Figure Requirements**:
- Color/shape by feats_coverage_band in all L7 plots
- Adequate: filled markers, Absent: hollow markers
- Include coverage % in appendix table
- Annotate key outliers with language names

## Robustness Extensions (Future)

**Reliability Weighting**: 
- Weight observations by coverage (w = coverage, rescaled to mean 1)
- Re-fit L7 models with weighted regression

**Sensitivity Suite**:
- Drop Absent/Sparse subset (covered by primary)
- Missing-data imputation for MorphPC1 using auxiliaries
- UD version sensitivity check for Absent languages

**Triangulation**:
- Combine MorphPC1 with WALS morphological indicators
- UniMorph coverage as additional validation
- Cross-tokenizer replication (XLM-R/SentencePiece)

## Expected Outcomes

**Primary Hypothesis**: Positive association between MorphPC1 and syntactic probe performance in Adequate-coverage languages

**Secondary Validation**: MorphPC1 × coverage interaction confirms measurement validity - effect present only where FEATS annotation exists

**Interpretation**: MorphPC1 captures annotation-expressed morphological richness, not true linguistic complexity for under-annotated languages

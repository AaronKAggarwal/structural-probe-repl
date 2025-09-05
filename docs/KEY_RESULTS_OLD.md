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

## Probing `Llama-3.2-3B` Base vs. Instruct Models on UD EWT

This phase directly investigates the impact of instruction tuning on syntactic representation. Both the base pretrained model and its instruction-tuned variant were probed across all layers.

### Summary of Peak Performance

| Metric          | Llama-3.2-3B (Base) | Llama-3.2-3B-Instruct | Key Takeaway                                           |
| :-------------- | :------------------ | :-------------------- | :----------------------------------------------------- |
| **NSpr (Depth)**| **0.8336** @ L12     | **0.8260** @ L14       | Performance is preserved, peak shifts **deeper (+2)**. |
| **Root Acc**    | **0.7552** @ L14     | **0.7552** @ L17       | Performance is preserved, peak shifts **deeper (+3)**. |
| **DSpr (Dist)** | **0.7318** @ L12     | **0.7295** @ L14       | Performance is preserved, peak shifts **deeper (+2)**. |
| **UUAS (Dist)** | **0.6767** @ L16     | **0.6769** @ L17       | Performance is preserved, peak shifts **deeper (+1)**. |

### Core Hypotheses & Interpretation

1.  **Instruction Tuning Reorganises, Not Degrades, Syntactic Knowledge:** The primary finding is that the peak performance of the Instruct model is virtually identical to the Base model across all metrics (e.g., Root Accuracy is exactly 0.7552 for both; UUAS is ~0.677 for both). This strongly suggests that the alignment process fine-tunes the model's behavior without sacrificing its core, pretrained geometric encoding of grammar.

2.  **Syntactic Information is Pushed to Deeper Layers After Instruction Tuning:** While performance is maintained, the location of the most linearly accessible syntactic information consistently shifts to deeper layers in the Instruct model. For every metric, the peak layer is 1-3 layers deeper than in the Base model. This suggests that while lower layers retain their fundamental linguistic representations, the alignment process may repurpose middle-to-upper layers for instruction-following and task-oriented behaviors, causing the most abstract syntactic representations to consolidate slightly later in the network.

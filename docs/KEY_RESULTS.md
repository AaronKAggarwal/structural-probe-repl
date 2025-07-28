# Key Research Findings & Observations

Last updated: July 28, 2025

This document serves as a high-level summary of the key quantitative results and qualitative hypotheses generated during the project. It is intended as a quick reference and a guide for the evolving research narrative.

---

## Part 1: Foundational Baselines on UD EWT

Before probing modern LLMs, we established performance baselines on the **Universal Dependencies English Web Treebank (UD EWT)** dataset using our validated pipeline. These scores serve as the primary point of comparison for all new models.

### ELMo (Modern Baseline Training)

| Probe Type | Metric     | Layer 0 | Layer 1 (Peak) | Layer 2 |
| :--------- | :--------- | :------ | :------------- | :------ |
| **Depth**  | NSpr       | 0.731   | **0.827**      | 0.789   |
|            | Root Acc   | 0.582   | **0.825**      | 0.793   |
| **Distance**| DSpr       | 0.315   | **0.712**      | 0.678   |
|            | UUAS       | 0.320   | **0.724**      | 0.658   |

*   **Observation:** Syntax is most linearly prominent in Layer 1 (the first BiLSTM layer), aligning with prior literature that suggests ELMo's lower layers are more syntactic. Layer 0 (CharCNN) has weak syntactic signal, and Layer 2 begins to mix in more semantic information.

### BERT (`bert-base-cased`, Modern Baseline Training)

*(Note: Populate with your BERT-base UD EWT baseline results)*

| Probe Type | Metric | Peak Score (Layer 7) |
| :--------- | :----- | :------------------- |
| **Depth**  | NSpr   | `[Enter L7 NSpr]`    |
|            | Root Acc| `[Enter L7 Root Acc]`|
| **Distance**| DSpr   | `[Enter L7 DSpr]`    |
|            | UUAS   | `[Enter L7 UUAS]`    |

---

## Part 2: Probing `Llama-3.2-3B` Base vs. Instruct Models

This phase directly investigates the impact of instruction tuning on syntactic representation. Both the base pretrained model and its instruction-tuned variant were probed across all 28 layers.

### Summary of Peak Performance

| Metric          | Llama-3.2-3B (Base) | Llama-3.2-3B-Instruct | Key Takeaway                                           |
| :-------------- | :------------------ | :-------------------- | :----------------------------------------------------- |
| **NSpr (Depth)**| **0.834** @ L12     | **0.826** @ L14       | Performance is preserved, peak shifts **deeper (+2)**. |
| **Root Acc**    | **0.755** @ L14     | **0.755** @ L17       | Performance is preserved, peak shifts **deeper (+3)**. |
| **DSpr (Dist)** | **0.732** @ L12     | **0.730** @ L14       | Performance is preserved, peak shifts **deeper (+2)**. |
| **UUAS (Dist)** | **0.677** @ L16     | **0.677** @ L17       | Performance is preserved, peak shifts **deeper (+1)**. |

### Core Hypotheses & Interpretation

1.  **Instruction Tuning Reorganizes, It Does Not Degrade, Syntactic Knowledge:** The primary finding is that the peak performance of the Instruct model is virtually identical to the Base model. This strongly suggests that the alignment process fine-tunes the model's behavior without sacrificing its core, pretrained understanding of grammar.
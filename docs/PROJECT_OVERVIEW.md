**`docs/PROJECT_OVERVIEW.md`**
# Project Overview: Replicating and Extending Structural Probes

Last updated: 2025-05-26

## 1. Introduction: Understanding Linguistic Knowledge in Neural Networks

The advent of large, pre-trained deep learning models for Natural Language Processing (NLP) has led to significant performance gains across a wide array of tasks. Models like ELMo, BERT, and their successors (e.g., Llama, Mistral) learn rich contextual representations of language. A key area of research in NLP interpretability, often termed "BERTology" or more broadly "modelology," seeks to understand *what* linguistic knowledge these models acquire during their extensive pre-training and *how* this knowledge is encoded in their internal vector representations (embeddings).

**Probing tasks** are a common methodology used to investigate these questions. A "probe" is typically a simple, supervised model trained to predict specific linguistic properties (e.g., part-of-speech tags, syntactic dependencies, semantic roles) from the internal representations of a larger, pre-trained language model. If a simple probe can successfully predict a property, it suggests that the information related to that property is readily accessible or "encoded" in the pre-trained model's embeddings.

## 2. Hewitt & Manning (2019): "A Structural Probe for Finding Syntax in Word Representations"

This project is directly inspired by and builds upon the seminal work by John Hewitt and Christopher D. Manning:

*   **Paper:** Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4129-4138). [Link to ACL Anthology](https://www.aclweb.org/anthology/N19-1042/)
*   **Original Code:** [https://github.com/john-hewitt/structural-probes](https://github.com/john-hewitt/structural-probes)

**2.1. Core Hypothesis:**
Hewitt & Manning hypothesized that the complex syntactic structure of sentences (specifically, dependency parse trees) is implicitly embedded in the *geometry* of the contextual word representation spaces learned by models like ELMo and BERT. They proposed that this structure could be revealed by finding a simple linear transformation of the embedding space.

**2.2. Methodology: The Structural Probe**
They introduced a "structural probe" with two main components:

*   **Distance Probe:** This probe learns a linear transformation matrix `B` such that, for any two words `w_i` and `w_j` in a sentence with embeddings `h_i` and `h_j`, the **squared L2 distance** between their transformed embeddings, `||B(h_i - h_j)||^2` (or equivalently `||B h_i - B h_j||^2`), approximates the **tree distance** (number of edges in the dependency parse tree) between `w_i` and `w_j`.
*   **Depth Probe (Norm Probe):** This probe learns a linear transformation matrix `B` such that the **squared L2 norm** of a transformed word embedding, `||B h_i||^2`, approximates the **depth** of word `w_i` in its dependency parse tree (distance from the root).

The probes are trained by minimizing the L1 loss between the predicted (squared L2) values and the true (non-squared, as per their code) tree-based values.

**2.3. Key Findings:**
*   They demonstrated that such linear transformations `B` (often low-rank, meaning syntax is encoded in a lower-dimensional subspace) can be found for ELMo and BERT representations.
*   The transformed spaces successfully encoded significant syntactic information, allowing for the reconstruction of dependency trees with notable accuracy (measured by Unlabeled Undirected Attachment Score - UUAS for distance, and Root Accuracy / Spearman correlation for depth).
*   BERT representations, particularly from its middle layers, showed stronger encoding of syntactic structure compared to ELMo.
*   Control baselines (e.g., non-contextual embeddings) did not show this strong geometric encoding of syntax.

**2.4. Significance:**
This work provided compelling evidence that deep language models, trained on self-supervised objectives like language modeling, learn and represent hierarchical syntactic structures implicitly within their vector spaces, even without explicit syntactic supervision during pre-training. It suggested that learning syntax is an emergent property of learning the language distribution effectively.

## 3. Motivation for This Project

This project aims to build upon Hewitt & Manning's foundational work with two primary motivations:

**3.1. Replication and Foundational Understanding:**
*   **Verification:** Replicating established research is a cornerstone of scientific progress. By first re-implementing and verifying their methodology (Phase 0a & Phase 1), we ensure a deep understanding of the original technique, its nuances, and its performance characteristics.
*   **Tooling:** Developing a modern, robust, and well-tested implementation of the structural probe provides a valuable tool for our own future research and for the community.

**3.2. Extension to Modern LLMs and New Research Questions:**
The NLP landscape has evolved significantly since 2019 with the advent of even larger and more powerful Transformer-based models (e.g., GPT series, Llama series, Mistral, T5, etc.), diverse pre-training objectives, and alignment techniques (SFT, RLHF). This raises new questions:

*   **Universality:** Is the geometric encoding of syntax found by H&M a general property of deep contextual language models, or was it specific to ELMo/BERT architectures and their training?
*   **Impact of Scale:** How does syntactic encoding change with model size (e.g., 7B vs. 70B parameters)?
*   **Architectural Differences:** Do decoder-only models (like Llama, GPT) encode syntax differently from encoder-based models (like BERT) or encoder-decoder models (like T5)?
*   **Effect of Pre-training Objectives:** How do different pre-training tasks beyond standard MLM affect syntactic representation?
*   **Effect of Fine-tuning & Alignment:** How do instruction fine-tuning (SFT) and alignment techniques like RLHF impact the foundational syntactic knowledge captured by the probes? Does alignment preserve, enhance, or potentially degrade these representations?
*   **Correlation with Performance:** Is there a correlation between how well a model's representations encode syntax (as measured by the probe) and its performance on downstream tasks, particularly syntactically demanding ones?
*   **Beyond Linearity:** Are there syntactic structures encoded non-linearly that the original linear probe might miss in newer models?

## 4. Project Aims & Research Questions

This project specifically aims to:

1.  **Aim 1 (Phase 0a & 1):** Successfully replicate the Hewitt & Manning (2019) structural probe methodology, first by running their legacy code on sample data, and then by creating a modern, validated PyTorch implementation.
2.  **Aim 2 (Phase 2 & 3):** Apply this modern structural probe to a diverse set of recent, high-performing LLMs available via Hugging Face, using the Penn Treebank (PTB) as the primary evaluation dataset. This involves:
    *   Extracting hidden state representations from various layers of these models.
    *   Training distance and depth probes for each layer.
    *   Analyzing and comparing the extent and location (which layers) of syntactic encoding across these models.
3.  **Aim 3 (Phase 4):** Explore selected extensions based on initial findings. Potential extensions include:
    *   Investigating non-linear probes.
    *   Comparing base pre-trained models with their instruction-tuned/aligned counterparts.
    *   Delving into more mechanistic interpretability approaches (e.g., connecting probe subspaces with Sparse Autoencoder features or performing causal interventions).
    *   Analyzing specific syntactic phenomena beyond general tree structure.

## 5. Expected Outcomes & Potential Impact

*   **Replication Results:** Confirmation (or nuanced understanding) of H&M's findings on ELMo/BERT with our setup.
*   **Modern LLM Analysis:** Novel insights into how current state-of-the-art LLMs represent syntax across different architectures, sizes, and training paradigms. This could reveal, for instance, if larger models inherently encode syntax better, or if certain architectures are more predisposed to learning specific types of syntactic geometry.
*   **Understanding Alignment:** Evidence on how SFT/RLHF affect the geometric encoding of syntax.
*   **Methodological Contribution:** A robust, open-source, modern implementation of the structural probe and associated tooling.
*   **Foundation for Future Work:** The findings and tools from this project can serve as a springboard for deeper interpretability research and for developing better models.

By systematically addressing these aims, this project hopes to contribute to the broader understanding of what linguistic structures are learned by powerful neural language models and how they are represented.

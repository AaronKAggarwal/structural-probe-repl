**`docs/PROJECT_OVERVIEW.md`**
# Project Overview: Replicating and Extending Structural Probes

Last updated: 2025-05-26

## 1. Introduction: Understanding Linguistic Knowledge in Neural Networks

The advent of large, pre-trained deep learning models for Natural Language Processing (NLP) has led to significant performance gains across a wide array of tasks. Models like ELMo, BERT, and their successors (e.g., Llama, Mistral) learn rich contextual representations of language. A key area of research in NLP interpretability, often termed "BERTology" or more broadly "modelology," seeks to understand *what* linguistic knowledge these models acquire during their extensive pre-training and *how* this knowledge is encoded in their internal vector representations (embeddings).

**Probing tasks** are a common methodology used to investigate these questions. A "probe" is typically a simple, supervised model trained to predict specific linguistic properties (e.g. part-of-speech tags, syntactic dependencies, semantic roles) from the internal representations of a larger, pre-trained language model. If a simple probe can successfully predict a property, it suggests that the information related to that property is readily accessible or "encoded" in the pre-trained model's embeddings.

## 2. Hewitt & Manning (2019): "A Structural Probe for Finding Syntax in Word Representations"

This project is directly inspired by and builds upon the seminal work by John Hewitt and Christopher D. Manning:

*   **Paper:** Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4129-4138). [Link to ACL Anthology](https://www.aclweb.org/anthology/N19-1042/)
*   **Original Code:** [https://github.com/john-hewitt/structural-probes](https://github.com/john-hewitt/structural-probes)

**2.1. Core Hypothesis:**
Hewitt & Manning hypothesised that the complex syntactic structure of sentences (specifically, dependency parse trees) is implicitly embedded in the *geometry* of the contextual word representation spaces learned by models like ELMo and BERT. They proposed that this structure could be revealed by finding a simple linear transformation of the embedding space.

**2.2. Methodology: The Structural Probe**
They introduced a "structural probe" with two main components:

*   **Distance Probe:** This probe learns a linear transformation matrix `B` such that, for any two words `w_i` and `w_j` in a sentence with embeddings `h_i` and `h_j`, the **squared L2 distance** between their transformed embeddings, `||B(h_i - h_j)||^2` (or equivalently `||B h_i - B h_j||^2`), approximates the **tree distance** (number of edges in the dependency parse tree) between `w_i` and `w_j`.
*   **Depth Probe (Norm Probe):** This probe learns a linear transformation matrix `B` such that the **squared L2 norm** of a transformed word embedding, `||B h_i||^2`, approximates the **depth** of word `w_i` in its dependency parse tree (distance from the root).

The probes are trained by minimising the L1 loss between the predicted (squared L2) values and the true (non-squared, as per their code) tree-based values.

**2.3. Key Findings:**
*   They demonstrated that such linear transformations `B` (often low-rank, meaning syntax is encoded in a lower-dimensional subspace) can be found for ELMo and BERT representations.
*   The transformed spaces successfully encoded significant syntactic information, allowing for the reconstruction of dependency trees with notable accuracy (measured by Unlabeled Undirected Attachment Score - UUAS for distance, and Root Accuracy / Spearman correlation for depth).
*   BERT representations, particularly from its middle layers, showed stronger encoding of syntactic structure compared to ELMo.
*   Control baselines (e.g., non-contextual embeddings) did not show this strong geometric encoding of syntax.
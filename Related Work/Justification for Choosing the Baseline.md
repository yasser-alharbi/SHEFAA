# Justification for Choosing the Baseline: "Fine-Tuning LLMs for Reliable Medical Question-Answering Services"

We have selected the paper *Fine-Tuning LLMs for Reliable Medical Question-Answering Services* as the baseline for implementing our SHEFAA project. Here are the two primary reasons for this choice:

### 1. Targeted Fine-Tuning Framework for Medical QA
This paper introduces a fine-tuning framework explicitly designed for medical question-answering tasks. It demonstrates how to adapt large language models (LLMs) to handle health-related queries and addresses key challenges like staying updated with rapidly changing medical knowledge. This approach aligns directly with our goal of building an Arabic medical Q&A system that fine-tunes pre-trained models to provide accurate and specialized clinical or health-related answers.

### 2. Focus on Robustness and Retrieval Augmentation
The paper emphasizes enhancing model reliability and reducing errors through its **ReRAG method**. This technique integrates:
- **Retrieval on Demand**: Ensures the model has access to verified and relevant context.
- **Question Rewriting**: Improves the quality of retrieved information and reduces the risk of generating incorrect or misleading answers.

These features are especially relevant to our project, as we aim to tackle hallucinations and improve the reliability of answers. By adopting this paper’s methods, we can establish a strong baseline while setting the stage for future enhancements, such as advanced retrieval mechanisms and hyperparameter optimization.

In summary, this paper provides a well-suited starting point for developing a robust and context-aware Arabic medical Q&A system.

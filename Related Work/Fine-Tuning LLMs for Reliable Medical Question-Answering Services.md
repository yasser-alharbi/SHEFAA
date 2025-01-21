# Fine-Tuning LLMs for Reliable Medical Question-Answering Services

**Paper Reference:**
[Anaissi, Ali et al., *“Fine-Tuning LLMs for Reliable Medical Question-Answering Services.”* (2024)](https://arxiv.org/pdf/2410.16088v1)

---

## Overview
This paper explores **fine-tuning** large language models (LLMs)—particularly **LLaMA-2** and **Mistral**—to increase the **reliability and accuracy** of healthcare question-answering. The authors introduce:
- **rsDoRA+**: A rank-stabilized, decomposed weight technique that boosts performance at higher ranks through differential learning rates for low-rank adapters.
- **ReRAG**: A *Retrieval on Demand* + *Question Rewrite* approach, which filters out irrelevant text and injects essential information into the model’s context.

The work highlights how these specialized fine-tuning methods can ensure **fast, dependable** medical QA, supporting patient trust and better decisions.

---

## Dataset 
They compile multiple medical QA resources:
- **Medical Meadow** (from medAlpaca, known medical databases)
- **Anki Flashcards** (rephrased with GPT-3.5)
- **MediQA** (manually generated summaries)
These collectively offer diverse medical question-answer pairs to train and test their approach.

---

## Methodology
1. **rsDoRA+**  
   - Combines LoRA (low-rank adapters) and DoRA (decomposed weights) with **rank stabilization** and *noise injection* (NEFtune) to reduce overfitting.
   - Adopts *different learning rates* for adapter matrices to improve efficiency, stability, and feature learning.

2. **ReRAG**  
   - A retrieval-augmented generation pipeline using “Retrieval Token,” “Relevance Tokens,” and a “Question Rewrite” node if initial retrieval is inadequate.
   - Ensures the model only uses the **most relevant** external text on demand, thus refining final QA outputs.

Models like **LLaMA2** and **Mistral** are fine-tuned with these techniques. They employ metrics like BLEU, ROUGE, and Mauve to compare baseline vs. proposed methods.

---

## Results & Findings
- **rsDoRA+** yields major improvements in high-rank fine-tuning tasks—overcoming scaling factor issues found in classic LoRA/DoRA.
- **ReRAG** further enhances question-answer accuracy, ensuring relevant textual evidence is integrated while discarding irrelevant content.
- Gains exceed **100%** on certain QA metrics under specific conditions, demonstrating strong potential for more accurate and rich medical answers.
- Conclude these synergy-based methods (fine-tuning + retrieval) can significantly **boost** LLM performance in medical QA, even on limited data.


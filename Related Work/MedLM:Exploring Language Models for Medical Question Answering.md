# MedLM: Exploring Language Models for Medical Question Answering Systems

**Paper Reference:**
[Yagnik, Niraj et al., *“MedLM: Exploring Language Models for Medical Question Answering Systems.”* (2024)](https://arxiv.org/pdf/2401.11389)

---

## Overview
This paper compares **general vs. domain-specific** large language models for **medical Q&A**, focusing on **closed-book generative** tasks. They evaluate:
- Fine-tuning smaller “distilled” LMs on Q&A corpora
- Different prompt-based strategies (static vs. dynamic)
- Reliability issues like hallucinations and response quality

It aims to find which model families (e.g., GPT-2 vs. T5) and what prompting methods best suit healthcare question-answering.

---

## Dataset
They utilize two main data sources:
- **MedQuAD** (47k Q&A pairs from NIH websites)
- **Icliniq** (29k Q&A pairs from various health forums like eHealth Forum, WebMD, etc.)

They truncate longer answers (300 tokens for MedQuAD, 150 for Icliniq) to keep training more efficient.

---

## Methodology
1. **Fine-Tuning Distilled LMs**  
   - Train smaller variants (e.g., GPT-2 Distil) on the consolidated Q&A pairs, ensuring each question-answer is concatenated.
   - Apply data augmentation by mixing MedQuAD and Icliniq to broaden coverage.

2. **Testing Base LLMs**  
   - Evaluates pretrained GPT-2, GPT-3.5, T5, and Bloom with direct prompting—no fine-tuning—to measure baseline capabilities.

3. **Prompting Approaches**  
   - **Static Prompting**: A fixed set of example Q&As is always appended to the prompt.
   - **Dynamic Prompting**: Finds the top-k similar training questions (via embeddings) for each test query.  
     - *Question-Type Specific* dynamic prompting further classifies question types (Symptoms, Treatments, etc.) before retrieving top examples.

---

## Results & Findings
- **Static Prompting** is sometimes strong but inconsistent across queries.
- **Dynamic Prompting** more consistently boosts BLEU/ROUGE and yields better human evaluations.
- Data augmentation with multiple sources (MedQuAD + Icliniq) improves fine-tuned model performance, mitigating hallucinations somewhat.
- **User/doctor surveys** reveal GPT-based models often appear more factual, though they can be verbose; smaller models hallucinate frequently but are more controllable.

**Implications for an Arabic Medical Q&A**:
- Emphasizes *prompt engineering*: retrieving relevant Q&A pairs might reduce hallucinations and ensure domain accuracy.
- Shows how **domain-labeled** question types can direct context retrieval and produce more tailored Arabic answers.
- Highlights the gap between automatic metrics and real-world acceptability, suggesting careful human evaluation for medical QA in Arabic.


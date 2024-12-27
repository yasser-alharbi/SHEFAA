# Evaluation Metrics in the SHAFAA Project

## Introduction
To comprehensively evaluate the SHAFAA Arabic Medical Question Answering system, we employ a combination of precision-based and semantic evaluation metrics. These metrics assess the system's ability to generate accurate, contextually appropriate, and semantically coherent answers. The evaluation includes BLEU, ROUGE, and BERTScore, offering a multi-faceted perspective on the system's performance.

---

## Evaluation Metrics

### 1. **BLEU (Bilingual Evaluation Understudy)**
- **Description**: BLEU measures n-gram precision by comparing the overlap of n-grams between the system's predictions and the reference answers.
- **Variants**:
  - **BLEU-1**: Unigram overlap.
  - **BLEU-2**: Bigram overlap.
  - **BLEU-4**: 4-gram overlap.
- **Purpose**: Measures lexical overlap and phrase-level accuracy in generated responses.
- **Implementation**: Calculated using the NLTK library with smoothing techniques to handle short text scenarios.

---

### 2. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- **Description**: ROUGE emphasizes recall, evaluating how much of the reference text is covered by the generated text.
- **Variants**:
  - **ROUGE-1**: Unigram overlap.
  - **ROUGE-2**: Bigram overlap.
  - **ROUGE-L**: Longest Common Subsequence (LCS), evaluating in-order matches.
- **Purpose**: Highlights both word-level and sequence-level similarities, providing insights into the coherence of generated answers.
- **Implementation**: Calculated using the `rouge_score` library.

---

### 3. **BERTScore**
- **Description**: BERTScore uses pretrained transformer embeddings (e.g., BERT) to compare the semantic similarity between generated and reference texts. Instead of exact word matching, it computes cosine similarity between token embeddings.
- **Metrics**:
  - **Precision**: Measures how well the generated text captures relevant parts of the reference.
  - **Recall**: Measures how much of the reference text is reflected in the generated output.
  - **F1-Score**: Balances precision and recall to provide an overall performance score.
- **Implementation**:
  - Leverages the `bert-score` library with multilingual model support (`lang='ar'`) for Arabic text.
  - Example usage:
    ```python
    P, R, F1 = compute_bertscore(y_valid_sample, y_pred, lang='ar')
    print(f"Average BERTScore Precision: {P.mean() * 100:.2f}%")
    print(f"Average BERTScore Recall: {R.mean() * 100:.2f}%")
    print(f"Average BERTScore F1: {F1.mean() * 100:.2f}%")
    ```
- **Purpose**: Evaluates semantic alignment between generated and reference answers, making it ideal for tasks where exact word matching is insufficient.


---

## Summary of Metrics

| Metric     | Precision | Recall | F1-Score | Purpose                               |
|------------|-----------|--------|----------|---------------------------------------|
| BLEU       | ✅         | ❌      | ❌        | Lexical overlap and phrase-level accuracy. |
| ROUGE      | ✅         | ✅      | ✅        | Word-level and sequence-level recall. |
| BERTScore  | ✅         | ✅      | ✅        | Semantic similarity using embeddings. |

---

## Conclusion
The combination of BLEU, ROUGE, BERTScore, and BARTScore ensures a comprehensive evaluation of the SHAFAA system. Each metric highlights different aspects of the system’s performance, from lexical and semantic accuracy to fluency and coherence. This multi-faceted evaluation helps identify strengths and areas for improvement in the model.


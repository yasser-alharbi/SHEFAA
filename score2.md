# Evaluation Metrics Documentation

This document describes the three evaluation metrics used in our project: BLEU, ROUGE, and BERTScore. Each metric has a formal definition, relevant citations, an example calculation, and instructions for running the evaluation script.

---

## 1. BLEU Score

### Formal Definition
BLEU (Bilingual Evaluation Understudy) is a precision-focused metric that evaluates the n-gram overlap between the generated text and the reference text. It is defined as:

**BLEU = BP * exp(Σ (w_n * log(p_n)))**

Where:
- **BP**: Brevity Penalty to account for length differences.
- **p_n**: Precision of n-grams.
- **w_n**: Weight assigned to n-grams (typically equal weights).

### Example
Reference: "The cat is on the mat."
Generated: "The cat is on a mat."

- BLEU-1 (unigrams): Precision = 5/6 (matches: "The", "cat", "is", "on", "mat")
- BLEU-2 (bigrams): Precision = 4/5 (matches: "The cat", "cat is", "is on", "on mat")

### Citations
- Papineni, K., et al., 2002. "BLEU: a Method for Automatic Evaluation of Machine Translation." [Link](https://aclanthology.org/P02-1040/)
- Wikipedia: [BLEU](https://en.wikipedia.org/wiki/BLEU)

---
## 2. ROUGE Score

### Formal Definition
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics that evaluate the quality of generated text by comparing it to reference text. ROUGE focuses on measuring recall, which reflects how much of the reference text is captured by the generated text. Key variants include:
- **ROUGE-1**: Measures the overlap of unigrams (single words) between the generated and reference text.
- **ROUGE-2**: Measures the overlap of bigrams (two consecutive words).
- **ROUGE-L**: Based on the longest common subsequence (LCS) between the generated and reference text.

### Equations

#### ROUGE-N:
![ROUGE-N Formula](https://github.com/yasser-alharbi/SHEFAA/blob/main/ROUGE-N.png)

**Description**:  
The ROUGE-N metric calculates the ratio of the number of matching n-grams between the generated text and the reference text to the total number of n-grams in the reference text. It emphasizes how much content from the reference is reflected in the generated text.

- **Matching n-grams**: Number of n-grams in the reference that also appear in the generated text.
- **Total n-grams in the reference**: The total number of n-grams in the reference text.

---

#### ROUGE-L:
![ROUGE-L Formula](https://github.com/yasser-alharbi/SHEFAA/blob/main/ROUGE-L.png)

**Description**:  
The ROUGE-L metric is based on the longest common subsequence (LCS), which represents the longest sequence of tokens that appear in both the generated and reference text in the same order. ROUGE-L considers both precision and recall through the F-measure, which balances the two using a weighting parameter \( \beta \).

- \( P \): Precision
![image alt](https://github.com/yasser-alharbi/SHEFAA/blob/main/Recall.png)
- \( R \): Recall
![image alt](https://github.com/yasser-alharbi/SHEFAA/blob/main/Recall.png)
- \( F_\beta \): A harmonic mean of Precision and Recall, controlled by \( \beta \) (usually set to 1 for equal weight).

---

These equations demonstrate how ROUGE measures text overlap, making it an effective evaluation metric for tasks like text summarization, translation, and question answering.


## 2. ROUGE Score

### Formal Definition
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures recall by comparing n-grams and sequences in the generated text to the reference text. Key variants include:
- ROUGE-1: Overlap of unigrams.
- ROUGE-2: Overlap of bigrams.
- ROUGE-L: Longest common subsequence.

![image alt](https://github.com/yasser-alharbi/SHEFAA/blob/main/ROUGE-N.png)
![image alt](https://github.com/yasser-alharbi/SHEFAA/blob/main/ROUGE-L.png)

### Example
Reference: "The quick brown fox jumps."
Generated: "The quick brown dog jumps."

- ROUGE-1: Recall = 4/5 (matches: "The", "quick", "brown", "jumps")
- ROUGE-2: Recall = 3/4 (matches: "The quick", "quick brown", "brown jumps")

### Citations
- Lin, C.-Y., 2004. "ROUGE: A Package for Automatic Evaluation of Summaries." [Link](https://aclanthology.org/W04-1013/)

---

## 3. BERTScore

### Formal Definition
BERTScore evaluates the semantic similarity between the generated and reference texts using embeddings from pre-trained BERT models. It computes:
- Precision: Semantic overlap in generated text.
- Recall: Semantic overlap in reference text.
- F1 Score: Harmonic mean of Precision and Recall.

### Example
Reference: "The sky is clear."
Generated: "The sky is blue."

The BERT embeddings capture that "clear" and "blue" are semantically similar, resulting in a high similarity score.

### Citations
- Zhang, T., et al., 2020. "BERTScore: Evaluating Text Generation with BERT." [Link](https://arxiv.org/abs/1904.09675)

---

## Instructions to Run the Evaluation Script

### Prerequisites
- Python 3.x
- Required libraries: `nltk`, `rouge-score`, `bert-score`
- Files:
  - `1000_Sample_Prediction.txt`: Generated text.
  - `1000_Sample_Reference.txt`: Reference text.

### Command to Run
```bash
python score.py 1000_Sample_Prediction.txt 1000_Sample_Reference.txt

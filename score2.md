# Evaluation Metrics Documentation

This document describes the three evaluation metrics used in our project: BLEU, ROUGE, and BERTScore. Each metric has a formal definition, relevant citations, an example calculation, and instructions for running the evaluation script.

---

## 1. BLEU Score

### Formal Definition
BLEU (Bilingual Evaluation Understudy) is a precision-focused metric that evaluates the n-gram overlap between the generated text and the reference text.
BLEU's output is always a number between 0 and 1.
It is defined as:


### Example
**Reference**: "The cat is on the mat."  
**Generated**: "The cat is on a mat."

- **BLEU-1 (Unigrams)**:  
  - **Matching Unigrams**: "The", "cat", "is", "on", "mat"  
  - **Total Unigrams in Generated Text**: 6  
  - **Precision** = `Number of Matching Unigrams / Total Unigrams = 5 / 6 = 0.833`

- **BLEU-2 (Bigrams)**:  
  - **Matching Bigrams**: "The cat", "cat is", "is on", "on mat"  
  - **Total Bigrams in Generated Text**: 5  
  - **Precision** = `Number of Matching Bigrams / Total Bigrams = 4 / 5 = 0.8`

- **BLEU-4 (4-grams)**:  
  - **Matching 4-grams**: "The cat is on", "cat is on a"  
  - **Total 4-grams in Generated Text**: 3  
  - **Precision** = `Number of Matching 4-grams / Total 4-grams = 2 / 3 ≈ 0.667`

**Final BLEU Scores**:  
- BLEU-1 = `0.833`  
- BLEU-2 = `0.8`  
- BLEU-4 = `0.667`


### Citations
- geeksforgeeks: [Understanding BLEU and ROUGE score](https://www.geeksforgeeks.org/understanding-bleu-and-rouge-score-for-nlp-evaluation/)
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

- \( F1 \): A harmonic mean of Precision and Recall, controlled by beta (usually set to 1 for equal weight).

### Example
Reference: "The quick brown fox jumps."
Generated: "The quick brown dog jumps."

- **ROUGE-1**: Recall = `4/5`  
  **Matches**: "The", "quick", "brown", "jumps"

- **ROUGE-2**: Recall = `3/4`  
  **Matches**: "The quick", "quick brown", "brown jumps"

- **ROUGE-L**:  
  - **Longest Common Subsequence (LCS)**: "The quick brown jumps"  
  - LCS Length = 4 tokens  
  - **Precision** = `LCS Length / Total Tokens in Generated Text = 4 / 5 = 0.8`  
  - **Recall** = `LCS Length / Total Tokens in Reference Text = 4 / 5 = 0.8`  
  - **F-measure** (`F_β`, where `β = 1`):  
    ```
    F_β = (1 + β²) * P * R / (β² * P + R)
        = (1 + 1²) * 0.8 * 0.8 / (1² * 0.8 + 0.8)
        = 0.8
    ```

**Final ROUGE Scores**:  
- ROUGE-1 = `0.8`  
- ROUGE-2 = `0.75`  
- ROUGE-L = `0.8` 



### Citations
- geeksforgeeks: [Understanding BLEU and ROUGE score](https://www.geeksforgeeks.org/understanding-bleu-and-rouge-score-for-nlp-evaluation/)

---
## 3. BERTScore

### Formal Definition
BERTScore evaluates the semantic similarity between the generated text and the reference text using embeddings from pre-trained BERT models. Unlike traditional n-gram-based metrics (e.g., BLEU and ROUGE), BERTScore captures the contextual and semantic meaning of the text by leveraging pairwise cosine similarity between token embeddings.

### BERTScore computes:
- **Precision**: Measures how much of the generated text matches the reference text in meaning.
- **Recall**: Measures how much of the reference text is captured by the generated text.
- **F1 Score**: The harmonic mean of Precision and Recall, balancing their contributions.

---

### Example: BERTScore Calculation
**Reference Text**: "The weather is cold today."  
**Generated Text**: "It is freezing today."

1. **Contextual Embeddings**:  
   Each word in the reference and generated text is represented as a high-dimensional vector using a pre-trained BERT model. These embeddings encode the semantic meaning of words in their context.



2. **Pairwise Cosine Similarity**:  
   The cosine similarity is computed pairwise between each token in the generated text and every token in the reference text. This results in a similarity matrix that highlights how semantically similar the tokens are.

   ![Pairwise Cosine Similarity](https://github.com/yasser-alharbi/SHEFAA/blob/main/Pairwise%20Cosine%20Similarity.png)

3. **Maximum Similarity**:  
   For each token in the generated text, the maximum similarity with any token in the reference text is calculated (and vice versa). This ensures that each word contributes its best semantic match.

   ![Maximum Similarity](https://github.com/yasser-alharbi/SHEFAA/blob/main/Maximum%20Similarity.png)

5. **Precision, Recall and F1**:  
   - **Precision**: The average of the maximum similarities for tokens in the generated text.
   - **Recall**: The average of the maximum similarities for tokens in the reference text.
   - **F1**: The harmonic mean of Precision and Recall, balancing their contributions
  
  
   ![Precision, Recall and F1](https://github.com/yasser-alharbi/SHEFAA/blob/main/Precision,%20Recall%20and%20F1%20for%20BERTScore.png)

**Result**:  
- Precision = `0.85`  
- Recall = `0.83`  
- F1 Score = `0.84`


### Citations
- Zhang, T., et al., 2020. "BERTScore: Evaluating Text Generation with BERT." [Link](https://arxiv.org/abs/1904.09675)

---

This markdown integrates the images and detailed step-by-step explanations to provide a comprehensive understanding of BERTScore.


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

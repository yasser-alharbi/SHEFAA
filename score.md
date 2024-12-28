# Evaluation Metrics
## Intoduction
To comprehensively evaluate the SHAFAA Arabic Medical Question Answering system, we employ a combination of precision-based and semantic evaluation metrics. These metrics assess the system's ability to generate accurate, contextually appropriate, and semantically coherent answers. The evaluation includes BLEU, ROUGE, and BERTScore.

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
- GeeksforGeeks: [Understanding BLEU and ROUGE score](https://www.geeksforgeeks.org/understanding-bleu-and-rouge-score-for-nlp-evaluation/)
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
![ROUGE-N Formula](https://github.com/yasser-alharbi/SHEFAA/blob/main/images/ROUGE-N.png)

**Description**:  
The ROUGE-N metric calculates the ratio of the number of matching n-grams between the generated text and the reference text to the total number of n-grams in the reference text. It emphasizes how much content from the reference is reflected in the generated text.

- **Matching n-grams**: Number of n-grams in the reference that also appear in the generated text.
- **Total n-grams in the reference**: The total number of n-grams in the reference text.

---

#### ROUGE-L:
![ROUGE-L Formula](https://github.com/yasser-alharbi/SHEFAA/blob/main/images/ROUGE-L.png)

**Description**:  
The ROUGE-L metric is based on the longest common subsequence (LCS), which represents the longest sequence of tokens that appear in both the generated and reference text in the same order. ROUGE-L considers both precision and recall through the F-measure, which balances the two using a weighting parameter \( \beta \).

- \( P \): Precision

![image alt](https://github.com/yasser-alharbi/SHEFAA/blob/main/images/Recall.png)

- \( R \): Recall

![image alt](https://github.com/yasser-alharbi/SHEFAA/blob/main/images/Recall.png)

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

1. **Contextual Embeddings**:  
   Each word in the reference and generated text is represented as a high-dimensional vector using a pre-trained BERT model. These embeddings encode the semantic meaning of words in their context.


2. **Pairwise Cosine Similarity**:  
   The cosine similarity is computed pairwise between each token in the generated text and every token in the reference text. This results in a similarity matrix that highlights how semantically similar the tokens are.

   ![Pairwise Cosine Similarity](https://github.com/yasser-alharbi/SHEFAA/blob/main/images/Pairwise%20Cosine%20Similarity.png)

3. **Maximum Similarity**:  
   For each token in the generated text, the maximum similarity with any token in the reference text is calculated (and vice versa). This ensures that each word contributes its best semantic match.

   ![Maximum Similarity](https://github.com/yasser-alharbi/SHEFAA/blob/main/images/Maximum%20Similarity.png)

5. **Precision, Recall and F1**:  
   - **Precision**: The average of the maximum similarities for tokens in the generated text.
   - **Recall**: The average of the maximum similarities for tokens in the reference text.
   - **F1**: The harmonic mean of Precision and Recall, balancing their contributions
  
  
   ![Precision, Recall and F1](https://github.com/yasser-alharbi/SHEFAA/blob/main/images/Precision,%20Recall%20and%20F1%20for%20BERTScore.png)




### Citations
- Medium: [BERTScore Explained in 5 minutes](https://medium.com/@abonia/bertscore-explained-in-5-minutes-0b98553bfb71)
- YouTube: [BERTScore: Evaluating Text Generation with BERT](https://www.youtube.com/watch?v=Nq4VKXhumSY&t=34s)

---


## Summary of Metrics


| Metric       | Precision Support | Recall Support | F1-Score Support | Main Focus                                      | Why?                                                                                   |
|--------------|-------------------|----------------|------------------|------------------------------------------------|----------------------------------------------------------------------------------------|
| **BLEU**     | ✅                 | ❌             | ❌               | Measures n-gram precision for exact word and phrase matching.                         | Best for fact-based QA requiring exact wording, but struggles with paraphrased or semantic answers. |
| **ROUGE**    | ✅                 | ✅             | ✅               | Measures how much of the reference is covered in the generated answer.               | Useful for recall-focused tasks, ensuring all key points from the reference are included. |
| **BERTScore**| ✅                 | ✅             | ✅               | Assesses semantic similarity using contextual embeddings.                             | Ideal for conversational or paraphrased QA, as it focuses on semantic correctness and meaning.        |


---


## Choosing the Best Metric for QA Answer Generation

For a **QA answer generation task**, the best metric depends on the type of answers your system produces and the evaluation priorities. Here’s a breakdown of each metric:

---

### 1. BLEU  
**Best for:** Fact-based QA tasks with short, precise answers where exact phrasing is important.

**Scoring Interpretation:** Higher BLEU scores indicate better performance, with a score of `1.0` representing a perfect match.

**For QA:** BLEU is useful only if your task demands exact wording. For example, in legal or medical QA tasks, precision is key.

---

### 2. ROUGE  
**Best for:** Long-form QA answers where recall (including all important information) is key.

**Scoring Interpretation:** Higher ROUGE scores indicate better performance, as they reflect greater overlap between the generated and reference answers.

**For QA:** ROUGE is useful if your task prioritizes covering all key points in the reference answer. However, it doesn’t handle semantic variations as well as other metrics.

---

### 3. BERTScore  
**Best for:** Generative QA systems that prioritize meaning and semantic similarity.

**Scoring Interpretation:** Higher BERTScore values indicate better semantic similarity, with scores closer to `1.0` representing strong alignment.

**For QA:** BERTScore is the most appropriate metric for QA systems that generate conversational answers, where meaning matters more than exact wording.

---

## Final Recommendation

For your **QA answer generation task**, **BERTScore** is the best choice because it aligns with the goal of generating semantically correct answers. It works particularly well

---

### Metric Ranking for QA Answer Generation

| Rank  | Metric       |
|-------|--------------|
| 1st   | **BERTScore**|
| 2nd   | **ROUGE**    |
| 3rd   | **BLEU**     |




---


## Instructions to Run the Evaluation Script

### Prerequisites
- Required downloads libraries: `nltk`, `rouge-score`, `bert-score`
- Files:
  - `1000_Sample_Prediction.txt`: Generated text.
  - `1000_Sample_Reference.txt`: Reference text.

### Command to Run
```bash
python score.py --predictions 1000_Sample_Prediction.txt --references 1000_Sample_Reference.txt

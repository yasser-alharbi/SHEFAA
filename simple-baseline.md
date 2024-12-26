# 2. Baseline

In this baseline approach, we use **TF-IDF** to encode the training questions and then identify the most similar question for each test input via **cosine similarity**. Due to the size of our validation set (**71,519** entries), we take a sample of **1,000** questions from **X_valid** for efficiency.

---

## Approach

1. **TF-IDF Vectorization**  
   - Fit a TF-IDF vectorizer on the **X_train** questions.
   - Transform **X_train** questions into TF-IDF vectors (`X_train_tfidf`).

2. **Sample the Validation Set**  
   - Select a subset of **1,000** questions from **X_valid** (along with the corresponding **y_valid**).

3. **Find Most Similar Question**  
   - For each question in the **X_valid** sample:
     1. Transform the question into a TF-IDF vector.
     2. Compute cosine similarity with all **X_train** TF-IDF vectors.
     3. Retrieve the index of the highest similarity score.
     4. Use that index to fetch the corresponding answer from **y_train**.

4. **Compare Predictions**  
   - Store all predicted answers in **y_pred** and compare with **y_valid** to evaluate how often the baseline predicts answers correctly or similarly.

---

## Code Snippet

```python
# Build TF-IDF on the training question
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Take a sample of 1,000 from y_valid for quick testing
y_valid_sample = y_valid[:1000]

# For each question in the sample of X_valid, get the most similar training question
y_pred = []
for test_question in X_valid[:1000]:
    test_question_tfidf = vectorizer.transform([test_question])
    similarities = cosine_similarity(test_question_tfidf, X_train_tfidf)
    best_index = np.argmax(similarities)
    predicted_answer = y_train.iloc[best_index]
    y_pred.append(predicted_answer)

print("y_pred shape", len(y_pred))
print("y_valid shape", len(y_valid_sample))

# Show example for one test question
print(f"Test question: \n{X_valid.iloc[50]}")
print(f"Predicted answer: \n{y_pred[50]}")
print(f"True answer: \n{y_valid_sample.iloc[50]}")

```

## **Results**

**Example**:


*   Test question:
```python
سؤال: السلام عليكم انا عمري ١٥ دائما تيجيني تشنجات في اصابع رجلي وفي اصابع يدي ومني عارف من ايش وهي ايش | التصنيف: الأمراض العصبية
```
*   Predicted answer:
```python
الإجابة: علیک فحص مقدار الکلسیم فی الدم
```
*   True answer:
```python
الإجابة: اجراء فحص الکلسیم
```


## Analysis of Results
In the example above:

1. **Similarity of Context**

*   The predicted answer directly advises checking calcium levels in the blood. This is closely aligned with the true answer, which suggests conducting a calcium test.

*   This indicates that the TF-IDF + cosine similarity approach can successfully pick up on key medical terms such as "الکلسیم" (calcium) to find the most relevant answer.


2. **Limitations**

*   Some questions may not have an exact or close match in the training set, leading to potentially irrelevant answers.

*   TF-IDF does not capture deeper semantic relationships beyond exact or partially matched terms.


## Conclusion

*   The baseline method provides a quick way to retrieve a contextually relevant answer by leveraging text similarity.

*   In many cases, as seen in the example, it can produce answers that are relevant to the context and helpful.

# Fine-Tuning Jais-family-256m with QLoRA (Baseline)

## Introduction
This document describes a baseline system for fine-tuning the **Jais-family-256m** model on an Arabic medical question-answering dataset using **QLoRA**. By loading the model weights in 4-bit precision (through [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)), we can reduce the hardware requirements while still adapting the model to our specific domain.

## Dataset Preparation
We assume three CSV files: **train.csv**, **valid.csv**, and **test.csv**, each containing:

- **Question**: the medical question asked by the user,
- **Category**: an associated category (e.g., الأمراض العصبية, الحمل والولادة, etc.),
- **Answer**: the ground-truth or reference response.

 > **Note**: Due to limitation in resources in this baseline, only **10% of the data** was used for the implementation (**715,187 * 10% = 71,515**).


We convert these CSV files into Hugging Face Datasets for streamlined tokenization and training.

## QLoRA Approach

1. **BitsAndBytes 4-bit Quantization**  
   We load the base model (`Jais-family-256m`) in 4-bit precision. This greatly reduces VRAM usage without heavily sacrificing performance.

2. **LoRA Adapters**  
   - **Rank (r):** 8  
   - **Alpha:** 32  
   - **Dropout:** 0.1  
   - **Target Modules:** `["c_attn", "c_proj", "c_fc", "c_fc2"]`  

   These LoRA modules adapt only a small set of trainable parameters on top of the frozen base model layers, making fine-tuning more efficient.

3. **Prompt Engineering**  
   For each training sample, we create a prompt of the form:
  ```python
   f"سؤال: {Question}\nالتصنيف: {Category}\nالإجابة:"
```
   We then tokenize both the prompt and the reference answer separately. The final input_ids (prompt) serve as **input**, and the tokenized answer serves as **labels** for causal language modeling.

4. **Training Configuration**  
   - **Epochs:** 1 (demonstration; can be increased)  
   - **Batch Size:** 8 (can be tuned based on GPU memory)  
   - **Learning Rate:** 1e-4  
   - **Mixed Precision (FP16):** Enabled to further reduce memory usage  
   - **Evaluation Strategy:** By epoch  
   - **Best Model Selection:** Based on validation loss  

## Code Snippet
Below is an abbreviated snippet of how we configure QLoRA:

```python
from bitsandbytes import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. Bits and Bytes Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 
    bnb_4bit_compute_dtype="float16",  
    bnb_4bit_quant_type="nf4",         
    bnb_4bit_use_double_quant=True     
)

# 2. Load Base Model & Tokenizer
model_path = "inceptionai/Jais-family-256m"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# 3. LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "c_proj", "c_fc", "c_fc2"]
)
model = get_peft_model(model, lora_config)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="Jais-family-256m-lora-SHEFAA",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    fp16=True,
    learning_rate=1e-4,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)
trainer.train()

```
## The Process For Generation and Evaluation

The SHEFAA project provides two key scripts—`baseline.py` and `score.py`—to generate answers to medical questions and evaluate the model's performance.

---

### **Functionality Overview**

#### **`baseline.py`**
- Generates answers to medical questions using a pre-trained language model.
- Reads a dataset containing questions, categories, and ground-truth answers, processes it, and produces predictions.

#### **`score.py`**
- Evaluates the quality of the model's predictions by comparing them against ground-truth answers.
- Computes performance metrics such as BLEU, ROUGE, or other specified evaluation metrics.

---

## **Inputs, Processing, and Outputs**

### **`baseline.py`**

#### **Inputs**
- **Dataset**: A CSV file with the following columns:
  - **Question**: The medical question to be answered.
  - **Category**: The type of the question (e.g., symptoms, diagnosis, treatment).
  - **Answer**: The reference or ground-truth answer.

#### **Processing**
1. **Data Loading and Preprocessing**:
   - The script reads the input CSV file to extract the "Question," "Category," and "Answer" columns.
   - Questions and categories are tokenized and formatted for input into the language model.

2. **Model Initialization**:
   - A pre-trained language model (likely from Hugging Face) is loaded.
   - The model is prepared for inference, ensuring it is compatible with CUDA for GPU-accelerated processing.

3. **Answer Generation**:
   - The tokenized input questions are fed into the model.
   - The model generates answers based on the context provided by the question and category.


#### **Outputs**
- **`references.txt`**: Contains the ground-truth answers extracted from the dataset.
- **`predictions.txt`**: Contains the model-generated answers for the input questions.

---

### **`score.py`**

#### **Inputs**
- **`references.txt`**: Produced by `baseline.py`, containing the ground-truth answers.
- **`predictions.txt`**: Produced by `baseline.py`, containing the model's generated answers.

#### **Processing**
1. **Loading Predictions and References**:
   - The script reads the `references.txt` and `predictions.txt` files.
   - It aligns the predictions with their corresponding ground-truth answers.

2. **Evaluation Metrics Computation**:
   - Metrics such as BLEU, ROUGE, or other evaluation scores are computed.
   - These metrics measure the similarity between the generated answers and the ground-truth answers, providing a quantitative assessment of the model's accuracy and relevance.

3. **Reporting Results**:
   - The computed metrics are displayed in the console or saved to a results file for further analysis.

#### **Outputs**
- Evaluation metrics (e.g., BLEU, ROUGE scores) that quantify the model's performance.

---

## **Key Requirements to Run the Scripts**

### **1. CUDA Compatibility (for `baseline.py`)**
- A CUDA-enabled GPU is essential for running `baseline.py` effectively.
- CUDA is required for the `bitsandbytes` library, which optimizes model inference.  
  Without CUDA, the following error will occur:


## Example Generation

**Example**:


*   Test question:
```python
سؤال: عفوا يمكنني معرفة تاخر دورة شهرية لمدة 15 يوم من 5 ايام عملت تحليل الحمل ولكن النتيجة كانت سلبية و اود ان اعرف سبب هدا التاخر لانها المرة الاولى | التصنيف: الحمل والولادة
```
*   Generated answer:
```python
الإجابة: حدوث التبويض بشكل غير صحيح, قد تكون الدورة الشهرية لديك غير منتظمة
```
*   True answer:
```python
الإجابة:  قد يكون السبب ارتفاع في نسبة هرمون الحليب او اضطراب في احدى الهرمونات المسؤولة عن تنظيم عمليات الاباضة والحمل 
```
We evaluate on a small subset of 1000 examples for both validation and test sets (to reduce computation time for demonstration).
# Evaluating the Model
## BLEU
- **BLEU-1, BLEU-2, BLEU-4**  
- Calculated with `nltk.translate.bleu_score`.

## ROUGE
- **ROUGE-1, ROUGE-2, ROUGE-L**  
- Uses `rouge_score.rouge_scorer`.  
- We report the **F-measure** of each.

## BERTScore
- **Precision, Recall, F1**  
- Computed with `bert_score.score`.  
- For Arabic text, `lang='ar'` is used.

---
# Result

| **Metric**       | **Baseline (10% Data)**<br>(Test, 1000 Sample, *After Cleaning*) | **Baseline (10% Data)**<br>(Valid, 1000 Sample, *After Cleaning*) | **Baseline (10% Data)**<br>(Test, 1000 Sample, *Before Cleaning*) | **Baseline (10% Data)**<br>(Valid, 1000 Sample, *Before Cleaning*) | **TF‑IDF Baseline (100% Data)**<br>(Test, 1000 Sample) |
|------------------|-----------------------------------------------------------------|--------------------------------------------------------------------|---------------------------------|-----------------------------------|-------------------------------------------|
| **BLEU‑1**       | 0.037                                                           | 0.033                                                              | 0.020                           | 0.017                             | 0.118                                     |
| **BLEU‑2**       | 0.015                                                           | 0.014                                                              | 0.010                           | 0.008                             | 0.081                                     |
| **BLEU‑4**       | 0.006                                                           | 0.006                                                              | 0.004                           | 0.003                             | 0.036                                     |
| **ROUGE‑1 (F1)** | 0.001                                                           | 0.002                                                              | 0.000                           | 0.000                             | 0.005                                     |
| **ROUGE‑2 (F1)** | 0.000                                                           | 0.001                                                              | 0.000                           | 0.000                             | 0.003                                     |
| **ROUGE‑L (F1)** | 0.001                                                           | 0.002                                                              | 0.000                           | 0.000                             | 0.005                                     |
| **BERTScore P**  | 61.40%                                                          | 60.75%                                                             | 60.93%                          | 60.42%                            | 71.08%                                    |
| **BERTScore R**  | 61.59%                                                          | 60.66%                                                             | 65.99%                          | 65.78%                            | 70.86%                                    |
| **BERTScore F1** | 61.33%                                                          | 60.57%                                                             | 63.18%                          | 62.83%                            | 70.78%                                    |

### Observations and Comparison Between Baseline and Simple-Baseline (TF-IDF)
The comparison shows that the simple baseline consistently outperforms the current baseline system trained on only **10% of the data** and for just **1 epoch** across all metrics. BLEU scores and ROUGE metrics highlight a gap in token overlap between the model’s predictions and ground truth, with the simple baseline performing 68–100% better. Similarly, BERTScore metrics show a smaller but notable advantage (~13%) in semantic similarity for the random baseline, suggesting that its predictions are more contextually aligned with the reference answers.

---

## Conclusion
- While the simple baseline performs better than the current baseline system, this is largely due to the limited training data (10%). Expanding the dataset to its full size (100%)  are expected to significantly improve performance, especially for BLEU and ROUGE metrics, which are sensitive to token overlaps.
- Additionally, optimizing hyperparameters (e.g. the number of epochs) and incorporating domain-specific fine-tuning can help close the gap and likely surpass the simple baseline. Moving forward, the model’s performance can be enhanced by focusing on error analysis, data augmentation, and leveraging advanced techniques like LoRA for lightweight fine-tuning.


# 🩺 **SHEFAA Project**  

## 🌟 Overview  
SHEFAA is an **Arabic medical question-answering (QA) system** designed to provide **reliable and contextually accurate** responses to health-related queries. The system is fine-tuned using **QLoRA (Quantized Low-Rank Adapter)** on the **Jais-family-256m** model, a state-of-the-art Arabic language model optimized for understanding and generating medical text.

By leveraging QLoRA, SHEFAA maintains the efficiency of large-scale language models while enabling fine-tuning on **specialized medical datasets**, ensuring domain-specific accuracy without requiring extensive computational resources.  





<p align="center">
  <img src="https://github.com/user-attachments/assets/9ea5f7e2-84d7-45e6-ba7d-b8d2d63ed976" width="80%">
</p>

🚀 *Powered by:* QLoRA, Jais-family-256m, Hugging Face Transformers  
🌍 *Language:* Arabic  
🏥 *Domain:* Healthcare & Medical AI  
---

## ⚙️ Prerequisites  

1. **Python 3.8+ 🐍** 
2. **Required Libraries** 📚:
   - **transformers** `ver4.31.0 Or newer` 
   - **accelerate** `ver0.21.0 Or newer`  
   - **bitsandbytes** `ver0.39.0 Or newer`
   - **scikit-learn** `ver1.3.0 Or newer` 
   - **sentencepiece** `ver0.1.99 Or newer`  
   - **pyyaml** `ver6.0 Or newer`
   - **numpy** `ver1.25.0 Or newer`

   You can install them with this line:
   ```bash
   pip install -r Requirements.txt
   ``` 
3. **📃 The Model**:  
   Download the pretrained **Jais-family-256m** model from [Hugging Face](https://huggingface.co/inceptionai/Jais-family-256m)

> **Note!:** At least **12GB of GPU memory** is recommended to speed up the process.

> **Note!:** The scripts requires CUDA-enabled GPUs to run efficiently. CUDA is critical for using the bitsandbytes library.  
---

## 📂 Dataset Preparation  

### Arabic Medical Dataset:  
- Includes **medical questions**, **categories**, and **answers**.  
- So far only **10%** of the data (~71,515 entries) is being used for baseline implementation.  

**Stucture**: three CSV files are used: **train.csv**, **valid.csv**, and **test.csv**, each containing:

- **Question**: the medical question asked by the user,
- **Category**: an associated category (e.g., الأمراض العصبية, الحمل والولادة, etc.),
- **Answer**: the ground-truth or reference response.
---

## 🚀 Running the Baseline with Qlora

### **BitsAndBytes 4-bit Quantization**  
We load the base model (**Jais-family-256m**) in 4-bit precision using **BitsAndBytes**. This significantly reduces VRAM usage without heavily sacrificing performance.

### **LoRA Adapters**  
- **Rank (r):** 8  
- **Alpha:** 32  
- **Dropout:** 0.1  
- **Target Modules:** `["c_attn", "c_proj", "c_fc", "c_fc2"]`  

These LoRA modules adapt only a small set of trainable parameters on top of the frozen base model layers, making fine-tuning highly memory efficient.

### **Prompt Engineering**  
For each training sample, we create a prompt of the form:  
```plaintext
سؤال: {Question}
التصنيف: {Category}
الإجابة:
```  
- The prompt and reference answer are tokenized separately.  
- The `input_ids` (prompt) serve as the input, and the tokenized answer serves as labels for causal language modeling.

### **Training Configuration**  
- **Epochs:** 1 (**due to limitation in resources**; can be increased)  
- **Batch Size:** 8 (can be tuned based on GPU memory)  
- **Learning Rate:** 1e-4  
- **Mixed Precision (FP16):** Enabled to further reduce memory usage  
- **Evaluation Strategy:** By epoch  
- **Best Model Selection:** Based on validation loss  

---

## 📊 Evaluation of the Model  

After fine-tuning, evaluate the model on dev and test sets:  
```bash
python score.py --model_path outputs/saved_model/
```  
- Replace `/path/to/saved/model` with the location of your trained model (e.g., `outputs/saved_model/`).  
- **Output:** Evaluation metrics (e.g., accuracy, F1 score) will be displayed in the terminal.  


## 🔵 BLEU
- **BLEU-1, BLEU-2, BLEU-4**  
- Calculated with `nltk.translate.bleu_score`.

## 🔴 ROUGE
- **ROUGE-1, ROUGE-2, ROUGE-L**  
- Uses `rouge_score.rouge_scorer`.  
- We report the **F-measure** of each.

## 🟠 BERTScore
- **Precision, Recall, F1**  
- Computed with `bert_score.score`.  
- For Arabic text, `lang='ar'` is used.

---

## 📊 Performance  
All of the results were tested on 1000 sample

| **Metric**       | **Baseline (10% Data)**<br>(Test, *After Cleaning*) | **Baseline (10% Data)**<br>(Valid, *After Cleaning*) | **Baseline (10% Data)**<br>(Test, *Before Cleaning*) | **Baseline (10% Data)**<br>(Valid, *Before Cleaning*) | **TF‑IDF Baseline (100% Data)**<br>(Test) |
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



---

## 📝 Notes  

1. **Common Issues:**  
   - **CUDA Out of Memory:** Reduce the batch size in `config.yaml`.  
   - **File Not Found:** Verify that datasets and models are in their correct locations.  

2. **Future Directions:**  
   - Experiment with different hyperparameters or larger datasets.  

3. **WIP:** As mentioned before, the project currantly only uses 10% of the data and is still a proof of concept.  

---

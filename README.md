
# 🩺 **SHEFAA Project**  

## 🌟 Overview  
**SHEFAA** is an Arabic medical question-answering system fine-tuned using **QLoRA** on the **Jais-family-256m** model.  
This project aims to advance Arabic NLP in healthcare by providing accurate responses to medical queries. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/9ea5f7e2-84d7-45e6-ba7d-b8d2d63ed976" width="50%">
</p>

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



---

## 📝 Notes  

1. **Common Issues:**  
   - **CUDA Out of Memory:** Reduce the batch size in `config.yaml`.  
   - **File Not Found:** Verify that datasets and models are in their correct locations.  

2. **Future Directions:**  
   - Experiment with different hyperparameters or larger datasets.  

3. **WIP:** As mentioned before, the project currantly only uses 10% of the data and is still a proof of concept.  

---

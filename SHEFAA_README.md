
# 🩺 **SHEFAA Project**  

## 🌟 Overview  
**SHEFAA** is an Arabic medical question-answering system fine-tuned using **QLoRA** on the **Jais-family-256m** model.  
This project aims to advance Arabic NLP in healthcare by providing accurate responses to medical queries.  

---

## ⚙️ Prerequisites  

1. **Python 3.8+**  
2. **Required Libraries**:  
   - `transformers==4.31.0` 🤗  
   - `accelerate==0.21.0`  
   - `bitsandbytes==0.39.0`  

> **Note:** At least **12GB of GPU memory** is recommended to speed up the process.  

3. **📂 Model Placement**:  
   Download the pretrained **Jais-family-256m** model from [Hugging Face](https://huggingface.co/) 

---

## 📂 Dataset Preparation  

### Arabic Medical Dataset:  
- Includes **medical questions**, **categories**, and **answers**.  
- Only **10%** of the data (~71,515 entries) is used for baseline implementation.  

## 📂 Placement
We assume three CSV files: **train.csv**, **valid.csv**, and **test.csv**, each containing:

- **Question**: the medical question asked by the user,
- **Category**: an associated category (e.g., الأمراض العصبية, الحمل والولادة, etc.),
- **Answer**: the ground-truth or reference response.
---

## 🧠 QLoRA Approach  

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
- **Epochs:** 1 (demonstration; can be increased)  
- **Batch Size:** 8 (can be tuned based on GPU memory)  
- **Learning Rate:** 1e-4  
- **Mixed Precision (FP16):** Enabled to further reduce memory usage  
- **Evaluation Strategy:** By epoch  
- **Best Model Selection:** Based on validation loss  

---

## 📊 Evaluating the Model  

After fine-tuning, evaluate the model using:  
```bash
python baseline.py --evaluate --model_path outputs/saved_model/
```  
- Replace `outputs/saved_model/` with the location of your fine-tuned model.  

---

## 📈 Expected Outputs  

1. **📂 Logs**:  
   - Training logs are saved in the `outputs/logs/` directory by default.  
   - Logs include information on training loss, evaluation metrics, and system performance.  

2. **📂 Checkpoints**:  
   - Fine-tuned model checkpoints are saved in the `outputs/checkpoints/` directory.  

3. **📂 Evaluation Results**:  
   - Metrics such as **F1 Score** and **Accuracy** are printed to the console.  
   - A summary of evaluation results is saved in `outputs/evaluation_results.txt`.  

---

## 📊 Performance  

- **Primary Metric:** [Insert metric, e.g., F1 Score]  
- **Results**:  
  - Dev Set: [Value]  
  - Test Set: [Value]  

🔄 **Comparison to Random Baseline**:  
  - Dev Set: [Value]  
  - Test Set: [Value]  
  - **Improvement:** [Details]  

---

## 📝 Notes  

1. **Common Issues:**  
   - **CUDA Out of Memory:** Reduce the batch size in `config.yaml`.  
   - **File Not Found:** Verify that datasets and models are in their correct locations.  

2. **Future Directions:**  
   - Experiment with different hyperparameters or larger datasets.  

3. **Contact:** Reach out at [Your Email].  

---

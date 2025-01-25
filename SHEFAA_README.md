
# 🩺 **SHEFAA Project**  

## 🌟 Overview  
**SHEFAA** is an Arabic medical question-answering system fine-tuned using **QLoRA** on the **Jais-family-256m** model.  
This project aims to advance Arabic NLP in healthcare by providing accurate responses to medical queries.

---

## ⚙️ Prerequisites  

### 🛠 Environment Setup  
1. **Python 3.8+**  
2. Install dependencies with:  
   ```bash
   pip install -r requirements.txt
   ```  
   **Key Libraries**:  
   - `PyTorch` 🟠  
   - `transformers` 🤗  
   - `accelerate`, `bitsandbytes`, and others  

3. **Hardware Requirements**:  
   - 🖥 **GPU:** 16GB+ memory (e.g., NVIDIA V100/A100)  
   - 🧠 **RAM:** 32GB+  

4. **Download the JAIS Model**:  
   Get the pretrained **Jais-family-256m** model from [Hugging Face](https://huggingface.co/) and place it in the appropriate directory.  

---

## 📂 Dataset Preparation  

1. **Arabic Medical Dataset**:  
   - Includes medical questions, categories, and answers.  
   - Only **10%** of the data (~71,515 entries) is used for baseline implementation.  

2. **Preprocessing**:  
   Run the following command to clean and split data:  
   ```bash
   python preprocess.py --data_path /path/to/dataset
   ```  

---

## 🚀 Running SHEFAA  

### 🏋️ Training the Model  
Fine-tune the model using QLoRA:  
```bash
python train.py --config config.yaml
```  
- **config.yaml** contains hyperparameters and file paths.  

### 📊 Evaluating the Model  
Evaluate the fine-tuned model on dev and test sets:  
```bash
python evaluate.py --model_path /path/to/saved/model
```  

---

## 📈 Performance  

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

1. **Scalability**: QLoRA enables memory-efficient fine-tuning.  
2. **Next Steps**: Explore additional datasets and architectures.  
3. **Contact**: Reach out at [Your Email].  

---

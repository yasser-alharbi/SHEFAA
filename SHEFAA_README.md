
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
   Get the pretrained **Jais-family-256m** model from [Hugging Face](https://huggingface.co/) and place it in a `models/` directory.  

---

## 📂 Dataset Preparation  

1. **Arabic Medical Dataset**:  
   - Includes medical questions, categories, and answers.  
   - Only **10%** of the data (~71,515 entries) is used for baseline implementation.  

2. **Preprocessing**:  
   Run the following command to clean and split the data:  
   ```bash
   python preprocess.py --data_path data/raw_dataset.csv --output_dir data/processed/
   ```  
   - Replace `/path/to/dataset` with the path to your dataset file (e.g., `data/raw_dataset.csv`).  
   - **Output:** Preprocessed files will be saved in `data/processed/`.  

---

## 🚀 Running SHEFAA  

### File Structure  
Ensure your project is structured like this:  
```
SHEFAA/
├── models/          # Pretrained Jais model
├── data/            # Dataset files
│   ├── raw_dataset.csv
│   ├── processed/
├── train.py         # Training script
├── evaluate.py      # Evaluation script
├── config.yaml      # Configuration file
```

### 🏋️ Training the Model  
To fine-tune the Jais-family-256m model using **QLoRA**, run:  
```bash
python train.py --config config.yaml
```  
- **config.yaml:** Modify this file to set paths, hyperparameters (e.g., batch size, learning rate), and output locations.  
- **Example Output:** Training logs and model checkpoints will be saved in `outputs/`.  

#### 🔧 QLoRA Fine-Tuning Details:  
1. **Efficient Memory Usage:** Uses **4-bit quantization** with `bitsandbytes` to significantly reduce GPU memory usage.  
2. **LoRA Adapters Configuration:**  
   - **Rank (r):** 8  
   - **Alpha:** 32  
   - **Dropout:** 0.1  
   - **Target Modules:** Fine-tunes specific layers (e.g., `c_attn`, `c_proj`).  

### 📊 Evaluating the Model  
After fine-tuning, evaluate the model on dev and test sets:  
```bash
python evaluate.py --model_path outputs/saved_model/
```  
- Replace `/path/to/saved/model` with the location of your trained model (e.g., `outputs/saved_model/`).  
- **Output:** Evaluation metrics (e.g., accuracy, F1 score) will be displayed in the terminal.  

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

1. **Troubleshooting Tips:**  
   - **CUDA Out of Memory:** Reduce batch size in `config.yaml`.  
   - **File Not Found:** Ensure datasets and models are in their correct directories.  

2. **Next Steps:**  
   - Fine-tune on additional datasets to improve performance.  
   - Experiment with different hyperparameters or architectures.  

3. **Contact:** Reach out at [Your Email].  

---

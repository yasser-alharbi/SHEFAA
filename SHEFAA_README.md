
# 🩺 **SHEFAA Project**  

## 🌟 Overview  
**SHEFAA** is an Arabic medical question-answering system fine-tuned using **QLoRA** on the **Jais-family-256m** model.  
This project aims to advance Arabic NLP in healthcare by providing accurate responses to medical queries.  

---

## ⚙️ Prerequisites  

1. **Python 3.8+**  
2. **Required Libraries**:  
   - `transformers ver4.31.0 Or newer`  
   - `accelerate ver0.21.0 Or newer`  
   - `bitsandbytes ver0.39.0 Or newer`  
🤗
> **Note:** At least **12GB of GPU memory** is recommended to speed up the process.  

3. **Download the Jais-family-256m Model**:  
   Download the pretrained model from [Hugging Face](https://huggingface.co/) and place it in the `models/` directory.  

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

## 🚀 Running the Baseline  

### 🏋️ Training the Model  
To fine-tune the Jais-family-256m model using **QLoRA**, run:  
```bash
python baseline.py --config config.yaml
```  
- **baseline.py**: This script handles the fine-tuning process.  
- **config.yaml**: Modify this file to set paths, hyperparameters (e.g., batch size, learning rate), and output locations.  

#### 🔧 QLoRA Fine-Tuning Details:  
1. **4-bit Quantization:** Reduces GPU memory usage by loading the model in 4-bit precision using `bitsandbytes`.  
2. **LoRA Adapters Configuration:**  
   - **Rank (r):** 8  
   - **Alpha:** 32  
   - **Dropout:** 0.1  

These configurations enable efficient fine-tuning with minimal memory overhead.  

### 📊 Evaluating the Model  
After fine-tuning, evaluate the model using:  
```bash
python baseline.py --evaluate --model_path outputs/saved_model/
```  
- Replace `outputs/saved_model/` with the location of your fine-tuned model.  

---

## 📈 Expected Outputs  

1. **Logs**:  
   - Training logs are saved in the `outputs/logs/` directory by default.  
   - Logs include information on training loss, evaluation metrics, and system performance.  

2. **Checkpoints**:  
   - Fine-tuned model checkpoints are saved in the `outputs/checkpoints/` directory.  

3. **Evaluation Results**:  
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

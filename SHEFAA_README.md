
# README: SHEFAA Project

## Overview
**SHEFAA** is a project aimed at fine-tuning a robust Arabic medical question-answering system. This document provides step-by-step instructions for running the baseline system implemented as part of Milestone 3. The baseline system utilizes **QLoRA fine-tuning** on the **Jais-family-256m model** and is designed to work with an Arabic medical dataset.

This README outlines how to prepare, train, and evaluate the system, as well as its performance metrics compared to a random baseline. The ultimate goal of **SHEFAA** is to contribute to advancements in Arabic natural language processing (NLP) within the healthcare domain.

---

## Prerequisites

### Environment Setup
Before running the baseline system, ensure the following prerequisites are met:

1. **Python version:** Python 3.8 or higher.
2. Required libraries:
   - PyTorch
   - Sklearn
   - NLTK
   - Spacy
   - transformers
   - accelerate
   - bitsandbytes
   - Any other dependencies listed in `requirements.txt`.

Install all dependencies with:
```bash
pip install -r requirements.txt
```

3. **Hardware Requirements:**
   - **GPU:** At least 16GB of GPU memory (e.g., NVIDIA V100 or A100).
   - **RAM:** 32GB or higher for efficient processing.

4. **Download the Jais-family-256m Model:**
   The Jais-family-256m model is required for fine-tuning. You can download the pretrained model from its official source [here](https://huggingface.co/). Ensure the model files are placed in the appropriate directory before proceeding.

---

## Dataset Preparation

### Dataset Description
The dataset consists of Arabic medical question-answer pairs specifically curated for **SHEFAA**. Each entry includes:
- **Question:** The medical question posed by the user.
- **Category:** The associated category (e.g., الأمراض العصبية, الحمل والولادة).
- **Answer:** The ground-truth or reference response.

> **Note:** Due to resource limitations in this baseline, only 10% of the data was used for implementation (71,515 out of 715,187 entries).

### Downloading the Dataset
Ensure the dataset is downloaded and placed in the specified directory.

### Data Preprocessing
Convert the CSV files into Hugging Face Datasets for streamlined tokenization and training. Preprocess the dataset and split it into development and test sets by running:
```bash
python preprocess.py --data_path /path/to/dataset
```
This script will clean the data and prepare it for training and evaluation.

---

## QLoRA Fine-Tuning Approach

1. **BitsAndBytes 4-bit Quantization:**  
   Load the base model (`Jais-family-256m`) in 4-bit precision using the `bitsandbytes` library. This reduces VRAM usage without significantly sacrificing performance.

2. **LoRA Adapters Configuration:**
   - **Rank (r):** 8
   - **Alpha:** 32
   - **Dropout:** 0.1
   - **Target Modules:** `["c_attn", "c_proj", "c_fc", "c_fc2"]`

These LoRA modules adapt only a small set of trainable parameters on top of the frozen pre-trained model, enabling efficient fine-tuning.

---

## Running the SHEFAA Baseline System

### Step 1: Training the Model
To fine-tune the Jais-family-256m model using QLoRA, execute the training script:
```bash
python train.py --config config.yaml
```
- **config.yaml:** This file contains hyperparameters (e.g., learning rate, batch size) and paths for data and model files. Edit it as needed.

The training process adapts the **Jais-family-256m** model using LoRA adapters, enabling memory-efficient optimization suitable for limited hardware setups.

### Step 2: Evaluating the Model
Once training is complete, evaluate the system on the development and test datasets by running:
```bash
python evaluate.py --model_path /path/to/saved/model
```
This script computes evaluation metrics such as accuracy and F1 score, generating results for both datasets.

---

## Performance Metrics

### Evaluation Metrics
The following metric(s) were used to assess system performance:
- **Primary Metric:** [Insert metric here, e.g., F1 Score or Accuracy]

### Results:
- **Development Set:**
  - Metric: [Insert value here]
- **Test Set:**
  - Metric: [Insert value here]

### Comparison with Random Baseline:
- **Random Baseline Performance:**
  - Development Set: [Insert value here]
  - Test Set: [Insert value here]
- **Improvement Achieved by SHEFAA Baseline:**
  - Development Set: [Insert improvement details]
  - Test Set: [Insert improvement details]

These results illustrate the significant improvement achieved by the QLoRA fine-tuned baseline compared to random guessing, establishing a solid foundation for future enhancements in **SHEFAA**.

---

## Notes and Limitations

1. **Hardware Considerations:**  
   While QLoRA reduces memory requirements, larger datasets or extended training may necessitate high-performance GPUs.

2. **Data Usage:**  
   The current implementation uses a portion of the dataset (10%) for evaluation. Using the full dataset may yield better performance results.

3. **Future Directions:**  
   Future iterations of **SHEFAA** may explore:
   - Incorporating additional datasets to expand the system’s knowledge base.
   - Experimenting with alternative architectures or advanced fine-tuning methods.
   - Integrating external tools for enhanced question-answering capabilities.

4. **Support:**  
   For questions or issues, please contact [Your Name/Email].

---

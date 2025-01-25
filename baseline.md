# Fine-Tuning Jais-family-256m with QLoRA (Baseline)

## Introduction
In this baseline, we fine-tune the **Jais-family-256m** model using **QLoRA** from the [PEFT](https://github.com/huggingface/peft) library. The model is loaded in **4-bit** precision through [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for memory efficiency.  
Our goal is to answer Arabic medical questions, leveraging LoRA adapters on top of Jais.

---

## Approach

1. **Data Loading & Splitting**  
   - We load three CSV files: **train.csv**, **valid.csv**, and **test.csv**, each containing columns:
     - `Question`
     - `Category`
     - `Answer`
   - We convert these CSVs into **Hugging Face Datasets** for easy tokenization and training.

2. **QLoRA Configuration**  
   - The base model (`Jais-family-256m`) is loaded in **4-bit** mode using `BitsAndBytesConfig`.
   - **LoRA** parameters are configured with:
     - `r = 8`
     - `lora_alpha = 32`
     - `lora_dropout = 0.1`
     - `target_modules = ["c_attn", "c_proj", "c_fc", "c_fc2"]`  
   - We then apply `get_peft_model()` to wrap the original model.

3. **Tokenization & Prompt Format**  
   - Each sample is tokenized with the prompt:  
     \[
       \text{"سؤال: {question}\nالتصنيف: {category}\nالإجابة:"}
     \]
   - The answer is separately tokenized as the label.

4. **Training**  
   - We use **1 epoch** for demonstration (can be increased).
   - Batch size is **8** (can be tuned).
   - **4-bit** precision plus LoRA drastically lowers GPU memory usage.
   - Model is evaluated each epoch on the validation set, saving the best checkpoint.

5. **Evaluation & Metrics**  
   - We generate predictions on a subset of the validation set and the test set (e.g., 100 examples) for quick checks.
   - Metrics computed:
     - **BLEU-1**, **BLEU-2**, **BLEU-4**
     - **ROUGE-1**, **ROUGE-2**, **ROUGE-L** (F-measure)
     - **BERTScore** (Precision, Recall, F1) for Arabic text.

---

## Code Snippet

Below is a concise version of how we perform the fine-tuning. For the full script, see **baseline.py**:
```python
# BitsAndBytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
).to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn","c_proj","c_fc","c_fc2"]
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="Jais-family-256m-lora-SHEFAA",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    ...
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)
trainer.train()

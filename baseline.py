import argparse
import pandas as pd
import numpy as np
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from tqdm import tqdm

# Hugging Face & PEFT
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes import BitsAndBytesConfig
# For Jais login (requires the Jais HF token in your environment)
from huggingface_hub import login

# Download NLTK tokenizer if needed
nltk.download("punkt")

def preprocess_hf_dataset(train_path, valid_path, test_path):
    """
    Reads CSV files, merges them into Hugging Face Datasets,
    and returns train_dataset, valid_dataset, test_dataset.
    """
    # Read data
    train_df_raw = pd.read_csv(train_path)
    valid_df_raw = pd.read_csv(valid_path)
    test_df_raw = pd.read_csv(test_path)
    
    # Convert them to HF Datasets
    train_dataset = Dataset.from_pandas(train_df_raw)
    valid_dataset = Dataset.from_pandas(valid_df_raw)
    test_dataset = Dataset.from_pandas(test_df_raw)
    
    print("Train size:", len(train_dataset))
    print("Valid size:", len(valid_dataset))
    print("Test size:", len(test_dataset))
    return train_dataset, valid_dataset, test_dataset

def preprocess_function(examples, tokenizer):
    """
    Custom function to tokenize and prepare input_ids and labels
    for causal language modeling tasks.
    """
    questions = examples["Question"]
    categories = examples["Category"]
    answers = examples["Answer"]  # or "answer", depending on CSV column name
    
    input_ids = []
    labels = []

    for q, cat, ans in zip(questions, categories, answers):
        # Prompt format
        prompt = f"سؤال: {q}\nالتصنيف: {cat}\nالإجابة:"
        
        # Tokenize prompt
        prompt_encodings = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        # Tokenize answer
        label_encodings = tokenizer(
            ans,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids.append(prompt_encodings["input_ids"].squeeze(0).tolist())
        labels.append(label_encodings["input_ids"].squeeze(0).tolist())

    return {
        "input_ids": input_ids,
        "labels": labels
    }

def generate_from_dataset(dataset, model, tokenizer, device, max_new_tokens=100):
    """
    Generates predictions (answers) for each question/category pair in 'dataset'.
    """
    predictions = []
    for q, cat in tqdm(zip(dataset["Question"], dataset["Category"]),
                       total=len(dataset["Question"]),
                       desc="Generating answers"):
        prompt = f"سؤال: {q}\nالتصنيف: {cat}\nالإجابة:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated_text)
    return predictions

def extract_answers_from_text(generated_lines, cutoff=1000):
    """
    Extracts answers after the substring 'الإجابة:'.
    Returns a list of answers truncated to 'cutoff' number of answers.
    """
    answers = []
    for line in generated_lines:
        line = line.strip()
        if "الإجابة:" in line:
            parts = line.split("الإجابة:", 1)
            answer = parts[1].strip()
            answers.append(answer)
        if len(answers) == cutoff:
            break
    return answers

def compute_bleu_scores(predictions, references):
    """
    Computes sentence-level BLEU scores (BLEU-1, BLEU-2, BLEU-4)
    for each prediction-reference pair and returns a dict of lists.
    """
    smoothie = SmoothingFunction().method4
    bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_4": []}

    for hyp, ref in zip(predictions, references):
        hyp_tokens = nltk.word_tokenize(hyp)
        ref_tokens = nltk.word_tokenize(ref)
        # BLEU-1
        bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        # BLEU-2
        bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        # BLEU-4
        bleu_4 = sentence_bleu([ref_tokens], hyp_tokens,
                               weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=smoothie)
        
        bleu_scores["bleu_1"].append(bleu_1)
        bleu_scores["bleu_2"].append(bleu_2)
        bleu_scores["bleu_4"].append(bleu_4)
    
    return bleu_scores

def compute_rouge_scores(predictions, references):
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L for each prediction-reference pair.
    Returns a list of dicts with these scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    for hyp, ref in zip(predictions, references):
        scores = scorer.score(ref, hyp)
        results.append(scores)
    return results

def compute_bertscore(predictions, references, lang="ar"):
    """
    Computes BERTScore Precision, Recall, and F1 for each prediction-reference pair.
    Returns three lists (P, R, F1).
    """
    P, R, F1 = score(predictions, references, lang=lang)
    return P, R, F1

def main(args):
    # 1. Login to Hugging Face if needed (requires valid token)
    if args.hf_token:
        login(token=args.hf_token)

    # 2. Prepare Datasets
    train_dataset, valid_dataset, test_dataset = preprocess_hf_dataset(
        args.train, args.valid, args.test
    )
    
    # 3. Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 4. BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",  # or "bfloat16" if your GPU supports BF16
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 5. Load base model & tokenizer
    model_path = "inceptionai/Jais-family-256m"  # Example model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure we have a padding token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    ).to(device)
    
    # 6. PEFT LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj", "c_fc", "c_fc2"]
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # 7. Preprocess (tokenize, create input_ids & labels) for each dataset
    train_dataset = train_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True
    )
    valid_dataset = valid_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True
    )
    
    # 8. TrainingArguments
    training_args = TrainingArguments(
        output_dir="Jais-family-256m-lora-SHEFAA",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=1e-4,
        fp16=True,
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none"  # disable W&B and TensorBoard
    )
    
    # 9. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    # 10. Train model
    trainer.train()
    
    # 11. Evaluate on validation & test sets (loss)
    valid_loss = trainer.evaluate(valid_dataset)
    test_loss = trainer.evaluate(test_dataset)
    print("Validation Loss:", valid_loss["eval_loss"])
    print("Test Loss:", test_loss["eval_loss"])
    
    # 12. Generate predictions on small subset for evaluation
    model.eval()
    max_examples = 1000  # for demonstration
    
    print("\nGenerating Predictions on Validation Subset...")
    y_pred_valid_raw = generate_from_dataset(
        valid_dataset[:max_examples], model, tokenizer, device
    )
    with open("y_pred_valid.txt", "w", encoding="utf-8") as f_valid:
        for pred in y_pred_valid_raw:
            f_valid.write(str(pred) + "\n")
    
    print("Extracting validation answers...")
    y_pred_valid = extract_answers_from_text(y_pred_valid_raw, cutoff=max_examples)
    y_valid = valid_dataset["Answer"][:max_examples]
    
    # Show one example
    print("\n[Validation Example]")
    print("Question:", valid_dataset["Question"][0])
    print("Generated answer:", y_pred_valid[0])
    print("Reference answer:", y_valid[0])
    
    print("\nGenerating Predictions on Test Subset...")
    y_pred_test_raw = generate_from_dataset(
        test_dataset[:max_examples], model, tokenizer, device
    )
    with open("y_pred_test.txt", "w", encoding="utf-8") as f_test:
        for pred in y_pred_test_raw:
            f_test.write(str(pred) + "\n")
    
    print("Extracting test answers...")
    y_pred_test = extract_answers_from_text(y_pred_test_raw, cutoff=max_examples)
    y_test = test_dataset["Answer"][:max_examples]
    
    # Show one test example
    print("\n[Test Example]")
    print("Question:", test_dataset["Question"][2])
    print("Generated answer:", y_pred_test[2])
    print("Reference answer:", y_test[2])
    
    # 13. Compute BLEU
    print("\nComputing BLEU scores on the Test Subset...")
    bleu_scores = compute_bleu_scores(y_pred_test, y_test)
    bleu_1_avg = np.mean(bleu_scores["bleu_1"])
    bleu_2_avg = np.mean(bleu_scores["bleu_2"])
    bleu_4_avg = np.mean(bleu_scores["bleu_4"])
    print(f"Average BLEU-1: {bleu_1_avg:.3f}")
    print(f"Average BLEU-2: {bleu_2_avg:.3f}")
    print(f"Average BLEU-4: {bleu_4_avg:.3f}")
    
    # 14. Compute ROUGE
    print("\nComputing ROUGE scores on the Test Subset...")
    rouge_results = compute_rouge_scores(y_pred_test, y_test)
    avg_rouge1 = np.mean([r['rouge1'].fmeasure for r in rouge_results])
    avg_rouge2 = np.mean([r['rouge2'].fmeasure for r in rouge_results])
    avg_rougeL = np.mean([r['rougeL'].fmeasure for r in rouge_results])
    print(f"Average ROUGE-1 F1: {avg_rouge1:.3f}")
    print(f"Average ROUGE-2 F1: {avg_rouge2:.3f}")
    print(f"Average ROUGE-L F1: {avg_rougeL:.3f}")
    
    # 15. Compute BERTScore
    print("\nComputing BERTScore on the Test Subset...")
    P, R, F1 = compute_bertscore(y_pred_test, y_test, lang='ar')
    print(f"Average BERTScore Precision: {P.mean() * 100:.2f}%")
    print(f"Average BERTScore Recall: {R.mean() * 100:.2f}%")
    print(f"Average BERTScore F1: {F1.mean() * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Jais-family-256m using QLoRA and evaluate.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset CSV.")
    parser.add_argument("--valid", type=str, required=True, help="Path to the validation dataset CSV.")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset CSV.")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face token for private model access.")
    args = parser.parse_args()
    main(args)

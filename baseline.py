import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
#from peft import BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_from_dataset(dataset, model, tokenizer, max_new_tokens=100):
    """
    Generates answers for each Question/Category entry in `dataset`.
    
    Expects dataset columns: "Question", "Category".
    Returns a list of predicted answers (strings).
    """
    predictions = []

    for q, cat in tqdm(
        zip(dataset["Question"], dataset["Category"]),
        total=len(dataset["Question"]),
        desc="Generating answers"
    ):
        # Construct the same style of prompt used in training (Arabic)
        prompt = f"سؤال: {q}\nالتصنيف: {cat}\nالإجابة:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract text after "الإجابة:"
        if "الإجابة:" in generated_text:
            # Split on the first occurrence of "الإجابة:"
            parts = generated_text.split("الإجابة:", 1)
            answer = parts[1].strip()
            predictions.append(answer)
        else:
            # If no "الإجابة:" is found, store entire generation
            predictions.append(generated_text)

    return predictions


def main(csv_file):
    """
    Main routine:
      1) Load CSV into a Hugging Face Dataset
      2) Load the fine-tuned QLoRA model
      3) Generate predictions
      4) Write references and predictions to text files
    """
    #---------------------------------------------------------------------
    # 1. Load CSV
    #    We assume your CSV has columns: "Question", "Category", "Answer"
    #---------------------------------------------------------------------
    dataset = load_dataset("csv", data_files={"test": csv_file})["test"]
    
    # If your CSV has a ground-truth "Answer" column, use it as references.
    # Otherwise, remove or adjust accordingly.
    references = dataset["Answer"]
    
    #---------------------------------------------------------------------
    # 2. Load the fine-tuned QLoRA model
    #---------------------------------------------------------------------
    saved_model_path = "/workspaces/SHEFAA/SHEFAA_Jais_qlora"
    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        saved_model_path,
        quantization_config=bnb_config,
        device_map="auto",   # For multi-GPU or single GPU auto placement
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path, trust_remote_code=True)

    # Make sure model is in evaluation mode
    model.eval()

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # (Optional) Merge LoRA weights into the base model
    # If you want to directly generate with the merged weights, do:
    merged_model = model.merge_and_unload()

    #---------------------------------------------------------------------
    # 3. Generate predictions using the function
    #---------------------------------------------------------------------
    y_pred = generate_from_dataset(dataset, merged_model, tokenizer)

    #---------------------------------------------------------------------
    # 4. Write references and predictions to .txt files
    #---------------------------------------------------------------------
    with open("references.txt", "w", encoding="utf-8") as ref_file:
        for ref in references:
            ref_file.write(str(ref) + "\n")

    with open("predictions.txt", "w", encoding="utf-8") as pred_file:
        for pred in y_pred:
            pred_file.write(str(pred) + "\n")

    print("Done! Wrote 'references.txt' and 'predictions.txt'.")


if __name__ == "__main__":
    # Example usage:
    # python my_inference_script.py
    #
    # Make sure "valid_or_test_dataset.csv" is in the same directory or specify a path.
    main("/workspaces/SHEFAA/Dataset & Samples/sample_test.csv")

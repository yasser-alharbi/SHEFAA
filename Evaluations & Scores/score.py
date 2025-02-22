import nltk
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

# Function to read text files
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# Compute BLEU Scores
def compute_bleu_scores(references, predictions):
    smoothie = SmoothingFunction().method4  # Smooth to avoid zero scores for short texts
    bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_4": []}  # Store scores for each level

    for ref, hyp in zip(references, predictions):
        # Tokenize
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        # Compute BLEU scores for different n-gram levels
        bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        # Append scores
        bleu_scores["bleu_1"].append(bleu_1)
        bleu_scores["bleu_2"].append(bleu_2)
        bleu_scores["bleu_4"].append(bleu_4)

    return bleu_scores

# Compute ROUGE Scores
def compute_rouge_scores(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, hyp in zip(references, predictions):
        scores = scorer.score(ref, hyp)
        results["rouge1"].append(scores['rouge1'].fmeasure)
        results["rouge2"].append(scores['rouge2'].fmeasure)
        results["rougeL"].append(scores['rougeL'].fmeasure)
    return results

# Compute BERTScore
def compute_bertscore(references, predictions, lang="ar"):
    # precision, recall, F1
    P, R, F1 = score(predictions, references, lang=lang)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

# Main evaluation function
def evaluate(predictions_file, references_file):
    # Read the files
    predictions = read_file(predictions_file)
    references = read_file(references_file)

    # Compute BLEU scores
    print("Computing BLEU Scores...")
    bleu_scores = compute_bleu_scores(references, predictions)
    print("BLEU-1:", sum(bleu_scores["bleu_1"]) / len(bleu_scores["bleu_1"]))
    print("BLEU-2:", sum(bleu_scores["bleu_2"]) / len(bleu_scores["bleu_2"]))
    print("BLEU-4:", sum(bleu_scores["bleu_4"]) / len(bleu_scores["bleu_4"]))

    # Compute ROUGE scores
    print("\nComputing ROUGE Scores...")
    rouge_scores = compute_rouge_scores(references, predictions)
    print("ROUGE-1 (F1):", sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]))
    print("ROUGE-2 (F1):", sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]))
    print("ROUGE-L (F1):", sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]))

    # Compute BERTScore
    print("\nComputing BERTScore...")
    bert_scores = compute_bertscore(references, predictions)
    print("BERTScore Precision:", bert_scores["precision"])
    print("BERTScore Recall:", bert_scores["recall"])
    print("BERTScore F1:", bert_scores["f1"])

if __name__ == "__main__":
    import argparse

    # Command line argument parser
    parser = argparse.ArgumentParser(description="Evaluate predictions against references using BLEU, ROUGE, and BERTScore.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the predictions file.")
    parser.add_argument("--references", type=str, required=True, help="Path to the references file.")
    args = parser.parse_args()

    # Run evaluation
    evaluate(args.predictions, args.references)

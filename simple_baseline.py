import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

nltk.download('punkt')

# Clean and preprocess data
def clean_data(df):
    df['Question'] = df['Question'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
    df['Answer'] = df['Answer'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
    return df

# Build TF-IDF model and predict answers
def generate_predictions(train_questions, train_answers, questions):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_questions)
    
    predictions = []
    for question in questions:
        question_tfidf = vectorizer.transform([question])
        similarities = cosine_similarity(question_tfidf, X_train_tfidf)
        best_index = np.argmax(similarities)
        predictions.append(train_answers.iloc[best_index])
    return predictions

# Compute evaluation metrics
def evaluate(predictions, references):
    # BLEU
    smoothie = SmoothingFunction().method4
    bleu_scores = {
        "BLEU-1": [],
        "BLEU-2": [],
        "BLEU-4": []
    }
    for ref, pred in zip(references, predictions):
        ref_tokens = nltk.word_tokenize(ref)
        pred_tokens = nltk.word_tokenize(pred)
        bleu_scores["BLEU-1"].append(sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu_scores["BLEU-2"].append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu_scores["BLEU-4"].append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
    
    # ROUGE
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}
    for ref, pred in zip(references, predictions):
        scores = rouge_scorer_obj.score(ref, pred)
        rouge_scores["ROUGE-1"].append(scores['rouge1'].fmeasure)
        rouge_scores["ROUGE-2"].append(scores['rouge2'].fmeasure)
        rouge_scores["ROUGE-L"].append(scores['rougeL'].fmeasure)
    
    # BERTScore
    P, R, F1 = score(predictions, references, lang='ar')
    bert_scores = {
        "BERTScore Precision": P.mean().item(),
        "BERTScore Recall": R.mean().item(),
        "BERTScore F1": F1.mean().item()
    }

    return bleu_scores, rouge_scores, bert_scores

def main(args):
    # Load datasets
    train_df = pd.read_csv(args.train)
    valid_df = pd.read_csv(args.valid)
    test_df = pd.read_csv(args.test)
    
    # Clean datasets
    train_df = clean_data(train_df)
    valid_df = clean_data(valid_df)
    test_df = clean_data(test_df)
    
    # Generate predictions for validation and test sets
    valid_predictions = generate_predictions(train_df['Question'], train_df['Answer'], valid_df['Question'])
    test_predictions = generate_predictions(train_df['Question'], train_df['Answer'], test_df['Question'])
    
    # Evaluate predictions on validation and test sets
    print("Evaluating Validation Set:")
    valid_bleu, valid_rouge, valid_bert = evaluate(valid_predictions, valid_df['Answer'])
    print("Validation BLEU Scores:", {k: np.mean(v) for k, v in valid_bleu.items()})
    print("Validation ROUGE Scores:", {k: np.mean(v) for k, v in valid_rouge.items()})
    print("Validation BERTScore:", valid_bert)

    print("\nEvaluating Test Set:")
    test_bleu, test_rouge, test_bert = evaluate(test_predictions, test_df['Answer'])
    print("Test BLEU Scores:", {k: np.mean(v) for k, v in test_bleu.items()})
    print("Test ROUGE Scores:", {k: np.mean(v) for k, v in test_rouge.items()})
    print("Test BERTScore:", test_bert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline model with BLEU, ROUGE, and BERTScore on validation and test datasets.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--valid", type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset.")
    args = parser.parse_args()
    main(args)

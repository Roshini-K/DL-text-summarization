"""
Evaluation utilities for text summarization.
Includes metrics calculation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any
import json

from .utils import timer

def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores for summarization.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    # Calculate averages
    avg_scores = {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL'])
    }
    
    return avg_scores

def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score for summarization.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        
    Returns:
        Average BLEU score
    """
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]
        
        # Calculate BLEU
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)

@timer
def generate_summaries(model, tokenizer, articles: List[str], max_len: int, device: str) -> List[str]:
    """
    Generate summaries for a list of articles.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        articles: List of articles to summarize
        max_len: Maximum length for generated summary
        device: Device to run inference on
        
    Returns:
        List of generated summaries
    """
    model.eval()
    summaries = []
    
    for article in articles:
        # Tokenize input
        inputs = tokenizer(
            article,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Generate summary
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_len // 2,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return summaries

def evaluate_model(model, tokenizer, test_articles, test_summaries, max_len, device):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_articles: Test articles
        test_summaries: Test reference summaries
        max_len: Maximum sequence length
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate predictions
    predictions = generate_summaries(model, tokenizer, test_articles, max_len, device)
    
    # Calculate metrics
    rouge_scores = calculate_rouge(predictions, test_summaries)
    bleu_score = calculate_bleu(predictions, test_summaries)
    
    # Combine results
    results = {
        'rouge': rouge_scores,
        'bleu': bleu_score,
        'predictions': predictions,
        'references': test_summaries
    }
    
    return results

def save_results(results: Dict[str, Any], save_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        save_path: Path to save results
        
    Returns:
        None
    """
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if key in ['predictions', 'references']:
            serializable_results[key] = value
        elif isinstance(value, dict):
            serializable_results[key] = {k: float(v) for k, v in value.items()}
        else:
            serializable_results[key] = float(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"Results saved to {save_path}")

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()
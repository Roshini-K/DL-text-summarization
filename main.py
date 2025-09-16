"""
Main script for text summarization project.
Orchestrates data loading, preprocessing, training, and evaluation.
"""

import argparse
from sklearn.model_selection import train_test_split

from src.data_loading import (
    extract_tgz_file, load_dataset, filter_short_summaries, create_dataframe
)

# from data_loading import load_dataset
from src.preprocessing import clean_texts, analyze_text_lengths

from src.model_training import (
    initialize_model, create_data_loaders, train_model, save_model, load_model
)
from src.evaluation import evaluate_model, save_results, plot_training_history
from src.utils import setup_device, set_seed
from src.config import (
    TGZ_FILE_PATH, EXTRACT_DIR, DATA_PATH, MIN_SUMMARY_LENGTH,
    MAX_LEN, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, MODEL_DIR, RESULTS_DIR
)

def main():
    """Main function to run the text summarization pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text Summarization Pipeline')
    parser.add_argument('--extract', action='store_true', help='Extract data from tgz file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = setup_device()
    
    # Extract data if requested
    if args.extract:
        print("Extracting data...")
        extract_tgz_file(TGZ_FILE_PATH, EXTRACT_DIR)
    
    # Load and preprocess data
    print("Loading dataset...")
    articles, highlights = load_dataset(DATA_PATH)
    
    print("Filtering short summaries...")
    articles, highlights = filter_short_summaries(articles, highlights, MIN_SUMMARY_LENGTH)
    
    # Limit dataset size for development
    articles = articles[:10000]
    highlights = highlights[:10000]
    
    print("Cleaning texts...")
    cleaned_articles, cleaned_highlights = clean_texts(articles, highlights)
    
    # Analyze text lengths
    length_stats = analyze_text_lengths(cleaned_articles, cleaned_highlights)
    print(f"Mean article length: {length_stats['mean_article_length']:.2f}")
    print(f"Mean summary length: {length_stats['mean_summary_length']:.2f}")
    
    # Split data
    print("Splitting data...")
    train_articles, temp_articles, train_highlights, temp_highlights = train_test_split(
        cleaned_articles, cleaned_highlights, test_size=0.2, random_state=args.seed
    )
    
    val_articles, test_articles, val_highlights, test_highlights = train_test_split(
        temp_articles, temp_highlights, test_size=0.5, random_state=args.seed
    )
    
    print(f"Train set size: {len(train_articles)}")
    print(f"Validation set size: {len(val_articles)}")
    print(f"Test set size: {len(test_articles)}")
    
    # Train model if requested
    if args.train:
        print("Initializing model...")
        model, tokenizer = initialize_model()
        
        print("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            train_articles, train_highlights, val_articles, val_highlights,
            tokenizer, MAX_LEN, BATCH_SIZE
        )
        
        print("Training model...")
        trained_model, history = train_model(
            model, train_loader, val_loader, LEARNING_RATE, NUM_EPOCHS, device
        )
        
        print("Saving model...")
        save_model(trained_model, tokenizer, MODEL_DIR / "text_summarization_model")
        
        # Plot training history
        plot_training_history(history, RESULTS_DIR / "training_history.png")
    
    # Evaluate model if requested
    if args.evaluate:
        print("Loading model for evaluation...")
        model, tokenizer = initialize_model()
        model_path = MODEL_DIR / "text_summarization_model"
        
        if model_path.exists():
            model, tokenizer = load_model(model_path, device)
            
            print("Evaluating model...")
            results = evaluate_model(
                model, tokenizer, test_articles, test_highlights, MAX_LEN, device
            )
            
            print("Evaluation results:")
            print(f"ROUGE-1: {results['rouge']['rouge1']:.4f}")
            print(f"ROUGE-2: {results['rouge']['rouge2']:.4f}")
            print(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
            print(f"BLEU: {results['bleu']:.4f}")
            
            print("Saving results...")
            save_results(results, RESULTS_DIR / "evaluation_results.json")
        else:
            print("Model not found. Please train the model first with --train flag.")

if __name__ == "__main__":
    main()
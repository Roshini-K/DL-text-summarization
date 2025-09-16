"""
Text preprocessing utilities for text summarization.
Includes cleaning, tokenization, and dataset preparation.
"""

import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Tuple
from collections import Counter

from .config import STOPWORD_MAPPING
from .utils import timer

# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources."""
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_resources()

# Get stopwords
stop_words = set(stopwords.words('english'))

def text_cleaner(text: str) -> str:
    """
    Clean text by removing HTML, special characters, and normalizing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    text_lowercased = text.lower()
    text_cleaned = BeautifulSoup(text_lowercased, "lxml").text
    text_cleaned = re.sub(r'\([^)]*\)', '', text_cleaned)
    text_cleaned = re.sub('"', '', text_cleaned)
    text_cleaned = ' '.join([STOPWORD_MAPPING[t] if t in STOPWORD_MAPPING else t 
                           for t in text_cleaned.split(" ")])
    text_cleaned = re.sub(r"'s\b", "", text_cleaned)
    text_cleaned = re.sub("[^a-zA-Z]", " ", text_cleaned)
    text_cleaned = re.sub('[m]{2,}', 'mm', text_cleaned)
    return text_cleaned

def articles_clean(article: str) -> str:
    """
    Clean article text by removing stopwords and short words.
    
    Args:
        article: Article text to clean
        
    Returns:
        Cleaned article text
    """
    clean_article = text_cleaner(article)
    word_tokens = [w for w in clean_article.split() if w not in stop_words]
    article_words = [i for i in word_tokens if len(i) > 1]
    return (" ".join(article_words)).strip()

def summary_clean(summary: str) -> str:
    """
    Clean summary text (keeps stopwords for better summary quality).
    
    Args:
        summary: Summary text to clean
        
    Returns:
        Cleaned summary text
    """
    clean_summary = text_cleaner(summary)
    word_tokens = clean_summary.split()
    summary_words = [i for i in word_tokens if len(i) > 1]
    return (" ".join(summary_words)).strip()

@timer
def clean_texts(articles: List[str], highlights: List[str]) -> Tuple[List[str], List[str]]:
    """
    Clean both articles and summaries.
    
    Args:
        articles: List of raw article texts
        highlights: List of raw summary texts
        
    Returns:
        Tuple of (cleaned_articles, cleaned_highlights)
    """
    cleaned_articles = [articles_clean(t) for t in articles]
    cleaned_highlights = [summary_clean(t) for t in highlights]
    
    return cleaned_articles, cleaned_highlights

def tokenize_texts(texts: List[str]) -> List[List[str]]:
    """
    Tokenize a list of texts.
    
    Args:
        texts: List of texts to tokenize
        
    Returns:
        List of tokenized texts
    """
    return [word_tokenize(text) for text in texts]

def build_vocab(tokenized_texts: List[List[str]], min_freq: int = 2) -> Tuple[dict, dict]:
    """
    Build vocabulary from tokenized texts.
    
    Args:
        tokenized_texts: List of tokenized texts
        min_freq: Minimum frequency for a word to be included in vocabulary
        
    Returns:
        Tuple of (word_to_idx, idx_to_word) dictionaries
    """
    # Flatten list of tokens
    all_tokens = [token for tokens in tokenized_texts for token in tokens]
    
    # Count token frequencies
    vocab = Counter(all_tokens)
    
    # Filter by minimum frequency
    vocab = {word: count for word, count in vocab.items() if count >= min_freq}
    
    # Create word to index mapping
    word_to_idx = {word: idx + 2 for idx, word in enumerate(vocab.keys())}  # +2 for pad and unk
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    
    # Create index to word mapping
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word

def analyze_text_lengths(articles: List[str], summaries: List[str]) -> dict:
    """
    Analyze text lengths for determining model parameters.
    
    Args:
        articles: List of article texts
        summaries: List of summary texts
        
    Returns:
        Dictionary with length statistics
    """
    article_lengths = [len(text.split()) for text in articles]
    summary_lengths = [len(text.split()) for text in summaries]
    
    return {
        'mean_article_length': sum(article_lengths) / len(article_lengths),
        'mean_summary_length': sum(summary_lengths) / len(summary_lengths),
        'max_article_length': max(article_lengths),
        'max_summary_length': max(summary_lengths),
        'article_lengths': article_lengths,
        'summary_lengths': summary_lengths
    }
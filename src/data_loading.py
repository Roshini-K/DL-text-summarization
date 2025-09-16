"""
Data loading utilities for text summarization project.
Handles extraction, reading, and dataset preparation.
"""

import tarfile
import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from .config import MIN_SUMMARY_LENGTH

from .config import TGZ_FILE_PATH, EXTRACT_DIR, DATA_PATH
from .utils import timer

@timer
def extract_tgz_file(tgz_file_path: str = TGZ_FILE_PATH, 
                    extract_dir: str = EXTRACT_DIR) -> None:
    """
    Extract .tgz file containing the dataset.
    
    Args:
        tgz_file_path: Path to the .tgz file
        extract_dir: Directory to extract files to
        
    Returns:
        None
    """
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(tgz_file_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    
    print("Extraction completed!")

def read_story_file(file_path: str) -> Tuple[str, str]:
    """
    Read a .story file and extract the story and highlights.
    
    Args:
        file_path: Path to the .story file
        
    Returns:
        Tuple of (story, highlights)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    story_lines = []
    highlight_lines = []
    reading_highlights = False
    
    for line in lines:
        if line.strip() == "@highlight":
            reading_highlights = True
            continue
        if reading_highlights:
            highlight_lines.append(line.strip())
        else:
            story_lines.append(line.strip())

    story = ' '.join(story_lines)
    highlights = ' '.join(highlight_lines)
    return story, highlights

@timer
def load_dataset(data_path: str = DATA_PATH) -> Tuple[List[str], List[str]]:
    """
    Load all story files from a directory.
    
    Args:
        data_path: Path to directory containing .story files
        
    Returns:
        Tuple of (articles, highlights) lists
    """
    articles = []
    highlights = []
    
    for filename in os.listdir(data_path):
        if filename.endswith('.story'):
            story, highlight = read_story_file(os.path.join(data_path, filename))
            articles.append(story)
            highlights.append(highlight)
    
    return articles, highlights

def filter_short_summaries(articles: List[str], highlights: List[str], 
                          min_length: int = MIN_SUMMARY_LENGTH) -> Tuple[List[str], List[str]]:
    """
    Filter out articles with summaries shorter than min_length.
    
    Args:
        articles: List of article texts
        highlights: List of summary texts
        min_length: Minimum number of words in summary
        
    Returns:
        Filtered tuple of (articles, highlights)
    """
    filtered_articles = []
    filtered_highlights = []
    
    for article, highlight in zip(articles, highlights):
        if len(highlight.split()) >= min_length:
            filtered_articles.append(article)
            filtered_highlights.append(highlight)
    
    return filtered_articles, filtered_highlights

def create_dataframe(articles: List[str], highlights: List[str]) -> pd.DataFrame:
    """
    Create a DataFrame from articles and highlights.
    
    Args:
        articles: List of article texts
        highlights: List of summary texts
        
    Returns:
        DataFrame with 'article' and 'summary' columns
    """
    return pd.DataFrame({
        'article': articles,
        'summary': highlights
    })
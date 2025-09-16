import pytest
from preprocessing import text_cleaner, articles_clean, clean_texts


def test_text_cleaner():
    """Test basic text cleaning functionality."""
    dirty_text = "Hello <html>World</html>! (ignore this) 's test"
    cleaned = text_cleaner(dirty_text)
    assert "html" not in cleaned
    assert "ignore this" not in cleaned
    assert "'s" not in cleaned
    assert cleaned.islower()


def test_clean_texts():
    """Test cleaning multiple texts."""
    articles = ["Test article <html>content</html>", "Another test"]
    summaries = ["Test summary", "Another summary"]
    
    cleaned_articles, cleaned_summaries = clean_texts(articles, summaries)
    
    assert len(cleaned_articles) == 2
    assert len(cleaned_summaries) == 2
    assert "html" not in cleaned_articles[0]
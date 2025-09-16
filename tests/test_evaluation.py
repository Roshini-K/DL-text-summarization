import pytest
from unittest.mock import Mock
from evaluation import calculate_rouge, calculate_bleu, evaluate_model

def test_calculate_rouge():
    """Test ROUGE score calculation."""
    reference = "This is a test summary"
    candidate = "This is a test summary"
    
    scores = calculate_rouge([reference], [candidate])
    
    assert 'rouge1' in scores
    assert 'rouge2' in scores
    assert 'rougeL' in scores
    assert scores['rouge1'].fmeasure > 0.9

def test_calculate_bleu():
    """Test BLEU score calculation."""
    references = [["This is a test summary"]]
    candidate = ["This is a test summary"]
    
    score = calculate_bleu(references, candidate)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
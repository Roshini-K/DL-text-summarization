import pytest
from unittest.mock import Mock, patch
from data_loading import load_dataset, preprocess_dataset, split_dataset

@patch('data_loading.datasets.load_dataset')
def test_load_dataset(mock_load):
    """Test dataset loading."""
    mock_load.return_value = {
        'train': [{'article': 'test1', 'highlights': 'summary1'}],
        'test': [{'article': 'test2', 'highlights': 'summary2'}]
    }
    
    train_data, test_data = load_dataset("cnn_dailymail", "3.0.0")
    
    assert len(train_data) == 1
    assert len(test_data) == 1
    assert 'article' in train_data[0]
    assert 'highlights' in train_data[0]

def test_split_dataset():
    """Test dataset splitting."""
    data = [{'article': f'test{i}', 'highlights': f'summary{i}'} for i in range(100)]
    
    train, val, test = split_dataset(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    assert len(train) == 70
    assert len(val) == 20
    assert len(test) == 10
    assert len(train) + len(val) + len(test) == 100
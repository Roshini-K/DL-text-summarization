from unittest.mock import Mock, patch
import torch
from model_training import TextSummarizationDataset, initialize_model


def test_text_summarization_dataset():
    """Test dataset creation and item retrieval."""
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    dataset = TextSummarizationDataset(
        articles=["test article"],
        summaries=["test summary"],
        tokenizer=mock_tokenizer,
        max_len=50
    )
    
    assert len(dataset) == 1
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item


@patch('model_training.AutoTokenizer.from_pretrained')
@patch('model_training.AutoModelForSeq2SeqLM.from_pretrained')
def test_initialize_model(mock_model, mock_tokenizer):
    """Test model initialization."""
    mock_tokenizer.return_value = "mock_tokenizer"
    mock_model.return_value = "mock_model"
    
    model, tokenizer = initialize_model("test-model")
    
    assert model == "mock_model"
    assert tokenizer == "mock_tokenizer"
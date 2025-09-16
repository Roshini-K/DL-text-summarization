"""
Model training utilities for text summarization.
Includes model definition, training loop, and saving/loading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from .config import PRETRAINED_MODELS, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
import torch
from datasets import Dataset as HFDataset
import numpy as np
from typing import Dict, Any
from .utils import timer

class TextSummarizationDataset(Dataset):
    """PyTorch Dataset for text summarization."""
    
    def __init__(self, articles, summaries, tokenizer, max_len):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles[idx])
        summary = str(self.summaries[idx])
        
        # Tokenize article and summary
        article_encoding = self.tokenizer(
            article,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        summary_encoding = self.tokenizer(
            summary,
            max_length=self.max_len // 2,  # Summaries are typically shorter
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': article_encoding['input_ids'].flatten(),
            'attention_mask': article_encoding['attention_mask'].flatten(),
            'labels': summary_encoding['input_ids'].flatten()
        }

def initialize_model(model_name: str) -> tuple:
    """
    Initialize model and tokenizer for different model architectures.
    
    Args:
        model_name: Name of pretrained model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Handle different model architectures
    if "bert2bert" in model_name or "bert" in model_name.lower():
        # For BERT2BERT models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EncoderDecoderModel.from_pretrained(model_name)
        
    elif "bart" in model_name.lower():
        # For BART models
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        
    elif "t5" in model_name.lower():
        # For T5 models
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    else:
        # Default to Auto classes
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return model, tokenizer

def create_data_loaders(train_articles, train_summaries, val_articles, val_summaries, 
                       tokenizer, max_len, batch_size):
    """
    Create DataLoader objects for training and validation.
    
    Args:
        train_articles: Training articles
        train_summaries: Training summaries
        val_articles: Validation articles
        val_summaries: Validation summaries
        tokenizer: Tokenizer object
        max_len: Maximum sequence length
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TextSummarizationDataset(
        train_articles, train_summaries, tokenizer, max_len
    )
    val_dataset = TextSummarizationDataset(
        val_articles, val_summaries, tokenizer, max_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

@timer
def train_model(model, train_loader, val_loader, learning_rate, num_epochs, device):
    """
    Train the summarization model.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model, history

def save_model(model, tokenizer, save_path):
    """
    Save model and tokenizer.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        save_path: Path to save model
        
    Returns:
        None
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

def load_model(model_path, device):
    """
    Load model and tokenizer.
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer
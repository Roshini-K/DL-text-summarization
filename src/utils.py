"""
Utility functions for text summarization project.
"""

import time
from functools import wraps
from typing import Callable, Any
import logging
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.2f}s")
        return result
    return wrapper

def setup_device() -> str:
    """
    Set up device for training (GPU if available).
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    return device

def check_gpu_memory() -> None:
    """
    Check GPU memory usage if CUDA is available.
    
    Returns:
        None
    """
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        
        logger.info(f"GPU Memory: Total: {gpu_memory:.2f}GB, "
                   f"Allocated: {allocated:.2f}GB, "
                   f"Cached: {cached:.2f}GB")

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        
    Returns:
        None
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Set random seed to {seed}")
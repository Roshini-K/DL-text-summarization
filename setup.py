"""
Setup script for text summarization project.
"""

from setuptools import setup, find_packages

setup(
    name="text_summarization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "datasets>=1.12.0",
        "nltk>=3.6.0",
        "rouge-score>=0.0.4",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "beautifulsoup4>=4.9.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Text summarization using transformer models",
    keywords="text summarization nlp transformers",
    python_requires=">=3.7",
)
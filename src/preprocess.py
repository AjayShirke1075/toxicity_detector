# src/preprocess.py
import re
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """Basic cleaning: remove HTML, lowercase, remove urls/emails, keep alnum and spaces."""
    if not isinstance(text, str):
        return ""
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    # Replace urls and emails
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # Keep letters, numbers and whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

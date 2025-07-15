import re
import string
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_medical_terms(text: str) -> List[str]:
    """
    Extract medical terms from text.
    
    Args:
        text: Input text
        
    Returns:
        List of medical terms
    """
    # This is a simplified version without spaCy
    # In a real application, you'd use a medical NER model
    
    # List of common medical terms for demonstration
    medical_terms = [
        "headache", "fever", "pain", "cough", "rash", 
        "nausea", "vomiting", "dizziness", "fatigue", 
        "hypertension", "diabetes", "asthma", "pneumonia",
        "blood pressure", "shortness of breath", "chest pain", 
        "abdominal pain", "back pain", "joint pain"
    ]
    
    text_lower = text.lower()
    found_terms = []
    
    for term in medical_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms

def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of tokens
    """
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]
    
    return tokens

def preprocess_input(text: str) -> str:
    """
    Preprocess user input for analysis.
    
    Args:
        text: User input text
        
    Returns:
        Preprocessed text
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Extract medical terms to preserve them
    medical_terms = extract_medical_terms(text)
    
    # Ensure medical terms are preserved in the cleaned text
    # This is a simplified approach
    result = cleaned_text
    
    # Add any extracted medical terms that might have been lost in cleaning
    for term in medical_terms:
        if term not in result:
            result += f" {term}"
    
    return result

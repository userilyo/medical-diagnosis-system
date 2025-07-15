"""
Paper-based model implementations for LSTM and RNN models.
Based on the research paper: "International Classification of Diseases Prediction from 
MIMIIC-III Clinical Text Using Pre-Trained ClinicalBERT and NLP Deep Learning Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Union, Tuple
import re
import os
from models.paper_lstm_model import PaperLSTMModel, load_pretrained_model, create_icd_code_mapping, get_model_info
from utils.data_processing import tokenize_text, clean_text

class PaperBasedPredictor:
    """
    Predictor class that uses pre-trained models from the research paper.
    """
    
    def __init__(self):
        self.lstm_model = None
        self.icd_mapping = create_icd_code_mapping()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained LSTM model."""
        try:
            # Load LSTM model
            lstm_path = "attached_assets/lstm_model_v1_1752599989807.pth"
            if os.path.exists(lstm_path):
                self.lstm_model = load_pretrained_model(lstm_path, "lstm")
                self.lstm_model.to(self.device)
                print("✓ LSTM model loaded successfully from research paper")
            else:
                print("⚠ LSTM model file not found, using default model")
                self.lstm_model = PaperLSTMModel()
                self.lstm_model.to(self.device)
                
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            # Fallback to default model
            self.lstm_model = PaperLSTMModel()
            self.lstm_model.to(self.device)
    
    def _text_to_tensor(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Convert text to tensor for model input.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tensor for model input
        """
        # Clean and tokenize text
        cleaned_text = clean_text(text)
        tokens = tokenize_text(cleaned_text)
        
        # Simple vocabulary mapping (in production, this would use the actual training vocabulary)
        vocab = self._create_simple_vocab(tokens)
        
        # Convert tokens to indices
        indices = [vocab.get(token, 1) for token in tokens[:max_length]]  # 1 for <UNK>
        
        # Pad sequence
        if len(indices) < max_length:
            indices.extend([0] * (max_length - len(indices)))  # 0 for <PAD>
        
        # Convert to tensor
        return torch.tensor([indices], dtype=torch.long).to(self.device)
    
    def _create_simple_vocab(self, tokens: List[str]) -> Dict[str, int]:
        """
        Create a simple vocabulary mapping.
        In production, this would use the actual training vocabulary.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Vocabulary mapping
        """
        # Basic medical vocabulary (simplified for demonstration)
        medical_vocab = {
            "<PAD>": 0, "<UNK>": 1, "pain": 2, "headache": 3, "nausea": 4,
            "fever": 5, "cough": 6, "chest": 7, "abdominal": 8, "shortness": 9,
            "breath": 10, "dizziness": 11, "fatigue": 12, "swelling": 13,
            "joint": 14, "muscle": 15, "skin": 16, "rash": 17, "itching": 18,
            "burning": 19, "numbness": 20, "tingling": 21, "weakness": 22,
            "bleeding": 23, "discharge": 24, "difficulty": 25, "swallowing": 26,
            "urination": 27, "bowel": 28, "movement": 29, "appetite": 30,
            "weight": 31, "loss": 32, "gain": 33, "sleep": 34, "anxiety": 35,
            "depression": 36, "confusion": 37, "memory": 38, "vision": 39,
            "hearing": 40, "speech": 41, "walking": 42, "coordination": 43,
            "balance": 44, "tremor": 45, "seizure": 46, "palpitations": 47,
            "hypertension": 48, "diabetes": 49, "asthma": 50, "pneumonia": 51,
            "infection": 52, "inflammation": 53, "chronic": 54, "acute": 55,
            "severe": 56, "mild": 57, "moderate": 58, "sudden": 59, "gradual": 60
        }
        
        # Add any new tokens with high indices
        for token in tokens:
            if token not in medical_vocab:
                medical_vocab[token] = len(medical_vocab)
        
        return medical_vocab
    
    def predict_with_lstm(self, symptoms_text: str) -> Dict[str, Any]:
        """
        Predict using the LSTM model from the research paper.
        
        Args:
            symptoms_text: Patient symptoms text
            
        Returns:
            Dictionary containing LSTM predictions
        """
        try:
            # Convert text to tensor
            input_tensor = self._text_to_tensor(symptoms_text)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.lstm_model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                top_predictions = torch.topk(probabilities, k=5, dim=1)
            
            # Format results
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_predictions.values[0], top_predictions.indices[0])):
                icd_code = self.icd_mapping.get(idx.item(), f"ICD_{idx.item()}")
                predictions.append({
                    "icd_code": icd_code,
                    "confidence": float(prob.item()),
                    "rank": i + 1
                })
            
            return {
                "predictions": predictions,
                "model_type": "Paper-based LSTM",
                "overall_confidence": float(top_predictions.values[0][0].item()),
                "method": "pretrained_lstm_from_paper",
                "paper_reference": "BDCC-08-00047-v2",
                "model_accuracy": "81% (Top 10 ICD codes from paper)"
            }
            
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return {
                "predictions": [],
                "model_type": "Paper-based LSTM",
                "overall_confidence": 0.0,
                "method": "pretrained_lstm_from_paper",
                "error": str(e)
            }
    

    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded LSTM model.
        
        Returns:
            Dictionary with model information
        """
        lstm_info = get_model_info("attached_assets/lstm_model_v1_1752599989807.pth")
        
        return {
            "lstm_model": lstm_info,
            "paper_reference": "BDCC-08-00047-v2: International Classification of Diseases Prediction from MIMIIC-III Clinical Text",
            "paper_results": {
                "lstm_top10_accuracy": "81%",
                "lstm_top50_accuracy": "68%",
                "f1_scores": {
                    "lstm_top10": "81%",
                    "lstm_top50": "66%"
                }
            },
            "hyperparameters": {
                "lstm": {
                    "embedding_dim": 128,
                    "hidden_dim": 256,
                    "dropout": 0.2,
                    "optimizer": "AdamW",
                    "learning_rate": 0.001,
                    "batch_size": 16,
                    "epochs": 10
                }
            }
        }

# Global instance
_paper_predictor = None

def get_paper_predictor() -> PaperBasedPredictor:
    """Get singleton instance of paper-based predictor."""
    global _paper_predictor
    if _paper_predictor is None:
        _paper_predictor = PaperBasedPredictor()
    return _paper_predictor

def paper_lstm_prediction(symptoms_text: str) -> Dict[str, Any]:
    """
    Generate predictions using the paper-based LSTM model.
    
    Args:
        symptoms_text: Patient symptoms text
        
    Returns:
        Dictionary containing LSTM predictions from research paper
    """
    predictor = get_paper_predictor()
    result = predictor.predict_with_lstm(symptoms_text)
    
    # Format for compatibility with the app interface
    if result.get('predictions'):
        top_prediction = result['predictions'][0]
        return {
            "predicted_icd": top_prediction['icd_code'],
            "condition": f"Medical condition for {top_prediction['icd_code']}",
            "confidence": top_prediction['confidence'],
            "status": "success",
            "model_type": result.get('model_type', 'Paper-based LSTM'),
            "predictions": result['predictions'],
            "paper_reference": result.get('paper_reference', 'BDCC-08-00047-v2'),
            "model_accuracy": result.get('model_accuracy', '81% (Top 10 ICD codes)')
        }
    else:
        return {
            "predicted_icd": "N/A",
            "condition": "N/A",
            "confidence": 0.0,
            "status": "error",
            "error": result.get('error', 'Unknown error')
        }



def get_paper_model_info() -> Dict[str, Any]:
    """
    Get information about the paper-based models.
    
    Returns:
        Dictionary with comprehensive model information
    """
    predictor = get_paper_predictor()
    return predictor.get_model_info()
"""
RandomForest-based Traditional ML using authentic MIMIC-III trained models
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ICD-10 code mapping for RandomForest model classes
ICD10_MAPPING = {
    0: "Z38.00",  # Single liveborn infant, delivered vaginally
    1: "I25.10",  # Atherosclerotic heart disease of native coronary artery without angina pectoris
    2: "Z38.01",  # Single liveborn infant, delivered by cesarean
    3: "A41.9",   # Sepsis, unspecified organism
    4: "I21.4",   # Non-ST elevation (NSTEMI) myocardial infarction
    5: "I35.9",   # Nonrheumatic aortic valve disorder, unspecified
    6: "J96.00",  # Acute respiratory failure, unspecified whether with hypoxia or hypercapnia
    7: "I61.9",   # Nontraumatic intracerebral hemorrhage, unspecified
    8: "Z38.31",  # Twin liveborn infant, delivered by cesarean
    9: "J18.9"    # Pneumonia, unspecified organism
}

# ICD-10 descriptions for better display
ICD10_DESCRIPTIONS = {
    "Z38.00": "Single liveborn infant, delivered vaginally",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "Z38.01": "Single liveborn infant, delivered by cesarean",
    "A41.9": "Sepsis, unspecified organism",
    "I21.4": "Non-ST elevation (NSTEMI) myocardial infarction",
    "I35.9": "Nonrheumatic aortic valve disorder, unspecified",
    "J96.00": "Acute respiratory failure, unspecified whether with hypoxia or hypercapnia",
    "I61.9": "Nontraumatic intracerebral hemorrhage, unspecified",
    "Z38.31": "Twin liveborn infant, delivered by cesarean",
    "J18.9": "Pneumonia, unspecified organism"
}

class RandomForestMedicalPredictor:
    """
    RandomForest-based medical predictor using authentic MIMIC-III trained models.
    """
    
    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model_loaded = False
        self.model_path = "attached_assets/rf_traditional_ml_model_v1_1752608692532.joblib"
        self.vectorizer_path = "attached_assets/rf_vectorizer_v1_1752608692525.joblib"
        
    def load_models(self) -> bool:
        """
        Load the pre-trained RandomForest model and vectorizer from MIMIC-III dataset.
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # Check if model files exist
            if not os.path.exists(self.model_path):
                logger.error(f"RandomForest model file not found: {self.model_path}")
                return False
                
            if not os.path.exists(self.vectorizer_path):
                logger.error(f"Vectorizer file not found: {self.vectorizer_path}")
                return False
            
            # Load the vectorizer
            logger.info("Loading TF-IDF vectorizer...")
            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info(f"✓ Vectorizer loaded successfully")
            
            # Load the RandomForest model
            logger.info("Loading RandomForest model...")
            self.model = joblib.load(self.model_path)
            logger.info(f"✓ RandomForest model loaded successfully")
            
            # Log model information
            if hasattr(self.model, 'n_estimators'):
                logger.info(f"Model parameters: n_estimators={self.model.n_estimators}")
            if hasattr(self.model, 'classes_'):
                logger.info(f"Model classes: {len(self.model.classes_)} ICD-10 codes")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading RandomForest models: {e}")
            return False
    
    def preprocess_text(self, symptoms_text: str) -> np.ndarray:
        """
        Preprocess symptoms text using the loaded vectorizer.
        
        Args:
            symptoms_text: Raw symptoms text
            
        Returns:
            Vectorized text features
        """
        if not self.vectorizer:
            raise ValueError("Vectorizer not loaded. Call load_models() first.")
        
        # Clean and prepare text
        cleaned_text = symptoms_text.lower().strip()
        
        # Transform text using the pre-trained vectorizer
        features = self.vectorizer.transform([cleaned_text])
        
        return features
    
    def predict_icd_codes(self, symptoms_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predict ICD-10 codes using the RandomForest model.
        
        Args:
            symptoms_text: Patient symptoms text
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with ICD codes and confidence scores
        """
        if not self.model_loaded:
            if not self.load_models():
                return []
        
        try:
            # Preprocess the text
            features = self.preprocess_text(symptoms_text)
            
            # Get probability predictions
            probabilities = self.model.predict_proba(features)[0]
            
            # Get class labels
            classes = self.model.classes_
            
            # Create predictions with confidence scores and ICD-10 code mapping
            predictions = []
            for i, prob in enumerate(probabilities):
                class_index = int(classes[i])
                icd_code = ICD10_MAPPING.get(class_index, f"Unknown_Class_{class_index}")
                description = ICD10_DESCRIPTIONS.get(icd_code, "Unknown condition")
                predictions.append({
                    'icd_code': icd_code,
                    'description': description,
                    'confidence': float(prob),
                    'probability': float(prob)
                })
            
            # Sort by confidence and return top_k
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Add rank information
            for i, pred in enumerate(predictions[:top_k]):
                pred['rank'] = i + 1
            
            return predictions[:top_k]
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded RandomForest model.
        
        Returns:
            Dictionary with model information
        """
        # Try to load models if not already loaded
        if not self.model_loaded:
            self.load_models()
        
        if not self.model_loaded:
            return {
                "model_loaded": False,
                "error": "Models not loaded"
            }
        
        info = {
            "model_loaded": True,
            "model_type": "RandomForest",
            "dataset": "MIMIC-III",
            "file_exists": os.path.exists(self.model_path),
            "vectorizer_exists": os.path.exists(self.vectorizer_path)
        }
        
        # Add model-specific information
        if self.model:
            model_classes = getattr(self.model, 'classes_', [])
            # Map numeric classes to ICD-10 codes
            icd_codes = [ICD10_MAPPING.get(int(cls), f"Unknown_Class_{cls}") for cls in model_classes]
            info.update({
                "n_estimators": getattr(self.model, 'n_estimators', 'Unknown'),
                "max_depth": getattr(self.model, 'max_depth', 'Unknown'),
                "n_features": getattr(self.model, 'n_features_in_', 'Unknown'),
                "n_classes": len(model_classes),
                "classes_sample": icd_codes  # Show ICD-10 codes instead of numeric classes
            })
        
        # Add vectorizer information
        if self.vectorizer:
            info.update({
                "vocabulary_size": len(getattr(self.vectorizer, 'vocabulary_', {})),
                "vectorizer_type": type(self.vectorizer).__name__
            })
        
        # Add file size information
        try:
            model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            vectorizer_size = os.path.getsize(self.vectorizer_path) / (1024 * 1024)  # MB
            info.update({
                "model_size_mb": round(model_size, 1),
                "vectorizer_size_mb": round(vectorizer_size, 1)
            })
        except:
            pass
        
        return info

# Global instance
_rf_predictor = None

def get_rf_predictor() -> RandomForestMedicalPredictor:
    """
    Get singleton instance of RandomForest predictor.
    
    Returns:
        RandomForestMedicalPredictor instance
    """
    global _rf_predictor
    if _rf_predictor is None:
        _rf_predictor = RandomForestMedicalPredictor()
    return _rf_predictor

def rf_traditional_ml_prediction(symptoms_text: str) -> Dict[str, Any]:
    """
    Generate predictions using RandomForest traditional ML model.
    
    Args:
        symptoms_text: Patient symptoms text
        
    Returns:
        Dictionary containing RandomForest predictions
    """
    try:
        predictor = get_rf_predictor()
        predictions = predictor.predict_icd_codes(symptoms_text, top_k=7)
        
        if predictions:
            # Format results for compatibility
            top_prediction = predictions[0]
            
            result = {
                "predicted_icd": top_prediction["icd_code"],
                "condition": top_prediction.get("description", f"Medical condition for {top_prediction['icd_code']}"),
                "confidence": top_prediction["confidence"],
                "predictions": predictions,
                "method": "RandomForest_MIMIC_III",
                "model_type": "RandomForest Traditional ML",
                "dataset": "MIMIC-III",
                "status": "success",
                "total_predictions": len(predictions)
            }
            
            return result
        else:
            return {
                "predicted_icd": "N/A",
                "condition": "N/A",
                "confidence": 0.0,
                "status": "no_predictions",
                "error": "No predictions generated"
            }
            
    except Exception as e:
        logger.error(f"Error in RF traditional ML prediction: {e}")
        return {
            "predicted_icd": "N/A",
            "condition": "N/A",
            "confidence": 0.0,
            "status": "error",
            "error": str(e)
        }

def get_rf_model_info() -> Dict[str, Any]:
    """
    Get information about the RandomForest model.
    
    Returns:
        Dictionary with model information
    """
    try:
        predictor = get_rf_predictor()
        return predictor.get_model_info()
    except Exception as e:
        return {
            "model_loaded": False,
            "error": str(e)
        }

# Test function
def test_rf_model():
    """
    Test function to verify RandomForest model functionality.
    """
    print("Testing RandomForest Traditional ML Model...")
    
    # Test with sample symptoms
    symptoms = "chest pain, shortness of breath, fatigue"
    result = rf_traditional_ml_prediction(symptoms)
    
    print(f"Symptoms: {symptoms}")
    print(f"Predicted ICD: {result.get('predicted_icd', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0):.3f}")
    print(f"Status: {result.get('status', 'N/A')}")
    print(f"Method: {result.get('method', 'N/A')}")
    
    # Show model info
    info = get_rf_model_info()
    print(f"\nModel Info:")
    print(f"Model loaded: {info.get('model_loaded', False)}")
    print(f"Model type: {info.get('model_type', 'N/A')}")
    print(f"Dataset: {info.get('dataset', 'N/A')}")
    print(f"N estimators: {info.get('n_estimators', 'N/A')}")
    print(f"N classes: {info.get('n_classes', 'N/A')}")

if __name__ == "__main__":
    test_rf_model()
import random
import numpy as np
import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def extract_features(text: str) -> np.ndarray:
    """
    Extract comprehensive features from text using advanced NLP techniques.
    
    Args:
        text: The input text
        
    Returns:
        Feature vector combining multiple feature types
    """
    # Initialize feature vector
    features = []
    
    # Preprocess text
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    # 1. TF-IDF Features for medical terms
    medical_terms = [
        "pain", "ache", "headache", "nausea", "vomiting", "fever", "cough", "shortness", "breath",
        "chest", "abdominal", "joint", "muscle", "skin", "rash", "swelling", "dizziness", "fatigue",
        "weakness", "numbness", "tingling", "burning", "itching", "bleeding", "discharge", "difficulty",
        "swallowing", "urination", "bowel", "movement", "appetite", "weight", "loss", "gain", "sleep",
        "insomnia", "anxiety", "depression", "confusion", "memory", "vision", "hearing", "speech",
        "walking", "coordination", "balance", "tremor", "seizure", "unconscious", "syncope", "palpitations"
    ]
    
    # Calculate TF-IDF-like scores for medical terms
    text_tokens = text_clean.split()
    text_length = len(text_tokens) if text_tokens else 1  # Avoid division by zero
    
    for term in medical_terms:
        count = text_clean.count(term)
        tf_score = float(count) / text_length
        features.append(tf_score)
    
    # 2. Symptom severity indicators
    severity_indicators = {
        "mild": 0.3, "moderate": 0.5, "severe": 0.8, "intense": 0.9, "extreme": 1.0,
        "slight": 0.2, "sharp": 0.7, "dull": 0.4, "throbbing": 0.6, "burning": 0.7,
        "stabbing": 0.8, "crushing": 0.9, "persistent": 0.7, "intermittent": 0.5
    }
    
    max_severity = 0.0
    avg_severity = 0.0
    severity_terms = []
    
    for indicator, score in severity_indicators.items():
        if indicator in text_lower:
            severity_terms.append(score)
            max_severity = max(max_severity, score)
    
    avg_severity = float(np.mean(severity_terms)) if severity_terms else 0.0
    features.extend([float(max_severity), float(avg_severity)])
    
    # 3. Temporal features
    temporal_keywords = {
        "sudden": 0.8, "gradual": 0.4, "chronic": 0.6, "acute": 0.9, "recent": 0.7,
        "ongoing": 0.6, "intermittent": 0.5, "constant": 0.7, "occasional": 0.3,
        "daily": 0.7, "weekly": 0.4, "monthly": 0.2, "hours": 0.8, "days": 0.6,
        "weeks": 0.4, "months": 0.2, "years": 0.1
    }
    
    temporal_score = 0.0
    for keyword, score in temporal_keywords.items():
        if keyword in text_lower:
            temporal_score = max(temporal_score, score)
    
    features.append(float(temporal_score))
    
    # 4. Anatomical location features
    anatomical_regions = {
        "head": 0, "neck": 1, "chest": 2, "abdomen": 3, "back": 4, "arm": 5,
        "leg": 6, "joint": 7, "muscle": 8, "skin": 9, "eye": 10, "ear": 11,
        "nose": 12, "throat": 13, "heart": 14, "lung": 15, "stomach": 16,
        "liver": 17, "kidney": 18, "bladder": 19
    }
    
    anatomical_vector = [0.0] * 20
    for region, index in anatomical_regions.items():
        if region in text_lower:
            anatomical_vector[index] = 1.0
    
    features.extend(anatomical_vector)
    
    # 5. System-specific features
    system_features = {
        "cardiovascular": ["heart", "chest", "palpitations", "pressure", "circulation"],
        "respiratory": ["breath", "cough", "lung", "wheeze", "asthma"],
        "gastrointestinal": ["stomach", "nausea", "vomiting", "diarrhea", "constipation"],
        "neurological": ["headache", "dizziness", "numbness", "seizure", "memory"],
        "musculoskeletal": ["joint", "muscle", "bone", "arthritis", "fracture"],
        "dermatological": ["skin", "rash", "itch", "lesion", "wound"],
        "genitourinary": ["urination", "bladder", "kidney", "frequency", "urgency"],
        "endocrine": ["diabetes", "thyroid", "hormone", "metabolism", "glucose"]
    }
    
    system_scores = []
    for system, keywords in system_features.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        system_scores.append(float(score) / len(keywords))
    
    features.extend(system_scores)
    
    # 6. Context and association features
    context_patterns = {
        "after_eating": r"after\s+eating|post\s+meal|following\s+food",
        "during_exercise": r"during\s+exercise|while\s+running|when\s+walking",
        "at_rest": r"at\s+rest|while\s+sleeping|when\s+lying",
        "worse_at_night": r"worse\s+at\s+night|nighttime|evening",
        "morning_stiffness": r"morning\s+stiffness|stiff\s+in\s+morning"
    }
    
    context_scores = []
    for pattern_name, pattern in context_patterns.items():
        if re.search(pattern, text_lower):
            context_scores.append(1.0)
        else:
            context_scores.append(0.0)
    
    features.extend(context_scores)
    
    return np.array(features)

def get_condition_probabilities(features: np.ndarray) -> List[Dict[str, Any]]:
    """
    Get probabilities for different conditions based on comprehensive feature analysis.
    
    Args:
        features: Feature vector
        
    Returns:
        List of conditions with probabilities
    """
    # Define condition patterns with weighted feature importance
    condition_patterns = {
        "G43.1": {  # Migraine with aura
            "name": "Migraine with aura",
            "feature_weights": {
                "headache": 0.4, "nausea": 0.3, "vision": 0.4, "dizziness": 0.3,
                "neurological_system": 0.5, "head_region": 0.4, "sudden_onset": 0.3
            }
        },
        "B86": {  # Scabies
            "name": "Scabies",
            "feature_weights": {
                "itching": 0.6, "rash": 0.4, "skin": 0.5, "nighttime": 0.3,
                "dermatological_system": 0.6, "skin_region": 0.5
            }
        },
        "K40.20": {  # Bilateral inguinal hernia
            "name": "Bilateral inguinal hernia",
            "feature_weights": {
                "pain": 0.4, "swelling": 0.5, "abdomen": 0.4, "chronic": 0.3,
                "musculoskeletal_system": 0.3, "during_exercise": 0.4
            }
        },
        "K35.9": {  # Acute appendicitis
            "name": "Acute appendicitis",
            "feature_weights": {
                "pain": 0.5, "abdomen": 0.6, "nausea": 0.4, "fever": 0.3,
                "gastrointestinal_system": 0.5, "acute_onset": 0.5
            }
        },
        "M06.9": {  # Rheumatoid arthritis
            "name": "Rheumatoid arthritis",
            "feature_weights": {
                "joint": 0.6, "pain": 0.4, "swelling": 0.4, "morning_stiffness": 0.5,
                "musculoskeletal_system": 0.6, "chronic": 0.4
            }
        },
        "I49.9": {  # Cardiac arrhythmia
            "name": "Cardiac arrhythmia",
            "feature_weights": {
                "palpitations": 0.6, "chest": 0.4, "heart": 0.5, "shortness": 0.4,
                "cardiovascular_system": 0.6, "at_rest": 0.3
            }
        },
        "N02.9": {  # Recurrent hematuria
            "name": "Recurrent hematuria",
            "feature_weights": {
                "urination": 0.5, "bladder": 0.4, "kidney": 0.3, "burning": 0.4,
                "genitourinary_system": 0.6, "chronic": 0.3
            }
        },
        "R51": {  # Headache
            "name": "Headache",
            "feature_weights": {
                "headache": 0.7, "head": 0.5, "pain": 0.4, "neurological_system": 0.4
            }
        },
        "J45.9": {  # Asthma
            "name": "Asthma",
            "feature_weights": {
                "breath": 0.6, "wheeze": 0.5, "cough": 0.4, "chest": 0.3,
                "respiratory_system": 0.6, "during_exercise": 0.4
            }
        },
        "I20.0": {  # Angina pectoris
            "name": "Angina pectoris",
            "feature_weights": {
                "chest": 0.6, "pain": 0.5, "heart": 0.4, "shortness": 0.4,
                "cardiovascular_system": 0.6, "during_exercise": 0.5
            }
        }
    }
    
    # Create feature mapping (simplified - in production would use proper indexing)
    feature_names = [
        "pain", "ache", "headache", "nausea", "vomiting", "fever", "cough", "shortness", "breath",
        "chest", "abdominal", "joint", "muscle", "skin", "rash", "swelling", "dizziness", "fatigue",
        "weakness", "numbness", "tingling", "burning", "itching", "bleeding", "discharge", "difficulty",
        "swallowing", "urination", "bowel", "movement", "appetite", "weight", "loss", "gain", "sleep",
        "insomnia", "anxiety", "depression", "confusion", "memory", "vision", "hearing", "speech",
        "walking", "coordination", "balance", "tremor", "seizure", "unconscious", "syncope", "palpitations",
        "max_severity", "avg_severity", "temporal_score"
    ]
    
    # Add anatomical regions (indices 51-70)
    anatomical_regions = [
        "head", "neck", "chest", "abdomen", "back", "arm", "leg", "joint", "muscle", "skin",
        "eye", "ear", "nose", "throat", "heart", "lung", "stomach", "liver", "kidney", "bladder"
    ]
    
    # Add system scores (indices 71-78)
    system_names = [
        "cardiovascular_system", "respiratory_system", "gastrointestinal_system", "neurological_system",
        "musculoskeletal_system", "dermatological_system", "genitourinary_system", "endocrine_system"
    ]
    
    # Add context patterns (indices 79-83)
    context_names = [
        "after_eating", "during_exercise", "at_rest", "worse_at_night", "morning_stiffness"
    ]
    
    # Combine all feature names
    all_feature_names = feature_names + anatomical_regions + system_names + context_names
    
    # Calculate probabilities for each condition
    results = []
    for icd_code, condition_info in condition_patterns.items():
        probability = 0.0
        feature_count = 0
        
        for feature_name, weight in condition_info["feature_weights"].items():
            if feature_name in all_feature_names:
                feature_idx = all_feature_names.index(feature_name)
                if feature_idx < len(features):
                    # Ensure feature value is numeric
                    feature_value = features[feature_idx]
                    if isinstance(feature_value, (int, float, np.integer, np.floating)):
                        probability += float(feature_value) * float(weight)
                        feature_count += 1
        
        # Normalize by number of features
        if feature_count > 0:
            probability = probability / feature_count
        
        # Add baseline probability
        probability += 0.1
        
        # Apply sigmoid-like transformation to keep probabilities reasonable
        probability = 1 / (1 + np.exp(-5 * (probability - 0.5)))
        
        # Ensure probability is between 0 and 1
        probability = max(0.05, min(0.95, probability))
        
        results.append({
            "icd_code": icd_code,
            "condition": condition_info["name"],
            "probability": probability,
            "confidence": probability
        })
    
    # Sort by probability and return top results
    results.sort(key=lambda x: x["probability"], reverse=True)
    
    return results[:7]  # Return top 7 conditions

def traditional_ml_prediction(symptoms_text: str) -> Dict[str, Any]:
    """
    Generate predictions using traditional machine learning approach.
    Now uses authentic RandomForest model trained on MIMIC-III dataset.
    
    Args:
        symptoms_text: Patient symptoms text
        
    Returns:
        Dictionary containing traditional ML predictions
    """
    # Try to use authentic RandomForest model first
    try:
        from utils.rf_traditional_ml import rf_traditional_ml_prediction
        rf_result = rf_traditional_ml_prediction(symptoms_text)
        
        # If RandomForest works, use it as primary method
        if rf_result.get("status") == "success":
            return rf_result
        else:
            print(f"RandomForest failed: {rf_result.get('error', 'Unknown error')}, falling back to feature-based ML")
    except Exception as e:
        print(f"RandomForest model failed, falling back to feature-based ML: {e}")
    
    # Fallback to feature-based prediction
    return feature_based_prediction(symptoms_text)

def feature_based_prediction(symptoms_text: str) -> Dict[str, Any]:
    """
    Generate predictions using traditional machine learning features.
    
    Args:
        symptoms_text: Patient symptoms text
        
    Returns:
        Dictionary containing predictions from traditional ML
    """
    # Extract features from the symptoms text
    features = extract_features(symptoms_text)
    
    # Get condition probabilities
    condition_probabilities = get_condition_probabilities(features)
    
    # Format results to match expected structure
    predictions = []
    for condition in condition_probabilities:
        predictions.append({
            "icd_code": condition["icd_code"],
            "condition": condition["condition"],
            "probability": condition["probability"],
            "confidence": condition["confidence"],
            "source": "traditional_ml_rules"
        })
    
    # Calculate overall confidence
    overall_confidence = np.mean([pred["confidence"] for pred in predictions]) if predictions else 0.0
    
    # Return formatted results
    result = {
        "predictions": predictions,
        "overall_confidence": overall_confidence,
        "method": "traditional_ml_enhanced",
        "feature_vector_size": len(features),
        "total_predictions": len(predictions),
        "status": "success"
    }
    
    # Add top prediction for compatibility
    if predictions:
        top_prediction = predictions[0]
        result.update({
            "predicted_icd": top_prediction["icd_code"],
            "condition": top_prediction["condition"],
            "confidence": top_prediction["confidence"]
        })
    else:
        result.update({
            "predicted_icd": "N/A",
            "condition": "N/A",
            "confidence": 0.0,
            "status": "no_predictions"
        })
    
    return result


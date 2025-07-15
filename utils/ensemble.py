import numpy as np
from typing import Dict, Any, List

def ensemble_predictions(
    llm_results: Dict[str, Any],
    lstm_verified_codes: Dict[str, Any],
    ml_predictions: Dict[str, Any],
    rag_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ensemble predictions from different models.
    
    Args:
        llm_results: Results from LLM predictions
        lstm_verified_codes: Verified ICD codes from LSTM model
        ml_predictions: Predictions from traditional ML model
        rag_results: Results from RAG system
        
    Returns:
        Ensemble prediction result
    """
    # Collect all unique ICD-10 codes from all models
    all_codes = set()
    code_info = {}
    
    # Add LLM predictions
    if "predictions" in llm_results:
        for pred in llm_results["predictions"]:
            code = pred.get("icd_code", "")
            if code:
                all_codes.add(code)
                if code not in code_info:
                    code_info[code] = {
                        "condition": pred.get("condition", ""),
                        "scores": [],
                        "sources": []
                    }
                code_info[code]["scores"].append(pred.get("confidence", 0.0))
                code_info[code]["sources"].append("llm")
    
    # Add LSTM verified codes
    if "verified_codes" in lstm_verified_codes:
        for code_info_item in lstm_verified_codes["verified_codes"]:
            code = code_info_item.get("icd_code", "")
            if code:
                all_codes.add(code)
                if code not in code_info:
                    code_info[code] = {
                        "condition": "",  # LSTM may not provide condition names
                        "scores": [],
                        "sources": []
                    }
                code_info[code]["scores"].append(code_info_item.get("confidence", 0.0))
                code_info[code]["sources"].append("lstm")
    
    # Add ML predictions
    if "predictions" in ml_predictions:
        for pred in ml_predictions["predictions"]:
            code = pred.get("icd_code", "")
            if code:
                all_codes.add(code)
                if code not in code_info:
                    code_info[code] = {
                        "condition": pred.get("condition", ""),
                        "scores": [],
                        "sources": []
                    }
                code_info[code]["scores"].append(pred.get("probability", 0.0))
                code_info[code]["sources"].append("ml")
    
    # Add RAG suggested codes
    if "suggested_codes" in rag_results:
        for code in rag_results["suggested_codes"]:
            if code:
                all_codes.add(code)
                if code not in code_info:
                    code_info[code] = {
                        "condition": "",  # RAG may not provide condition names
                        "scores": [],
                        "sources": []
                    }
                # Add a fixed score for RAG suggestions (could be refined)
                code_info[code]["scores"].append(0.7)  # Medium-high confidence
                code_info[code]["sources"].append("rag")
    
    # Calculate ensemble scores
    ensemble_scores = []
    for code in all_codes:
        info = code_info[code]
        
        # Calculate average score, with more weight for LSTM (it's a verifier)
        scores = info["scores"]
        sources = info["sources"]
        
        # Enhanced confidence calculation
        if not scores:
            weighted_score = 0.0
        else:
            # Simple weighted average based on sources
            weights = [1.0 for _ in scores]  # Default weight
            for i, source in enumerate(sources):
                if source == "lstm":  # Give LSTM verification more weight
                    weights[i] = 1.5
                elif source == "rag":  # Give RAG a bit more weight
                    weights[i] = 1.2
            
            # Weighted average
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            # Apply boost for codes verified by multiple models
            if len(set(sources)) > 1:
                # More substantial boost for multi-model agreement
                if len(set(sources)) >= 3:
                    weighted_score *= 1.15  # 15% boost for 3+ models
                else:
                    weighted_score *= 1.08  # 8% boost for 2 models
            
            # Apply minimum confidence boost for ensemble predictions
            # This ensures that ensemble predictions don't undervalue good individual predictions
            if weighted_score > 0.5:
                weighted_score = min(weighted_score * 1.1, 1.0)  # 10% boost for good predictions
            
            # Cap at 1.0
            weighted_score = min(weighted_score, 1.0)
        
        # Create entry with condition name from any source
        ensemble_scores.append({
            "icd_code": code,
            "condition": info["condition"],
            "confidence": weighted_score,
            "sources": list(set(sources))
        })
    
    # Sort by confidence
    ensemble_scores.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Primary diagnosis is the highest confidence one
    primary_diagnosis = ensemble_scores[0] if ensemble_scores else {}
    
    # Differential diagnoses are the rest
    differential_diagnoses = ensemble_scores[1:4] if len(ensemble_scores) > 1 else []
    
    return {
        "primary_diagnosis": primary_diagnosis,
        "differential_diagnoses": differential_diagnoses,
        "all_scores": ensemble_scores
    }

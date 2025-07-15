from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(
    final_prediction: Dict[str, Any],
    llm_results: Dict[str, Any],
    lstm_verified_codes: Dict[str, Any],
    ml_predictions: Dict[str, Any],
    ground_truth: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for the prediction.
    
    Args:
        final_prediction: The final ensemble prediction
        llm_results: Results from LLM predictions
        lstm_verified_codes: Verified ICD codes from LSTM model
        ml_predictions: Predictions from traditional ML model
        ground_truth: Ground truth ICD code for evaluation (optional)
        
    Returns:
        Dictionary containing comprehensive evaluation metrics
    """
    try:
        # Initialize metrics dictionary
        metrics = {
            "model_agreement": 0.0,
            "confidence_scores": {},
            "inter_model_agreement": {},
            "prediction_quality": {},
            "ensemble_metrics": {}
        }
        
        # Get the primary diagnosis code
        primary_code = ""
        if "primary_diagnosis" in final_prediction and "icd_code" in final_prediction["primary_diagnosis"]:
            primary_code = final_prediction["primary_diagnosis"]["icd_code"]
        
        if not primary_code:
            logger.warning("No primary diagnosis code found in final prediction")
            return metrics
        
        # Collect predictions from all models
        model_predictions = {
            "llm": [],
            "lstm": [],
            "ml": []
        }
        
        # Extract LLM predictions
        if "predictions" in llm_results:
            model_predictions["llm"] = [
                p.get("icd_code", "") for p in llm_results["predictions"]
            ]
        
        # Extract LSTM predictions
        if "verified_codes" in lstm_verified_codes:
            model_predictions["lstm"] = [
                c.get("icd_code", "") for c in lstm_verified_codes["verified_codes"]
            ]
        
        # Extract ML predictions
        if "predictions" in ml_predictions:
            model_predictions["ml"] = [
                p.get("icd_code", "") for p in ml_predictions["predictions"]
            ]
        
        # Calculate model agreement metrics
        metrics["model_agreement"] = calculate_model_agreement(primary_code, model_predictions)
        
        # Calculate inter-model agreement (Cohen's kappa-like metric)
        metrics["inter_model_agreement"] = calculate_inter_model_agreement(model_predictions)
        
        # Extract confidence scores
        metrics["confidence_scores"] = extract_confidence_scores(
            primary_code, llm_results, lstm_verified_codes, ml_predictions, final_prediction
        )
        
        # Calculate prediction quality metrics
        metrics["prediction_quality"] = calculate_prediction_quality(model_predictions)
        
        # Calculate ensemble-specific metrics
        metrics["ensemble_metrics"] = calculate_ensemble_metrics(final_prediction, model_predictions)
        
        # If ground truth is provided, calculate accuracy metrics
        if ground_truth:
            metrics["accuracy_metrics"] = calculate_accuracy_metrics(
                primary_code, ground_truth, model_predictions
            )
        
        logger.info(f"Calculated metrics for prediction: {primary_code}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            "model_agreement": 0.0,
            "confidence_scores": {},
            "error": str(e)
        }

def calculate_model_agreement(primary_code: str, model_predictions: Dict[str, List[str]]) -> float:
    """Calculate the percentage of models that agree on the primary diagnosis."""
    models_with_primary = 0
    total_models = sum(1 for preds in model_predictions.values() if preds)
    
    for model_preds in model_predictions.values():
        if primary_code in model_preds:
            models_with_primary += 1
    
    return models_with_primary / total_models if total_models > 0 else 0.0

def calculate_inter_model_agreement(model_predictions: Dict[str, List[str]]) -> Dict[str, float]:
    """Calculate pairwise agreement between models."""
    agreement_scores = {}
    model_names = list(model_predictions.keys())
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Avoid duplicate pairs
                preds1 = set(model_predictions[model1])
                preds2 = set(model_predictions[model2])
                
                if preds1 or preds2:
                    # Calculate Jaccard similarity
                    intersection = len(preds1.intersection(preds2))
                    union = len(preds1.union(preds2))
                    agreement = intersection / union if union > 0 else 0.0
                    agreement_scores[f"{model1}_{model2}"] = agreement
    
    return agreement_scores

def extract_confidence_scores(
    primary_code: str,
    llm_results: Dict[str, Any],
    lstm_verified_codes: Dict[str, Any],
    ml_predictions: Dict[str, Any],
    final_prediction: Dict[str, Any]
) -> Dict[str, float]:
    """Extract confidence scores from all models for the primary diagnosis."""
    confidence_scores = {}
    
    # LLM confidence
    if "predictions" in llm_results:
        for pred in llm_results["predictions"]:
            if pred.get("icd_code", "") == primary_code:
                confidence_scores["llm"] = float(pred.get("confidence", 0.0))
                break
    
    # LSTM confidence
    if "verified_codes" in lstm_verified_codes:
        for code in lstm_verified_codes["verified_codes"]:
            if code.get("icd_code", "") == primary_code:
                confidence_scores["lstm"] = float(code.get("verification_confidence", 0.0))
                break
    
    # ML confidence
    if "predictions" in ml_predictions:
        for pred in ml_predictions["predictions"]:
            if pred.get("icd_code", "") == primary_code:
                confidence_scores["ml"] = float(pred.get("probability", 0.0))
                break
    
    # Ensemble confidence
    if "primary_diagnosis" in final_prediction:
        confidence_scores["ensemble"] = float(final_prediction["primary_diagnosis"].get("confidence", 0.0))
    
    return confidence_scores

def calculate_prediction_quality(model_predictions: Dict[str, List[str]]) -> Dict[str, Any]:
    """Calculate quality metrics for predictions."""
    quality_metrics = {}
    
    for model_name, predictions in model_predictions.items():
        if predictions:
            # Calculate diversity (number of unique predictions)
            unique_preds = len(set(predictions))
            total_preds = len(predictions)
            
            quality_metrics[model_name] = {
                "total_predictions": total_preds,
                "unique_predictions": unique_preds,
                "diversity_ratio": unique_preds / total_preds if total_preds > 0 else 0.0
            }
    
    return quality_metrics

def calculate_ensemble_metrics(final_prediction: Dict[str, Any], model_predictions: Dict[str, List[str]]) -> Dict[str, Any]:
    """Calculate ensemble-specific metrics."""
    ensemble_metrics = {}
    
    if "primary_diagnosis" in final_prediction:
        primary_code = final_prediction["primary_diagnosis"]["icd_code"]
        
        # Calculate how many models contributed to the final prediction
        contributing_models = sum(1 for preds in model_predictions.values() if primary_code in preds)
        total_models = len([preds for preds in model_predictions.values() if preds])
        
        ensemble_metrics["contributing_models"] = contributing_models
        ensemble_metrics["total_models"] = total_models
        ensemble_metrics["contribution_ratio"] = contributing_models / total_models if total_models > 0 else 0.0
    
    return ensemble_metrics

def calculate_accuracy_metrics(
    predicted_code: str,
    ground_truth: str,
    model_predictions: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Calculate accuracy metrics against ground truth."""
    accuracy_metrics = {
        "ensemble_correct": predicted_code == ground_truth,
        "model_accuracies": {}
    }
    
    # Calculate individual model accuracies
    for model_name, predictions in model_predictions.items():
        if predictions:
            # Check if ground truth is in top predictions
            accuracy_metrics["model_accuracies"][model_name] = {
                "top_1_correct": ground_truth in predictions[:1] if predictions else False,
                "top_3_correct": ground_truth in predictions[:3] if predictions else False,
                "top_5_correct": ground_truth in predictions[:5] if predictions else False
            }
    
    return accuracy_metrics

def calculate_weighted_kappa(model_predictions: Dict[str, List[str]]) -> float:
    """Calculate weighted kappa for inter-model agreement (simplified implementation)."""
    # This is a simplified version - in production, use scikit-learn's cohen_kappa_score
    model_names = list(model_predictions.keys())
    if len(model_names) < 2:
        return 0.0
    
    agreements = []
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                preds1 = set(model_predictions[model1])
                preds2 = set(model_predictions[model2])
                
                if preds1 or preds2:
                    intersection = len(preds1.intersection(preds2))
                    union = len(preds1.union(preds2))
                    agreement = intersection / union if union > 0 else 0.0
                    agreements.append(agreement)
    
    return np.mean(agreements) if agreements else 0.0

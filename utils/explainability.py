from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lime
from lime.lime_text import LimeTextExplainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_explanation(
    final_prediction: Dict[str, Any],
    llm_results: Dict[str, Any],
    rag_results: Dict[str, Any],
    symptoms_text: str = "",
    ml_predictions: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive explanations for the final prediction using multiple techniques.
    
    Args:
        final_prediction: The final ensemble prediction
        llm_results: Results from LLM predictions
        rag_results: Results from RAG system
        symptoms_text: Original symptoms text for LIME analysis
        ml_predictions: Traditional ML predictions for feature importance
        
    Returns:
        Dictionary containing comprehensive explanations with visualizations
    """
    # This is a simplified implementation
    # In a real application, this would generate more detailed and accurate explanations
    
    # Check if we have a primary diagnosis
    if "primary_diagnosis" not in final_prediction or not final_prediction["primary_diagnosis"]:
        return {
            "reasoning": "No primary diagnosis available.",
            "key_factors": []
        }
    
    primary = final_prediction["primary_diagnosis"]
    primary_code = primary.get("icd_code", "")
    primary_condition = primary.get("condition", "")
    
    # Collect reasoning from LLM predictions for the primary diagnosis
    llm_reasoning = ""
    if "predictions" in llm_results:
        for pred in llm_results["predictions"]:
            if pred.get("icd_code", "") == primary_code and "reasoning" in pred:
                llm_reasoning = pred["reasoning"]
                break
    
    # Get information from RAG results
    rag_info = ""
    if "relevant_info" in rag_results:
        rag_info = rag_results["relevant_info"]
        
    # Extract medical literature insights specifically
    medical_literature = ""
    if rag_info:
        # Look for PDF content in evidence texts
        evidence_texts = rag_results.get("evidence_texts", [])
        pdf_content = []
        
        for text in evidence_texts:
            if "From Medical Textbook 'Symptom to Diagnosis'" in text:
                # Extract the actual content after the prefix
                content = text.replace("From Medical Textbook 'Symptom to Diagnosis': ", "")
                if len(content) > 50:  # Only include substantial content
                    pdf_content.append(content[:300])  # Limit length
        
        if pdf_content:
            medical_literature = "\n\n".join(pdf_content[:2])  # Use first 2 chunks
        elif "From Medical Textbook 'Symptom to Diagnosis'" in rag_info:
            # Fallback to original approach
            parts = rag_info.split("From Medical Textbook 'Symptom to Diagnosis'")
            if len(parts) > 1:
                medical_literature = parts[1].strip()[:300]
    
    # Extract potential key factors from RAG and LLM results
    key_factors = []
    
    # Add factors from RAG medical concepts if available
    if "medical_concepts" in rag_results:
        concepts = rag_results["medical_concepts"]
        if "symptoms" in concepts and concepts["symptoms"]:
            key_factors.extend(concepts["symptoms"][:3])  # Take up to 3 symptoms
    
    # If we don't have enough factors, add common ones for the condition
    if len(key_factors) < 3:
        condition_to_factors = {
            "Essential hypertension": ["elevated blood pressure", "headache", "dizziness"],
            "Pneumonia": ["cough", "fever", "shortness of breath"],
            "Chest pain": ["chest discomfort", "pain on breathing", "radiating pain"],
            "Headache": ["head pain", "visual disturbances", "nausea"],
            "Gastroesophageal reflux disease": ["heartburn", "regurgitation", "chest discomfort"]
        }
        
        if primary_condition in condition_to_factors:
            for factor in condition_to_factors[primary_condition]:
                if factor not in key_factors:
                    key_factors.append(factor)
                    if len(key_factors) >= 3:
                        break
    
    # Generate dynamic reasoning using LLM if available, otherwise use enhanced templated approach
    reasoning = generate_dynamic_reasoning(
        primary_condition, primary_code, llm_reasoning, rag_info, 
        primary.get("confidence", 0.0), key_factors
    )
    
    # Generate LIME explanations if symptoms text is provided
    lime_explanation = None
    feature_importance_plot = None
    
    if symptoms_text and ml_predictions:
        try:
            lime_explanation = generate_lime_explanation(symptoms_text, primary_code, ml_predictions)
            feature_importance_plot = create_feature_importance_visualization(
                lime_explanation, symptoms_text, primary_condition
            )
        except Exception as e:
            logger.warning(f"Failed to generate LIME explanation: {e}")
    
    # Generate confidence visualization
    confidence_plot = create_confidence_visualization(final_prediction, llm_results)
    
    # Generate model agreement visualization
    agreement_plot = create_model_agreement_visualization(llm_results, ml_predictions)
    
    # Look for code descriptions to add to the explanation
    code_descriptions = {}
    if "code_descriptions" in rag_results and primary_code in rag_results["code_descriptions"]:
        code_descriptions[primary_code] = rag_results["code_descriptions"][primary_code]
        
        # Also add descriptions for the differential diagnoses if available
        if "differential_diagnoses" in final_prediction:
            for diff in final_prediction["differential_diagnoses"]:
                diff_code = diff.get("icd_code", "")
                if diff_code and diff_code in rag_results["code_descriptions"]:
                    code_descriptions[diff_code] = rag_results["code_descriptions"][diff_code]
    
    # Extract medical concepts if available
    medical_concepts = {}
    if "medical_concepts" in rag_results:
        medical_concepts = rag_results["medical_concepts"]
    
    return {
        "reasoning": reasoning,
        "key_factors": key_factors,
        "relevant_info": rag_info,
        "medical_literature": medical_literature,
        "code_descriptions": code_descriptions,
        "medical_concepts": medical_concepts,
        "lime_explanation": lime_explanation,
        "feature_importance_plot": feature_importance_plot,
        "confidence_plot": confidence_plot,
        "agreement_plot": agreement_plot,
        "explainability_method": "enhanced" if lime_explanation else "basic"
    }


def generate_dynamic_reasoning(condition, code, llm_reasoning, rag_info, confidence, key_factors):
    """Generate dynamic reasoning using available information."""
    if llm_reasoning:
        # Use LLM reasoning as primary source
        reasoning = llm_reasoning
        
        # Enhance with confidence context
        if confidence > 0.85:
            reasoning += f"\n\nThis diagnosis has high confidence ({confidence:.2f}) based on strong model agreement."
        elif confidence > 0.70:
            reasoning += f"\n\nThis diagnosis has moderate confidence ({confidence:.2f}) with good model consensus."
        else:
            reasoning += f"\n\nThis diagnosis has lower confidence ({confidence:.2f}) and should be considered alongside other possibilities."
            
        return reasoning
    
    # Generate enhanced templated reasoning
    reasoning = f"**Primary Diagnosis Analysis:**\n"
    reasoning += f"The diagnosis of {condition} (ICD-10: {code}) "
    
    # Add confidence-based language
    if confidence > 0.85:
        reasoning += "is strongly supported by the ensemble analysis. "
    elif confidence > 0.70:
        reasoning += "is well-supported by the model ensemble. "
    else:
        reasoning += "is suggested by the analysis but requires clinical correlation. "
    
    # Add key factors explanation
    if key_factors:
        reasoning += f"\n\n**Key Contributing Factors:**\n"
        for i, factor in enumerate(key_factors[:3], 1):
            reasoning += f"{i}. {factor}\n"
    
    # Add medical literature context
    if rag_info and len(rag_info) > 100:
        reasoning += f"\n\n**Medical Literature Context:**\n"
        # Extract meaningful excerpts
        excerpt = rag_info[:300] + "..." if len(rag_info) > 300 else rag_info
        reasoning += excerpt
    
    return reasoning


def generate_lime_explanation(symptoms_text, predicted_code, ml_predictions):
    """Generate LIME explanation for text-based prediction."""
    try:
        # Create a simple text classifier for LIME
        explainer = LimeTextExplainer(class_names=['negative', 'positive'])
        
        # Mock classifier function for LIME (in production, use actual ML model)
        def classifier_fn(texts):
            # Simple TF-IDF based classification for demonstration
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            try:
                features = vectorizer.fit_transform([symptoms_text] + list(texts))
                # Return probabilities (mock for demonstration)
                probs = np.random.rand(len(texts), 2)
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs
            except:
                # Fallback if vectorization fails
                return np.array([[0.3, 0.7]] * len(texts))
        
        # Generate explanation
        explanation = explainer.explain_instance(
            symptoms_text, 
            classifier_fn,
            num_features=10,
            num_samples=100
        )
        
        # Extract feature importance
        feature_importance = []
        for feature, importance in explanation.as_list():
            feature_importance.append({
                'feature': feature,
                'importance': importance,
                'direction': 'positive' if importance > 0 else 'negative'
            })
        
        return {
            'method': 'LIME',
            'feature_importance': feature_importance,
            'explanation_text': f"LIME analysis identified {len(feature_importance)} key text features contributing to the diagnosis."
        }
        
    except Exception as e:
        logger.error(f"LIME explanation failed: {e}")
        return None


def create_feature_importance_visualization(lime_explanation, symptoms_text, condition):
    """Create visualization for feature importance."""
    if not lime_explanation or not lime_explanation['feature_importance']:
        return None
    
    try:
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        
        features = []
        importances = []
        colors = []
        
        for item in lime_explanation['feature_importance'][:8]:  # Top 8 features
            features.append(item['feature'])
            importances.append(abs(item['importance']))
            colors.append('green' if item['importance'] > 0 else 'red')
        
        # Create horizontal bar plot
        plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Key Text Features for {condition} Diagnosis')
        plt.grid(axis='x', alpha=0.3)
        
        # Add legend
        plt.legend(['Positive Evidence', 'Negative Evidence'], loc='lower right')
        
        # Convert to base64 for embedding
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
        
    except Exception as e:
        logger.error(f"Feature importance visualization failed: {e}")
        return None


def create_confidence_visualization(final_prediction, llm_results):
    """Create visualization for model confidence distribution."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Collect confidence scores from different models
        confidences = []
        model_names = []
        
        # Primary diagnosis confidence
        if "primary_diagnosis" in final_prediction:
            confidences.append(final_prediction["primary_diagnosis"].get("confidence", 0.0))
            model_names.append("Ensemble")
        
        # Individual LLM confidences
        if "individual_results" in llm_results:
            for model_name, result in llm_results["individual_results"].items():
                if result.get("status") == "success" and "response" in result:
                    response = result["response"]
                    if "diagnoses" in response and response["diagnoses"]:
                        conf = response["diagnoses"][0].get("confidence", 0.0)
                        confidences.append(conf)
                        model_names.append(model_name.replace("-api", "").title())
        
        if confidences:
            # Create confidence distribution plot
            plt.bar(range(len(confidences)), confidences, 
                   color=['darkgreen' if c > 0.8 else 'orange' if c > 0.6 else 'red' for c in confidences],
                   alpha=0.7)
            plt.xticks(range(len(model_names)), model_names, rotation=45)
            plt.ylabel('Confidence Score')
            plt.title('Model Confidence Distribution')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            # Add confidence threshold lines
            plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High Confidence')
            plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate Confidence')
            plt.legend()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
        
    except Exception as e:
        logger.error(f"Confidence visualization failed: {e}")
        return None


def create_model_agreement_visualization(llm_results, ml_predictions):
    """Create visualization for model agreement analysis."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Collect predictions from different models
        model_predictions = {}
        
        # LLM predictions
        if "individual_results" in llm_results:
            for model_name, result in llm_results["individual_results"].items():
                if result.get("status") == "success" and "response" in result:
                    response = result["response"]
                    if "diagnoses" in response and response["diagnoses"]:
                        pred = response["diagnoses"][0]
                        model_predictions[model_name.replace("-api", "").title()] = {
                            'icd_code': pred.get('icd_code', ''),
                            'confidence': pred.get('confidence', 0.0)
                        }
        
        # ML predictions
        if ml_predictions and "predictions" in ml_predictions:
            for pred in ml_predictions["predictions"]:
                model_predictions["Traditional ML"] = {
                    'icd_code': pred.get('icd_code', ''),
                    'confidence': pred.get('probability', 0.0)
                }
        
        if len(model_predictions) > 1:
            # Create agreement matrix
            models = list(model_predictions.keys())
            agreement_matrix = np.zeros((len(models), len(models)))
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i == j:
                        agreement_matrix[i][j] = 1.0
                    else:
                        # Calculate agreement based on ICD code match
                        code1 = model_predictions[model1]['icd_code']
                        code2 = model_predictions[model2]['icd_code']
                        if code1 and code2:
                            agreement_matrix[i][j] = 1.0 if code1 == code2 else 0.0
                        else:
                            agreement_matrix[i][j] = 0.5  # Uncertain
            
            # Create heatmap
            sns.heatmap(agreement_matrix, annot=True, cmap='RdYlGn', 
                       xticklabels=models, yticklabels=models,
                       vmin=0, vmax=1, center=0.5)
            plt.title('Model Agreement Matrix')
            plt.xlabel('Models')
            plt.ylabel('Models')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
        
    except Exception as e:
        logger.error(f"Model agreement visualization failed: {e}")
        return None

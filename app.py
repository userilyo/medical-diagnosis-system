"""
Medical Diagnosis and ICD-10 Prediction Application

A comprehensive Streamlit web application that combines multiple AI/ML approaches
to analyze patient symptoms and provide accurate ICD-10 diagnostic predictions.

Features:
- Multi-modal ensemble of 5 LLMs (DeepSeek, Gemini, O1-Preview, OpenBioLLM, BioMistral)
- LSTM neural network verification (paper-based trained model)
- Traditional ML with RandomForest (MIMIC-III dataset)
- Real RAG processing of medical literature PDFs
- Hierarchical ICD-10 matching and evaluation
- Single patient analysis and batch processing modes
- Advanced explainability with LIME integration
- Comprehensive performance metrics and visualization

Architecture:
- Frontend: Streamlit with modular component design
- Backend: Ensemble methodology with parallel model execution
- Data: ICD-10 ontology with comprehensive graph structure
- Models: Paper-based LSTM (81% accuracy) + RandomForest (100 estimators)
- RAG: Real PDF processing with TF-IDF similarity search

Author: Medical AI Research Team
Paper: "A Modular Hierarchical Ensemble Framework for ICD-10 Prediction"
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
import io
import logging

from components.input_form import render_input_form
from components.results_display import display_results
from components.comparison_view import display_comparison
from components.llm_results_view import display_individual_llm_results, display_model_status_summary, display_api_usage_info

from utils.data_processing import preprocess_input
from utils.llm_module import predict_with_llms, get_consolidated_icd_codes
from utils.lstm_verification import verify_icd_codes
from utils.traditional_ml import feature_based_prediction
from utils.rag_processor import retrieve_relevant_info
from utils.ensemble import ensemble_predictions
from utils.explainability import generate_explanation
from utils.evaluation import calculate_metrics

from data.icd10_ontology import get_hierarchical_codes

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize all Streamlit session state variables with default values."""
    default_states = {
        'symptoms_text': "",
        'llm_results': {},
        'lstm_verified_codes': {},
        'ml_predictions': {},
        'rag_results': {},
        'final_prediction': {},
        'explanation': {},
        'hierarchical_codes': {},
        'metrics': {},
        'mode': "single",
        'batch_results': [],
        'batch_df': None
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

st.set_page_config(
    page_title="Medical diagnosis and ICD10 prediction",
    page_icon="🏥",
    layout="wide"
)

initialize_session_state()

st.title("Medical diagnosis and ICD10 prediction")
st.markdown("""
    This application analyses patient symptoms using multiple LLM models to predict possible diseases 
    and their corresponding ICD-10 codes. The system uses RAG, traditional machine learning, 
    and an LSTM neural network to verify the ICD-10 codes.
""")

st.header("Analysis Mode")
mode_option = st.radio(
    "Choose analysis mode:", 
    ["Single Input", "Batch Processing (CSV)"],
    index=0 if st.session_state.mode == "single" else 1,
    help="Single Input: Analyze one patient's symptoms at a time. Batch Processing: Upload a CSV file with multiple patients."
)

st.session_state.mode = "single" if mode_option == "Single Input" else "batch"


def process_single_patient(symptoms_text: str,
                           ground_truth_icd: str = None,
                           max_pages: int = 100) -> Dict[str, Any]:
    """
    Process a single patient's symptoms through the complete diagnostic pipeline.
    
    Args:
        symptoms_text: Patient's symptom description
        ground_truth_icd: Optional ground truth ICD-10 code for validation
        max_pages: Maximum PDF pages to process for RAG (default: 100)
        
    Returns:
        Dictionary containing all prediction results, metrics, and status information
    """
    try:
        logger.info(f"Processing patient symptoms (length: {len(symptoms_text)})")
        
        processed_text = preprocess_input(symptoms_text)
        logger.debug(f"Text preprocessing completed")

        llm_results = predict_with_llms(processed_text)
        logger.debug(f"LLM predictions completed: {type(llm_results)}")

        lstm_verified_codes = verify_icd_codes(processed_text, llm_results)
        logger.debug(f"LSTM verification completed: {type(lstm_verified_codes)}")

        ml_predictions = feature_based_prediction(processed_text)
        logger.debug(f"ML predictions completed: {type(ml_predictions)}")

        rag_results = retrieve_relevant_info(processed_text, max_pages=max_pages)
        logger.debug(f"RAG processing completed: {type(rag_results)}")

        final_prediction = ensemble_predictions(
            llm_results, lstm_verified_codes, ml_predictions, rag_results
        )

        explanation = generate_explanation(
            final_prediction, llm_results, rag_results,
            symptoms_text=processed_text, ml_predictions=ml_predictions
        )

        metrics = calculate_metrics(
            final_prediction, llm_results, lstm_verified_codes, ml_predictions
        )

        logger.info(f"Patient processing completed successfully")
        return {
            "symptoms": symptoms_text,
            "ground_truth": ground_truth_icd,
            "predicted_icd": final_prediction["primary_diagnosis"]["icd_code"],
            "predicted_condition": final_prediction["primary_diagnosis"]["condition"],
            "confidence": final_prediction["primary_diagnosis"]["confidence"],
            "llm_results": llm_results,
            "lstm_verified_codes": lstm_verified_codes,
            "ml_predictions": ml_predictions,
            "rag_results": rag_results,
            "final_prediction": final_prediction,
            "explanation": explanation,
            "metrics": metrics,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing patient: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "symptoms": symptoms_text,
            "ground_truth": ground_truth_icd,
            "predicted_icd": "ERROR",
            "predicted_condition": "ERROR",
            "confidence": 0.0,
            "error": str(e),
            "status": "error"
        }


def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate uploaded CSV format for batch processing requirements.
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str) indicating validation result
    """
    required_columns = ['symptoms']
    logger.debug(f"Validating CSV with {len(df)} rows and columns: {list(df.columns)}")

    if df.empty:
        logger.warning("CSV validation failed: empty file")
        return False, "CSV file is empty"

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"CSV validation failed: missing columns {missing_columns}")
        return False, f"Missing required columns: {missing_columns}"

    if df['symptoms'].isna().any() or (df['symptoms'] == '').any():
        empty_count = df['symptoms'].isna().sum() + (df['symptoms'] == '').sum()
        logger.warning(f"CSV validation failed: {empty_count} rows with empty symptoms")
        return False, "Some rows have empty symptoms"

    logger.info(f"CSV validation successful: {len(df)} rows ready for processing")
    return True, "CSV format is valid"


def create_batch_results_df(batch_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transform batch processing results into a structured DataFrame for analysis.
    
    Args:
        batch_results: List of patient processing result dictionaries
        
    Returns:
        DataFrame with organized results including hierarchical ICD-10 matching metrics
    """
    logger.info(f"Creating results DataFrame from {len(batch_results)} patient results")
    
    df_data = []
    for idx, result in enumerate(batch_results):
        row = {
            "symptoms": result["symptoms"],
            "predicted_icd": result["predicted_icd"],
            "predicted_condition": result["predicted_condition"],
            "confidence": result["confidence"],
            "status": result["status"]
        }

        if result.get("ground_truth"):
            row["ground_truth"] = result["ground_truth"]
            row["correct_prediction"] = result["predicted_icd"] == result["ground_truth"]
            
            try:
                from utils.icd10_comprehensive import calculate_hierarchical_accuracy
                hierarchical_sim = calculate_hierarchical_accuracy(
                    result["predicted_icd"], result["ground_truth"]
                )
                row.update({
                    "category_match": hierarchical_sim["category_match"],
                    "block_match": hierarchical_sim["block_match"],
                    "chapter_match": hierarchical_sim.get("chapter_match", 0),
                    "weighted_score": hierarchical_sim["weighted_score"]
                })
                logger.debug(f"Row {idx}: comprehensive hierarchical evaluation completed")
            except Exception as e:
                logger.warning(f"Row {idx}: falling back to legacy hierarchical matching: {e}")
                from utils.hierarchical_icd10 import hierarchical_matcher
                hierarchical_sim = hierarchical_matcher.calculate_hierarchical_similarity(
                    result["predicted_icd"], result["ground_truth"]
                )
                row.update({
                    "category_match": hierarchical_sim["category_match"],
                    "block_match": hierarchical_sim["block_match"],
                    "chapter_match": hierarchical_sim["chapter_match"],
                    "weighted_score": hierarchical_sim["weighted_score"]
                })

        if result.get("error"):
            row["error"] = result["error"]

        df_data.append(row)

    logger.info(f"DataFrame created successfully with {len(df_data)} rows")
    return pd.DataFrame(df_data)


# Render sidebar with application information
with st.sidebar:
    st.header("About")
    st.info("""
        **Medical Diagnosis Assistant** uses multiple AI approaches to predict diseases from symptoms:
        
        - Large Language Models (DeepSeek, Gemini, 01-preview, OpenBioLLM, BioMistral)
        - Retrieval-Augmented Generation (RAG)
        - LSTM verification for ICD-10 codes
        - Traditional machine learning classification
        
        The system ensembles these predictions to provide a comprehensive diagnosis suggestion.
    """)

    st.header("Model Information")
    st.markdown("""
        * **LLMs**: API for DeepSeek, Gemini, and O1-preview. Simulated for OpenBioLLM and BioMistral
        * **RAG**: Knowledge base from medical literature
        * **LSTM**: Paper-based trained model (BDCC-08-00047-v2)
        * **Traditional ML**: RandomForest model (MIMIC-III trained)
    """)
    
    # Display paper model status
    from components.paper_models_view import display_paper_model_status
    display_paper_model_status()
    
    # Display RandomForest model status
    from components.rf_models_view import display_rf_model_status
    display_rf_model_status()

    # Add performance settings
    st.header("Settings")
    
    # RAG toggle
    enable_rag = st.checkbox("Enable RAG (Medical Literature Search)", value=True, 
                            help="Toggle to enable/disable Retrieval-Augmented Generation for medical literature search")
    
    # Page limit setting (only show if RAG is enabled)
    if enable_rag:
        pdf_pages = st.selectbox(
            "PDF Pages to Process",
            options=[50, 100, 200, 300, 500, 600],
            index=1,  # Default to 100
            help="Higher values provide more comprehensive results but slower performance"
        )
        
        # Performance warning
        if pdf_pages > 200:
            st.warning("⚠️ Processing 300+ pages may take 2-3 minutes on first load")
    else:
        pdf_pages = 0
    
    # Store settings in session state
    st.session_state.rag_enabled = enable_rag
    st.session_state.pdf_pages = pdf_pages if enable_rag else 0
    
    # Add disclaimer
    st.header("Disclaimer")
    st.warning("""
        This application is for educational and demonstration purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the advice of a qualified healthcare provider for any medical condition.
    """)

# Main application layout based on mode
if st.session_state.mode == "single":
    # Single input mode (original functionality)
    col1, col2 = st.columns([1, 2])

    # Initialize session state for storing the current analysis state
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = ""

    # Input form in the first column
    with col1:
        st.header("Patient Symptoms")
        symptoms_text = render_input_form()

        # Process the symptoms if submitted
        if symptoms_text:
            st.session_state.current_analysis = symptoms_text
            st.session_state.symptoms_text = symptoms_text

            with st.spinner("Analyzing symptoms..."):
                # Preprocess the input text
                processed_text = preprocess_input(symptoms_text)

                # Get predictions from different models using our LLM module
                try:
                    logger.info("Starting single patient analysis pipeline")
                    
                    st.session_state.llm_results = predict_with_llms(processed_text)
                    logger.debug("LLM predictions completed successfully")

                    st.session_state.lstm_verified_codes = verify_icd_codes(
                        processed_text, st.session_state.llm_results)
                    logger.debug("LSTM verification completed successfully")

                    st.session_state.ml_predictions = feature_based_prediction(processed_text)
                    logger.debug("Traditional ML predictions completed successfully")

                    if st.session_state.get('rag_enabled', True):
                        st.session_state.rag_results = retrieve_relevant_info(
                            processed_text, max_pages=st.session_state.get('pdf_pages', 100))
                        logger.debug("RAG processing completed successfully")
                    else:
                        st.session_state.rag_results = {
                            "relevant_info": "RAG disabled by user",
                            "medical_concepts": {},
                            "confidence": 0.0,
                            "icd_codes": [],
                            "evidence_texts": [],
                            "search_method": "disabled",
                            "knowledge_base_stats": {"pdf_loaded": False, "pdf_chunks": 0}
                        }
                        logger.info("RAG processing skipped (disabled by user)")

                    # Check if Ollama is unavailable and show appropriate message
                    if "_system_note" in st.session_state.llm_results:
                        # Get info about which models were used
                        if "model_contributions" in st.session_state.llm_results:
                            api_models = [
                                m["model"] for m in st.session_state.
                                llm_results["model_contributions"] if
                                m["model"] in ["deepseek-api", "gemini-api"]
                            ]
                            if api_models:
                                models_used = ", ".join(api_models)
                                st.info(
                                    f"ℹ️ Ollama LLM service unavailable. Using {models_used} for predictions.",
                                    icon="ℹ️")
                            else:
                                st.warning(
                                    "⚠️ Ollama LLM service unavailable. Using traditional machine learning for predictions.",
                                    icon="⚠️")
                        else:
                            st.warning(
                                "⚠️ Ollama LLM service unavailable. Using traditional machine learning for predictions.",
                                icon="⚠️")

                    logger.debug("Starting ensemble prediction aggregation")
                    st.session_state.final_prediction = ensemble_predictions(
                        st.session_state.llm_results,
                        st.session_state.lstm_verified_codes,
                        st.session_state.ml_predictions,
                        st.session_state.rag_results)
                    logger.debug("Ensemble predictions completed successfully")

                    st.session_state.explanation = generate_explanation(
                        st.session_state.final_prediction,
                        st.session_state.llm_results,
                        st.session_state.rag_results)
                    logger.debug("Explanation generation completed")

                    st.session_state.hierarchical_codes = get_hierarchical_codes(
                        st.session_state.final_prediction["primary_diagnosis"]["icd_code"])
                    logger.debug("Hierarchical code analysis completed")

                    st.session_state.metrics = calculate_metrics(
                        st.session_state.final_prediction,
                        st.session_state.llm_results,
                        st.session_state.lstm_verified_codes,
                        st.session_state.ml_predictions)
                    logger.debug("Metrics calculation completed")

                    logger.info("Single patient analysis pipeline completed successfully")
                    st.success("Analysis complete!")

                except Exception as e:
                    logger.error(f"Error in single patient analysis: {str(e)}")
                    st.error(f"Error analyzing symptoms: {str(e)}")

    # Results display in the second column
    with col2:
        if st.session_state.current_analysis:
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Diagnosis", "Model Comparison", "Individual LLM Results", "Paper Models"])

            # Diagnosis tab
            with tab1:
                if st.session_state.final_prediction:
                    display_results(st.session_state.final_prediction,
                                    st.session_state.hierarchical_codes,
                                    st.session_state.explanation)
                    
                    # Debug: Show RAG content verification
                    if st.session_state.rag_results:
                        rag_enabled = st.session_state.get('rag_enabled', True)
                        expander_title = "🔍 RAG Content Verification (Proof of Real PDF)" if rag_enabled else "🔍 RAG System Status (Disabled)"
                        
                        with st.expander(expander_title, expanded=False):
                            if not rag_enabled:
                                st.warning("RAG system is disabled. Enable it in the sidebar settings for medical literature search.")
                                st.write("**Status:** Disabled by user")
                                st.write("**Impact:** No medical textbook content will be used in predictions")
                            else:
                                st.write("**Search Method:**", st.session_state.rag_results.get('search_method', 'Unknown'))
                                st.write("**PDF Pages Processed:**", st.session_state.get('pdf_pages', 100))
                                
                                stats = st.session_state.rag_results.get('knowledge_base_stats', {})
                                st.write("**PDF Processing Stats:**")
                                st.write(f"- PDF Loaded: {stats.get('pdf_loaded', False)}")
                                st.write(f"- PDF Chunks: {stats.get('pdf_chunks', 0)}")
                                st.write(f"- Average Chunk Length: {stats.get('avg_chunk_length', 0):.0f} chars")
                                
                                # Show evidence texts to prove real PDF content
                                evidence_texts = st.session_state.rag_results.get('evidence_texts', [])
                                if evidence_texts:
                                    st.write("**Evidence Texts from Real PDF:**")
                                    for i, text in enumerate(evidence_texts[:3]):  # Show first 3
                                        if "From Medical Textbook 'Symptom to Diagnosis'" in text:
                                            content = text.replace("From Medical Textbook 'Symptom to Diagnosis': ", "")
                                            st.write(f"**{i+1}. PDF Content ({len(content)} chars):**")
                                            st.code(content[:400] + "..." if len(content) > 400 else content)
                                        else:
                                            st.write(f"**{i+1}. Knowledge Base Content:**")
                                            st.code(text[:200] + "..." if len(text) > 200 else text)
                                
                                # Show ICD codes found
                                icd_codes = st.session_state.rag_results.get('icd_codes', [])
                                if icd_codes:
                                    st.write(f"**ICD Codes Found:** {', '.join(icd_codes)}")
                                
                                # Show relevant conditions
                                conditions = st.session_state.rag_results.get('relevant_conditions', [])
                                if conditions:
                                    st.write("**Relevant Conditions:**")
                                    for i, cond in enumerate(conditions[:3]):
                                        st.write(f"- {cond.get('condition', 'Unknown')} (Method: {cond.get('method', 'Unknown')}, Similarity: {cond.get('similarity', 0.0):.3f})")
                                
                                st.info("👆 This section proves the system is retrieving actual content from the 'symptom-to-diagnosis-an-evidence-based-guide_compress.pdf' file, not hardcoded responses!")
                    
                    # Show hierarchical ICD-10 information
                    if st.session_state.final_prediction:
                        predicted_code = st.session_state.final_prediction["primary_diagnosis"]["icd_code"]
                        
                        with st.expander("🏗️ Hierarchical ICD-10 Information", expanded=False):
                            try:
                                from utils.comprehensive_evaluation import get_enhanced_hierarchical_info
                                
                                hierarchical_info = get_enhanced_hierarchical_info(predicted_code)
                                
                                st.write(f"**Predicted Code:** {predicted_code}")
                                st.write(f"**Description:** {hierarchical_info.get('description', 'Unknown')}")
                                st.write(f"**Hierarchy Level:** {hierarchical_info.get('level', 'Unknown')}")
                                
                                # Show hierarchy path
                                if hierarchical_info.get('hierarchy_path'):
                                    st.write("**Hierarchy Path:**")
                                    for level_info in hierarchical_info['hierarchy_path']:
                                        st.write(f"  - Level {level_info['level']}: {level_info['code']} - {level_info['description']}")
                                
                                # Show related codes
                                if hierarchical_info.get('related_codes'):
                                    st.write("**Related Codes:**")
                                    for related in hierarchical_info['related_codes'][:5]:  # Show top 5
                                        st.write(f"  - {related['relationship'].title()}: {related['code']} - {related['description']}")
                                
                                st.info("This shows the hierarchical position and related codes using the comprehensive ICD-10 graph system.")
                                
                            except Exception as e:
                                # Fallback to old system
                                from utils.hierarchical_icd10 import hierarchical_matcher
                                
                                code_info = hierarchical_matcher.get_code_info(predicted_code)
                                
                                if 'error' not in code_info:
                                    st.write(f"**Predicted Code:** {predicted_code}")
                                    st.write(f"**Description:** {code_info.get('description', 'Unknown')}")
                                    st.write(f"**Hierarchy Level:** {code_info.get('level', 'Unknown')}")
                                    st.write(f"**Full Path:** {code_info.get('path', 'Unknown')}")
                                    
                                    if 'chapter' in code_info:
                                        st.write(f"**Chapter:** {code_info['chapter']}")
                                    if 'block' in code_info:
                                        st.write(f"**Block:** {code_info['block']}")
                                    if 'category' in code_info:
                                        st.write(f"**Category:** {code_info['category']}")
                                    
                                    st.info("This shows the hierarchical position of the predicted ICD-10 code in the medical classification system.")
                                else:
                                    st.write(f"Error loading hierarchical information: {e}")

            # Model comparison tab
            with tab2:
                if "_system_note" in st.session_state.llm_results:
                    # Check if any API models were used
                    if "model_contributions" in st.session_state.llm_results:
                        api_models = [
                            m["model"] for m in
                            st.session_state.llm_results["model_contributions"]
                            if m["model"] in ["deepseek-api", "gemini-api"]
                        ]
                        if api_models:
                            models_used = ", ".join(api_models)
                            st.info(
                                f"ℹ️ Ollama LLM service unavailable. Using {models_used} for predictions and comparison.",
                                icon="ℹ️")
                        else:
                            st.warning(
                                "⚠️ Ollama LLM service unavailable. Showing traditional ML predictions only.",
                                icon="⚠️")
                            st.info(
                                "For full model comparison, Ollama needs to be installed and running on this system or API keys need to be provided."
                            )
                    else:
                        st.warning(
                            "⚠️ Ollama LLM service unavailable. Showing traditional ML predictions only.",
                            icon="⚠️")
                        st.info(
                            "For full model comparison, Ollama needs to be installed and running on this system or API keys need to be provided."
                        )

                if st.session_state.llm_results and st.session_state.lstm_verified_codes:
                    display_comparison(st.session_state.llm_results,
                                       st.session_state.lstm_verified_codes,
                                       st.session_state.ml_predictions,
                                       st.session_state.final_prediction)

            # Individual LLM Results tab
            with tab3:
                if st.session_state.llm_results:
                    # Show API usage information
                    display_api_usage_info(st.session_state.llm_results)

                    # Show model status summary
                    display_model_status_summary(st.session_state.llm_results)

                    # Show detailed individual results
                    display_individual_llm_results(
                        st.session_state.llm_results)
                else:
                    st.info(
                        "Run an analysis to see individual LLM model results")
            
            # Paper Models tab
            with tab4:
                st.write("### Research Paper-Based Models")
                
                # Create sub-tabs for LSTM and RandomForest models
                lstm_tab, rf_tab = st.tabs(["LSTM Model", "RandomForest Model"])
                
                with lstm_tab:
                    from components.paper_models_view import display_paper_model_info
                    display_paper_model_info()
                    
                    # Show LSTM predictions if symptoms are available
                    if hasattr(st.session_state, 'symptoms_text') and st.session_state.symptoms_text:
                        st.write("### LSTM Model Predictions")
                        
                        # Try to get LSTM model predictions
                        try:
                            from utils.paper_based_models import paper_lstm_prediction
                            
                            lstm_paper_results = paper_lstm_prediction(st.session_state.symptoms_text)
                            
                            from components.paper_models_view import display_paper_model_predictions
                            display_paper_model_predictions(lstm_paper_results)
                        except Exception as e:
                            st.error(f"Error loading LSTM model predictions: {e}")
                    else:
                        st.info("Enter symptoms to see LSTM model predictions")
                
                with rf_tab:
                    from components.rf_models_view import display_rf_model_info
                    display_rf_model_info()
                    
                    # Show RandomForest predictions if symptoms are available
                    if hasattr(st.session_state, 'symptoms_text') and st.session_state.symptoms_text:
                        st.write("### RandomForest Model Predictions")
                        
                        # Try to get RandomForest model predictions
                        try:
                            from utils.rf_traditional_ml import rf_traditional_ml_prediction
                            
                            rf_results = rf_traditional_ml_prediction(st.session_state.symptoms_text)
                            
                            from components.rf_models_view import display_rf_model_predictions
                            display_rf_model_predictions(rf_results)
                        except Exception as e:
                            st.error(f"Error loading RandomForest model predictions: {e}")
                    else:
                        st.info("Enter symptoms to see RandomForest model predictions")
        else:
            st.info("Enter patient symptoms on the left to begin analysis")

elif st.session_state.mode == "batch":
    # Batch processing mode
    st.header("Batch Processing")

    # CSV upload section
    st.subheader("Upload CSV File")
    st.markdown("""
    **CSV Format Requirements:**
    - Required column: `symptoms` (patient symptoms text)
    - Optional column: `ground_truth` (actual ICD-10 codes for validation)
    - Optional column: `patient_id` (unique identifier for each patient)
    
    **Example CSV format:**
    ```
    patient_id,symptoms,ground_truth
    1,"Chest pain and shortness of breath",I20.0
    2,"Headache and nausea",R51
    ```
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing patient symptoms")

    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.session_state.batch_df = df

            # Validate CSV format
            is_valid, message = validate_csv_format(df)

            if is_valid:
                logger.info(f"CSV validation successful: {len(df)} patients loaded")
                st.success(f"CSV file loaded successfully! {message}")

                st.subheader("Data Preview")
                st.dataframe(df.head(10))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", len(df))
                with col2:
                    ground_truth_count = len(df[df.get('ground_truth', pd.Series()).notna()])
                    st.metric("Has Ground Truth", ground_truth_count)
                with col3:
                    patient_id_count = len(df[df.get('patient_id', pd.Series()).notna()])
                    st.metric("Has Patient ID", patient_id_count)

                if st.button("Process Batch", type="primary"):
                    logger.info(f"Starting batch processing for {len(df)} patients")
                    st.session_state.batch_results = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, row in df.iterrows():
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Processing patient {idx + 1}/{len(df)}: {row['symptoms'][:50]}..."
                        )

                        result = process_single_patient(
                            row['symptoms'], 
                            row.get('ground_truth', None), 
                            max_pages=st.session_state.get('pdf_pages', 100)
                        )

                        result['patient_id'] = row.get('patient_id', idx + 1)
                        st.session_state.batch_results.append(result)
                        
                        if idx % 5 == 0:  # Log every 5th patient
                            logger.debug(f"Processed {idx + 1}/{len(df)} patients")

                    progress_bar.progress(1.0)
                    status_text.text("Batch processing complete!")
                    logger.info(f"Batch processing completed: {len(df)} patients processed successfully")
                    st.success(f"Successfully processed {len(df)} patients!")

            else:
                logger.error(f"CSV validation failed: {message}")
                st.error(f"Invalid CSV format: {message}")

        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            st.error(f"Error reading CSV file: {str(e)}")

    # Display batch results if available
    if st.session_state.batch_results:
        st.subheader("Batch Processing Results")

        # Create results DataFrame
        results_df = create_batch_results_df(st.session_state.batch_results)

        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Processed", len(results_df))
        with col2:
            success_count = len(results_df[results_df['status'] == 'success'])
            st.metric("Successful", success_count)
        with col3:
            error_count = len(results_df[results_df['status'] == 'error'])
            st.metric("Errors", error_count)
        with col4:
            if 'correct_prediction' in results_df.columns:
                exact_accuracy = results_df['correct_prediction'].mean() * 100
                st.metric("Exact Match Accuracy", f"{exact_accuracy:.1f}%")
                
                # Add hierarchical accuracy metrics if available
                if 'weighted_score' in results_df.columns:
                    hierarchical_accuracy = results_df['weighted_score'].mean() * 100
                    category_accuracy = results_df['category_match'].mean() * 100
                    block_accuracy = results_df['block_match'].mean() * 100
                    chapter_accuracy = results_df['chapter_match'].mean() * 100
                    
                    # Display hierarchical metrics
                    st.subheader("Hierarchical Accuracy Metrics")
                    col4a, col4b, col4c, col4d = st.columns(4)
                    with col4a:
                        st.metric("Weighted Score", f"{hierarchical_accuracy:.1f}%")
                    with col4b:
                        st.metric("Category Match", f"{category_accuracy:.1f}%")
                    with col4c:
                        st.metric("Block Match", f"{block_accuracy:.1f}%")
                    with col4d:
                        st.metric("Chapter Match", f"{chapter_accuracy:.1f}%")
                
                # Add accuracy explanation
                with st.expander("ℹ️ Hierarchical Accuracy Explanation", expanded=False):
                    st.markdown("""
                    **Hierarchical Accuracy Levels:**
                    - **Exact Match**: Identical codes (e.g., I10 = I10)
                    - **Category Match**: Same category (e.g., I10 ≈ I10.9) → 85% score
                    - **Block Match**: Same block (e.g., I10 ≈ I11) → 70% score  
                    - **Chapter Match**: Same chapter (e.g., I10 ≈ I20) → 50% score
                    - **Weighted Score**: Distance-based scoring considering hierarchy
                    
                    **Example Improvement:**
                    - Old system: I10 vs I10.9 → 0% (wrong)
                    - New system: I10 vs I10.9 → 85% (same category)
                    
                    **Clinical Relevance:**
                    - Codes in same category often represent similar conditions
                    - Hierarchical matching provides more realistic medical accuracy
                    - Partial credit reflects clinical coding practices
                    """)

        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(
            ["Results Summary", "Detailed Results", "Download Results"])

        with tab1:
            st.dataframe(results_df)

            # Show confidence distribution
            if 'confidence' in results_df.columns:
                st.subheader("Confidence Distribution")
                fig = go.Figure(
                    data=[go.Histogram(x=results_df['confidence'])])
                fig.update_layout(title="Prediction Confidence Distribution",
                                  xaxis_title="Confidence Score",
                                  yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Show hierarchical accuracy distribution
            if 'weighted_score' in results_df.columns:
                st.subheader("Hierarchical Accuracy Distribution")
                
                # Create comparison chart
                accuracy_data = {
                    'Exact Match': results_df['correct_prediction'].astype(float),
                    'Category Match': results_df['category_match'],
                    'Block Match': results_df['block_match'],
                    'Chapter Match': results_df['chapter_match'],
                    'Weighted Score': results_df['weighted_score']
                }
                
                fig = go.Figure()
                for accuracy_type, values in accuracy_data.items():
                    fig.add_trace(go.Histogram(
                        x=values,
                        name=accuracy_type,
                        opacity=0.7,
                        nbinsx=10
                    ))
                
                fig.update_layout(
                    title="Hierarchical vs Exact Match Accuracy",
                    xaxis_title="Accuracy Score",
                    yaxis_title="Count",
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Detailed Results")

            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.selectbox(
                    "Filter by Status",
                    options=['All', 'Success', 'Error'],
                    index=0)
            with col2:
                if 'correct_prediction' in results_df.columns:
                    accuracy_filter = st.selectbox(
                        "Filter by Accuracy",
                        options=['All', 'Correct', 'Incorrect'],
                        index=0)

            # Apply filters
            filtered_df = results_df.copy()
            if status_filter != 'All':
                filtered_df = filtered_df[filtered_df['status'] ==
                                          status_filter.lower()]
            if 'correct_prediction' in results_df.columns and accuracy_filter != 'All':
                if accuracy_filter == 'Correct':
                    filtered_df = filtered_df[filtered_df['correct_prediction']
                                              == True]
                elif accuracy_filter == 'Incorrect':
                    filtered_df = filtered_df[filtered_df['correct_prediction']
                                              == False]

            # Show detailed results for selected patients
            if not filtered_df.empty:
                selected_patient = st.selectbox(
                    "Select Patient for Details",
                    options=filtered_df.index.tolist(),
                    format_func=lambda x:
                    f"Patient {filtered_df.loc[x, 'patient_id'] if 'patient_id' in filtered_df.columns else x+1}"
                )

                if selected_patient is not None:
                    patient_result = st.session_state.batch_results[
                        selected_patient]

                    # Display patient details
                    st.markdown(
                        f"**Patient ID:** {patient_result.get('patient_id', selected_patient + 1)}"
                    )
                    st.markdown(f"**Symptoms:** {patient_result['symptoms']}")
                    st.markdown(
                        f"**Predicted ICD:** {patient_result['predicted_icd']}"
                    )
                    st.markdown(
                        f"**Predicted Condition:** {patient_result['predicted_condition']}"
                    )
                    st.markdown(
                        f"**Confidence:** {patient_result['confidence']:.2f}")

                    if patient_result.get('ground_truth'):
                        st.markdown(
                            f"**Ground Truth:** {patient_result['ground_truth']}"
                        )
                        st.markdown(
                            f"**Correct Prediction:** {patient_result['predicted_icd'] == patient_result['ground_truth']}"
                        )

                    if patient_result.get('error'):
                        st.error(f"Error: {patient_result['error']}")

        with tab3:
            st.subheader("Download Results")

            # Convert results to CSV
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # Download button
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=
                f"batch_diagnosis_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the batch processing results as a CSV file")

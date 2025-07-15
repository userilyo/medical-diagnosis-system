import streamlit as st
import pandas as pd
from typing import Dict, Any

def display_individual_llm_results(llm_results: Dict[str, Any]):
    """
    Display individual results from each LLM model.
    
    Args:
        llm_results: Results from LLM predictions containing individual_results
    """
    if "individual_results" not in llm_results:
        st.warning("No individual LLM results available")
        return
    
    st.subheader("Individual LLM Model Results")
    
    # Create tabs for each model
    individual_results = llm_results["individual_results"]
    model_names = list(individual_results.keys())
    
    if not model_names:
        st.info("No LLM models were called")
        return
    
    # Create tabs for each model
    tabs = st.tabs(model_names)
    
    for i, model_name in enumerate(model_names):
        with tabs[i]:
            model_result = individual_results[model_name]
            status = model_result["status"]
            
            # Show model status with color coding
            if status == "success":
                st.success(f"✅ {model_name} - Success")
            elif status == "failed":
                st.error(f"❌ {model_name} - Failed")
            elif status == "error":
                st.error(f"🔥 {model_name} - Error")
                
            # Show error if exists
            if model_result.get("error"):
                st.error(f"Error: {model_result['error']}")
            
            # Show the response if available
            response = model_result.get("response", {})
            if "diagnoses" in response:
                st.write("**Diagnoses from this model:**")
                
                # Create a dataframe for better display
                diagnoses_data = []
                for diagnosis in response["diagnoses"]:
                    diagnoses_data.append({
                        "ICD-10 Code": diagnosis.get("icd_code", "N/A"),
                        "Condition": diagnosis.get("condition", "N/A"),
                        "Confidence": f"{diagnosis.get('confidence', 0):.2f}",
                        "Reasoning": diagnosis.get("reasoning", "N/A")
                    })
                
                if diagnoses_data:
                    df = pd.DataFrame(diagnoses_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No diagnoses returned from this model")
            else:
                st.info("No valid diagnoses returned from this model")

def display_model_status_summary(llm_results: Dict[str, Any]):
    """
    Display a summary of model statuses.
    
    Args:
        llm_results: Results from LLM predictions containing model_status
    """
    if "model_status" not in llm_results:
        return
    
    st.subheader("Model Status Summary")
    
    model_status = llm_results["model_status"]
    
    # Create columns for status display
    cols = st.columns(len(model_status))
    
    for i, (model_name, status) in enumerate(model_status.items()):
        with cols[i]:
            if status == "success":
                st.metric(
                    label=model_name,
                    value="✅ Success",
                    delta="Working"
                )
            elif status == "failed":
                st.metric(
                    label=model_name,
                    value="❌ Failed",
                    delta="No response",
                    delta_color="inverse"
                )
            elif status == "error":
                st.metric(
                    label=model_name,
                    value="🔥 Error",
                    delta="Connection issue",
                    delta_color="inverse"
                )

def display_api_usage_info(llm_results: Dict[str, Any]):
    """
    Display information about which APIs are being used.
    
    Args:
        llm_results: Results from LLM predictions
    """
    if "models" not in llm_results:
        return
    
    models = llm_results["models"]
    
    # Categorize models
    api_models = [m for m in models if "-api" in m]
    local_models = [m for m in models if "-api" not in m and m != "fallback"]
    fallback_models = [m for m in models if m == "fallback"]
    
    st.subheader("LLM Configuration")
    
    if api_models:
        st.info(f"**API Models Used:** {', '.join(api_models)}")
    
    if local_models:
        st.info(f"**Local Models Used:** {', '.join(local_models)}")
    
    if fallback_models:
        st.warning(f"**Fallback Used:** Traditional ML predictions only")
    
    if "_system_note" in llm_results:
        st.warning(f"**System Note:** {llm_results['_system_note']}")
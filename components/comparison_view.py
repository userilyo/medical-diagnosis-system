import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List

def display_comparison(
    llm_results: Dict[str, Any],
    lstm_verified_codes: Dict[str, Any],
    ml_predictions: Dict[str, Any],
    final_prediction: Dict[str, Any]
):

    st.header("Model Comparison")
    st.markdown("Comparing predictions across different models helps understand the confidence and agreement in the diagnosis.")
    
    # Create a comparison table
    st.subheader("ICD-10 Code Predictions")
    
    # Prepare data for comparison table
    comparison_data = []
    
    # Add LLM predictions
    if "predictions" in llm_results and llm_results["predictions"]:
        for pred in llm_results["predictions"]:
            comparison_data.append({
                "ICD Code": pred.get("icd_code", ""),
                "Condition": pred.get("condition", ""),
                "Source": "LLMs",
                "Confidence": pred.get("confidence", 0.0)
            })
    
    # Add LSTM verified codes
    if "verified_codes" in lstm_verified_codes and lstm_verified_codes["verified_codes"]:
        for code in lstm_verified_codes["verified_codes"]:
            comparison_data.append({
                "ICD Code": code.get("icd_code", ""),
                "Condition": "", # LSTM model might not provide condition names
                "Source": "LSTM Verification",
                "Confidence": code.get("confidence", 0.0)
            })
    
    # Add ML predictions
    if "predictions" in ml_predictions and ml_predictions["predictions"]:
        for pred in ml_predictions["predictions"]:
            comparison_data.append({
                "ICD Code": pred.get("icd_code", ""),
                "Condition": "", # ML model might not provide condition names
                "Source": "Traditional ML",
                "Confidence": pred.get("probability", 0.0)
            })
    
    # Add final prediction
    if "primary_diagnosis" in final_prediction:
        primary = final_prediction["primary_diagnosis"]
        comparison_data.append({
            "ICD Code": primary.get("icd_code", ""),
            "Condition": primary.get("condition", ""),
            "Source": "Final (Ensemble)",
            "Confidence": primary.get("confidence", 0.0)
        })
    
    if comparison_data:
        # Convert to DataFrame and display
        comparison_df = pd.DataFrame(comparison_data)
        
        # Define a function to highlight rows based on confidence level
        def highlight_confidence(s):
            return ['background-color: rgba(76, 175, 80, {})'.format(val) 
                    if col == 'Confidence' else '' for col, val in s.items()]
        
        # Display styled table
        st.dataframe(comparison_df.style.apply(highlight_confidence, axis=1))
        
        # Create a bar chart for confidence comparison
        st.subheader("Confidence Comparison")
        
        # Group by ICD code and source, taking the max confidence for each
        grouped_data = {}
        for row in comparison_data:
            key = (row["ICD Code"], row["Source"])
            if key not in grouped_data or row["Confidence"] > grouped_data[key]:
                grouped_data[key] = row["Confidence"]
        
        # Prepare data for the bar chart
        icd_codes = sorted(set(k[0] for k in grouped_data.keys()))
        sources = sorted(set(k[1] for k in grouped_data.keys()))
        
        # Create traces for each source
        fig = go.Figure()
        for source in sources:
            y_values = [grouped_data.get((code, source), 0) for code in icd_codes]
            fig.add_trace(go.Bar(
                x=icd_codes,
                y=y_values,
                name=source
            ))
        
        # Update layout
        fig.update_layout(
            title="Confidence by Model for Each ICD-10 Code",
            xaxis_title="ICD-10 Code",
            yaxis_title="Confidence",
            legend_title="Model Source",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No comparison data available.")

"""
Component for displaying paper-based LSTM model information and results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

def display_paper_model_info():
    """
    Display information about the paper-based LSTM model.
    """
    st.write("### 📄 Research Paper-Based LSTM Model")
    st.write("**Paper Reference:** BDCC-08-00047-v2: International Classification of Diseases Prediction from MIMIC-III Clinical Text")
    
    try:
        from utils.paper_based_models import get_paper_model_info
        model_info = get_paper_model_info()
        
        # Display LSTM model information
        st.write("**LSTM Model Information:**")
        lstm_info = model_info.get("lstm_model", {})
        if lstm_info.get("file_exists"):
            st.success(f"✅ LSTM model loaded ({lstm_info.get('file_size_mb', 0):.1f} MB)")
            if "vocab_size" in lstm_info:
                st.write(f"- Vocabulary size: {lstm_info['vocab_size']:,}")
            if "embedding_dim" in lstm_info:
                st.write(f"- Embedding dimension: {lstm_info['embedding_dim']}")
            if "output_dim" in lstm_info:
                st.write(f"- Output classes: {lstm_info['output_dim']}")
        else:
            st.warning("⚠️ LSTM model file not found, using default")
        
        # Display paper results
        st.write("**📊 Paper Results:**")
        paper_results = model_info.get("paper_results", {})
        
        # Create performance chart
        metrics = ['Top 10 Accuracy', 'Top 50 Accuracy', 'F1 Score (Top 10)', 'F1 Score (Top 50)']
        values = [81, 68, 81, 66]  # From paper
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title='LSTM Model Performance from Research Paper',
            xaxis_title='Metric',
            yaxis_title='Accuracy (%)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display hyperparameters
        with st.expander("🔧 LSTM Model Hyperparameters (from paper)"):
            hyperparams = model_info.get("hyperparameters", {})
            lstm_params = hyperparams.get("lstm", {})
            for key, value in lstm_params.items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        return True
        
    except Exception as e:
        st.error(f"Error displaying paper model information: {e}")
        return False

def display_paper_model_predictions(lstm_results: Dict[str, Any]):
    """
    Display predictions from paper-based LSTM model.
    
    Args:
        lstm_results: Results from paper-based LSTM model
    """
    st.subheader("🧠 Paper-Based LSTM Model Predictions")
    
    st.write("**LSTM Model Results (81% accuracy from paper):**")
    
    if lstm_results.get("predictions"):
        # Create DataFrame for display
        predictions_df = pd.DataFrame(lstm_results["predictions"])
        predictions_df['confidence'] = predictions_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            predictions_df[['rank', 'icd_code', 'confidence']].rename(columns={
                'rank': 'Rank',
                'icd_code': 'ICD Code',
                'confidence': 'Confidence'
            }),
            use_container_width=True
        )
        
        # Create confidence chart
        fig = go.Figure(data=[
            go.Bar(
                x=[p['icd_code'] for p in lstm_results["predictions"]],
                y=[p['confidence'] for p in lstm_results["predictions"]],
                marker_color='#1f77b4'
            )
        ])
        
        fig.update_layout(
            title='LSTM Model Confidence Scores',
            xaxis_title='ICD Code',
            yaxis_title='Confidence',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top prediction details
        top_pred = lstm_results["predictions"][0]
        st.success(f"**Top Prediction:** {top_pred['icd_code']} (Confidence: {top_pred['confidence']:.1%})")
        
    else:
        st.warning("No predictions available from LSTM model")
    
    # Display model metadata
    with st.expander("🔍 Model Details"):
        st.write(f"**Model Type:** {lstm_results.get('model_type', 'Unknown')}")
        st.write(f"**Method:** {lstm_results.get('method', 'Unknown')}")
        st.write(f"**Paper Reference:** {lstm_results.get('paper_reference', 'Unknown')}")
        st.write(f"**Model Accuracy:** {lstm_results.get('model_accuracy', 'Unknown')}")

def display_paper_model_status():
    """
    Display the status of paper-based LSTM model.
    """
    try:
        from utils.paper_based_models import get_paper_model_info
        model_info = get_paper_model_info()
        
        lstm_info = model_info.get("lstm_model", {})
        if lstm_info.get("file_exists"):
            st.success("✅ LSTM model loaded")
        else:
            st.warning("⚠️ LSTM model not found")
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
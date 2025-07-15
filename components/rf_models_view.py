"""
Component for displaying RandomForest model information and results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

def display_rf_model_info():
    """
    Display information about the RandomForest model.
    """
    st.write("### 🌲 RandomForest Traditional ML Model")
    st.write("**Dataset:** MIMIC-III Clinical Text (Same as LSTM model)")
    st.write("**Model Type:** RandomForest Classifier with TF-IDF Vectorization")
    
    try:
        from utils.rf_traditional_ml import get_rf_model_info
        model_info = get_rf_model_info()
        
        if model_info.get("model_loaded"):
            st.success("✅ RandomForest model loaded successfully")
            
            # Display model specifications
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Specifications:**")
                st.write(f"- Model Type: {model_info.get('model_type', 'N/A')}")
                st.write(f"- Dataset: {model_info.get('dataset', 'N/A')}")
                st.write(f"- N Estimators: {model_info.get('n_estimators', 'N/A')}")
                st.write(f"- Max Depth: {model_info.get('max_depth', 'N/A')}")
                st.write(f"- Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
            
            with col2:
                st.write("**Text Processing:**")
                st.write(f"- Vectorizer: {model_info.get('vectorizer_type', 'N/A')}")
                st.write(f"- Vocabulary Size: {model_info.get('vocabulary_size', 'N/A'):,}")
                st.write(f"- N Features: {model_info.get('n_features', 'N/A')}")
                st.write(f"- N Classes: {model_info.get('n_classes', 'N/A')}")
                st.write(f"- Vectorizer Size: {model_info.get('vectorizer_size_mb', 0):.1f} MB")
            
            # Display sample classes
            if model_info.get('classes_sample'):
                st.write("**Sample ICD Classes:**")
                classes_str = ", ".join(str(c) for c in model_info['classes_sample'])
                st.code(classes_str)
            
            return True
        else:
            st.error(f"❌ Failed to load RandomForest model: {model_info.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        st.error(f"Error displaying RandomForest model information: {e}")
        return False

def display_rf_model_predictions(rf_results: Dict[str, Any]):
    """
    Display predictions from RandomForest model.
    
    Args:
        rf_results: Results from RandomForest model
    """
    st.subheader("🌲 RandomForest Model Predictions")
    
    if rf_results.get("status") == "success":
        st.write("**RandomForest Results (MIMIC-III trained):**")
        
        # Show top prediction
        top_pred = rf_results.get("predicted_icd", "N/A")
        top_conf = rf_results.get("confidence", 0.0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top Prediction", top_pred)
        with col2:
            st.metric("Confidence", f"{top_conf:.1%}")
        
        # Show all predictions if available
        if rf_results.get("predictions"):
            st.write("**All Predictions:**")
            
            # Create DataFrame for display
            predictions_df = pd.DataFrame(rf_results["predictions"])
            
            # Format confidence as percentage
            predictions_df['confidence_pct'] = predictions_df['confidence'].apply(lambda x: f"{x:.1%}")
            
            # Display table with descriptions
            display_columns = ['rank', 'icd_code', 'confidence_pct']
            column_names = {
                'rank': 'Rank',
                'icd_code': 'ICD Code',
                'confidence_pct': 'Confidence'
            }
            
            # Add description column if available
            if 'description' in predictions_df.columns:
                display_columns.append('description')
                column_names['description'] = 'Description'
            
            # Display table
            st.dataframe(
                predictions_df[display_columns].rename(columns=column_names),
                use_container_width=True
            )
            
            # Create confidence chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[str(p['icd_code']) for p in rf_results["predictions"]],
                    y=[p['confidence'] for p in rf_results["predictions"]],
                    marker_color='#2E8B57'  # Forest green color
                )
            ])
            
            fig.update_layout(
                title='RandomForest Model Confidence Scores',
                xaxis_title='ICD Code',
                yaxis_title='Confidence',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display model metadata
        with st.expander("🔍 Model Details"):
            st.write(f"**Method:** {rf_results.get('method', 'Unknown')}")
            st.write(f"**Model Type:** {rf_results.get('model_type', 'Unknown')}")
            st.write(f"**Dataset:** {rf_results.get('dataset', 'Unknown')}")
            st.write(f"**Total Predictions:** {rf_results.get('total_predictions', 0)}")
            st.write(f"**Status:** {rf_results.get('status', 'Unknown')}")
    
    else:
        st.error(f"❌ RandomForest prediction failed: {rf_results.get('error', 'Unknown error')}")
        
        # Show debugging information
        with st.expander("Debug Information"):
            st.write(f"**Status:** {rf_results.get('status', 'Unknown')}")
            st.write(f"**Error:** {rf_results.get('error', 'Unknown')}")

def display_rf_model_status():
    """
    Display the status of RandomForest model.
    """
    try:
        from utils.rf_traditional_ml import get_rf_model_info
        model_info = get_rf_model_info()
        
        if model_info.get("model_loaded"):
            st.success("✅ RandomForest model loaded")
        else:
            st.warning("⚠️ RandomForest model not loaded")
            
    except Exception as e:
        st.error(f"❌ Error loading RandomForest model: {str(e)}")
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List

def display_results(final_prediction: Dict[str, Any], hierarchical_codes: Dict[str, Any], explanation: Dict[str, Any]):
    """
    Display the final prediction results.
    
    Args:
        final_prediction: The final diagnosis prediction
        hierarchical_codes: Hierarchical ICD-10 information
        explanation: Explanation for the prediction
    """
    st.header("Diagnosis Results")
    
    # Display primary diagnosis
    if "primary_diagnosis" in final_prediction:
        primary = final_prediction["primary_diagnosis"]
        
        st.subheader("Primary Diagnosis")
        primary_col1, primary_col2 = st.columns([3, 1])
        
        with primary_col1:
            st.markdown(f"**{primary.get('condition', 'Unknown Condition')}**")
            st.markdown(f"ICD-10 Code: **{primary.get('icd_code', '')}**")
            
            if "description" in hierarchical_codes:
                st.markdown(f"Description: {hierarchical_codes['description']}")
        
        with primary_col2:
            # Display confidence as a gauge chart
            confidence = primary.get("confidence", 0.0)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkgreen" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "lightblue"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Display hierarchical information
    if hierarchical_codes and "full_hierarchy" in hierarchical_codes:
        st.subheader("ICD-10 Classification Hierarchy")
        
        hierarchy = hierarchical_codes["full_hierarchy"]
        if hierarchy:
            # Create a hierarchical display
            for i, level in enumerate(hierarchy):
                # Indent more for deeper levels
                indent = "&nbsp;" * (i * 4)
                if level.get("level") == "chapter":
                    st.markdown(f"{indent}📚 **Chapter:** {level.get('title', '')} ({level.get('range', '')})", unsafe_allow_html=True)
                elif level.get("level") == "subcategory":
                    st.markdown(f"{indent}📋 **Category:** {level.get('title', '')}", unsafe_allow_html=True)
                elif level.get("level") == "code":
                    st.markdown(f"{indent}🏷️ **Code {level.get('code', '')}:** {level.get('description', '')}", unsafe_allow_html=True)
        else:
            st.info("No hierarchical information available for this code.")
    
    # Display differential diagnoses
    if "differential_diagnoses" in final_prediction and final_prediction["differential_diagnoses"]:
        st.subheader("Differential Diagnoses")
        
        for i, diff in enumerate(final_prediction["differential_diagnoses"]):
            with st.expander(f"{diff.get('condition', 'Unknown')} (ICD-10: {diff.get('icd_code', '')})"):
                st.markdown(f"**Confidence:** {diff.get('confidence', 0.0) * 100:.1f}%")
                # You could add more detailed information about each differential diagnosis here
    
    # Display medical literature insights from our expanded knowledge base
    if "relevant_info" in explanation:
        st.subheader("Medical Literature Insights")
        st.markdown(explanation["relevant_info"])
        
        # Display medical concepts if available
        if "medical_concepts" in explanation:
            with st.expander("📋 Extracted Medical Concepts"):
                concepts = explanation["medical_concepts"]
                
                if "symptoms" in concepts and concepts["symptoms"]:
                    st.markdown("**Identified Symptoms:**")
                    symptoms_list = ""
                    for symptom in concepts["symptoms"]:
                        symptoms_list += f"- {symptom}\n"
                    st.markdown(symptoms_list)
                
                if "conditions" in concepts and concepts["conditions"]:
                    st.markdown("**Identified Conditions:**")
                    conditions_list = ""
                    for condition in concepts["conditions"]:
                        conditions_list += f"- {condition}\n"
                    st.markdown(conditions_list)
                
                if "measurements" in concepts and concepts["measurements"]:
                    st.markdown("**Clinical Measurements:**")
                    measurements_list = ""
                    for measurement in concepts["measurements"]:
                        measurements_list += f"- {measurement}\n"
                    st.markdown(measurements_list)
                    
        # Display code descriptions if available
        if "code_descriptions" in explanation:
            with st.expander("🔍 ICD-10 Code Details"):
                for code, description in explanation["code_descriptions"].items():
                    st.markdown(f"**{code}**: {description}")
    
    # Display explanation with enhanced features
    if explanation:
        st.subheader("Explanation")
        
        # Show explainability method used
        method = explanation.get("explainability_method", "basic")
        if method == "enhanced":
            st.success("🔬 Enhanced AI Explainability with LIME Analysis")
        else:
            st.info("📊 Standard Explainability Analysis")
        
        if "reasoning" in explanation:
            st.markdown(f"**Reasoning:**")
            st.markdown(explanation["reasoning"])
        
        if "key_factors" in explanation and explanation["key_factors"]:
            st.markdown(f"**Key Factors:**")
            factors_list = ""
            for factor in explanation["key_factors"]:
                factors_list += f"- {factor}\n"
            st.markdown(factors_list)
        
        # Display LIME explanation if available
        if "lime_explanation" in explanation and explanation["lime_explanation"]:
            with st.expander("🔍 LIME Text Feature Analysis"):
                lime_exp = explanation["lime_explanation"]
                st.markdown(f"**Method:** {lime_exp['method']}")
                st.markdown(lime_exp["explanation_text"])
                
                if "feature_importance" in lime_exp:
                    st.markdown("**Top Contributing Text Features:**")
                    for feature in lime_exp["feature_importance"][:5]:
                        direction = "🟢 Positive" if feature["direction"] == "positive" else "🔴 Negative"
                        st.markdown(f"- **{feature['feature']}** ({direction}): {feature['importance']:.3f}")
        
        # Display feature importance visualization
        if "feature_importance_plot" in explanation and explanation["feature_importance_plot"]:
            st.subheader("🎯 Feature Importance Analysis")
            import base64
            img_data = base64.b64decode(explanation["feature_importance_plot"])
            st.image(img_data, caption="Text Features Contributing to Diagnosis", use_column_width=True)
        
        # Display confidence visualization
        if "confidence_plot" in explanation and explanation["confidence_plot"]:
            st.subheader("📊 Model Confidence Distribution")
            import base64
            img_data = base64.b64decode(explanation["confidence_plot"])
            st.image(img_data, caption="Confidence Scores Across Different Models", use_column_width=True)
        
        # Display model agreement visualization
        if "agreement_plot" in explanation and explanation["agreement_plot"]:
            st.subheader("🤝 Model Agreement Analysis")
            import base64
            img_data = base64.b64decode(explanation["agreement_plot"])
            st.image(img_data, caption="Agreement Matrix Between Different AI Models", use_column_width=True)

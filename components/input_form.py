import streamlit as st

# Callback to clear input text and previous analysis results
def clear_text():
    st.session_state["symptom_input"] = ""
    
    # Clear all analysis results
    st.session_state.current_analysis = ""
    st.session_state.llm_results = {}
    st.session_state.lstm_verified_codes = {}
    st.session_state.ml_predictions = {}
    st.session_state.rag_results = {}
    st.session_state.final_prediction = {}
    st.session_state.explanation = {}
    st.session_state.hierarchical_codes = {}
    st.session_state.metrics = {}

def render_input_form():
    """
    Render the form for symptom input and returns symptom text if submitted, empty string otherwise
    """
    # Clear button outside the form
    st.button("Clear Input", on_click=clear_text)
            
    # The form for input
    with st.form(key="symptom_form"):
        # Input field for symptoms with key to track state
        symptoms_text = st.text_area(
            label="Enter patient symptoms",
            placeholder="Example: Patient reports persistent headache for 3 days, dizziness, elevated blood pressure (160/95), and occasional chest discomfort.",
            height=150,
            key="symptom_input"
        )
        
        # Submit button
        col1, col2 = st.columns([1, 1])
        
        with col1:
            submit_button = st.form_submit_button(label="Analyse Symptoms")
        
        if submit_button and symptoms_text:
            return symptoms_text
    
    # Return empty string if not submitted
    return ""

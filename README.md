# Medical Diagnosis AI System

A comprehensive medical diagnostic AI application that leverages multi-modal machine learning techniques to provide accurate ICD-10 diagnostic predictions with explainable AI features.

## Features

- **Multi-Modal AI Ensemble**: Combines 5 LLM models (DeepSeek, Gemini, OpenAI O1 Preview, OpenBioLLM, BioMistral)
- **LSTM Verification**: Neural network-based ICD-10 code verification
- **Real RAG Processing**: Actual medical literature processing from PDF textbooks
- **LIME Explainability**: Feature importance analysis with visualizations
- **Performance Controls**: Configurable PDF processing (50-600 pages)
- **Batch Processing**: CSV upload for multiple patient analysis
- **Interactive Web Interface**: Streamlit-based application

## Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
git clone <repository-url>
cd medical-diagnosis-system
python setup_local.py
```

### Option 2: Manual Setup
```bash
git clone <repository-url>
cd medical-diagnosis-ai-ensemble

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run the application
streamlit run app.py
```

## Configuration

1. **API Key**: Set up your OpenRouter API key in `.env` file:
   ```
   OPENROUTER_API_KEY=your_openrouter_key
   ```
   
   **Note**: We use OpenRouter API to access all LLM models (DeepSeek, Gemini, OpenAI O1 Preview, OpenBioLLM, BioMistral). You only need one OpenRouter API key, not individual API keys for each model.

2. **Performance Settings**: Configure in the sidebar:
   - RAG toggle (enable/disable medical literature search)
   - PDF page limits (50-600 pages)
   - Processing mode (speed vs comprehensiveness)

## Usage

### Single Patient Analysis
1. Enter patient symptoms in the text area
2. Click "Analyze Symptoms"
3. View ensemble predictions with confidence scores
4. Explore individual model results and explanations

### Batch Processing
1. Upload CSV file with columns: `symptoms`, `ground_truth_icd` (optional)
2. Process multiple patients simultaneously
3. Download results with performance metrics

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with modular component architecture
- **Models**: PyTorch LSTM, scikit-learn ML, multiple LLMs
- **Data**: Real medical literature processing with RAG
- **Explainability**: LIME integration with visualizations

## Testing

Run individual tests:
```bash
python tests/quick_pdf_test.py
python tests/test_rag_content.py
python tests/verify_pdf_content.py
```

## Debugging Local Issues

If you're getting different results locally (like "R69 - Illness, unspecified" for all inputs), run the debug script:

```bash
python debug_local.py
```

### Common Issues:

1. **Missing API Key**: If `OPENROUTER_API_KEY` is not set, the system falls back to simulated models
2. **Environment Variables**: Make sure `.env` file is in the project root and `python-dotenv` is installed
3. **Different Results**: Verify your API key is working by checking the debug output
4. **SSL Certificate Errors**: 
   ```
   [nltk_data] Error loading punkt: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
   ```
   - **Cause**: macOS SSL certificate verification issues
   - **Solution**: Run `python fix_ssl_certificates.py` or `/Applications/Python\ 3.11/Install\ Certificates.command`
5. **Missing Enhanced RAG**: 
   ```
   SentenceTransformers not available - using TF-IDF search only
   FAISS not available - using sklearn cosine similarity
   ```
   - **Solution**: Run `python install_missing_packages.py` to install sentence-transformers and faiss-cpu

### Quick Fix:
```bash
# 1. Make sure you have the API key in .env
echo "OPENROUTER_API_KEY=your_actual_key_here" > .env

# 2. Install dotenv if missing
pip install python-dotenv

# 3. Run the app
streamlit run app.py
```

## Development

The project uses a modular architecture:
- `app.py`: Main Streamlit application
- `components/`: UI components
- `utils/`: Core processing modules
- `models/`: ML model implementations
- `data/`: ICD-10 ontology and knowledge base
- `tests/`: Testing utilities

## Academic Paper

This work supports the research paper:
"Modular Hierarchical Ensemble Learning for ICD-10 Prediction: A Framework Integrating Large Language Models, LSTM Verification, Traditional ML, and Retrieval-Augmented Generation"

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Disclaimer

This application is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical conditions.

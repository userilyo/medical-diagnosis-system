# Medical Diagnosis and ICD-10 Prediction System

## Overview

This is a comprehensive medical diagnosis and ICD-10 prediction system built with Streamlit. The application combines multiple AI/ML approaches to analyze patient symptoms and provide accurate ICD-10 diagnostic codes. It uses an ensemble methodology that integrates Large Language Models (LLMs), LSTM neural networks, traditional machine learning, and Retrieval-Augmented Generation (RAG) to provide robust diagnostic predictions.

The system now supports both single patient analysis and batch processing of multiple patients via CSV upload, making it suitable for both individual consultations and research/evaluation scenarios.

## System Architecture
![Design architecture](https://github.com/userilyo/multi-modal-medical-ai/blob/main/attached_assets/architecture_diag.png)

### Frontend Architecture
- **Framework**: Streamlit-based web application
- **Layout**: Wide layout with modular components
- **State Management**: Streamlit session state for maintaining analysis results across user interactions
- **Interactive Elements**: Forms, buttons, tabs, and charts for user interaction

### Backend Architecture
- **Ensemble Approach**: Multiple AI/ML models working together
- **Modular Design**: Separate components for different prediction methods
- **Data Processing Pipeline**: Text preprocessing, feature extraction, and prediction aggregation
- **Explainability Layer**: Generates explanations for diagnostic decisions

### Model Architecture
1. **LLM Integration**: Multiple language models for symptom analysis
2. **LSTM Verification**: PyTorch-based LSTM model for ICD-10 code verification
3. **Traditional ML**: Feature-based prediction using numpy/sklearn-style algorithms
4. **RAG System**: Knowledge base retrieval for medical literature matching

## Key Components

### Core Application (`app.py`)
- Main Streamlit application entry point
- Session state management for all prediction results
- Orchestrates the complete prediction pipeline
- Mode selector for single vs batch processing
- CSV upload and validation for batch analysis
- Progress tracking and results visualization

### Component Modules (`components/`)
- **Input Form**: User interface for symptom input with clear functionality
- **Results Display**: Primary diagnosis presentation with confidence visualization
- **Comparison View**: Side-by-side comparison of different model predictions
- **LLM Results View**: Individual LLM model results with status tracking

### Utility Modules (`utils/`)
- **Data Processing**: Text cleaning, medical term extraction, and preprocessing
- **LLM Module**: Interface for multiple language models with fallback mechanisms
- **LSTM Verification**: Neural network-based code verification
- **Traditional ML**: Feature-based prediction algorithms
- **RAG Processor**: Medical knowledge base retrieval and matching
- **Ensemble**: Prediction aggregation and confidence scoring
- **Explainability**: Advanced reasoning generation with LIME integration, dynamic explanations, and comprehensive visualizations
- **Evaluation**: Model performance metrics and agreement analysis

### Data Layer (`data/`)
- **ICD-10 Ontology**: Hierarchical ICD-10 code structure and relationships
- **Medical Knowledge Base**: Symptom-to-diagnosis mappings from medical literature

### Model Layer (`models/`)
- **LSTM Model**: PyTorch-based neural network for ICD-10 verification
- **Feature Extractors**: Traditional ML feature engineering components

## Data Flow

1. **Input Processing**: User enters symptoms → Text preprocessing and cleaning
2. **Multi-Model Prediction**: 
   - LLMs analyze symptoms and suggest ICD-10 codes
   - Traditional ML extracts features and predicts codes
   - RAG system retrieves relevant medical literature
3. **LSTM Verification**: Validates predicted codes against learned patterns
4. **Ensemble Aggregation**: Combines predictions with confidence weighting
5. **Explanation Generation**: Creates reasoning for final diagnosis
6. **Results Display**: Shows primary diagnosis, confidence scores, and explanations

## External Dependencies

### Core Dependencies
- **streamlit**: Web application framework
- **plotly**: Interactive charts and visualizations
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **torch**: PyTorch for neural networks
- **nltk**: Natural language processing
- **requests**: HTTP client for API calls

### Model Dependencies
- **PyTorch**: LSTM neural network implementation
- **NLTK**: Text tokenization and preprocessing
- **Plotly**: Confidence visualization gauges

### API Integrations
- **Multiple LLM APIs**: Configured for various language model providers
- **Fallback Mechanisms**: Graceful degradation when APIs are unavailable

## Deployment Strategy

### Development Setup
- Streamlit development server for local testing
- Modular architecture allows individual component testing
- Mock data and simulated responses for development without external dependencies

### Production Considerations
- **Scalability**: Session state management for multi-user environments
- **Error Handling**: Comprehensive error catching and user feedback
- **Performance**: Lazy loading of models and caching of results
- **Security**: Input validation and sanitization

### Infrastructure Requirements
- Python 3.8+ runtime environment
- GPU support recommended for LSTM model training
- Sufficient memory for multiple ML models
- Network connectivity for LLM API calls

## Changelog

- February-March 2025: Initial Setup. REAL RAG Implementation with Actual PDF Processing and Proper Chunking
  - Implemented authentic PDF processing using pdfplumber and PyPDF2 libraries for real document retrieval
  - Created RealRAGProcessor class that actually processes the "symptom-to-diagnosis-an-evidence-based-guide_compress.pdf" file
  - Added efficient text chunking with overlap preservation for medical context continuity
  - Implemented TF-IDF-based similarity search on real PDF content chunks (not hardcoded text)
  - Added ICD-10 code extraction from actual PDF content using pattern matching
  - Created hybrid RAG system combining real PDF content with traditional knowledge base
  - Added comprehensive error handling and fallback mechanisms when PDF processing fails
  - Implemented page-by-page processing with memory-efficient chunking (processes 100 pages max)
  - Added real-time PDF statistics tracking (chunks processed, ICD codes found, text length)
  - Fixed confidence score calculation issues in ensemble system allowing scores above 50%
  - Enhanced LSTM verification with better pattern matching and confidence scaling
- April 2025: Major explainability system enhancement with LIME integration and advanced visualizations
  - Implemented LIME (Local Interpretable Model-agnostic Explanations) for traditional ML feature importance analysis
  - Added dynamic reasoning generation that enhances LLM-based explanations with confidence context
  - Created comprehensive visualization suite including:
    - Feature importance plots showing key text features contributing to diagnosis
    - Model confidence distribution charts across different AI models
    - Model agreement matrix visualizing consensus between different prediction systems
  - Enhanced reasoning templates with structured markdown formatting and medical literature context
  - Integrated matplotlib and seaborn for professional-grade medical visualization
  - Added explainability method indicators (basic vs enhanced) to inform users of analysis depth
  - Improved error handling and fallback mechanisms for visualization failures
  - Enhanced results display component with expandable LIME analysis sections
- May 2025: Completely overhauled RAG processor with vector-based search capabilities and comprehensive medical knowledge base
  - Enhanced from ~20 conditions to comprehensive medical coverage across all major organ systems
  - Implemented vector embeddings using SentenceTransformers for semantic similarity search
  - Added FAISS indexing for optimized large-scale medical literature search
  - Included fallback TF-IDF search for systems without vector capabilities
  - Expanded knowledge base to include cardiovascular, respiratory, neurological, gastrointestinal, musculoskeletal, endocrine, genitourinary, hematological, infectious, and mental health conditions
  - Added scalable architecture for loading external medical databases and literature
- June 2025: Added batch processing functionality with CSV upload support for multiple patient analysis and reviewed set up.
- July 2025: Implemented graph-based hierarchical ICD-10 matching system for improved accuracy evaluation
  - Added comprehensive hierarchical ICD-10 tree structure with chapters, blocks, categories, and subcategories
  - Implemented multi-level accuracy metrics: exact match, category match, block match, chapter match
  - Added weighted scoring system that gives partial credit based on hierarchical distance
  - Enhanced batch processing with hierarchical accuracy visualizations
  - Added hierarchical code information display for single patient analysis
  - Improved accuracy from strict exact matching to clinically relevant hierarchical matching
  - Example: I10 vs I10.9 now scores 85% instead of 0% (same category match)

## User Preferences

Preferred communication style: Simple, everyday language.

## Academic Paper Considerations

User is considering publishing this work with the title:
"Modular Ensemble Framework for ICD-10 Prediction: Integrating Large Language Models, LSTM Verification, Traditional ML, and Retrieval-Augmented Generation in a Streamlit Clinical Application."


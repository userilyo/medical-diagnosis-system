#!/usr/bin/env python3
"""
Enhanced Local setup script for the Medical Diagnosis System on MacBook Pro
With comprehensive ICD-10 graph, paper-based LSTM, and RandomForest models
"""

import os
import sys
import subprocess
import platform
import json
import urllib.request

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_packages():
    """Install required packages"""
    packages = [
        "streamlit>=1.29.0",
        "plotly>=5.17.0", 
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "torch>=2.1.0",
        "nltk>=3.8.1",
        "requests>=2.31.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "pdfplumber>=0.9.0",
        "PyPDF2>=3.0.1",
        "trafilatura>=1.6.2",
        "lime>=0.2.0.1",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "joblib>=1.4.2",
        "networkx>=3.4.2"
    ]
    
    print("📦 Installing packages...")
    pip_executable = os.path.join("venv", "bin", "pip") if platform.system() != "Windows" else os.path.join("venv", "Scripts", "pip")
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([pip_executable, "install", package], check=True)
    
    print("✅ All packages installed")

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    python_executable = os.path.join("venv", "bin", "python") if platform.system() != "Windows" else os.path.join("venv", "Scripts", "python")
    
    nltk_script = """
import ssl
import nltk

# Fix SSL certificate issues on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print('NLTK data downloaded successfully!')
"""
    
    subprocess.run([python_executable, "-c", nltk_script], check=True)
    print("✅ NLTK data downloaded")

def create_env_file():
    """Create .env file template"""
    if not os.path.exists(".env"):
        print("📝 Creating .env file template...")
        with open(".env", "w") as f:
            f.write("""# OpenRouter API Key (required - provides access to all LLM models)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Note: We use OpenRouter to access all models:
# - DeepSeek API
# - Gemini API  
# - OpenAI O1 Preview
# - OpenBioLLM
# - BioMistral
# You only need the OpenRouter API key above.
""")
        print("✅ .env file created - Please add your API keys!")
    else:
        print("✅ .env file already exists")

def create_streamlit_config():
    """Create Streamlit configuration"""
    os.makedirs(".streamlit", exist_ok=True)
    
    config_content = """[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
    
    with open(".streamlit/config.toml", "w") as f:
        f.write(config_content)
    
    print("✅ Streamlit configuration created")

def create_run_script():
    """Create run script for easy startup"""
    script_content = f"""#!/bin/bash
# Activate virtual environment and run the app

{get_activation_command()}
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
if not os.getenv('OPENROUTER_API_KEY'):
    print('❌ ERROR: OPENROUTER_API_KEY not found in .env file')
    print('Please add your OpenRouter API key to the .env file')
    print('Without this key, the system will use simulated models and return incorrect results')
    exit(1)
else:
    print('✅ Environment variables loaded successfully')
    print('✅ OpenRouter API key found - real models will be used')
"
streamlit run app.py
"""
    
    with open("run_local.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("run_local.sh", 0o755)
    print("✅ Run script created (run_local.sh)")

def main():
    """Main setup function"""
    print("🔧 Setting up Medical Diagnosis System for MacBook Pro...")
    print("=" * 50)
    
    if not check_python_version():
        return
    
    create_virtual_environment()
    install_packages()
    download_nltk_data()
    create_env_file()
    create_streamlit_config()
    create_run_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Add your API keys to the .env file")
    print("2. Run the app with: ./run_local.sh")
    print("   or manually: source venv/bin/activate && streamlit run app.py")
    print("\n📖 See LOCAL_SETUP_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()
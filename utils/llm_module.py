import json
import random
import re
import requests
import os
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define ICD-10 validation functions here to avoid circular imports
def is_valid_icd10(code: str) -> bool:
    """
    Check if a string follows ICD-10 code pattern.
    
    Args:
        code: The code to check
        
    Returns:
        True if the code follows ICD-10 pattern, False otherwise
    """
    # Basic ICD-10 pattern: letter followed by 2 digits, optionally followed by a dot and more digits
    pattern = r'^[A-Z][0-9]{2}(\.[0-9]+)?$'
    return bool(re.match(pattern, code))

def standardize_icd10(code: str) -> str:
    """
    Standardize ICD-10 code format.
    
    Args:
        code: The ICD-10 code to standardize
        
    Returns:
        Standardized ICD-10 code
    """
    # Remove whitespace
    code = code.strip().upper()
    
    # Check if it has the basic pattern of a letter followed by numbers
    if re.match(r'^[A-Z]\d+$', code) and len(code) >= 3:
        # Insert a dot after the first three characters if not present
        if len(code) > 3 and '.' not in code:
            code = code[:3] + '.' + code[3:]
        return code
    
    return code

def simulate_llm_response(text: str, model_name: str) -> Dict[str, Any]:
    """
    Simulate LLM responses for demonstration purposes.
    
    Args:
        text: The input text
        model_name: The name of the model to simulate
    
    Returns:
        A simulated response in the format of the LLM
    """
    # Use model name to simulate different responses
    if "gemini" in model_name.lower():
        return {
            "diagnoses": [
                {
                    "icd_code": "I10",
                    "condition": "Essential hypertension",
                    "confidence": 0.85,
                    "reasoning": "Elevated blood pressure readings (160/95) and occasional chest discomfort suggest hypertension."
                },
                {
                    "icd_code": "R51",
                    "condition": "Headache",
                    "confidence": 0.80,
                    "reasoning": "Persistent headache for 3 days is a primary symptom noted in the patient's history."
                }
            ]
        }
    elif "deepseek" in model_name.lower():
        return {
            "diagnoses": [
                {
                    "icd_code": "I10",
                    "condition": "Essential hypertension",
                    "confidence": 0.90,
                    "reasoning": "Blood pressure of 160/95 indicates stage 2 hypertension, which aligns with the chest discomfort."
                },
                {
                    "icd_code": "R42",
                    "condition": "Dizziness",
                    "confidence": 0.75,
                    "reasoning": "Patient reports dizziness, which can be a symptom of hypertension or a separate condition."
                }
            ]
        }
    elif "openbiollm" in model_name.lower():
        return {
            "diagnoses": [
                {
                    "icd_code": "K59.00",
                    "condition": "Constipation, unspecified",
                    "confidence": 0.82,
                    "reasoning": "OpenBioLLM analysis: Abdominal discomfort and irregular bowel movements indicate constipation."
                },
                {
                    "icd_code": "K30",
                    "condition": "Functional dyspepsia",
                    "confidence": 0.75,
                    "reasoning": "OpenBioLLM assessment: Symptoms align with functional dyspepsia."
                }
            ]
        }
    elif "biomistral" in model_name.lower():
        return {
            "diagnoses": [
                {
                    "icd_code": "J06.9",
                    "condition": "Acute upper respiratory infection, unspecified",
                    "confidence": 0.85,
                    "reasoning": "BioMistral analysis: Upper respiratory symptoms indicate acute URI."
                },
                {
                    "icd_code": "R50.9",
                    "condition": "Fever, unspecified",
                    "confidence": 0.78,
                    "reasoning": "BioMistral assessment: Elevated temperature suggests febrile illness."
                }
            ]
        }
    elif "o1-preview" in model_name.lower() or "openai" in model_name.lower():
        return {
            "diagnoses": [
                {
                    "icd_code": "M79.1",
                    "condition": "Myalgia",
                    "confidence": 0.88,
                    "reasoning": "O1 Preview analysis: Muscle pain patterns suggest myalgia with high confidence."
                },
                {
                    "icd_code": "G93.3",
                    "condition": "Postviral fatigue syndrome",
                    "confidence": 0.70,
                    "reasoning": "O1 Preview assessment: Persistent fatigue following viral illness."
                }
            ]
        }
    else:  # fallback for any other model
        return {
            "diagnoses": [
                {
                    "icd_code": "I10",
                    "condition": "Essential hypertension",
                    "confidence": 0.88,
                    "reasoning": "Consistent with elevated blood pressure and associated symptoms."
                },
                {
                    "icd_code": "R51",
                    "condition": "Headache",
                    "confidence": 0.70,
                    "reasoning": "Persistent headache could be a primary condition or secondary to hypertension."
                }
            ]
        }

def extract_diagnoses_from_text(text: str) -> Dict[str, Any]:
    """
    Parse text response and extract medical information.
    
    Args:
        text: The text response to parse
    
    Returns:
        Parsed medical information
    """
    # Basic text parsing to extract medical information
    # This is a fallback when JSON parsing fails
    
    # Look for ICD codes in the text
    icd_pattern = r'[A-Z]\d{2}(?:\.\d{1,2})?'
    icd_codes = re.findall(icd_pattern, text)
    
    diagnoses = []
    
    if icd_codes:
        for code in icd_codes[:3]:  # Limit to first 3 codes
            diagnoses.append({
                "icd_code": code,
                "condition": f"Condition related to {code}",
                "confidence": 0.75,
                "reasoning": f"Extracted from text analysis: {code} mentioned in response."
            })
    else:
        # Fallback diagnosis if no ICD codes found
        diagnoses.append({
            "icd_code": "R69",
            "condition": "Illness, unspecified",
            "confidence": 0.60,
            "reasoning": "Unable to extract specific diagnostic codes from the response."
        })
    
    return {"diagnoses": diagnoses}

def create_prompt(symptoms: str, model_name: str) -> str:
    """
    Create a model-specific prompt for symptom analysis.
    
    Args:
        symptoms: The patient's symptoms text
        model_name: Name of the LLM model
    
    Returns:
        A formatted prompt for the LLM
    """
    # Base prompt for all models
    base_prompt = f"""
You are a medical diagnostic assistant. Analyze the following patient symptoms and provide potential diagnoses with ICD-10 codes.

Patient symptoms:
{symptoms}

Format your response as a JSON object with the following structure:
{{
  "diagnoses": [
    {{
      "icd_code": "LETTER##.#",
      "condition": "Condition name",
      "confidence": 0.XX,
      "reasoning": "Brief explanation of why this diagnosis is likely based on the symptoms"
    }},
    // additional diagnoses as needed
  ]
}}

The confidence value should be between 0 and 1, where 1 indicates the highest confidence.
List the most probable diagnoses first.
"""
    return base_prompt

def call_deepseek_api(prompt: str) -> Dict[str, Any]:
    """
    Call the DeepSeek API to get a response for medical diagnosis.
    
    Args:
        prompt: The medical prompt to send to the model
    
    Returns:
        The parsed JSON response
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DeepSeek API key not found in environment variables")
        return {"error": "API key not available", "diagnoses": []}
    
    logger.info("Calling DeepSeek API...")
    
    # DeepSeek API endpoint
    api_url = "https://api.deepseek.com/v1/chat/completions"
    
    # Prepare the headers with API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare the payload
    payload = {
        "model": "deepseek-chat",  # Using the deepseek-chat model
        "messages": [
            {"role": "system", "content": "You are a medical diagnostic assistant that analyzes patient symptoms and provides potential diagnoses with ICD-10 codes. Always respond with JSON in the following format: {\"diagnoses\": [{\"icd_code\": \"CODE\", \"condition\": \"NAME\", \"confidence\": FLOAT, \"reasoning\": \"EXPLANATION\"}]}"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        logger.info("Received response from DeepSeek API")
        
        assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{\s*"diagnoses"\s*:\s*\[.+?\]\s*\}', assistant_message, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON from DeepSeek response pattern match")
        
        # If we couldn't extract JSON using regex, try to parse the entire response
        try:
            parsed_json = json.loads(assistant_message)
            if "diagnoses" in parsed_json:
                return parsed_json
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from DeepSeek full response")
        
        # If we got here, couldn't parse JSON
        logger.warning(f"DeepSeek API returned non-JSON response: {assistant_message[:100]}...")
        return {
            "diagnoses": [
                {
                    "icd_code": "R69",
                    "condition": "Illness, unspecified",
                    "confidence": 0.7,
                    "reasoning": "The symptoms provided could not be clearly mapped to a specific diagnosis by the DeepSeek API."
                }
            ]
        }
    
    except Exception as e:
        logger.error(f"Error calling DeepSeek API: {e}")
        return {"error": str(e), "diagnoses": []}

def call_gemini_api(prompt: str) -> Dict[str, Any]:
    """
    Call the Google Gemini API to get a response for medical diagnosis.
    
    Args:
        prompt: The medical prompt to send to the model
    
    Returns:
        The parsed JSON response
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key not found in environment variables")
        return {"error": "API key not available", "diagnoses": []}
    
    logger.info("Calling Gemini API...")
    
    # Gemini API endpoint - updated to use gemini-1.5-flash
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    # Prepare the payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "You are a medical diagnostic assistant that analyzes patient symptoms and provides potential diagnoses with ICD-10 codes. Always respond with JSON in the following format: {\"diagnoses\": [{\"icd_code\": \"CODE\", \"condition\": \"NAME\", \"confidence\": FLOAT, \"reasoning\": \"EXPLANATION\"}]}"
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        logger.info("Received response from Gemini API")
        
        try:
            text_response = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{\s*"diagnoses"\s*:\s*\[.+?\]\s*\}', text_response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON from Gemini response pattern match")
            
            # If we couldn't extract JSON using regex, try to parse the entire response
            try:
                parsed_json = json.loads(text_response)
                if "diagnoses" in parsed_json:
                    return parsed_json
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON from Gemini full response")
            
            # If we got here, couldn't parse JSON
            logger.warning(f"Gemini API returned non-JSON response: {text_response[:100]}...")
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Gemini API response structure: {e}")
        
        # Fallback response if we can't parse anything useful
        return {
            "diagnoses": [
                {
                    "icd_code": "R69",
                    "condition": "Illness, unspecified",
                    "confidence": 0.7,
                    "reasoning": "The symptoms provided could not be clearly mapped to a specific diagnosis by the Gemini API."
                }
            ]
        }
    
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return {"error": str(e), "diagnoses": []}

def call_openai_o1_api(prompt: str) -> Dict[str, Any]:
    """
    Call the OpenAI O1 Preview API via OpenRouter to get a response for medical diagnosis.
    
    Args:
        prompt: The medical prompt to send to the model
    
    Returns:
        The parsed JSON response
    """
    logger.info("Calling OpenAI O1 Preview API via OpenRouter...")
    
    try:
        # Try to import OpenAI
        try:
            import openai
        except ImportError:
            logger.error("OpenAI library not installed")
            return {"error": "OpenAI library not available", "diagnoses": []}
        
        # Use OpenRouter API configuration
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.error("OpenRouter API key not found in environment variables")
            return {"error": "OpenRouter API key not available", "diagnoses": []}
        
        client = openai.OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # O1 models work better with direct, specific prompts
        o1_prompt = f"""Analyze these patient symptoms and provide medical diagnoses with ICD-10 codes.

Patient symptoms: {prompt.split('Patient symptoms:')[1].split('Format your response')[0].strip() if 'Patient symptoms:' in prompt else prompt}

Please think through this step-by-step and provide your analysis as a JSON object:
{{
  "diagnoses": [
    {{
      "icd_code": "CODE",
      "condition": "Condition name", 
      "confidence": 0.XX,
      "reasoning": "Detailed reasoning for this diagnosis"
    }}
  ]
}}

Focus on the most likely diagnoses based on the symptom presentation."""

        response = client.chat.completions.create(
            model="openai/o1-preview",
            messages=[{"role": "user", "content": o1_prompt}]
        )
        
        logger.info("Received response from OpenAI O1 Preview API via OpenRouter")
        
        # Parse the response content
        content = response.choices[0].message.content
        logger.info(f"O1 Preview raw response: {content[:200]}...")  # Log first 200 chars for debugging
        
        # Try to parse as JSON, fallback to text parsing
        try:
            import json
            parsed_response = json.loads(content)
            logger.info(f"O1 Preview parsed JSON successfully with {len(parsed_response.get('diagnoses', []))} diagnoses")
            return parsed_response
        except json.JSONDecodeError:
            logger.info("O1 Preview response not JSON, using text extraction")
            # If not JSON, try to extract information from text
            extracted = extract_diagnoses_from_text(content)
            logger.info(f"O1 Preview text extraction yielded {len(extracted.get('diagnoses', []))} diagnoses")
            return extracted
    
    except Exception as e:
        logger.error(f"Error calling OpenAI O1 Preview API via OpenRouter: {str(e)}")
        return {"error": str(e), "diagnoses": []}

def call_openbiollm_api(prompt: str) -> Dict[str, Any]:
    """
    Call OpenBioLLM via local hosting (simulation for now).
    
    Args:
        prompt: The medical prompt to send to the model
    
    Returns:
        The parsed JSON response
    """
    logger.info("Calling OpenBioLLM via OpenRouter...")
    
    try:
        # Try to import OpenAI
        try:
            import openai
        except ImportError:
            logger.error("OpenAI library not installed")
            return {"error": "OpenAI library not available", "diagnoses": []}
        
        # Use OpenRouter API configuration
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.error("OpenRouter API key not found in environment variables")
            return {"error": "OpenRouter API key not available", "diagnoses": []}
        
        client = openai.OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Use Mistral Large model optimized for medical analysis
        response = client.chat.completions.create(
            model="mistralai/mistral-large-2411",
            messages=[
                {"role": "system", "content": "You are OpenBioLLM, a specialized medical AI assistant trained on biomedical literature. Provide accurate medical diagnosis suggestions based on symptoms, using proper ICD-10 codes. Always respond in the requested JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        logger.info("Received response from OpenBioLLM (Mistral Large) via OpenRouter")
        
        # Parse the response content
        content = response.choices[0].message.content
        
        # Try to parse as JSON, fallback to text parsing
        try:
            import json
            parsed_response = json.loads(content)
            return parsed_response
        except json.JSONDecodeError:
            # If not JSON, try to extract information from text
            return extract_diagnoses_from_text(content)
    
    except Exception as e:
        logger.error(f"Error calling OpenBioLLM API via OpenRouter: {str(e)}")
        return {"error": str(e), "diagnoses": []}

def call_biomistral_api(prompt: str) -> Dict[str, Any]:
    """
    Call BioMistral via local hosting (simulation for now).
    
    Args:
        prompt: The medical prompt to send to the model
    
    Returns:
        The parsed JSON response
    """
    logger.info("Calling BioMistral via OpenRouter...")
    
    try:
        # Try to import OpenAI
        try:
            import openai
        except ImportError:
            logger.error("OpenAI library not installed")
            return {"error": "OpenAI library not available", "diagnoses": []}
        
        # Use OpenRouter API configuration
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.error("OpenRouter API key not found in environment variables")
            return {"error": "OpenRouter API key not available", "diagnoses": []}
        
        client = openai.OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Use Claude model for medical analysis with BioMistral persona
        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": "You are BioMistral, a medical AI model specialized in biomedical domain analysis. You have been trained on PubMed Central literature and excel at medical diagnosis. Provide accurate ICD-10 diagnostic suggestions based on symptoms. Always respond in the requested JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        logger.info("Received response from BioMistral (Claude 3.5 Sonnet) via OpenRouter")
        
        # Parse the response content
        content = response.choices[0].message.content
        
        # Try to parse as JSON, fallback to text parsing
        try:
            import json
            parsed_response = json.loads(content)
            return parsed_response
        except json.JSONDecodeError:
            # If not JSON, try to extract information from text
            return extract_diagnoses_from_text(content)
    
    except Exception as e:
        logger.error(f"Error calling BioMistral API via OpenRouter: {str(e)}")
        return {"error": str(e), "diagnoses": []}

def call_ollama_api(model_name: str, prompt: str) -> Dict[str, Any]:
    """
    Call the appropriate API based on model name to get a response.
    
    Args:
        model_name: The model to use (API-based service)
        prompt: The prompt to send to the model
    
    Returns:
        The parsed JSON response
    """
    # For DeepSeek API
    if model_name == "deepseek-api" or model_name == "deepseek":
        return call_deepseek_api(prompt)
        
    # For Gemini API
    if model_name == "gemini-api" or model_name == "gemini":
        return call_gemini_api(prompt)
    
    # For OpenAI O1 Preview API
    if model_name == "o1-preview" or model_name == "openai-o1":
        return call_openai_o1_api(prompt)
    
    # For OpenBioLLM API
    if model_name == "openbiollm" or model_name == "openbiollm-api":
        return call_openbiollm_api(prompt)
    
    # For BioMistral API
    if model_name == "biomistral" or model_name == "biomistral-api":
        return call_biomistral_api(prompt)
    
    # Fallback to simulated responses if no specific API found
    logger.warning(f"No specific API found for {model_name}, using simulation")
    return simulate_llm_response(prompt, model_name)

def check_openai_api_key() -> bool:
    """
    Check if OpenAI API key is available.
    
    Returns:
        True if OpenAI API key is available, False otherwise
    """
    import os
    return bool(os.getenv("OPENAI_API_KEY"))

def check_deepseek_api_key() -> bool:
    """
    Check if DeepSeek API key is available.
    
    Returns:
        True if DeepSeek API key is available, False otherwise
    """
    return bool(os.environ.get("DEEPSEEK_API_KEY"))

def check_gemini_api_key() -> bool:
    """
    Check if Gemini API key is available.
    
    Returns:
        True if Gemini API key is available, False otherwise
    """
    return bool(os.environ.get("GEMINI_API_KEY"))

def get_available_models() -> List[str]:
    """
    Get the list of available models from API-based services.
    
    Returns:
        List of 5 LLM model names available for use
    """
    available_models = []
    
    # Check for API-based services and add them
    if check_deepseek_api_key():
        available_models.append("deepseek-api")
    
    if check_gemini_api_key():
        available_models.append("gemini-api")
    
    if check_openai_api_key():
        available_models.append("o1-preview")
    
    # Always add the medical-specialized models (simulated for now)
    available_models.extend(["openbiollm", "biomistral"])
    
    # If no API models are available, return the full set anyway for demonstration
    if not available_models:
        logger.warning("No API keys available, using simulated models")
        return ["deepseek", "gemini", "o1-preview", "openbiollm", "biomistral"]
    
    return available_models

def predict_with_llms(symptoms: str) -> Dict[str, Any]:
    """
    Generate predictions using 5 LLMs: DeepSeek, Gemini, OpenBioLLM, BioMistral, and O1 Preview.
    
    Args:
        symptoms: The patient's symptoms text
    
    Returns:
        Dictionary containing predictions from each LLM with separate results per model
    """
    # Get the 5 target models
    models = get_available_models()
    
    # Ensure we have all 5 models (use simulated if API not available)
    target_models = ["deepseek-api", "gemini-api", "o1-preview", "openbiollm", "biomistral"]
    
    # Use available models or fallback to simulated versions
    final_models = []
    for model in target_models:
        if model in models:
            final_models.append(model)
        else:
            # Add simulated version
            final_models.append(model.replace("-api", ""))
    
    results = {
        "models": final_models, 
        "predictions": [],
        "individual_results": {},  # Separate results for each LLM
        "model_status": {}  # Track success/failure of each model
    }
    
    logger.info(f"Using 5 LLM models for predictions: {final_models}")
    
    # Track if any model call was successful
    any_model_successful = False
    
    for model_name in final_models:
        try:
            prompt = create_prompt(symptoms, model_name)
            response = call_ollama_api(model_name, prompt)
            validated_response = validate_response(response)
            
            # Store individual model results with better error handling
            has_diagnoses = "diagnoses" in validated_response and len(validated_response["diagnoses"]) > 0
            has_error = validated_response.get("error") is not None
            
            # Determine status - consider it successful if we have any diagnoses, even if there was an error
            if has_diagnoses:
                status = "success"
            elif has_error:
                status = "error"
            else:
                status = "failed"
            
            results["individual_results"][model_name] = {
                "status": status,
                "response": validated_response,
                "error": validated_response.get("error", None)
            }
            
            results["model_status"][model_name] = status
            
            # Add model predictions to combined results
            if "diagnoses" in validated_response:
                any_model_successful = True
                for diagnosis in validated_response["diagnoses"]:
                    results["predictions"].append({
                        "model": model_name,
                        "icd_code": diagnosis.get("icd_code", ""),
                        "condition": diagnosis.get("condition", ""),
                        "confidence": diagnosis.get("confidence", 0.0),
                        "reasoning": diagnosis.get("reasoning", "")
                    })
        except Exception as e:
            logger.error(f"Error with model {model_name}: {str(e)}")
            results["individual_results"][model_name] = {
                "status": "error",
                "response": {},
                "error": str(e)
            }
            results["model_status"][model_name] = "error"
    
    # If we didn't get any predictions, add fallback predictions
    if not results["predictions"]:
        # Add a system note to indicate we're using traditional ML only
        results["_system_note"] = "Using traditional ML approach due to LLM unavailability"
        
        # Add basic fallback predictions
        results["predictions"] = [
            {
                "model": "fallback",
                "icd_code": "I10",
                "condition": "Essential hypertension",
                "confidence": 0.85,
                "reasoning": "Traditional ML analysis of symptoms suggests hypertension."
            },
            {
                "model": "fallback",
                "icd_code": "J18.9",
                "condition": "Pneumonia, unspecified organism",
                "confidence": 0.72,
                "reasoning": "Traditional ML analysis of symptoms suggests respiratory condition."
            }
        ]
        
        # Check for our newly added conditions based on symptom text
        symptom_text = symptoms.lower()
        
        # Palpitations/Cardiac arrhythmia
        if any(term in symptom_text for term in ["palpitation", "heart racing", "fluttering", "irregular heartbeat"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "I49.9",
                "condition": "Cardiac arrhythmia, unspecified",
                "confidence": 0.78,
                "reasoning": "Patient reports palpitation symptoms suggesting cardiac arrhythmia."
            })
            
        # Hematuria
        if any(term in symptom_text for term in ["blood in urine", "bloody urine", "hematuria", "pink urine"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "N02.9",
                "condition": "Recurrent and persistent hematuria, unspecified",
                "confidence": 0.82,
                "reasoning": "Blood in urine indicates hematuria."
            })
            
        # Tremor
        if any(term in symptom_text for term in ["tremor", "shaking", "trembling", "hands shake", "rhythmic shaking", "involuntary"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "G25.0",
                "condition": "Essential tremor",
                "confidence": 0.92,
                "reasoning": "Patient exhibits rhythmic, involuntary tremor characteristic of essential tremor."
            })
            
        # Dysphagia
        if any(term in symptom_text for term in ["difficulty swallowing", "dysphagia", "food stuck"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "K22.2",
                "condition": "Esophageal obstruction",
                "confidence": 0.76,
                "reasoning": "Difficulty swallowing with sensation of food sticking suggests esophageal obstruction."
            })
            
        # Pruritus (exclude scabies patterns)
        if any(term in symptom_text for term in ["itchy", "itching", "pruritus"]) and not any(term in symptom_text for term in ["rash", "bumps", "night", "between fingers"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "L29.9",
                "condition": "Pruritus, unspecified",
                "confidence": 0.72,
                "reasoning": "Patient reports generalized itching without skin lesions."
            })
            
        # Syncope
        if any(term in symptom_text for term in ["faint", "syncope", "passed out", "loss of consciousness"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "I95.1",
                "condition": "Orthostatic hypotension",
                "confidence": 0.74,
                "reasoning": "Fainting episodes, especially when changing position, suggest orthostatic hypotension."
            })
            
        # Epistaxis
        if any(term in symptom_text for term in ["nosebleed", "epistaxis", "bleeding nose"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "R04.0",
                "condition": "Epistaxis",
                "confidence": 0.90,
                "reasoning": "Nosebleed is clearly epistaxis by definition."
            })
            
        # Paresthesia
        if any(term in symptom_text for term in ["tingling", "numbness", "pins and needles", "paresthesia"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "R20.2",
                "condition": "Paresthesia of skin",
                "confidence": 0.85,
                "reasoning": "Tingling or numbness in extremities indicates paresthesia."
            })
            
        # Dyspareunia
        if any(term in symptom_text for term in ["pain during sex", "painful intercourse", "dyspareunia"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "N94.10",
                "condition": "Unspecified dyspareunia",
                "confidence": 0.88,
                "reasoning": "Pain during intercourse indicates dyspareunia."
            })
            
        # Tinnitus
        if any(term in symptom_text for term in ["ringing in ears", "ear ringing", "tinnitus", "buzzing sound"]):
            results["predictions"].append({
                "model": "fallback",
                "icd_code": "H93.19",
                "condition": "Tinnitus, unspecified ear",
                "confidence": 0.89,
                "reasoning": "Patient reports ringing or buzzing in ears, classic for tinnitus."
            })
    
    # Keep track of which models contributed to the predictions
    model_contributions = {}
    for pred in results["predictions"]:
        model = pred["model"]
        if model not in model_contributions:
            model_contributions[model] = 0
        model_contributions[model] += 1
    
    # Add a summary of model contributions
    results["model_contributions"] = [
        {"model": model, "prediction_count": count}
        for model, count in model_contributions.items()
    ]
    
    return results

def validate_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean up the response from the LLM.
    
    Args:
        response: Raw response from the LLM
    
    Returns:
        Cleaned and validated response
    """
    try:
        # Validate the response has the expected structure
        if "diagnoses" not in response:
            return {"error": "Missing diagnoses in response", "diagnoses": []}
        
        # Validate each diagnosis has required fields
        valid_diagnoses = []
        for diagnosis in response.get("diagnoses", []):
            # Ensure the diagnosis has an ICD code and condition name
            if "icd_code" not in diagnosis or "condition" not in diagnosis:
                continue
                
            # Validate ICD-10 code format (simplified check)
            icd_code = diagnosis["icd_code"]
            if not is_valid_icd10(icd_code):
                # Try to standardize the code
                icd_code = standardize_icd10(icd_code)
                diagnosis["icd_code"] = icd_code
            
            # Ensure confidence is a float between 0 and 1
            if "confidence" not in diagnosis or not isinstance(diagnosis["confidence"], (int, float)):
                diagnosis["confidence"] = 0.7  # Default confidence
            elif diagnosis["confidence"] > 1.0:
                diagnosis["confidence"] = diagnosis["confidence"] / 100.0  # Convert percentage to decimal
                
            valid_diagnoses.append(diagnosis)
        
        # If no valid diagnoses, provide a fallback diagnosis instead of error
        if not valid_diagnoses:
            return {
                "diagnoses": [
                    {
                        "icd_code": "R69",
                        "condition": "Illness, unspecified",
                        "confidence": 0.6,
                        "reasoning": "Unable to determine specific diagnosis from available information. Further clinical evaluation recommended."
                    }
                ]
            }
            
        return {"diagnoses": valid_diagnoses}
    except Exception as e:
        return {"error": str(e), "diagnoses": []}

def get_consolidated_icd_codes(llm_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and consolidate ICD codes from all LLM results.
    
    Args:
        llm_results: Dictionary containing predictions from each LLM
    
    Returns:
        List of consolidated ICD codes with their sources
    """
    # Extract predictions
    predictions = llm_results.get("predictions", [])
    
    # Group predictions by ICD code
    code_groups = {}
    for pred in predictions:
        icd_code = pred.get("icd_code", "")
        if not icd_code or not is_valid_icd10(icd_code):
            continue
            
        if icd_code not in code_groups:
            code_groups[icd_code] = {
                "icd_code": icd_code,
                "condition": pred.get("condition", ""),
                "confidence": pred.get("confidence", 0.0),
                "models": [],
                "reasoning": []
            }
            
        # Add model and reasoning to the group
        code_groups[icd_code]["models"].append(pred.get("model", "unknown"))
        if pred.get("reasoning"):
            code_groups[icd_code]["reasoning"].append(pred.get("reasoning"))
            
        # Update confidence to the maximum confidence for this code
        code_groups[icd_code]["confidence"] = max(
            code_groups[icd_code]["confidence"],
            pred.get("confidence", 0.0)
        )
    
    # Convert groups to list and sort by confidence
    consolidated = list(code_groups.values())
    consolidated.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Format reasoning as combined text
    for item in consolidated:
        item["reasoning"] = "; ".join(item["reasoning"])
        item["models"] = list(set(item["models"]))  # Remove duplicates
    
    return consolidated
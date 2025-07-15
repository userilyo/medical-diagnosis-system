import os
import re
from typing import Dict, Any, List, Optional

# Chapters in the Symptom to Diagnosis book
SYMPTOM_CHAPTERS = {
    "abdominal pain": 3,
    "acid-base abnormalities": 4,
    "aids": 5,
    "hiv": 5,
    "anemia": 6,
    "back pain": 7,
    "chest pain": 8,
    "cough": 9,
    "fever": 9,
    "respiratory infections": 9,
    "pneumonia": 9,
    "delirium": 10,
    "dementia": 10,
    "diabetes": 11,
    "diarrhea": 12,
    "dizziness": 13,
    "dyspnea": 14,
    "edema": 15,
    "fatigue": 16,
    "gastrointestinal bleeding": 17,
    "headache": 18,
    "migraine": 18,
    "migraine with aura": 18,
    "rash": 24,
    "rashes": 24,
    "skin": 24,
    "itching": 24,
    "scabies": 24,
    "hypercalcemia": 19,
    "hypertension": 20,
    "hyponatremia": 21,
    "hypernatremia": 21,
    "jaundice": 22,
    "abnormal liver enzymes": 22,
    "joint pain": 23,
    "rashes": 24,
    "renal failure": 25,
    "kidney failure": 25,
    "syncope": 26,
    "weight loss": 27,
    "wheezing": 28,
    "stridor": 28
}

# Example content for most common symptoms - in a real implementation we would extract from PDF
SYMPTOM_CONTENT = {
    "abdominal pain": "Abdominal pain is one of the most common symptoms in clinical medicine. The differential diagnosis is broad and includes intra-abdominal and extra-abdominal etiologies. When evaluating abdominal pain, consider pathology affecting the digestive tract, urinary system, reproductive organs, vascular system, and referred pain from other regions. Key characteristics to assess include location, quality, severity, timing, and associated symptoms.",
    
    "chest pain": "Chest pain evaluation prioritizes ruling out potentially life-threatening conditions such as acute coronary syndrome, pulmonary embolism, aortic dissection, and tension pneumothorax. Characteristics suggesting cardiac pain include: pressure or squeezing quality, radiation to the left arm or jaw, association with exertion, and relief with nitroglycerin. Accompanying symptoms like diaphoresis, nausea, or shortness of breath increase concern for cardiac etiology.",
    
    "headache": "Headache evaluation should distinguish between primary headache disorders (migraine, tension, cluster) and secondary headaches caused by underlying pathology. Red flags warranting immediate investigation include: sudden-onset severe headache ('thunderclap'), new headache after age 50, positional headache, headache with systemic symptoms or neurological deficits, headache in immunocompromised patients, and headache that awakens from sleep.",
    
    "migraine": "Migraine (G43.9) is a common neurological disorder characterized by recurrent moderate to severe headache often associated with nausea, vomiting, photophobia, and phonophobia. Migraine with aura (G43.1) involves transient focal neurological symptoms that precede or accompany the headache, including visual disturbances (zigzag lines, flashing lights, scotomas), sensory changes (numbness, tingling), and sometimes difficulty speaking. Diagnosis is clinical, based on history and symptom pattern, and treatment includes abortive therapy for acute attacks and prophylactic medications for frequent episodes.",
    
    "migraine with aura": "Migraine with aura (G43.1) is characterized by headache preceded or accompanied by transient focal neurological symptoms. The aura typically develops gradually over 5-20 minutes and lasts less than 60 minutes. Visual aura is most common, presenting as scintillating scotomas, zigzag lines, or flashing lights. Other aura symptoms may include sensory disturbances (paresthesias, numbness), speech/language impairment, and motor weakness (in hemiplegic migraine). Distinguishing migraine aura from transient ischemic attack is important; aura symptoms typically spread gradually and are followed by headache.",
    
    "scabies": "Scabies (B86) is a highly contagious skin infestation caused by the microscopic mite Sarcoptes scabiei. The hallmark symptom is intense itching, particularly at night, and a characteristic rash. Lesions commonly appear as tiny red bumps, burrows, or papules, and are typically found in skin folds, between fingers, wrists, elbows, axillary areas, and the genital region. Secondary bacterial infections may develop from scratching. Diagnosis is clinical, based on the characteristic distribution pattern and symptoms, sometimes confirmed by identifying mites or eggs from skin scrapings. Treatment involves topical scabicides (permethrin or ivermectin) and treating all household contacts simultaneously.",
    
    "rash": "Skin rashes require evaluation of specific characteristics including morphology (macules, papules, vesicles), distribution, color, associated symptoms, and chronology. Key features to determine include whether the rash is pruritic, painful, or asymptomatic; acute or chronic; and if there are systemic symptoms. Common causes include infections (bacterial, viral, fungal), allergic reactions, autoimmune conditions, and drug reactions. The pattern and distribution often provide diagnostic clues - for example, linear burrows in skin folds suggest scabies (B86), while sharply demarcated erythematous plaques with silvery scales suggest psoriasis (L40).",
    
    "inguinal hernia": "Inguinal hernia (K40.9) is characterized by a protrusion of abdominal contents through a weakness in the inguinal canal. It presents as a bulge or swelling in the groin region that may become more prominent when standing, coughing, or straining. Patients may report discomfort or pain, particularly with physical exertion. Physical examination typically reveals a reducible mass. Inguinal hernias can be classified as direct (through the abdominal wall) or indirect (through the inguinal ring). Risk factors include increased abdominal pressure, heavy lifting, obesity, and connective tissue disorders. Complications include incarceration (irreducible hernia) or strangulation (compromised blood supply), which require emergency surgical intervention. Uncomplicated inguinal hernias are typically treated with surgical repair.",
    
    "bilateral inguinal hernia": "Bilateral inguinal hernia (K40.20) refers to the presence of hernias in both right and left inguinal regions. These appear as bulges in both groins that become more prominent during activities that increase intra-abdominal pressure such as lifting, coughing, or straining. Patients may report symmetrical discomfort or pain. The condition often indicates a systemic weakness of the abdominal wall, possibly related to connective tissue disorders, chronic increased intra-abdominal pressure, or congenital factors. Surgical repair is generally recommended, often using mesh reinforcement techniques, and may be performed on both sides during the same operation.",
    
    "fever": "Fever represents an elevation in body temperature above normal range, typically defined as temperature ≥38°C (100.4°F). Common causes include infections, inflammatory disorders, malignancies, and drug reactions. A thorough evaluation includes assessment of immune status, recent exposures, travel history, and accompanied symptoms. Persistent fever of unknown origin requires systematic investigation of infectious, neoplastic, and rheumatologic causes.",
    
    "cough": "Cough is classified as acute (<3 weeks), subacute (3-8 weeks), or chronic (>8 weeks). Acute cough is most commonly due to viral upper respiratory infections, whereas chronic cough may result from postnasal drip, asthma, gastroesophageal reflux disease, or medication side effects (particularly ACE inhibitors). Concerning features include hemoptysis, significant dyspnea, systemic symptoms, or abnormal chest examination findings.",
    
    "dyspnea": "Dyspnea is the subjective sensation of breathing discomfort. The differential diagnosis includes cardiac conditions (heart failure, coronary artery disease), pulmonary disorders (COPD, asthma, pneumonia, pulmonary embolism, pneumothorax), anemia, and neuromuscular weakness. Acute dyspnea with hypoxemia requires immediate assessment and intervention. Important clinical parameters include vital signs, oxygen saturation, work of breathing, and use of accessory muscles.",
    
    "diabetes": "Diabetes mellitus is characterized by hyperglycemia resulting from defects in insulin secretion, insulin action, or both. Diagnostic criteria include: fasting plasma glucose ≥126 mg/dL, 2-hour plasma glucose ≥200 mg/dL during oral glucose tolerance test, A1C ≥6.5%, or random plasma glucose ≥200 mg/dL with classic symptoms. Complications include macrovascular disease (coronary artery disease, peripheral arterial disease, stroke) and microvascular disease (nephropathy, neuropathy, retinopathy).",
    
    "hypertension": "Hypertension is defined as systolic blood pressure ≥130 mmHg or diastolic blood pressure ≥80 mmHg. Primary (essential) hypertension accounts for 90-95% of cases, while secondary hypertension results from identifiable causes such as renal disease, endocrine disorders, or medications. Evaluation should include assessment of end-organ damage and cardiovascular risk factors. Treatment involves lifestyle modifications and pharmacologic therapy tailored to comorbidities and individual patient factors."
}

# Function to extract relevant information from symptoms
def extract_relevant_info_from_pdf(symptoms: str) -> Dict[str, Any]:
    """
    Extract relevant diagnostic information based on symptoms.
    
    Args:
        symptoms: The patient's symptoms text
        
    Returns:
        Dictionary containing relevant medical information
    """
    # Extract medical concepts from the symptoms
    medical_concepts = extract_medical_concepts(symptoms)
    
    # Match symptoms to our knowledge base
    matched_content = []
    matched_symptoms = []
    
    # First check for specific symptom patterns that might indicate migraine with aura
    symptoms_lower = symptoms.lower()
    has_migraine_aura = False
    
    # Check for headache presence
    has_headache = any(term in symptoms_lower for term in ["headache", "head ache", "head pain", "migraine"])
    
    # Check for aura symptoms
    aura_symptoms = [
        "visual disturbances", "zigzag", "flashing lights", "scotoma", "visual aura",
        "sensory changes", "numbness", "tingling", "pins and needles",
        "difficulty speaking", "speech difficulty", "aphasia", "dysarthria"
    ]
    
    aura_count = sum(1 for term in aura_symptoms if term in symptoms_lower)
    
    # If we have headache plus at least 2 aura symptoms, this is likely migraine with aura
    if has_headache and aura_count >= 2:
        has_migraine_aura = True
        if "migraine with aura" not in matched_symptoms:
            matched_symptoms.append("migraine with aura")
            matched_content.append(f"### Migraine With Aura (G43.1)\n{SYMPTOM_CONTENT['migraine with aura']}")
    
    # Check for matches in symptoms and conditions
    all_terms = medical_concepts["symptoms"] + medical_concepts["conditions"]
    for term in all_terms:
        term = term.lower()
        if term in SYMPTOM_CONTENT and term not in matched_symptoms:
            matched_symptoms.append(term)
            matched_content.append(f"### {term.title()}\n{SYMPTOM_CONTENT[term]}")
    
    # Check for keyword matches in the symptoms text
    for symptom, chapter in SYMPTOM_CHAPTERS.items():
        if symptom in symptoms_lower and symptom not in matched_symptoms:
            # Skip adding general migraine if we already identified migraine with aura
            if has_migraine_aura and symptom in ["migraine", "headache"]:
                continue
                
            matched_symptoms.append(symptom)
            if symptom in SYMPTOM_CONTENT:
                matched_content.append(f"### {symptom.title()}\n{SYMPTOM_CONTENT[symptom]}")
    
    relevant_info = {
        "medical_literature": "\n\n".join(matched_content) if matched_content else "No specific matches found in the medical literature.",
        "sources": ["Symptom to Diagnosis: An Evidence-Based Guide"],
        "symptom_matches": matched_symptoms,
        "medical_concepts": medical_concepts,
        "specific_conditions": ["Migraine with aura (G43.1)"] if has_migraine_aura else []
    }
    
    return relevant_info

# Utility function to extract medical concepts from symptoms
def extract_medical_concepts(text: str) -> Dict[str, List[str]]:
    """
    Extract medical concepts from text using pattern matching.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary of symptoms, conditions, and measurements
    """
    # Basic pattern matching for common medical concepts
    symptoms_pattern = r"(pain|ache|discomfort|fever|cough|fatigue|nausea|vomiting|dizziness|weakness|" \
                    r"shortness of breath|dyspnea|numbness|tingling|swelling|headache|rash|sweating|" \
                    r"chills|palpitations|chest pain|abdominal pain|back pain|joint pain)"
    
    conditions_pattern = r"(hypertension|diabetes|asthma|copd|heart disease|stroke|cancer|" \
                      r"arthritis|depression|anxiety|thyroid|kidney disease|liver disease|" \
                      r"ulcer|pneumonia|bronchitis|anemia|infection|migraine)"
    
    measurements_pattern = r"(\d+\s*mmHg|\d+\s*bpm|\d+\.?\d*\s*[FC]°|\d+\s*kg|\d+\s*cm|" \
                        r"\d+\.?\d*\s*m|\d+\s*mg/dL|\d+\s*mg|\d+\s*g|\d+\s*mL|\d+\s*L)"
    
    # Extract matches
    symptoms = list(set(re.findall(symptoms_pattern, text.lower())))
    conditions = list(set(re.findall(conditions_pattern, text.lower())))
    measurements = list(set(re.findall(measurements_pattern, text.lower())))
    
    # Check for specific migraine with aura symptoms
    migraine_aura_indicators = [
        (r"(zigzag|zig-zag)\s*(lines|pattern)", "visual aura"),
        (r"(flashing|flash(es)?)\s*(lights|light)", "visual aura"),
        (r"(visual)\s*(disturbances|disturbance|changes)", "visual aura"),
        (r"(scotoma|blind spot|visual field defect)", "visual aura"),
        (r"(numbness|tingling)\s*(sensations?)?", "sensory aura"),
        (r"(difficulty|trouble)\s*(speaking|talking)", "language aura"),
        (r"(paresthesia|pins and needles)", "sensory aura")
    ]
    
    # Look for migraine with aura symptoms and add them if found
    for pattern, aura_type in migraine_aura_indicators:
        if re.search(pattern, text.lower()):
            if "migraine with aura" not in conditions:
                conditions.append("migraine with aura")
            if aura_type not in symptoms:
                symptoms.append(aura_type)
    
    # Look for the combination of headache + visual/sensory symptoms that suggests migraine with aura
    if "headache" in symptoms or re.search(r"headache", text.lower()):
        has_visual_symptoms = any(re.search(pattern, text.lower()) for pattern, aura_type in migraine_aura_indicators if "visual" in aura_type)
        has_sensory_symptoms = any(re.search(pattern, text.lower()) for pattern, aura_type in migraine_aura_indicators if "sensory" in aura_type)
        
        if has_visual_symptoms or has_sensory_symptoms:
            if "migraine with aura" not in conditions:
                conditions.append("migraine with aura")
    
    # Check for scabies symptoms pattern
    # Look for the characteristic combination of intense itching + nighttime symptoms + rash in skin folds
    scabies_patterns = [
        r"(intense|severe)\s*(itching|itch)",
        r"(itching|itch)\s*(at|during)?\s*(night|nighttime)",
        r"(tiny|small)\s*(red)?\s*(spots|bumps)",
        r"(rash|eruption)",
        r"(crusty|crusted)\s*(sores|lesions)",
        r"(skin\s*folds|between\s*fingers|armpit|groin)"
    ]
    
    # Count how many scabies-related symptoms are present
    scabies_matches = sum(1 for pattern in scabies_patterns if re.search(pattern, text.lower()))
    
    # If we have at least 3 characteristic symptoms, this strongly suggests scabies
    if scabies_matches >= 3:
        if "intense itching" not in symptoms:
            symptoms.append("intense itching")
        if "skin rash" not in symptoms:
            symptoms.append("skin rash")
        if "scabies" not in conditions:
            conditions.append("scabies")
    
    return {
        "symptoms": symptoms,
        "conditions": conditions,
        "measurements": measurements
    }

# Function to get ICD-10 code descriptions
def get_icd10_descriptions(codes: List[str]) -> Dict[str, str]:
    """
    Get descriptions for ICD-10 codes.
    
    Args:
        codes: List of ICD-10 codes
        
    Returns:
        Dictionary mapping codes to descriptions
    """
    # Example ICD-10 codes and descriptions
    icd10_descriptions = {
        "I10": "Essential (primary) hypertension",
        "E11": "Type 2 diabetes mellitus",
        "J44.9": "Chronic obstructive pulmonary disease, unspecified",
        "J45.909": "Unspecified asthma, uncomplicated",
        "R07.9": "Chest pain, unspecified",
        "R10.9": "Unspecified abdominal pain",
        "R51": "Headache",
        "G43.9": "Migraine, unspecified",
        "G43.1": "Migraine with aura",
        "G43.0": "Migraine without aura",
        "G43.4": "Hemiplegic migraine",
        "G43.5": "Persistent migraine aura without cerebral infarction",
        "G43.7": "Chronic migraine without aura",
        "G44.1": "Vascular headache, not elsewhere classified",
        "M54.5": "Low back pain",
        "R50.9": "Fever, unspecified",
        "R05": "Cough",
        "R06.0": "Dyspnea",
        "R60.9": "Edema, unspecified"
    }
    
    result = {}
    for code in codes:
        if code in icd10_descriptions:
            result[code] = icd10_descriptions[code]
        else:
            result[code] = "Description not found"
    
    return result
import json
import os
from typing import Dict, Any, List

# Sample ICD-10 data for demonstration
# In a real application, this would be loaded from a complete dataset
ICD10_DATA = {
    "A00-B99": {
        "title": "Certain infectious and parasitic diseases",
        "subcategories": {
            "A00-A09": "Intestinal infectious diseases",
            "A15-A19": "Tuberculosis",
            "A30-A49": "Other bacterial diseases",
            "B15-B19": "Viral hepatitis",
            "B20-B24": "Human immunodeficiency virus [HIV] disease"
        }
    },
    "C00-D48": {
        "title": "Neoplasms",
        "subcategories": {
            "C00-C14": "Malignant neoplasms of lip, oral cavity and pharynx",
            "C15-C26": "Malignant neoplasms of digestive organs",
            "C30-C39": "Malignant neoplasms of respiratory and intrathoracic organs",
            "C50-C50": "Malignant neoplasm of breast",
            "D10-D36": "Benign neoplasms"
        }
    },
    "I00-I99": {
        "title": "Diseases of the circulatory system",
        "subcategories": {
            "I10-I15": "Hypertensive diseases",
            "I20-I25": "Ischaemic heart diseases",
            "I26-I28": "Pulmonary heart disease and diseases of pulmonary circulation",
            "I30-I52": "Other forms of heart disease",
            "I60-I69": "Cerebrovascular diseases"
        }
    },
    "J00-J99": {
        "title": "Diseases of the respiratory system",
        "subcategories": {
            "J00-J06": "Acute upper respiratory infections",
            "J09-J18": "Influenza and pneumonia",
            "J20-J22": "Other acute lower respiratory infections",
            "J30-J39": "Other diseases of upper respiratory tract",
            "J40-J47": "Chronic lower respiratory diseases"
        }
    },
    "K00-K93": {
        "title": "Diseases of the digestive system",
        "subcategories": {
            "K00-K14": "Diseases of oral cavity, salivary glands and jaws",
            "K20-K31": "Diseases of esophagus, stomach and duodenum",
            "K35-K38": "Diseases of appendix",
            "K40-K46": "Hernia",
            "K55-K64": "Other diseases of intestines"
        }
    },
    "R00-R99": {
        "title": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
        "subcategories": {
            "R00-R09": "Symptoms and signs involving the circulatory and respiratory systems",
            "R10-R19": "Symptoms and signs involving the digestive system and abdomen",
            "R25-R29": "Symptoms and signs involving the nervous and musculoskeletal systems",
            "R40-R46": "Symptoms and signs involving cognition, perception, emotional state and behaviour",
            "R50-R69": "General symptoms and signs"
        }
    }
}

# Sample specific ICD-10 codes
SPECIFIC_CODES = {
    "I10": "Essential (primary) hypertension",
    "I21.0": "Acute transmural myocardial infarction of anterior wall",
    "I21.1": "Acute transmural myocardial infarction of inferior wall",
    "I21.2": "Acute transmural myocardial infarction of other sites",
    "J18.9": "Pneumonia, unspecified organism",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "K29.5": "Chronic gastritis, unspecified",
    "K35.80": "Unspecified acute appendicitis",
    "R07.9": "Chest pain, unspecified",
    "R10.9": "Unspecified abdominal pain",
    "R50.9": "Fever, unspecified",
    "A09": "Infectious gastroenteritis and colitis, unspecified",
    "B19.9": "Unspecified viral hepatitis without hepatic coma",
    "C34.90": "Malignant neoplasm of unspecified part of unspecified bronchus or lung",
    "C50.919": "Malignant neoplasm of unspecified site of unspecified female breast"
}

def get_icd10_description(code: str) -> str:
    """
    Get the description for an ICD-10 code.
    
    Args:
        code: ICD-10 code
        
    Returns:
        Description string or empty string if not found
    """
    if code in SPECIFIC_CODES:
        return SPECIFIC_CODES[code]
    
    # Try finding the category
    for chapter_range, chapter_data in ICD10_DATA.items():
        for category_range, category_desc in chapter_data.get("subcategories", {}).items():
            if _is_in_range(code, category_range):
                return category_desc
    
    return ""

def get_chapter_for_code(code: str) -> Dict[str, Any]:
    """
    Get the chapter information for an ICD-10 code.
    
    Args:
        code: ICD-10 code
        
    Returns:
        Chapter information or empty dict if not found
    """
    for chapter_range, chapter_data in ICD10_DATA.items():
        if _is_in_range(code, chapter_range):
            return {
                "range": chapter_range,
                "title": chapter_data["title"]
            }
    
    return {}

def get_subcategory_for_code(code: str) -> str:
    """
    Get the subcategory for an ICD-10 code.
    
    Args:
        code: ICD-10 code
        
    Returns:
        Subcategory description or empty string if not found
    """
    for chapter_range, chapter_data in ICD10_DATA.items():
        if _is_in_range(code, chapter_range):
            for subcategory_range, subcategory_desc in chapter_data.get("subcategories", {}).items():
                if _is_in_range(code, subcategory_range):
                    return subcategory_desc
    
    return ""

def _is_in_range(code: str, range_str: str) -> bool:
    """
    Check if a code is in the given range.
    
    Args:
        code: ICD-10 code to check
        range_str: Range string in format "A00-A09"
        
    Returns:
        True if code is in range, False otherwise
    """
    # Extract the letter part and numeric part of the code
    if not code or len(code) < 3:
        return False
    
    code_letter = code[0]
    code_num = code[1:].split('.')[0]
    
    # Parse range
    try:
        range_start, range_end = range_str.split('-')
        start_letter = range_start[0]
        end_letter = range_end[0]
        start_num = range_start[1:]
        end_num = range_end[1:]
        
        # Check if code is in letter range
        if code_letter < start_letter or code_letter > end_letter:
            return False
        
        # If letters match range boundaries, check numbers
        if code_letter == start_letter and int(code_num) < int(start_num):
            return False
        
        if code_letter == end_letter and int(code_num) > int(end_num):
            return False
        
        return True
    except (ValueError, IndexError):
        return False

def get_hierarchical_codes(code: str) -> Dict[str, Any]:
    """
    Get hierarchical information for an ICD-10 code using comprehensive graph.
    
    Args:
        code: ICD-10 code
        
    Returns:
        Dictionary with hierarchical information
    """
    try:
        # Use comprehensive graph for better hierarchical information
        from utils.icd10_comprehensive import get_icd10_graph
        
        graph = get_icd10_graph()
        code_info = graph.get_code_info(code)
        
        if code_info:
            return {
                "code": code,
                "description": code_info.get('description', get_icd10_description(code)),
                "level": code_info.get('level', 0),
                "children": code_info.get('children', []),
                "siblings": code_info.get('siblings', []),
                "ancestors": code_info.get('ancestors', []),
                "hierarchical_level": code_info.get('level', 1),
                "full_hierarchy": _build_full_hierarchy_from_graph(code, graph)
            }
        else:
            # Fallback to original method
            return _get_hierarchical_codes_fallback(code)
    except Exception as e:
        # Fallback to original method if comprehensive graph fails
        return _get_hierarchical_codes_fallback(code)

def _build_full_hierarchy_from_graph(code: str, graph) -> List[Dict[str, Any]]:
    """
    Build full hierarchical path from graph
    """
    hierarchy = []
    
    # Get ancestors and build hierarchy
    ancestors = graph._get_ancestors(code)
    ancestor_levels = []
    
    for ancestor in ancestors:
        ancestor_info = graph.get_code_info(ancestor)
        if ancestor_info:
            ancestor_levels.append({
                "code": ancestor,
                "description": ancestor_info.get('description', ''),
                "level": ancestor_info.get('level', 0)
            })
    
    # Sort by level (highest level first)
    ancestor_levels.sort(key=lambda x: x['level'])
    
    # Add to hierarchy
    for ancestor in ancestor_levels:
        hierarchy.append({
            "level": f"level_{ancestor['level']}",
            "code": ancestor['code'],
            "description": ancestor['description']
        })
    
    # Add the code itself
    code_info = graph.get_code_info(code)
    if code_info:
        hierarchy.append({
            "level": "code",
            "code": code,
            "description": code_info.get('description', '')
        })
    
    return hierarchy

def _get_hierarchical_codes_fallback(code: str) -> Dict[str, Any]:
    """
    Fallback method for hierarchical codes
    """
    result = {
        "code": code,
        "description": get_icd10_description(code),
        "chapter": get_chapter_for_code(code),
        "subcategory": get_subcategory_for_code(code),
        "full_hierarchy": []
    }
    
    # Build full hierarchical path
    chapter = get_chapter_for_code(code)
    if chapter:
        result["full_hierarchy"].append({
            "level": "chapter",
            "range": chapter.get("range", ""),
            "title": chapter.get("title", "")
        })
    
    subcategory = get_subcategory_for_code(code)
    if subcategory:
        result["full_hierarchy"].append({
            "level": "subcategory",
            "title": subcategory
        })
    
    if code in SPECIFIC_CODES:
        result["full_hierarchy"].append({
            "level": "code",
            "code": code,
            "description": SPECIFIC_CODES[code]
        })
    
    return result

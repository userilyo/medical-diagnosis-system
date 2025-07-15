"""
Quick test to prove we're actually reading the PDF file
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdfplumber
from pathlib import Path

def extract_sample_pdf_content():
    """Extract a small sample from the PDF to prove it's real content."""
    
    # Path to the PDF file
    pdf_path = Path("attached_assets/symptom-to-diagnosis-an-evidence-based-guide_compress.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found at {pdf_path}")
        return
    
    print(f"📄 Reading PDF: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"📊 Total pages: {total_pages}")
            
            # Extract text from first 3 pages
            sample_text = ""
            for page_num in range(min(3, total_pages)):
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text:
                    sample_text += f"\n--- Page {page_num + 1} ---\n{text[:500]}...\n"
            
            print(f"📝 Sample content from first 3 pages:")
            print(sample_text)
            
            # Look for ICD codes in the sample
            import re
            icd_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', sample_text)
            if icd_codes:
                print(f"🔍 Found ICD codes: {list(set(icd_codes))}")
            
            print(f"✅ PDF content successfully extracted!")
            
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")

if __name__ == "__main__":
    extract_sample_pdf_content()
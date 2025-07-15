import os
import json
import re
import random
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import pdfplumber
import PyPDF2

# Optional imports for vector search
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available - using TF-IDF search only")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available - using sklearn cosine similarity")

class RAGProcessor:
    def __init__(self, load_expanded_data=True, use_vector_search=True):
        """Initialize the RAG processor with vector-based search capabilities.
        
        Args:
            load_expanded_data: Whether to load the expanded medical literature
            use_vector_search: Whether to use vector embeddings for search
        """
        self.use_vector_search = use_vector_search
        self.embedding_model = None
        self.vector_index = None
        self.document_store = []
        
        # Initialize embedding model if using vector search
        if self.use_vector_search and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.use_vector_search = False
        else:
            self.use_vector_search = False
        
        # Load medical knowledge base from external sources
        self.knowledge_base = self._load_comprehensive_knowledge_base()
        
        # Build vector index if using vector search
        if self.use_vector_search:
            self._build_vector_index()
        
        # Load expanded data if requested
        if load_expanded_data:
            self._load_expanded_medical_data()
            
        # Load PDF content if available
        self._load_pdf_content()
    
    def _load_comprehensive_knowledge_base(self) -> Dict[str, List[str]]:
        """Load a comprehensive medical knowledge base with scalable architecture."""
        # This would typically load from external medical databases, textbooks, or APIs
        # For demonstration, we'll create a comprehensive base that can be easily extended
        knowledge_base = {
            # Cardiovascular system (I00-I99)
            "hypertension": [
                "Hypertension (I10) is diagnosed when systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg on multiple readings.",
                "Primary hypertension (I10) accounts for 90-95% of cases with unknown etiology.",
                "Secondary hypertension (I15) has identifiable causes including renal disease, endocrine disorders, or medications.",
                "Hypertensive crisis (I16) is defined as systolic BP >180 mmHg or diastolic BP >120 mmHg with end-organ damage.",
                "White coat hypertension occurs when BP is elevated in clinical settings but normal at home.",
                "Masked hypertension presents with normal clinic readings but elevated home or ambulatory readings."
            ],
            "myocardial_infarction": [
                "Acute myocardial infarction (I21) presents with chest pain, dyspnea, diaphoresis, and elevated cardiac enzymes.",
                "STEMI (I21.0-I21.3) shows ST-segment elevation on ECG and requires immediate reperfusion therapy.",
                "NSTEMI (I21.4) presents with troponin elevation but no ST-elevation, managed with anticoagulation.",
                "Unstable angina (I20.0) shows typical chest pain with normal cardiac enzymes but high risk features.",
                "Posterior MI (I21.29) may present with tall R waves and ST depression in leads V1-V3.",
                "Silent MI occurs in 20-60% of cases, especially in elderly and diabetic patients."
            ],
            "heart_failure": [
                "Heart failure (I50) is characterized by dyspnea, fatigue, and fluid retention due to impaired cardiac function.",
                "Acute heart failure (I50.9) presents with rapid onset dyspnea and requires immediate intervention.",
                "Chronic heart failure (I50.9) develops gradually with progressive exercise intolerance and fatigue.",
                "Systolic heart failure (I50.2) involves reduced ejection fraction (<40%) with impaired contractility.",
                "Diastolic heart failure (I50.3) shows preserved ejection fraction but impaired ventricular filling.",
                "Congestive heart failure (I50.0) presents with pulmonary and systemic congestion."
            ],
            "atrial_fibrillation": [
                "Atrial fibrillation (I48) is characterized by irregular heart rhythm with absent P waves on ECG.",
                "Paroxysmal AF (I48.0) terminates spontaneously within 7 days of onset.",
                "Persistent AF (I48.1) lasts >7 days and requires cardioversion for termination.",
                "Permanent AF (I48.2) is long-standing with failed cardioversion attempts.",
                "Lone AF occurs in patients <60 years without structural heart disease.",
                "AF increases stroke risk 5-fold, requiring anticoagulation in most patients."
            ],
            
            # Respiratory system (J00-J99)
            "pneumonia": [
                "Community-acquired pneumonia (J18) typically presents with fever, cough, dyspnea, and chest pain.",
                "Bacterial pneumonia (J15) often shows lobar consolidation on chest X-ray with purulent sputum.",
                "Viral pneumonia (J12) presents with bilateral infiltrates and less pronounced systemic symptoms.",
                "Aspiration pneumonia (J69) occurs in patients with impaired consciousness or swallowing disorders.",
                "Hospital-acquired pneumonia (J15.5) develops ≥48 hours after hospital admission.",
                "Ventilator-associated pneumonia (J95.851) occurs in mechanically ventilated patients."
            ],
            "asthma": [
                "Asthma (J45) is characterized by variable airflow obstruction with wheezing, cough, and dyspnea.",
                "Allergic asthma (J45.0) is triggered by specific allergens and shows positive skin tests.",
                "Non-allergic asthma (J45.1) lacks identifiable triggers and typically develops in adulthood.",
                "Exercise-induced asthma (J45.990) occurs during or after physical activity.",
                "Occupational asthma (J45.8) is caused by workplace exposures and improves away from work.",
                "Status asthmaticus (J46) is a severe, life-threatening asthma exacerbation."
            ],
            "copd": [
                "Chronic obstructive pulmonary disease (J44) is characterized by persistent airflow limitation.",
                "Emphysema (J43) involves destruction of alveolar walls with reduced gas exchange surface.",
                "Chronic bronchitis (J42) is defined by chronic cough and sputum production for ≥3 months.",
                "COPD exacerbation (J44.1) presents with worsening dyspnea, cough, and sputum production.",
                "Alpha-1 antitrypsin deficiency (E88.01) causes early-onset emphysema and liver disease.",
                "Smoking cessation is the most effective intervention to slow COPD progression."
            ],
            
            # Neurological system (G00-G99)
            "migraine": [
                "Migraine (G43) is characterized by recurrent moderate-to-severe headaches with associated symptoms.",
                "Migraine with aura (G43.1) involves transient neurological symptoms preceding headache.",
                "Migraine without aura (G43.0) presents with unilateral throbbing headache and photophobia.",
                "Chronic migraine (G43.7) occurs ≥15 days per month for >3 months.",
                "Status migrainosus (G43.001) is a severe migraine lasting >72 hours.",
                "Medication overuse headache (G44.4) results from frequent analgesic use in migraine patients."
            ],
            "stroke": [
                "Ischemic stroke (I63) accounts for 87% of strokes and results from arterial occlusion.",
                "Hemorrhagic stroke (I61) involves bleeding into brain tissue with higher mortality.",
                "Transient ischemic attack (G45) causes temporary neurological symptoms lasting <24 hours.",
                "Lacunar stroke (I63.8) affects small penetrating arteries causing pure motor or sensory deficits.",
                "Cardioembolic stroke (I63.4) results from embolism from the heart, often in atrial fibrillation.",
                "Large vessel stroke (I63.5) involves major cerebral arteries with significant neurological deficits."
            ],
            "seizure": [
                "Epileptic seizures (G40) result from abnormal electrical activity in the brain.",
                "Generalized tonic-clonic seizures (G40.4) involve loss of consciousness with motor activity.",
                "Focal seizures (G40.1) originate from a specific brain region with preserved consciousness.",
                "Status epilepticus (G41) is continuous seizure activity lasting >30 minutes.",
                "Febrile seizures (R56.00) occur in children with fever and usually have good prognosis.",
                "Breakthrough seizures occur in epileptic patients despite adequate medication."
            ],
            
            # Gastrointestinal system (K00-K93)
            "gastritis": [
                "Gastritis (K29) is inflammation of the gastric mucosa with epigastric pain and nausea.",
                "Acute gastritis (K29.0) presents with sudden onset pain, often related to NSAIDs or alcohol.",
                "Chronic gastritis (K29.5) develops gradually and may be associated with H. pylori infection.",
                "Erosive gastritis (K29.0) shows mucosal breaks on endoscopy with higher bleeding risk.",
                "Atrophic gastritis (K29.4) involves loss of gastric glands and increased cancer risk.",
                "Stress gastritis (K29.0) occurs in critically ill patients with mucosal ischemia."
            ],
            "inflammatory_bowel_disease": [
                "Crohn's disease (K50) can affect any part of the GI tract with transmural inflammation.",
                "Ulcerative colitis (K51) involves continuous inflammation of the colonic mucosa.",
                "IBD presents with abdominal pain, diarrhea, weight loss, and systemic symptoms.",
                "Extraintestinal manifestations include arthritis, uveitis, and skin lesions.",
                "Toxic megacolon (K59.3) is a life-threatening complication requiring immediate surgery.",
                "Colorectal cancer risk is increased in long-standing IBD patients."
            ],
            
            # Musculoskeletal system (M00-M99)
            "rheumatoid_arthritis": [
                "Rheumatoid arthritis (M06) is a chronic autoimmune disease affecting synovial joints.",
                "RA typically presents with symmetric polyarthritis involving small joints of hands and feet.",
                "Morning stiffness lasting >30 minutes is characteristic of inflammatory arthritis.",
                "Rheumatoid factor (RF) and anti-CCP antibodies are important diagnostic markers.",
                "Extra-articular manifestations include rheumatoid nodules, lung disease, and vasculitis.",
                "Early aggressive treatment with DMARDs prevents joint destruction and disability."
            ],
            "osteoarthritis": [
                "Osteoarthritis (M15-M19) is a degenerative joint disease affecting cartilage and subchondral bone.",
                "Primary OA develops without identifiable cause, typically in weight-bearing joints.",
                "Secondary OA results from trauma, infection, or metabolic disorders.",
                "Pain worsens with activity and improves with rest in early disease.",
                "Heberden's nodes affect DIP joints, Bouchard's nodes affect PIP joints.",
                "Joint space narrowing, osteophytes, and subchondral sclerosis are radiographic features."
            ],
            
            # Endocrine system (E00-E90)
            "diabetes_mellitus": [
                "Type 1 diabetes (E10) results from autoimmune destruction of pancreatic beta cells.",
                "Type 2 diabetes (E11) involves insulin resistance and progressive beta cell dysfunction.",
                "Gestational diabetes (O24) develops during pregnancy and increases future diabetes risk.",
                "Diabetic ketoacidosis (E10.1) presents with hyperglycemia, ketosis, and metabolic acidosis.",
                "Hyperosmolar hyperglycemic state (E11.0) shows severe hyperglycemia without ketosis.",
                "Diabetic complications include neuropathy, retinopathy, nephropathy, and cardiovascular disease."
            ],
            "thyroid_disorders": [
                "Hyperthyroidism (E05) presents with weight loss, tremor, palpitations, and heat intolerance.",
                "Hypothyroidism (E03) causes fatigue, cold intolerance, weight gain, and depression.",
                "Graves' disease (E05.0) is the most common cause of hyperthyroidism with diffuse goiter.",
                "Hashimoto's thyroiditis (E06.3) is an autoimmune condition causing hypothyroidism.",
                "Thyroid storm (E05.5) is a life-threatening hyperthyroid crisis requiring immediate treatment.",
                "Myxedema coma (E03.5) is severe hypothyroidism with altered mental status."
            ],
            
            # Genitourinary system (N00-N99)
            "urinary_tract_infection": [
                "Urinary tract infection (N39.0) commonly presents with dysuria, frequency, and urgency.",
                "Cystitis (N30) involves bladder inflammation with suprapubic pain and hematuria.",
                "Pyelonephritis (N10) is upper UTI with flank pain, fever, and systemic symptoms.",
                "Complicated UTI occurs in patients with structural abnormalities or immunocompromise.",
                "Recurrent UTI is defined as ≥2 episodes in 6 months or ≥3 episodes in 12 months.",
                "Asymptomatic bacteriuria requires treatment only in pregnant women and before urologic procedures."
            ],
            "chronic_kidney_disease": [
                "Chronic kidney disease (N18) is defined by decreased GFR <60 mL/min/1.73m² for >3 months.",
                "CKD staging is based on GFR: Stage 1 (≥90), Stage 2 (60-89), Stage 3 (30-59), Stage 4 (15-29), Stage 5 (<15).",
                "Diabetic nephropathy (E11.2) is the leading cause of CKD in developed countries.",
                "Hypertensive nephrosclerosis (I12) is the second most common cause of CKD.",
                "CKD complications include anemia, bone disease, cardiovascular disease, and electrolyte disorders.",
                "Renal replacement therapy is needed when GFR <15 mL/min/1.73m² or uremic symptoms develop."
            ],
            
            # Hematological disorders (D50-D89)
            "anemia": [
                "Iron deficiency anemia (D50) is the most common cause of anemia worldwide.",
                "Vitamin B12 deficiency (D51) causes megaloblastic anemia with neurological symptoms.",
                "Folate deficiency (D52) also causes megaloblastic anemia but without neurological involvement.",
                "Chronic disease anemia (D63.8) results from inflammatory cytokines affecting iron metabolism.",
                "Hemolytic anemia (D58-D59) involves shortened red cell survival with elevated bilirubin.",
                "Aplastic anemia (D61) is pancytopenia due to bone marrow failure."
            ],
            
            # Infectious diseases (A00-B99)
            "bacterial_infections": [
                "Streptococcal pharyngitis (J02.0) presents with sore throat, fever, and tonsillar exudate.",
                "Pneumococcal pneumonia (J13) is the most common cause of bacterial pneumonia.",
                "Staphylococcal skin infections (L08.9) include cellulitis, abscesses, and wound infections.",
                "Urinary tract infections (N39.0) are commonly caused by E. coli and other gram-negative bacteria.",
                "Meningococcal meningitis (A39.0) presents with fever, headache, and petechial rash.",
                "Clostridium difficile colitis (A04.7) causes antibiotic-associated diarrhea and colitis."
            ],
            
            # Mental health disorders (F00-F99)
            "depression": [
                "Major depressive disorder (F32) is characterized by persistent low mood and anhedonia.",
                "Symptoms include sleep disturbances, appetite changes, fatigue, and concentration difficulties.",
                "Suicidal ideation (R45.851) requires immediate risk assessment and safety planning.",
                "Seasonal affective disorder (F33.2) occurs during winter months with light deprivation.",
                "Postpartum depression (F53.0) affects 10-15% of new mothers within the first year.",
                "Treatment includes antidepressants, psychotherapy, and lifestyle modifications."
            ],
            "anxiety_disorders": [
                "Generalized anxiety disorder (F41.1) involves excessive worry about multiple life domains.",
                "Panic disorder (F41.0) is characterized by recurrent panic attacks with physical symptoms.",
                "Social anxiety disorder (F40.1) involves fear of social situations and performance anxiety.",
                "Specific phobias (F40.2) are irrational fears of specific objects or situations.",
                "Post-traumatic stress disorder (F43.1) develops after exposure to traumatic events.",
                "Treatment includes cognitive-behavioral therapy and anxiolytic medications."
            ]
        }
        
        return knowledge_base
    
    def _load_expanded_medical_data(self):
        """Load additional medical data from external sources."""
        # This method would typically load from:
        # - Medical textbooks (digitized)
        # - PubMed abstracts
        # - Medical guidelines
        # - Clinical decision support systems
        # - Electronic health records (anonymized)
        
        # For now, we'll add some additional conditions to demonstrate scalability
        additional_conditions = {
            "skin_disorders": [
                "Psoriasis (L40) is a chronic autoimmune skin condition with scaly plaques.",
                "Eczema (L20) presents with itchy, inflamed skin patches.",
                "Melanoma (C43) is the most dangerous form of skin cancer.",
                "Basal cell carcinoma (C44) is the most common skin cancer.",
                "Acne vulgaris (L70) affects sebaceous follicles with comedones and inflammation."
            ],
            "eye_disorders": [
                "Glaucoma (H40) is increased intraocular pressure causing optic nerve damage.",
                "Cataracts (H25) involve lens opacity causing visual impairment.",
                "Diabetic retinopathy (H36.0) is a complication of diabetes affecting retinal vessels.",
                "Macular degeneration (H35.3) causes central vision loss in elderly patients.",
                "Conjunctivitis (H10) is inflammation of the conjunctiva causing red eyes."
            ],
            "ear_disorders": [
                "Otitis media (H65) is middle ear inflammation common in children.",
                "Hearing loss (H90) can be conductive, sensorineural, or mixed.",
                "Tinnitus (H93.1) is perception of sound without external source.",
                "Vertigo (H81) involves sensation of spinning or movement.",
                "Meniere's disease (H81.0) causes vertigo, hearing loss, and tinnitus."
            ]
        }
        
        # Add to existing knowledge base
        self.knowledge_base.update(additional_conditions)
        
        # Rebuild vector index if using vector search
        if self.use_vector_search:
            self._build_vector_index()
    
    def _load_pdf_content(self):
        """Load content from the symptom-to-diagnosis PDF file."""
        pdf_path = "attached_assets/symptom-to-diagnosis-an-evidence-based-guide_compress.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return
            
        try:
            print("Loading content from symptom-to-diagnosis PDF...")
            
            # Extract text from PDF using pdfplumber
            pdf_content = self._extract_pdf_content(pdf_path)
            
            if pdf_content:
                # Process PDF content into structured knowledge
                structured_content = self._process_pdf_content(pdf_content)
                
                # Add to knowledge base
                for condition, texts in structured_content.items():
                    if condition not in self.knowledge_base:
                        self.knowledge_base[condition] = []
                    self.knowledge_base[condition].extend(texts)
                
                print(f"Successfully loaded {len(structured_content)} conditions from PDF")
                
                # Rebuild vector index if using vector search
                if self.use_vector_search:
                    self._build_vector_index()
            else:
                print("No content extracted from PDF")
                
        except Exception as e:
            print(f"Error loading PDF content: {e}")
    
    def _extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF file with efficient chunking."""
        try:
            # Use pdfplumber for better text extraction
            full_text = ""
            page_count = 0
            max_pages = 50  # Limit to first 50 pages for efficiency
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    if page_num >= max_pages:
                        break
                    
                    text = page.extract_text()
                    if text:
                        # Clean up text formatting
                        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                        text = re.sub(r'(\n\s*){3,}', '\n\n', text)  # Reduce excessive newlines
                        full_text += text + "\n"
                        page_count += 1
            
            print(f"Extracted text from {page_count} pages")
            return full_text
            
        except Exception as e:
            print(f"Error extracting PDF content with pdfplumber: {e}")
            
            # Fallback to PyPDF2 with same limitations
            try:
                full_text = ""
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        if page_num >= 50:  # Same limit
                            break
                        
                        text = page.extract_text()
                        if text:
                            text = re.sub(r'\s+', ' ', text)
                            full_text += text + "\n"
                
                print(f"Extracted text using PyPDF2 fallback")
                return full_text
                
            except Exception as e2:
                print(f"Error with PyPDF2 fallback: {e2}")
                return ""
    
    def _process_pdf_content(self, pdf_text: str) -> Dict[str, List[str]]:
        """Process PDF content into structured medical knowledge with efficient chunking."""
        structured_content = {}
        
        # Split text into manageable chunks first
        chunks = self._create_text_chunks(pdf_text, max_chunk_size=2000)
        
        for i, chunk in enumerate(chunks):
            # Extract medical concepts from each chunk
            medical_concepts = self._extract_medical_concepts_from_chunk(chunk)
            
            # Create structured entries
            for concept, descriptions in medical_concepts.items():
                condition_key = concept.lower().replace(' ', '_').replace('-', '_')
                if condition_key not in structured_content:
                    structured_content[condition_key] = []
                structured_content[condition_key].extend(descriptions)
        
        return structured_content
    
    def _create_text_chunks(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """Create text chunks with medical context preservation."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current paragraph
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_medical_concepts_from_chunk(self, chunk_text: str) -> Dict[str, List[str]]:
        """Extract medical concepts from a text chunk using efficient pattern matching."""
        concepts = {}
        
        # Extract ICD-10 codes and associated descriptions
        icd_pattern = r'([A-Z]\d{2}(?:\.\d{1,2})?)\s*[:\-]?\s*([^.\n]{20,200})'
        icd_matches = re.finditer(icd_pattern, chunk_text)
        
        for match in icd_matches:
            code = match.group(1)
            description = match.group(2).strip()
            
            # Try to extract condition name from description
            condition_words = description.split()[:3]  # First 3 words usually contain condition name
            condition_name = ' '.join(condition_words).lower()
            condition_key = re.sub(r'[^a-z0-9_]', '_', condition_name)
            
            if condition_key not in concepts:
                concepts[condition_key] = []
            
            concepts[condition_key].append(f"{description} (ICD-10: {code})")
        
        # Extract symptom-diagnosis patterns more efficiently
        symptom_patterns = [
            r'([A-Z][a-z]+(?:\s+[a-z]+){0,2})\s+(?:presents?|symptoms?|signs?|characterized\s+by)\s+([^.\n]{20,150}\.)',
            r'([A-Z][a-z]+(?:\s+[a-z]+){0,2})\s*:\s*([^.\n]{20,150}\.)',
            r'Patients?\s+with\s+([A-Z][a-z]+(?:\s+[a-z]+){0,2})\s+(?:present|show|have)\s+([^.\n]{20,150}\.)',
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, chunk_text, re.IGNORECASE)
            for match in matches:
                condition = match.group(1).strip().lower()
                description = match.group(2).strip()
                
                condition_key = re.sub(r'[^a-z0-9_]', '_', condition)
                
                if condition_key not in concepts:
                    concepts[condition_key] = []
                
                concepts[condition_key].append(f"{condition.title()}: {description}")
        
        return concepts
    
    def _split_into_medical_sections(self, text: str) -> Dict[str, str]:
        """Split PDF text into medical sections."""
        sections = {}
        
        # Common medical section patterns
        section_patterns = [
            r'Chapter\s+\d+[:\s]*([^\n]+)',
            r'CHAPTER\s+\d+[:\s]*([^\n]+)',
            r'(\d+\.\d+\s+[A-Z][^.\n]+)',
            r'([A-Z][A-Z\s]+SYMPTOMS?)',
            r'([A-Z][A-Z\s]+DIAGNOSIS)',
            r'([A-Z][A-Z\s]+TREATMENT)',
            r'(CARDIOVASCULAR[^.\n]*)',
            r'(RESPIRATORY[^.\n]*)',
            r'(NEUROLOGICAL[^.\n]*)',
            r'(GASTROINTESTINAL[^.\n]*)',
            r'(MUSCULOSKELETAL[^.\n]*)',
            r'(ENDOCRINE[^.\n]*)',
            r'(GENITOURINARY[^.\n]*)',
        ]
        
        # Find all section headers
        section_headers = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                section_headers.append((match.start(), match.group(1).strip()))
        
        # Sort by position in text
        section_headers.sort(key=lambda x: x[0])
        
        # Extract content for each section
        for i, (start_pos, title) in enumerate(section_headers):
            # Get end position (start of next section or end of text)
            end_pos = section_headers[i + 1][0] if i + 1 < len(section_headers) else len(text)
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            
            # Clean up title
            clean_title = re.sub(r'^\d+\.?\d*\s*', '', title).strip()
            
            if len(section_content) > 50:  # Only include substantial sections
                sections[clean_title] = section_content
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections['General Medical Content'] = text
        
        return sections
    
    def _extract_medical_concepts_from_section(self, section_text: str) -> Dict[str, List[str]]:
        """Extract medical concepts and their descriptions from a section."""
        concepts = {}
        
        # Extract ICD-10 codes and associated descriptions
        icd_pattern = r'([A-Z]\d{2}(?:\.\d{1,2})?)\s*[:\-]?\s*([^.\n]+(?:\.[^.\n]+)*)'
        icd_matches = re.finditer(icd_pattern, section_text)
        
        for match in icd_matches:
            code = match.group(1)
            description = match.group(2).strip()
            
            if len(description) > 20:  # Only include substantial descriptions
                # Try to extract condition name
                condition_match = re.search(r'^([A-Za-z\s]+)', description)
                if condition_match:
                    condition_name = condition_match.group(1).strip().lower()
                    condition_key = condition_name.replace(' ', '_')
                    
                    if condition_key not in concepts:
                        concepts[condition_key] = []
                    
                    concepts[condition_key].append(f"{description} ({code})")
        
        # Extract symptom-diagnosis patterns
        symptom_patterns = [
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:presents?|symptoms?|signs?|characterized by)\s+([^.\n]+\.)',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s*:\s*([^.\n]+\.)',
            r'Patients?\s+with\s+([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:present|show|have)\s+([^.\n]+\.)',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+(?:characterized by|defined as|diagnosed when)\s+([^.\n]+\.)',
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, section_text, re.IGNORECASE)
            for match in matches:
                condition = match.group(1).strip().lower()
                description = match.group(2).strip()
                
                if len(description) > 20:
                    condition_key = condition.replace(' ', '_')
                    
                    if condition_key not in concepts:
                        concepts[condition_key] = []
                    
                    concepts[condition_key].append(f"{condition.title()} {description}")
        
        # Extract diagnostic criteria
        criteria_patterns = [
            r'Diagnosis(?:\s+of\s+([A-Za-z\s]+))?\s*:\s*([^.\n]+\.)',
            r'Diagnostic\s+criteria\s*(?:for\s+([A-Za-z\s]+))?\s*:\s*([^.\n]+\.)',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+diagnosed\s+when\s+([^.\n]+\.)',
        ]
        
        for pattern in criteria_patterns:
            matches = re.finditer(pattern, section_text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    condition = match.group(1).strip().lower()
                    description = match.group(2).strip()
                else:
                    # Extract condition from description
                    condition_match = re.search(r'^([A-Za-z\s]+)', match.group(2))
                    if condition_match:
                        condition = condition_match.group(1).strip().lower()
                        description = match.group(2).strip()
                    else:
                        continue
                
                if len(description) > 20:
                    condition_key = condition.replace(' ', '_')
                    
                    if condition_key not in concepts:
                        concepts[condition_key] = []
                    
                    concepts[condition_key].append(f"Diagnostic criteria for {condition.title()}: {description}")
        
        return concepts
    
    def _build_vector_index(self):
        """Build FAISS vector index for efficient similarity search."""
        if not self.use_vector_search or not self.embedding_model:
            return
            
        # Prepare documents for indexing
        documents = []
        metadata = []
        
        for condition, chunks in self.knowledge_base.items():
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    "condition": condition,
                    "chunk_id": i,
                    "source": f"{condition}_{i}"
                })
        
        # Store document metadata
        self.document_store = metadata
        
        # Generate embeddings
        try:
            embeddings = self.embedding_model.encode(documents, convert_to_tensor=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Build FAISS index
            if FAISS_AVAILABLE:
                dimension = embeddings.shape[1]
                self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                self.vector_index.add(embeddings)
            else:
                # Fallback to storing embeddings for sklearn cosine similarity
                self.vector_index = {
                    'embeddings': embeddings,
                    'use_sklearn': True
                }
                
        except Exception as e:
            print(f"Failed to build vector index: {e}")
            self.use_vector_search = False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search function with vector embeddings and TF-IDF fallback.
        
        Args:
            query: The search query (patient symptoms)
            top_k: Number of results to return
            
        Returns:
            List of relevant text chunks with similarity scores
        """
        # Use vector search if available
        if self.use_vector_search and self.vector_index is not None:
            return self._vector_search(query, top_k)
        else:
            return self._tfidf_search(query, top_k)
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector-based similarity search using sentence embeddings."""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding).astype('float32')
            
            if FAISS_AVAILABLE and hasattr(self.vector_index, 'search'):
                # Normalize query embedding for cosine similarity
                faiss.normalize_L2(query_embedding)
                
                # Search using FAISS
                scores, indices = self.vector_index.search(query_embedding, min(top_k * 2, len(self.document_store)))
                
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.document_store):
                        metadata = self.document_store[idx]
                        condition = metadata["condition"]
                        chunk_id = metadata["chunk_id"]
                        
                        results.append({
                            "text": self.knowledge_base[condition][chunk_id],
                            "similarity": float(score),
                            "source": metadata["source"],
                            "condition": condition,
                            "similarity_method": "vector_faiss"
                        })
                
            else:
                # Fallback to sklearn cosine similarity
                doc_embeddings = self.vector_index['embeddings']
                similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
                
                # Get top results
                top_indices = np.argsort(similarities)[::-1][:top_k * 2]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Minimum similarity threshold
                        metadata = self.document_store[idx]
                        condition = metadata["condition"]
                        chunk_id = metadata["chunk_id"]
                        
                        results.append({
                            "text": self.knowledge_base[condition][chunk_id],
                            "similarity": float(similarities[idx]),
                            "source": metadata["source"],
                            "condition": condition,
                            "similarity_method": "vector_sklearn"
                        })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Vector search failed: {e}")
            return self._tfidf_search(query, top_k)
    
    def _tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback TF-IDF search method."""
        # Prepare documents for TF-IDF
        documents = []
        doc_metadata = []
        
        for condition, chunks in self.knowledge_base.items():
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                doc_metadata.append({
                    "condition": condition,
                    "source": f"{condition}_{i}",
                    "chunk_index": i
                })
        
        if not documents:
            return []
        
        # Create TF-IDF vectorizer with medical stop words
        medical_stopwords = [
            'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'how', 'what', 'who', 'why',
            'this', 'that', 'these', 'those', 'a', 'an', 'are', 'is', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
            'must', 'shall', 'should', 'ought', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'without'
        ]
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=medical_stopwords,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform the documents
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Transform the query
        query_vector = vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Create results with similarity scores
        results = []
        for i, score in enumerate(similarity_scores):
            if score > 0.05:  # Minimum similarity threshold
                results.append({
                    "text": documents[i],
                    "similarity": float(score),
                    "source": doc_metadata[i]["source"],
                    "condition": doc_metadata[i]["condition"],
                    "similarity_method": "tfidf_cosine"
                })
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        total_conditions = len(self.knowledge_base)
        total_documents = sum(len(chunks) for chunks in self.knowledge_base.values())
        
        condition_stats = {}
        for condition, chunks in self.knowledge_base.items():
            condition_stats[condition] = {
                "document_count": len(chunks),
                "avg_length": np.mean([len(chunk) for chunk in chunks]),
                "total_length": sum(len(chunk) for chunk in chunks)
            }
        
        return {
            "total_conditions": total_conditions,
            "total_documents": total_documents,
            "vector_search_enabled": self.use_vector_search,
            "embedding_model": "all-MiniLM-L6-v2" if self.use_vector_search else None,
            "index_type": "FAISS" if FAISS_AVAILABLE and self.use_vector_search else "sklearn",
            "condition_stats": condition_stats
        }


def retrieve_relevant_info(symptoms: str, max_pages: int = 100) -> Dict[str, Any]:
    """
    Retrieve relevant medical information based on symptoms using real PDF RAG with chunking.
    
    Args:
        symptoms: Patient symptoms text
        max_pages: Maximum number of PDF pages to process
        
    Returns:
        Dictionary containing relevant medical information from actual PDF content
    """
    try:
        # Import here to avoid circular imports
        from utils.cached_rag import CachedRAGProcessor
        
        # Use cached RAG processor to avoid reloading PDF every time
        pdf_results = CachedRAGProcessor.search(symptoms, top_k=5, max_pages=max_pages)
        
        # Also use the traditional knowledge base as fallback
        rag_processor = RAGProcessor(load_expanded_data=True, use_vector_search=False)
        traditional_results = rag_processor.search(symptoms, top_k=3)
        
        # Combine results with priority to PDF content
        all_results = []
        icd_codes = []
        evidence_texts = []
        
        # Process PDF results first (highest priority)
        for result in pdf_results:
            evidence_texts.append(f"From Medical Textbook 'Symptom to Diagnosis': {result['text']}")
            icd_codes.extend(result.get('icd_codes', []))
            
            all_results.append({
                "condition": f"PDF Literature Finding {len(all_results) + 1}",
                "similarity": result['similarity'],
                "evidence": result['text'],
                "source": result['source'],
                "method": "pdf_real_rag",
                "icd_codes": result.get('icd_codes', [])
            })
        
        # Add traditional knowledge base results as supplementary
        for result in traditional_results:
            icd_matches = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', result['text'])
            for code in icd_matches:
                if code not in icd_codes:
                    icd_codes.append(code)
            
            evidence_texts.append(f"From Knowledge Base: {result['text']}")
            
            all_results.append({
                "condition": result['condition'],
                "similarity": result['similarity'],
                "evidence": result['text'],
                "source": result['source'],
                "method": result['similarity_method'],
                "icd_codes": icd_matches
            })
        
        # Calculate confidence based on similarity scores
        avg_similarity = np.mean([r['similarity'] for r in all_results]) if all_results else 0.0
        max_similarity = max([r['similarity'] for r in all_results]) if all_results else 0.0
        
        # Get stats from cached RAG processor
        pdf_stats = CachedRAGProcessor.get_stats(max_pages=max_pages)
        
        return {
            "relevant_conditions": all_results,
            "icd_codes": list(set(icd_codes)),  # Remove duplicates
            "evidence_texts": evidence_texts,
            "confidence": float(avg_similarity),
            "max_similarity": float(max_similarity),
            "total_results": len(all_results),
            "search_method": "real_pdf_rag_with_fallback",
            "knowledge_base_stats": {
                "pdf_loaded": pdf_stats.get('loaded', False),
                "pdf_chunks": pdf_stats.get('total_chunks', 0),
                "pdf_icd_codes": pdf_stats.get('total_icd_codes', 0),
                "avg_chunk_length": pdf_stats.get('avg_chunk_length', 0),
                "traditional_kb_conditions": len(rag_processor.knowledge_base)
            }
        }
        
    except Exception as e:
        print(f"Error in retrieve_relevant_info with real RAG: {e}")
        
        # Fallback to traditional RAG if real RAG fails
        rag_processor = RAGProcessor(load_expanded_data=True, use_vector_search=False)
        search_results = rag_processor.search(symptoms, top_k=7)
        
        relevant_conditions = []
        icd_codes = []
        evidence_texts = []
        
        for result in search_results:
            icd_matches = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', result['text'])
            for code in icd_matches:
                if code not in icd_codes:
                    icd_codes.append(code)
            
            condition_info = {
                "condition": result['condition'],
                "similarity": result['similarity'],
                "evidence": result['text'],
                "source": result['source'],
                "method": result['similarity_method']
            }
            relevant_conditions.append(condition_info)
            evidence_texts.append(result['text'])
        
        avg_similarity = np.mean([r['similarity'] for r in search_results]) if search_results else 0.0
        max_similarity = max([r['similarity'] for r in search_results]) if search_results else 0.0
        
        return {
            "relevant_conditions": relevant_conditions,
            "icd_codes": icd_codes,
            "evidence_texts": evidence_texts,
            "confidence": float(avg_similarity),
            "max_similarity": float(max_similarity),
            "total_results": len(search_results),
            "search_method": "fallback_traditional_rag",
            "knowledge_base_stats": {
                "pdf_loaded": False,
                "fallback_used": True,
                "traditional_kb_conditions": len(rag_processor.knowledge_base)
            }
        }
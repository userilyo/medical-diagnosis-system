"""
Real RAG processor that actually uses the PDF content with proper chunking.
"""
import os
import re
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import PyPDF2

class RealRAGProcessor:
    def __init__(self, max_pages: int = 100):
        """Initialize the real RAG processor.
        
        Args:
            max_pages: Maximum number of pages to process (default: 100, max recommended: 600)
        """
        self.chunks = []
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for medical terms
            lowercase=True,
            min_df=1,  # Include rare terms (medical terminology)
            max_df=0.95,  # Exclude very common terms
            sublinear_tf=True  # Use sublinear TF scaling
        )
        self.chunk_vectors = None
        self.pdf_loaded = False
        self.max_pages = min(max_pages, 600)  # Cap at 600 for performance
        
        # Load PDF content
        self._load_pdf_content()
    
    def _load_pdf_content(self):
        """Load and chunk PDF content efficiently."""
        pdf_path = "attached_assets/symptom-to-diagnosis-an-evidence-based-guide_compress.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return
        
        try:
            print("Loading PDF content with chunking...")
            
            # Extract text from specific pages (skip first page, configurable limit)
            pdf_text = ""
            page_count = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(1, min(len(pdf.pages), self.max_pages + 1)):  # Skip page 0
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:
                        # Clean and normalize text
                        text = re.sub(r'\s+', ' ', text.strip())
                        pdf_text += text + "\n\n"
                        page_count += 1
                        
                        # Process in smaller batches to avoid memory issues
                        if page_count % 10 == 0:
                            print(f"Processed {page_count} pages...")
            
            print(f"Extracted text from {page_count} pages, total length: {len(pdf_text)} characters")
            
            # Create chunks
            self.chunks = self._create_chunks(pdf_text)
            print(f"Created {len(self.chunks)} chunks")
            
            # Create TF-IDF vectors for chunks
            if self.chunks:
                chunk_texts = [chunk['text'] for chunk in self.chunks]
                self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
                print(f"Created TF-IDF vectors for {len(self.chunks)} chunks")
                self.pdf_loaded = True
            
        except Exception as e:
            print(f"Error loading PDF: {e}")
            self.pdf_loaded = False
    
    def _create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
        """Create overlapping chunks from text."""
        chunks = []
        
        # Split by double newlines to preserve paragraph structure
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size:
                # Save current chunk if it's substantial
                if len(current_chunk.strip()) > 200:  # Only save substantial chunks
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'source': 'Symptom to Diagnosis PDF',
                        'icd_codes': self._extract_icd_codes(current_chunk)
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Keep last few words for context
                    words = current_chunk.split()
                    overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if len(current_chunk.strip()) > 200:
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'source': 'Symptom to Diagnosis PDF',
                'icd_codes': self._extract_icd_codes(current_chunk)
            })
        
        return chunks
    
    def _extract_icd_codes(self, text: str) -> List[str]:
        """Extract ICD-10 codes from text."""
        icd_pattern = r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b'
        codes = re.findall(icd_pattern, text)
        return list(set(codes))  # Remove duplicates
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using TF-IDF similarity."""
        if not self.pdf_loaded or not self.chunks:
            print("PDF not loaded or no chunks available")
            return []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
            
            # Debug: Print similarity statistics
            max_sim = np.max(similarities)
            min_sim = np.min(similarities)
            avg_sim = np.mean(similarities)
            print(f"Similarity stats - Max: {max_sim:.4f}, Min: {min_sim:.4f}, Avg: {avg_sim:.4f}")
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                # Lower threshold and always return top results even if similarity is low
                if similarity > 0.001 or len(results) < 3:  # Much lower threshold
                    chunk = self.chunks[idx]
                    results.append({
                        'text': chunk['text'],
                        'similarity': float(similarity),
                        'source': chunk['source'],
                        'icd_codes': chunk['icd_codes'],
                        'chunk_id': chunk['id']
                    })
                    print(f"Added chunk {idx} with similarity {similarity:.4f}")
            
            print(f"Returning {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error in search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded content."""
        if not self.pdf_loaded:
            return {"loaded": False, "error": "PDF not loaded"}
        
        total_text_length = sum(len(chunk['text']) for chunk in self.chunks)
        total_icd_codes = len(set(code for chunk in self.chunks for code in chunk['icd_codes']))
        
        return {
            "loaded": True,
            "total_chunks": len(self.chunks),
            "total_text_length": total_text_length,
            "total_icd_codes": total_icd_codes,
            "avg_chunk_length": total_text_length / len(self.chunks) if self.chunks else 0
        }
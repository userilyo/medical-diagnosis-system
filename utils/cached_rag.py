"""
Cached RAG processor to avoid reloading PDF content every time
"""

import os
from typing import Dict, Any, Optional
from utils.real_rag_processor import RealRAGProcessor

class CachedRAGProcessor:
    _instance: Optional[RealRAGProcessor] = None
    _initialized = False
    _current_max_pages = 100  # Track current page limit
    
    @classmethod
    def get_instance(cls, max_pages: int = 100) -> RealRAGProcessor:
        """Get singleton instance of RAG processor."""
        # Reset cache if page limit changes significantly
        if cls._instance is not None and abs(cls._current_max_pages - max_pages) > 10:
            print(f"Page limit changed from {cls._current_max_pages} to {max_pages}, resetting cache...")
            cls.reset()
        
        if cls._instance is None or not cls._initialized:
            print(f"Initializing cached RAG processor with {max_pages} pages...")
            cls._instance = RealRAGProcessor(max_pages=max_pages)
            cls._initialized = True
            cls._current_max_pages = max_pages
        return cls._instance
    
    @classmethod
    def search(cls, query: str, top_k: int = 5, max_pages: int = 100) -> list:
        """Search using cached RAG processor."""
        instance = cls.get_instance(max_pages=max_pages)
        return instance.search(query, top_k)
    
    @classmethod
    def get_stats(cls, max_pages: int = 100) -> Dict[str, Any]:
        """Get stats from cached RAG processor."""
        instance = cls.get_instance(max_pages=max_pages)
        return instance.get_stats()
    
    @classmethod
    def reset(cls):
        """Reset the cached instance."""
        cls._instance = None
        cls._initialized = False
        cls._current_max_pages = 100


def retrieve_relevant_info_cached(symptoms: str, max_pages: int = 100) -> Dict[str, Any]:
    """
    Retrieve relevant medical information using cached RAG processor.
    This version is much faster as it doesn't reload the PDF every time.
    """
    try:
        # Use cached RAG processor with specified page limit
        pdf_results = CachedRAGProcessor.search(symptoms, top_k=5, max_pages=max_pages)
        
        # Get stats to verify PDF loading
        pdf_stats = CachedRAGProcessor.get_stats(max_pages=max_pages)
        
        # Process results
        all_results = []
        icd_codes = []
        evidence_texts = []
        
        # Process PDF results
        for result in pdf_results:
            evidence_texts.append(f"From Medical Textbook 'Symptom to Diagnosis': {result['text']}")
            icd_codes.extend(result.get('icd_codes', []))
            
            all_results.append({
                "condition": f"PDF Literature Finding {len(all_results) + 1}",
                "similarity": result['similarity'],
                "evidence": result['text'],
                "source": result['source'],
                "method": "pdf_real_rag_cached",
                "icd_codes": result.get('icd_codes', [])
            })
        
        # Calculate confidence
        avg_similarity = sum(r['similarity'] for r in all_results) / len(all_results) if all_results else 0.0
        max_similarity = max(r['similarity'] for r in all_results) if all_results else 0.0
        
        return {
            "relevant_conditions": all_results,
            "icd_codes": list(set(icd_codes)),
            "evidence_texts": evidence_texts,
            "confidence": float(avg_similarity),
            "max_similarity": float(max_similarity),
            "total_results": len(all_results),
            "search_method": "cached_real_pdf_rag",
            "knowledge_base_stats": {
                "pdf_loaded": pdf_stats.get('loaded', False),
                "pdf_chunks": pdf_stats.get('total_chunks', 0),
                "pdf_icd_codes": pdf_stats.get('total_icd_codes', 0),
                "avg_chunk_length": pdf_stats.get('avg_chunk_length', 0)
            }
        }
        
    except Exception as e:
        print(f"Error in cached RAG: {e}")
        return {
            "relevant_conditions": [],
            "icd_codes": [],
            "evidence_texts": [],
            "confidence": 0.0,
            "max_similarity": 0.0,
            "total_results": 0,
            "search_method": "cached_rag_failed",
            "knowledge_base_stats": {"pdf_loaded": False, "error": str(e)}
        }
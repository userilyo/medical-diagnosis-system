"""
Quick test to verify we're getting real PDF content
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cached_rag import CachedRAGProcessor

def main():
    print("Testing cached RAG processor...")
    
    # Test search with cached processor
    symptoms = "shortness of breath and chest pain"
    results = CachedRAGProcessor.search(symptoms, top_k=3, max_pages=50)
    
    print(f"\nSearching for: '{symptoms}'")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Source: {result['source']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Text Preview: {result['text'][:150]}...")
        if result.get('icd_codes'):
            print(f"   ICD Codes: {result['icd_codes']}")
    
    # Get stats
    stats = CachedRAGProcessor.get_stats(max_pages=50)
    print(f"\nCached Processor Stats:")
    print(f"- PDF Loaded: {stats['pdf_loaded']}")
    print(f"- Total Chunks: {stats['pdf_chunks']}")
    print(f"- Average Chunk Length: {stats['avg_chunk_length']:.0f}")

if __name__ == "__main__":
    main()
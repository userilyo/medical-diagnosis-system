"""
Simple verification that we're getting real PDF content
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.real_rag_processor import RealRAGProcessor

def main():
    print("Testing PDF content extraction...")
    
    # Initialize RAG processor
    processor = RealRAGProcessor(max_pages=50)
    
    # Test search
    results = processor.search("chest pain", top_k=3)
    
    print(f"\nFound {len(results)} results for 'chest pain':")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Source: {result['source']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Text: {result['text'][:200]}...")
        print(f"   ICD Codes: {result.get('icd_codes', [])}")
    
    # Get stats
    stats = processor.get_stats()
    print(f"\nProcessor Stats:")
    print(f"- PDF Loaded: {stats['pdf_loaded']}")
    print(f"- Total Chunks: {stats['pdf_chunks']}")
    print(f"- Average Chunk Length: {stats['avg_chunk_length']:.0f}")

if __name__ == "__main__":
    main()
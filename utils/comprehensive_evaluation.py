"""
Comprehensive evaluation module using the full ICD-10 hierarchical graph
"""

from typing import Dict, List, Any, Tuple
import logging
from utils.icd10_comprehensive import get_icd10_graph, calculate_hierarchical_accuracy

# Set up logging
logger = logging.getLogger(__name__)

def evaluate_predictions_comprehensive(predictions: List[Dict[str, Any]], 
                                     ground_truth: List[str]) -> Dict[str, Any]:
    """
    Evaluate predictions using comprehensive hierarchical ICD-10 matching
    
    Args:
        predictions: List of prediction dictionaries with 'predicted_icd' key
        ground_truth: List of ground truth ICD-10 codes
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    # Initialize metrics
    metrics = {
        'total_predictions': len(predictions),
        'exact_matches': 0,
        'category_matches': 0,
        'block_matches': 0,
        'hierarchical_similarities': [],
        'weighted_scores': [],
        'detailed_results': []
    }
    
    # Get ICD-10 graph
    graph = get_icd10_graph()
    
    # Evaluate each prediction
    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        predicted_code = pred.get('predicted_icd', '')
        
        # Calculate hierarchical accuracy
        hierarchical_metrics = calculate_hierarchical_accuracy(predicted_code, truth)
        
        # Update overall metrics
        metrics['exact_matches'] += hierarchical_metrics['exact_match']
        metrics['category_matches'] += hierarchical_metrics['category_match']
        metrics['block_matches'] += hierarchical_metrics['block_match']
        metrics['hierarchical_similarities'].append(hierarchical_metrics['hierarchical_similarity'])
        metrics['weighted_scores'].append(hierarchical_metrics['weighted_score'])
        
        # Add detailed result
        detailed_result = {
            'index': i,
            'predicted_code': predicted_code,
            'ground_truth_code': truth,
            'predicted_description': graph.code_to_description.get(predicted_code, ''),
            'ground_truth_description': graph.code_to_description.get(truth, ''),
            'hierarchical_metrics': hierarchical_metrics,
            'hierarchical_distance': graph.get_hierarchical_distance(predicted_code, truth)
        }
        
        metrics['detailed_results'].append(detailed_result)
    
    # Calculate final percentages
    total = metrics['total_predictions']
    if total > 0:
        metrics['exact_match_percentage'] = (metrics['exact_matches'] / total) * 100
        metrics['category_match_percentage'] = (metrics['category_matches'] / total) * 100
        metrics['block_match_percentage'] = (metrics['block_matches'] / total) * 100
        metrics['avg_hierarchical_similarity'] = sum(metrics['hierarchical_similarities']) / total
        metrics['avg_weighted_score'] = sum(metrics['weighted_scores']) / total
    else:
        metrics['exact_match_percentage'] = 0
        metrics['category_match_percentage'] = 0
        metrics['block_match_percentage'] = 0
        metrics['avg_hierarchical_similarity'] = 0
        metrics['avg_weighted_score'] = 0
    
    return metrics

def get_enhanced_hierarchical_info(code: str) -> Dict[str, Any]:
    """
    Get enhanced hierarchical information for a code using comprehensive graph
    
    Args:
        code: ICD-10 code
        
    Returns:
        Enhanced hierarchical information
    """
    graph = get_icd10_graph()
    
    # Get basic info
    code_info = graph.get_code_info(code)
    
    if not code_info:
        return {
            'code': code,
            'description': 'Unknown code',
            'level': 0,
            'hierarchy_path': [],
            'related_codes': []
        }
    
    # Build hierarchy path
    hierarchy_path = []
    ancestors = code_info.get('ancestors', [])
    
    # Sort ancestors by level
    ancestor_info = []
    for ancestor in ancestors:
        ancestor_data = graph.get_code_info(ancestor)
        if ancestor_data:
            ancestor_info.append({
                'code': ancestor,
                'description': ancestor_data.get('description', ''),
                'level': ancestor_data.get('level', 0)
            })
    
    # Sort by level (root to leaf)
    ancestor_info.sort(key=lambda x: x['level'])
    
    for ancestor in ancestor_info:
        if ancestor['level'] > 0:  # Skip root
            hierarchy_path.append({
                'level': ancestor['level'],
                'code': ancestor['code'],
                'description': ancestor['description']
            })
    
    # Add current code
    hierarchy_path.append({
        'level': code_info.get('level', 0),
        'code': code,
        'description': code_info.get('description', '')
    })
    
    # Get related codes (siblings and children)
    related_codes = []
    
    # Add siblings
    siblings = code_info.get('siblings', [])
    for sibling in siblings[:5]:  # Limit to avoid too many
        sibling_info = graph.get_code_info(sibling)
        if sibling_info:
            related_codes.append({
                'relationship': 'sibling',
                'code': sibling,
                'description': sibling_info.get('description', '')
            })
    
    # Add children
    children = code_info.get('children', [])
    for child in children[:5]:  # Limit to avoid too many
        child_info = graph.get_code_info(child)
        if child_info:
            related_codes.append({
                'relationship': 'child',
                'code': child,
                'description': child_info.get('description', '')
            })
    
    return {
        'code': code,
        'description': code_info.get('description', ''),
        'level': code_info.get('level', 0),
        'hierarchy_path': hierarchy_path,
        'related_codes': related_codes
    }

def find_similar_codes(code: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Find codes similar to the given code using hierarchical relationships
    
    Args:
        code: ICD-10 code to find similar codes for
        max_results: Maximum number of similar codes to return
        
    Returns:
        List of similar codes with similarity scores
    """
    graph = get_icd10_graph()
    
    # Get code info
    code_info = graph.get_code_info(code)
    if not code_info:
        return []
    
    similar_codes = []
    
    # Get all nodes in the graph
    all_nodes = list(graph.graph.nodes())
    
    # Calculate similarity to each node
    for node in all_nodes:
        if node == code or node.startswith('ROOT') or node.startswith('CHAPTER') or node.startswith('BLOCK'):
            continue
            
        # Calculate hierarchical distance
        similarity = graph.get_hierarchical_distance(code, node)
        
        if similarity > 0:
            node_info = graph.get_code_info(node)
            if node_info:
                similar_codes.append({
                    'code': node,
                    'description': node_info.get('description', ''),
                    'similarity': similarity,
                    'level': node_info.get('level', 0)
                })
    
    # Sort by similarity and return top results
    similar_codes.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_codes[:max_results]

def expand_query_with_comprehensive_hierarchy(query: str, max_expansions: int = 10) -> List[str]:
    """
    Expand a query with hierarchically related terms using comprehensive graph
    
    Args:
        query: Original query string
        max_expansions: Maximum number of expanded terms
        
    Returns:
        List of expanded query terms
    """
    graph = get_icd10_graph()
    
    # Search for codes matching the query
    matches = graph.search_codes(query, max_results=5)
    
    expanded_terms = [query]
    
    for match in matches:
        code = match['code']
        
        # Skip structural nodes
        if code.startswith('ROOT') or code.startswith('CHAPTER') or code.startswith('BLOCK'):
            continue
            
        # Get related codes
        code_info = graph.get_code_info(code)
        if code_info:
            # Add children descriptions
            for child in code_info.get('children', [])[:3]:
                child_info = graph.get_code_info(child)
                if child_info:
                    expanded_terms.append(child_info.get('description', ''))
            
            # Add sibling descriptions
            for sibling in code_info.get('siblings', [])[:3]:
                sibling_info = graph.get_code_info(sibling)
                if sibling_info:
                    expanded_terms.append(sibling_info.get('description', ''))
    
    # Remove duplicates while preserving order
    unique_terms = []
    seen = set()
    for term in expanded_terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)
    
    return unique_terms[:max_expansions]

def get_comprehensive_evaluation_summary(predictions: List[Dict[str, Any]], 
                                        ground_truth: List[str]) -> str:
    """
    Get a comprehensive evaluation summary with hierarchical metrics
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth codes
        
    Returns:
        Formatted summary string
    """
    metrics = evaluate_predictions_comprehensive(predictions, ground_truth)
    
    summary = f"""
=== Comprehensive ICD-10 Evaluation Summary ===

Total Predictions: {metrics['total_predictions']}

Hierarchical Accuracy Metrics:
• Exact Match: {metrics['exact_match_percentage']:.1f}% ({metrics['exact_matches']}/{metrics['total_predictions']})
• Category Match: {metrics['category_match_percentage']:.1f}% ({metrics['category_matches']}/{metrics['total_predictions']})
• Block Match: {metrics['block_match_percentage']:.1f}% ({metrics['block_matches']}/{metrics['total_predictions']})

Similarity Metrics:
• Average Hierarchical Similarity: {metrics['avg_hierarchical_similarity']:.3f}
• Average Weighted Score: {metrics['avg_weighted_score']:.3f}

Performance Interpretation:
• Exact Match: Predicted code exactly matches ground truth
• Category Match: Predicted code is in same 3-character category
• Block Match: Predicted code is in same ICD-10 block
• Hierarchical Similarity: Closeness in the ICD-10 hierarchy tree
• Weighted Score: Combined metric considering all hierarchy levels

Graph Coverage: {len(get_icd10_graph().graph.nodes())} ICD-10 codes in hierarchy
"""
    
    return summary

def test_comprehensive_evaluation():
    """
    Test comprehensive evaluation functionality
    """
    print("=== Testing Comprehensive Evaluation ===")
    
    # Test with sample predictions
    predictions = [
        {'predicted_icd': 'I21.4'},
        {'predicted_icd': 'I25.10'},
        {'predicted_icd': 'J18.9'},
        {'predicted_icd': 'Z38.00'},
        {'predicted_icd': 'A41.9'}
    ]
    
    ground_truth = ['I21.4', 'I25.10', 'J96.00', 'Z38.01', 'A41.9']
    
    # Evaluate
    metrics = evaluate_predictions_comprehensive(predictions, ground_truth)
    
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Exact matches: {metrics['exact_match_percentage']:.1f}%")
    print(f"Category matches: {metrics['category_match_percentage']:.1f}%")
    print(f"Block matches: {metrics['block_match_percentage']:.1f}%")
    print(f"Average hierarchical similarity: {metrics['avg_hierarchical_similarity']:.3f}")
    print(f"Average weighted score: {metrics['avg_weighted_score']:.3f}")
    
    # Test hierarchical info
    print("\n=== Testing Hierarchical Info ===")
    code = 'I21.4'
    hierarchical_info = get_enhanced_hierarchical_info(code)
    print(f"Code: {code}")
    print(f"Description: {hierarchical_info['description']}")
    print(f"Level: {hierarchical_info['level']}")
    print(f"Hierarchy path: {len(hierarchical_info['hierarchy_path'])} levels")
    print(f"Related codes: {len(hierarchical_info['related_codes'])}")
    
    # Test similar codes
    print("\n=== Testing Similar Codes ===")
    similar = find_similar_codes(code, max_results=5)
    print(f"Similar codes to {code}:")
    for similar_code in similar:
        print(f"  {similar_code['code']}: {similar_code['description']} (similarity: {similar_code['similarity']:.3f})")
    
    print("\n=== Comprehensive Evaluation Test Complete ===")

if __name__ == "__main__":
    test_comprehensive_evaluation()
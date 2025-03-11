from typing import Dict, Any

def evaluate_document_value(content: str, filename: str) -> Dict[str, Any]:
    """
    Evaluate the estimated value of a document based on its content.
    
    Args:
        content (str): Document content
        filename (str): Name of the document
        
    Returns:
        dict: Dictionary with estimated value and factors
    """
    # Default value
    base_value = 10.0
    
    # Initialize factors list
    factors = []
    
    # Calculate length factor (longer documents may be more valuable)
    length = len(content)
    length_factor = min(length / 5000, 5)  # Cap at 5x
    factors.append(f"Length factor: {length_factor:.2f}x (document has {length} characters)")
    
    # Check for technical content
    technical_terms = ['algorithm', 'analysis', 'methodology', 'framework', 'implementation', 
                      'architecture', 'infrastructure', 'configuration', 'specification']
    technical_count = sum(1 for term in technical_terms if term.lower() in content.lower())
    technical_factor = 1 + (technical_count / 10)
    factors.append(f"Technical content factor: {technical_factor:.2f}x ({technical_count} technical terms found)")
    
    # Check for data and statistics
    data_terms = ['data', 'statistics', 'metrics', 'measurement', 'percentage', 'analysis', 
                 'figure', 'table', 'chart', 'graph']
    data_count = sum(1 for term in data_terms if term.lower() in content.lower())
    data_factor = 1 + (data_count / 20)
    factors.append(f"Data richness factor: {data_factor:.2f}x ({data_count} data-related terms found)")
    
    # Calculate final value
    estimated_value = base_value * length_factor * technical_factor * data_factor
    
    # Round to 2 decimal places
    estimated_value = round(estimated_value, 2)
    
    return {
        'estimated_value': estimated_value,
        'factors': factors
    }

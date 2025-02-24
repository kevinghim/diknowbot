from typing import Dict, Any

def evaluate_document_value(content: str, filename: str) -> Dict[str, Any]:
    """
    Evaluate a document's potential value using a comprehensive formula:
    Value = Base Value × Multiplier × Uniqueness Factor × Quality Factor × Demand Adjustment
    """
    factors = []
    
    # 1. Determine Base Value based on document type and content indicators
    base_value = 50.0  # Default base value
    
    # Check for document type indicators
    scientific_indicators = {'methodology', 'research', 'analysis', 'study', 'findings', 
                           'hypothesis', 'experiment', 'data', 'conclusion'}
    technical_indicators = {'algorithm', 'implementation', 'architecture', 'system', 
                          'framework', 'code', 'api', 'database', 'protocol'}
    business_indicators = {'strategy', 'market', 'financial', 'business', 'revenue', 
                         'commercial', 'industry', 'enterprise'}
    
    # Count occurrences of each type of indicator
    scientific_score = sum(1 for word in scientific_indicators if word.lower() in content.lower())
    technical_score = sum(1 for word in technical_indicators if word.lower() in content.lower())
    business_score = sum(1 for word in business_indicators if word.lower() in content.lower())
    
    # Adjust base value according to document type
    if scientific_score > 5:
        base_value = 500.0
        factors.append("Scientific document (high base value)")
    elif technical_score > 5:
        base_value = 300.0
        factors.append("Technical document (medium-high base value)")
    elif business_score > 5:
        base_value = 200.0
        factors.append("Business document (medium base value)")
    
    # 2. Calculate Size Multiplier
    word_count = len(content.split())
    if word_count < 500:
        size_multiplier = 0.8
        factors.append("Short document (0.8x multiplier)")
    elif word_count < 2000:
        size_multiplier = 1.0
        factors.append("Standard length document (1.0x multiplier)")
    elif word_count < 5000:
        size_multiplier = 1.3
        factors.append("Long document (1.3x multiplier)")
    else:
        size_multiplier = 1.5
        factors.append("Comprehensive document (1.5x multiplier)")
    
    # 3. Calculate Uniqueness Factor
    uniqueness_score = 0
    
    # Check for unique/specialized content indicators
    unique_indicators = {
        'proprietary': 10,
        'novel': 8,
        'innovative': 7,
        'patent': 10,
        'exclusive': 9,
        'breakthrough': 9,
        'original': 7,
        'unique': 6
    }
    
    for term, score in unique_indicators.items():
        if term.lower() in content.lower():
            uniqueness_score += score
    
    # Normalize uniqueness score to 0-100 range and apply square formula
    uniqueness_score = min(uniqueness_score, 100)
    uniqueness_factor = (uniqueness_score / 100) ** 2
    factors.append(f"Uniqueness score: {uniqueness_score}/100 (factor: {uniqueness_factor:.2f})")
    
    # 4. Calculate Quality Factor
    quality_score = 1.0
    
    # Check for quality indicators
    quality_indicators = {
        'well-documented': 0.1,
        'comprehensive': 0.1,
        'detailed': 0.08,
        'accurate': 0.08,
        'verified': 0.1,
        'peer-reviewed': 0.15,
        'validated': 0.1,
        'certified': 0.12
    }
    
    for indicator, bonus in quality_indicators.items():
        if indicator.lower() in content.lower():
            quality_score += bonus
    
    quality_factor = min(max(quality_score, 0.5), 1.5)
    factors.append(f"Quality factor: {quality_factor:.2f}")
    
    # 5. Calculate Demand Adjustment
    demand_adjustment = 1.0
    
    # High-demand topics
    high_demand_topics = {
        'machine learning': 0.3,
        'artificial intelligence': 0.3,
        'data science': 0.25,
        'blockchain': 0.2,
        'cybersecurity': 0.25,
        'cloud computing': 0.2,
        'deep learning': 0.3,
        'neural networks': 0.25
    }
    
    for topic, bonus in high_demand_topics.items():
        if topic.lower() in content.lower():
            demand_adjustment += bonus
    
    demand_adjustment = min(demand_adjustment, 2.0)
    factors.append(f"Market demand adjustment: {demand_adjustment:.2f}x")
    
    # Calculate final value
    final_value = base_value * size_multiplier * uniqueness_factor * quality_factor * demand_adjustment
    
    # Add file format considerations
    file_ext = filename.split('.')[-1].lower()
    if file_ext == 'pdf':
        final_value *= 1.1
        factors.append("PDF format bonus (1.1x)")
    elif file_ext == 'docx':
        final_value *= 1.05
        factors.append("DOCX format bonus (1.05x)")
    
    # Summary calculation
    factors.append(f"Base value: ${base_value:.2f}")
    factors.append(f"Final calculation: ${base_value:.2f} × {size_multiplier:.2f} × {uniqueness_factor:.2f} × {quality_factor:.2f} × {demand_adjustment:.2f}")
    
    return {
        'estimated_value': round(final_value, 2),
        'factors': factors,
        'currency': 'USD'
    }

from typing import Dict, Any

def evaluate_document_value(content: str, filename: str) -> Dict[str, Any]:
    """
    Evaluate a document's potential value using comprehensive formula including authority metrics:
    Value = Base Value × Multiplier × Uniqueness Factor × Quality Factor × Demand Adjustment × Authority Multiplier
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
    
    # 6. Calculate Authority Multiplier
    authority_score = 0.7  # Base authority score
    
    # Credential indicators
    credential_indicators = {
        'ph.d': 0.8,
        'professor': 0.8,
        'dr.': 0.6,
        'expert': 0.4,
        'researcher': 0.4,
        'scientist': 0.4,
        'director': 0.3,
        'chief': 0.3,
        'lead': 0.2
    }
    
    # Citation impact indicators
    citation_indicators = {
        'cited': 0.3,
        'referenced': 0.3,
        'published in': 0.4,
        'journal': 0.3,
        'peer-reviewed': 0.5,
        'conference': 0.3
    }
    
    # Domain recognition indicators
    domain_indicators = {
        'renowned': 0.4,
        'recognized': 0.3,
        'leading': 0.3,
        'pioneer': 0.4,
        'authority': 0.4,
        'specialist': 0.3
    }
    
    # Community standing indicators
    community_indicators = {
        'committee': 0.3,
        'board member': 0.4,
        'chairman': 0.4,
        'founder': 0.3,
        'advisor': 0.3
    }
    
    # Industry authority indicators
    industry_indicators = {
        'industry leader': 0.4,
        'thought leader': 0.4,
        'innovator': 0.3,
        'veteran': 0.3,
        'executive': 0.3
    }
    
    content_lower = content.lower()
    
    # Calculate credential score
    credential_score = sum(score for term, score in credential_indicators.items() 
                         if term in content_lower)
    credential_score = min(credential_score, 1.0)
    
    # Calculate citation impact
    citation_score = sum(score for term, score in citation_indicators.items() 
                        if term in content_lower)
    citation_score = min(citation_score, 1.0)
    
    # Calculate domain recognition
    domain_score = sum(score for term, score in domain_indicators.items() 
                      if term in content_lower)
    domain_score = min(domain_score, 1.0)
    
    # Calculate community standing
    community_score = sum(score for term, score in community_indicators.items() 
                         if term in content_lower)
    community_score = min(community_score, 1.0)
    
    # Calculate industry authority
    industry_score = sum(score for term, score in industry_indicators.items() 
                        if term in content_lower)
    industry_score = min(industry_score, 1.0)
    
    # Calculate final authority multiplier
    authority_multiplier = (0.7 + 
                          (credential_score * 0.05) +
                          (citation_score * 0.1) +
                          (domain_score * 0.05) +
                          (community_score * 0.05) +
                          (industry_score * 0.05))
    
    authority_multiplier = min(max(authority_multiplier, 0.7), 1.3)
    
    # Add authority factors to the explanation
    if credential_score > 0:
        factors.append(f"Credential score: {credential_score:.2f}")
    if citation_score > 0:
        factors.append(f"Citation impact: {citation_score:.2f}")
    if domain_score > 0:
        factors.append(f"Domain recognition: {domain_score:.2f}")
    if community_score > 0:
        factors.append(f"Community standing: {community_score:.2f}")
    if industry_score > 0:
        factors.append(f"Industry authority: {industry_score:.2f}")
    
    factors.append(f"Authority multiplier: {authority_multiplier:.2f}x")
    
    # Calculate final value with authority multiplier
    final_value = (base_value * size_multiplier * uniqueness_factor * 
                  quality_factor * demand_adjustment * authority_multiplier)
    
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
    factors.append(
        f"Final calculation: ${base_value:.2f} × {size_multiplier:.2f} × "
        f"{uniqueness_factor:.2f} × {quality_factor:.2f} × "
        f"{demand_adjustment:.2f} × {authority_multiplier:.2f}"
    )
    
    return {
        'estimated_value': round(final_value, 2),
        'factors': factors,
        'currency': 'USD'
    }

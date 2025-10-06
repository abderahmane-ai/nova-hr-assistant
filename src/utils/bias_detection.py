"""
Bias Detection and Mitigation Utilities for Nova HR Assistant.

This module provides tools to detect and mitigate potential biases in candidate scoring
and evaluation processes.
"""

import logging
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BiasIndicator:
    """Represents a detected bias indicator."""
    bias_type: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    affected_components: List[str]
    mitigation_suggestion: str


class BiasDetector:
    """Detects potential biases in candidate evaluation."""
    
    # Common bias patterns and keywords
    GENDER_BIAS_PATTERNS = [
        r'\b(aggressive|assertive|bossy|emotional|hysterical)\b',
        r'\b(leadership|management|technical|analytical)\b.*\b(woman|female|girl)\b',
        r'\b(soft|gentle|nurturing|caring)\b.*\b(man|male|guy)\b'
    ]
    
    AGE_BIAS_PATTERNS = [
        r'\b(young|old|senior|junior|experienced|fresh|veteran)\b.*\b(energy|vitality|stamina|wisdom|maturity)\b',
        r'\b(digital native|millennial|gen z|boomer|senior citizen)\b',
        r'\b(overqualified|underqualified)\b'
    ]
    
    RACE_ETHNICITY_BIAS_PATTERNS = [
        r'\b(cultural fit|diversity hire|affirmative action)\b',
        r'\b(foreign|international|immigrant|native)\b.*\b(accent|language|communication)\b'
    ]
    
    EDUCATION_BIAS_PATTERNS = [
        r'\b(ivy league|prestigious|elite|top tier)\b.*\b(university|college|school)\b',
        r'\b(community college|state school|online degree)\b.*\b(lesser|inferior|lower quality)\b'
    ]
    
    EXPERIENCE_BIAS_PATTERNS = [
        r'\b(overqualified|underqualified|too experienced|not enough experience)\b',
        r'\b(career gap|unemployment|job hopping|stability)\b'
    ]
    
    @classmethod
    def detect_biases(cls, analysis_data: Dict[str, Any]) -> List[BiasIndicator]:
        """
        Detect potential biases in the analysis data.
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            List of detected bias indicators
        """
        biases = []
        
        # Check text content for bias patterns
        text_content = cls._extract_text_content(analysis_data)
        biases.extend(cls._detect_text_biases(text_content))
        
        # Check scoring patterns for bias
        biases.extend(cls._detect_scoring_biases(analysis_data))
        
        # Check for demographic bias indicators
        biases.extend(cls._detect_demographic_biases(analysis_data))
        
        return biases
    
    @classmethod
    def _extract_text_content(cls, analysis_data: Dict[str, Any]) -> str:
        """Extract all text content from analysis data."""
        text_parts = []
        
        # Extract from various analysis components
        for component in ['experience', 'skills', 'education', 'certifications']:
            if component in analysis_data:
                component_data = analysis_data[component]
                if isinstance(component_data, dict):
                    for key, value in component_data.items():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            text_parts.extend([str(item) for item in value if isinstance(item, str)])
        
        return ' '.join(text_parts).lower()
    
    @classmethod
    def _detect_text_biases(cls, text_content: str) -> List[BiasIndicator]:
        """Detect biases in text content."""
        biases = []
        
        # Check for gender bias
        for pattern in cls.GENDER_BIAS_PATTERNS:
            if re.search(pattern, text_content, re.IGNORECASE):
                biases.append(BiasIndicator(
                    bias_type="gender",
                    severity="medium",
                    description="Potential gender bias detected in language",
                    affected_components=["text_analysis"],
                    mitigation_suggestion="Review language for gender-neutral alternatives"
                ))
        
        # Check for age bias
        for pattern in cls.AGE_BIAS_PATTERNS:
            if re.search(pattern, text_content, re.IGNORECASE):
                biases.append(BiasIndicator(
                    bias_type="age",
                    severity="medium",
                    description="Potential age bias detected in language",
                    affected_components=["text_analysis"],
                    mitigation_suggestion="Focus on skills and qualifications rather than age-related attributes"
                ))
        
        # Check for race/ethnicity bias
        for pattern in cls.RACE_ETHNICITY_BIAS_PATTERNS:
            if re.search(pattern, text_content, re.IGNORECASE):
                biases.append(BiasIndicator(
                    bias_type="race_ethnicity",
                    severity="high",
                    description="Potential race/ethnicity bias detected",
                    affected_components=["text_analysis"],
                    mitigation_suggestion="Remove any references to cultural background or ethnicity"
                ))
        
        # Check for education bias
        for pattern in cls.EDUCATION_BIAS_PATTERNS:
            if re.search(pattern, text_content, re.IGNORECASE):
                biases.append(BiasIndicator(
                    bias_type="education",
                    severity="medium",
                    description="Potential education bias detected",
                    affected_components=["education_analysis"],
                    mitigation_suggestion="Focus on relevant skills and knowledge rather than institution prestige"
                ))
        
        return biases
    
    @classmethod
    def _detect_scoring_biases(cls, analysis_data: Dict[str, Any]) -> List[BiasIndicator]:
        """Detect biases in scoring patterns."""
        biases = []
        
        # Check for extreme score variations
        scoring_details = analysis_data.get("scoring_details", {})
        component_scores = scoring_details.get("component_scores", {})
        
        if component_scores:
            scores = list(component_scores.values())
            if scores:
                score_range = max(scores) - min(scores)
                if score_range > 60:
                    biases.append(BiasIndicator(
                        bias_type="scoring_consistency",
                        severity="medium",
                        description="Large variation in component scores may indicate bias",
                        affected_components=["scoring"],
                        mitigation_suggestion="Review scoring criteria for consistency across components"
                    ))
        
        # Check for education over-weighting
        education_score = component_scores.get("education", 0)
        overall_score = scoring_details.get("overall_score", 0)
        
        if education_score > overall_score + 20:
            biases.append(BiasIndicator(
                bias_type="education_overweighting",
                severity="low",
                description="Education score significantly higher than overall score",
                affected_components=["education_analysis", "scoring"],
                mitigation_suggestion="Ensure education weight is appropriate for role requirements"
            ))
        
        return biases
    
    @classmethod
    def _detect_demographic_biases(cls, analysis_data: Dict[str, Any]) -> List[BiasIndicator]:
        """Detect demographic bias indicators."""
        biases = []
        
        # Check for name-based assumptions
        candidate_info = analysis_data.get("candidate", {})
        candidate_name = candidate_info.get("name", "")
        
        if candidate_name:
            # Check for potential name-based bias indicators
            if any(name_part in candidate_name.lower() for name_part in 
                   ['jr', 'sr', 'iii', 'iv', 'v']):
                biases.append(BiasIndicator(
                    bias_type="name_assumptions",
                    severity="low",
                    description="Name contains generational indicators that might influence assessment",
                    affected_components=["candidate_info"],
                    mitigation_suggestion="Focus on qualifications rather than name characteristics"
                ))
        
        return biases


class BiasMitigator:
    """Provides bias mitigation strategies."""
    
    @classmethod
    def apply_mitigation_strategies(cls, analysis_data: Dict[str, Any], 
                                  biases: List[BiasIndicator]) -> Dict[str, Any]:
        """
        Apply bias mitigation strategies to analysis data.
        
        Args:
            analysis_data: Original analysis data
            biases: List of detected biases
            
        Returns:
            Mitigated analysis data
        """
        mitigated_data = analysis_data.copy()
        
        for bias in biases:
            if bias.bias_type == "gender":
                mitigated_data = cls._mitigate_gender_bias(mitigated_data)
            elif bias.bias_type == "age":
                mitigated_data = cls._mitigate_age_bias(mitigated_data)
            elif bias.bias_type == "education":
                mitigated_data = cls._mitigate_education_bias(mitigated_data)
            elif bias.bias_type == "scoring_consistency":
                mitigated_data = cls._mitigate_scoring_bias(mitigated_data)
        
        return mitigated_data
    
    @classmethod
    def _mitigate_gender_bias(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply gender bias mitigation strategies."""
        # Remove gender-specific language from text fields
        text_fields = ['justification', 'summary', 'description']
        
        for field in text_fields:
            if field in data:
                data[field] = cls._neutralize_gender_language(data[field])
        
        return data
    
    @classmethod
    def _mitigate_age_bias(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply age bias mitigation strategies."""
        # Focus on skills and experience rather than age-related attributes
        if "experience" in data:
            exp_data = data["experience"]
            if "summary" in exp_data:
                exp_data["summary"] = cls._neutralize_age_language(exp_data["summary"])
        
        return data
    
    @classmethod
    def _mitigate_education_bias(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply education bias mitigation strategies."""
        # Adjust education scoring to focus on relevance rather than prestige
        if "scoring_details" in data:
            scoring = data["scoring_details"]
            if "component_scores" in scoring:
                # Reduce education weight if it's over-weighted
                education_score = scoring["component_scores"].get("education", 0)
                if education_score > 80:
                    # Cap education score to prevent over-weighting
                    scoring["component_scores"]["education"] = min(education_score, 80)
        
        return data
    
    @classmethod
    def _mitigate_scoring_bias(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scoring bias mitigation strategies."""
        # Normalize extreme score variations
        if "scoring_details" in data:
            scoring = data["scoring_details"]
            if "component_scores" in scoring:
                scores = scoring["component_scores"]
                
                # Apply smoothing to reduce extreme variations
                smoothed_scores = cls._smooth_score_variations(scores)
                scoring["component_scores"] = smoothed_scores
        
        return data
    
    @classmethod
    def _neutralize_gender_language(cls, text: str) -> str:
        """Replace gender-specific language with neutral alternatives."""
        replacements = {
            'aggressive': 'assertive',
            'bossy': 'direct',
            'emotional': 'passionate',
            'hysterical': 'enthusiastic',
            'man up': 'step up',
            'girly': 'feminine',
            'manly': 'masculine'
        }
        
        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result
    
    @classmethod
    def _neutralize_age_language(cls, text: str) -> str:
        """Replace age-specific language with neutral alternatives."""
        replacements = {
            'young and energetic': 'energetic',
            'old and experienced': 'experienced',
            'fresh out of college': 'recent graduate',
            'veteran': 'experienced professional',
            'digital native': 'tech-savvy',
            'overqualified': 'highly qualified',
            'underqualified': 'developing qualifications'
        }
        
        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result
    
    @classmethod
    def _smooth_score_variations(cls, scores: Dict[str, int]) -> Dict[str, int]:
        """Smooth extreme score variations to reduce bias."""
        score_values = list(scores.values())
        if not score_values:
            return scores
        
        mean_score = sum(score_values) / len(score_values)
        
        # Apply smoothing factor to reduce extreme variations
        smoothing_factor = 0.3
        smoothed_scores = {}
        
        for component, score in scores.items():
            # Move score closer to mean if it's too far away
            if abs(score - mean_score) > 30:
                smoothed_score = int(score * (1 - smoothing_factor) + mean_score * smoothing_factor)
                smoothed_scores[component] = max(0, min(100, smoothed_score))
            else:
                smoothed_scores[component] = score
        
        return smoothed_scores


def detect_and_mitigate_biases(analysis_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[BiasIndicator]]:
    """
    Detect and mitigate biases in analysis data.
    
    Args:
        analysis_data: Analysis data to check for biases
        
    Returns:
        Tuple of (mitigated_data, detected_biases)
    """
    # Detect biases
    biases = BiasDetector.detect_biases(analysis_data)
    
    # Apply mitigation strategies
    mitigated_data = BiasMitigator.apply_mitigation_strategies(analysis_data, biases)
    
    # Log detected biases
    if biases:
        logger.warning(f"Detected {len(biases)} potential biases in analysis")
        for bias in biases:
            logger.warning(f"Bias detected: {bias.bias_type} - {bias.description}")
    
    return mitigated_data, biases


def generate_bias_report(biases: List[BiasIndicator]) -> Dict[str, Any]:
    """
    Generate a bias detection report.
    
    Args:
        biases: List of detected biases
        
    Returns:
        Dictionary containing bias report
    """
    if not biases:
        return {
            "bias_detected": False,
            "total_biases": 0,
            "summary": "No biases detected in the analysis"
        }
    
    # Categorize biases by type
    bias_by_type = {}
    for bias in biases:
        if bias.bias_type not in bias_by_type:
            bias_by_type[bias.bias_type] = []
        bias_by_type[bias.bias_type].append(bias)
    
    # Calculate severity distribution
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    for bias in biases:
        severity_counts[bias.severity] += 1
    
    return {
        "bias_detected": True,
        "total_biases": len(biases),
        "bias_by_type": {bias_type: len(bias_list) for bias_type, bias_list in bias_by_type.items()},
        "severity_distribution": severity_counts,
        "high_severity_biases": [bias for bias in biases if bias.severity == "high"],
        "mitigation_suggestions": [bias.mitigation_suggestion for bias in biases],
        "summary": f"Detected {len(biases)} potential biases requiring attention"
    }

"""
Main configuration class for Nova HR Assistant
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from .llm_config import LLMConfig


@dataclass
class NovaConfig:
    """Main configuration for Nova HR Assistant"""
    llm: LLMConfig
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "experience": 0.25,  # Reduced from 0.3 - less harsh on junior candidates
        "skills": 0.3,       # Increased from 0.25 - more emphasis on skills
        "education": 0.2,    # Same
        "certifications": 0.15,  # Same
        "overall_fit": 0.1   # Same
    })
    output_format: Dict[str, Any] = field(default_factory=lambda: {
        "include_reasoning": True,
        "include_confidence_scores": True
    })
    debug_mode: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate scoring weights sum to 1.0
        total_weight = sum(self.scoring_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Validate all weights are positive
        for weight_name, weight_value in self.scoring_weights.items():
            if weight_value < 0:
                raise ValueError(f"Scoring weight '{weight_name}' must be non-negative")
        

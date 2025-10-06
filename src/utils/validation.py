"""
Validation utilities for Nova HR Assistant
"""

from typing import Any, Dict, List
import re


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class ConfigValidator:
    """Utility class for validating configuration values"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> None:
        """Validate that weights are positive and sum to 1.0"""
        if not weights:
            raise ValidationError("Weights dictionary cannot be empty")
        
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValidationError(f"Weights must sum to 1.0, got {total}")
        
        for name, weight in weights.items():
            if weight < 0:
                raise ValidationError(f"Weight '{name}' must be non-negative, got {weight}")
    
    @staticmethod
    def validate_positive_int(value: int, name: str) -> None:
        """Validate that a value is a positive integer"""
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{name} must be a positive integer, got {value}")
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
        """Validate that a value is within a specified range"""
        if not (min_val <= value <= max_val):
            raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")


# Re-export output validation utilities for convenience
from .output_validation import (
    OutputValidationError,
    ReportFormatter,
    AnalysisCompleteness,
    ErrorReportGenerator,
    ReportConsistencyValidator,
    validate_complete_report,
    format_report_with_validation,
    create_fallback_report
)
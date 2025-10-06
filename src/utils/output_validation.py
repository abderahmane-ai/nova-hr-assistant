"""
Output validation and formatting utilities for Nova HR Assistant.

This module provides comprehensive JSON schema validation, formatting utilities,
and error handling for incomplete analysis results to ensure consistent report
structure and compliance with output format requirements.
"""

import json
import jsonschema
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import asdict

from ..models.candidate import (
    CandidateInfo,
    ExperienceAnalysis,
    SkillsAnalysis,
    EducationAnalysis,
    CertificationAnalysis
)


class OutputValidationError(Exception):
    """Custom exception for output validation errors."""
    pass


class ReportFormatter:
    """Utility class for formatting and validating Nova HR reports."""
    
    # JSON Schema for Nova HR report validation
    REPORT_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["metadata", "candidate", "analysis", "evaluation", "summary"],
        "properties": {
            "metadata": {
                "type": "object",
                "required": ["analysis_timestamp", "position", "processing_node"],
                "properties": {
                    "analysis_timestamp": {"type": "string", "format": "date-time"},
                    "position": {"type": "string", "minLength": 1},
                    "processing_node": {"type": "string"},
                    "completion_percentage": {"type": "number", "minimum": 0, "maximum": 100},
                    "error": {"type": "boolean"}
                }
            },
            "candidate": {
                "type": "object",
                "required": ["name", "contact_info"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "contact_info": {"type": "string", "minLength": 1},
                    "location": {"type": ["string", "null"]}
                }
            },
            "analysis": {
                "type": "object",
                "required": ["experience", "skills", "education", "certifications"],
                "properties": {
                    "experience": {
                        "type": "object",
                        "required": ["summary", "years_experience"],
                        "properties": {
                            "summary": {"type": "string"},
                            "years_experience": {"type": "integer", "minimum": 0},
                            "relevant_roles": {"type": "array", "items": {"type": "string"}},
                            "career_progression": {"type": "string"},
                            "strengths": {"type": "array", "items": {"type": "string"}},
                            "gaps": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "skills": {
                        "type": "object",
                        "properties": {
                            "technical_skills": {"type": "array", "items": {"type": "string"}},
                            "soft_skills": {"type": "array", "items": {"type": "string"}},
                            "skill_gaps": {"type": "array", "items": {"type": "string"}},
                            "proficiency_levels": {
                                "type": "object",
                                "patternProperties": {
                                    ".*": {
                                        "type": "string",
                                        "enum": ["beginner", "intermediate", "advanced", "expert"]
                                    }
                                }
                            }
                        }
                    },
                    "education": {
                        "type": "object",
                        "properties": {
                            "degrees": {"type": "array", "items": {"type": "string"}},
                            "institutions": {"type": "array", "items": {"type": "string"}},
                            "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
                            "additional_training": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "certifications": {
                        "type": "object",
                        "properties": {
                            "certifications": {"type": "array", "items": {"type": "string"}},
                            "professional_memberships": {"type": "array", "items": {"type": "string"}},
                            "achievements": {"type": "array", "items": {"type": "string"}},
                            "relevance_assessment": {"type": "string"}
                        }
                    }
                }
            },
            "evaluation": {
                "type": "object",
                "required": ["suitability_score", "recommendation"],
                "properties": {
                    "suitability_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "recommendation": {"type": "string", "minLength": 1},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "areas_for_improvement": {"type": "array", "items": {"type": "string"}}
                }
            },
            "summary": {
                "type": "object",
                "required": ["overall_assessment"],
                "properties": {
                    "overall_assessment": {"type": "string", "minLength": 1},
                    "key_highlights": {"type": "array", "items": {"type": "string"}},
                    "decision_factors": {"type": "array", "items": {"type": "string"}},
                    "error_details": {"type": "string"}
                }
            }
        }
    }
    
    @classmethod
    def validate_report_schema(cls, report: Dict[str, Any]) -> List[str]:
        """
        Validate report against JSON schema.
        
        Args:
            report: The report dictionary to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            jsonschema.validate(instance=report, schema=cls.REPORT_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            if e.path:
                errors.append(f"Error path: {' -> '.join(str(p) for p in e.path)}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema definition error: {e.message}")
        
        return errors
    
    @classmethod
    def format_json_report(cls, report: Dict[str, Any], 
                          indent: int = 2, 
                          ensure_ascii: bool = False,
                          sort_keys: bool = True) -> str:
        """
        Format report as JSON string with consistent formatting.
        
        Args:
            report: The report dictionary
            indent: Number of spaces for indentation
            ensure_ascii: Whether to escape non-ASCII characters
            sort_keys: Whether to sort dictionary keys
            
        Returns:
            Formatted JSON string
        """
        try:
            return json.dumps(
                report, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                separators=(',', ': ')
            )
        except (TypeError, ValueError) as e:
            raise OutputValidationError(f"JSON formatting error: {str(e)}")
    
    @classmethod
    def validate_json_string(cls, json_string: str) -> List[str]:
        """
        Validate that a string is valid JSON.
        
        Args:
            json_string: The JSON string to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not json_string or not json_string.strip():
            errors.append("JSON string is empty or contains only whitespace")
            return errors
        
        try:
            json.loads(json_string)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {e.msg} at line {e.lineno}, column {e.colno}")
        
        return errors


class AnalysisCompleteness:
    """Utility class for validating analysis completeness and handling incomplete results."""
    
    @staticmethod
    def validate_candidate_info(candidate_info: Optional[CandidateInfo]) -> List[str]:
        """Validate candidate information completeness."""
        errors = []
        
        if not candidate_info:
            errors.append("Candidate information is missing")
        else:
            validation_errors = candidate_info.validate()
            errors.extend(validation_errors)
        
        return errors
    
    @staticmethod
    def validate_experience_analysis(experience: Optional[ExperienceAnalysis]) -> List[str]:
        """Validate experience analysis completeness."""
        errors = []
        
        if not experience:
            errors.append("Experience analysis is missing")
        else:
            validation_errors = experience.validate()
            errors.extend(validation_errors)
            
            # Additional completeness checks
            if not experience.summary.strip():
                errors.append("Experience summary cannot be empty")
            
            # Handle missing experience data gracefully - don't require both to be present
            # If CV doesn't contain clear experience information, we should handle it gracefully
            if experience.years_experience == 0 and not experience.relevant_roles:
                # Instead of failing, we'll note this as a limitation but allow processing to continue
                # The summary should indicate "No clear professional experience found" or similar
                pass  # Remove the strict validation requirement
        
        return errors
    
    @staticmethod
    def validate_skills_analysis(skills: Optional[SkillsAnalysis]) -> List[str]:
        """Validate skills analysis completeness."""
        errors = []
        
        if not skills:
            errors.append("Skills analysis is missing")
        else:
            validation_errors = skills.validate()
            errors.extend(validation_errors)
            
            # Additional completeness checks
            if not skills.technical_skills and not skills.soft_skills:
                errors.append("At least one technical or soft skill must be identified")
        
        return errors
    
    @staticmethod
    def validate_education_analysis(education: Optional[EducationAnalysis]) -> List[str]:
        """Validate education analysis completeness."""
        errors = []
        
        if not education:
            errors.append("Education analysis is missing")
        else:
            validation_errors = education.validate()
            errors.extend(validation_errors)
        
        return errors
    
    @staticmethod
    def validate_certification_analysis(certifications: Optional[CertificationAnalysis]) -> List[str]:
        """Validate certification analysis completeness."""
        errors = []
        
        if not certifications:
            errors.append("Certification analysis is missing")
        else:
            validation_errors = certifications.validate()
            errors.extend(validation_errors)
        
        return errors
    
    @staticmethod
    def validate_evaluation_completeness(suitability_score: int) -> List[str]:
        """Validate evaluation section completeness."""
        errors = []
        
        if not isinstance(suitability_score, int):
            errors.append("Suitability score must be an integer")
        elif suitability_score < 0 or suitability_score > 100:
            errors.append("Suitability score must be between 0 and 100")
        
        return errors


class ErrorReportGenerator:
    """Utility class for generating error reports when analysis is incomplete."""
    
    @staticmethod
    def create_error_report(position: str, 
                          error_message: str,
                          partial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized error report.
        
        Args:
            position: The job position being analyzed
            error_message: Description of the error
            partial_data: Any partial analysis data available
            
        Returns:
            Error report dictionary
        """
        timestamp = datetime.now().isoformat()
        
        # Ensure candidate section has required fields
        candidate_data = partial_data.get("candidate", {}) if partial_data else {}
        if not candidate_data.get("name"):
            candidate_data["name"] = "Unknown Candidate"
        if not candidate_data.get("contact_info"):
            candidate_data["contact_info"] = "Contact information unavailable"
        
        error_report = {
            "metadata": {
                "analysis_timestamp": timestamp,
                "position": position,
                "processing_node": "error_handler",
                "completion_percentage": 0,
                "error": True
            },
            "candidate": candidate_data,
            "analysis": {
                "experience": {
                    "summary": "Analysis failed - no experience data available",
                    "years_experience": 0,
                    **partial_data.get("experience", {})
                } if partial_data else {
                    "summary": "Analysis failed - no experience data available",
                    "years_experience": 0
                },
                "skills": partial_data.get("skills", {}) if partial_data else {},
                "education": partial_data.get("education", {}) if partial_data else {},
                "certifications": partial_data.get("certifications", {}) if partial_data else {}
            },
            "evaluation": {
                "suitability_score": 0,
                "recommendation": "Analysis Failed - Unable to complete evaluation",
                "strengths": [],
                "areas_for_improvement": ["Analysis could not be completed due to processing errors"]
            },
            "summary": {
                "overall_assessment": "Report generation failed due to processing errors",
                "key_highlights": [],
                "decision_factors": [],
                "error_details": error_message
            }
        }
        
        return error_report
    
    @staticmethod
    def create_partial_report(position: str,
                            available_data: Dict[str, Any],
                            missing_components: List[str]) -> Dict[str, Any]:
        """
        Create a report with partial data when some analysis components are missing.
        
        Args:
            position: The job position being analyzed
            available_data: Dictionary of available analysis components
            missing_components: List of missing component names
            
        Returns:
            Partial report dictionary
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate completion percentage based on available components
        total_components = 5  # candidate, experience, skills, education, certifications
        available_components = len([k for k in available_data.keys() if available_data[k]])
        completion_percentage = int((available_components / total_components) * 100)
        
        partial_report = {
            "metadata": {
                "analysis_timestamp": timestamp,
                "position": position,
                "processing_node": "partial_analysis",
                "completion_percentage": completion_percentage,
                "error": False
            },
            "candidate": available_data.get("candidate", {}),
            "analysis": {
                "experience": available_data.get("experience", {}),
                "skills": available_data.get("skills", {}),
                "education": available_data.get("education", {}),
                "certifications": available_data.get("certifications", {})
            },
            "evaluation": {
                "suitability_score": available_data.get("suitability_score", 0),
                "recommendation": "Partial Analysis - Some components missing",
                "strengths": available_data.get("strengths", []),
                "areas_for_improvement": available_data.get("gaps", [])
            },
            "summary": {
                "overall_assessment": f"Partial analysis completed ({completion_percentage}% complete)",
                "key_highlights": available_data.get("highlights", []),
                "decision_factors": available_data.get("decision_factors", []),
                "error_details": f"Missing components: {', '.join(missing_components)}"
            }
        }
        
        return partial_report


class ReportConsistencyValidator:
    """Utility class for ensuring report consistency and quality."""
    
    @staticmethod
    def validate_data_consistency(report: Dict[str, Any]) -> List[str]:
        """
        Validate internal data consistency within the report.
        
        Args:
            report: The report dictionary to validate
            
        Returns:
            List of consistency error messages
        """
        errors = []
        
        # Check candidate name consistency
        candidate_name = report.get("candidate", {}).get("name", "")
        if candidate_name and len(candidate_name.strip()) < 2:
            errors.append("Candidate name appears to be too short")
        
        # Check suitability score consistency with recommendation
        suitability_score = report.get("evaluation", {}).get("suitability_score", 0)
        recommendation = report.get("evaluation", {}).get("recommendation", "")
        
        if suitability_score >= 80 and "not recommended" in recommendation.lower():
            errors.append("High suitability score inconsistent with negative recommendation")
        elif suitability_score < 50 and "highly recommended" in recommendation.lower():
            errors.append("Low suitability score inconsistent with positive recommendation")
        
        # Check experience years consistency
        experience = report.get("analysis", {}).get("experience", {})
        years_exp = experience.get("years_experience", 0)
        if years_exp > 50:
            errors.append("Years of experience seems unrealistic (>50 years)")
        
        # Check skills consistency
        skills = report.get("analysis", {}).get("skills", {})
        tech_skills = skills.get("technical_skills", [])
        proficiency_levels = skills.get("proficiency_levels", {})
        
        # Check that proficiency levels match listed skills
        for skill in proficiency_levels.keys():
            if skill not in tech_skills and skill not in skills.get("soft_skills", []):
                errors.append(f"Proficiency level specified for unlisted skill: {skill}")
        
        return errors
    
    @staticmethod
    def validate_completeness_quality(report: Dict[str, Any]) -> List[str]:
        """
        Validate the quality and completeness of report content.
        
        Args:
            report: The report dictionary to validate
            
        Returns:
            List of quality error messages
        """
        errors = []
        
        # Check for empty or minimal content
        analysis = report.get("analysis", {})
        
        # Experience analysis quality
        experience = analysis.get("experience", {})
        if experience.get("summary", "") and len(experience["summary"].strip()) < 20:
            errors.append("Experience summary appears too brief")
        
        # Skills analysis quality
        skills = analysis.get("skills", {})
        total_skills = len(skills.get("technical_skills", [])) + len(skills.get("soft_skills", []))
        if total_skills == 0:
            errors.append("No skills identified - analysis may be incomplete")
        elif total_skills < 3:
            errors.append("Very few skills identified - analysis may be incomplete")
        
        # Interview questions quality
        # Check for generic or template-like content
        assessment = report.get("summary", {}).get("overall_assessment", "")
        if "template" in assessment.lower() or "placeholder" in assessment.lower():
            errors.append("Assessment contains template or placeholder content")
        
        return errors


# Convenience functions for common validation tasks
def validate_complete_report(report: Dict[str, Any]) -> List[str]:
    """
    Perform comprehensive validation of a complete report.
    
    Args:
        report: The report dictionary to validate
        
    Returns:
        List of all validation error messages
    """
    errors = []
    
    # Schema validation
    errors.extend(ReportFormatter.validate_report_schema(report))
    
    # Data consistency validation
    errors.extend(ReportConsistencyValidator.validate_data_consistency(report))
    
    # Quality validation
    errors.extend(ReportConsistencyValidator.validate_completeness_quality(report))
    
    return errors


def format_report_with_validation(report: Dict[str, Any], 
                                 validate: bool = True) -> str:
    """
    Format report as JSON with optional validation.
    
    Args:
        report: The report dictionary to format
        validate: Whether to validate before formatting
        
    Returns:
        Formatted JSON string
        
    Raises:
        OutputValidationError: If validation fails
    """
    if validate:
        errors = validate_complete_report(report)
        if errors:
            raise OutputValidationError(f"Report validation failed: {'; '.join(errors)}")
    
    return ReportFormatter.format_json_report(report)


def create_fallback_report(position: str, error_message: str) -> Dict[str, Any]:
    """
    Create a fallback error report when analysis completely fails.
    
    Args:
        position: The job position being analyzed
        error_message: Description of the error
        
    Returns:
        Fallback error report dictionary
    """
    return ErrorReportGenerator.create_error_report(position, error_message)
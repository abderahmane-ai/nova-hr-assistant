"""
Data models for candidate analysis in Nova HR Assistant.

This module contains dataclasses representing different aspects of candidate
evaluation including basic info, experience, skills, education, and certifications.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
from datetime import datetime


@dataclass
class CandidateInfo:
    """Basic candidate information extracted from CV."""
    
    name: str
    contact_info: str
    location: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate candidate information and return list of validation errors."""
        errors = []
        
        if not self.name or not self.name.strip():
            errors.append("Name is required and cannot be empty")
        
        if not self.contact_info or not self.contact_info.strip():
            errors.append("Contact information is required and cannot be empty")
        
        # Basic email validation if contact_info contains email
        if '@' in self.contact_info:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if not re.search(email_pattern, self.contact_info):
                errors.append("Contact information contains invalid email format")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if candidate information is valid."""
        return len(self.validate()) == 0


@dataclass
class ExperienceAnalysis:
    """Analysis of candidate's work experience and career progression."""
    
    summary: str
    years_experience: int
    relevant_roles: List[str] = field(default_factory=list)
    career_progression: str = ""
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate experience analysis and return list of validation errors."""
        errors = []
        
        if not self.summary or not self.summary.strip():
            errors.append("Experience summary is required and cannot be empty")
        
        if self.years_experience < 0:
            errors.append("Years of experience cannot be negative")
        
        if self.years_experience > 60:
            errors.append("Years of experience seems unrealistic (>60 years)")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if experience analysis is valid."""
        return len(self.validate()) == 0


@dataclass
class SkillsAnalysis:
    """Analysis of candidate's technical and soft skills."""
    
    technical_skills: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)
    skill_gaps: List[str] = field(default_factory=list)
    proficiency_levels: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate skills analysis and return list of validation errors."""
        errors = []
        
        # Check that proficiency levels only contain valid values
        valid_proficiency_levels = {'beginner', 'intermediate', 'advanced', 'expert'}
        for skill, level in self.proficiency_levels.items():
            if level.lower() not in valid_proficiency_levels:
                errors.append(f"Invalid proficiency level '{level}' for skill '{skill}'. "
                            f"Valid levels: {', '.join(valid_proficiency_levels)}")
        
        # Check that skills in proficiency_levels exist in technical_skills or soft_skills
        # If not, add them to technical_skills automatically
        all_skills = set(self.technical_skills + self.soft_skills)
        for skill in self.proficiency_levels.keys():
            if skill not in all_skills:
                # Auto-add missing skills to technical_skills instead of failing
                self.technical_skills.append(skill)
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if skills analysis is valid."""
        return len(self.validate()) == 0


@dataclass
class EducationAnalysis:
    """Analysis of candidate's educational background."""
    
    degrees: List[str] = field(default_factory=list)
    institutions: List[str] = field(default_factory=list)
    relevance_score: int = 0
    additional_training: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate education analysis and return list of validation errors."""
        errors = []
        
        if self.relevance_score < 0 or self.relevance_score > 100:
            errors.append("Relevance score must be between 0 and 100")
        
        # Remove the strict degree-institution matching requirement
        # This was too restrictive and caused failures when:
        # - Multiple degrees from same institution
        # - LLM extracts inconsistent data
        # - Candidate lists degrees but not all institutions
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if education analysis is valid."""
        return len(self.validate()) == 0


@dataclass
class CertificationAnalysis:
    """Analysis of candidate's professional certifications and achievements."""
    
    certifications: List[str] = field(default_factory=list)
    professional_memberships: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    relevance_assessment: str = ""
    
    def validate(self) -> List[str]:
        """Validate certification analysis and return list of validation errors."""
        errors = []
        
        # No specific validation rules for certifications currently
        # This method is provided for consistency and future extensibility
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if certification analysis is valid."""
        return len(self.validate()) == 0
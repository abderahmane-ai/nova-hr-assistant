"""
State management for Nova HR Assistant LangGraph workflow.

This module contains the CVAnalysisState dataclass that represents the state
passed between nodes in the LangGraph workflow.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .candidate import (
    CandidateInfo, 
    ExperienceAnalysis, 
    SkillsAnalysis, 
    EducationAnalysis, 
    CertificationAnalysis
)


@dataclass
class CVAnalysisState:
    """
    State object passed between graph nodes in the CV analysis workflow.
    
    This class maintains all the data and analysis results as the workflow
    progresses through different analysis nodes.
    """
    
    # Input data
    cv_text: str = ""
    position: str = ""
    
    # Analysis results from individual nodes
    candidate_info: Optional[CandidateInfo] = None
    experience_analysis: Optional[ExperienceAnalysis] = None
    skills_analysis: Optional[SkillsAnalysis] = None
    education_analysis: Optional[EducationAnalysis] = None
    certifications_analysis: Optional[CertificationAnalysis] = None
    
    # Scoring and recommendations
    suitability_score: int = 0
    
    # Final output
    final_report: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow metadata
    current_node: str = ""
    errors: List[str] = field(default_factory=list)
    processing_start_time: Optional[str] = None
    
    def validate_input(self) -> List[str]:
        """Validate input data and return list of validation errors."""
        errors = []
        
        if not self.cv_text or not self.cv_text.strip():
            errors.append("CV text is required and cannot be empty")
        
        if not self.position or not self.position.strip():
            errors.append("Position description is required and cannot be empty")
        
        if len(self.cv_text.strip()) < 50:
            errors.append("CV text seems too short (less than 50 characters)")
        
        return errors
    
    def validate_analysis_results(self) -> List[str]:
        """Validate all analysis results and return list of validation errors."""
        errors = []
        
        # Validate individual analysis components if they exist
        if self.candidate_info:
            errors.extend([f"CandidateInfo: {error}" for error in self.candidate_info.validate()])
        
        if self.experience_analysis:
            errors.extend([f"ExperienceAnalysis: {error}" for error in self.experience_analysis.validate()])
        
        if self.skills_analysis:
            errors.extend([f"SkillsAnalysis: {error}" for error in self.skills_analysis.validate()])
        
        if self.education_analysis:
            errors.extend([f"EducationAnalysis: {error}" for error in self.education_analysis.validate()])
        
        if self.certifications_analysis:
            errors.extend([f"CertificationAnalysis: {error}" for error in self.certifications_analysis.validate()])
        
        # Validate suitability score
        if self.suitability_score < 0 or self.suitability_score > 100:
            errors.append("Suitability score must be between 0 and 100")
        
        return errors
    
    def is_input_valid(self) -> bool:
        """Check if input data is valid."""
        return len(self.validate_input()) == 0
    
    def is_analysis_valid(self) -> bool:
        """Check if all analysis results are valid."""
        return len(self.validate_analysis_results()) == 0
    
    def has_required_analysis(self) -> bool:
        """Check if all required analysis components are present."""
        return all([
            self.candidate_info is not None,
            self.experience_analysis is not None,
            self.skills_analysis is not None,
            self.education_analysis is not None,
            self.certifications_analysis is not None
        ])
    
    def get_completion_percentage(self) -> float:
        """Get the completion percentage of the analysis workflow."""
        completed_components = 0
        total_components = 6  # candidate_info, experience, skills, education, certs, score
        
        if self.candidate_info is not None:
            completed_components += 1
        if self.experience_analysis is not None:
            completed_components += 1
        if self.skills_analysis is not None:
            completed_components += 1
        if self.education_analysis is not None:
            completed_components += 1
        if self.certifications_analysis is not None:
            completed_components += 1
        if self.suitability_score > 0:
            completed_components += 1
        
        return (completed_components / total_components) * 100
    
    def add_error(self, error: str) -> None:
        """Add an error to the errors list."""
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if there are any errors in the state."""
        return len(self.errors) > 0
    
    def clear_errors(self) -> None:
        """Clear all errors from the state."""
        self.errors.clear()
    
    def set_current_node(self, node_name: str) -> None:
        """Set the current processing node."""
        self.current_node = node_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'cv_text': self.cv_text,
            'position': self.position,
            'candidate_info': self.candidate_info.__dict__ if self.candidate_info else None,
            'experience_analysis': self.experience_analysis.__dict__ if self.experience_analysis else None,
            'skills_analysis': self.skills_analysis.__dict__ if self.skills_analysis else None,
            'education_analysis': self.education_analysis.__dict__ if self.education_analysis else None,
            'certifications_analysis': self.certifications_analysis.__dict__ if self.certifications_analysis else None,
            'suitability_score': self.suitability_score,
            'final_report': self.final_report,
            'current_node': self.current_node,
            'errors': self.errors,
            'processing_start_time': self.processing_start_time,
            'completion_percentage': self.get_completion_percentage()
        }


    
    def validate_input(self) -> List[str]:
        """
        Validate input data in the state.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        if not self.cv_text or not self.cv_text.strip():
            errors.append("CV text is empty or missing")
        elif len(self.cv_text.strip()) < 50:
            errors.append("CV text is too short (minimum 50 characters)")
        
        if not self.position or not self.position.strip():
            errors.append("Position description is empty or missing")
        elif len(self.position.strip()) < 20:
            errors.append("Position description is too short (minimum 20 characters)")
        
        return errors

    def has_required_analysis(self) -> bool:
        """
        Check if state has all required analysis components.
        
        Returns:
            True if all required components are present
        """
        return (self.candidate_info is not None and
                self.experience_analysis is not None and
                self.skills_analysis is not None and
                self.education_analysis is not None and
                self.certifications_analysis is not None)

    def get_completion_percentage(self) -> float:
        """
        Calculate completion percentage of the analysis.
        
        Returns:
            Completion percentage (0-100)
        """
        total_components = 7  # All analysis components + score + report
        completed = 0
        
        if self.candidate_info:
            completed += 1
        if self.experience_analysis:
            completed += 1
        if self.skills_analysis:
            completed += 1
        if self.education_analysis:
            completed += 1
        if self.certifications_analysis:
            completed += 1
        if self.suitability_score > 0:
            completed += 1
        if self.final_report:
            completed += 1
        
        return (completed / total_components) * 100

    def has_errors(self) -> bool:
        """
        Check if state has any errors.
        
        Returns:
            True if there are errors
        """
        return bool(self.errors)

    def add_error(self, error: str) -> None:
        """
        Add an error to the state.
        
        Args:
            error: Error message to add
        """
        self.errors.append(error)

    def set_current_node(self, node_name: str) -> None:
        """
        Set the current node in the state.
        
        Args:
            node_name: Name of the current node
        """
        self.current_node = node_name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        from dataclasses import asdict
        return asdict(self)
"""
Nova HR Assistant data models.

This package contains all data models used in the Nova HR Assistant system,
including candidate analysis models and workflow state management.
"""

from .candidate import (
    CandidateInfo,
    ExperienceAnalysis,
    SkillsAnalysis,
    EducationAnalysis,
    CertificationAnalysis
)

from .state import CVAnalysisState

__all__ = [
    'CandidateInfo',
    'ExperienceAnalysis', 
    'SkillsAnalysis',
    'EducationAnalysis',
    'CertificationAnalysis',
    'CVAnalysisState'
]
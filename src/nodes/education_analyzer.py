"""
Education Analyzer Node for Nova HR Assistant.

This module contains the education analyzer node that evaluates academic background
and relevance using structured LLM prompts.
"""

import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.state import CVAnalysisState
from ..models.candidate import EducationAnalysis
from ..utils.llm_manager import LLMManager, LLMProviderError


logger = logging.getLogger(__name__)


class EducationAnalysisOutput(BaseModel):
    """Pydantic model for structured education analyzer output."""
    
    degrees: List[str] = Field(description="List of academic degrees obtained")
    institutions: List[str] = Field(description="List of educational institutions attended")
    relevance_score: int = Field(description="Relevance score of education to position (0-100)")
    additional_training: List[str] = Field(description="List of additional training, courses, certifications, or professional development", default_factory=list)


class EducationAnalyzerError(Exception):
    """Exception raised when education analysis fails."""
    pass


def create_education_analyzer_prompt() -> ChatPromptTemplate:
    """
    Create the structured prompt for education analysis.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for education analysis
    """
    
    system_message = """You are an expert HR analyst specialized in evaluating candidate educational backgrounds and academic qualifications.
Your task is to analyze the provided CV and position requirements to assess the candidate's educational qualifications.

ANALYSIS GUIDELINES:
1. Extract all academic degrees, diplomas, and formal qualifications
2. Identify educational institutions attended (universities, colleges, schools)
3. Assess relevance of educational background to the target position (score 0-100)
4. Identify additional training, courses, online certifications, or professional development

EVALUATION CRITERIA:
- Relevance of degree field to the position requirements
- Prestige and reputation of educational institutions
- Level of education (Bachelor's, Master's, PhD, etc.)
- Recency of education and continuous learning
- Specialized training relevant to the role
- Academic achievements and honors if mentioned

SCORING GUIDELINES FOR RELEVANCE (0-100):
- 90-100: Perfect match - degree directly related to position, top-tier institution
- 80-89: Excellent match - closely related field, reputable institution
- 70-79: Good match - related field or strong institution, some relevance
- 60-69: Moderate match - somewhat related or decent institution
- 50-59: Basic match - general education meets minimum requirements
- 40-49: Weak match - education somewhat relevant but gaps exist
- 30-39: Poor match - education minimally relevant to position
- 20-29: Very poor match - education not well suited for position
- 10-19: Minimal match - basic education only, major gaps
- 0-9: No match - education not relevant or insufficient

IMPORTANT INSTRUCTIONS:
- Be objective and evidence-based in your assessment
- Consider both formal degrees and additional training/certifications
- Assess relevance specifically to the target position requirements
- Include online courses, bootcamps, and professional development if mentioned
- Note any academic achievements, honors, or distinctions
- Consider the progression and consistency of educational background
- Extract degrees and institutions separately - they don't need to be perfectly paired
- If a candidate has multiple degrees from the same institution, list the institution once
- Focus on extracting all available educational information accurately

{format_instructions}"""

    human_message = """Please analyze the candidate's educational background for the following position:

POSITION: {position}

CV TEXT:
{cv_text}

Provide a comprehensive analysis of the candidate's educational qualifications according to the specified format."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def education_analyzer_node(state: CVAnalysisState, llm_manager: LLMManager) -> Dict[str, Any]:
    """
    Education Analyzer node function for LangGraph workflow.
    
    Evaluates academic background and relevance against position requirements.
    
    Args:
        state: Current workflow state containing CV text and position
        llm_manager: LLM manager instance for making LLM calls
        
    Returns:
        Dict containing updated state with education_analysis
        
    Raises:
        EducationAnalyzerError: If education analysis fails
    """
    
    logger.info("Starting education analyzer node")
    
    # Update current node in state
    updated_state = {
        "current_node": "education_analyzer",
        "errors": state.errors.copy()
    }
    
    try:
        # Validate input
        if not state.cv_text or not state.cv_text.strip():
            error_msg = "CV text is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise EducationAnalyzerError(error_msg)
        
        if not state.position or not state.position.strip():
            error_msg = "Position description is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise EducationAnalyzerError(error_msg)
        
        # Get LLM instance
        llm = llm_manager.get_llm()
        
        # Create output parser
        output_parser = PydanticOutputParser(pydantic_object=EducationAnalysisOutput)
        
        # Create prompt template
        prompt_template = create_education_analyzer_prompt()
        
        # Create the chain
        chain = prompt_template | llm | output_parser
        
        # Execute the chain
        logger.debug(f"Analyzing education for position: {state.position}")
        
        result = chain.invoke({
            "cv_text": state.cv_text,
            "position": state.position,
            "format_instructions": output_parser.get_format_instructions()
        })
        
        # Convert to EducationAnalysis model
        education_analysis = EducationAnalysis(
            degrees=[degree.strip() for degree in result.degrees],
            institutions=[institution.strip() for institution in result.institutions],
            relevance_score=result.relevance_score,
            additional_training=[training.strip() for training in result.additional_training]
        )
        
        # Validate the analysis
        validation_errors = education_analysis.validate()
        if validation_errors:
            error_msg = f"Education analysis validation failed: {', '.join(validation_errors)}"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise EducationAnalyzerError(error_msg)
        
        # Update state with successful result
        updated_state["education_analysis"] = education_analysis
        
        logger.info(f"Successfully analyzed education: {len(education_analysis.degrees)} degrees, "
                   f"relevance score: {education_analysis.relevance_score}")
        logger.debug(f"Degrees: {', '.join(education_analysis.degrees)}")
        logger.debug(f"Institutions: {', '.join(education_analysis.institutions)}")
        
        return updated_state
        
    except LLMProviderError as e:
        error_msg = f"LLM provider error in education analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise EducationAnalyzerError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error in education analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise EducationAnalyzerError(error_msg)


def validate_education_analysis_output(education_analysis: EducationAnalysis) -> bool:
    """
    Validate education analysis output for completeness and accuracy.
    
    Args:
        education_analysis: Analyzed education information
        
    Returns:
        bool: True if output is valid, False otherwise
    """
    
    if not education_analysis:
        return False
    
    # Check basic validation
    if not education_analysis.is_valid():
        return False
    
    # Additional checks for education analyzer specific requirements
    if education_analysis.relevance_score < 0 or education_analysis.relevance_score > 100:
        logger.warning("Education relevance score is out of valid range (0-100)")
        return False
    
    # Check that we have some educational information
    if not education_analysis.degrees and not education_analysis.additional_training:
        logger.warning("No educational information found")
        return False
    
    # Note: Removed strict degree-institution pairing check
    # This was causing validation failures when LLM extracts inconsistent data
    # Multiple degrees from same institution or incomplete institution data is common
    
    return True


def calculate_education_relevance_score(education_analysis: EducationAnalysis, 
                                      position_keywords: List[str]) -> float:
    """
    Calculate relevance score based on education analysis and position keywords.
    
    Args:
        education_analysis: Analyzed education information
        position_keywords: Keywords from the position description
        
    Returns:
        float: Relevance score between 0.0 and 1.0
    """
    
    if not education_analysis or not position_keywords:
        return 0.0
    
    # Use the LLM-generated relevance score as base
    base_score = education_analysis.relevance_score / 100.0
    
    # Combine all education text for keyword analysis
    education_text = " ".join([
        " ".join(education_analysis.degrees),
        " ".join(education_analysis.institutions),
        " ".join(education_analysis.additional_training)
    ]).lower()
    
    # Count keyword matches for additional validation
    matches = 0
    for keyword in position_keywords:
        if keyword.lower() in education_text:
            matches += 1
    
    # Calculate keyword relevance score
    keyword_score = matches / len(position_keywords) if position_keywords else 0.0
    
    # Combine scores with more weight on LLM assessment
    final_score = (base_score * 0.8) + (keyword_score * 0.2)
    return min(final_score, 1.0)


def extract_education_highlights(education_analysis: EducationAnalysis) -> List[str]:
    """
    Extract key educational highlights from education analysis.
    
    Args:
        education_analysis: Analyzed education information
        
    Returns:
        List[str]: List of educational highlights
    """
    
    highlights = []
    
    if not education_analysis:
        return highlights
    
    # Add degree highlights
    for degree in education_analysis.degrees:
        if any(keyword in degree.lower() for keyword in 
               ['master', 'phd', 'doctorate', 'mba', 'engineering', 'computer science', 'technology']):
            highlights.append(f"Advanced degree: {degree}")
    
    # Add institution highlights
    for institution in education_analysis.institutions:
        if any(keyword in institution.lower() for keyword in 
               ['university', 'institute of technology', 'polytechnic', 'college']):
            highlights.append(f"Formal education at: {institution}")
    
    # Add training highlights
    for training in education_analysis.additional_training:
        if any(keyword in training.lower() for keyword in 
               ['certification', 'course', 'training', 'bootcamp', 'workshop']):
            highlights.append(f"Additional training: {training}")
    
    # Add relevance score if high
    if education_analysis.relevance_score >= 80:
        highlights.append(f"High education relevance score: {education_analysis.relevance_score}/100")
    
    return highlights


def handle_education_analyzer_errors(state: CVAnalysisState, error: Exception) -> Dict[str, Any]:
    """
    Handle errors that occur during education analysis.
    
    Args:
        state: Current workflow state
        error: Exception that occurred
        
    Returns:
        Dict containing updated state with error information
    """
    
    error_msg = f"Education Analyzer failed: {str(error)}"
    logger.error(error_msg)
    
    return {
        "current_node": "education_analyzer",
        "education_analysis": None,
        "errors": state.errors + [error_msg]
    }
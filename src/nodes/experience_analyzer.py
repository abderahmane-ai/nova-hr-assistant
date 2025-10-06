"""
Experience Analyzer Node for Nova HR Assistant.

This module contains the experience analyzer node that analyzes work history
and career progression using structured LLM prompts.
"""

import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.state import CVAnalysisState
from ..models.candidate import ExperienceAnalysis
from ..utils.llm_manager import LLMManager, LLMProviderError


logger = logging.getLogger(__name__)


class ExperienceAnalysisOutput(BaseModel):
    """Pydantic model for structured experience analyzer output."""
    
    summary: str = Field(description="Summary of candidate's work experience")
    years_experience: int = Field(description="Total years of professional experience")
    relevant_roles: List[str] = Field(description="List of roles relevant to the position")
    career_progression: str = Field(description="Analysis of career progression and growth")
    strengths: List[str] = Field(description="Key strengths identified from experience")
    gaps: List[str] = Field(description="Experience gaps or areas for improvement")


class ExperienceAnalyzerError(Exception):
    """Exception raised when experience analysis fails."""
    pass


def create_experience_analyzer_prompt() -> ChatPromptTemplate:
    """
    Create the structured prompt for experience analysis.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for experience analysis
    """
    
    system_message = """You are an expert HR analyst specialized in evaluating candidate work experience and career progression.
Your task is to analyze the provided CV and position requirements to assess the candidate's professional experience.

ANALYSIS GUIDELINES:
1. Summarize the candidate's overall work experience in 2-3 sentences
2. Calculate total years of professional experience (exclude internships unless specified)
3. Identify roles that are directly relevant to the target position
4. Analyze career progression: promotions, skill development, responsibility growth
5. Identify key strengths demonstrated through work experience
6. Identify experience gaps or areas where the candidate may need development

EVALUATION CRITERIA:
- Relevance of previous roles to the target position
- Consistency and progression in career path
- Leadership and responsibility growth over time
- Technical skills development through experience
- Industry experience and domain knowledge
- Achievement and impact in previous roles

IMPORTANT INSTRUCTIONS:
- Be objective and evidence-based in your analysis
- Focus on experience that directly relates to the position requirements
- Consider both technical and soft skills demonstrated through experience
- Identify specific achievements and quantifiable results where mentioned
- Note any career gaps or transitions and their potential impact
- If no professional experience is found, set years_experience to 0 and provide appropriate summary
- For candidates with no experience, focus on education, projects, internships, or transferable skills
- Handle missing information gracefully - use "No professional experience found" or similar phrases

{format_instructions}"""

    human_message = """Please analyze the candidate's work experience for the following position:

POSITION: {position}

CV TEXT:
{cv_text}

Provide a comprehensive analysis of the candidate's work experience according to the specified format."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def experience_analyzer_node(state: CVAnalysisState, llm_manager: LLMManager) -> Dict[str, Any]:
    """
    Experience Analyzer node function for LangGraph workflow.
    
    Analyzes work history and career progression against position requirements.
    
    Args:
        state: Current workflow state containing CV text and position
        llm_manager: LLM manager instance for making LLM calls
        
    Returns:
        Dict containing updated state with experience_analysis
        
    Raises:
        ExperienceAnalyzerError: If experience analysis fails
    """
    
    logger.info("Starting experience analyzer node")
    
    # Update current node in state
    updated_state = {
        "current_node": "experience_analyzer",
        "errors": state.errors.copy()
    }
    
    try:
        # Validate input
        if not state.cv_text or not state.cv_text.strip():
            error_msg = "CV text is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise ExperienceAnalyzerError(error_msg)
        
        if not state.position or not state.position.strip():
            error_msg = "Position description is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise ExperienceAnalyzerError(error_msg)
        
        # Get LLM instance
        llm = llm_manager.get_llm()
        
        # Create output parser
        output_parser = PydanticOutputParser(pydantic_object=ExperienceAnalysisOutput)
        
        # Create prompt template
        prompt_template = create_experience_analyzer_prompt()
        
        # Create the chain
        chain = prompt_template | llm | output_parser
        
        # Execute the chain
        logger.debug(f"Analyzing experience for position: {state.position}")
        
        result = chain.invoke({
            "cv_text": state.cv_text,
            "position": state.position,
            "format_instructions": output_parser.get_format_instructions()
        })
        
        # Convert to ExperienceAnalysis model
        experience_analysis = ExperienceAnalysis(
            summary=result.summary.strip(),
            years_experience=result.years_experience,
            relevant_roles=[role.strip() for role in result.relevant_roles],
            career_progression=result.career_progression.strip(),
            strengths=[strength.strip() for strength in result.strengths],
            gaps=[gap.strip() for gap in result.gaps]
        )
        
        # Validate the analysis
        validation_errors = experience_analysis.validate()
        if validation_errors:
            error_msg = f"Experience analysis validation failed: {', '.join(validation_errors)}"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise ExperienceAnalyzerError(error_msg)
        
        # Update state with successful result
        updated_state["experience_analysis"] = experience_analysis
        
        logger.info(f"Successfully analyzed experience: {experience_analysis.years_experience} years, "
                   f"{len(experience_analysis.relevant_roles)} relevant roles")
        logger.debug(f"Experience summary: {experience_analysis.summary[:100]}...")
        
        return updated_state
        
    except LLMProviderError as e:
        error_msg = f"LLM provider error in experience analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise ExperienceAnalyzerError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error in experience analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise ExperienceAnalyzerError(error_msg)


def validate_experience_analysis_output(experience_analysis: ExperienceAnalysis) -> bool:
    """
    Validate experience analysis output for completeness and accuracy.
    
    Args:
        experience_analysis: Analyzed experience information
        
    Returns:
        bool: True if output is valid, False otherwise
    """
    
    if not experience_analysis:
        return False
    
    # Check basic validation
    if not experience_analysis.is_valid():
        return False
    
    # Additional checks for experience analyzer specific requirements
    if len(experience_analysis.summary.strip()) < 20:
        logger.warning("Experience summary is too short")
        return False
    
    # Check reasonable years of experience
    if experience_analysis.years_experience > 50:
        logger.warning("Years of experience seems unrealistic")
        return False
    
    # Check that we have some analysis content
    if not experience_analysis.career_progression.strip():
        logger.warning("Career progression analysis is empty")
        return False
    
    return True


def calculate_experience_relevance_score(experience_analysis: ExperienceAnalysis, 
                                       position_keywords: List[str]) -> float:
    """
    Calculate relevance score based on experience analysis and position keywords.
    
    Args:
        experience_analysis: Analyzed experience information
        position_keywords: Keywords from the position description
        
    Returns:
        float: Relevance score between 0.0 and 1.0
    """
    
    if not experience_analysis or not position_keywords:
        return 0.0
    
    # Combine all experience text for analysis
    experience_text = " ".join([
        experience_analysis.summary,
        experience_analysis.career_progression,
        " ".join(experience_analysis.relevant_roles),
        " ".join(experience_analysis.strengths)
    ]).lower()
    
    # Count keyword matches
    matches = 0
    for keyword in position_keywords:
        if keyword.lower() in experience_text:
            matches += 1
    
    # Calculate relevance score
    if len(position_keywords) == 0:
        return 0.0
    
    relevance_score = matches / len(position_keywords)
    return min(relevance_score, 1.0)


def extract_key_achievements(experience_analysis: ExperienceAnalysis) -> List[str]:
    """
    Extract key achievements from experience analysis.
    
    Args:
        experience_analysis: Analyzed experience information
        
    Returns:
        List[str]: List of key achievements
    """
    
    achievements = []
    
    if not experience_analysis:
        return achievements
    
    # Extract achievements from strengths
    for strength in experience_analysis.strengths:
        if any(keyword in strength.lower() for keyword in 
               ['achieved', 'improved', 'increased', 'reduced', 'led', 'managed', 'delivered']):
            achievements.append(strength)
    
    # Extract quantifiable results from career progression
    career_text = experience_analysis.career_progression.lower()
    if any(keyword in career_text for keyword in 
           ['%', 'percent', 'million', 'thousand', 'team of', 'budget of']):
        achievements.append("Quantifiable results mentioned in career progression")
    
    return achievements


def handle_experience_analyzer_errors(state: CVAnalysisState, error: Exception) -> Dict[str, Any]:
    """
    Handle errors that occur during experience analysis.
    
    Args:
        state: Current workflow state
        error: Exception that occurred
        
    Returns:
        Dict containing updated state with error information
    """
    
    error_msg = f"Experience Analyzer failed: {str(error)}"
    logger.error(error_msg)
    
    return {
        "current_node": "experience_analyzer",
        "experience_analysis": None,
        "errors": state.errors + [error_msg]
    }
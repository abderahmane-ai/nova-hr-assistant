"""
Skills Analyzer Node for Nova HR Assistant.

This module contains the skills analyzer node that identifies and evaluates
technical and soft skills from CV using structured LLM prompts.
"""

import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.state import CVAnalysisState
from ..models.candidate import SkillsAnalysis
from ..utils.llm_manager import LLMManager, LLMProviderError


logger = logging.getLogger(__name__)


class SkillsAnalysisOutput(BaseModel):
    """Pydantic model for structured skills analyzer output."""
    
    technical_skills: List[str] = Field(description="List of technical skills identified")
    soft_skills: List[str] = Field(description="List of soft skills identified")
    skill_gaps: List[str] = Field(description="Skills gaps compared to position requirements")
    proficiency_levels: Dict[str, str] = Field(description="Proficiency levels for key skills")


class SkillsAnalyzerError(Exception):
    """Exception raised when skills analysis fails."""
    pass


def create_skills_analyzer_prompt() -> ChatPromptTemplate:
    """
    Create the structured prompt for skills analysis.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for skills analysis
    """
    
    system_message = """You are an expert HR analyst specialized in identifying and evaluating candidate skills from CVs.
Your task is to analyze the provided CV and position requirements to assess the candidate's technical and soft skills.

ANALYSIS GUIDELINES:
1. Identify all technical skills mentioned or implied in the CV (programming languages, frameworks, tools, technologies)
2. Identify soft skills demonstrated through experience descriptions and achievements
3. Assess proficiency levels for key skills based on experience context and years of use
4. Compare candidate skills against position requirements to identify gaps
5. Focus on skills that are relevant to the target position

TECHNICAL SKILLS CATEGORIES:
- Programming languages (Python, Java, JavaScript, etc.)
- Frameworks and libraries (React, Django, Spring, etc.)
- Databases (MySQL, PostgreSQL, MongoDB, etc.)
- Cloud platforms (AWS, Azure, GCP, etc.)
- DevOps tools (Docker, Kubernetes, Jenkins, etc.)
- Development tools (Git, IDEs, testing frameworks, etc.)

SOFT SKILLS CATEGORIES:
- Leadership and management
- Communication and collaboration
- Problem-solving and analytical thinking
- Project management and organization
- Adaptability and learning ability
- Customer service and stakeholder management

PROFICIENCY LEVELS:
- beginner: Basic knowledge, limited practical experience
- intermediate: Good understanding, some practical experience
- advanced: Strong expertise, extensive practical experience
- expert: Deep expertise, can mentor others, thought leader

IMPORTANT INSTRUCTIONS:
- Only include skills that are explicitly mentioned or clearly demonstrated in the CV
- Be conservative with proficiency level assessments - require clear evidence
- Focus on skills relevant to the target position
- Identify specific skill gaps that could impact job performance
- Consider both breadth and depth of skills

{format_instructions}"""

    human_message = """Please analyze the candidate's skills for the following position:

POSITION: {position}

CV TEXT:
{cv_text}

Provide a comprehensive analysis of the candidate's technical and soft skills according to the specified format."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def skills_analyzer_node(state: CVAnalysisState, llm_manager: LLMManager) -> Dict[str, Any]:
    """
    Skills Analyzer node function for LangGraph workflow.
    
    Identifies and evaluates technical and soft skills from CV against position requirements.
    
    Args:
        state: Current workflow state containing CV text and position
        llm_manager: LLM manager instance for making LLM calls
        
    Returns:
        Dict containing updated state with skills_analysis
        
    Raises:
        SkillsAnalyzerError: If skills analysis fails
    """
    
    logger.info("Starting skills analyzer node")
    
    # Update current node in state
    updated_state = {
        "current_node": "skills_analyzer",
        "errors": state.errors.copy()
    }
    
    try:
        # Validate input
        if not state.cv_text or not state.cv_text.strip():
            error_msg = "CV text is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise SkillsAnalyzerError(error_msg)
        
        if not state.position or not state.position.strip():
            error_msg = "Position description is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise SkillsAnalyzerError(error_msg)
        
        # Get LLM instance
        llm = llm_manager.get_llm()
        
        # Create output parser
        output_parser = PydanticOutputParser(pydantic_object=SkillsAnalysisOutput)
        
        # Create prompt template
        prompt_template = create_skills_analyzer_prompt()
        
        # Create the chain
        chain = prompt_template | llm | output_parser
        
        # Execute the chain
        logger.debug(f"Analyzing skills for position: {state.position}")
        
        result = chain.invoke({
            "cv_text": state.cv_text,
            "position": state.position,
            "format_instructions": output_parser.get_format_instructions()
        })
        
        # Convert to SkillsAnalysis model
        skills_analysis = SkillsAnalysis(
            technical_skills=[skill.strip() for skill in result.technical_skills],
            soft_skills=[skill.strip() for skill in result.soft_skills],
            skill_gaps=[gap.strip() for gap in result.skill_gaps],
            proficiency_levels={k.strip(): v.strip().lower() for k, v in result.proficiency_levels.items()}
        )
        
        # Validate the analysis
        validation_errors = skills_analysis.validate()
        if validation_errors:
            error_msg = f"Skills analysis validation failed: {', '.join(validation_errors)}"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise SkillsAnalyzerError(error_msg)
        
        # Update state with successful result
        updated_state["skills_analysis"] = skills_analysis
        
        logger.info(f"Successfully analyzed skills: {len(skills_analysis.technical_skills)} technical, "
                   f"{len(skills_analysis.soft_skills)} soft skills")
        logger.debug(f"Technical skills: {', '.join(skills_analysis.technical_skills[:5])}")
        
        return updated_state
        
    except LLMProviderError as e:
        error_msg = f"LLM provider error in skills analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise SkillsAnalyzerError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error in skills analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise SkillsAnalyzerError(error_msg)


def validate_skills_analysis_output(skills_analysis: SkillsAnalysis) -> bool:
    """
    Validate skills analysis output for completeness and accuracy.
    
    Args:
        skills_analysis: Analyzed skills information
        
    Returns:
        bool: True if output is valid, False otherwise
    """
    
    if not skills_analysis:
        return False
    
    # Check basic validation
    if not skills_analysis.is_valid():
        return False
    
    # Additional checks for skills analyzer specific requirements
    if len(skills_analysis.technical_skills) == 0 and len(skills_analysis.soft_skills) == 0:
        logger.warning("No skills identified in analysis")
        return False
    
    # Check that proficiency levels are reasonable
    for skill, level in skills_analysis.proficiency_levels.items():
        if level not in ['beginner', 'intermediate', 'advanced', 'expert']:
            logger.warning(f"Invalid proficiency level: {level}")
            return False
    
    return True


def calculate_skills_match_score(skills_analysis: SkillsAnalysis, 
                                required_skills: List[str]) -> float:
    """
    Calculate skills match score based on required skills for the position.
    
    Args:
        skills_analysis: Analyzed skills information
        required_skills: List of required skills for the position
        
    Returns:
        float: Skills match score between 0.0 and 1.0
    """
    
    if not skills_analysis or not required_skills:
        return 0.0
    
    # Combine all candidate skills
    all_candidate_skills = [skill.lower() for skill in 
                           skills_analysis.technical_skills + skills_analysis.soft_skills]
    
    # Count matches - use exact matching or meaningful partial matching
    matches = 0
    for required_skill in required_skills:
        required_skill_lower = required_skill.lower()
        # Check for exact match or meaningful partial match (at least 3 characters)
        if (required_skill_lower in all_candidate_skills or 
            any(len(required_skill_lower) >= 3 and required_skill_lower in candidate_skill 
                for candidate_skill in all_candidate_skills)):
            matches += 1
    
    # Calculate match score
    if len(required_skills) == 0:
        return 0.0
    
    match_score = matches / len(required_skills)
    return min(match_score, 1.0)


def identify_critical_skill_gaps(skills_analysis: SkillsAnalysis, 
                                critical_skills: List[str]) -> List[str]:
    """
    Identify critical skill gaps that could significantly impact job performance.
    
    Args:
        skills_analysis: Analyzed skills information
        critical_skills: List of critical skills for the position
        
    Returns:
        List[str]: List of critical skills that are missing
    """
    
    if not skills_analysis or not critical_skills:
        return []
    
    # Combine all candidate skills
    all_candidate_skills = [skill.lower() for skill in 
                           skills_analysis.technical_skills + skills_analysis.soft_skills]
    
    # Find missing critical skills
    critical_gaps = []
    for critical_skill in critical_skills:
        critical_skill_lower = critical_skill.lower()
        if not any(critical_skill_lower in candidate_skill for candidate_skill in all_candidate_skills):
            critical_gaps.append(critical_skill)
    
    return critical_gaps


def categorize_technical_skills(technical_skills: List[str]) -> Dict[str, List[str]]:
    """
    Categorize technical skills into different technology areas.
    
    Args:
        technical_skills: List of technical skills
        
    Returns:
        Dict[str, List[str]]: Categorized technical skills
    """
    
    categories = {
        "Programming Languages": [],
        "Frameworks & Libraries": [],
        "Databases": [],
        "Cloud Platforms": [],
        "DevOps & Tools": [],
        "Other": []
    }
    
    # Define keyword mappings for categorization
    category_keywords = {
        "Programming Languages": [
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", 
            "php", "ruby", "swift", "kotlin", "scala", "r", "matlab"
        ],
        "Frameworks & Libraries": [
            "react", "angular", "vue", "django", "flask", "spring", "express", 
            "node.js", "laravel", "rails", "bootstrap", "jquery"
        ],
        "Databases": [
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", 
            "sql server", "sqlite", "cassandra", "dynamodb"
        ],
        "Cloud Platforms": [
            "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean", 
            "cloudflare", "vercel", "netlify"
        ],
        "DevOps & Tools": [
            "docker", "kubernetes", "jenkins", "git", "gitlab", "github", 
            "terraform", "ansible", "chef", "puppet", "ci/cd"
        ]
    }
    
    for skill in technical_skills:
        skill_lower = skill.lower()
        categorized = False
        
        # Find the best matching category (exact match first, then partial)
        best_match = None
        best_score = 0
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if skill_lower == keyword:  # Exact match gets highest priority
                    best_match = category
                    best_score = 100
                    break
                elif keyword in skill_lower and len(keyword) > best_score:
                    best_match = category
                    best_score = len(keyword)
            
            if best_score == 100:  # Found exact match
                break
        
        if best_match:
            categories[best_match].append(skill)
            categorized = True
        
        if not categorized:
            categories["Other"].append(skill)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def assess_skill_depth(skills_analysis: SkillsAnalysis) -> Dict[str, str]:
    """
    Assess the depth of skills based on proficiency levels and skill count.
    
    Args:
        skills_analysis: Analyzed skills information
        
    Returns:
        Dict[str, str]: Assessment of skill depth in different areas
    """
    
    if not skills_analysis:
        return {}
    
    assessment = {}
    
    # Assess technical skill depth
    tech_skills_count = len(skills_analysis.technical_skills)
    expert_tech_skills = sum(1 for skill, level in skills_analysis.proficiency_levels.items() 
                            if level == 'expert' and skill in skills_analysis.technical_skills)
    
    if tech_skills_count == 0:
        assessment["Technical Skills"] = "No technical skills identified"
    elif expert_tech_skills > 2:
        assessment["Technical Skills"] = "Deep expertise in multiple technical areas"
    elif expert_tech_skills > 0:
        assessment["Technical Skills"] = "Strong expertise in some technical areas"
    elif tech_skills_count > 10:
        assessment["Technical Skills"] = "Broad technical knowledge"
    else:
        assessment["Technical Skills"] = "Limited technical skill breadth"
    
    # Assess soft skill depth
    soft_skills_count = len(skills_analysis.soft_skills)
    if soft_skills_count == 0:
        assessment["Soft Skills"] = "No soft skills clearly demonstrated"
    elif soft_skills_count > 5:
        assessment["Soft Skills"] = "Well-rounded soft skill set"
    else:
        assessment["Soft Skills"] = "Basic soft skills demonstrated"
    
    return assessment


def handle_skills_analyzer_errors(state: CVAnalysisState, error: Exception) -> Dict[str, Any]:
    """
    Handle errors that occur during skills analysis.
    
    Args:
        state: Current workflow state
        error: Exception that occurred
        
    Returns:
        Dict containing updated state with error information
    """
    
    error_msg = f"Skills Analyzer failed: {str(error)}"
    logger.error(error_msg)
    
    return {
        "current_node": "skills_analyzer",
        "skills_analysis": None,
        "errors": state.errors + [error_msg]
    }
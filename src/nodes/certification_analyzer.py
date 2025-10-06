"""
Certification Analyzer Node for Nova HR Assistant.

This module contains the certification analyzer node that identifies professional
certifications and achievements using structured LLM prompts.
"""

import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.state import CVAnalysisState
from ..models.candidate import CertificationAnalysis
from ..utils.llm_manager import LLMManager, LLMProviderError


logger = logging.getLogger(__name__)


class CertificationAnalysisOutput(BaseModel):
    """Pydantic model for structured certification analyzer output."""
    
    certifications: List[str] = Field(description="List of professional certifications identified")
    professional_memberships: List[str] = Field(description="List of professional memberships and associations")
    achievements: List[str] = Field(description="List of professional achievements and awards")
    relevance_assessment: str = Field(description="Assessment of certification relevance to position")


class CertificationAnalyzerError(Exception):
    """Exception raised when certification analysis fails."""
    pass


def create_certification_analyzer_prompt() -> ChatPromptTemplate:
    """
    Create the structured prompt for certification analysis.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for certification analysis
    """
    
    system_message = """You are an expert HR analyst specialized in evaluating professional certifications, achievements, and industry credentials.
Your task is to analyze the provided CV and position requirements to assess the candidate's professional certifications and achievements.

ANALYSIS GUIDELINES:
1. Identify all professional certifications, licenses, and industry credentials
2. Identify professional memberships, associations, and societies
3. Identify professional achievements, awards, honors, and recognitions
4. Assess the relevance and value of certifications to the target position
5. Consider the credibility and industry recognition of certifying bodies

CERTIFICATION CATEGORIES:
- Technical certifications (AWS, Azure, Google Cloud, Cisco, Microsoft, etc.)
- Project management certifications (PMP, Scrum Master, Agile, etc.)
- Industry-specific certifications (CISSP, CISA, Six Sigma, etc.)
- Professional licenses (PE, CPA, Bar admission, etc.)
- Vendor-specific certifications (Oracle, SAP, Salesforce, etc.)
- Academic certifications and continuing education credits

PROFESSIONAL MEMBERSHIPS:
- Professional societies (IEEE, ACM, PMI, etc.)
- Industry associations and trade organizations
- Alumni networks and professional networks
- Board memberships and advisory positions

ACHIEVEMENTS CATEGORIES:
- Professional awards and recognitions
- Publications and research contributions
- Speaking engagements and conference presentations
- Patents and intellectual property
- Leadership roles and positions
- Competition wins and honors

RELEVANCE ASSESSMENT CRITERIA:
- Direct relevance to position requirements and responsibilities
- Industry recognition and credibility of certifying body
- Recency and currency of certifications (active vs. expired)
- Level of difficulty and selectivity of certification
- Alignment with career progression and role expectations
- Demonstration of continuous learning and professional development

IMPORTANT INSTRUCTIONS:
- Only include certifications and achievements explicitly mentioned in the CV
- Distinguish between active/current and expired certifications when possible
- Consider the reputation and industry standing of certifying organizations
- Assess both breadth and depth of professional credentials
- Focus on certifications relevant to the target position and industry
- Provide specific assessment of how certifications enhance candidacy

{format_instructions}"""

    human_message = """Please analyze the candidate's professional certifications and achievements for the following position:

POSITION: {position}

CV TEXT:
{cv_text}

Provide a comprehensive analysis of the candidate's professional certifications, memberships, and achievements according to the specified format."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def certification_analyzer_node(state: CVAnalysisState, llm_manager: LLMManager) -> Dict[str, Any]:
    """
    Certification Analyzer node function for LangGraph workflow.
    
    Identifies professional certifications and achievements against position requirements.
    
    Args:
        state: Current workflow state containing CV text and position
        llm_manager: LLM manager instance for making LLM calls
        
    Returns:
        Dict containing updated state with certifications_analysis
        
    Raises:
        CertificationAnalyzerError: If certification analysis fails
    """
    
    logger.info("Starting certification analyzer node")
    
    # Update current node in state
    updated_state = {
        "current_node": "certification_analyzer",
        "errors": state.errors.copy()
    }
    
    try:
        # Validate input
        if not state.cv_text or not state.cv_text.strip():
            error_msg = "CV text is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise CertificationAnalyzerError(error_msg)
        
        if not state.position or not state.position.strip():
            error_msg = "Position description is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise CertificationAnalyzerError(error_msg)
        
        # Get LLM instance
        llm = llm_manager.get_llm()
        
        # Create output parser
        output_parser = PydanticOutputParser(pydantic_object=CertificationAnalysisOutput)
        
        # Create prompt template
        prompt_template = create_certification_analyzer_prompt()
        
        # Create the chain
        chain = prompt_template | llm | output_parser
        
        # Execute the chain
        logger.debug(f"Analyzing certifications for position: {state.position}")
        
        result = chain.invoke({
            "cv_text": state.cv_text,
            "position": state.position,
            "format_instructions": output_parser.get_format_instructions()
        })
        
        # Convert to CertificationAnalysis model
        certifications_analysis = CertificationAnalysis(
            certifications=[cert.strip() for cert in result.certifications],
            professional_memberships=[membership.strip() for membership in result.professional_memberships],
            achievements=[achievement.strip() for achievement in result.achievements],
            relevance_assessment=result.relevance_assessment.strip()
        )
        
        # Validate the analysis
        validation_errors = certifications_analysis.validate()
        if validation_errors:
            error_msg = f"Certification analysis validation failed: {', '.join(validation_errors)}"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise CertificationAnalyzerError(error_msg)
        
        # Update state with successful result
        updated_state["certifications_analysis"] = certifications_analysis
        
        logger.info(f"Successfully analyzed certifications: {len(certifications_analysis.certifications)} certifications, "
                   f"{len(certifications_analysis.professional_memberships)} memberships, "
                   f"{len(certifications_analysis.achievements)} achievements")
        logger.debug(f"Certifications: {', '.join(certifications_analysis.certifications[:3])}")
        
        return updated_state
        
    except LLMProviderError as e:
        error_msg = f"LLM provider error in certification analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise CertificationAnalyzerError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error in certification analyzer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise CertificationAnalyzerError(error_msg)


def validate_certification_analysis_output(certifications_analysis: CertificationAnalysis) -> bool:
    """
    Validate certification analysis output for completeness and accuracy.
    
    Args:
        certifications_analysis: Analyzed certification information
        
    Returns:
        bool: True if output is valid, False otherwise
    """
    
    if not certifications_analysis:
        return False
    
    # Check basic validation
    if not certifications_analysis.is_valid():
        return False
    
    # Additional checks for certification analyzer specific requirements
    total_items = (len(certifications_analysis.certifications) + 
                  len(certifications_analysis.professional_memberships) + 
                  len(certifications_analysis.achievements))
    
    # Allow empty results if no certifications are found
    if total_items == 0:
        logger.info("No certifications, memberships, or achievements found")
        return True
    
    # If we have certifications/achievements, we should have a relevance assessment
    if total_items > 0 and not certifications_analysis.relevance_assessment:
        logger.warning("Missing relevance assessment for identified certifications")
        return False
    
    return True


def calculate_certification_relevance_score(certifications_analysis: CertificationAnalysis, 
                                          position_keywords: List[str]) -> float:
    """
    Calculate relevance score based on certification analysis and position keywords.
    
    Args:
        certifications_analysis: Analyzed certification information
        position_keywords: Keywords from the position description
        
    Returns:
        float: Relevance score between 0.0 and 1.0
    """
    
    if not certifications_analysis or not position_keywords:
        return 0.0
    
    # Combine all certification text for keyword analysis
    certification_text = " ".join([
        " ".join(certifications_analysis.certifications),
        " ".join(certifications_analysis.professional_memberships),
        " ".join(certifications_analysis.achievements),
        certifications_analysis.relevance_assessment
    ]).lower()
    
    # Count keyword matches
    matches = 0
    for keyword in position_keywords:
        if keyword.lower() in certification_text:
            matches += 1
    
    # Calculate keyword relevance score
    keyword_score = matches / len(position_keywords) if position_keywords else 0.0
    
    # Boost score based on number and quality of certifications
    cert_count = len(certifications_analysis.certifications)
    achievement_count = len(certifications_analysis.achievements)
    membership_count = len(certifications_analysis.professional_memberships)
    
    # Quality boost based on content
    quality_boost = 0.0
    if cert_count > 0:
        quality_boost += 0.3
    if achievement_count > 0:
        quality_boost += 0.2
    if membership_count > 0:
        quality_boost += 0.1
    
    # Combine scores
    final_score = (keyword_score * 0.7) + (quality_boost * 0.3)
    return min(final_score, 1.0)


def extract_certification_highlights(certifications_analysis: CertificationAnalysis) -> List[str]:
    """
    Extract key certification highlights from certification analysis.
    
    Args:
        certifications_analysis: Analyzed certification information
        
    Returns:
        List[str]: List of certification highlights
    """
    
    highlights = []
    
    if not certifications_analysis:
        return highlights
    
    # Add certification highlights
    high_value_certs = []
    for cert in certifications_analysis.certifications:
        cert_lower = cert.lower()
        if any(keyword in cert_lower for keyword in 
               ['aws', 'azure', 'google cloud', 'pmp', 'cissp', 'cisa', 'microsoft', 'oracle', 'cisco']):
            high_value_certs.append(cert)
    
    if high_value_certs:
        highlights.append(f"Industry-recognized certifications: {', '.join(high_value_certs[:3])}")
    
    # Add professional membership highlights
    if certifications_analysis.professional_memberships:
        highlights.append(f"Professional memberships: {len(certifications_analysis.professional_memberships)} organizations")
    
    # Add achievement highlights
    notable_achievements = []
    for achievement in certifications_analysis.achievements:
        achievement_lower = achievement.lower()
        if any(keyword in achievement_lower for keyword in 
               ['award', 'recognition', 'patent', 'publication', 'speaker', 'winner']):
            notable_achievements.append(achievement)
    
    if notable_achievements:
        highlights.append(f"Notable achievements: {', '.join(notable_achievements[:2])}")
    
    # Add relevance assessment if positive
    if certifications_analysis.relevance_assessment:
        assessment_lower = certifications_analysis.relevance_assessment.lower()
        if any(keyword in assessment_lower for keyword in 
               ['highly relevant', 'strong', 'excellent', 'valuable', 'significant']):
            highlights.append("Strong certification relevance to position")
    
    return highlights


def categorize_certifications(certifications: List[str]) -> Dict[str, List[str]]:
    """
    Categorize certifications into different technology and professional areas.
    
    Args:
        certifications: List of certifications
        
    Returns:
        Dict[str, List[str]]: Categorized certifications
    """
    
    categories = {
        "Cloud Platforms": [],
        "Project Management": [],
        "Security": [],
        "Technical": [],
        "Professional": [],
        "Other": []
    }
    
    # Define keyword mappings for categorization
    category_keywords = {
        "Cloud Platforms": [
            "aws", "azure", "google cloud", "gcp", "cloud", "solutions architect",
            "cloud practitioner", "devops engineer"
        ],
        "Project Management": [
            "pmp", "scrum master", "agile", "project management", "pmbok", "prince2",
            "certified scrum", "safe", "kanban"
        ],
        "Security": [
            "cissp", "cisa", "cism", "security+", "ethical hacker", "penetration testing",
            "cybersecurity", "information security"
        ],
        "Technical": [
            "microsoft", "oracle", "cisco", "vmware", "red hat", "linux", "java",
            "python", "developer", "administrator", "engineer"
        ],
        "Professional": [
            "cpa", "chartered", "professional engineer", "licensed", "board certified",
            "fellow", "member"
        ]
    }
    
    for cert in certifications:
        cert_lower = cert.lower()
        categorized = False
        
        # Find the best matching category
        best_match = None
        best_score = 0
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in cert_lower and len(keyword) > best_score:
                    best_match = category
                    best_score = len(keyword)
        
        if best_match:
            categories[best_match].append(cert)
            categorized = True
        
        if not categorized:
            categories["Other"].append(cert)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def assess_certification_currency(certifications_analysis: CertificationAnalysis) -> Dict[str, str]:
    """
    Assess the currency and recency of certifications.
    
    Args:
        certifications_analysis: Analyzed certification information
        
    Returns:
        Dict[str, str]: Assessment of certification currency
    """
    
    if not certifications_analysis:
        return {}
    
    assessment = {}
    
    # Count certifications by category
    cert_count = len(certifications_analysis.certifications)
    achievement_count = len(certifications_analysis.achievements)
    membership_count = len(certifications_analysis.professional_memberships)
    
    # Assess certification portfolio
    if cert_count == 0:
        assessment["Certification Portfolio"] = "No professional certifications identified"
    elif cert_count >= 5:
        assessment["Certification Portfolio"] = "Extensive certification portfolio"
    elif cert_count >= 3:
        assessment["Certification Portfolio"] = "Good range of professional certifications"
    else:
        assessment["Certification Portfolio"] = "Limited professional certifications"
    
    # Assess professional engagement
    if membership_count == 0 and achievement_count == 0:
        assessment["Professional Engagement"] = "Limited professional community involvement"
    elif membership_count > 0 and achievement_count > 0:
        assessment["Professional Engagement"] = "Active professional community participation"
    elif membership_count > 0:
        assessment["Professional Engagement"] = "Professional association memberships"
    else:
        assessment["Professional Engagement"] = "Some professional achievements noted"
    
    # Assess relevance based on assessment text
    if certifications_analysis.relevance_assessment:
        assessment_lower = certifications_analysis.relevance_assessment.lower()
        if any(keyword in assessment_lower for keyword in 
               ['highly relevant', 'strong alignment', 'excellent match']):
            assessment["Position Relevance"] = "High relevance to target position"
        elif any(keyword in assessment_lower for keyword in 
                ['relevant', 'applicable', 'useful']):
            assessment["Position Relevance"] = "Moderate relevance to target position"
        else:
            assessment["Position Relevance"] = "Limited relevance to target position"
    
    return assessment


def handle_certification_analyzer_errors(state: CVAnalysisState, error: Exception) -> Dict[str, Any]:
    """
    Handle errors that occur during certification analysis.
    
    Args:
        state: Current workflow state
        error: Exception that occurred
        
    Returns:
        Dict containing updated state with error information
    """
    
    error_msg = f"Certification Analyzer failed: {str(error)}"
    logger.error(error_msg)
    
    return {
        "current_node": "certification_analyzer",
        "certifications_analysis": None,
        "errors": state.errors + [error_msg]
    }
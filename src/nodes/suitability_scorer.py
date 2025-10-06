"""
Suitability Scorer Node for Nova HR Assistant.

This module contains the suitability scorer node that calculates overall candidate
fit score using all analysis results and configurable weights.
"""

import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.state import CVAnalysisState
from ..utils.llm_manager import LLMManager, LLMProviderError
from ..utils.bias_detection import detect_and_mitigate_biases, generate_bias_report
from ..config.nova_config import NovaConfig


logger = logging.getLogger(__name__)


class SuitabilityScoreOutput(BaseModel):
    """Pydantic model for structured suitability scorer output."""
    
    overall_score: int = Field(description="Overall suitability score from 0-100", ge=0, le=100)
    experience_score: int = Field(description="Experience relevance score from 0-100", ge=0, le=100)
    skills_score: int = Field(description="Skills alignment score from 0-100", ge=0, le=100)
    education_score: int = Field(description="Education relevance score from 0-100", ge=0, le=100)
    certifications_score: int = Field(description="Certifications relevance score from 0-100", ge=0, le=100)
    overall_fit_score: int = Field(description="Overall cultural and role fit score from 0-100", ge=0, le=100)
    strengths: List[str] = Field(description="Key strengths that make candidate suitable")
    weaknesses: List[str] = Field(description="Areas where candidate may need development")
    recommendation: str = Field(description="Clear recommendation: 'Highly Recommended', 'Recommended', 'Consider with Reservations', or 'Not Recommended'")
    justification: str = Field(description="Detailed justification for the score and recommendation")
    confidence_level: float = Field(description="Confidence level in the assessment (0.0-1.0)", ge=0.0, le=1.0)
    risk_factors: List[str] = Field(description="Identified risk factors that could impact performance", default_factory=list)
    potential_score: int = Field(description="Potential score with development (0-100)", ge=0, le=100)
    development_timeline: str = Field(description="Estimated timeline for reaching full potential", default="")


class SuitabilityScorerError(Exception):
    """Exception raised when suitability scoring fails."""
    pass


def create_suitability_scorer_prompt() -> ChatPromptTemplate:
    """
    Create the structured prompt for suitability scoring.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for suitability scoring
    """
    
    system_message = """You are an expert HR analyst specialized in holistic candidate evaluation and scoring.
Your task is to analyze all the candidate information and provide a comprehensive suitability assessment.

SCORING GUIDELINES:
1. Overall Score (0-100): Weighted combination of all factors with confidence assessment
2. Experience Score (0-100): Relevance and quality of work experience, considering career progression
3. Skills Score (0-100): Technical and soft skills alignment with position requirements
4. Education Score (0-100): Educational background relevance and continuous learning
5. Certifications Score (0-100): Professional certifications and achievements relevance
6. Overall Fit Score (0-100): Cultural fit, career trajectory, and role alignment

ADVANCED EVALUATION CRITERIA:
- Direct relevance to the specific position requirements
- Quality and depth of experience in relevant areas
- Skill gaps and development potential with realistic timelines
- Educational foundation and continuous learning indicators
- Professional certifications and industry recognition
- Career progression and growth trajectory analysis
- Cultural fit indicators and soft skills assessment
- Leadership potential and team collaboration abilities
- Risk factors that could impact performance
- Potential for growth and development within the role
- Transferable skills and adaptability indicators

CONFIDENCE ASSESSMENT:
- High confidence (0.8-1.0): Clear evidence, consistent profile, strong indicators
- Medium confidence (0.5-0.79): Good evidence with some uncertainties
- Low confidence (0.0-0.49): Limited evidence, conflicting indicators, or significant gaps

RISK FACTOR IDENTIFICATION:
- Critical skill gaps that could impact immediate performance
- Experience gaps in key areas
- Educational misalignment with role requirements
- Career instability or concerning patterns
- Limited evidence of continuous learning
- Potential cultural fit issues

POTENTIAL ASSESSMENT:
- Consider candidate's growth trajectory and learning ability
- Assess transferable skills and adaptability
- Evaluate development timeline based on role complexity
- Factor in mentorship and training availability

RECOMMENDATION CATEGORIES:
- "Highly Recommended" (80-100): Exceptional candidate, strong fit, low risk
- "Recommended" (60-79): Good candidate, meets most requirements, manageable risks
- "Consider with Reservations" (40-59): Potential candidate with significant concerns
- "Not Recommended" (0-39): Poor fit for the position, high risk factors

IMPORTANT INSTRUCTIONS:
- Be objective and evidence-based in your scoring
- Consider both strengths and weaknesses fairly with realistic assessment
- Provide specific justification for scores with concrete examples
- Focus on job-relevant factors and role-specific requirements
- Consider potential for growth and development with realistic timelines
- Be consistent in your evaluation criteria across all candidates
- For candidates with minimal experience, focus on education, skills, potential, and transferable abilities
- Be fair but realistic with scoring for junior candidates - consider their potential AND limitations
- For entry-level positions, weight skills and education more heavily than experience
- Don't penalize candidates heavily for lack of experience if they show strong skills and potential
- Identify specific risk factors that could impact performance
- Assess confidence level based on available evidence quality
- Provide realistic development timelines for improvement areas
- Ensure recommendation aligns with score ranges and risk assessment

{format_instructions}"""

    human_message = """Please provide a comprehensive suitability assessment for the following candidate:

POSITION: {position}

CANDIDATE INFORMATION:
Name: {candidate_name}
Contact: {candidate_contact}
Location: {candidate_location}

EXPERIENCE ANALYSIS:
Summary: {experience_summary}
Years of Experience: {years_experience}
Relevant Roles: {relevant_roles}
Career Progression: {career_progression}
Experience Strengths: {experience_strengths}
Experience Gaps: {experience_gaps}

SKILLS ANALYSIS:
Technical Skills: {technical_skills}
Soft Skills: {soft_skills}
Skill Gaps: {skill_gaps}
Proficiency Levels: {proficiency_levels}

EDUCATION ANALYSIS:
Degrees: {degrees}
Institutions: {institutions}
Education Relevance Score: {education_relevance_score}
Additional Training: {additional_training}

CERTIFICATIONS ANALYSIS:
Certifications: {certifications}
Professional Memberships: {professional_memberships}
Achievements: {achievements}
Relevance Assessment: {relevance_assessment}

Provide a comprehensive suitability assessment with detailed scoring and clear recommendation."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def suitability_scorer_node(state: CVAnalysisState, llm_manager: LLMManager) -> Dict[str, Any]:
    """
    Suitability Scorer node function for LangGraph workflow.
    
    Calculates overall candidate fit score using all analysis results.
    
    Args:
        state: Current workflow state containing all analysis results
        llm_manager: LLM manager instance for making LLM calls
        config: Nova configuration with scoring weights
        
    Returns:
        Dict containing updated state with suitability_score
        
    Raises:
        SuitabilityScorerError: If suitability scoring fails
    """
    
    logger.info("Starting suitability scorer node")
    
    # Update current node in state
    updated_state = {
        "current_node": "suitability_scorer",
        "errors": state.errors.copy()
    }
    
    try:
        # Validate that all required analysis components are present
        if not state.has_required_analysis():
            error_msg = "Missing required analysis components for scoring"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise SuitabilityScorerError(error_msg)
        
        # Validate input
        if not state.position or not state.position.strip():
            error_msg = "Position description is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise SuitabilityScorerError(error_msg)
        
        # Get LLM instance
        llm = llm_manager.get_llm()
        
        # Create output parser
        output_parser = PydanticOutputParser(pydantic_object=SuitabilityScoreOutput)
        
        # Create prompt template
        prompt_template = create_suitability_scorer_prompt()
        
        # Create the chain
        chain = prompt_template | llm | output_parser
        
        # Prepare input data
        input_data = _prepare_scoring_input_data(state)
        input_data["format_instructions"] = output_parser.get_format_instructions()
        
        # Execute the chain
        logger.debug(f"Calculating suitability score for position: {state.position}")
        
        result = chain.invoke(input_data)
        
        # Calculate dynamic weights based on position and candidate profile
        dynamic_weights = _calculate_dynamic_weights(state, result)
        
        # Calculate weighted overall score using dynamic weights
        weighted_score = _calculate_weighted_score(result, dynamic_weights)
        
        # Update the overall score with weighted calculation
        result.overall_score = weighted_score
        
        # Calculate additional scoring metrics
        result.confidence_level = _calculate_confidence_score(state, result)
        result.risk_factors = _identify_risk_factors(state, result)
        result.potential_score = _calculate_potential_score(state, result)
        result.development_timeline = _estimate_development_timeline(state, result)
        
        # Detect and mitigate biases
        analysis_data = _prepare_analysis_data_for_bias_detection(state, result)
        mitigated_data, detected_biases = detect_and_mitigate_biases(analysis_data)
        
        # Update result with bias-mitigated data if significant biases were found
        if detected_biases:
            logger.warning(f"Detected {len(detected_biases)} potential biases, applying mitigation")
            result = _apply_bias_mitigation_to_result(result, mitigated_data)
        
        # Validate the scoring result
        validation_errors = _validate_scoring_result(result)
        if validation_errors:
            error_msg = f"Suitability scoring validation failed: {', '.join(validation_errors)}"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise SuitabilityScorerError(error_msg)
        
        # Update state with successful result
        updated_state["suitability_score"] = result.overall_score
        
        # Store detailed scoring information in final_report for later use
        if "scoring_details" not in state.final_report:
            updated_state["final_report"] = state.final_report.copy()
        else:
            updated_state["final_report"] = state.final_report
            
        # Generate bias report
        bias_report = generate_bias_report(detected_biases) if detected_biases else None
        
        # Enhanced logging with key metrics (using ASCII-safe characters)
        logger.info("ENHANCED SCORING RESULTS:")
        logger.info(f"   Current Score: {result.overall_score}/100")
        logger.info(f"   Potential Score: {result.potential_score}/100")
        logger.info(f"   Confidence Level: {result.confidence_level:.1%}")
        logger.info(f"   Risk Factors: {len(result.risk_factors)} identified")
        logger.info(f"   Development Timeline: {result.development_timeline}")
        
        # Log component scores with weights
        logger.info("   Component Scores:")
        component_scores = {
            "experience": result.experience_score,
            "skills": result.skills_score,
            "education": result.education_score,
            "certifications": result.certifications_score,
            "overall_fit": result.overall_fit_score
        }
        # Use the already calculated dynamic weights
        for component, score in component_scores.items():
            weight = dynamic_weights.get(component, 0.0)
            logger.info(f"      - {component.title()}: {score}/100 (weight: {weight:.1%})")
        
        # Log risk factors if any
        if result.risk_factors:
            logger.info("   Risk Factors:")
            for i, risk in enumerate(result.risk_factors[:3], 1):  # Show top 3 risks
                logger.info(f"      {i}. {risk}")
        
        # Log bias detection results
        if detected_biases:
            logger.info(f"   Bias Detection: {len(detected_biases)} biases detected and mitigated")
        else:
            logger.info("   Bias Detection: No significant biases detected")
        
        updated_state["final_report"]["scoring_details"] = {
            "overall_score": result.overall_score,
            "component_scores": {
                "experience": result.experience_score,
                "skills": result.skills_score,
                "education": result.education_score,
                "certifications": result.certifications_score,
                "overall_fit": result.overall_fit_score
            },
            "strengths": result.strengths,
            "weaknesses": result.weaknesses,
            "recommendation": result.recommendation,
            "justification": result.justification,
            "scoring_weights": dynamic_weights,
            "confidence_level": result.confidence_level,
            "risk_factors": result.risk_factors,
            "potential_score": result.potential_score,
            "development_timeline": result.development_timeline,
            "bias_detection": bias_report
        }
        
        logger.info(f"Successfully calculated suitability score: {result.overall_score}/100 "
                   f"({result.recommendation})")
        logger.debug(f"Component scores - Experience: {result.experience_score}, "
                    f"Skills: {result.skills_score}, Education: {result.education_score}, "
                    f"Certifications: {result.certifications_score}, Fit: {result.overall_fit_score}")
        
        return updated_state
        
    except LLMProviderError as e:
        error_msg = f"LLM provider error in suitability scorer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise SuitabilityScorerError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error in suitability scorer: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise SuitabilityScorerError(error_msg)


def _prepare_scoring_input_data(state: CVAnalysisState) -> Dict[str, Any]:
    """
    Prepare input data for the scoring prompt.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict containing formatted input data for the prompt
    """
    
    return {
        "position": state.position,
        "candidate_name": state.candidate_info.name if state.candidate_info else "Unknown",
        "candidate_contact": state.candidate_info.contact_info if state.candidate_info else "Unknown",
        "candidate_location": state.candidate_info.location or "Not specified" if state.candidate_info else "Unknown",
        "experience_summary": state.experience_analysis.summary if state.experience_analysis else "No experience analysis",
        "years_experience": state.experience_analysis.years_experience if state.experience_analysis else 0,
        "relevant_roles": ", ".join(state.experience_analysis.relevant_roles) if state.experience_analysis and state.experience_analysis.relevant_roles else "No relevant professional roles identified",
        "career_progression": state.experience_analysis.career_progression if state.experience_analysis else "No progression analysis",
        "experience_strengths": ", ".join(state.experience_analysis.strengths) if state.experience_analysis else "None identified",
        "experience_gaps": ", ".join(state.experience_analysis.gaps) if state.experience_analysis else "None identified",
        "technical_skills": ", ".join(state.skills_analysis.technical_skills) if state.skills_analysis else "None identified",
        "soft_skills": ", ".join(state.skills_analysis.soft_skills) if state.skills_analysis else "None identified",
        "skill_gaps": ", ".join(state.skills_analysis.skill_gaps) if state.skills_analysis else "None identified",
        "proficiency_levels": str(state.skills_analysis.proficiency_levels) if state.skills_analysis else "None specified",
        "degrees": ", ".join(state.education_analysis.degrees) if state.education_analysis else "None",
        "institutions": ", ".join(state.education_analysis.institutions) if state.education_analysis else "None",
        "education_relevance_score": state.education_analysis.relevance_score if state.education_analysis else 0,
        "additional_training": ", ".join(state.education_analysis.additional_training) if state.education_analysis else "None",
        "certifications": ", ".join(state.certifications_analysis.certifications) if state.certifications_analysis else "None",
        "professional_memberships": ", ".join(state.certifications_analysis.professional_memberships) if state.certifications_analysis else "None",
        "achievements": ", ".join(state.certifications_analysis.achievements) if state.certifications_analysis else "None",
        "relevance_assessment": state.certifications_analysis.relevance_assessment if state.certifications_analysis else "No assessment"
    }


def _calculate_weighted_score(result: SuitabilityScoreOutput, weights: Dict[str, float]) -> int:
    """
    Calculate weighted overall score based on component scores and weights.
    
    Args:
        result: Scoring result with component scores
        weights: Scoring weights configuration
        
    Returns:
        int: Weighted overall score (0-100)
    """
    
    weighted_score = (
        result.experience_score * weights.get("experience", 0.3) +
        result.skills_score * weights.get("skills", 0.25) +
        result.education_score * weights.get("education", 0.2) +
        result.certifications_score * weights.get("certifications", 0.15) +
        result.overall_fit_score * weights.get("overall_fit", 0.1)
    )
    
    return int(round(weighted_score))


def _validate_scoring_result(result: SuitabilityScoreOutput) -> List[str]:
    """
    Validate scoring result for completeness and consistency.
    
    Args:
        result: Scoring result to validate
        
    Returns:
        List[str]: List of validation errors
    """
    
    errors = []
    
    # Check score ranges (already validated by Pydantic, but double-check)
    scores = [
        ("overall_score", result.overall_score),
        ("experience_score", result.experience_score),
        ("skills_score", result.skills_score),
        ("education_score", result.education_score),
        ("certifications_score", result.certifications_score),
        ("overall_fit_score", result.overall_fit_score),
        ("potential_score", result.potential_score)
    ]
    
    for score_name, score_value in scores:
        if not (0 <= score_value <= 100):
            errors.append(f"{score_name} must be between 0 and 100, got {score_value}")
    
    # Check confidence level
    if not (0.0 <= result.confidence_level <= 1.0):
        errors.append(f"confidence_level must be between 0.0 and 1.0, got {result.confidence_level}")
    
    # Check recommendation consistency with overall score and auto-correct if needed
    score_recommendation_mapping = {
        "Highly Recommended": (80, 100),
        "Recommended": (60, 79),
        "Consider with Reservations": (40, 59),
        "Not Recommended": (0, 39)
    }
    
    # Auto-correct recommendation based on score
    correct_recommendation = None
    for rec, (min_score, max_score) in score_recommendation_mapping.items():
        if min_score <= result.overall_score <= max_score:
            correct_recommendation = rec
            break
    
    if correct_recommendation and result.recommendation != correct_recommendation:
        logger.warning(f"Auto-correcting recommendation from '{result.recommendation}' to '{correct_recommendation}' for score {result.overall_score}")
        result.recommendation = correct_recommendation
    
    # Check that we have meaningful content
    if len(result.justification.strip()) < 50:
        errors.append("Justification is too short (minimum 50 characters)")
    
    if not result.strengths:
        errors.append("At least one strength must be identified")
    
    if not result.weaknesses:
        errors.append("At least one weakness must be identified")
    
    # Check potential score is reasonable
    if result.potential_score < result.overall_score:
        errors.append("Potential score should not be lower than current score")
    
    if result.potential_score - result.overall_score > 30:
        errors.append("Potential score improvement seems unrealistic (more than 30 points)")
    
    return errors


def validate_suitability_score_output(score: int, scoring_details: Dict[str, Any]) -> bool:
    """
    Validate suitability score output for completeness and accuracy.
    
    Args:
        score: Overall suitability score
        scoring_details: Detailed scoring information
        
    Returns:
        bool: True if output is valid, False otherwise
    """
    
    if not (0 <= score <= 100):
        logger.warning(f"Invalid suitability score: {score}")
        return False
    
    if not scoring_details:
        logger.warning("Missing scoring details")
        return False
    
    required_fields = ["component_scores", "strengths", "weaknesses", "recommendation", "justification"]
    for field in required_fields:
        if field not in scoring_details:
            logger.warning(f"Missing required field in scoring details: {field}")
            return False
    
    # Check component scores
    component_scores = scoring_details.get("component_scores", {})
    required_components = ["experience", "skills", "education", "certifications", "overall_fit"]
    for component in required_components:
        if component not in component_scores:
            logger.warning(f"Missing component score: {component}")
            return False
        
        component_score = component_scores[component]
        if not (0 <= component_score <= 100):
            logger.warning(f"Invalid component score for {component}: {component_score}")
            return False
    
    return True


def calculate_score_confidence(scoring_details: Dict[str, Any]) -> float:
    """
    Calculate confidence level for the scoring based on component score consistency.
    
    Args:
        scoring_details: Detailed scoring information
        
    Returns:
        float: Confidence level between 0.0 and 1.0
    """
    
    if not scoring_details or "component_scores" not in scoring_details:
        return 0.0
    
    component_scores = list(scoring_details["component_scores"].values())
    if not component_scores:
        return 0.0
    
    # Calculate standard deviation of component scores
    mean_score = sum(component_scores) / len(component_scores)
    variance = sum((score - mean_score) ** 2 for score in component_scores) / len(component_scores)
    std_dev = variance ** 0.5
    
    # Convert standard deviation to confidence (lower std_dev = higher confidence)
    # Normalize to 0-1 range where 0 std_dev = 1.0 confidence, 50 std_dev = 0.0 confidence
    confidence = max(0.0, 1.0 - (std_dev / 50.0))
    
    return confidence


def _calculate_dynamic_weights(state: CVAnalysisState, result: SuitabilityScoreOutput) -> Dict[str, float]:
    """
    Calculate dynamic scoring weights based on position type and candidate profile.
    
    Args:
        state: Current workflow state
        result: Scoring result with component scores
        
    Returns:
        Dict containing dynamic weights for each scoring component
    """
    
    # Base weights
    base_weights = {
        "experience": 0.25,
        "skills": 0.3,
        "education": 0.2,
        "certifications": 0.15,
        "overall_fit": 0.1
    }
    
    # Position type adjustments
    position_lower = state.position.lower()
    
    # Entry-level positions: emphasize education and skills over experience
    if any(keyword in position_lower for keyword in ['entry', 'junior', 'graduate', 'trainee', 'intern']):
        base_weights["experience"] = 0.15  # Reduced
        base_weights["skills"] = 0.35      # Increased
        base_weights["education"] = 0.25   # Increased
        base_weights["overall_fit"] = 0.15 # Increased
        base_weights["certifications"] = 0.1 # Reduced
    
    # Senior/leadership positions: emphasize experience and fit
    elif any(keyword in position_lower for keyword in ['senior', 'lead', 'manager', 'director', 'principal', 'architect']):
        base_weights["experience"] = 0.35   # Increased
        base_weights["skills"] = 0.25      # Reduced
        base_weights["overall_fit"] = 0.2  # Increased
        base_weights["education"] = 0.15   # Reduced
        base_weights["certifications"] = 0.05 # Reduced
    
    # Technical positions: emphasize skills and certifications
    elif any(keyword in position_lower for keyword in ['engineer', 'developer', 'technical', 'programmer', 'analyst']):
        base_weights["skills"] = 0.4       # Increased
        base_weights["experience"] = 0.25  # Same
        base_weights["certifications"] = 0.2 # Increased
        base_weights["education"] = 0.1    # Reduced
        base_weights["overall_fit"] = 0.05 # Reduced
    
    # Management positions: emphasize experience and fit
    elif any(keyword in position_lower for keyword in ['manager', 'supervisor', 'team lead', 'head of']):
        base_weights["experience"] = 0.3   # Increased
        base_weights["overall_fit"] = 0.25  # Increased
        base_weights["skills"] = 0.25      # Reduced
        base_weights["education"] = 0.15    # Reduced
        base_weights["certifications"] = 0.05 # Reduced
    
    # Candidate profile adjustments
    if state.experience_analysis:
        years_exp = state.experience_analysis.years_experience
        
        # For candidates with no experience, reduce experience weight
        if years_exp == 0:
            base_weights["experience"] *= 0.5
            base_weights["skills"] *= 1.2
            base_weights["education"] *= 1.3
        
        # For very experienced candidates, increase experience weight
        elif years_exp > 10:
            base_weights["experience"] *= 1.2
            base_weights["education"] *= 0.8
    
    # Skills profile adjustments
    if state.skills_analysis:
        tech_skills_count = len(state.skills_analysis.technical_skills or [])
        skill_gaps_count = len(state.skills_analysis.skill_gaps or [])
        
        # If many skills but many gaps, reduce skills weight
        if tech_skills_count > 10 and skill_gaps_count > 5:
            base_weights["skills"] *= 0.9
        
        # If few skills, increase skills weight to highlight the gap
        elif tech_skills_count < 3:
            base_weights["skills"] *= 1.1
    
    # Education profile adjustments
    if state.education_analysis:
        relevance_score = state.education_analysis.relevance_score or 0
        
        # If education is highly relevant, increase its weight
        if relevance_score >= 80:
            base_weights["education"] *= 1.2
        
        # If education is poorly relevant, decrease its weight
        elif relevance_score < 30:
            base_weights["education"] *= 0.7
    
    # Normalize weights to sum to 1.0
    total_weight = sum(base_weights.values())
    normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
    
    logger.debug(f"Dynamic weights calculated: {normalized_weights}")
    return normalized_weights


def _calculate_confidence_score(state: CVAnalysisState, result: SuitabilityScoreOutput) -> float:
    """
    Calculate confidence score based on data quality and consistency.
    
    Args:
        state: Current workflow state
        result: Scoring result
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    
    confidence_factors = []
    
    # Data completeness factor
    completeness_score = 0.0
    if state.candidate_info and state.candidate_info.name:
        completeness_score += 0.2
    if state.experience_analysis and state.experience_analysis.summary:
        completeness_score += 0.2
    if state.skills_analysis and state.skills_analysis.technical_skills:
        completeness_score += 0.2
    if state.education_analysis and state.education_analysis.degrees:
        completeness_score += 0.2
    if state.certifications_analysis and state.certifications_analysis.certifications:
        completeness_score += 0.2
    
    confidence_factors.append(completeness_score)
    
    # Score consistency factor
    component_scores = [
        result.experience_score,
        result.skills_score,
        result.education_score,
        result.certifications_score,
        result.overall_fit_score
    ]
    
    if component_scores:
        mean_score = sum(component_scores) / len(component_scores)
        variance = sum((score - mean_score) ** 2 for score in component_scores) / len(component_scores)
        std_dev = variance ** 0.5
        
        # Lower standard deviation = higher consistency = higher confidence
        consistency_score = max(0.0, 1.0 - (std_dev / 50.0))
        confidence_factors.append(consistency_score)
    
    # Evidence quality factor
    evidence_score = 0.0
    
    # Check for specific evidence indicators
    if state.experience_analysis:
        if state.experience_analysis.years_experience > 0:
            evidence_score += 0.2
        if state.experience_analysis.relevant_roles:
            evidence_score += 0.2
        if state.experience_analysis.strengths:
            evidence_score += 0.1
    
    if state.skills_analysis:
        if state.skills_analysis.technical_skills:
            evidence_score += 0.2
        if state.skills_analysis.proficiency_levels:
            evidence_score += 0.1
        if state.skills_analysis.soft_skills:
            evidence_score += 0.1
    
    if state.education_analysis:
        if state.education_analysis.degrees:
            evidence_score += 0.1
        if state.education_analysis.relevance_score > 50:
            evidence_score += 0.1
    
    confidence_factors.append(min(evidence_score, 1.0))
    
    # Calculate overall confidence
    overall_confidence = sum(confidence_factors) / len(confidence_factors)
    return min(max(overall_confidence, 0.0), 1.0)


def _identify_risk_factors(state: CVAnalysisState, result: SuitabilityScoreOutput) -> List[str]:
    """
    Identify risk factors that could impact candidate performance.
    
    Args:
        state: Current workflow state
        result: Scoring result
        
    Returns:
        List of identified risk factors
    """
    
    risk_factors = []
    
    # Experience-based risks
    if state.experience_analysis:
        years_exp = state.experience_analysis.years_experience
        relevant_roles = state.experience_analysis.relevant_roles or []
        
        if years_exp == 0:
            risk_factors.append("No professional experience - may require extensive onboarding")
        elif years_exp < 2:
            risk_factors.append("Limited professional experience - may need additional support")
        
        if not relevant_roles:
            risk_factors.append("No directly relevant role experience")
        elif len(relevant_roles) < 2:
            risk_factors.append("Limited relevant role experience")
    
    # Skills-based risks
    if state.skills_analysis:
        skill_gaps = state.skills_analysis.skill_gaps or []
        tech_skills = state.skills_analysis.technical_skills or []
        
        if len(skill_gaps) > 5:
            risk_factors.append("Multiple significant skill gaps identified")
        
        if len(tech_skills) < 3:
            risk_factors.append("Limited technical skill portfolio")
        
        # Check for critical skill gaps
        critical_gaps = [gap for gap in skill_gaps if any(keyword in gap.lower() 
                       for keyword in ['required', 'essential', 'critical', 'must have'])]
        if critical_gaps:
            risk_factors.append(f"Critical skill gaps: {', '.join(critical_gaps[:3])}")
    
    # Education-based risks
    if state.education_analysis:
        relevance_score = state.education_analysis.relevance_score or 0
        
        if relevance_score < 30:
            risk_factors.append("Educational background poorly aligned with role requirements")
        elif relevance_score < 50:
            risk_factors.append("Educational background has limited relevance to role")
    
    # Overall score risks
    if result.overall_score < 50:
        risk_factors.append("Overall suitability score indicates significant gaps")
    
    # Confidence-based risks
    if result.confidence_level < 0.5:
        risk_factors.append("Low confidence in assessment due to limited evidence")
    
    return risk_factors


def _calculate_potential_score(state: CVAnalysisState, result: SuitabilityScoreOutput) -> int:
    """
    Calculate potential score with development.
    
    Args:
        state: Current workflow state
        result: Scoring result
        
    Returns:
        int: Potential score (0-100)
    """
    
    current_score = result.overall_score
    
    # Base potential is current score + development potential
    potential_bonus = 0
    
    # Experience potential
    if state.experience_analysis:
        years_exp = state.experience_analysis.years_experience
        if years_exp < 3:
            potential_bonus += 10  # Room for growth
        elif years_exp < 5:
            potential_bonus += 5   # Some room for growth
    
    # Skills potential
    if state.skills_analysis:
        skill_gaps = state.skills_analysis.skill_gaps or []
        if len(skill_gaps) <= 3:
            potential_bonus += 8   # Few gaps to address
        elif len(skill_gaps) <= 6:
            potential_bonus += 5   # Moderate gaps
    
    # Education potential
    if state.education_analysis:
        relevance_score = state.education_analysis.relevance_score or 0
        if relevance_score >= 60:
            potential_bonus += 5   # Good educational foundation
        elif relevance_score >= 40:
            potential_bonus += 3   # Some educational relevance
    
    # Cap potential score at 100
    potential_score = min(current_score + potential_bonus, 100)
    
    return potential_score


def _estimate_development_timeline(state: CVAnalysisState, result: SuitabilityScoreOutput) -> str:
    """
    Estimate development timeline based on gaps and candidate profile.
    
    Args:
        state: Current workflow state
        result: Scoring result
        
    Returns:
        str: Development timeline estimate
    """
    
    if result.overall_score >= 80:
        return "Immediate readiness - minimal development needed"
    
    # Calculate development complexity
    complexity_factors = 0
    
    # Experience factors
    if state.experience_analysis:
        years_exp = state.experience_analysis.years_experience
        if years_exp == 0:
            complexity_factors += 3
        elif years_exp < 2:
            complexity_factors += 2
        elif years_exp < 5:
            complexity_factors += 1
    
    # Skills factors
    if state.skills_analysis:
        skill_gaps = state.skills_analysis.skill_gaps or []
        if len(skill_gaps) > 5:
            complexity_factors += 2
        elif len(skill_gaps) > 3:
            complexity_factors += 1
    
    # Education factors
    if state.education_analysis:
        relevance_score = state.education_analysis.relevance_score or 0
        if relevance_score < 30:
            complexity_factors += 2
        elif relevance_score < 50:
            complexity_factors += 1
    
    # Determine timeline based on complexity
    if complexity_factors <= 2:
        return "3-6 months with focused development"
    elif complexity_factors <= 4:
        return "6-12 months with structured development plan"
    elif complexity_factors <= 6:
        return "12-18 months with comprehensive development program"
    else:
        return "18+ months with extensive training and mentorship"


def _prepare_analysis_data_for_bias_detection(state: CVAnalysisState, result: SuitabilityScoreOutput) -> Dict[str, Any]:
    """
    Prepare analysis data for bias detection.
    
    Args:
        state: Current workflow state
        result: Scoring result
        
    Returns:
        Dictionary containing analysis data for bias detection
    """
    
    analysis_data = {
        "candidate": {
            "name": state.candidate_info.name if state.candidate_info else "Unknown",
            "contact_info": state.candidate_info.contact_info if state.candidate_info else "Unknown",
            "location": state.candidate_info.location if state.candidate_info else "Unknown"
        },
        "experience": {
            "summary": state.experience_analysis.summary if state.experience_analysis else "",
            "years_experience": state.experience_analysis.years_experience if state.experience_analysis else 0,
            "relevant_roles": state.experience_analysis.relevant_roles if state.experience_analysis else [],
            "career_progression": state.experience_analysis.career_progression if state.experience_analysis else "",
            "strengths": state.experience_analysis.strengths if state.experience_analysis else [],
            "gaps": state.experience_analysis.gaps if state.experience_analysis else []
        },
        "skills": {
            "technical_skills": state.skills_analysis.technical_skills if state.skills_analysis else [],
            "soft_skills": state.skills_analysis.soft_skills if state.skills_analysis else [],
            "skill_gaps": state.skills_analysis.skill_gaps if state.skills_analysis else [],
            "proficiency_levels": state.skills_analysis.proficiency_levels if state.skills_analysis else {}
        },
        "education": {
            "degrees": state.education_analysis.degrees if state.education_analysis else [],
            "institutions": state.education_analysis.institutions if state.education_analysis else [],
            "relevance_score": state.education_analysis.relevance_score if state.education_analysis else 0,
            "additional_training": state.education_analysis.additional_training if state.education_analysis else []
        },
        "certifications": {
            "certifications": state.certifications_analysis.certifications if state.certifications_analysis else [],
            "professional_memberships": state.certifications_analysis.professional_memberships if state.certifications_analysis else [],
            "achievements": state.certifications_analysis.achievements if state.certifications_analysis else [],
            "relevance_assessment": state.certifications_analysis.relevance_assessment if state.certifications_analysis else ""
        },
        "scoring_details": {
            "overall_score": result.overall_score,
            "component_scores": {
                "experience": result.experience_score,
                "skills": result.skills_score,
                "education": result.education_score,
                "certifications": result.certifications_score,
                "overall_fit": result.overall_fit_score
            },
            "justification": result.justification,
            "strengths": result.strengths,
            "weaknesses": result.weaknesses
        }
    }
    
    return analysis_data


def _apply_bias_mitigation_to_result(result: SuitabilityScoreOutput, mitigated_data: Dict[str, Any]) -> SuitabilityScoreOutput:
    """
    Apply bias mitigation to the scoring result.
    
    Args:
        result: Original scoring result
        mitigated_data: Bias-mitigated analysis data
        
    Returns:
        Updated scoring result with bias mitigation applied
    """
    
    # Update justification with bias-mitigated content
    if "scoring_details" in mitigated_data:
        scoring_details = mitigated_data["scoring_details"]
        if "justification" in scoring_details:
            result.justification = scoring_details["justification"]
        
        # Update component scores if they were adjusted
        if "component_scores" in scoring_details:
            component_scores = scoring_details["component_scores"]
            result.experience_score = component_scores.get("experience", result.experience_score)
            result.skills_score = component_scores.get("skills", result.skills_score)
            result.education_score = component_scores.get("education", result.education_score)
            result.certifications_score = component_scores.get("certifications", result.certifications_score)
            result.overall_fit_score = component_scores.get("overall_fit", result.overall_fit_score)
            
            # Recalculate overall score with mitigated component scores
            result.overall_score = int(round(
                result.experience_score * 0.25 +
                result.skills_score * 0.3 +
                result.education_score * 0.2 +
                result.certifications_score * 0.15 +
                result.overall_fit_score * 0.1
            ))
    
    return result


def handle_suitability_scorer_errors(state: CVAnalysisState, error: Exception) -> Dict[str, Any]:
    """
    Handle errors that occur during suitability scoring.
    
    Args:
        state: Current workflow state
        error: Exception that occurred
        
    Returns:
        Dict containing updated state with error information
    """
    
    error_msg = f"Suitability Scorer failed: {str(error)}"
    logger.error(error_msg)
    
    return {
        "current_node": "suitability_scorer",
        "suitability_score": 0,
        "errors": state.errors + [error_msg]
    }
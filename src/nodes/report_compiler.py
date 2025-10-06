"""
Report Compiler Node for Nova HR Assistant.

This module contains the report compiler node that assembles final JSON reports
from all analysis results in the CV analysis workflow.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from ..models.state import CVAnalysisState
from ..models.candidate import (
    CandidateInfo,
    ExperienceAnalysis,
    SkillsAnalysis,
    EducationAnalysis,
    CertificationAnalysis
)
from ..utils.output_validation import (
    validate_complete_report,
    format_report_with_validation,
    create_fallback_report,
    ReportFormatter,
    AnalysisCompleteness
)


def report_compiler_node(state: CVAnalysisState) -> CVAnalysisState:
    """
    Compile final JSON report from all analysis results.
    
    This node assembles all the analysis results from previous nodes into a
    structured JSON report that follows the required output format.
    
    Args:
        state: Current workflow state containing all analysis results
        
    Returns:
        Updated state with final_report populated
    """
    state.set_current_node("report_compiler")
    
    try:
        # Validate that all required analysis components are present
        validation_errors = validate_analysis_completeness(state)
        if validation_errors:
            # Log validation errors but don't fail completely - try to compile what we have
            for error in validation_errors:
                state.add_error(f"Validation warning: {error}")
            # Continue with compilation using available data
        
        # Compile the final report
        final_report = compile_json_report(state)
        
        # Validate the compiled report using comprehensive validation
        comprehensive_errors = validate_complete_report(final_report)
        if comprehensive_errors:
            state.add_error(f"Report validation failed: {'; '.join(comprehensive_errors)}")
            return state
        
        # Store the final report in state
        state.final_report = final_report
        
    except Exception as e:
        state.add_error(f"Report compilation error: {str(e)}")
    
    return state


def compile_json_report(state: CVAnalysisState) -> Dict[str, Any]:
    """
    Compile all analysis results into a structured JSON report.
    
    Args:
        state: Current workflow state with analysis results
        
    Returns:
        Dictionary representing the final JSON report
    """
    # Get current timestamp
    timestamp = datetime.now().isoformat()
    
    # Extract comprehensive candidate information
    candidate_name = state.candidate_info.name if state.candidate_info else "Unknown"
    schools = extract_schools_info(state.education_analysis)
    pros = extract_overall_strengths(state)
    cons = extract_overall_gaps(state)
    flaws = extract_critical_flaws(state)
    
    # Build the report structure matching the expected schema
    report = {
        "metadata": {
            "analysis_timestamp": timestamp,
            "position": state.position,
            "processing_node": state.current_node,
            "completion_percentage": state.get_completion_percentage()
        },
        "candidate": {
            "name": candidate_name,
            "contact_info": state.candidate_info.contact_info if state.candidate_info else "Contact information unavailable",
            "location": state.candidate_info.location if state.candidate_info else None
        },
        "analysis": {
            "experience": transform_experience_analysis(state.experience_analysis),
            "skills": transform_skills_analysis(state.skills_analysis),
            "education": transform_education_analysis(state.education_analysis),
            "certifications": transform_certification_analysis(state.certifications_analysis)
        },
        "evaluation": {
            "suitability_score": state.suitability_score,
            "recommendation": generate_recommendation(state.suitability_score),
            "strengths": pros,
            "areas_for_improvement": cons,
            "detailed_strengths_analysis": generate_detailed_strengths_narrative(state),
            "detailed_weaknesses_analysis": generate_detailed_weaknesses_narrative(state),
            "comprehensive_candidate_profile": generate_comprehensive_candidate_profile(state),
            "scoring_details": state.final_report.get("scoring_details", {})
        },
        "summary": {
            "overall_assessment": generate_overall_assessment(state),
            "key_highlights": extract_key_highlights(state),
            "decision_factors": extract_decision_factors(state)
        }
    }
    
    return report


def transform_candidate_info(candidate_info: CandidateInfo) -> Dict[str, Any]:
    """Transform CandidateInfo to dictionary format."""
    if not candidate_info:
        return {}
    
    return {
        "name": candidate_info.name,
        "contact_info": candidate_info.contact_info,
        "location": candidate_info.location
    }


def transform_experience_analysis(experience: ExperienceAnalysis) -> Dict[str, Any]:
    """Transform ExperienceAnalysis to dictionary format."""
    if not experience:
        return {
            "summary": "No professional experience analysis available - candidate may be entry-level or career-changing",
            "years_experience": 0,
            "relevant_roles": [],
            "career_progression": "No professional career progression identified",
            "strengths": ["Entry-level candidate with potential for growth"],
            "gaps": ["Limited professional experience"]
        }
    
    return {
        "summary": experience.summary or "No detailed summary available",
        "years_experience": experience.years_experience or 0,
        "relevant_roles": experience.relevant_roles or [],
        "career_progression": experience.career_progression or "No clear career progression identified",
        "strengths": experience.strengths or ["Potential for growth and development"],
        "gaps": experience.gaps or ["Limited professional experience" if experience.years_experience == 0 else "No significant gaps identified"]
    }


def transform_skills_analysis(skills: SkillsAnalysis) -> Dict[str, Any]:
    """Transform SkillsAnalysis to dictionary format."""
    if not skills:
        return {
            "technical_skills": [],
            "soft_skills": [],
            "skill_gaps": [],
            "proficiency_levels": {}
        }
    
    return {
        "technical_skills": skills.technical_skills or [],
        "soft_skills": skills.soft_skills or [],
        "skill_gaps": skills.skill_gaps or [],
        "proficiency_levels": skills.proficiency_levels or {}
    }


def transform_education_analysis(education: EducationAnalysis) -> Dict[str, Any]:
    """Transform EducationAnalysis to dictionary format."""
    if not education:
        return {
            "degrees": [],
            "institutions": [],
            "relevance_score": 0,
            "additional_training": []
        }
    
    return {
        "degrees": education.degrees or [],
        "institutions": education.institutions or [],
        "relevance_score": education.relevance_score or 0,
        "additional_training": education.additional_training or []
    }


def transform_certification_analysis(certifications: CertificationAnalysis) -> Dict[str, Any]:
    """Transform CertificationAnalysis to dictionary format."""
    if not certifications:
        return {
            "certifications": [],
            "professional_memberships": [],
            "achievements": [],
            "relevance_assessment": ""
        }
    
    return {
        "certifications": certifications.certifications or [],
        "professional_memberships": certifications.professional_memberships or [],
        "achievements": certifications.achievements or [],
        "relevance_assessment": certifications.relevance_assessment or ""
    }


def generate_recommendation(suitability_score: int) -> str:
    """Generate hiring recommendation based on suitability score."""
    if suitability_score >= 80:
        return "Highly Recommended - Strong candidate for the position"
    elif suitability_score >= 60:
        return "Recommended - Good candidate with minor gaps"
    elif suitability_score >= 40:
        return "Consider with Reservations - Moderate fit, requires further evaluation"
    else:
        return "Not Recommended - Significant gaps for this position"


def extract_overall_strengths(state: CVAnalysisState) -> List[str]:
    """Extract overall candidate strengths from all analysis results."""
    strengths = []
    
    if state.experience_analysis and state.experience_analysis.strengths:
        strengths.extend(state.experience_analysis.strengths)
    
    if state.skills_analysis and state.skills_analysis.technical_skills:
        # Add top technical skills as strengths
        top_skills = state.skills_analysis.technical_skills[:3]
        strengths.extend([f"Strong {skill} skills" for skill in top_skills])
    
    if state.education_analysis and state.education_analysis.relevance_score >= 70:
        strengths.append("Relevant educational background")
    
    if state.certifications_analysis and state.certifications_analysis.certifications:
        if len(state.certifications_analysis.certifications) > 0:
            strengths.append("Professional certifications demonstrate commitment to continuous learning")
    
    return list(set(strengths))  # Remove duplicates


def extract_schools_info(education_analysis: EducationAnalysis) -> List[str]:
    """Extract school/institution information from education analysis."""
    if not education_analysis or not education_analysis.institutions:
        return []
    
    return education_analysis.institutions


def extract_overall_gaps(state: CVAnalysisState) -> List[str]:
    """Extract overall areas for improvement from all analysis results."""
    gaps = []
    
    if state.experience_analysis and state.experience_analysis.gaps:
        gaps.extend(state.experience_analysis.gaps)
    
    if state.skills_analysis and state.skills_analysis.skill_gaps:
        gaps.extend(state.skills_analysis.skill_gaps)
    
    if state.education_analysis and state.education_analysis.relevance_score < 50:
        gaps.append("Educational background may not be directly relevant to the position")
    
    return list(set(gaps))  # Remove duplicates


def generate_overall_assessment(state: CVAnalysisState) -> str:
    """Generate overall assessment summary."""
    score = state.suitability_score
    
    if score >= 80:
        return f"Excellent candidate with {score}% suitability. Strong alignment with position requirements."
    elif score >= 65:
        return f"Good candidate with {score}% suitability. Minor gaps that can be addressed through training."
    elif score >= 50:
        return f"Moderate candidate with {score}% suitability. Requires careful consideration of gaps vs. potential."
    else:
        return f"Limited fit with {score}% suitability. Significant development would be required."


def extract_key_highlights(state: CVAnalysisState) -> List[str]:
    """Extract key highlights from the candidate profile."""
    highlights = []
    
    if state.experience_analysis:
        if state.experience_analysis.years_experience > 5:
            highlights.append(f"{state.experience_analysis.years_experience} years of professional experience")
        
        if state.experience_analysis.relevant_roles:
            highlights.append(f"Experience in {len(state.experience_analysis.relevant_roles)} relevant roles")
    
    if state.skills_analysis:
        tech_skills_count = len(state.skills_analysis.technical_skills)
        if tech_skills_count > 5:
            highlights.append(f"Proficient in {tech_skills_count} technical skills")
    
    if state.education_analysis and state.education_analysis.degrees:
        highlights.append(f"Holds {len(state.education_analysis.degrees)} degree(s)")
    
    if state.certifications_analysis and state.certifications_analysis.certifications:
        cert_count = len(state.certifications_analysis.certifications)
        if cert_count > 0:
            highlights.append(f"{cert_count} professional certification(s)")
    
    return highlights


def extract_decision_factors(state: CVAnalysisState) -> List[str]:
    """Extract key factors that influenced the suitability decision."""
    factors = []
    
    # Experience factors
    if state.experience_analysis:
        if state.experience_analysis.years_experience >= 3:
            factors.append("Sufficient professional experience")
        if state.experience_analysis.relevant_roles:
            factors.append("Relevant role experience")
    
    # Skills factors
    if state.skills_analysis:
        if len(state.skills_analysis.technical_skills) >= 5:
            factors.append("Strong technical skill set")
        if state.skills_analysis.skill_gaps:
            factors.append("Some skill gaps identified")
    
    # Education factors
    if state.education_analysis:
        if state.education_analysis.relevance_score >= 70:
            factors.append("Relevant educational background")
        elif state.education_analysis.relevance_score < 50:
            factors.append("Limited educational relevance")
    
    return factors


def extract_critical_flaws(state: CVAnalysisState) -> List[str]:
    """Extract critical flaws that could disqualify the candidate."""
    flaws = []
    
    # Experience-based flaws
    if state.experience_analysis:
        if state.experience_analysis.years_experience < 1:
            flaws.append("Insufficient professional experience")
        if not state.experience_analysis.relevant_roles:
            flaws.append("No relevant role experience")
    
    # Skills-based flaws
    if state.skills_analysis:
        critical_gaps = [gap for gap in state.skills_analysis.skill_gaps 
                        if any(keyword in gap.lower() for keyword in ['required', 'essential', 'critical', 'must have'])]
        flaws.extend(critical_gaps)
    
    # Education-based flaws
    if state.education_analysis and state.education_analysis.relevance_score < 30:
        flaws.append("Educational background significantly misaligned with position requirements")
    
    return flaws


def assess_interview_readiness(state: CVAnalysisState) -> str:
    """Assess candidate's readiness for interview process."""
    score = state.suitability_score
    
    if score >= 75:
        return "Ready for technical and behavioral interviews"
    elif score >= 60:
        return "Ready for initial screening, may need preparation for technical rounds"
    elif score >= 45:
        return "Requires significant preparation before interview process"
    else:
        return "Not recommended for interview at this time"


def generate_development_recommendations(state: CVAnalysisState) -> List[str]:
    """Generate recommendations for candidate development."""
    recommendations = []
    
    # Skills development
    if state.skills_analysis and state.skills_analysis.skill_gaps:
        for gap in state.skills_analysis.skill_gaps[:3]:  # Top 3 gaps
            recommendations.append(f"Develop skills in: {gap}")
    
    # Experience development
    if state.experience_analysis:
        if state.experience_analysis.years_experience < 3:
            recommendations.append("Gain more hands-on experience in relevant roles")
        if state.experience_analysis.gaps:
            for gap in state.experience_analysis.gaps[:2]:  # Top 2 gaps
                recommendations.append(f"Address experience gap: {gap}")
    
    # Education/Certification development
    if state.certifications_analysis:
        if not state.certifications_analysis.certifications:
            recommendations.append("Consider obtaining relevant professional certifications")
    
    return recommendations


def generate_executive_summary(state: CVAnalysisState, candidate_name: str) -> str:
    """Generate executive summary for the candidate."""
    score = state.suitability_score
    experience_years = state.experience_analysis.years_experience if state.experience_analysis else 0
    
    summary = f"{candidate_name} is a candidate with {experience_years} years of experience "
    
    if score >= 80:
        summary += f"who demonstrates excellent alignment ({score}%) with the position requirements. "
        summary += "Strong technical skills and relevant experience make this candidate highly suitable."
    elif score >= 65:
        summary += f"who shows good alignment ({score}%) with the position requirements. "
        summary += "Minor skill gaps can be addressed through targeted training."
    elif score >= 50:
        summary += f"who shows moderate alignment ({score}%) with the position requirements. "
        summary += "Significant gaps exist but candidate shows potential for growth."
    else:
        summary += f"who shows limited alignment ({score}%) with the position requirements. "
        summary += "Substantial development would be required for success in this role."
    
    return summary


def generate_detailed_breakdown(state: CVAnalysisState) -> Dict[str, Any]:
    """Generate detailed breakdown of analysis results."""
    return {
        "experience_assessment": {
            "years": state.experience_analysis.years_experience if state.experience_analysis else 0,
            "relevant_roles": len(state.experience_analysis.relevant_roles) if state.experience_analysis and state.experience_analysis.relevant_roles else 0,
            "strengths_count": len(state.experience_analysis.strengths) if state.experience_analysis and state.experience_analysis.strengths else 0,
            "gaps_count": len(state.experience_analysis.gaps) if state.experience_analysis and state.experience_analysis.gaps else 0
        },
        "skills_assessment": {
            "technical_skills_count": len(state.skills_analysis.technical_skills) if state.skills_analysis and state.skills_analysis.technical_skills else 0,
            "soft_skills_count": len(state.skills_analysis.soft_skills) if state.skills_analysis and state.skills_analysis.soft_skills else 0,
            "skill_gaps_count": len(state.skills_analysis.skill_gaps) if state.skills_analysis and state.skills_analysis.skill_gaps else 0
        },
        "education_assessment": {
            "degrees_count": len(state.education_analysis.degrees) if state.education_analysis and state.education_analysis.degrees else 0,
            "relevance_score": state.education_analysis.relevance_score if state.education_analysis else 0,
            "institutions_count": len(state.education_analysis.institutions) if state.education_analysis and state.education_analysis.institutions else 0
        },
        "certification_assessment": {
            "certifications_count": len(state.certifications_analysis.certifications) if state.certifications_analysis and state.certifications_analysis.certifications else 0,
            "professional_memberships_count": len(state.certifications_analysis.professional_memberships) if state.certifications_analysis and state.certifications_analysis.professional_memberships else 0
        }
    }


def generate_final_verdict(state: CVAnalysisState) -> str:
    """Generate final hiring verdict with enhanced decision logic."""
    score = state.suitability_score
    
    # Get additional scoring details if available
    scoring_details = state.final_report.get("scoring_details", {})
    confidence_level = scoring_details.get("confidence_level", 0.5)
    risk_factors = scoring_details.get("risk_factors", [])
    potential_score = scoring_details.get("potential_score", score)
    
    # Enhanced decision logic considering confidence and risk factors
    if score >= 80:
        if confidence_level >= 0.7 and len(risk_factors) <= 2:
            return "HIRE - Excellent candidate, proceed with offer"
        elif confidence_level >= 0.5:
            return "HIRE - Strong candidate, proceed with interview process"
        else:
            return "CONSIDER - High potential but requires additional assessment"
    elif score >= 65:
        if confidence_level >= 0.6 and len(risk_factors) <= 3:
            return "HIRE - Good candidate, proceed with interview process"
        elif potential_score >= 75:
            return "CONSIDER - Good potential with development, proceed with detailed interview"
        else:
            return "CONSIDER - Moderate fit, requires careful evaluation"
    elif score >= 50:
        if potential_score >= 70 and len(risk_factors) <= 4:
            return "CONSIDER - Moderate fit with development potential, requires structured interview"
        else:
            return "CONSIDER - Limited fit, requires comprehensive evaluation"
    else:
        if potential_score >= 60 and len(risk_factors) <= 5:
            return "CONSIDER - Low current fit but high potential, requires extensive development plan"
        else:
            return "PASS - Insufficient alignment with position requirements"


def generate_next_steps(state: CVAnalysisState) -> List[str]:
    """Generate recommended next steps based on enhanced analysis."""
    score = state.suitability_score
    
    # Get additional scoring details if available
    scoring_details = state.final_report.get("scoring_details", {})
    confidence_level = scoring_details.get("confidence_level", 0.5)
    risk_factors = scoring_details.get("risk_factors", [])
    potential_score = scoring_details.get("potential_score", score)
    development_timeline = scoring_details.get("development_timeline", "")
    
    steps = []
    
    # Enhanced decision logic with confidence and risk considerations
    if score >= 75:
        if confidence_level >= 0.7 and len(risk_factors) <= 2:
            steps.extend([
                "Schedule technical interview with senior team members",
                "Prepare behavioral interview questions focused on leadership",
                "Conduct reference checks with previous managers",
                "Prepare competitive offer package",
                "Plan onboarding and integration strategy"
            ])
        else:
            steps.extend([
                "Schedule comprehensive technical interview",
                "Conduct additional reference checks",
                "Review portfolio and work samples in detail",
                "Assess cultural fit through team interviews",
                "Prepare structured interview questions addressing identified concerns"
            ])
    elif score >= 60:
        if confidence_level >= 0.6 and len(risk_factors) <= 3:
            steps.extend([
                "Conduct initial phone screening with hiring manager",
                "Schedule technical assessment with practical exercises",
                "Review portfolio and work samples",
                "Assess cultural fit through team interviews",
                "Evaluate development plan feasibility"
            ])
        elif potential_score >= 75:
            steps.extend([
                "Conduct detailed phone screening focusing on growth mindset",
                "Schedule technical assessment with learning potential evaluation",
                "Review educational background and continuous learning indicators",
                "Assess willingness to address skill gaps",
                "Develop structured development plan with timeline"
            ])
        else:
            steps.extend([
                "Conduct comprehensive phone screening",
                "Schedule technical assessment with gap analysis",
                "Review portfolio with focus on identified weaknesses",
                "Assess cultural fit and team dynamics",
                "Evaluate training and mentorship requirements"
            ])
    elif score >= 45:
        if potential_score >= 70 and len(risk_factors) <= 4:
            steps.extend([
                "Conduct detailed phone screening with development focus",
                "Schedule technical assessment with learning potential evaluation",
                "Assess willingness to address skill gaps and development timeline",
                "Review educational background and continuous learning indicators",
                "Consider for junior or training positions with structured development",
                "Evaluate mentorship and training resource requirements"
            ])
        else:
            steps.extend([
                "Conduct comprehensive phone screening with gap analysis",
                "Schedule technical assessment focusing on critical skills",
                "Review portfolio with detailed gap analysis",
                "Assess cultural fit and team integration potential",
                "Evaluate comprehensive development plan requirements",
                "Consider alternative roles or future opportunities"
            ])
    else:
        if potential_score >= 60 and len(risk_factors) <= 5:
            steps.extend([
                "Conduct detailed phone screening with development potential focus",
                "Assess willingness to address significant skill gaps",
                "Review educational background and learning indicators",
                "Evaluate comprehensive development plan with extended timeline",
                "Consider for junior roles or extensive training programs",
                "Assess mentorship and training resource availability"
            ])
        else:
            steps.extend([
                "Send polite rejection email with constructive feedback",
                "Keep profile for future opportunities that may be a better fit",
                "Provide specific feedback on areas for improvement if requested",
                "Consider for different roles or future positions",
                "Maintain professional relationship for future opportunities"
            ])
    
    # Add development-specific steps if timeline is provided
    if development_timeline and score < 80:
        steps.append(f"Develop structured development plan with timeline: {development_timeline}")
    
    # Add risk mitigation steps if significant risks identified
    if len(risk_factors) > 3:
        steps.append("Develop risk mitigation plan for identified concerns")
    
    return steps


def generate_detailed_strengths_narrative(state: CVAnalysisState) -> str:
    """Generate a comprehensive narrative analysis of candidate strengths."""
    candidate_name = state.candidate_info.name if state.candidate_info else "The candidate"
    
    # Build comprehensive strengths narrative
    narrative_parts = []
    
    # Experience strengths
    if state.experience_analysis:
        exp_years = state.experience_analysis.years_experience
        relevant_roles = state.experience_analysis.relevant_roles or []
        exp_strengths = state.experience_analysis.strengths or []
        
        if exp_years > 0 or relevant_roles:
            exp_narrative = f"**Professional Experience Excellence:** {candidate_name} brings "
            if exp_years > 5:
                exp_narrative += f"substantial professional experience with {exp_years} years in the field, demonstrating career longevity and sustained professional growth. "
            elif exp_years > 0:
                exp_narrative += f"{exp_years} years of professional experience, showing solid foundation-building in their career trajectory. "
            else:
                exp_narrative += "emerging professional experience through meaningful roles and projects. "
            
            if relevant_roles:
                exp_narrative += f"Their background includes {len(relevant_roles)} directly relevant positions: {', '.join(relevant_roles[:3])}{'...' if len(relevant_roles) > 3 else ''}. "
            
            if exp_strengths:
                exp_narrative += "Key experiential strengths include: " + "; ".join(exp_strengths[:4]) + ". "
                if len(exp_strengths) > 4:
                    exp_narrative += f"Additionally, they demonstrate {len(exp_strengths) - 4} other notable professional competencies. "
            
            narrative_parts.append(exp_narrative)
    
    # Technical skills strengths
    if state.skills_analysis:
        tech_skills = state.skills_analysis.technical_skills or []
        soft_skills = state.skills_analysis.soft_skills or []
        proficiency = state.skills_analysis.proficiency_levels or {}
        
        if tech_skills:
            tech_narrative = f"**Technical Proficiency & Innovation:** {candidate_name} demonstrates impressive technical breadth with expertise across {len(tech_skills)} key technologies. "
            
            # Categorize skills by proficiency
            expert_skills = [skill for skill, level in proficiency.items() if level == 'expert']
            advanced_skills = [skill for skill, level in proficiency.items() if level == 'advanced']
            intermediate_skills = [skill for skill, level in proficiency.items() if level == 'intermediate']
            
            if expert_skills:
                tech_narrative += f"They exhibit expert-level mastery in {', '.join(expert_skills[:3])}{'...' if len(expert_skills) > 3 else ''}, "
                tech_narrative += "indicating deep technical understanding and ability to mentor others in these areas. "
            
            if advanced_skills:
                tech_narrative += f"Advanced proficiency in {', '.join(advanced_skills[:4])}{'...' if len(advanced_skills) > 4 else ''} "
                tech_narrative += "demonstrates their ability to handle complex technical challenges independently. "
            
            if intermediate_skills:
                tech_narrative += f"Solid intermediate skills in {', '.join(intermediate_skills[:3])}{'...' if len(intermediate_skills) > 3 else ''} "
                tech_narrative += "show continuous learning and adaptability to new technologies. "
            
            # Highlight top technical skills
            top_skills = tech_skills[:5]
            tech_narrative += f"Core technical competencies include {', '.join(top_skills)}, "
            tech_narrative += "providing a strong foundation for tackling diverse technical requirements. "
            
            narrative_parts.append(tech_narrative)
        
        if soft_skills:
            soft_narrative = f"**Leadership & Interpersonal Excellence:** Beyond technical capabilities, {candidate_name} brings strong interpersonal and leadership qualities. "
            soft_narrative += f"They demonstrate {', '.join(soft_skills[:4])}{'...' if len(soft_skills) > 4 else ''}, "
            soft_narrative += "essential skills for collaborative environments and team success. "
            
            # Highlight leadership and communication skills specifically
            leadership_skills = [skill for skill in soft_skills if any(keyword in skill.lower() for keyword in ['leadership', 'lead', 'manage', 'mentor'])]
            communication_skills = [skill for skill in soft_skills if any(keyword in skill.lower() for keyword in ['communication', 'present', 'collaborate', 'team'])]
            
            if leadership_skills:
                soft_narrative += f"Particularly noteworthy are their leadership capabilities in {', '.join(leadership_skills)}, "
                soft_narrative += "indicating potential for senior roles and team management responsibilities. "
            
            if communication_skills:
                soft_narrative += f"Strong communication and collaboration skills ({', '.join(communication_skills)}) "
                soft_narrative += "ensure effective stakeholder engagement and cross-functional team success. "
            
            narrative_parts.append(soft_narrative)
    
    # Educational strengths
    if state.education_analysis:
        degrees = state.education_analysis.degrees or []
        institutions = state.education_analysis.institutions or []
        relevance_score = state.education_analysis.relevance_score or 0
        additional_training = state.education_analysis.additional_training or []
        
        if degrees or relevance_score >= 60:
            edu_narrative = f"**Educational Foundation & Continuous Learning:** {candidate_name}'s educational background provides "
            
            if relevance_score >= 80:
                edu_narrative += f"exceptional alignment with the role requirements (relevance score: {relevance_score}/100). "
            elif relevance_score >= 60:
                edu_narrative += f"strong alignment with the role requirements (relevance score: {relevance_score}/100). "
            else:
                edu_narrative += f"foundational knowledge relevant to the position (relevance score: {relevance_score}/100). "
            
            if degrees:
                edu_narrative += f"They hold {len(degrees)} degree(s): {', '.join(degrees)}, "
                if institutions:
                    edu_narrative += f"from respected institutions including {', '.join(institutions[:2])}{'...' if len(institutions) > 2 else ''}. "
                else:
                    edu_narrative += "demonstrating formal academic achievement and structured learning. "
            
            if additional_training:
                edu_narrative += f"Beyond formal education, they have pursued {len(additional_training)} additional training programs: "
                edu_narrative += f"{', '.join(additional_training[:3])}{'...' if len(additional_training) > 3 else ''}, "
                edu_narrative += "showing commitment to continuous professional development. "
            
            narrative_parts.append(edu_narrative)
    
    # Certification and achievement strengths
    if state.certifications_analysis:
        certifications = state.certifications_analysis.certifications or []
        memberships = state.certifications_analysis.professional_memberships or []
        achievements = state.certifications_analysis.achievements or []
        
        if certifications or achievements or memberships:
            cert_narrative = f"**Professional Recognition & Achievements:** {candidate_name} demonstrates commitment to professional excellence through "
            
            if certifications:
                cert_narrative += f"{len(certifications)} professional certifications: {', '.join(certifications[:3])}{'...' if len(certifications) > 3 else ''}. "
                cert_narrative += "These credentials validate their expertise and commitment to industry standards. "
            
            if memberships:
                cert_narrative += f"Active participation in {len(memberships)} professional organizations ({', '.join(memberships[:2])}{'...' if len(memberships) > 2 else ''}) "
                cert_narrative += "demonstrates engagement with the professional community and commitment to staying current with industry trends. "
            
            if achievements:
                cert_narrative += f"Notable achievements include: {'. '.join(achievements[:3])}{'...' if len(achievements) > 3 else ''}. "
                cert_narrative += "These accomplishments showcase their ability to deliver tangible results and drive meaningful impact. "
            
            narrative_parts.append(cert_narrative)
    
    # Overall strengths synthesis
    if narrative_parts:
        synthesis = f"\n\n**Strengths Synthesis:** {candidate_name} presents a compelling profile combining "
        
        strength_areas = []
        if state.experience_analysis and (state.experience_analysis.years_experience > 0 or state.experience_analysis.relevant_roles):
            strength_areas.append("relevant professional experience")
        if state.skills_analysis and state.skills_analysis.technical_skills:
            strength_areas.append("strong technical capabilities")
        if state.skills_analysis and state.skills_analysis.soft_skills:
            strength_areas.append("excellent interpersonal skills")
        if state.education_analysis and state.education_analysis.relevance_score >= 60:
            strength_areas.append("solid educational foundation")
        if state.certifications_analysis and (state.certifications_analysis.certifications or state.certifications_analysis.achievements):
            strength_areas.append("professional recognition")
        
        if len(strength_areas) >= 3:
            synthesis += f"{', '.join(strength_areas[:-1])}, and {strength_areas[-1]}. "
        elif len(strength_areas) == 2:
            synthesis += f"{strength_areas[0]} and {strength_areas[1]}. "
        elif len(strength_areas) == 1:
            synthesis += f"{strength_areas[0]}. "
        
        synthesis += "This multifaceted profile positions them as a valuable contributor who can bring both technical expertise and collaborative leadership to drive organizational success."
        narrative_parts.append(synthesis)
    
    return "\n\n".join(narrative_parts) if narrative_parts else f"{candidate_name} demonstrates foundational capabilities with potential for growth and development in key areas relevant to the position."


def generate_detailed_weaknesses_narrative(state: CVAnalysisState) -> str:
    """Generate a comprehensive narrative analysis of candidate weaknesses and areas for improvement."""
    candidate_name = state.candidate_info.name if state.candidate_info else "The candidate"
    
    # Build comprehensive weaknesses narrative
    narrative_parts = []
    
    # Experience gaps
    if state.experience_analysis:
        exp_years = state.experience_analysis.years_experience
        exp_gaps = state.experience_analysis.gaps or []
        relevant_roles = state.experience_analysis.relevant_roles or []
        
        if exp_years < 3 or exp_gaps:
            exp_narrative = f"**Experience Development Opportunities:** "
            
            if exp_years == 0:
                exp_narrative += f"{candidate_name} is at the early stages of their professional journey with limited formal work experience. "
                exp_narrative += "While this presents opportunities for growth, it may require additional mentorship and structured onboarding to reach full productivity. "
            elif exp_years < 2:
                exp_narrative += f"With {exp_years} year(s) of professional experience, {candidate_name} is still building their professional foundation. "
                exp_narrative += "They may need additional support in complex project management and independent decision-making scenarios. "
            elif exp_years < 5:
                exp_narrative += f"While {candidate_name} has {exp_years} years of experience, they may still be developing senior-level competencies. "
                exp_narrative += "Additional experience in leadership roles and strategic thinking would strengthen their profile. "
            
            if not relevant_roles:
                exp_narrative += "The absence of directly relevant role experience means they may require a longer adjustment period to understand industry-specific challenges and best practices. "
            elif len(relevant_roles) < 2:
                exp_narrative += "Limited exposure to diverse relevant roles may restrict their understanding of different approaches and methodologies within the field. "
            
            if exp_gaps:
                exp_narrative += f"Specific experience gaps include: {'; '.join(exp_gaps[:4])}{'...' if len(exp_gaps) > 4 else ''}. "
                exp_narrative += "Addressing these gaps through targeted projects or training would significantly enhance their effectiveness. "
            
            narrative_parts.append(exp_narrative)
    
    # Skills gaps
    if state.skills_analysis:
        skill_gaps = state.skills_analysis.skill_gaps or []
        tech_skills = state.skills_analysis.technical_skills or []
        proficiency = state.skills_analysis.proficiency_levels or {}
        
        if skill_gaps or len(tech_skills) < 5:
            skills_narrative = f"**Technical Skill Enhancement Needs:** "
            
            if skill_gaps:
                critical_gaps = [gap for gap in skill_gaps if any(keyword in gap.lower() for keyword in ['required', 'essential', 'critical', 'must have'])]
                moderate_gaps = [gap for gap in skill_gaps if gap not in critical_gaps]
                
                if critical_gaps:
                    skills_narrative += f"Critical skill gaps that require immediate attention include: {'; '.join(critical_gaps[:3])}{'...' if len(critical_gaps) > 3 else ''}. "
                    skills_narrative += "These deficiencies could significantly impact job performance and should be prioritized for development. "
                
                if moderate_gaps:
                    skills_narrative += f"Additional skill development opportunities exist in: {'; '.join(moderate_gaps[:4])}{'...' if len(moderate_gaps) > 4 else ''}. "
                    skills_narrative += "While not immediately critical, strengthening these areas would enhance overall effectiveness and career progression. "
            
            # Analyze proficiency levels for improvement areas
            beginner_skills = [skill for skill, level in proficiency.items() if level == 'beginner']
            if beginner_skills:
                skills_narrative += f"Several technical areas show beginner-level proficiency ({', '.join(beginner_skills[:3])}{'...' if len(beginner_skills) > 3 else ''}), "
                skills_narrative += "indicating opportunities for focused skill development and hands-on practice. "
            
            if len(tech_skills) < 5:
                skills_narrative += f"The current technical skill portfolio ({len(tech_skills)} skills) may benefit from expansion to meet the diverse requirements of modern technical roles. "
            
            narrative_parts.append(skills_narrative)
    
    # Educational limitations
    if state.education_analysis:
        relevance_score = state.education_analysis.relevance_score or 0
        degrees = state.education_analysis.degrees or []
        additional_training = state.education_analysis.additional_training or []
        
        if relevance_score < 60 or not degrees:
            edu_narrative = f"**Educational Alignment Considerations:** "
            
            if relevance_score < 30:
                edu_narrative += f"The educational background shows limited alignment with the position requirements (relevance score: {relevance_score}/100). "
                edu_narrative += "This significant gap may require substantial additional training or alternative learning pathways to build necessary foundational knowledge. "
            elif relevance_score < 60:
                edu_narrative += f"While the educational foundation provides some relevant knowledge (relevance score: {relevance_score}/100), "
                edu_narrative += "there are notable gaps that could benefit from targeted supplementary education or professional development programs. "
            
            if not degrees:
                edu_narrative += "The absence of formal degree credentials may limit opportunities in organizations that require specific educational qualifications. "
            
            if not additional_training:
                edu_narrative += "Limited evidence of continuous learning through additional training programs suggests opportunities for professional development and skill updating. "
            
            narrative_parts.append(edu_narrative)
    
    # Certification and professional development gaps
    if state.certifications_analysis:
        certifications = state.certifications_analysis.certifications or []
        memberships = state.certifications_analysis.professional_memberships or []
        
        if not certifications and not memberships:
            cert_narrative = f"**Professional Development & Credentialing Opportunities:** "
            cert_narrative += f"{candidate_name} currently lacks formal professional certifications and industry memberships, "
            cert_narrative += "which could limit their credibility in specialized technical areas and reduce networking opportunities within the professional community. "
            cert_narrative += "Pursuing relevant certifications would validate their expertise and demonstrate commitment to professional standards. "
            cert_narrative += "Joining professional organizations would provide access to industry insights, best practices, and career development resources. "
            
            narrative_parts.append(cert_narrative)
    
    # Suitability score analysis
    score = state.suitability_score
    if score < 70:
        score_narrative = f"**Overall Fit Assessment:** "
        
        if score < 40:
            score_narrative += f"The overall suitability score of {score}/100 indicates substantial gaps between the candidate's current profile and the position requirements. "
            score_narrative += "Significant investment in training, mentorship, and skill development would be necessary to achieve role readiness. "
            score_narrative += "Consider whether the organization has the resources and timeline to support this level of development. "
        elif score < 60:
            score_narrative += f"With a suitability score of {score}/100, {candidate_name} shows potential but requires considerable development to meet role expectations. "
            score_narrative += "A structured development plan with clear milestones and regular progress assessments would be essential for success. "
        else:
            score_narrative += f"The suitability score of {score}/100 suggests {candidate_name} has a solid foundation but needs targeted improvements in key areas. "
            score_narrative += "With focused development efforts, they could reach full role effectiveness within a reasonable timeframe. "
        
        narrative_parts.append(score_narrative)
    
    # Development recommendations synthesis
    if narrative_parts:
        synthesis = f"\n\n**Development Strategy Recommendations:** To address these areas for improvement, {candidate_name} would benefit from "
        
        development_areas = []
        if state.experience_analysis and (state.experience_analysis.years_experience < 3 or state.experience_analysis.gaps):
            development_areas.append("structured mentorship and progressive responsibility assignments")
        if state.skills_analysis and state.skills_analysis.skill_gaps:
            development_areas.append("targeted technical training and hands-on project experience")
        if state.education_analysis and state.education_analysis.relevance_score < 60:
            development_areas.append("supplementary education or professional development programs")
        if state.certifications_analysis and not state.certifications_analysis.certifications:
            development_areas.append("pursuit of relevant professional certifications")
        
        if len(development_areas) >= 3:
            synthesis += f"{', '.join(development_areas[:-1])}, and {development_areas[-1]}. "
        elif len(development_areas) == 2:
            synthesis += f"{development_areas[0]} and {development_areas[1]}. "
        elif len(development_areas) == 1:
            synthesis += f"{development_areas[0]}. "
        
        synthesis += "A comprehensive development plan addressing these areas would maximize their potential and accelerate their path to full role effectiveness."
        narrative_parts.append(synthesis)
    
    return "\n\n".join(narrative_parts) if narrative_parts else f"{candidate_name} demonstrates a strong profile with minimal areas requiring development, positioning them well for immediate contribution to the role."


def generate_comprehensive_candidate_profile(state: CVAnalysisState) -> str:
    """Generate a comprehensive narrative profile of the candidate combining all analysis aspects."""
    candidate_name = state.candidate_info.name if state.candidate_info else "The candidate"
    score = state.suitability_score
    
    # Professional summary
    profile = f"**Comprehensive Candidate Assessment for {candidate_name}**\n\n"
    
    # Executive overview
    profile += f"**Executive Overview:** {candidate_name} presents "
    
    if score >= 80:
        profile += f"an exceptional profile with {score}% alignment to the position requirements, demonstrating strong readiness for immediate contribution and potential for leadership growth. "
    elif score >= 65:
        profile += f"a strong profile with {score}% alignment to the position requirements, showing solid capabilities with minor development needs. "
    elif score >= 50:
        profile += f"a developing profile with {score}% alignment to the position requirements, indicating potential with targeted development opportunities. "
    else:
        profile += f"an emerging profile with {score}% alignment to the position requirements, requiring significant investment in development to reach full role effectiveness. "
    
    # Experience profile
    if state.experience_analysis:
        exp_years = state.experience_analysis.years_experience
        relevant_roles = state.experience_analysis.relevant_roles or []
        
        profile += f"\n\n**Professional Experience Profile:** "
        if exp_years > 5:
            profile += f"With {exp_years} years of professional experience, {candidate_name} brings seasoned expertise and proven track record. "
        elif exp_years > 0:
            profile += f"With {exp_years} years of professional experience, {candidate_name} demonstrates growing expertise and professional development. "
        else:
            profile += f"{candidate_name} is at the beginning of their professional journey, bringing fresh perspective and eagerness to learn. "
        
        if relevant_roles:
            profile += f"Their background includes {len(relevant_roles)} relevant positions, providing direct industry experience and contextual understanding. "
        
        career_progression = state.experience_analysis.career_progression
        if career_progression:
            profile += f"Career trajectory shows: {career_progression} "
    
    # Skills and competencies profile
    if state.skills_analysis:
        tech_skills = state.skills_analysis.technical_skills or []
        soft_skills = state.skills_analysis.soft_skills or []
        
        profile += f"\n\n**Skills & Competencies Profile:** {candidate_name} demonstrates "
        
        if len(tech_skills) > 10:
            profile += f"extensive technical breadth with {len(tech_skills)} technical competencies, "
        elif len(tech_skills) > 5:
            profile += f"solid technical foundation with {len(tech_skills)} technical competencies, "
        else:
            profile += f"focused technical expertise with {len(tech_skills)} core competencies, "
        
        if len(soft_skills) > 5:
            profile += f"complemented by strong interpersonal capabilities across {len(soft_skills)} soft skill areas. "
        else:
            profile += f"supported by developing interpersonal skills in {len(soft_skills)} key areas. "
        
        # Highlight key technical areas
        if tech_skills:
            profile += f"Core technical strengths include {', '.join(tech_skills[:5])}{'...' if len(tech_skills) > 5 else ''}. "
    
    # Educational and professional development profile
    if state.education_analysis:
        degrees = state.education_analysis.degrees or []
        relevance_score = state.education_analysis.relevance_score or 0
        
        profile += f"\n\n**Educational & Development Profile:** "
        
        if degrees:
            profile += f"{candidate_name} holds {len(degrees)} degree(s) ({', '.join(degrees)}), "
        
        if relevance_score >= 70:
            profile += f"with strong educational alignment to the role (relevance: {relevance_score}%). "
        elif relevance_score >= 50:
            profile += f"with moderate educational alignment to the role (relevance: {relevance_score}%). "
        else:
            profile += f"with foundational educational background (relevance: {relevance_score}%). "
    
    if state.certifications_analysis:
        certifications = state.certifications_analysis.certifications or []
        achievements = state.certifications_analysis.achievements or []
        
        if certifications:
            profile += f"Professional credentials include {len(certifications)} certifications, demonstrating commitment to industry standards. "
        
        if achievements:
            profile += f"Notable achievements encompass {len(achievements)} significant accomplishments, showcasing ability to deliver results. "
    
    # Fit assessment and recommendations
    profile += f"\n\n**Role Fit Assessment:** "
    
    if score >= 80:
        profile += f"Excellent alignment with position requirements. {candidate_name} is ready for immediate contribution with potential for expanded responsibilities. "
        profile += "Recommend proceeding with offer and discussing growth opportunities. "
    elif score >= 65:
        profile += f"Strong alignment with position requirements. {candidate_name} can contribute effectively with minimal onboarding. "
        profile += "Recommend proceeding with interview process and reference checks. "
    elif score >= 50:
        profile += f"Moderate alignment with position requirements. {candidate_name} shows potential but requires development planning. "
        profile += "Recommend detailed interview to assess growth mindset and development timeline. "
    else:
        profile += f"Limited current alignment with position requirements. {candidate_name} would require significant development investment. "
        profile += "Consider for junior roles or future opportunities with appropriate development support. "
    
    # Strategic value proposition
    profile += f"\n\n**Strategic Value Proposition:** "
    
    value_points = []
    if state.experience_analysis and state.experience_analysis.years_experience > 3:
        value_points.append("proven professional experience")
    if state.skills_analysis and len(state.skills_analysis.technical_skills) > 7:
        value_points.append("broad technical capabilities")
    if state.skills_analysis and any('leadership' in skill.lower() for skill in (state.skills_analysis.soft_skills or [])):
        value_points.append("leadership potential")
    if state.education_analysis and state.education_analysis.relevance_score >= 70:
        value_points.append("strong educational foundation")
    if state.certifications_analysis and state.certifications_analysis.certifications:
        value_points.append("professional credentialing")
    
    if value_points:
        profile += f"{candidate_name} brings {', '.join(value_points[:-1]) + ' and ' + value_points[-1] if len(value_points) > 1 else value_points[0]} to the organization. "
    
    profile += f"Their profile suggests {'immediate value addition' if score >= 70 else 'strong potential for growth' if score >= 50 else 'long-term development opportunity'} "
    profile += f"with {'minimal' if score >= 80 else 'moderate' if score >= 60 else 'significant'} investment in development and integration."
    
    return profile


def validate_analysis_completeness(state: CVAnalysisState) -> List[str]:
    """
    Validate that all required analysis components are present and valid.
    
    Args:
        state: Current workflow state
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Use the new validation utilities for comprehensive validation
    errors.extend(AnalysisCompleteness.validate_candidate_info(state.candidate_info))
    errors.extend(AnalysisCompleteness.validate_experience_analysis(state.experience_analysis))
    errors.extend(AnalysisCompleteness.validate_skills_analysis(state.skills_analysis))
    errors.extend(AnalysisCompleteness.validate_education_analysis(state.education_analysis))
    errors.extend(AnalysisCompleteness.validate_certification_analysis(state.certifications_analysis))
    errors.extend(AnalysisCompleteness.validate_evaluation_completeness(
        state.suitability_score
    ))
    
    return errors





def format_report_json(report: Dict[str, Any], indent: int = 2) -> str:
    """
    Format the report as a JSON string with proper indentation.
    
    Args:
        report: The report dictionary
        indent: Number of spaces for indentation
        
    Returns:
        Formatted JSON string
    """
    return ReportFormatter.format_json_report(report, indent=indent)


def validate_json_format(json_string: str) -> List[str]:
    """
    Validate that a string is valid JSON format.
    
    Args:
        json_string: The JSON string to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    try:
        json.loads(json_string)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {str(e)}")
    
    return errors


# Error handling utilities
def handle_report_compiler_errors(state: CVAnalysisState, error: Exception) -> CVAnalysisState:
    """
    Handle errors that occur during report compilation.
    
    Args:
        state: Current workflow state
        error: The exception that occurred
        
    Returns:
        Updated state with error information
    """
    error_message = f"Report compilation failed: {str(error)}"
    state.add_error(error_message)
    
    # Collect any partial data available
    partial_data = {}
    if state.candidate_info:
        partial_data["candidate"] = transform_candidate_info(state.candidate_info)
    if state.experience_analysis:
        partial_data["experience"] = transform_experience_analysis(state.experience_analysis)
    if state.skills_analysis:
        partial_data["skills"] = transform_skills_analysis(state.skills_analysis)
    if state.education_analysis:
        partial_data["education"] = transform_education_analysis(state.education_analysis)
    if state.certifications_analysis:
        partial_data["certifications"] = transform_certification_analysis(state.certifications_analysis)
    
    # Create error report using the new utility
    from ..utils.output_validation import ErrorReportGenerator
    error_report = ErrorReportGenerator.create_error_report(state.position, error_message, partial_data)
    
    state.final_report = error_report
    return state
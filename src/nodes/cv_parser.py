"""
CV Parser Node for Nova HR Assistant.

This module contains the CV parser node that extracts basic candidate information
from CV text using structured LLM prompts.
"""

import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.state import CVAnalysisState
from ..models.candidate import CandidateInfo
from ..utils.llm_manager import LLMManager, LLMProviderError


logger = logging.getLogger(__name__)


class CandidateInfoOutput(BaseModel):
    """Pydantic model for structured CV parser output."""
    
    name: str = Field(description="Full name of the candidate")
    contact_info: str = Field(description="Contact information (email, phone, etc.)")
    location: str = Field(default="", description="Location/address of the candidate")


class CVParserError(Exception):
    """Exception raised when CV parsing fails."""
    pass


def create_cv_parser_prompt() -> ChatPromptTemplate:
    """
    Create the structured prompt for CV parsing.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for CV parsing
    """
    
    system_message = """You are an expert HR assistant specialized in extracting candidate information from CVs.
Your task is to carefully analyze the provided CV text and extract the candidate's basic information.

IMPORTANT INSTRUCTIONS:
1. Extract the candidate's full name as it appears in the CV
2. Extract all available contact information (email, phone, LinkedIn, etc.)
3. Extract location information if available (city, country, address)
4. If information is not clearly available, use empty string for optional fields
5. Be precise and only extract information that is explicitly stated
6. For contact_info, combine all contact methods into a single string separated by commas

EXTRACTION GUIDELINES:
- Name: Look for the candidate's full name, usually at the top of the CV
- Contact Info: Include email, phone number, LinkedIn profile, website, etc.
- Location: Include city, state/province, country as available

{format_instructions}"""

    human_message = """Please extract the candidate's basic information from the following CV:

CV TEXT:
{cv_text}

Extract the information according to the specified format."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])


def cv_parser_node(state: CVAnalysisState, llm_manager: LLMManager) -> Dict[str, Any]:
    """
    CV Parser node function for LangGraph workflow.
    
    Extracts basic candidate information from CV text using structured LLM prompts.
    
    Args:
        state: Current workflow state containing CV text
        llm_manager: LLM manager instance for making LLM calls
        
    Returns:
        Dict containing updated state with candidate_info
        
    Raises:
        CVParserError: If CV parsing fails
    """
    
    logger.info("Starting CV parser node")
    
    # Update current node in state
    updated_state = {
        "current_node": "cv_parser",
        "errors": state.errors.copy()
    }
    
    try:
        # Validate input
        if not state.cv_text or not state.cv_text.strip():
            error_msg = "CV text is empty or missing"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise CVParserError(error_msg)
        
        # Get LLM instance
        llm = llm_manager.get_llm()
        
        # Create output parser
        output_parser = PydanticOutputParser(pydantic_object=CandidateInfoOutput)
        
        # Create prompt template
        prompt_template = create_cv_parser_prompt()
        
        # Create the chain
        chain = prompt_template | llm | output_parser
        
        # Execute the chain
        logger.debug(f"Processing CV text of length: {len(state.cv_text)}")
        
        result = chain.invoke({
            "cv_text": state.cv_text,
            "format_instructions": output_parser.get_format_instructions()
        })
        
        # Convert to CandidateInfo model
        candidate_info = CandidateInfo(
            name=result.name.strip(),
            contact_info=result.contact_info.strip(),
            location=result.location.strip() if result.location else None
        )
        
        # Validate the extracted information
        validation_errors = candidate_info.validate()
        if validation_errors:
            error_msg = f"Extracted candidate info validation failed: {', '.join(validation_errors)}"
            logger.error(error_msg)
            updated_state["errors"].append(error_msg)
            raise CVParserError(error_msg)
        
        # Update state with successful result
        updated_state["candidate_info"] = candidate_info
        
        logger.info(f"Successfully extracted candidate info for: {candidate_info.name}")
        logger.debug(f"Extracted contact info: {candidate_info.contact_info}")
        
        return updated_state
        
    except LLMProviderError as e:
        error_msg = f"LLM provider error in CV parser: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise CVParserError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error in CV parser: {str(e)}"
        logger.error(error_msg)
        updated_state["errors"].append(error_msg)
        raise CVParserError(error_msg)


def validate_cv_parser_output(candidate_info: CandidateInfo) -> bool:
    """
    Validate CV parser output for completeness and accuracy.
    
    Args:
        candidate_info: Extracted candidate information
        
    Returns:
        bool: True if output is valid, False otherwise
    """
    
    if not candidate_info:
        return False
    
    # Check basic validation
    if not candidate_info.is_valid():
        return False
    
    # Additional checks for CV parser specific requirements
    if len(candidate_info.name.strip()) < 2:
        return False
    
    # Check if contact info contains at least email or phone pattern
    contact_lower = candidate_info.contact_info.lower()
    has_email = '@' in contact_lower and '.' in contact_lower
    has_phone = any(char.isdigit() for char in candidate_info.contact_info)
    
    if not (has_email or has_phone):
        logger.warning("No email or phone number detected in contact info")
        return False
    
    return True


def handle_cv_parser_errors(state: CVAnalysisState, error: Exception) -> Dict[str, Any]:
    """
    Handle errors that occur during CV parsing.
    
    Args:
        state: Current workflow state
        error: Exception that occurred
        
    Returns:
        Dict containing updated state with error information
    """
    
    error_msg = f"CV Parser failed: {str(error)}"
    logger.error(error_msg)
    
    return {
        "current_node": "cv_parser",
        "candidate_info": None,
        "errors": state.errors + [error_msg]
    }
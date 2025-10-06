"""
NovaGraph - Main workflow orchestrator for Nova HR Assistant

This module contains the NovaGraph class that orchestrates the CV analysis workflow
using LangGraph for state management and node execution.
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from .models.state import CVAnalysisState
from .config.nova_config import NovaConfig
from .utils.llm_manager import LLMManager, LLMProviderError
from .utils.output_validation import create_fallback_report
from .nodes import (
    cv_parser_node,
    experience_analyzer_node,
    skills_analyzer_node,
    education_analyzer_node,
    certification_analyzer_node,
    suitability_scorer_node,
    report_compiler_node
)


logger = logging.getLogger(__name__)


class NovaGraphError(Exception):
    """Exception raised when NovaGraph operations fail"""
    pass


class NovaGraph:
    """
    Main workflow orchestrator using LangGraph for CV analysis pipeline.
    
    This class creates and manages the LangGraph workflow that processes
    candidate CVs through various analysis nodes to generate comprehensive
    evaluation reports.
    """
    
    def __init__(self, config: NovaConfig):
        """
        Initialize NovaGraph with configuration
        
        Args:
            config: NovaConfig instance with LLM and workflow settings
        """
        self.config = config
        self.llm_manager = LLMManager(config.llm)
        self._workflow: Optional[CompiledGraph] = None
        
        # Validate configuration
        self._validate_config()
        
        logger.info("NovaGraph initialized successfully")
    
    def create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow with all analysis nodes and edges
        
        Returns:
            StateGraph: Configured workflow graph
        """
        # Create the state graph
        workflow = StateGraph(CVAnalysisState)
        
        # Add all analysis nodes
        workflow.add_node("cv_parser", self._wrap_node(cv_parser_node))
        workflow.add_node("experience_analyzer", self._wrap_node(experience_analyzer_node))
        workflow.add_node("skills_analyzer", self._wrap_node(skills_analyzer_node))
        workflow.add_node("education_analyzer", self._wrap_node(education_analyzer_node))
        workflow.add_node("certification_analyzer", self._wrap_node(certification_analyzer_node))
        workflow.add_node("suitability_scorer", self._wrap_node(suitability_scorer_node))
        workflow.add_node("report_compiler", self._wrap_node(report_compiler_node))
        
        # Set entry point
        workflow.set_entry_point("cv_parser")
        
        # Add sequential edges for the main workflow
        workflow.add_edge("cv_parser", "experience_analyzer")
        workflow.add_edge("experience_analyzer", "skills_analyzer")
        workflow.add_edge("skills_analyzer", "education_analyzer")
        workflow.add_edge("education_analyzer", "certification_analyzer")
        workflow.add_edge("certification_analyzer", "suitability_scorer")
        
        # Final edges
        workflow.add_edge("suitability_scorer", "report_compiler")
        workflow.add_edge("report_compiler", END)
        
        logger.info("Workflow created with all nodes and edges")
        return workflow
    
    def get_compiled_workflow(self) -> CompiledGraph:
        """
        Get the compiled workflow, creating it if necessary
        
        Returns:
            CompiledGraph: Compiled workflow ready for execution
        """
        if self._workflow is None:
            workflow = self.create_workflow()
            self._workflow = workflow.compile()
            logger.info("Workflow compiled successfully")
        
        return self._workflow
    
    def process_candidate(self, cv_text: str, position: str, 
                         enable_recovery: bool = True) -> Dict[str, Any]:
        """
        Process a candidate CV through the complete analysis workflow
        
        Args:
            cv_text: Raw CV text content
            position: Job position description and requirements
            enable_recovery: Whether to enable error recovery mechanisms
            
        Returns:
            Dict containing the complete analysis report
            
        Raises:
            NovaGraphError: If processing fails and recovery is disabled
        """
        processing_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger.info(f"Starting CV analysis workflow [ID: {processing_id}]")
        
        try:
            # Create initial state with processing metadata
            initial_state = CVAnalysisState(
                cv_text=cv_text,
                position=position,
                processing_start_time=datetime.now().isoformat()
            )
            
            # Validate input with detailed logging
            input_errors = initial_state.validate_input()
            if input_errors:
                logger.error(f"Input validation failed [ID: {processing_id}]: {input_errors}")
                if not enable_recovery:
                    raise NovaGraphError(f"Invalid input: {'; '.join(input_errors)}")
                else:
                    # Create fallback report for invalid input
                    return self._create_error_report(
                        "INPUT_VALIDATION_ERROR",
                        f"Invalid input: {'; '.join(input_errors)}",
                        processing_id
                    )
            
            logger.info(f"Input validation passed [ID: {processing_id}]")
            logger.debug(f"CV text length: {len(cv_text)} characters")
            logger.debug(f"Position description length: {len(position)} characters")
            
            # Test LLM connectivity before processing
            if not self._test_llm_connectivity():
                error_msg = "LLM provider is not available"
                logger.error(f"LLM connectivity test failed [ID: {processing_id}]: {error_msg}")
                if not enable_recovery:
                    raise NovaGraphError(error_msg)
                else:
                    return self._create_error_report(
                        "LLM_CONNECTIVITY_ERROR",
                        error_msg,
                        processing_id
                    )
            
            # Get compiled workflow
            workflow = self._get_workflow_with_retry()
            
            # Execute workflow with monitoring
            logger.info(f"Executing workflow [ID: {processing_id}]")
            result = self._execute_workflow_with_monitoring(workflow, initial_state, processing_id)
            
            # Validate and process final result
            return self._process_workflow_result(result, processing_id, enable_recovery)
                
        except NovaGraphError:
            # Re-raise NovaGraphError as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error in workflow execution: {str(e)}"
            logger.error(f"Critical error [ID: {processing_id}]: {error_msg}")
            logger.debug(f"Full traceback [ID: {processing_id}]: {traceback.format_exc()}")
            
            if not enable_recovery:
                raise NovaGraphError(error_msg)
            else:
                return self._create_error_report(
                    "CRITICAL_ERROR",
                    error_msg,
                    processing_id
                )
    
    def _test_llm_connectivity(self) -> bool:
        """
        Test LLM connectivity before processing
        
        Returns:
            bool: True if LLM is accessible, False otherwise
        """
        try:
            return self.llm_manager.test_connection()
        except Exception as e:
            logger.warning(f"LLM connectivity test failed: {str(e)}")
            return False
    
    def _get_workflow_with_retry(self, max_retries: int = 2) -> CompiledGraph:
        """
        Get compiled workflow with retry logic
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            CompiledGraph: Compiled workflow
            
        Raises:
            NovaGraphError: If workflow compilation fails after retries
        """
        for attempt in range(max_retries + 1):
            try:
                return self.get_compiled_workflow()
            except Exception as e:
                if attempt == max_retries:
                    raise NovaGraphError(f"Workflow compilation failed after {max_retries} retries: {str(e)}")
                else:
                    logger.warning(f"Workflow compilation attempt {attempt + 1} failed, retrying: {str(e)}")
                    # Reset workflow to force recompilation
                    self.reset_workflow()
    
    def _execute_workflow_with_monitoring(self, workflow: CompiledGraph, 
                                        initial_state: CVAnalysisState, 
                                        processing_id: str) -> CVAnalysisState:
        """
        Execute workflow with comprehensive monitoring and error handling
        
        Args:
            workflow: Compiled workflow to execute
            initial_state: Initial state for workflow
            processing_id: Unique processing identifier
            
        Returns:
            CVAnalysisState: Final workflow state
            
        Raises:
            NovaGraphError: If workflow execution fails
        """
        try:
            start_time = datetime.now()
            logger.info(f"Workflow execution started [ID: {processing_id}]")
            
            # Execute workflow
            result = workflow.invoke(initial_state)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Workflow execution completed [ID: {processing_id}] in {execution_time:.2f} seconds")
            
            # Convert result back to CVAnalysisState if needed
            if not isinstance(result, CVAnalysisState):
                # LangGraph returns AddableValuesDict, convert back to CVAnalysisState
                logger.debug(f"Converting result from {type(result)} to CVAnalysisState")
                result = CVAnalysisState(
                    cv_text=result.get("cv_text", ""),
                    position=result.get("position", ""),
                    candidate_info=result.get("candidate_info"),
                    experience_analysis=result.get("experience_analysis"),
                    skills_analysis=result.get("skills_analysis"),
                    education_analysis=result.get("education_analysis"),
                    certifications_analysis=result.get("certifications_analysis"),
                    suitability_score=result.get("suitability_score", 0),
                    final_report=result.get("final_report", {}),
                    current_node=result.get("current_node", ""),
                    errors=result.get("errors", []),
                    processing_start_time=result.get("processing_start_time")
                )
            
            # Log workflow completion status
            completion_percentage = result.get_completion_percentage()
            logger.info(f"Workflow completion: {completion_percentage:.1f}% [ID: {processing_id}]")
            
            if result.has_errors():
                logger.warning(f"Workflow completed with {len(result.errors)} errors [ID: {processing_id}]: {result.errors}")
            
            return result
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(f"Workflow execution error [ID: {processing_id}]: {error_msg}")
            raise NovaGraphError(error_msg)
    
    def _process_workflow_result(self, result: CVAnalysisState, 
                               processing_id: str, 
                               enable_recovery: bool) -> Dict[str, Any]:
        """
        Process and validate workflow result
        
        Args:
            result: Final workflow state
            processing_id: Unique processing identifier
            enable_recovery: Whether to enable error recovery
            
        Returns:
            Dict: Final analysis report
            
        Raises:
            NovaGraphError: If result processing fails and recovery is disabled
        """
        try:
            # Check for critical errors that prevent report generation
            if result.has_errors() and not result.final_report:
                error_msg = f"Workflow failed with errors: {'; '.join(result.errors)}"
                logger.error(f"Critical workflow errors [ID: {processing_id}]: {error_msg}")
                
                if not enable_recovery:
                    raise NovaGraphError(error_msg)
                else:
                    return self._create_error_report(
                        "WORKFLOW_EXECUTION_ERROR",
                        error_msg,
                        processing_id,
                        partial_state=result
                    )
            
            # Validate analysis completeness
            missing_components = self._check_analysis_completeness(result)
            if missing_components:
                error_msg = f"Incomplete analysis - missing: {', '.join(missing_components)}"
                logger.warning(f"Analysis incomplete [ID: {processing_id}]: {error_msg}")
                
                if not enable_recovery:
                    raise NovaGraphError(error_msg)
                else:
                    # Try to create partial report
                    return self._create_partial_report(result, missing_components, processing_id)
            
            # Validate final report
            if result.final_report:
                logger.info(f"Analysis completed successfully [ID: {processing_id}]")
                return result.final_report
            else:
                logger.warning(f"Final report is empty, returning state dict [ID: {processing_id}]")
                return result.to_dict()
                
        except Exception as e:
            error_msg = f"Result processing failed: {str(e)}"
            logger.error(f"Result processing error [ID: {processing_id}]: {error_msg}")
            
            if not enable_recovery:
                raise NovaGraphError(error_msg)
            else:
                return self._create_error_report(
                    "RESULT_PROCESSING_ERROR",
                    error_msg,
                    processing_id
                )
    
    def _check_analysis_completeness(self, state: CVAnalysisState) -> List[str]:
        """
        Check which analysis components are missing
        
        Args:
            state: Workflow state to check
            
        Returns:
            List of missing component names
        """
        missing_components = []
        
        if not state.candidate_info:
            missing_components.append("candidate_info")
        if not state.experience_analysis:
            missing_components.append("experience_analysis")
        if not state.skills_analysis:
            missing_components.append("skills_analysis")
        if not state.education_analysis:
            missing_components.append("education_analysis")
        if not state.certifications_analysis:
            missing_components.append("certifications_analysis")
        if state.suitability_score <= 0:
            missing_components.append("suitability_score")
            
        return missing_components
    
    def _create_error_report(self, error_type: str, error_message: str, 
                           processing_id: str, partial_state: Optional[CVAnalysisState] = None) -> Dict[str, Any]:
        """
        Create an error report when processing fails
        
        Args:
            error_type: Type of error that occurred
            error_message: Detailed error message
            processing_id: Unique processing identifier
            partial_state: Partial state if available
            
        Returns:
            Dict: Error report
        """
        logger.info(f"Creating error report [ID: {processing_id}] for error type: {error_type}")
        
        try:
            return create_fallback_report(
                position=partial_state.position if partial_state and partial_state.position else "Unknown Position",
                error_message=f"{error_type}: {error_message}"
            )
        except Exception as e:
            logger.error(f"Failed to create error report [ID: {processing_id}]: {str(e)}")
            # Return minimal error report
            return {
                "error": True,
                "error_type": error_type,
                "error_message": error_message,
                "processing_id": processing_id,
                "timestamp": datetime.now().isoformat(),
                "status": "FAILED"
            }
    
    def _create_partial_report(self, state: CVAnalysisState, 
                             missing_components: List[str], 
                             processing_id: str) -> Dict[str, Any]:
        """
        Create a partial report when some analysis components are missing
        
        Args:
            state: Partial workflow state
            missing_components: List of missing components
            processing_id: Unique processing identifier
            
        Returns:
            Dict: Partial analysis report
        """
        logger.info(f"Creating partial report [ID: {processing_id}] - missing: {missing_components}")
        
        try:
            # Use existing report if available, otherwise create from state
            if state.final_report:
                report = state.final_report.copy()
            else:
                report = state.to_dict()
            
            # Add metadata about partial completion
            report.update({
                "status": "PARTIAL",
                "completion_percentage": state.get_completion_percentage(),
                "missing_components": missing_components,
                "processing_id": processing_id,
                "warnings": [f"Analysis incomplete - missing: {', '.join(missing_components)}"]
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create partial report [ID: {processing_id}]: {str(e)}")
            return self._create_error_report(
                "PARTIAL_REPORT_ERROR",
                f"Failed to create partial report: {str(e)}",
                processing_id,
                state
            )
    
    def _create_debug_aware_llm_manager(self):
        """
        Create a wrapper around the LLM manager that automatically passes debug mode
        
        Returns:
            LLM manager wrapper that handles debug mode automatically
        """
        class DebugAwareLLMManagerWrapper:
            def __init__(self, llm_manager, debug_mode):
                self.llm_manager = llm_manager
                self.debug_mode = debug_mode
            
            def get_llm(self, debug_mode=None):
                # Use the wrapper's debug mode if not explicitly overridden
                effective_debug_mode = debug_mode if debug_mode is not None else self.debug_mode
                return self.llm_manager.get_llm(debug_mode=effective_debug_mode)
            
            def __getattr__(self, name):
                # Delegate all other attributes to the wrapped LLM manager
                return getattr(self.llm_manager, name)
        
        return DebugAwareLLMManagerWrapper(self.llm_manager, self.config.debug_mode)
    
    def _wrap_node(self, node_func):
        """
        Wrap a node function with LLM manager injection and error handling
        
        Args:
            node_func: The node function to wrap
            
        Returns:
            Wrapped node function
        """
        def wrapped_node(state: CVAnalysisState) -> CVAnalysisState:
            try:
                # Set current node for tracking
                node_name = node_func.__name__.replace('_node', '')
                state.set_current_node(node_name)
                
                logger.debug(f"Executing node: {node_name}")
                
                # Execute node with appropriate parameters based on function signature
                import inspect
                sig = inspect.signature(node_func)
                if len(sig.parameters) == 1:
                    # Node only expects state parameter
                    result = node_func(state)
                else:
                    # Node expects state and llm_manager parameters
                    # Create a wrapper that automatically passes debug mode to get_llm()
                    llm_manager_wrapper = self._create_debug_aware_llm_manager()
                    result = node_func(state, llm_manager_wrapper)
                
                # Handle different return types from nodes
                if isinstance(result, CVAnalysisState):
                    # Node returned complete state object
                    logger.debug(f"Node {node_name} completed successfully")
                    return result
                elif isinstance(result, dict):
                    # Node returned state updates - merge with current state
                    for key, value in result.items():
                        if hasattr(state, key):
                            setattr(state, key, value)
                    logger.debug(f"Node {node_name} completed successfully")
                    return state
                else:
                    # Unexpected return type
                    error_msg = f"Node {node_func.__name__} returned unexpected type: {type(result)}"
                    logger.error(error_msg)
                    state.add_error(error_msg)
                    return state
                
            except Exception as e:
                error_msg = f"Node {node_func.__name__} failed: {str(e)}"
                logger.error(error_msg)
                state.add_error(error_msg)
                return state
        
        return wrapped_node
    

    
    def _validate_config(self) -> None:
        """
        Validate the NovaGraph configuration
        
        Raises:
            NovaGraphError: If configuration is invalid
        """
        try:
            # Test LLM manager initialization
            self.llm_manager.get_llm()
            logger.info("LLM configuration validated successfully")
            
        except LLMProviderError as e:
            raise NovaGraphError(f"LLM configuration invalid: {str(e)}")
        except Exception as e:
            raise NovaGraphError(f"Configuration validation failed: {str(e)}")
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the current workflow configuration
        
        Returns:
            Dict containing workflow information
        """
        return {
            "llm_provider": self.llm_manager.get_provider_info(),
            "scoring_weights": self.config.scoring_weights,
            "output_format": self.config.output_format,
            "debug_mode": self.config.debug_mode,
            "workflow_compiled": self._workflow is not None,
            "workflow_nodes": [
                "cv_parser", "experience_analyzer", "skills_analyzer",
                "education_analyzer", "certification_analyzer", 
                "suitability_scorer", "report_compiler"
            ]
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics and health information
        
        Returns:
            Dict containing processing statistics
        """
        try:
            llm_status = self.llm_manager.test_connection()
            provider_info = self.llm_manager.get_provider_info()
        except Exception as e:
            llm_status = False
            provider_info = {"error": str(e)}
        
        return {
            "llm_connectivity": llm_status,
            "llm_provider_info": provider_info,
            "workflow_ready": self._workflow is not None,
            "configuration_valid": True,  # If we got this far, config is valid
            "debug_mode": self.config.debug_mode,
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check
        
        Returns:
            Dict containing health check results
        """
        health_results = {
            "overall_status": "HEALTHY",
            "checks": {},
            "warnings": [],
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Check LLM connectivity
        try:
            llm_connected = self.llm_manager.test_connection()
            health_results["checks"]["llm_connectivity"] = {
                "status": "PASS" if llm_connected else "FAIL",
                "details": "LLM provider is accessible" if llm_connected else "LLM provider is not accessible"
            }
            if not llm_connected:
                health_results["errors"].append("LLM provider connectivity failed")
                health_results["overall_status"] = "UNHEALTHY"
        except Exception as e:
            health_results["checks"]["llm_connectivity"] = {
                "status": "ERROR",
                "details": f"LLM connectivity test failed: {str(e)}"
            }
            health_results["errors"].append(f"LLM connectivity error: {str(e)}")
            health_results["overall_status"] = "UNHEALTHY"
        
        # Check workflow compilation
        try:
            workflow = self.get_compiled_workflow()
            health_results["checks"]["workflow_compilation"] = {
                "status": "PASS",
                "details": "Workflow compiled successfully"
            }
        except Exception as e:
            health_results["checks"]["workflow_compilation"] = {
                "status": "FAIL",
                "details": f"Workflow compilation failed: {str(e)}"
            }
            health_results["errors"].append(f"Workflow compilation error: {str(e)}")
            health_results["overall_status"] = "UNHEALTHY"
        
        # Check configuration validity
        try:
            self._validate_config()
            health_results["checks"]["configuration"] = {
                "status": "PASS",
                "details": "Configuration is valid"
            }
        except Exception as e:
            health_results["checks"]["configuration"] = {
                "status": "FAIL",
                "details": f"Configuration validation failed: {str(e)}"
            }
            health_results["errors"].append(f"Configuration error: {str(e)}")
            health_results["overall_status"] = "UNHEALTHY"
        
        # Add warnings for non-critical issues
        if self.config.debug_mode:
            health_results["warnings"].append("Debug mode is enabled - may impact performance")
        
        return health_results
    
    def reset_workflow(self) -> None:
        """Reset the compiled workflow to force recompilation"""
        self._workflow = None
        logger.info("Workflow reset - will be recompiled on next use")
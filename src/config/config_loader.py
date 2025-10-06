"""
Configuration loading utilities for Nova HR Assistant
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from .llm_config import LLMConfig
from .nova_config import NovaConfig
from ..utils.rate_limiter import configure_rate_limiting


class ConfigLoader:
    """Utility class for loading configuration from environment variables"""
    
    @staticmethod
    def load_env_file(env_path: Optional[str] = None) -> None:
        """Load environment variables from .env file"""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()  # Load from default .env file
    
    @staticmethod
    def load_llm_config() -> LLMConfig:
        """Load LLM configuration from environment variables"""
        # Default to OpenRouter with Grok 4 unless overridden by env vars
        provider = os.getenv("NOVA_LLM_PROVIDER", "openrouter")
        model_name = os.getenv("NOVA_LLM_MODEL", "grok-4")
        temperature = float(os.getenv("NOVA_LLM_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("NOVA_LLM_MAX_TOKENS", "2000"))
        api_key = os.getenv("NOVA_LLM_API_KEY")
        base_url = os.getenv("NOVA_LLM_BASE_URL")
        
        return LLMConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url
        )
    
    @staticmethod
    def load_scoring_weights() -> Dict[str, float]:
        """Load scoring weights from environment variables"""
        return {
            "experience": float(os.getenv("NOVA_WEIGHT_EXPERIENCE", "0.3")),
            "skills": float(os.getenv("NOVA_WEIGHT_SKILLS", "0.25")),
            "education": float(os.getenv("NOVA_WEIGHT_EDUCATION", "0.2")),
            "certifications": float(os.getenv("NOVA_WEIGHT_CERTIFICATIONS", "0.15")),
            "overall_fit": float(os.getenv("NOVA_WEIGHT_OVERALL_FIT", "0.1"))
        }
    
    @staticmethod
    def load_output_format() -> Dict[str, Any]:
        """Load output format settings from environment variables"""
        return {
            "include_reasoning": os.getenv("NOVA_INCLUDE_REASONING", "true").lower() == "true",
            "include_confidence_scores": os.getenv("NOVA_INCLUDE_CONFIDENCE", "true").lower() == "true"
        }
    
    @staticmethod
    def load_rate_limiting_config() -> Dict[str, Any]:
        """Load rate limiting configuration from environment variables"""
        return {
            "max_requests_per_minute": int(os.getenv("NOVA_MAX_REQUESTS_PER_MINUTE", "30")),
            "request_delay": float(os.getenv("NOVA_REQUEST_DELAY", "2.0")),
            "max_retries": int(os.getenv("NOVA_MAX_RETRIES", "3")),
            "retry_base_delay": float(os.getenv("NOVA_RETRY_BASE_DELAY", "1.0")),
            "retry_max_delay": float(os.getenv("NOVA_RETRY_MAX_DELAY", "60.0"))
        }
    
    @staticmethod
    def load_nova_config(env_path: Optional[str] = None) -> NovaConfig:
        """Load complete Nova configuration from environment variables"""
        ConfigLoader.load_env_file(env_path)
        
        llm_config = ConfigLoader.load_llm_config()
        scoring_weights = ConfigLoader.load_scoring_weights()
        output_format = ConfigLoader.load_output_format()
        debug_mode = os.getenv("NOVA_DEBUG_MODE", "false").lower() == "true"
        
        # Configure rate limiting
        rate_config = ConfigLoader.load_rate_limiting_config()
        configure_rate_limiting(**rate_config)
        
        return NovaConfig(
            llm=llm_config,
            scoring_weights=scoring_weights,
            output_format=output_format,
            debug_mode=debug_mode
        )
"""
Factory methods for creating LLM instances with latest LangChain syntax
"""

from typing import Dict, Any, Optional
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from ..config.llm_config import LLMConfig, LLMProvider


class LLMFactory:
    """Factory class for creating LLM instances using latest LangChain syntax"""
    
    @staticmethod
    def create_openai_llm(
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> ChatOpenAI:
        """
        Create OpenAI LLM instance with latest LangChain syntax
        
        Args:
            model_name: OpenAI model name (e.g., gpt-3.5-turbo, gpt-4)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key
            base_url: Optional custom base URL
            **kwargs: Additional parameters for ChatOpenAI
            
        Returns:
            ChatOpenAI: Configured OpenAI LLM instance
        """
        params = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if api_key:
            params["api_key"] = api_key
        
        if base_url:
            params["base_url"] = base_url
        
        return ChatOpenAI(**params)
    
    @staticmethod
    def create_anthropic_llm(
        model_name: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> ChatAnthropic:
        """
        Create Anthropic LLM instance with latest LangChain syntax
        
        Args:
            model_name: Anthropic model name (e.g., claude-3-sonnet-20240229)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            api_key: Anthropic API key
            base_url: Optional custom base URL
            **kwargs: Additional parameters for ChatAnthropic
            
        Returns:
            ChatAnthropic: Configured Anthropic LLM instance
        """
        params = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if api_key:
            params["api_key"] = api_key
        
        if base_url:
            params["base_url"] = base_url
        
        return ChatAnthropic(**params)
    
    @staticmethod
    def create_groq_llm(
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ChatGroq:
        """
        Create Groq LLM instance with latest LangChain syntax
        
        Args:
            model_name: Groq model name (e.g., mixtral-8x7b-32768, llama2-70b-4096)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            api_key: Groq API key
            **kwargs: Additional parameters for ChatGroq
            
        Returns:
            ChatGroq: Configured Groq LLM instance
        """
        params = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if api_key:
            params["api_key"] = api_key
        
        return ChatGroq(**params)
    
    @staticmethod
    def create_local_llm(
        model_name: str = "llama2",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> Ollama:
        """
        Create local LLM instance using Ollama with latest LangChain syntax
        
        Args:
            model_name: Local model name (e.g., llama2, mistral)
            temperature: Sampling temperature (0.0 to 1.0)
            base_url: Ollama server URL
            **kwargs: Additional parameters for Ollama
            
        Returns:
            Ollama: Configured local LLM instance
        """
        params = {
            "model": model_name,
            "temperature": temperature,
            "base_url": base_url,
            **kwargs
        }
        
        return Ollama(**params)
    
    @staticmethod
    def create_from_config(config: LLMConfig) -> BaseLLM:
        """
        Create LLM instance from LLMConfig using appropriate factory method
        
        Args:
            config: LLMConfig instance with provider settings
            
        Returns:
            BaseLLM: Configured LLM instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if config.provider == LLMProvider.OPENAI.value:
            return LLMFactory.create_openai_llm(
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key,
                base_url=config.base_url
            )
        
        elif config.provider == LLMProvider.ANTHROPIC.value:
            return LLMFactory.create_anthropic_llm(
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key,
                base_url=config.base_url
            )
        
        elif config.provider == LLMProvider.GROQ.value:
            return LLMFactory.create_groq_llm(
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key
            )
        
        elif config.provider == LLMProvider.OPENROUTER.value:
            # OpenRouter is OpenAI-compatible; use ChatOpenAI with base_url
            return LLMFactory.create_openai_llm(
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1"
            )
        
        elif config.provider == LLMProvider.LOCAL.value:
            return LLMFactory.create_local_llm(
                model_name=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434"
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    @staticmethod
    def get_default_models() -> Dict[str, str]:
        """
        Get default model names for each provider
        
        Returns:
            Dict mapping provider names to default model names
        """
        return {
            LLMProvider.OPENAI.value: "gpt-3.5-turbo",
            LLMProvider.ANTHROPIC.value: "claude-3-sonnet-20240229",
            LLMProvider.GROQ.value: "openai/gpt-oss-120b",
            LLMProvider.OPENROUTER.value: "xai/grok-4o-mini",  # placeholder; override via env
            LLMProvider.LOCAL.value: "llama2"
        }
    
    @staticmethod
    def get_supported_providers() -> list[str]:
        """
        Get list of supported provider names
        
        Returns:
            List of supported provider names
        """
        return [provider.value for provider in LLMProvider]
    
    @staticmethod
    def validate_provider_config(provider: str, model_name: str, api_key: Optional[str] = None) -> bool:
        """
        Validate provider configuration without creating LLM instance
        
        Args:
            provider: Provider name
            model_name: Model name
            api_key: Optional API key
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if provider not in LLMFactory.get_supported_providers():
            raise ValueError(f"Unsupported provider: {provider}")
        
        if not model_name:
            raise ValueError("Model name is required")
        
        # Check API key requirements for cloud providers
        if provider in [LLMProvider.OPENAI.value, LLMProvider.ANTHROPIC.value, LLMProvider.GROQ.value]:
            if not api_key:
                raise ValueError(f"API key is required for {provider}")
        
        return True
"""
LLM Manager for dynamic provider support in Nova HR Assistant
"""

import logging
from typing import Optional, Dict, Any, List
from langchain_core.language_models import BaseLLM
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from ..config.llm_config import LLMConfig, LLMProvider
from .rate_limiter import with_rate_limiting, get_rate_limiter, get_retry_handler


logger = logging.getLogger(__name__)


class DebugLLMWrapper:
    """Wrapper class that logs LLM interactions for debugging"""
    
    def __init__(self, llm: BaseLLM, debug_mode: bool = False):
        self.llm = llm
        self.debug_mode = debug_mode
        self._call_count = 0
    
    @with_rate_limiting
    def invoke(self, input_data, **kwargs):
        """Invoke the LLM with debug logging and rate limiting"""
        self._call_count += 1
        
        if self.debug_mode:
            print(f"\n{'='*60}")
            print(f"🤖 LLM CALL #{self._call_count}")
            print(f"{'='*60}")
            
            # Show rate limiter stats
            rate_stats = get_rate_limiter().get_stats()
            print(f"📊 Rate Limiter: {rate_stats['requests_in_last_minute']}/{rate_stats['max_requests_per_minute']} requests/min")
            
            # Log input - show just the key parts
            if isinstance(input_data, (list, tuple)):
                print("📝 INPUT MESSAGES:")
                for i, msg in enumerate(input_data):
                    if hasattr(msg, 'content'):
                        content = str(msg.content)
                        # Show first and last parts of long content
                        if len(content) > 300:
                            print(f"  Message {i+1}: {content[:150]}...[{len(content)-300} chars]...{content[-150:]}")
                        else:
                            print(f"  Message {i+1}: {content}")
                    else:
                        print(f"  Message {i+1}: {str(msg)[:200]}...")
            else:
                content = str(input_data)
                if len(content) > 300:
                    print(f"📝 INPUT: {content[:150]}...[{len(content)-300} chars]...{content[-150:]}")
                else:
                    print(f"📝 INPUT: {content}")
            
            print(f"\n⏳ Calling {self.llm.__class__.__name__}...")
        
        # Make the actual LLM call (rate limiting is handled by decorator)
        try:
            result = self.llm.invoke(input_data, **kwargs)
            
            if self.debug_mode:
                print(f"✅ RESPONSE RECEIVED")
                print(f"📤 OUTPUT: {str(result)[:1000]}...")
                print(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ LLM CALL FAILED: {str(e)}")
                print(f"{'='*80}\n")
            raise
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM"""
        return getattr(self.llm, name)


class LLMProviderError(Exception):
    """Exception raised when LLM provider operations fail"""
    pass


class LLMManager:
    """Manages LLM provider configuration and initialization with dynamic switching support"""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM Manager with configuration
        
        Args:
            config: LLMConfig instance with provider settings
        """
        self.config = config
        self._llm_instance: Optional[BaseLLM] = None
        self._provider_factories = {
            LLMProvider.OPENAI.value: self._create_openai_llm,
            LLMProvider.ANTHROPIC.value: self._create_anthropic_llm,
            LLMProvider.GOOGLE.value: self._create_google_llm,
            LLMProvider.GROQ.value: self._create_groq_llm,
            LLMProvider.OPENROUTER.value: self._create_openrouter_llm,
            LLMProvider.LOCAL.value: self._create_local_llm
        }
        
        # Validate configuration on initialization
        self._validate_config()
    
    def get_llm(self, debug_mode: bool = False) -> BaseLLM:
        """
        Get the configured LLM instance, creating it if necessary
        
        Args:
            debug_mode: Enable debug logging for LLM interactions
        
        Returns:
            BaseLLM: Configured LLM instance (wrapped with debug if enabled)
            
        Raises:
            LLMProviderError: If LLM creation or configuration fails
        """
        if self._llm_instance is None:
            self._llm_instance = self._create_llm()
        
        if debug_mode:
            return DebugLLMWrapper(self._llm_instance, debug_mode=True)
        else:
            return self._llm_instance
    
    def switch_provider(self, provider: str, model_name: Optional[str] = None, 
                       api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        """
        Switch to a different LLM provider dynamically
        
        Args:
            provider: New provider name (openai, anthropic, local)
            model_name: Optional new model name
            api_key: Optional new API key
            base_url: Optional new base URL
            
        Raises:
            LLMProviderError: If provider switching fails
        """
        try:
            # Update configuration
            self.config.provider = provider
            if model_name:
                self.config.model_name = model_name
            if api_key:
                self.config.api_key = api_key
            if base_url:
                self.config.base_url = base_url
            
            # Validate new configuration
            self._validate_config()
            
            # Clear existing instance to force recreation
            self._llm_instance = None
            
            # Test new configuration by creating instance
            self.get_llm()
            
            logger.info(f"Successfully switched to provider: {provider}")
            
        except Exception as e:
            logger.error(f"Failed to switch to provider {provider}: {str(e)}")
            raise LLMProviderError(f"Provider switching failed: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test connection to the current LLM provider
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            llm = self.get_llm()
            # Try a simple test prompt
            test_response = llm.invoke("Hello")
            return bool(test_response)
        except Exception as e:
            logger.warning(f"Connection test failed for provider {self.config.provider}: {str(e)}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider configuration
        
        Returns:
            Dict containing provider information
        """
        return {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "has_api_key": bool(self.config.api_key),
            "base_url": self.config.base_url,
            "connection_status": self.test_connection()
        }
    
    def _create_llm(self) -> BaseLLM:
        """
        Create LLM instance based on current configuration
        
        Returns:
            BaseLLM: Configured LLM instance
            
        Raises:
            LLMProviderError: If LLM creation fails
        """
        try:
            factory = self._provider_factories.get(self.config.provider)
            if not factory:
                raise LLMProviderError(f"Unsupported provider: {self.config.provider}")
            
            return factory()
            
        except Exception as e:
            logger.error(f"Failed to create LLM for provider {self.config.provider}: {str(e)}")
            raise LLMProviderError(f"LLM creation failed: {str(e)}")
    
    def _create_openai_llm(self) -> ChatOpenAI:
        """Create OpenAI LLM instance"""
        if not self.config.api_key:
            raise LLMProviderError("OpenAI API key is required")
        
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key": self.config.api_key
        }
        
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        
        return ChatOpenAI(**kwargs)

    def _create_openrouter_llm(self) -> ChatOpenAI:
        """Create OpenRouter LLM instance (OpenAI-compatible)"""
        if not self.config.api_key:
            raise LLMProviderError("OpenRouter API key is required")
        
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key": self.config.api_key,
            "base_url": self.config.base_url or "https://openrouter.ai/api/v1",
        }
        
        return ChatOpenAI(**kwargs)
    
    def _create_anthropic_llm(self) -> ChatAnthropic:
        """Create Anthropic LLM instance"""
        if not self.config.api_key:
            raise LLMProviderError("Anthropic API key is required")
        
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key": self.config.api_key
        }
        
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        
        return ChatAnthropic(**kwargs)
    
    def _create_google_llm(self) -> ChatGoogleGenerativeAI:
        """Create Google Gemini LLM instance"""
        if not self.config.api_key:
            raise LLMProviderError("Google API key is required")
        
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "google_api_key": self.config.api_key
        }
        
        return ChatGoogleGenerativeAI(**kwargs)
    
    def _create_groq_llm(self) -> ChatGroq:
        """Create Groq LLM instance"""
        if not self.config.api_key:
            raise LLMProviderError("Groq API key is required")
        
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key": self.config.api_key
        }
        
        return ChatGroq(**kwargs)
    
    def _create_local_llm(self) -> Ollama:
        """Create local LLM instance (using Ollama)"""
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature
        }
        
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        else:
            kwargs["base_url"] = "http://localhost:11434"  # Default Ollama URL
        
        return Ollama(**kwargs)
    
    def _validate_config(self) -> None:
        """
        Validate current configuration
        
        Raises:
            LLMProviderError: If configuration is invalid
        """
        try:
            # This will raise ValueError if config is invalid
            self.config.__post_init__()
        except ValueError as e:
            raise LLMProviderError(f"Invalid configuration: {str(e)}")
        
        # Additional provider-specific validation
        if self.config.provider == LLMProvider.OPENAI.value:
            if not self.config.model_name.startswith(('gpt-', 'text-')):
                logger.warning(f"Unusual OpenAI model name: {self.config.model_name}")
        
        elif self.config.provider == LLMProvider.ANTHROPIC.value:
            if not self.config.model_name.startswith('claude-'):
                logger.warning(f"Unusual Anthropic model name: {self.config.model_name}")
        
        elif self.config.provider == LLMProvider.GOOGLE.value:
            if not self.config.model_name.startswith('gemini-'):
                logger.warning(f"Unusual Google model name: {self.config.model_name}")
        
        elif self.config.provider == LLMProvider.GROQ.value:
            # Groq supports various models, so we'll be more flexible with validation
            if not self.config.model_name:
                raise LLMProviderError("Model name is required for Groq provider")
        
        elif self.config.provider == LLMProvider.OPENROUTER.value:
            if not self.config.model_name:
                raise LLMProviderError("Model name is required for OpenRouter provider")
        
        elif self.config.provider == LLMProvider.LOCAL.value:
            if not self.config.model_name:
                raise LLMProviderError("Model name is required for local provider")
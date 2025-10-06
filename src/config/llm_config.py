"""
LLM configuration classes for Nova HR Assistant
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM provider settings"""
    provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.provider not in [p.value for p in LLMProvider]:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        
        # Validate required API key for cloud providers
        if self.provider in [
            LLMProvider.OPENAI.value,
            LLMProvider.ANTHROPIC.value,
            LLMProvider.GOOGLE.value,
            LLMProvider.GROQ.value,
            LLMProvider.OPENROUTER.value,
        ]:
            if not self.api_key:
                raise ValueError(f"API key required for {self.provider} provider")
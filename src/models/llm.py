"""
LLM model implementations for the Datasheet Analyzer.
"""
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from ..config.settings import LLM_CONFIG

class LLMModel:
    """
    A wrapper class for LLM models.
    """
    def __init__(self, model_type: str = "gpt4", api_key: Optional[str] = None):
        """
        Initialize the LLM model.
        
        Args:
            model_type: Type of model to use (gpt4 or claude)
            api_key: API key for the model
        """
        self.model_type = model_type
        self.config = LLM_CONFIG[model_type]
        self.api_key = api_key
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the appropriate model based on model_type.
        
        Returns:
            Initialized LLM model
        """
        if self.model_type == "gpt4":
            return ChatOpenAI(
                model_name=self.config["model_name"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                api_key=self.api_key
            )
        elif self.model_type == "claude":
            return ChatAnthropic(
                model_name=self.config["model_name"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                anthropic_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def get_model(self):
        """
        Get the initialized model.
        
        Returns:
            Initialized LLM model
        """
        return self.model
    
    def update_api_key(self, api_key: str):
        """
        Update the API key and reinitialize the model.
        
        Args:
            api_key: New API key
        """
        self.api_key = api_key
        self.model = self._initialize_model() 
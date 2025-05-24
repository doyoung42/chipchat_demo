"""
Token management for the Datasheet Analyzer.
"""
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TokenManager:
    """Token management class for handling API tokens."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the token manager.
        
        Args:
            config_path: Path to the token configuration file
        """
        self.config_path = config_path or Path("tokens.json")
        self.tokens: Dict[str, str] = {}
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from the configuration file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.tokens = json.load(f)
                logger.info("Tokens loaded successfully")
            else:
                logger.info("No token configuration file found")
        except Exception as e:
            logger.error(f"Error loading tokens: {str(e)}")
            self.tokens = {}
    
    def save_tokens(self):
        """Save tokens to the configuration file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.tokens, f, indent=2)
            logger.info("Tokens saved successfully")
        except Exception as e:
            logger.error(f"Error saving tokens: {str(e)}")
    
    def get_token(self, service: str) -> Optional[str]:
        """
        Get token for a specific service.
        
        Args:
            service: Service name (e.g., 'openai', 'huggingface', 'anthropic')
            
        Returns:
            Token string if exists, None otherwise
        """
        return self.tokens.get(service)
    
    def set_token(self, service: str, token: str):
        """
        Set token for a specific service.
        
        Args:
            service: Service name
            token: Token string
        """
        self.tokens[service] = token
        self.save_tokens()
        logger.info(f"Token set for {service}")
    
    def remove_token(self, service: str):
        """
        Remove token for a specific service.
        
        Args:
            service: Service name
        """
        if service in self.tokens:
            del self.tokens[service]
            self.save_tokens()
            logger.info(f"Token removed for {service}")
    
    def has_token(self, service: str) -> bool:
        """
        Check if token exists for a specific service.
        
        Args:
            service: Service name
            
        Returns:
            True if token exists, False otherwise
        """
        return service in self.tokens 
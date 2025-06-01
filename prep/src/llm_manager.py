import json
import requests
import os
from typing import Dict, List, Optional, Tuple, Any

class LLMManager:
    def __init__(self, provider: str = "openai", model_name: Optional[str] = None):
        """Initialize LLM Manager
        
        Args:
            provider: LLM provider ("openai" or "claude")
            model_name: Model name to use (optional, defaults to provider's default model)
        """
        self.provider = provider.lower()
        
        # Load API keys from key.json
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc', 'key.json')
        with open(key_path, 'r') as f:
            keys = json.load(f)
        
        # OpenAI settings
        self.openai_api_key = keys['openai_api_key']
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
        self.openai_headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Claude settings
        self.claude_api_key = keys['anthropic_api_key']
        self.claude_api_url = "https://api.anthropic.com/v1/messages"
        self.claude_headers = {
            "x-api-key": self.claude_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Load model configuration from models.json
        models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc', 'models.json')
        with open(models_path, 'r') as f:
            models_config = json.load(f)
        
        # Model configuration
        self.model_name = model_name
        self.openai_models = models_config['openai_models']
        self.claude_models = models_config['claude_models']
        self.default_openai_model = models_config['default_model']['openai']
        self.default_claude_model = models_config['default_model']['claude']
    
    def _get_model_name(self) -> str:
        """Get the appropriate model name based on provider and user selection"""
        if self.model_name:
            return self.model_name
        
        # Return default model based on provider
        if self.provider == "openai":
            return self.default_openai_model
        else:
            return self.default_claude_model
    
    def set_model(self, model_name: str) -> None:
        """Set the model to use for API calls
        
        Args:
            model_name: Name of the model to use
        
        Raises:
            ValueError: If the model name is not supported for the current provider
        """
        if self.provider == "openai" and model_name not in self.openai_models:
            raise ValueError(f"Unsupported OpenAI model: {model_name}. Available models: {', '.join(self.openai_models)}")
        elif self.provider == "claude" and model_name not in self.claude_models:
            raise ValueError(f"Unsupported Claude model: {model_name}. Available models: {', '.join(self.claude_models)}")
        
        self.model_name = model_name
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "claude":
            return self._call_claude(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        data = {
            "model": self._get_model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(self.openai_api_url, headers=self.openai_headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API"""
        if not self.claude_api_key:
            raise ValueError("Claude API key has not been set.")
            
        data = {
            "model": self._get_model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(self.claude_api_url, headers=self.claude_headers, json=data)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
    
    def check_page_usefulness(self, page_text: str) -> bool:
        """Judge the usefulness of a page through LLM"""
        prompt = f"""Please determine if the following text contains useful information from a datasheet or technical document.
        Useful information includes product specifications, technical characteristics, performance metrics, etc.
        Table of contents, copyright information, blank pages, etc. are considered not useful.
        
        Text:
        {page_text}
        
        Please answer only with 'yes' if it contains useful information, or 'no' if it doesn't."""
        
        response = self._call_llm(prompt)
        return response.strip().lower() == 'yes'
    
    def analyze_chunk(self, pages: List[str], page_numbers: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """Analyze a chunk of pages to evaluate the usefulness and key content of each page
        
        Args:
            pages: List of page contents
            page_numbers: List of page numbers
            
        Returns:
            (Whether there is at least one useful page in the chunk, Dictionary of page information)
        """
        # Combine chunk contents (including page numbers)
        combined_text = ""
        for i, page in enumerate(pages):
            combined_text += f"=== PAGE {page_numbers[i]} ===\n{page}\n\n"
        
        prompt = f"""The following text contains multiple pages extracted from a datasheet PDF.
        For each page, please evaluate:
        
        1. Whether the page contains useful information (product specifications, technical characteristics, performance metrics, etc.)
        2. If useful, extract the most important information from that page (exact text, not summarized)
        
        Non-useful content: covers, table of contents, copyright information, blank pages, indexes, etc.
        
        Text:
        {combined_text}
        
        Please output in the following JSON format:
        {{
            "page_numbers": {{
                "page_number1": {{
                    "is_useful": true or false,
                    "content": "Important information here if useful (exact text)"
                }},
                "page_number2": {{
                    "is_useful": true or false,
                    "content": "Important information here if useful (exact text)"
                }},
                ...
            }}
        }}
        
        Please use the actual page numbers from the document.
        """
        
        response = self._call_llm(prompt)
        
        try:
            # Parse JSON response
            result = json.loads(response)
            page_info = result.get('page_numbers', {})
            
            # Check if there is at least one useful page
            has_useful_page = any(info.get('is_useful', False) for info in page_info.values())
            
            return has_useful_page, page_info
        except json.JSONDecodeError:
            # Return empty result if JSON parsing fails
            return False, {}
    
    def extract_features(self, pages: List[str]) -> str:
        """Extract features from useful pages and return in JSON format"""
        combined_text = "\n".join(pages)
        return self.extract_features_from_content(combined_text)
    
    def extract_features_from_content(self, content: str) -> str:
        """Extract features from the extracted content and return in JSON format"""
        prompt = f"""You are an expert in analyzing electronic component datasheets. 
Your task is to extract and organize key specifications and information from the provided datasheet.
Focus on technical details, specifications, and important characteristics of the component.
Maintain accuracy and use the original terminology from the datasheet.

Please provide the analysis in the following JSON format:
{{
    "component_name": "string",
    "specifications": {{
        "voltage": "string",
        "current": "string",
        "package": "string",
        "temperature_range": "string"
    }},
    "key_features": ["string"],
    "applications": ["string"],
    "notes": "string"
}}

Text:
{content}"""
        
        response = self._call_llm(prompt)
        return response 
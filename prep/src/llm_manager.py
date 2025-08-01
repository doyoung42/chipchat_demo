import json
import requests
import os
from typing import Dict, List, Optional, Tuple, Any

class LLMManager:
    def __init__(self, provider: str = None, model_name: Optional[str] = None):
        """Initialize LLM Manager
        
        Args:
            provider: LLM provider ("openai" or "claude")
            model_name: Model name to use (optional, defaults to provider's default model)
        """
        # Load API keys and selected models from key.json
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc', 'key.json')
        with open(key_path, 'r') as f:
            keys = json.load(f)
        
        # Determine provider based on available API keys and selected models
        if provider is None:
            if keys.get('anthropic_api_key') and not keys.get('openai_api_key'):
                self.provider = "claude"
            elif keys.get('openai_api_key') and not keys.get('anthropic_api_key'):
                self.provider = "openai"
            else:
                # If both keys are available, use the selected model from key.json
                selected_models = keys.get('selected_models', {})
                if selected_models.get('claude'):
                    self.provider = "claude"
                else:
                    self.provider = "openai"
        else:
            self.provider = provider.lower()
        
        # OpenAI settings
        self.openai_api_key = keys.get('openai_api_key', '')
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
        self.openai_headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Claude settings
        self.claude_api_key = keys.get('anthropic_api_key', '')
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
        
        # If no model_name is provided, use the selected model from key.json
        if self.model_name is None:
            selected_models = keys.get('selected_models', {})
            if self.provider == "claude" and selected_models.get('claude'):
                self.model_name = selected_models['claude']
            elif self.provider == "openai" and selected_models.get('openai'):
                self.model_name = selected_models['openai']
    
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
        
        # 현재 사용 중인 모델명 가져오기
        model_name = self._get_model_name()
        
        # Claude API 헤더 설정 - 모든 Claude 모델에 적용
        self.claude_headers = {
            "x-api-key": self.claude_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # 기본 요청 데이터 구성
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        # Claude 3 모델인 경우 (claude-3로 시작하는 모든 모델)
        if "claude-3" in model_name:
            data["max_tokens"] = 1000
        else:
            # Claude 2 이하 모델인 경우 (이전 API 형식 사용)
            data["max_tokens_to_sample"] = 1000
        
        try:
            response = requests.post(self.claude_api_url, headers=self.claude_headers, json=data)
            response.raise_for_status()
            
            # Claude 3 모델인 경우 응답 형식이 다름
            if "claude-3" in model_name:
                return response.json()["content"][0]["text"]
            else:
                # Claude 2 이하 모델인 경우
                return response.json()["completion"]
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"응답 상태 코드: {e.response.status_code}")
                print(f"응답 내용: {e.response.text}")
            raise
    
    def analyze_pages_with_categories(self, pages: List[str], page_numbers: List[int]) -> Dict[str, Any]:
        """Analyze a chunk of pages to evaluate the usefulness and categorize content
        
        Args:
            pages: List of page contents
            page_numbers: List of page numbers
            
        Returns:
            Dictionary containing page analysis with categories
        """
        # Combine chunk contents (including page numbers)
        combined_text = ""
        for i, page in enumerate(pages):
            combined_text += f"=== PAGE {page_numbers[i]} ===\n{page}\n\n"
        
        prompt = f"""The following text contains multiple pages extracted from a datasheet PDF.
        For each page, please evaluate:
        
        1. Whether the page contains useful information (product specifications, technical characteristics, performance metrics, etc.)
        2. If useful, identify which category(s) the page belongs to from the following options:
            - Product Summary
            - Electrical Characteristics
            - Application Circuits
            - Mechanical Characteristics
            - Reliability and Environmental Conditions
            - Packaging Information
        3. Extract the most important information from that page (exact text, not summarized)
        
        IMPORTANT: Pay special attention to summary pages that often appear at the beginning of datasheets. These pages typically contain critical overview information, key features, and condensed specifications. These summary pages SHOULD BE MARKED AS USEFUL and categorized as "Product Summary".
        
        Non-useful content: copyright notices, blank pages, detailed indexes, etc. Note that table of contents pages might be useful if they contain overview information along with the contents.
        
        Text:
        {combined_text}
        
        Please output in the following JSON format:
        {{
            "page_analysis": {{
                "page_number1": {{
                    "is_useful": true or false,
                    "categories": ["Category1", "Category2"],
                    "content": "Important information here if useful (exact text)"
                }},
                "page_number2": {{
                    "is_useful": true or false,
                    "categories": ["Category1"],
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
            page_analysis = result.get('page_analysis', {})
            
            # Return the page analysis with categories
            return page_analysis
        except json.JSONDecodeError:
            # Return empty result if JSON parsing fails
            return {}
    
    def create_semantic_chunks_for_category(self, category_name: str, category_content: str, target_part_number: str = None) -> List[str]:
        """Create semantic chunks for a specific category
        
        Args:
            category_name: Name of the category
            category_content: Combined content for the category
            target_part_number: Specific part number to focus on (optional)
            
        Returns:
            List of semantic chunks
        """
        category_descriptions = {
            "Product Summary": "Overview, key features, and general description of the product",
            "Electrical Characteristics": "Voltage, current, power specifications, timing characteristics, and other electrical parameters",
            "Application Circuits": "Example circuits, application notes, and implementation examples",
            "Mechanical Characteristics": "Physical dimensions, form factor, mounting information, and mechanical specifications",
            "Reliability and Environmental Conditions": "Operating conditions, temperature ranges, reliability tests, and environmental specifications",
            "Packaging Information": "Package types, ordering information, marking details, and shipment specifications"
        }
        
        description = category_descriptions.get(category_name, "")
        
        part_specific_instruction = ""
        if target_part_number:
            part_specific_instruction = f"""
        IMPORTANT: This datasheet may contain information about multiple components.
        Focus ONLY on information related to part number: {target_part_number}.
        When creating chunks, ONLY include information relevant to this specific part number.
        """
        
        prompt = f"""You are an expert in analyzing electronic component datasheets.
        Your task is to create semantic chunks from content related to "{category_name}".
        
        {description}
        {part_specific_instruction}
        
        Please divide the following text into 3-5 meaningful chunks that would be useful for vector search.
        Each chunk should be self-contained and meaningful on its own.
        Create chunks with some semantic overlap for better retrieval.
        
        Text related to {category_name}:
        {category_content}
        
        Please output in the following JSON format:
        {{
            "chunks": [
                "Chunk 1 content...",
                "Chunk 2 content...",
                "Chunk 3 content..."
            ]
        }}
        """
        
        response = self._call_llm(prompt)
        
        try:
            # Parse JSON response
            result = json.loads(response)
            chunks = result.get('chunks', [])
            
            # Enhance chunks with part number information for better retrieval
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = chunk
                if target_part_number:
                    # Add part number context to each chunk for better retrieval
                    enhanced_chunk = f"[Part Number: {target_part_number}] {chunk}"
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
        except json.JSONDecodeError:
            # Return empty list if JSON parsing fails
            return []
    
    def extract_metadata(self, all_useful_content: str, target_part_number: str = None) -> Dict[str, str]:
        """Extract metadata from all useful content
        
        Args:
            all_useful_content: Combined content from all useful pages
            target_part_number: Specific part number to focus on (optional)
            
        Returns:
            Dictionary containing metadata
        """
        part_specific_instruction = ""
        if target_part_number:
            part_specific_instruction = f"""
        IMPORTANT: This datasheet contains information about multiple components. 
        Focus ONLY on the component with part number: {target_part_number}.
        Extract information ONLY related to this specific part number.
        """
            
        prompt = f"""You are an expert in analyzing electronic component datasheets. 
        Your task is to extract key metadata from the provided datasheet content.
        {part_specific_instruction}
        
        Please extract the following information:
        - Component name/model
        - Manufacturer
        - Key specifications (brief)
        - Main applications
        
        Text:
        {all_useful_content}
        
        Please output in the following JSON format:
        {{
            "component_name": "string",
            "manufacturer": "string",
            "key_specifications": "string",
            "applications": "string"
        }}
        """
        
        response = self._call_llm(prompt)
        
        try:
            # Parse JSON response
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # Return empty dictionary if JSON parsing fails
            return {
                "component_name": "",
                "manufacturer": "",
                "key_specifications": "",
                "applications": ""
            } 
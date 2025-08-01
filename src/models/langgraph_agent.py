"""
LangGraph-based multi-tool agent for ChipChat.
Handles query classification and tool selection for component search tasks.
"""

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from pathlib import Path

from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import add_messages

from .agent_tools import ChipChatTools
from .llm_manager import LLMManager
from .vectorstore_manager import VectorstoreManager
from ..utils.prompt_manager import get_prompt_manager


class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: Annotated[List, add_messages]
    query: str
    query_type: str
    tools_to_use: List[str]
    tool_results: Dict[str, str]
    final_response: str


class ChipChatAgent:
    """LangGraph-based agent for multi-tool component search"""
    
    def __init__(self, csv_path: str, vectorstore_manager: VectorstoreManager, 
                 vectorstore, llm_manager: LLMManager):
        """Initialize the agent with tools and LLM"""
        self.llm_manager = llm_manager
        
        # Initialize prompt manager
        self.prompt_manager = get_prompt_manager()
        
        # Initialize tools
        self.tools = ChipChatTools(
            csv_path=csv_path,
            vectorstore_manager=vectorstore_manager,
            vectorstore=vectorstore,
            llm_manager=llm_manager
        )
        
        # Create tool executor
        self.tool_executor = ToolExecutor(self.tools.get_tools())
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create the graph
        self.graph = self._create_graph()
        
        # Compile the graph
        self.app = self.graph.compile()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("tool_selector", self._select_tools)
        workflow.add_node("tool_executor", self._execute_tools)
        workflow.add_node("response_generator", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "tool_selector")
        workflow.add_edge("tool_selector", "tool_executor")
        workflow.add_edge("tool_executor", "response_generator")
        workflow.add_edge("response_generator", END)
        
        return workflow
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """Classify the user query to determine intent"""
        query = state["query"]
        
        try:
            # Get classification prompt from template
            classification_prompt = self.prompt_manager.get_classification_prompt(query)
            
            response = self.llm_manager._call_llm(classification_prompt, temperature=0.1)
            query_type = response.strip().upper()
            
            # Validate classification using prompt manager
            valid_types = self.prompt_manager.get_valid_classifications()
            if query_type not in valid_types:
                query_type = self.prompt_manager.get_fallback_classification()
                
            self.logger.info(f"Query classified as: {query_type}")
            
        except Exception as e:
            self.logger.error(f"Classification error: {str(e)}")
            query_type = self.prompt_manager.get_fallback_classification()
        
        state["query_type"] = query_type
        return state
    
    def _select_tools(self, state: AgentState) -> AgentState:
        """Select appropriate tools based on query classification"""
        query_type = state["query_type"]
        query = state["query"]
        
        # Tool selection logic
        tools_to_use = []
        
        if query_type == "COMPONENT_LIST":
            tools_to_use = ["search_chip_database"]
            
        elif query_type == "TECHNICAL_DETAIL":
            # Check if specific part number mentioned
            if any(part in query.upper() for part in ["LM", "W25Q", "TPS", "BZX", "1N"]):
                tools_to_use = ["search_vectorstore"]
            else:
                tools_to_use = ["search_chip_database", "search_vectorstore"]
                
        elif query_type == "PDF_UPLOAD":
            tools_to_use = ["process_new_pdf"]
            
        else:  # HYBRID
            tools_to_use = ["search_chip_database", "search_vectorstore"]
        
        state["tools_to_use"] = tools_to_use
        self.logger.info(f"Selected tools: {tools_to_use}")
        
        return state
    
    def _execute_tools(self, state: AgentState) -> AgentState:
        """Execute the selected tools"""
        query = state["query"]
        tools_to_use = state["tools_to_use"]
        tool_results = {}
        
        for tool_name in tools_to_use:
            try:
                if tool_name == "search_chip_database":
                    result = self.tools.search_chip_database(query)
                    tool_results["chip_database"] = result
                    
                elif tool_name == "search_vectorstore":
                    # Extract part number if mentioned in query
                    part_number = self._extract_part_number(query)
                    result = self.tools.search_vectorstore(
                        query=query,
                        part_number=part_number,
                        k=3
                    )
                    tool_results["vectorstore"] = result
                    
                elif tool_name == "process_new_pdf":
                    # This would be handled by file upload in Streamlit
                    tool_results["pdf_upload"] = "PDF upload functionality available in the interface"
                
                self.logger.info(f"Executed tool: {tool_name}")
                
            except Exception as e:
                self.logger.error(f"Tool execution error for {tool_name}: {str(e)}")
                tool_results[tool_name] = f"❌ Error executing {tool_name}: {str(e)}"
        
        state["tool_results"] = tool_results
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response by combining tool results"""
        query = state["query"]
        query_type = state["query_type"]
        tool_results = state["tool_results"]
        
        # Combine results intelligently
        response_parts = []
        
        if "chip_database" in tool_results:
            response_parts.append(f"📊 **Component Database Search:**\n{tool_results['chip_database']}")
        
        if "vectorstore" in tool_results:
            response_parts.append(f"📚 **Detailed Technical Information:**\n{tool_results['vectorstore']}")
        
        if "pdf_upload" in tool_results:
            response_parts.append(f"📄 **PDF Processing:**\n{tool_results['pdf_upload']}")
        
        # Create final response
        if response_parts:
            final_response = f"🤖 **ChipChat Agent Response**\n\n"
            final_response += f"❓ **Your Query:** {query}\n"
            final_response += f"🏷️ **Query Type:** {query_type}\n\n"
            final_response += "\n\n".join(response_parts)
        else:
            final_response = f"❌ I couldn't find relevant information for your query: '{query}'"
        
        state["final_response"] = final_response
        return state
    
    def _extract_part_number(self, query: str) -> str:
        """Extract part number from query if present"""
        # Simple pattern matching for common part number formats
        import re
        
        patterns = [
            r'\b([A-Z]{1,3}\d+[A-Z]*\d*[A-Z]*)\b',  # LM324, W25Q32JV, etc.
            r'\b(\d+[A-Z]+\d+[A-Z]*)\b',            # 1N4728A, etc.
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query.upper())
            if matches:
                return matches[0]
        
        return ""
    
    def process_query(self, query: str, uploaded_file=None) -> str:
        """Process a user query through the agent system"""
        try:
            # Handle PDF upload if provided
            if uploaded_file is not None:
                pdf_content = uploaded_file.read()
                filename = uploaded_file.name
                return self.tools.process_new_pdf(pdf_content, filename)
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "query_type": "",
                "tools_to_use": [],
                "tool_results": {},
                "final_response": ""
            }
            
            # Run the agent
            result = self.app.invoke(initial_state)
            
            return result["final_response"]
            
        except Exception as e:
            self.logger.error(f"Agent processing error: {str(e)}")
            return f"❌ Error processing query: {str(e)}"
    
    def get_agent_info(self) -> str:
        """Get information about the agent's capabilities"""
        return f"""
🤖 **ChipChat Agent System**

**Capabilities:**
• 🔍 **Smart Query Classification**: Automatically determines what you're looking for
• 🛠️ **Multi-Tool Execution**: Uses the right tools for each question type
• 📊 **Component Search**: Find parts by functionality from chipDB.csv
• 📚 **Technical Details**: Get detailed specs from vectorstore
• 📄 **PDF Processing**: Upload and integrate new datasheets

**Query Types Supported:**
1. **Component Lists**: "What components do voltage regulation?"
2. **Technical Details**: "W25Q32JV electrical characteristics"
3. **PDF Upload**: Upload new datasheets via file upload
4. **Hybrid Queries**: Complex questions combining multiple aspects

**Available Data:**
• 📊 **ChipDB**: {len(self.tools.chip_db)} components with basic specs
• 📚 **Vectorstore**: Detailed technical documentation
• 🔄 **Live Processing**: Can add new components in real-time

{self.tools.get_tool_descriptions()}
        """.strip() 






import re
import os
import sys
import json
import time
import chromadb
from typing import Optional, TypedDict, Literal, List, Tuple, Dict, Any, Callable, AsyncGenerator
from loguru import logger
import asyncio
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
# --- LangChain Core & Community Imports ---
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, BaseTool
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.factory_tools import ToolFactory
from pydantic import BaseModel, Field as PydanticField
from langchain_tavily import TavilySearch
import re
# --- Tool Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
from app.core.config import get_settings
from app.agents.load_config import load_config
from app.agents.workflow.state import GraphState
from app.agents.retrievers.company_retriever import CompanyRetrieverTool
from app.agents.factory.tools.summary_tool import SummaryTool
from app.agents.factory.tools.search_tool import SearchTool
from app.agents.workflow.initalize import llm_instance, settings, agent_config
# class GraphState(TypedDict):
#     session_id: Optional[str]
#     original_query: str
#     employee_id: Optional[int]
#     guest_id: Optional[str]
#     customer_id: Optional[str]
#     user_role: str  # Added user role field
#     rewritten_query: str
#     intents: Optional[List[str]]
#     contexts: Optional[List[Dict[str, Any]]]
#     classified_agent: Literal["CompanyAgent", "CustomerAgent", "MedicalAgent", "ProductAgent", "DrugAgent", "NaiveAgent", "GeneticAgent", "QuestionGeneratorAgent", "RewriterAgent", "VisualAgent", "EmployeeAgent", "GuestAgent"]
#     task_assigned: List[Literal["searchweb", "retrieve", "chitchat", "summary"]] # Changed to List[Literal[...]]
#     agent_response: str
#     agent_thinks: Dict[str, Any]
#     reflection_feedback: str
#     is_final_answer: bool
#     error_message: Optional[str]
#     retry_count: int
#     suggested_questions: List[str]
#     chat_history: List[Tuple[str, str]]
#     suggested_followup_agents: Literal["CompanyAgent", "CustomerAgent", "MedicalAgent", "ProductAgent", "DrugAgent", "NaiveAgent", "GeneticAgent", "QuestionGeneratorAgent", "RewriterAgent", "VisualAgent", "SummaryAgent", "EmployeeAgent", "GuestAgent"]
#     # Added fields for context summarization
#     tool_summaries: Optional[Dict[str, str]]  # Tool name to summarized output mapping
#     context_summaries: Optional[Dict[str, str]]  # Context type to summary mapping
#     visual_context: Optional[Dict[str, Any]]  # For visual data and chart information
#     entity_summaries: Optional[Dict[str, Any]]  # For entity-specific summarized information
#     tool_usage_history: Optional[List[Dict[str, Any]]]  # Track tool usage across conversation
#     combined_context: Optional[str]  # Synthesized context from multiple sources

# logger.info("LangGraph State Defined .") 
agent_config = load_config("app/agents/config/vie/agent.yaml")


class BaseAgentNode:
    def __init__(self, llm: BaseChatModel, agent_name: str, system_prompt: Optional[str] = None):
        self.llm = llm
        self.agent_name = agent_name
        self.system_prompt = system_prompt or f"You are a helpful agent named {agent_name}."
        logger.info(f"Agent '{self.agent_name}' initialized.")


    def execute(self, state: GraphState) -> Dict[str, Any]:
        """
        Execute the agent's logic based on the current state.
        
        Args:
            state (GraphState): The current state of the graph.
        
        Returns:
            Dict[str, Any]: The updated state after execution.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")


    async def async_execute(self, state: GraphState) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement the execute method.")

    def _prepare_execution(self, state: GraphState, critical_error_check: Optional[str] = None) -> Optional[Dict[str, Any]]:
        logger.info(f"--- Agent Node: {self.agent_name} ---")
        if critical_error_check and state.get("error_message") and critical_error_check in state["error_message"]:
            logger.warning(f"Skipping {self.agent_name} due to critical error: {state['error_message']}")
            # Return proper error state instead of empty dict to prevent workflow hanging
            return {**state, "error_message": f"Critical error in {self.agent_name}: {state['error_message']}"}
            

        partial_state_update = {"error_message": None}
        if self.agent_name not in ["ReflectionAgent", "QuestionGeneratorAgent"]: # These don't directly set agent_response
            partial_state_update["agent_response"] = ""
        if self.agent_name == "RewriterAgent": # Rewriter specific resets
             partial_state_update.update({"reflection_feedback": "", "suggested_questions": [], "is_final_answer": False})
        return {**state, **partial_state_update}  # Merge with existing state

    def _handle_execution_error(self, e: Exception, current_partial_state: Dict) -> Dict[str, Any]:
        logger.error(f"Error in {self.agent_name} execution: {e}", exc_info=True)
        # current_partial_state["error_message"] = f"{self.agent_name} execution failed: {e}"
        current_partial_state["error_message"] = f"Thực thi {self.agent_name} thất bại: {e}"
        if "agent_response" in current_partial_state: # Only if it's an agent that produces direct response
            # current_partial_state["agent_response"] = f"Sorry, an error occurred with the {self.agent_name}. Please try again."
            current_partial_state["agent_response"] = f"Xin lỗi, đã xảy ra lỗi với {self.agent_name}. Vui lòng thử lại."
        return current_partial_state




class Agent(BaseAgentNode):
    def __init__(self, llm: BaseChatModel, agent_name: str, system_prompt: str, default_tools: List[BaseAgentTool], vectorstore: Optional[Chroma] = None):
        super().__init__(llm, agent_name)
        self.system_prompt = system_prompt # Can be an f-string template
        self.default_tools = default_tools
        self.vectorstore = vectorstore or None
        self.retriever = None
        
        """
        Initialize the agent with a language model, name, system prompt, and configuration.
        
        Args:
            llm (BaseChatModel): The language model to be used by the agent.
            agent_name (str): The name of the agent.
            system_prompt (Optional[str]): The system prompt for the agent.
            default_tools (List[BaseAgentTool]): A list of default tools available to the agent.
        """
        super().__init__(llm, agent_name, system_prompt)
        
        logger.info(f"Agent '{self.agent_name}' initialized with tools: {self.default_tools}")

    
    def _prepare_execution(self, state, critical_error_check = None):
        return super()._prepare_execution(state, critical_error_check)


    def get_dynamic_tools(self, state: GraphState) -> List[BaseTool]:
        """
        Get dynamic tools based on the current state.
        
        Args:
            state (GraphState): The current state of the graph.
        
        Returns:
            List[BaseTool]: A list of dynamic tools for the agent.
        """
        factory = ToolFactory(state)
        intents = state.get("intents", '')
        if intents:
            tool = factory.get_tool(intents)

            if tool:
                logger.info(f"Using dynamic tool: {tool.name} for intents: {intents}")
                return [tool]
        if intents  == 'retrieve':
            retrieve_tool = self.retriever if self.retriever else CompanyRetrieverTool(collection_name="company_docs", watch_directory='app/agents/retrievers/storages/company')
            return [retrieve_tool]
        if intents == 'chitchat':
            return [SearchTool(n_results=3)]

        return []
        
        
        
        
    def get_tool_contexts(self, state: GraphState) -> Dict[str, Any]:
        current_tools = self.default_tools + self.get_dynamic_tools(state)
        tool_contexts = {}
        query = state.get("rewritten_query", state.get("original_query", ""))
        for tool in current_tools:
            if isinstance(tool, BaseAgentTool):
                tool_name = tool.name
                context = tool.run(query)
                tool_contexts[tool_name] = f"Ten cong cu: {tool_name}\nMo ta cong cu: {tool.description}\nKet qua: {context}"
            elif isinstance(tool, BaseTool):
                tool_name = tool.name
                context = tool.run(query)
                tool_contexts[tool_name] = f"Ten cong cu: {tool_name}\nMo ta cong cu: {tool.description}\nKet qua: {context}"
            else:
                logger.warning(f"Tool {tool} is not a recognized BaseAgentTool or BaseTool.")
            
            logger.debug(f"Tool {tool_name} context: {context[:100]}...")  # Log first 100 characters of context for brevity
        logger.info(f"Tools chosen for {self.agent_name}: {list(tool_contexts.keys())}")
        tool_contexts = self.handle_tool_summaries(state, tool_contexts)
        return tool_contexts

    def handle_tool_summaries(self, state: GraphState, tool_contexts: Dict[str, Any]) -> Dict[str, str]:
        intent = state.get("intents", [None])[0]
        summary_tool = SummaryTool(llm=self.llm)
        
        if intent == "summary" and len(state.get('original_query')) < 1000:
            all_documents = self.vectorstore.get()
            content = "\n".join([doc for doc in all_documents['documents']])
            logger.info(f"Tóm tắt cho tất cả tài liệu: {tool_contexts['summary'][:100]}...")
            
            if len(content) > 10000:
                tool_contexts['summary'] = "Tóm tắt cho tất cả tài liệu: "
                for i in range(0, len(content), 10000):
                    chunk = content[i:i+10000]
                    summary = summary_tool.run(chunk)
                    tool_contexts['summary'] += f"Tóm tắt đoạn: {i}: " + summary + "\n"
            else:
                tool_contexts['summary'] = summary_tool.run(content)
            logger.info(f"Tóm tắt cho tất cả tài liệu: {tool_contexts['summary'][:100]}...")
        elif intent == "summary" and 1000 <= len(state.get('original_query')) <= 10000:
            
            logger.warning("Tóm tắt ý định được phát hiện nhưng truy vấn quá dài để tóm tắt.")
            tool_contexts['summary'] = summary_tool.run(state.get('original_query', ''))
        elif intent == "summary" and len(state.get('original_query')) > 10000:
            logger.warning("Tóm tắt ý định được phát hiện nhưng truy vấn quá dài để tóm tắt.")
            
            tool_contexts['summary'] = summary_tool.run(state.get('original_query', '')[:10000])
        else:
            tool_contexts['summary'] = "Không có tóm tắt không được yêu cầu hoặc không đủ dữ liệu để tóm tắt."
        
        
        logger.info(f"Summary context for {self.agent_name}: {tool_contexts['summary'][:100]}...")
        return tool_contexts

    def execute(self, state):
        partial_state = self._prepare_execution(state, critical_error_check="Query processing failed")
        if not isinstance(partial_state, dict):
            return {}
        
        
        query = state.get("rewritten_query", state.get("original_query", ""))
        chat_history = state.get("chat_history", [])
        history_messages = [HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=a) for i, (q, a) in enumerate(chat_history[-3:])]
        
        logger.info(f"Chat history for {self.agent_name}: {history_messages}")
        contexts = self.get_tool_contexts(state)
        
        tool_messages = []
        for tool_name, context in contexts.items():
            if isinstance(context, str):
                tool_messages.append(ToolMessage(content=context, tool_call_id=tool_name))
            elif isinstance(context, Document):
                tool_messages.append(ToolMessage(content=context.page_content, tool_call_id=tool_name))
            else:
                logger.warning(f"Unexpected context type for tool {tool_name}: {type(context)}")
        logger.info(f"Tool contexts for {self.agent_name}: {tool_messages}")
        partial_state["contexts"] = contexts

        
        prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "User query: {query}"),
                MessagesPlaceholder(variable_name="history", optional=True),
                ("human", "Dưới đây là các kết quả từ các công cụ đã truy xuất dựa trên câu hỏi của người dùng. "
                           "- Ưu tiên: Tài liệu công ty (nguồn chính thống và đáng tin cậy nhất)\n"
                           "- Chỉ sử dụng dữ liệu từ tìm kiếm web để bổ sung khi tài liệu công ty không có đủ thông tin\n"
                           "- Luôn trả lời rõ ràng, chính xác và chuyên nghiệp\n\n"
                           "- Đưa ra câu trả lời ngắn gọn, rõ ràng và chính xác\n"
                           "- Tránh sử dụng các thông tin không chính xác hoặc không rõ ràng\n"
                           "Kết quả từ các công cụ:\n{tools_results}. Dựa trên những thông tin trên, hãy đưa ra câu trả lời theo câu hỏi của người dùng: {query}."
                )

                # ("human", "Use the following tool results to generate a final response, remembering to PRIORITIZE company documents: {tools_results}")
            ])
        chain = prompt_template | self.llm
        logger.info(f"Executing {self.agent_name} with query: {query} and contexts: {contexts}")
        try:
            current_partial_state = self._prepare_execution(state, critical_error_check="critical")
            if not current_partial_state:
                return {**state, "error_message": "Failed to prepare execution state."}  # Skip execution if critical error check fails

            response = chain.invoke({
                "query": query,
                "chat_history": history_messages,
                "tools_results": "\n".join([msg.content for msg in tool_messages])
            })
            partial_state["agent_response"] = response.content
            logger.info(f"{self.agent_name} response: {current_partial_state['agent_response']}")

        except Exception as e:
            partial_state = self._handle_execution_error(e, partial_state)

        return {**state, **partial_state}

    async def stream_execute(self, state: GraphState) :
        partial_state = self._prepare_execution(state, critical_error_check="Query processing failed")
        if not isinstance(partial_state, dict):
            yield {**state, "error_message": "Failed to prepare execution state."}
            return

        query = state.get("rewritten_query", state.get("original_query", ""))
        chat_history = state.get("chat_history", [])
        history_messages = [HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=a) for i, (q, a) in enumerate(chat_history[-3:])]
        
        logger.info(f"Chat history for {self.agent_name}: {history_messages}")
        contexts = self.get_tool_contexts(state)
        
        tool_messages = []
        for tool_name, context in contexts.items():
            if isinstance(context, str):
                tool_messages.append(ToolMessage(content=context, tool_call_id=tool_name))
            elif isinstance(context, Document):
                tool_messages.append(ToolMessage(content=context.page_content, tool_call_id=tool_name))
            else:
                logger.warning(f"Unexpected context type for tool {tool_name}: {type(context)}")
        logger.info(f"Tool contexts for {self.agent_name}: {tool_messages}")
        partial_state["contexts"] = tool_messages

        partial_state["agent_thinks"] = {"query": query, "contexts": contexts, "chat_history": chat_history}
        prompt_template = ChatPromptTemplate.from_messages([
                {"role": "system", "content": self.system_prompt},
                {"role": "human", "content": "User query: {query}"},
                MessagesPlaceholder(variable_name="history", optional=True),
                {"role": "human", "content": "Dưới đây là các kết quả từ các công cụ đã truy xuất dựa trên câu hỏi của người dùng. "
                           "- Ưu tiên: Tài liệu công ty (nguồn chính thống và đáng tin cậy nhất)\n"
                           "- Chỉ sử dụng dữ liệu từ tìm kiếm web để bổ sung khi tài liệu công ty không có đủ thông tin\n"
                           "- Luôn trả lời rõ ràng, chính xác và chuyên nghiệp\n\n"
                           "- Đưa ra câu trả lời ngắn gọn, rõ ràng và chính xác\n"
                           "- Tránh sử dụng các thông tin không chính xác hoặc không rõ ràng\n"
                           "Kết quả từ các công cụ:\n{tools_results}. Dựa trên những thông tin trên, hãy đưa ra câu trả lời theo câu hỏi của người dùng: {query}."
                }

                # ("human", "Use the following tool results to generate a final response, remembering to PRIORITIZE company documents: {tools_results}")
            ])
        chain = prompt_template | self.llm
        logger.info(f"Executing {self.agent_name} with query: {query} and contexts: {contexts}")
        try:
             # Skip execution if critical error check fails
            
            response = chain.astream({
                "query": query,
                "chat_history": history_messages,
                "tools_results": "\n".join([msg.content for msg in tool_messages])
            })
            partial_state["agent_response"] = ""
            async for chunk in response:
                if hasattr(chunk, 'content'):
                    partial_state["agent_response"] += chunk.content
                    yield partial_state
            logger.info(f"{self.agent_name} response: {partial_state['agent_response']}")
        except Exception as e:
            partial_state = self._handle_execution_error(e, partial_state)
            logger.error(f"Error during streaming execution of {self.agent_name}: {e}")
            yield {**state, **partial_state}
        yield {**state, **partial_state}

    async def async_execute(self, state: GraphState) -> Dict[str, Any]:
        """
        Asynchronously execute the agent's logic based on the current state.
        
        Args:
            state (GraphState): The current state of the graph.
        
        Returns:
            Dict[str, Any]: The updated state after execution.
        """
        raise NotImplementedError("Subclasses must implement the async_execute method.")
    


    async def async_stream_execute(self, state: GraphState) -> Dict[str, Any]:
        """
        Asynchronously stream the agent's execution based on the current state.
        
        Args:
            state (GraphState): The current state of the graph.
        
        Returns:
            Dict[str, Any]: The updated state after execution.
        """
        raise NotImplementedError("Subclasses must implement the stream_execute method.")
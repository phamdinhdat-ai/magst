import asyncio
import json
import sys
import time
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from loguru import logger
from pathlib import Path

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

# --- Local/App Imports ---
# Giả sử các import này đã hoạt động đúng
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from app.agents.workflow.state import GraphState
    from app.agents.factory.tools.base import BaseAgentTool
    from app.agents.workflow.initalize import llm_instance
except ImportError:
    # Fallback cho môi trường test độc lập
    class GraphState(dict): pass
    class BaseAgentTool(BaseTool): pass
    class MockLLM(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return "Mock response"
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return "Mock response"
    llm_instance = MockLLM()


class BaseAgentNode:
    """Lớp cơ sở cho tất cả các node trong đồ thị, bao gồm các agent và các node logic khác."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        logger.info(f"Node '{self.agent_name}' initialized.")

    def execute(self, state: GraphState) -> Dict[str, Any]:
        """Thực thi logic của node một cách đồng bộ."""
        raise NotImplementedError("Subclasses must implement the execute method.")

    async def aexecute(self, state: GraphState) -> Dict[str, Any]:
        """Thryc thi logic của node một cách bất đồng bộ."""
        # Cung cấp một triển khai mặc định bất đồng bộ bằng cách chạy hàm đồng bộ trong một thread khác
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, state)

    def _prepare_execution(self, state: GraphState) -> GraphState:
        """Chuẩn bị trạng thái trước khi thực thi, xóa các lỗi cũ."""
        logger.info(f"--- Preparing execution for: {self.agent_name} ---")
        state['error_message'] = None  # Xóa lỗi từ bước trước
        return state

    def _handle_execution_error(self, e: Exception, state: GraphState) -> GraphState:
        """Xử lý lỗi xảy ra trong quá trình thực thi."""
        error_msg = f"Error in {self.agent_name}: {type(e).__name__} - {e}"
        logger.error(error_msg, exc_info=True)
        state['error_message'] = f"Lỗi thực thi {self.agent_name}: {e}"
        state['agent_response'] = f"Xin lỗi, đã có lỗi xảy ra với {self.agent_name}. Vui lòng thử lại."
        return state


class Agent(BaseAgentNode):
    """
    Lớp cơ sở cho các agent có khả năng sử dụng LLM và các công cụ (tools).
    Tối ưu hóa với việc thực thi tool song song và xử lý lỗi/timeout.
    """
    def __init__(self, llm: BaseChatModel, agent_name: str, system_prompt: str, default_tools: List[BaseAgentTool]):
        super().__init__(agent_name)
        self.llm = llm
        self.system_prompt = system_prompt
        self.default_tools = default_tools
        self.thread_pool = ThreadPoolExecutor(max_workers=5) # Pool riêng cho các tool của agent này
        logger.info(f"Agent '{self.agent_name}' initialized with {len(self.default_tools)} default tool(s).")

    def _get_tools_to_run(self, state: GraphState) -> List[BaseAgentTool]:
        """Xác định các tool cần chạy cho truy vấn hiện tại."""
        # Trong phiên bản tối ưu này, chúng ta mặc định chạy tất cả các tool được cung cấp cho agent.
        # Việc quyết định DỮ LIỆU nào từ tool để sử dụng sẽ do LLM đảm nhiệm,
        # giúp hệ thống linh hoạt hơn là cố gắng đoán trước tool nào là "đúng".
        return self.default_tools

    def _run_tools_in_parallel(self, tools: List[BaseAgentTool], query: str, timeout: int = 10) -> Dict[str, Any]:
        """Chạy các tool song song và thu thập kết quả."""
        tool_contexts = {}
        future_to_tool = {self.thread_pool.submit(tool.run, query): tool for tool in tools}

        for future in as_completed(future_to_tool):
            tool = future_to_tool[future]
            tool_name = tool.name
            try:
                # Lấy kết quả với timeout
                result = future.result(timeout=timeout)
                # Đảm bảo kết quả là một chuỗi có thể join được
                if isinstance(result, list):
                    context = "\n".join(map(str, result))
                else:
                    context = str(result)
                
                tool_contexts[tool_name] = f"--- Kết quả từ công cụ: {tool_name} ---\n{context}"
                logger.info(f"Tool '{tool_name}' executed successfully.")
            except TimeoutError:
                error_msg = f"Tool '{tool_name}' timed out after {timeout} seconds."
                logger.warning(error_msg)
                tool_contexts[tool_name] = f"--- Lỗi từ công cụ: {tool_name} ---\n{error_msg}"
            except Exception as e:
                error_msg = f"Tool '{tool_name}' failed with error: {e}"
                logger.error(error_msg, exc_info=True)
                tool_contexts[tool_name] = f"--- Lỗi từ công cụ: {tool_name} ---\n{error_msg}"
        
        return tool_contexts

    def _construct_prompt(self, query: str, history_messages: List[BaseMessage], tool_contexts: Dict[str, str]) -> ChatPromptTemplate:
        """Xây dựng prompt hoàn chỉnh cho LLM."""
        # Ghép tất cả các context từ tool thành một chuỗi duy nhất
        tools_results = "\n\n".join(tool_contexts.values())
        if not tools_results.strip():
            tools_results = "Không có công cụ nào trả về kết quả."

        prompt_messages = [
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="history", optional=True),
            HumanMessage(content=f"""
Dựa trên lịch sử trò chuyện và thông tin từ các công cụ dưới đây, hãy đưa ra câu trả lời toàn diện và chính xác cho câu hỏi của người dùng.

### Câu hỏi của người dùng:
{query}

### Thông tin từ các công cụ:
{tools_results}

### Câu trả lời của bạn:
""")
        ]
        return ChatPromptTemplate.from_messages(prompt_messages)

    def execute(self, state: GraphState) -> GraphState:
        """
        Thực thi luồng công việc hoàn chỉnh của agent:
        1. Chuẩn bị.
        2. Chạy các tool song song.
        3. Xây dựng prompt.
        4. Gọi LLM.
        5. Xử lý kết quả và lỗi.
        """
        state = self._prepare_execution(state)
        try:
            query = state.get("rewritten_query", state.get("original_query"))
            history_messages = self._format_chat_history(state.get("chat_history", []))

            tools_to_run = self._get_tools_to_run(state)
            tool_contexts = self._run_tools_in_parallel(tools_to_run, query)
            state['contexts'] = tool_contexts # Lưu lại context để debug

            prompt = self._construct_prompt(query, history_messages, tool_contexts)
            chain = prompt | self.llm
            
            logger.info(f"Invoking LLM for agent '{self.agent_name}'...")
            response = chain.invoke({
                "history": history_messages
            })
            
            state["agent_response"] = response.content
            logger.info(f"Agent '{self.agent_name}' generated response successfully.")

        except Exception as e:
            state = self._handle_execution_error(e, state)

        return state

    async def astream_execute(self, state: GraphState) -> AsyncGenerator[GraphState, None]:
        """Thực thi agent và stream kết quả trả về theo từng chunk."""
        state = self._prepare_execution(state)
        try:
            query = state.get("rewritten_query", state.get("original_query"))
            history_messages = self._format_chat_history(state.get("chat_history", []))

            tools_to_run = self._get_tools_to_run(state)
            
            # Chạy tool vẫn là tác vụ blocking, nên chạy trong executor
            loop = asyncio.get_event_loop()
            tool_contexts = await loop.run_in_executor(
                self.thread_pool, 
                self._run_tools_in_parallel, 
                tools_to_run, 
                query
            )
            state['contexts'] = tool_contexts

            prompt = self._construct_prompt(query, history_messages, tool_contexts)
            chain = prompt | self.llm
            
            logger.info(f"Streaming LLM response for agent '{self.agent_name}'...")
            
            full_response = ""
            async for chunk in chain.astream({"history": history_messages}):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    state["agent_response"] = full_response
                    yield state
            
            logger.info(f"Agent '{self.agent_name}' finished streaming response.")
            # Yield trạng thái cuối cùng một lần nữa để đảm bảo client nhận được đầy đủ
            yield state

        except Exception as e:
            state = self._handle_execution_error(e, state)
            yield state
            
    def _format_chat_history(self, history: List[Tuple[str, str]]) -> List[BaseMessage]:
        """Chuyển đổi lịch sử chat từ tuple sang đối tượng message của LangChain."""
        messages = []
        for q, a in history[-3:]: # Lấy 3 cặp hội thoại gần nhất
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
        return messages
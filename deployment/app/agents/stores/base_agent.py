import asyncio
import sys
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple, TypedDict
from concurrent.futures import ThreadPoolExecutor
import time
from loguru import logger
from pathlib import Path

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

# --- Local/App Imports ---
# Giả định các file này đã tồn tại và hoạt động đúng
# sys.path.append(...) nếu cần

# Import instance factory đã tối ưu
from app.agents.workflow.initalize import llm_instance, agent_config
from app.agents.factory.factory_tools import TOOL_FACTORY
from app.agents.workflow.state import GraphState as AgentState
from app.agents.factory.tools.search_tool import SearchTool
# --- Cấu hình trung tâm cho việc ánh xạ ---
# Đây là "bộ não" logic quyết định tool nào cho intent nào.
# Nên đặt ở một file config chung để dễ quản lý.
INTENT_TO_TOOL_MAP = {
    "searchweb": ["searchweb_tool"],
    "product": ["product_retriever_tool"],
    # 'retrieve' là một intent chung, nó sẽ được xử lý đặc biệt bởi các agent chuyên môn.
    "retrieve": [] # Thêm các retriever tĩnh ở đây
}

# Các tool động cần được xử lý riêng bởi các agent chuyên môn
# DYNAMIC_TOOLS = {
#     "CustomerAgent": "customer_retriever_tool",
#     "EmployeeAgent": "employee_retriever_tool"
# }
DYNAMIC_TOOLS = {
    "CustomerAgent": "customer_retriever_mcp_tool",
    "EmployeeAgent": "employee_retriever_mcp_tool"
}
# --- Shared Resources ---
# Executor dùng chung cho các tác vụ blocking (tool đồng bộ)
SHARED_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=10)


class BaseAgentNode:
    """
    Lớp cơ sở cho tất cả các node trong đồ thị.
    Cung cấp các phương thức thực thi và xử lý lỗi chung.
    """
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        logger.info(f"Node '{self.agent_name}' initialized.")

    def execute(self, state: AgentState) -> AgentState:
        """
        Thực thi đồng bộ. 
        CẢNH BÁO: Không nên gọi phương thức này từ một event loop đang chạy (ví dụ: trong LangGraph).
        Phương thức này chủ yếu dành cho việc test các node một cách độc lập.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                logger.error("`execute` was called from a running event loop. This is not supported.")
                raise RuntimeError(
                    "`execute` cannot be called from a running event loop. "
                    "Please use `aexecute` or configure your graph to be fully asynchronous."
                )
        except RuntimeError:
            # Không có event loop nào đang chạy, có thể an toàn sử dụng asyncio.run()
            pass
        
        logger.warning(f"Running `aexecute` for '{self.agent_name}' via `asyncio.run()`. This is only for standalone testing.")
        return asyncio.run(self.aexecute(state))

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi logic bất đồng bộ. Các lớp con PHẢI triển khai phương thức này.
        """
        raise NotImplementedError("Subclasses must implement the aexecute method.")

    def _prepare_execution(self, state: AgentState) -> AgentState:
        """Chuẩn bị state trước khi thực thi."""
        logger.info(f"--- Preparing execution for: {self.agent_name} ---")
        state['error_message'] = None # Xóa lỗi từ bước trước
        return state

    def _handle_execution_error(self, e: Exception, state: AgentState) -> AgentState:
        """Xử lý lỗi xảy ra trong quá trình thực thi."""
        # error_msg = f"Error in {self.agent_name}: {type(e).__name__} - {e}"
        # logger.error(error_msg, exc_info=True)
        state['error_message'] = f"Lỗi thực thi {self.agent_name}: {e}"
        state['agent_response'] = f"Xin lỗi, đã có lỗi xảy ra với {self.agent_name}. Vui lòng thử lại."
        return state


class Agent(BaseAgentNode):
    """
    Lớp cơ sở cho các agent có khả năng sử dụng LLM và các công cụ (tools).
    Tích hợp chặt chẽ với ToolFactory và logic chọn tool thông minh.
    """
    def __init__(self, 
                 llm: BaseChatModel, 
                 agent_name: str, 
                 system_prompt: str,
                 default_tool_names: List[str] = None,
                 **kwargs):
        super().__init__(agent_name)
        self.llm = llm
        self.system_prompt = system_prompt
        self.tool_factory = TOOL_FACTORY
        self.default_tool_names = default_tool_names or []
        
        # Các tham số cấu hình khác
        self.tool_timeout = kwargs.get('tool_timeout', 120)
        self.history_k = kwargs.get('history_k', 5)
        self.tool_executor = kwargs.get('tool_executor', SHARED_TOOL_EXECUTOR)
      
        logger.info(
            f"Agent '{self.agent_name}' initialized. "
            f"Default tools: {self.default_tool_names}. "
            "Will request additional tools from factory based on intent."
        )
        
    def _get_tools_to_run(self, state: AgentState) -> List[BaseTool]:
        """
        Xác định các tool cần chạy bằng cách kết hợp tool mặc định và tool theo intent.
        """
        final_tool_names = set(self.default_tool_names)
        if self.default_tool_names:
            logger.info(f"Adding default tools: {self.default_tool_names}")
        if not final_tool_names:
            logger.warning("No default tools specified. Will only use tools from intent mapping.")
            final_tool_names = set()
        intents = set(state.get('intents', []))
        if intents:
            logger.info(f"Requesting tools for intents: {intents}")
            for intent in intents:
                final_tool_names.update(INTENT_TO_TOOL_MAP.get(intent, []))
            
            if 'retrieve' in intents and self.agent_name in DYNAMIC_TOOLS:
                final_tool_names.add(DYNAMIC_TOOLS[self.agent_name])

            elif 'summary' in intents and self.agent_name in DYNAMIC_TOOLS:
                final_tool_names.add('summary_tool')
            
            # elif 'greeting' in intents:
            #     final_tool_names.add('get_datetime_tool')
            #     final_tool_names.add('searchweb_tool')  # Giả sử chào hỏi có thể bao gồm tìm kiếm web
            elif 'datetime' in intents:
                final_tool_names.add('get_datetime_tool')
                
            elif 'searchweb' in intents:
                final_tool_names.add('searchweb_tool')
                
            elif 'product' in intents and self.agent_name != "ReflectionAgent":
                final_tool_names.add('product_retriever_tool')
            
            elif 'retrieve' in intents and self.agent_name == "NaiveAgent":
                final_tool_names.add('searchweb_tool')
            
        

        selected_tools = []
        for tool_name in final_tool_names:
            logger.info(f"Retrieving tool: {tool_name}")
            if tool_name in DYNAMIC_TOOLS.values():
                is_dynamic = True
            elif tool_name == 'summary_tool':
                is_dynamic = True
            else:
                is_dynamic = False
            try:
                tool = self.tool_factory.get_dynamic_tool(tool_name, state) if is_dynamic else self.tool_factory.get_static_tool(tool_name)
                logger.info(f"Retrieved tool: {tool.name if tool else 'None'} (is_dynamic={is_dynamic})")
            except Exception as e:
                logger.error(f"Failed to retrieve tool '{tool_name}': {e}")
                continue
            logger.info(f"Retrieved tool: {tool.name if tool else 'None'} (is_dynamic={is_dynamic})")
            logger.debug(f"Tool details: {tool}")
            if tool:
                selected_tools.append(tool)
        
        logger.info(f"Successfully retrieved {len(selected_tools)} tools: {[t.name for t in selected_tools]}")
        return selected_tools

    async def _arun_tools_in_parallel(self, tools: List[BaseTool], query: str) -> Dict[str, Any]:
        """Chạy các tool (cả sync và async) song song với timeout."""
        if not tools:
            return {} # Tránh lỗi ValueError nếu không có tool nào được chọn

        tasks, tool_map = [], {}
        loop = asyncio.get_event_loop()
        
        for tool in tools:
            if hasattr(tool, '_initialize_all'):
                logger.info(f"Initializing tool: {tool.name}")
                await tool._initialize_all()
            logger.info(f"Query: {query}")
            logger.info(f"Preparing to run tool: {tool.name} (is_async={getattr(tool, 'async', False)})")
            try: 
                task = asyncio.create_task(tool.arun(query), name=tool.name)
                # if hasattr(tool, 'cleanup'):
                    # task.add_done_callback(lambda t: tool.cleanup())
                tool_map[task] = tool.name
                
            except Exception as e:
                logger.error(f"Failed to prepare task for tool '{tool.name}': {e}")
                continue
            tasks.append(task)
            
        tool_contexts = {}
        done, pending = await asyncio.wait(tasks, timeout=self.tool_timeout)
        logger.info(f"Executed {len(done)} tools, {len(pending)} tools are still pending after {self.tool_timeout} seconds.")
        for future in done:
            # caculate time taken
            start_time = time.time()
            tool_name = future.get_name() if hasattr(future, 'get_name') else tool_map.get(future, "unknown_tool")
            try:
                result = await future if asyncio.iscoroutine(future) or asyncio.isfuture(future) else future.result()
                context = "\n".join(map(str, result)) if isinstance(result, list) else str(result)
                tool_contexts[tool_name] = f"--- Kết quả từ công cụ: {tool_name} ---\n{context}"
                end_time = time.time()  
                time_taken = end_time - start_time
                logger.info(f"Tool '{tool_name}' executed successfully. Time taken: {time_taken:.2f} seconds.")
            except Exception as e:
                error_msg = f"Tool '{tool_name}' failed with error: {e}"
                logger.error(error_msg, exc_info=True)
                tool_contexts[tool_name] = f"--- Lỗi từ công cụ: {tool_name} ---\n{error_msg}"
        
        for future in pending:
            tool_name = future.get_name() if hasattr(future, 'get_name') else tool_map.get(future, "unknown_tool")
            error_msg = f"Tool '{tool_name}' timed out after {self.tool_timeout} seconds."
            logger.warning(error_msg)
            tool_contexts[tool_name] = f"--- Lỗi từ công cụ: {tool_name} ---\n{error_msg}"
            future.cancel()
        logger.info(f"Tool execution completed. Total tools executed: {len(tool_contexts)}")
        logger.debug(f"Tool contexts: {tool_contexts}")
        return tool_contexts

    def _construct_prompt(self, query: str, history_messages: List[BaseMessage], tool_contexts: Dict[str, str]) -> ChatPromptTemplate:
        """Xây dựng prompt hoàn chỉnh cho LLM."""
        if tool_contexts:
            logger.info(f"Constructing prompt with {len(tool_contexts)} tool contexts.")
            tools_results = "\n\n".join(tool_contexts.values())
        else:
            search_results =  SearchTool().run(query)
            search_context = "\n\n".join(search_results)
            tools_results = "Không có thông tin từ công cụ nào được sử dụng. Sử dụng kết quả tìm kiếm web:\n\n" + search_context
        logger.info(f"Constructing prompt for agent '{self.agent_name}' with query: {query}")
        logger.info(f"Tools results for agent '{self.agent_name}': {tools_results}")
        
        template = [
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="history", optional=True),
            HumanMessage(
                content=f"""Dựa trên lịch sử trò chuyện và thông tin từ các công cụ dưới đây, hãy đưa ra câu trả lời toàn diện và chính xác cho câu hỏi của người dùng.

                    ### Câu hỏi của người dùng:
                    {query}

                    ### Thông tin ngữ cảnh từ các công cụ:
                    {tools_results}

                    ### Câu trả lời của bạn:"""
                                ),
                            ]
        return ChatPromptTemplate.from_messages(template)
    
    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi agent bằng cách stream và trả về state cuối cùng.
        Đây là phương thức chính được gọi bởi graph.
        """
        final_state = state
        async for partial_state in self.astream_execute(state):
            final_state = partial_state
        return final_state

    async def astream_execute(self, state: AgentState) -> AsyncGenerator[AgentState, None]:
        """Thực thi agent và stream kết quả trả về theo từng chunk."""
        state = self._prepare_execution(state)
        try:
            query = state.get("original_query", "") 
            history_messages = self._format_chat_history(state.get("chat_history", []))
            # logger calculate time tool running
            logger.info(f"Executing agent '{self.agent_name}' with query: {query}")
            logger.info(f"Chat history for agent '{self.agent_name}': {history_messages}")
            # start = time.time()
            tools_to_run = self._get_tools_to_run(state)
            if not tools_to_run:
                logger.warning(f"No tools found for agent '{self.agent_name}'. Using default search tool.")
                tools_to_run = [SearchTool()]
            logger.info(f"Tools to run for agent '{self.agent_name}': {[tool.name for tool in tools_to_run]}")
            # get the last message from history
            
            history_content = "\n".join([f"{msg.content}" for msg in history_messages[-self.history_k:]])
            retriever_query = f"Truy van gan nhat: {query}\nLich su gan nhat: {history_content}"
            logger.info(f"Last history context for agent '{self.agent_name}': {history_content}")
            try:
                start = time.time() 
                tool_contexts = await self._arun_tools_in_parallel(tools_to_run, query)
                end = time.time()
                time_taken = end - start
                logger.info(f"Tool execution completed. Total tools executed: {len(tool_contexts)}")
                logger.info(f"Tool execution time: {time_taken:.2f} seconds.")
            except Exception as e:
                logger.error(f"Error running tools for agent '{self.agent_name}': {e}")
                state['error_message'] = f"Không thể chạy công cụ: {e}"
                yield state
                return
            # end = time.time()
            logger.debug(f"tools_to_run: {tools_to_run}")
            # logger.info(f"Tools executed in {end - start:.2f} seconds for agent '{self.agent_name}'.")  
            # logger.info(f"Tools executed in {end - start:.2f} seconds for agent '{self.agent_name}'.")
            state['contexts'] = tool_contexts
            # logger.info(f"Contexts prepared for agent '{self.agent_name}': {state['contexts'] }")
            # prompt = self._construct_prompt(query, history_messages, tool_contexts)
            
            """Xây dựng prompt hoàn chỉnh cho LLM."""
            logger.info(f"Constructing prompt for agent '{self.agent_name}' with query: {query}")
            tools_results = "\n\n".join(tool_contexts.values()) if tool_contexts else "Không có thông tin từ công cụ nào được sử dụng."
            logger.info(f"Tools results for agent '{self.agent_name}': {tools_results}")
            template = [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="history", optional=True),
                HumanMessage(
                    content=f"""Dựa trên lịch sử trò chuyện và thông tin từ các công cụ dưới đây, hãy đưa ra câu trả lời toàn diện và chính xác cho câu hỏi của người dùng.

                        ### Câu hỏi của người dùng:
                        {query}

                        ### Thông tin ngữ cảnh từ các công cụ:
                        {tools_results}

                        ### Câu trả lời của bạn:"""
                                    ),
                                ]
            logger.info(f"Constructing prompt for agent '{self.agent_name}' with query: {query}")
            prompt = ChatPromptTemplate.from_messages(template)
            # logger.info(f"Constructed prompt for agent '{self.agent_name}': {prompt}")
            chain = prompt | self.llm
            
            logger.info(f"Streaming LLM response for agent '{self.agent_name}'...")
            
            full_response = ""
            generator = chain.astream({
                "query": query,
                "tools_results": tool_contexts,
                "history": history_messages
            })
            async for chunk in generator:
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    state["agent_response"] = full_response
                    # logger.info(f"Streaming chunk for agent '{self.agent_name}': {chunk.content}")
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
        # logger.debug(f"show chat history: {history}")
        for item in history[-self.history_k:]: # Lấy k cặp hội thoại gần nhất
            if item['role'] == 'user':
                messages.append(HumanMessage(content=item['content']))
            elif item['role'] == 'assistant':
                messages.append(AIMessage(content=item['content']))
            else:
                logger.warning(f"Unknown role in chat history: {item['role']}. Skipping this message.")
        # logger.debug(f"Formatted chat history: {messages}")
        return messages
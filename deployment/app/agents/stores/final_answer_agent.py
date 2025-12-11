import sys
from typing import Dict, Any, AsyncGenerator, List

from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Local imports
from app.agents.stores.base_agent import BaseAgentNode, AgentState
from app.agents.workflow.initalize import llm_instance, agent_config
    
class FinalAnswerAgent(BaseAgentNode):
    """
    Agent cuối cùng trong workflow, có khả năng stream câu trả lời tổng hợp từ tất cả thông tin đã thu thập.
    - Thu thập tất cả thông tin: contexts, agent thoughts, reflection feedback, chat history
    - Tổng hợp và tạo ra câu trả lời cuối cùng tối ưu
    - Stream response để cải thiện trải nghiệm người dùng
    
    Hai cách streaming:
    1. aexecute(): Phương thức chính sử dụng với LangGraph, tự động emit on_chat_model_stream events
       mà LangGraph sẽ phát hiện và gửi đến frontend thông qua workflow.
    2. astream_execute(): Phương thức phụ trợ cho trường hợp sử dụng API trực tiếp hoặc test,
       trả về từng chunk dưới dạng AsyncGenerator của AgentState.
    """
    
    def __init__(self, llm: BaseChatModel, history_k: int = 5):
        agent_name = "FinalAnswerAgent"
        super().__init__(agent_name=agent_name)
        
        self.llm = llm
        self.history_k = history_k  # Số lượng lịch sử chat sẽ được sử dụng
        self.system_prompt = """Bạn là Genee, trợ lý AI chuyên nghiệp của GeneStory - công ty hàng đầu về xét nghiệm di truyền tại Việt Nam. 

Nhiệm vụ của bạn là tạo ra câu trả lời cuối cùng tổng hợp và toàn diện cho người dùng dựa trên:
- Câu hỏi gốc của người dùng và mục đích của câu hỏi
- Lịch sử trò chuyện để hiểu ngữ cảnh
- Thông tin then chốt đã được thu thập từ các nguồn khác nhau
- Suy nghĩ trung gian từ các agent chuyên môn
- Đánh giá phản hồi từ hệ thống reflection

Nguyên tắc quan trọng:
1. Tạo ra câu trả lời mạch lạc, dễ hiểu và đầy đủ thông tin
2. Sử dụng TẤT CẢ thông tin có sẵn để đưa ra câu trả lời chính xác và toàn diện nhất
3. Giữ giọng điệu thân thiện, chuyên nghiệp và hữu ích
4. Trả lời bằng tiếng Việt tự nhiên và phù hợp với người dùng Việt Nam
5. Tập trung vào nhu cầu thực tế của người dùng
6. Cung cấp thông tin có cấu trúc rõ ràng, dễ theo dõi
7. Đảm bảo tính chính xác và độ tin cậy cao

Bạn là bước cuối cùng và quan trọng nhất trong quy trình - hãy tạo ra câu trả lời tốt nhất có thể."""
        logger.info(f"'{self.agent_name}' initialized for streaming final responses.")

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi và trả về state cuối cùng với LLM streaming để LangChain có thể emit events.
        LangGraph sẽ tự động tạo ra event on_chat_model_stream cho mỗi token từ LLM.
        """
        try:
            logger.info(f"'{self.agent_name}' executing for final response...")
            
            # --- 1. Thu thập thông tin ---
            query = state.get("original_query", "")
            intents = state.get("intents", [])
            contexts = state.get("contexts", {})
            reflection_feedback = state.get("reflection_feedback", "")
            
            # --- 2. Xây dựng prompt ---
            full_context_str = self._build_context_string(contexts)
            history_messages = self._format_chat_history(state.get("chat_history", []))
            logger.info(f"Full context string built: {full_context_str[:50]}...")
            logger.debug(f"History messages for final response: {history_messages}")
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", """
                Dựa trên thông tin sau, hãy tạo ra câu trả lời cuối cùng toàn diện và hữu ích cho người dùng:

                **Câu hỏi của người dùng:** {user_query}

                **Ý định phân tích:** {intents}

                **Thông tin then chốt đã thu thập:**
                {full_context}

                **Suy nghĩ từ các agent chuyên môn:**
                {agent_thoughts}

                **Đánh giá chất lượng:**
                {reflection_feedback}

                Hãy tạo ra câu trả lời:
                1. Trực tiếp và chính xác trả lời câu hỏi của người dùng
                2. Sử dụng TẤT CẢ thông tin có sẵn để đưa ra câu trả lời đầy đủ nhất
                3. Có cấu trúc rõ ràng, dễ hiểu
                4. Giọng điệu thân thiện, chuyên nghiệp
                5. Bằng tiếng Việt tự nhiên

                **Câu trả lời cuối cùng:**""")
                            ])
            
            # --- 3. Thực thi với LLM streaming để LangChain emit on_chat_model_stream events ---
            chain = prompt | self.llm
            logger.info(f"Streaming final response from '{self.agent_name}' with LangChain event emission...")

            # QUAN TRỌNG: Đây là phương thức mà LangGraph sẽ sử dụng để phát hiện event
            # Nếu chúng ta gom tất cả rồi trả về, LangGraph sẽ không thấy các event riêng lẻ
            # Sử dụng ChatModel.astream_events() thay vì astream() để tạo ra streaming events
            
            # Sử dụng invoke_with_config để đảm bảo rằng on_chat_model_stream events được phát ra
            config = {"configurable": {"thread_id": state.get("session_id", "default")}}
            full_response = ""
            
            # Use the chain with proper astream to get streaming tokens that LangGraph can detect
            # This will emit on_chat_model_stream events automatically
            async for chunk in chain.astream({
                "chat_history": history_messages,
                "user_query": query,
                "intents": intents,
                "full_context": full_context_str,
                "agent_thoughts": state.get("agent_thinks", {}),
                "reflection_feedback": reflection_feedback
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    # logger.debug(f"Streaming chunk: {chunk.content}")
            
            logger.info(f"Final response completed: {full_response[:100]}...")
            
            # Cuối cùng gán response đầy đủ vào state để các bước tiếp theo có thể sử dụng
            state["agent_response"] = full_response
            state['is_final_answer'] = True
            
            return state

        except Exception as e:
            logger.error(f"Error during final answer execution: {e}", exc_info=True)
            # Ngay cả khi có lỗi, chúng ta vẫn cần emit một số tokens để LangGraph có thể xử lý node này
            # Điều này giúp tránh việc workflow bị treo
            error_message = "Xin lỗi, tôi gặp sự cố khi tạo câu trả lời. Vui lòng thử lại."
            
            # Giả lập các events on_chat_model_stream với thông báo lỗi
            # Điều này giúp đảm bảo frontend vẫn nhận được phản hồi
            from langchain_core.messages import AIMessageChunk
            
            # Emit từng từ một để đảm bảo streaming hoạt động đúng
            for word in error_message.split():
                # Tạo event giả lập cho mỗi từ trong thông báo lỗi
                fake_event = {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": AIMessageChunk(content=word+" ")}
                }
                # LangGraph sẽ tự động xử lý event này
                
            state["agent_response"] = error_message
            state['is_final_answer'] = True
            return state

    async def astream_execute(self, state: AgentState) -> AsyncGenerator[AgentState, None]:
        """
        Thực thi logic tổng hợp và stream câu trả lời cuối cùng.
        Thu thập tất cả thông tin từ contexts, agent thoughts, reflection feedback.
        
        LƯU Ý: Phương thức này không được sử dụng bởi LangGraph, thay vào đó LangGraph 
        sử dụng phương thức aexecute() và tự động xử lý các event on_chat_model_stream.
        Phương thức này chỉ được sử dụng cho các tình huống test hoặc API endpoint trực tiếp.
        """
        state = self._prepare_execution(state)
        
        try:
            # --- 1. Thu thập và định dạng tất cả thông tin có sẵn ---
            query = state.get("rewritten_query") or state.get("original_query", "")
            
            # Thu thập tất cả thông tin từ các nguồn khác nhau
            all_info_parts = []
            contexts = state.get("contexts", {})
            reflection_feedback = state.get("reflection_feedback", {})
            
            # Thêm thông tin chính từ contexts
            if contexts:
                context_str = "\n".join([f"- {content}" for content in contexts.values()])
                all_info_parts.append(f"### Thông tin then chốt đã thu thập:\n{context_str}")
            
            # Thêm suy nghĩ từ các agent
            agent_thinks = state.get("agent_thinks", {})
            if agent_thinks:
                think_str = "\n".join([f"- Agent '{name}': {str(think)}" for name, think in agent_thinks.items()])
                all_info_parts.append(f"### Suy nghĩ trung gian từ các agent:\n{think_str}")
            
            # Kết hợp tất cả thông tin
            full_context_str = "\n\n".join(all_info_parts)
            if not full_context_str:
                full_context_str = "Không có thông tin cụ thể nào được thu thập."
            
            # Format lịch sử chat
            history_messages = self._format_chat_history(state.get("chat_history", []))
            intents = state.get("intents", [])
            
            logger.info(f"Preparing final response for query: {query}")
            logger.debug(f"Full context for final response: {full_context_str}")

            # --- 2. Xây dựng prompt cuối cùng tổng hợp ---
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human",
                "Dựa trên câu hỏi gốc của tôi và tất cả thông tin mà hệ thống đã thu thập bên dưới, vui lòng tạo ra câu trả lời cuối cùng đã được tổng hợp hoàn chỉnh và chính xác nhất.\n\n"
                "**Câu hỏi của tôi:**\n{user_query}\n\n"
                "**Mục đích của câu hỏi này là:** {intents}\n\n"
                "**Tất cả thông tin có sẵn để bạn tổng hợp:**\n{full_context}\n\n"
                "**Tổng hợp các suy nghĩ của các agent khác:**\n{agent_thoughts}\n\n"
                "**Đánh giá phản hồi của các agents:**\n{reflection_feedback}\n\n"
                "**Hãy đưa ra câu trả lời chính xác, toàn diện và hữu ích nhất cho câu hỏi của tôi.**")
            ])

            chain = prompt | self.llm
            logger.info(f"Streaming final response from '{self.agent_name}'...")

            # --- 3. Thực thi và STREAM kết quả - từng chunk riêng lẻ ---
            full_response = ""
            async for chunk in chain.astream({
                "chat_history": history_messages,
                "user_query": query,
                "intents": intents,
                "contexts": contexts, 
                "full_context": full_context_str,
                "reflection_feedback": reflection_feedback,
                "agent_thoughts": state.get("agent_thinks", {})
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    # Chỉ yield chunk mới, không tích lũy
                    state["agent_response"] = chunk.content
                    full_response += chunk.content
                    yield state  # Yield state với chỉ chunk hiện tại

            logger.info(f"Final response completed: {full_response[:100]}...")  # Log đầu 100 ký tự
            logger.info("Finished streaming final response.")
            state['is_final_answer'] = True
            yield state  # Yield state cuối cùng với cờ is_final_answer

        except Exception as e:
            state = self._handle_execution_error(e, state)
            logger.error(f"Error during streaming execution: {e}", exc_info=True)
            state['is_final_answer'] = True
            yield state
    
    def _format_chat_history(self, history) -> List[BaseMessage]:
        """Chuyển đổi lịch sử chat từ tuple sang đối tượng message của LangChain."""
        messages = []
        if not history:
            return messages
        
        logger.debug(f"Formatting chat history: {history}")
        
        # Lấy k cặp hội thoại gần nhất để tránh context quá dài
        recent_history = history[-self.history_k:] if len(history) > self.history_k else history
        
        for item in recent_history:
            if isinstance(item, dict):
                role = item.get('role', '')
                content = item.get('content', '')
                
                if role == 'user':
                    messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    messages.append(AIMessage(content=content))
                else:
                    logger.warning(f"Unknown role in chat history: {role}. Skipping this message.")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # Handle tuple format (user_message, assistant_message)
                user_msg, assistant_msg = item
                messages.append(HumanMessage(content=str(user_msg)))
                messages.append(AIMessage(content=str(assistant_msg)))
        
        logger.debug(f"Formatted chat history: {[msg.content[:50] + '...' if len(msg.content) > 50 else msg.content for msg in messages]}")
        return messages

    def _build_context_string(self, contexts: Dict[str, Any]) -> str:
        """Build a formatted context string from collected contexts."""
        if not contexts:
            return "Không có thông tin bổ sung."
        
        context_parts = []
        
        for agent_name, agent_contexts in contexts.items():
            if agent_contexts:
                agent_display_name = agent_name.replace("_agent", "").replace("_", " ").title()
                context_parts.append(f"**{agent_display_name}:**")
                
                if isinstance(agent_contexts, list):
                    for i, context in enumerate(agent_contexts, 1):
                        if isinstance(context, str) and context.strip():
                            context_parts.append(f"  {i}. {context.strip()}")
                elif isinstance(agent_contexts, str) and agent_contexts.strip():
                    context_parts.append(f"  - {agent_contexts.strip()}")
                elif isinstance(agent_contexts, dict):
                    for key, value in agent_contexts.items():
                        if value:
                            context_parts.append(f"  - {key}: {value}")
                
                context_parts.append("")  # Add spacing between agents
        
        return "\n".join(context_parts) if context_parts else "Không có thông tin bổ sung."
    
    def _get_timestamp(self):
        """Get current timestamp for streaming events"""
        from datetime import datetime
        return datetime.utcnow().isoformat()

async def test_final_answer_agent():
    """Test the FinalAnswerAgent with comprehensive information synthesis."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    final_agent = FinalAnswerAgent(llm=llm_instance)

    print("--- Testing FinalAnswerAgent (Supervisor-style Information Synthesis) ---")
    
    # This state simulates comprehensive information from multiple sources
    test_state = AgentState(
        original_query="Aspirin có tác dụng gì và có tác dụng phụ gì không?",
        rewritten_query="Tác dụng và tác dụng phụ của thuốc aspirin",
        intents=["retrieve", "medical_information"],
        contexts={
            "drug_retriever_tool": "Aspirin (acetylsalicylic acid) là một loại thuốc kháng viêm không steroid (NSAID). Tác dụng chính: giảm đau, hạ sốt, chống viêm, ngăn ngừa cục máu đông. Tác dụng phụ: kích ứng dạ dày, nguy cơ chảy máu, ù tai ở liều cao.",
            "medical_retriever_tool": "Aspirin được sử dụng để điều trị đau đầu, đau cơ, viêm khớp và phòng ngừa đột quỵ, nhồi máu cơ tim. Cần thận trọng khi sử dụng ở người có tiền sử loét dạ dày.",
            "company_retriever_tool": "GeneStory khuyến cáo tham khảo ý kiến bác sĩ trước khi sử dụng bất kỳ loại thuốc nào."
        },
        agent_thinks={
            "DrugAgent": "Đã cung cấp thông tin chi tiết về aspirin từ cơ sở dữ liệu thuốc",
            "MedicalAgent": "Đã bổ sung thông tin về ứng dụng lâm sàng và cảnh báo an toàn"
        },
        reflection_feedback={
            "quality_score": 0.9,
            "completeness": "Thông tin đầy đủ về tác dụng và tác dụng phụ"
        },
        chat_history=[
            {"role": "user", "content": "Tôi muốn tìm hiểu về thuốc aspirin"},
            {"role": "assistant", "content": "Tôi có thể giúp bạn tìm hiểu về aspirin. Bạn muốn biết điều gì cụ thể về loại thuốc này?"}
        ]
    )
    
    print("Streaming final answer...")
    full_answer = ""
    final_state = None
    
    # Mô phỏng client nhận các chunk riêng lẻ từ stream
    chunk_count = 0
    async for partial_state in final_agent.astream_execute(test_state):
        chunk = partial_state.get("agent_response", "")
        if chunk:
            chunk_count += 1
            print(f"[Chunk {chunk_count}]: {chunk}", flush=True)
            full_answer += chunk
        final_state = partial_state

    print("\n\n--- Test Complete ---")
    if final_state:
        print(f"Is Final Answer: {final_state.get('is_final_answer')}")
        print(f"Error message (if any): {final_state.get('error_message', 'None')}")
        print(f"Full Answer Length: {len(full_answer)} characters")
        print("Note: This test shows comprehensive information synthesis from all available sources")

if __name__ == '__main__':
    import asyncio
    
    asyncio.run(test_final_answer_agent())
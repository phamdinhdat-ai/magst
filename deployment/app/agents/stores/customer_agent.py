import sys
import asyncio
import re
from typing import List, AsyncGenerator

from loguru import logger
from pathlib import Path

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
# Sửa các import này cho đúng với cấu trúc dự án của bạn
from app.agents.stores.base_agent import Agent, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.tools.base import BaseAgentTool

class CustomerAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi liên quan đến một khách hàng cụ thể.
    Nó có system_prompt động và sử dụng tool retriever động.
    Supports two main scenarios:
    - retrieve: Answer specific questions about customer's genetic report
    - summary: Provide comprehensive summary of customer's entire genetic report
    """
    def __init__(self, llm: BaseChatModel, **kwargs):
        """
        Khởi tạo CustomerAgent.
        Nó không cần tham số tool vì mọi thứ sẽ được lấy từ ToolFactory.
        """
        agent_name = "CustomerAgent"
        # Lấy template prompt. Các placeholder sẽ được điền sau.
        system_prompt_template = agent_config["customer_agent"]["description"]
        
        super().__init__(
            llm=llm,
            agent_name=agent_name,
            system_prompt=system_prompt_template,
            # Không cần default_tool_names nếi tool động là tool chính
            # Hoặc bạn có thể thêm: default_tool_names=["customer_retriever_tool"]
            # để đảm bảo nó luôn chạy. Logic trong base class đã xử lý việc này.
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized. It will request tools dynamically.")

    def _detect_intent(self, query: str) -> List[str]:
        """
        Analyze the user query to determine if they want retrieval or summary.
        
        Args:
            query: The user's query about their genetic report
            
        Returns:
            List of intents: ['retrieve'] for specific questions, ['summary'] for comprehensive summary
        """
        query_lower = query.lower()
        
        # Keywords that indicate user wants a comprehensive summary
        summary_keywords = [
            'tóm tắt', 'summary', 'summarize', 'tổng kết', 'tổng quan', 'toàn bộ', 'all', 'entire',
            'báo cáo đầy đủ', 'full report', 'complete report', 'tất cả thông tin', 'all information',
            'overview', 'comprehensive', 'hoàn chỉnh', 'chi tiết đầy đủ', 'detailed summary',
            'kết quả tổng thể', 'overall results', 'tình trạng chung', 'general condition',
            'xem tất cả', 'see all', 'show all', 'hiển thị tất cả', 'toàn diện'
        ]
        
        # Keywords that indicate user wants specific retrieval
        retrieve_keywords = [
            'có nguy cơ', 'risk of', 'nguy cơ mắc', 'prone to', 'dễ mắc', 'susceptible',
            'gen nào', 'which gene', 'gene nào', 'what gene', 'gene gì', 'genetic variant',
            'đột biến', 'mutation', 'allele', 'snp', 'biến thể gen', 'variant',
            'bệnh gì', 'what disease', 'disease', 'bệnh nào', 'which disease',
            'thuốc gì', 'what drug', 'medication', 'thuốc nào', 'which medication',
            'có thể', 'might', 'could', 'có khả năng', 'likely', 'possible',
            'liều lượng', 'dosage', 'dose', 'metabolism', 'chuyển hóa',
            'phản ứng', 'reaction', 'response', 'đáp ứng', 'effectiveness'
        ]
        
        # Check for summary intent first (more specific)
        if any(keyword in query_lower for keyword in summary_keywords):
            logger.info(f"Detected summary intent for query: {query[:50]}...")
            return ['summary']
        
        # Check for retrieve intent (more common)
        if any(keyword in query_lower for keyword in retrieve_keywords):
            logger.info(f"Detected retrieve intent for query: {query[:50]}...")
            return ['retrieve']
        
        # Pattern-based detection for questions vs summary requests
        question_patterns = [
            r'\b(có|is|are|do|does|can|will|would|should|might|could)\b',
            r'\?(.*)?',  # Questions with question marks
            r'\b(tại sao|why|how|làm sao|như thế nào|what|gì|nào)\b'
        ]
        
        summary_patterns = [
            r'\b(cho tôi|give me|show me|hiển thị|xem)\b.*\b(tất cả|all|toàn bộ|complete|full)\b',
            r'\b(báo cáo|report)\b.*\b(của tôi|my|mine)\b'
        ]
        
        # Check summary patterns first
        for pattern in summary_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Detected summary intent via pattern for query: {query[:50]}...")
                return ['summary']
        
        # Check question patterns
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Detected retrieve intent via pattern for query: {query[:50]}...")
                return ['retrieve']
        
        # Default to retrieve for genetic-related queries
        logger.info(f"Defaulting to retrieve intent for query: {query[:50]}...")
        return ['retrieve']

    def _prepare_dynamic_prompt(self, state: AgentState) -> str:
        """
        Tạo system prompt hoàn chỉnh bằng cách điền thông tin từ state.
        """
        customer_id = state.get('customer_id')
        if not customer_id:
            raise ValueError("CustomerAgent requires 'customer_id' to format its prompt.")
        
        return self.system_prompt.format(
            customer_id=customer_id
        )

    async def astream_execute(self, state: AgentState) -> AsyncGenerator[AgentState, None]:
        """
        Ghi đè luồng stream chính để:
        1. Detect intent from user query
        2. Set appropriate intent in state 
        3. Format dynamic prompt
        4. Call parent execution logic
        """
        original_prompt = self.system_prompt
        try:
            # Bước 1: Detect intent from user query
            query = state.get('original_query', '')
            if query:
                detected_intents = self._detect_intent(query)
                # Update state with detected intents, but preserve existing intents if any
                existing_intents = state.get('intents', [])
                all_intents = list(set(existing_intents + detected_intents))
                state['intents'] = all_intents
                logger.info(f"CustomerAgent detected intents: {detected_intents}, final intents: {all_intents}")
            
            # Bước 2: Chuẩn bị prompt động
            self.system_prompt = self._prepare_dynamic_prompt(state)
            
            # Bước 3: Gọi logic của lớp cha với prompt đã được cập nhật tạm thời
            async for chunk_state in super().astream_execute(state):
                # logger.debug(f"Streaming chunk state: {chunk_state}")
                yield chunk_state

        except Exception as e:
            state = self._handle_execution_error(e, state)
            yield state
        finally:
            # Bước 4 (Quan trọng): Khôi phục lại prompt gốc
            self.system_prompt = original_prompt

# ==============================================================================
# === TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        # Khởi tạo agent rất đơn giản
        customer_agent = CustomerAgent(llm=llm)
        
        # --- Test Case 1: Retrieve specific information ---
        print("--- Test Case 1: Retrieve specific genetic information ---")
        state = AgentState(
            original_query="Tôi có nguy cơ mắc bệnh tim mạch không dựa trên gen của tôi?",
            customer_id="789122254025",
            user_role="customer",
            chat_history=[]
        )

        final_state = None
        async for partial_state in customer_agent.astream_execute(state):
            print(f"Streaming response: ...{partial_state.get('agent_response', '')[-30:]}", end="\r")
            final_state = partial_state
        
        print("\n\n--- Final Result for Test Case 1 ---")
        if final_state.get('error_message'):
            print(f"Error: {final_state['error_message']}")
        else:
            print(f"Final Answer: {final_state.get('agent_response')}")
        print(f"Intents Detected: {final_state.get('intents', [])}")
        print(f"Tools Selected & Run: {list(final_state.get('contexts', {}).keys())}")
        
        # --- Test Case 2: Summary request ---
        print("\n--- Test Case 2: Genetic report summary ---")
        state_summary = AgentState(
            original_query="Tóm tắt toàn bộ báo cáo gen của tôi",
            customer_id="789122254025",
            user_role="customer",
            chat_history=[]
        )
        
        final_state_summary = None
        async for partial_state in customer_agent.astream_execute(state_summary):
            print(f"Streaming response: ...{partial_state.get('agent_response', '')[-30:]}", end="\r")
            final_state_summary = partial_state
            
        print("\n\n--- Final Result for Test Case 2 ---")
        if final_state_summary.get('error_message'):
            print(f"Error: {final_state_summary['error_message']}")
        else:
            print(f"Final Answer: {final_state_summary.get('agent_response')}")
        print(f"Intents Detected: {final_state_summary.get('intents', [])}")
        print(f"Tools Selected & Run: {list(final_state_summary.get('contexts', {}).keys())}")
        
        # --- Test Case 3: Thiếu customer_id ---
        print("\n--- Test Case 3: Missing customer_id ---")
        state_no_id = AgentState(
            original_query="Hồ sơ sức khỏe gần đây của khách hàng này có gì đáng chú ý không?",
            user_role="customer",
            chat_history=[]
        )
        
        async for final_state_no_id in customer_agent.astream_execute(state_no_id):
            pass # Chỉ cần lấy state cuối cùng
            
        print("\n--- Final Result for Test Case 3 ---")
        print(f"Error: {final_state_no_id.get('error_message')}")
        
        # --- Test Case 4: Different summary patterns ---
        print("\n--- Test Case 4: Different query patterns ---")
        test_queries = [
            "Cho tôi xem tất cả thông tin gen của tôi",  # Should be summary
            "Tôi có gen BRCA1 đột biến không?",           # Should be retrieve  
            "Báo cáo gen đầy đủ của tôi",                # Should be summary
            "Thuốc nào tôi có thể có phản ứng xấu?"     # Should be retrieve
        ]
        
        for i, query in enumerate(test_queries, 5):
            print(f"\n--- Test Case {i}: '{query}' ---")
            intents = customer_agent._detect_intent(query)
            print(f"Detected intents: {intents}")

    # Chạy kịch bản test
    asyncio.run(main())
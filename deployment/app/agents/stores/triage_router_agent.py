from pydantic import BaseModel, Field
from typing import List, Literal
import sys
from typing import List, AsyncGenerator, Dict, Any
from pathlib import Path

from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import asyncio
# Assuming a central place for Pydantic models


# Local imports
from .base_agent import BaseAgentNode, AgentState
from app.agents.workflow.initalize import llm_instance, agent_config
# The different paths the workflow can take.
NextStep = Literal[
    "direct_answer",      # For simple queries, answer directly without tools.
    "specialist_agent",   # For standard queries needing one specialist.
    "multi_agent_plan",   # For complex queries needing multiple specialists.
    "clarify_question",   # If the query is too ambiguous to proceed.
    "re_execute_query"    # If the user is dissatisfied with the previous answer.
]

class TriageOutput(BaseModel):
    """
    Defines the structured output plan for the TriageRouterAgent.
    This plan dictates the entire subsequent workflow for a given user query.
    """
    
    rewritten_query: str = Field(
    ...,
    description="Một phiên bản rõ ràng, độc lập của truy vấn từ người dùng, được viết lại sao cho có thể hiểu mà không cần lịch sử hội thoại. Nếu không cần viết lại, đây chính là truy vấn gốc."
    )

    classified_agent: str = Field(
        ...,
        description="Tác tử chuyên trách phù hợp nhất để xử lý truy vấn đã viết lại. Nếu truy vấn chỉ là trò chuyện đơn giản hoặc kiến thức chung, hãy phân loại thành 'DirectAnswerAgent'. Nếu cần nhiều tác tử chuyên trách, hãy chọn tác tử quan trọng nhất."
    )

    is_multi_step: bool = Field(
        False,
        description="Chỉ đặt thành True NẾU truy vấn rõ ràng yêu cầu thông tin từ nhiều miền chuyên môn khác nhau (ví dụ: 'So sánh giá sản phẩm với chính sách đổi trả của công ty')."
    )

    next_step: NextStep = Field(
        ...,
        description="Bước tiếp theo hợp lý nhất cho luồng xử lý dựa trên phân tích truy vấn. Đây là quyết định định tuyến chính."
    )

    clarification_question: str = Field(
        "",
        description="Nếu truy vấn chưa rõ ràng, hãy đưa ra một câu hỏi để yêu cầu người dùng cung cấp thêm chi tiết. Chỉ điền trường này nếu next_step là 'clarify_question'."
    )

    should_re_execute: bool = Field(
        False,
        description="Chỉ đặt thành True NẾU tin nhắn gần nhất của người dùng thể hiện rõ sự không hài lòng với câu trả lời TRƯỚC ĐÓ và yêu cầu thực hiện lại. Các dấu hiệu bao gồm: 'sai rồi', 'thực hiện lại', 'làm lại', 'hỏi lại', 'không đúng', 'chưa đúng', 'không hiểu', 'chưa rõ'. Trường hợp này sẽ kích hoạt bước 're_execute_query'."
    )

    original_query_from_history: str = Field(
        "",
        description="Nếu should_re_execute là True, đây là câu hỏi gốc từ lịch sử trò chuyện mà người dùng muốn thực hiện lại. Tìm câu hỏi người dùng gần nhất từ chat_history (không phải câu hiện tại)."
    )


    class Config:
        arbitrary_types_allowed = True
        


class TriageRouterAgent(BaseAgentNode):
    """
    The first and most critical agent in the workflow. It acts as a planner and router.
    - Analyzes user intent, sentiment, and query clarity.
    - Rewrites the query for standalone context.
    - Creates a structured plan (TriageOutput) that dictates the next steps in the graph.
    - This agent does NOT use tools and does not answer the user directly.
    """
    
    def __init__(self, llm: BaseChatModel):
        agent_name = "TriageRouterAgent"
        super().__init__(agent_name=agent_name)
        
        self.llm = llm
        self.system_prompt = agent_config['triage_router_agent_prompt']['description']
        
        # Add guidance for workflow-specific agent selection
        workflow_guidance = """
        IMPORTANT AGENT SELECTION CONSTRAINTS:
        - For 'guest' workflows, only use: CompanyAgent, ProductAgent, MedicalAgent, DrugAgent, GeneticAgent, or DirectAnswerAgent
        - For 'customer' workflows, you can also use: CustomerAgent, PersonalizedAgent
        - For 'employee' workflows, you can also use: EmployeeAgent, HRAgent, InternalAgent
        
        If the workflow_type is 'guest', DO NOT select any agent that would require authentication or access to private data.
        """
        
        # Create the structured LLM chain
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.system_prompt + workflow_guidance),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Workflow Type: {workflow_type}\nUser Query: {query}")
            ])
            | self.llm.with_structured_output(TriageOutput)
        )
        logger.info(f"'{self.agent_name}' initialized successfully.")

    def _format_chat_history(self, history: List[Dict[str, str]], k: int = 3) -> List[BaseMessage]:
        """Formats the last k interactions from chat history."""
        messages = []
        if not history:
            return messages
        
        # Each item in history is a dict {'role': 'user'/'assistant', 'content': '...'}
        for item in history[-k*2:]: # Get last k pairs
            if item.get('role') == 'user':
                messages.append(HumanMessage(content=item['content']))
            elif item.get('role') == 'assistant':
                messages.append(AIMessage(content=item['content']))
        return messages

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Asynchronously executes the planning and routing logic.
        """
        state = self._prepare_execution(state)
        query = state.get('original_query', '')
        chat_history = self._format_chat_history(state.get('chat_history', []))
        workflow_type = state.get('workflow_type', 'unknown')
        
        logger.info(f"Executing Triage for query: '{query}' in workflow type: {workflow_type}")
        
        try:
            # Invoke the chain to get the structured plan
            plan: TriageOutput = await self.chain.ainvoke({
                "query": query,
                "chat_history": chat_history,
                "workflow_type": workflow_type,
            })
            
            logger.info(f"Triage plan received: next_step='{plan.next_step}', agent='{plan.classified_agent}', multi_step={plan.is_multi_step}")
            logger.debug(f"Full Triage Plan: {plan.model_dump()}")

            # Handle re-execution logic - if user is dissatisfied and wants to re-execute
            if plan.should_re_execute and hasattr(plan, 'original_query_from_history') and plan.original_query_from_history:
                logger.info(f"Re-execution requested. Original query from history: '{plan.original_query_from_history}'")
                # Override the rewritten_query to be the original query from history
                plan.rewritten_query = plan.original_query_from_history
                # Ensure we route to re_execute_query path
                plan.next_step = "re_execute_query"
                logger.info(f"Re-execution setup: rewritten_query='{plan.rewritten_query}', next_step='{plan.next_step}'")

            # For guest workflow, ensure the agent is compatible
            if workflow_type == 'guest':
                valid_guest_agents = [
                    "CompanyAgent", "ProductAgent", "MedicalAgent",
                    "DrugAgent", "GeneticAgent", "DirectAnswerAgent"
                ]
                
                # If the triage selects an invalid agent for guest workflow, fall back to a safe agent
                if plan.classified_agent not in valid_guest_agents:
                    logger.warning(f"Agent '{plan.classified_agent}' selected by triage is not valid for guest workflow. Falling back to DirectAnswerAgent.")
                    plan.classified_agent = "DirectAnswerAgent"
                    # If we need specialist path but have to fallback, go to direct answer
                    if plan.next_step == "specialist_agent":
                        plan.next_step = "direct_answer"

            # Update the graph state with the plan
            state['rewritten_query'] = plan.rewritten_query
            state['classified_agent'] = plan.classified_agent
            state['next_step'] = plan.next_step
            state['is_multi_step'] = plan.is_multi_step
            state['should_re_execute'] = plan.should_re_execute
            
            # Handle re-execution specific state updates
            if plan.should_re_execute:
                state['re_execution_requested'] = True
                if hasattr(plan, 'original_query_from_history') and plan.original_query_from_history:
                    state['original_query_from_history'] = plan.original_query_from_history
                    logger.info(f"Stored original query from history for re-execution: '{plan.original_query_from_history}'")
            
            if plan.next_step == 'clarify_question':
                # If clarification is needed, this becomes the primary response
                state['agent_response'] = plan.clarification_question

        except Exception as e:
            logger.error(f"Error during TriageRouterAgent execution: {e}", exc_info=True)
            state = self._handle_execution_error(e, state)
            
            # CRITICAL FALLBACK: If triage fails, we default to a safe, simple path.
            logger.warning("Triage failed. Applying default fallback plan.")
            state['rewritten_query'] = query
            state['classified_agent'] = "DirectAnswerAgent"
            state['next_step'] = "direct_answer" # The safest fallback is a direct answer
            state['is_multi_step'] = False
            state['should_re_execute'] = False

        return state
async def test_triage_agent():
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    triage_agent = TriageRouterAgent(llm=llm_instance)

    test_cases = [
        {"name": "Simple Greeting", "query": "Hello there", "history": []},
        {"name": "Standard Specialist Query", "query": "What are the side effects of Aspirin?", "history": []},
        {"name": "Query needing rewrite", "query": "what about for my heart?", "history": [{'role': 'user', 'content': 'Is aspirin good for pain?'}, {'role': 'assistant', 'content': 'Yes, it is an effective pain reliever.'}]},
        {"name": "Ambiguous Query", "query": "Tell me about that product", "history": []},
        {"name": "Dissatisfied User (Re-execution)", "query": "No, explain the risks better", "history": [{'role': 'user', 'content': 'What are the risks of taking a daily aspirin?'}, {'role': 'assistant', 'content': 'Daily aspirin can help prevent heart attacks but may cause bleeding.'}]},
        {"name": "Multi-step Query", "query": "Can you compare the side effects of Lipitor with your company's return policy?", "history": []}
    ]
    
    for case in test_cases:
        print(f"\n--- Testing Case: {case['name']} ---")
        print(f"Query: {case['query']}")
        
        initial_state = AgentState(
            original_query=case['query'],
            chat_history=case['history']
        )
        
        result_state = await triage_agent.aexecute(initial_state)
        
        print(f"  - Rewritten Query: {result_state.get('rewritten_query')}")
        print(f"  - Next Step: {result_state.get('next_step')}")
        print(f"  - Classified Agent: {result_state.get('classified_agent')}")
        print(f"  - Multi-Step: {result_state.get('is_multi_step')}")
        print(f"  - Re-execute: {result_state.get('should_re_execute')}")
        if result_state.get('next_step') == 'clarify_question':
            print(f"  - Clarification: {result_state.get('agent_response')}")
if __name__ == '__main__':
    # Example of how to test the agent in isolation


    asyncio.run(test_triage_agent())
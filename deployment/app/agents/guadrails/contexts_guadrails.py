import asyncio
from typing import Dict, Any, List
from loguru import logger

# --- Local/App Imports ---
# Giả sử các file này đã tồn tại và hoạt động đúng
from app.agents.workflow.initalize import llm_instance # Sử dụng cùng một LLM instance để nhất quán
from app.agents.workflow.state import GraphState as AgentState
from app.agents.stores.base_agent import BaseAgentNode # Kế thừa từ lớp cơ sở chung
from app.agents.factory.tools.search_tool import SearchTool
from langchain_core.language_models.chat_models import BaseChatModel
# --- LangChain Core Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from pydantic import BaseModel, Field

# --- Cấu hình Prompt cho Guardrail ---
# Prompt này được thiết kế để yêu cầu LLM đưa ra một quyết định rõ ràng (CÓ/KHÔNG)
# và cung cấp lý do, giúp việc parsing kết quả trở nên đáng tin cậy.
class GuardrailDecisionOutput(BaseModel):
    is_relevant: bool = Field(..., description="Whether the context is relevant to the user's question.")
    reason: str = Field(..., description="Reasoning behind the relevance decision.")
    relevant_context: str = Field("", description="The relevant context if applicable.")

GUARDRAIL_PROMPT_TEMPLATE = """
Bạn là một trợ lý kiểm duyệt thông minh với nhiệm vụ đánh giá xem các thông tin ngữ cảnh được cung cấp có thực sự liên quan và hữu ích để trả lời một câu hỏi cụ thể của người dùng hay không.

### Nhiệm vụ:
Dựa vào **Câu hỏi của người dùng** và **Thông tin ngữ cảnh từ các công cụ**, hãy đưa ra quyết định:

1.  **Đánh giá:** Thông tin ngữ cảnh có chứa đủ dữ kiện để trả lời trực tiếp và chính xác câu hỏi không?
2.  **Quyết định:** Đưa ra câu trả lời cuối cùng ở định dạng "QUYẾT ĐỊNH: [CÓ/KHÔNG]".
3.  **Lý do:** Giải thích ngắn gọn cho quyết định của bạn trong một dòng.

---
### Ví dụ 1:
**Câu hỏi của người dùng:** "Tác dụng phụ của thuốc Paracetamol là gì?"
**Thông tin ngữ cảnh từ các công cụ:** "--- Kết quả từ công cụ: drug_retriever_tool ---\nParacetamol là một loại thuốc giảm đau, hạ sốt. Các tác dụng phụ thường gặp bao gồm phát ban da và các phản ứng dị ứng khác. Sử dụng quá liều có thể gây tổn thương gan nghiêm trọng."
**Phân tích của bạn:**
{{
    "is_relevant": True,
    "reason": "Thông tin ngữ cảnh đã liệt kê rõ ràng các tác dụng phụ của Paracetamol.",
    "relevant_context": "Paracetamol là một loại thuốc giảm đau, hạ sốt. Các tác dụng phụ thường gặp bao gồm phát ban da và các phản ứng dị ứng khác. Sử dụng quá liều có thế gây tổn thương gan nghiêm trọng."
}}

---
### Ví dụ 2:
**Câu hỏi của người dùng:** "Chính sách nghỉ phép của công ty GeneStory như thế nào?"
**Thông tin ngữ cảnh từ các công cụ:** "--- Kết quả từ công cụ: searchweb_tool ---\nGeneStory là một công ty công nghệ sinh học hàng đầu. Trang web chính thức là genestory.com."
**Phân tích của bạn:**
{{
    "is_relevant": False,
    "reason": "Thông tin ngữ cảnh chỉ giới thiệu chung về công ty, không đề cập đến chính sách nghỉ phép.",
    "relevant_context": ""
}}

---
### Yêu cầu thực tế:

**Câu hỏi của người dùng:**
{query}

**Thông tin ngữ cảnh từ các công cụ:**
{context}

**Phân tích của bạn:**
"""

class ContextGuardrail(BaseAgentNode):
    """    Node kiểm duyệt ngữ cảnh để xác định tính liên quan của thông tin.
    Nó sử dụng LLM để đánh giá xem các thông tin ngữ cảnh có đủ dữ kiện để trả lời câu hỏi của người dùng hay không.
    Nếu không, nó sẽ đánh dấu là không liên quan và cung cấp lý do.
    """ 
    def __init__(self, llm: BaseChatModel):
        super().__init__(agent_name="ContextGuardrail")
        self.llm = llm
        # self.chain = None
        # Xây dựng chain xử lý logic cho guardrail
        prompt = ChatPromptTemplate.from_template(GUARDRAIL_PROMPT_TEMPLATE)
        # StrOutputParser để lấy kết quả dạng chuỗi thô từ LLM
        self.chain = prompt | self.llm.with_structured_output(GuardrailDecisionOutput)
        self.search_tool = SearchTool()  # Khởi tạo công cụ tìm kiếm nếu cần
        logger.info(f"Node '{self.agent_name}' initialized successfully.")

    def _parse_decision(self, llm_output: str) -> Dict[str, Any]:
        """
        Phân tích chuỗi đầu ra từ LLM để lấy quyết định và lý do.
        Hàm này được thiết kế để hoạt động ổn định ngay cả khi LLM không tuân thủ định dạng 100%.
        """
        is_relevant = False
        reason = "Không thể xác định được sự liên quan từ phản hồi của mô hình."
        
        # Chuyển đổi sang chữ thường để tìm kiếm không phân biệt hoa/thường
        output_lower = llm_output.lower()
        
        # Tìm dòng chứa "QUYẾT ĐỊNH:"
        decision_line = next((line for line in output_lower.split('\n') if 'quyết định:' in line), None)
        
        if decision_line:
            if 'có' in decision_line or 'yes' in decision_line:
                is_relevant = True

        # Tìm dòng chứa "LÝ DO:"
        reason_line = next((line for line in llm_output.split('\n') if 'LÝ DO:' in line or 'Lý do:' in line), None)
        if reason_line:
            # Lấy phần văn bản sau "LÝ DO:"
            reason = reason_line.split(':', 1)[-1].strip()
        
        logger.info(f"Guardrail decision: Relevant={is_relevant}, Reason='{reason}'")
        return {"is_relevant": is_relevant, "reason": reason}

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi node guardrail.
        """
        state = self._prepare_execution(state)
        logger.info(f"--- Executing: {self.agent_name} ---")

        try:
            query = state.get("rewritten_query") or state.get("original_query")
            contexts = state.get("contexts", {})

            # Nếu không có context nào được truy xuất, mặc định là không liên quan.
            if not contexts:
                logger.warning("No context found to evaluate. Marking as not relevant.")
                state['is_context_relevant'] = False
                state['relevance_reason'] = "Không có thông tin nào được truy xuất từ các công cụ."
                return state

            # Ghép nối các context lại thành một chuỗi duy nhất
            context_str = "\n\n".join(contexts.values())

            # Gọi chain để LLM đưa ra quyết định
            llm_decision_output = await self.chain.ainvoke({
                "query": query,
                "context": context_str
            })

            # Phân tích kết quả từ LLM
            state['is_context_relevant'] = llm_decision_output.is_relevant
            state['relevance_reason'] = llm_decision_output.reason
            # Giữ nguyên context để sử dụng sau này
            

            if not llm_decision_output.is_relevant:
                logger.warning(f"Context deemed NOT relevant for query '{query}'. Reason: {llm_decision_output.reason}")
                # Nếu không liên quan, ta có thể đặt một câu trả lời mặc định
                
                state['agent_response'] = (
                    "Xin lỗi, tôi không tìm thấy thông tin cụ thể trong tài liệu để trả lời câu hỏi của bạn. "
                    "Bạn có muốn tôi thử tìm kiếm trên web không?"
                )
                # neu context khong lien quan, co the chay tool tim kiem
                if self.search_tool:
                    logger.info("Context is not relevant. Running search tool to find more information.")
                    
                    search_results = await self.search_tool.arun(query)
                    search_str = "Ket qua tìm kiếm:\n".join(
                        f"- {result}" for result in search_results if result
                    )
                    logger.info(f"Search results: {search_str}")
                    relevant_context = llm_decision_output.relevant_context or search_str
                    state['contexts'] = {"Context by Guardrail": relevant_context}

        except Exception as e:
            state = self._handle_execution_error(e, state)

        return state


context_guardrail_node = ContextGuardrail(llm=llm_instance)
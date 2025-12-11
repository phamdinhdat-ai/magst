from pydantic import BaseModel, Field
from typing import List, Literal
import sys
from typing import List, AsyncGenerator, Dict, Any
from pathlib import Path
from enum import Enum

from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import asyncio

# Local imports
from .base_agent import BaseAgentNode, AgentState
from app.agents.workflow.initalize import llm_instance, agent_config

# The different paths the workflow can take.
NextStep = Literal[
    "direct_answer",      # For simple queries, answer directly without tools.
    "specialist_agent",   # For standard queries needing one specialist.
    "multi_agent_plan",   # For complex queries needing multiple specialists.
    "clarify_question",   # If the query is too ambiguous to proceed.
    "re_execute_query",   # If the user is dissatisfied with the previous answer.
    "toxic_content_block" # If the query contains toxic content.
]

# Enum for strict agent validation
class AgentName(str, Enum):
    COMPANY = "CompanyAgent"
    PRODUCT = "ProductAgent"
    CUSTOMER = "CustomerAgent"
    EMPLOYEE = "EmployeeAgent"
    MEDICAL = "MedicalAgent"
    DRUG = "DrugAgent"
    GENETIC = "GeneticAgent"
    DIRECT_ANSWER = "DirectAnswerAgent"
    VISUAL = "VisualAgent"
    NAIVE = "NaiveAgent"

class TriageGuardrailOutput(BaseModel):
    """
    Simplified structured output for the TriageGuardrailAgent.
    Focuses on core triage functions: toxicity detection, agent classification, and query rewriting.
    """
    
    rewritten_query: str = Field(
        ...,
        description="Một phiên bản rõ ràng, độc lập của truy vấn từ người dùng, được viết lại sao cho có thể hiểu mà không cần lịch sử hội thoại. Nếu không cần viết lại, đây chính là truy vấn gốc."
    )

    is_toxic: bool = Field(
        False,
        description="Xác định xem truy vấn có chứa nội dung độc hại, không phù hợp, vi phạm chính sách không. Ví dụ: ngôn từ thù địch, bạo lực, khiêu dâm, phân biệt đối xử, hoặc yêu cầu thông tin có thể gây hại."
    )

    toxicity_reason: str = Field(
        "",
        description="Nếu is_toxic=True, giải thích ngắn gọn lý do tại sao truy vấn được coi là độc hại. Để trống nếu không độc hại."
    )

    safety_response: str = Field(
        "",
        description="Nếu is_toxic=True, cung cấp một phản hồi lịch sự để từ chối trả lời và hướng dẫn người dùng đặt câu hỏi phù hợp hơn."
    )

    classified_agent: AgentName = Field(
        ...,
        description="CHÍNH XÁC tên agent từ danh sách cho phép. CHỈ được chọn MỘT trong các tên sau: 'CompanyAgent', 'ProductAgent', 'CustomerAgent', 'EmployeeAgent', 'MedicalAgent', 'DrugAgent', 'GeneticAgent', 'DirectAnswerAgent', 'VisualAgent', 'NaiveAgent'. KHÔNG được sử dụng tên khác. CustomerAgent chỉ cho customer workflow, EmployeeAgent chỉ cho employee workflow."
    )

    need_analysis: bool = Field(
        False,
        description="True nếu truy vấn cần phân tích thêm để kiểm tra liên quan đến chat session hoặc context phức tạp. False nếu có thể chuyển thẳng đến classified_agent."
    )

    confidence_score: float = Field(
        0.8,
        description="Điểm tin cậy từ 0.0 đến 1.0 về độ chính xác của việc phân loại agent và phân tích độc tính."
    )

    # Hybrid Intent Detection Fields (for internal processing)
    detected_intents: List[str] = Field(
        default_factory=list,
        description="Danh sách các ý định được phát hiện trong truy vấn (personal, genetic, medical, product, company, account, drug)"
    )

    is_hybrid_query: bool = Field(
        False,
        description="True nếu truy vấn chứa nhiều ý định hoặc kết hợp cá nhân + tổng quát"
    )

    primary_intent: str = Field(
        "general",
        description="Ý định chính được ưu tiên cho việc định tuyến (personal > account > domain-specific > general)"
    )

    complexity_score: float = Field(
        1.0,
        description="Điểm phức tạp từ 1.0-5.0 dựa trên số lượng ý định và chỉ báo phức tạp"
    )

    class Config:
        arbitrary_types_allowed = True


class TriageGuardrailAgent(BaseAgentNode):
    """
    Enhanced triage agent with built-in guardrails, toxicity detection, and hybrid intent analysis.
    
    Key features:
    1. Analyzes user intent and query clarity
    2. Detects toxic content and provides safety responses
    3. Classifies queries to appropriate agents with hybrid intent support
    4. Creates structured execution plans
    5. Provides comprehensive examples and reasoning
    6. Handles complex multi-intent queries with priority routing
    7. Performance-optimized intent detection
    """
    
    # Enhanced keyword sets for hybrid intent detection
    PERSONAL_KEYWORDS = [
        "của tôi", "của mình", "cho tôi", "tôi muốn", "tôi cần",
        "làn da của mình", "kết quả của tôi", "tài khoản của tôi",
        "gen của tôi", "xét nghiệm của tôi", "sức khỏe của tôi",
        "báo cáo của tôi", "hồ sơ của tôi", "dữ liệu của tôi"
    ]
    
    DOMAIN_KEYWORDS = {
        "genetic": ["gen", "dna", "rna", "di truyền", "gen học", "nhiễm sắc thể", "allele", "brca"],
        "medical": ["sức khỏe", "y tế", "bệnh", "chỉ số", "triệu chứng", "điều trị", "lão hóa", "da"],
        "product": ["gói", "dịch vụ", "sản phẩm", "giá", "so sánh", "mua", "đặt hàng", "premium", "basic"],
        "company": ["công ty", "chi nhánh", "liên hệ", "về chúng tôi", "đội ngũ", "genestory"],
        "account": ["tài khoản", "đăng nhập", "mật khẩu", "thông tin", "cập nhật", "hồ sơ"],
        "drug": ["thuốc", "dược", "tác dụng phụ", "liều dùng", "tương tác"]
    }
    
    COMPLEXITY_INDICATORS = [
        "so sánh", "và", "cùng với", "ngoài ra", "thêm vào đó", 
        "kết hợp", "bao gồm", "cả", "không chỉ", "mà còn"
    ]
    
    def __init__(self, llm: BaseChatModel):
        agent_name = "TriageGuardrailAgent"
        super().__init__(agent_name=agent_name)
        
        self.llm = llm
        self.system_prompt = self._build_comprehensive_prompt()
        
        # Create the structured LLM chain
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Loại Workflow: {workflow_type}\nNếu có cuộc hội thoại trước đó, quan tâm đến  nội dung để thực thi câu quy vấn của người dùng: {query}")
            ])
            | self.llm.with_structured_output(TriageGuardrailOutput)
        )
        logger.info(f"'{self.agent_name}' initialized successfully with enhanced guardrails and hybrid intent detection.")

    def _detect_personal_intent(self, query: str) -> bool:
        """Detect if query contains personal intent indicators."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.PERSONAL_KEYWORDS)
    
    def _detect_domain_intents(self, query: str) -> List[str]:
        """Detect which domains the query relates to."""
        query_lower = query.lower()
        detected_domains = []
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)
        
        return detected_domains
    
    def _calculate_complexity_score(self, query: str, detected_intents: List[str]) -> float:
        """Calculate query complexity based on intents and indicators."""
        base_complexity = 1.0
        
        # Add complexity for multiple domains
        base_complexity += len(detected_intents) * 0.5
        
        # Add complexity for comparison/combination indicators
        query_lower = query.lower()
        complexity_indicators_found = sum(1 for indicator in self.COMPLEXITY_INDICATORS 
                                         if indicator in query_lower)
        base_complexity += complexity_indicators_found * 0.3
        
        # Add complexity for personal + general combination
        if self._detect_personal_intent(query) and any(word in query_lower 
                                                      for word in ["là gì", "như thế nào", "giải thích"]):
            base_complexity += 0.5
        
        return min(5.0, base_complexity)
    
    def _determine_primary_intent(self, query: str, detected_domains: List[str]) -> str:
        """Determine the primary intent for hybrid queries."""
        # Priority order: personal > account > specific domains > general
        if self._detect_personal_intent(query):
            return "personal"
        elif "account" in detected_domains:
            return "account"
        elif len(detected_domains) == 1:
            return detected_domains[0]
        elif "product" in detected_domains:
            return "product"  # Product queries often have priority for business
        elif "company" in detected_domains:
            return "company"
        elif detected_domains:
            return detected_domains[0]  # First detected domain
        else:
            return "general"

    def _build_comprehensive_prompt(self) -> str:
        return """
            Bạn là "Triage Guardrail Agent" – agent phân luồng chính cho hệ thống trợ lý AI GeneStory. 

            ## VAI TRÒ VÀ TRÁCH NHIỆM CHÍNH ##

            ### 1. PHÂN TÍCH VÀ VIẾT LẠI TRUY VẤN
            - Đọc hiểu ý định thực sự của người dùng
            - Viết lại truy vấn để độc lập với ngữ cảnh hội thoại
            - Đảm bảo truy vấn rõ ràng và đầy đủ

            ### 2. KIỂM TRA ĐỘC TÍNH VÀ AN TOÀN (TOXICITY DETECTION)
            Bạn PHẢI từ chối các truy vấn chứa:
            - **Ngôn từ thù địch**: Chửi bới, kỳ thị sắc tộc, tôn giáo, giới tính
            - **Nội dung bạo lực**: Mô tả bạo lực, đe dọa, tự tử
            - **Thông tin có hại**: Hướng dẫn làm vũ khí, chất độc, hoạt động bất hợp pháp
            - **Nội dung khiêu dâm**: Mô tả tình dục, khiêu dâm
            - **Spam/Lừa đảo**: Quảng cáo bất hợp pháp, lừa đảo tài chính
            - **Vi phạm quyền riêng tư**: Yêu cầu thông tin cá nhân của người khác

            **LƯU Ý**: Các thuật ngữ y tế và khoa học bình thường như "lão hóa da", "gen", "xét nghiệm", "sức khỏe" KHÔNG được coi là độc hại.

            ### 3. PHÂN LOẠI AGENT CHUYÊN TRÁCH
            **DANH SÁCH AGENT CHO PHÉP:**

            **"CompanyAgent"**: Thông tin về GeneStory (lịch sử, sứ mệnh, địa điểm, liên hệ, chính sách công ty)

            **"ProductAgent"**: Chi tiết sản phẩm xét nghiệm gen và dịch vụ, so sánh gói, giá cả, quy trình

            **"CustomerAgent"** (CHỈ CUSTOMER WORKFLOW): Thông tin tài khoản CỤ THỂ, kết quả xét nghiệm CÁ NHÂN, lịch sử đơn hàng CÁ NHÂN

            **"EmployeeAgent"** (CHỈ EMPLOYEE WORKFLOW): Hỗ trợ nhân viên nội bộ, quy trình công việc, chính sách nội bộ

            **"GeneticAgent"**: Kiến thức di truyền học TỔNG QUÁT, giải thích khái niệm gen, DNA, RNA

            **"MedicalAgent"**: Thông tin y khoa TỔNG QUÁT, thuật ngữ y tế, chăm sóc sức khỏe cơ bản

            **"DrugAgent"**: Thông tin thuốc, dược phẩm tổng quát, tác dụng phụ, tương tác thuốc

            **"VisualAgent"**: Phân tích hình ảnh, biểu đồ, charts

            **"DirectAnswerAgent"**: Câu hỏi đơn giản, trò chuyện thường ngày, chào hỏi

            **"NaiveAgent"**: Fallback cho các truy vấn không phân loại được

            ### 4. XÁC ĐỊNH CẦN PHÂN TÍCH THÊM (need_analysis)

            **Đặt need_analysis = True CHỈ KHI:**
            - Truy vấn CỰC KỲ MƠ HỒ với tham chiếu không rõ ("Cái đó", "Nó", "Cái này", "Điều đó")
            - Thiếu hoàn toàn ngữ cảnh để định tuyến (ví dụ: "Bao nhiều tiền?" mà không biết nói về gì)
            - Câu hỏi phụ thuộc hoàn toàn vào tin nhắn trước trong cuộc trò chuyện

            **Đặt need_analysis = False cho hầu hết các truy vấn rõ ràng:**
            - "Tôi muốn hiểu về chỉ số lão hóa da" → FALSE (rõ ràng, cá nhân)
            - "Chỉ số lão hóa da là gì?" → FALSE (rõ ràng, tổng quát) 
            - "Gen BRCA1 có nghĩa gì?" → FALSE (rõ ràng, tổng quát)
            - "So sánh gói Premium và Basic" → FALSE (rõ ràng, sản phẩm)
            - "GeneStory có những gói xét nghiệm nào?" → FALSE (rõ ràng, sản phẩm)
            - "Kết quả xét nghiệm của tôi như thế nào?" → FALSE (rõ ràng, cá nhân)

            **NGUYÊN TẮC QUAN TRỌNG**: Hầu hết các truy vấn cụ thể và rõ ràng KHÔNG cần phân tích thêm.
            - Có thể chuyển thẳng đến classified_agent
            - Không cần thông tin thêm từ chat history
            - Đối với các câu truy vấn không liên quan đế công ty, khách hàng, báo cáo Gene thuốc và dược phẩm, hãy chuyển thẳng đến DirectAnswerAgent để đưa ra trả lời. 

            ## LOGIC PHÂN LOẠI NGHIÊM NGẶT ##

            **BƯỚC 1: KIỂM TRA TỪ KHÓA CÁ NHÂN**
            - Có "của tôi" HOẶC "của mình" HOẶC "làn da của mình" HOẶC "gen của tôi" HOẶC "kết quả của tôi" → CÁ NHÂN
            - Không có các từ này → TỔNG QUÁT

            **BƯỚC 2: ÁP DỤNG QUY TẮC**
            - Nếu CÁ NHÂN + Customer workflow → **CustomerAgent** (BẮT BUỘC)
            - Nếu CÁ NHÂN + Employee workflow → **EmployeeAgent** (BẮT BUỘC)
            - Nếu TỔNG QUÁT + bất kỳ workflow → **MedicalAgent/GeneticAgent/ProductAgent/CompanyAgent**
            - Nếu Guest workflow → **KHÔNG BAO GIỜ CustomerAgent/EmployeeAgent**

            ## VÍ DỤ PHÂN TÍCH ##

            ### Ví dụ 1: Truy vấn rõ ràng - Không cần phân tích thêm
            ```
            User: "GeneStory có những gói xét nghiệm nào?"
            Analysis:
            - is_toxic: False
            - classified_agent: "ProductAgent"
            - need_analysis: False
            - rewritten_query: "GeneStory có những gói xét nghiệm nào?"
            ```

            ### Ví dụ 2: Truy vấn cá nhân rõ ràng - Không cần phân tích thêm
            ```
            User: "Tôi muốn xem kết quả xét nghiệm của mình"
            Analysis:
            - is_toxic: False
            - classified_agent: "CustomerAgent"
            - need_analysis: False
            - rewritten_query: "Xem kết quả xét nghiệm cá nhân của khách hàng"
            ```

            ### Ví dụ 2B: Truy vấn y tế tổng quát rõ ràng - Không cần phân tích thêm
            ```
            User: "Chỉ số lão hóa da là gì?"
            Analysis:
            - is_toxic: False
            - classified_agent: "MedicalAgent"
            - need_analysis: False
            - rewritten_query: "Chỉ số lão hóa da là gì?"
            ```

            ### Ví dụ 2C: Truy vấn cá nhân về y tế - Không cần phân tích thêm
            ```
            User: "Tôi muốn hiểu về chỉ số lão hóa da của mình"
            Analysis:
            - is_toxic: False
            - classified_agent: "CustomerAgent"
            - need_analysis: False
            - rewritten_query: "Giải thích chỉ số lão hóa da cho khách hàng cụ thể"
            ```

            ### Ví dụ 3: Truy vấn mơ hồ - Cần phân tích thêm
            ```
            User: "Cái đó bao nhiều tiền?"
            Analysis:
            - is_toxic: False
            - classified_agent: "ProductAgent"
            - need_analysis: True
            - rewritten_query: "Hỏi về giá của sản phẩm/dịch vụ (cần làm rõ sản phẩm cụ thể)"
            ```

            ### Ví dụ 4: Người dùng không hài lòng - Cần phân tích thêm
            ```
            Chat history: [
            {{user: "GeneStory có chi nhánh ở đâu?"}},
            {{assistant: "GeneStory có trụ sở tại Hà Nội."}}
            ]
            User: "Không, tôi muốn biết tất cả chi nhánh"
            Analysis:
            - is_toxic: False
            - classified_agent: "CompanyAgent"
            - need_analysis: True
            - rewritten_query: "Thông tin chi tiết về tất cả chi nhánh của GeneStory"
            ```

            **LƯU Ý QUAN TRỌNG**: 
            1. Toàn bộ đầu ra PHẢI là JSON tuân thủ schema `TriageGuardrailOutput`
            2. Trường `classified_agent` CHỈ được chứa tên chính xác từ danh sách enum
            3. TUYỆT ĐỐI KHÔNG sử dụng tên khác như "medical", "support", "Skin Care Advisor"
            4. Ưu tiên tính rõ ràng và an toàn trong mọi quyết định
            """

    def _format_chat_history(self, history: List[Dict[str, str]], k: int = 3) -> List[BaseMessage]:
        """Formats the last k interactions from chat history."""
        messages = []
        if not history:
            return messages
        
        for item in history[-k*2:]:
            if item.get('role') == 'user':
                messages.append(HumanMessage(content=item['content']))
            elif item.get('role') == 'assistant':
                messages.append(AIMessage(content=item['content']))
        return messages

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Simplified triage execution focusing on core responsibilities:
        1. Toxicity detection
        2. Agent classification  
        3. Query rewriting
        4. Determining if further analysis is needed
        """
        state = self._prepare_execution(state)
        query = state.get('original_query', '')
        chat_history = self._format_chat_history(state.get('chat_history', []))
        workflow_type = state.get('workflow_type', 'unknown')
        
        logger.info("Executing Triage Guardrail for query: '{}' in workflow type: {}".format(query, workflow_type))
        
        try:
            # Invoke the chain to get the structured plan
            plan = await self.chain.ainvoke({
                "query": query,
                "chat_history": chat_history,
                "workflow_type": workflow_type,
            })
            
            logger.info("Guardrail analysis: toxic={}, agent='{}', need_analysis={}".format(
                plan.is_toxic, plan.classified_agent, plan.need_analysis))
            
            # Handle toxic content first
            if plan.is_toxic:
                logger.warning("Toxic content detected: {}".format(plan.toxicity_reason))
                state['agent_response'] = plan.safety_response
                state['is_toxic'] = True
                state['toxicity_reason'] = plan.toxicity_reason
                state['next_step'] = 'toxic_content_block'
                return state

            # Validate agent selection based on workflow type
            valid_agents = self._get_valid_agents_for_workflow(workflow_type)
            agent_name = plan.classified_agent.value if isinstance(plan.classified_agent, AgentName) else str(plan.classified_agent)
            if agent_name not in valid_agents:
                logger.warning("Agent '{}' not valid for {} workflow. Falling back.".format(agent_name, workflow_type))
                plan.classified_agent = AgentName.DIRECT_ANSWER
                agent_name = "DirectAnswerAgent"

            # Perform hybrid intent analysis (for internal processing)
            detected_domains = self._detect_domain_intents(query)
            is_personal = self._detect_personal_intent(query)
            detected_intents = []
            
            if is_personal:
                detected_intents.append("personal")
            detected_intents.extend(detected_domains)
            
            is_hybrid = len(detected_intents) > 1 or (is_personal and any(word in query.lower() 
                                                                       for word in ["là gì", "như thế nào"]))
            primary_intent = self._determine_primary_intent(query, detected_domains)
            complexity_score = self._calculate_complexity_score(query, detected_intents)
            
            # Log hybrid intent analysis
            if is_hybrid:
                logger.info("Hybrid query detected: intents={}, primary='{}'".format(detected_intents, primary_intent))

            # Update the graph state with simplified output
            state['rewritten_query'] = plan.rewritten_query
            state['classified_agent'] = agent_name
            state['need_analysis'] = plan.need_analysis
            state['confidence_score'] = plan.confidence_score
            state['is_toxic'] = False
            
            # Add hybrid intent fields to state (for internal processing)
            state['detected_intents'] = detected_intents
            state['is_hybrid_query'] = is_hybrid
            state['primary_intent'] = primary_intent
            state['complexity_score'] = complexity_score
            
            # Determine next step based on need_analysis
            if plan.need_analysis:
                state['next_step'] = 'need_analysis'
            else:
                state['next_step'] = 'specialist_agent'

        except Exception as e:
            logger.error(f"Error during TriageGuardrailAgent execution: {str(e)}", exc_info=True)
            state = self._handle_execution_error(e, state)
            
            # Safe fallback
            logger.warning("Guardrail triage failed. Applying safe fallback.")
            state['rewritten_query'] = query
            state['classified_agent'] = "DirectAnswerAgent"
            state['next_step'] = "specialist_agent"
            state['need_analysis'] = False
            state['is_toxic'] = False
            state['confidence_score'] = 0.3

        return state

    def _get_valid_agents_for_workflow(self, workflow_type: str) -> List[str]:
        """Return list of valid agents based on workflow type."""
        base_agents = [
            "CompanyAgent", "ProductAgent", "MedicalAgent", "DrugAgent", "GeneticAgent", 
            "VisualAgent", "DirectAnswerAgent", "NaiveAgent"
        ]
        
        if workflow_type == 'customer':
            base_agents.append("CustomerAgent")
        elif workflow_type == 'employee':
            base_agents.append("EmployeeAgent")
        
        return base_agents


async def test_triage_guardrail_agent():
    """Test function for the enhanced triage agent."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    triage_agent = TriageGuardrailAgent(llm=llm_instance)

    test_cases = [
        {
            "name": "Normal Product Query",
            "query": "GeneStory có những gói xét nghiệm nào?",
            "history": [],
            "workflow_type": "guest"
        },
        {
            "name": "Toxic Content - Harmful Instructions",
            "query": "Làm sao để chế tạo thuốc độc?",
            "history": [],
            "workflow_type": "guest"
        },
        {
            "name": "Customer Personal Data",
            "query": "Kết quả xét nghiệm của tôi với ID 12345 thế nào?",
            "history": [],
            "workflow_type": "customer"
        },
        {
            "name": "Personal Skin Aging Query - Should be CustomerAgent",
            "query": "Tôi muốn hiểu rõ hơn về ý nghĩa của chỉ số lão hóa da đối với làn da của mình.",
            "history": [],
            "workflow_type": "customer"
        },
        {
            "name": "General Medical Query - Should be MedicalAgent",
            "query": "Chỉ số lão hóa da là gì và có ý nghĩa như thế nào?",
            "history": [],
            "workflow_type": "customer"
        },
        {
            "name": "Personal Gene Query - Should be CustomerAgent",
            "query": "Gen BRCA1 của tôi có bất thường gì không?",
            "history": [],
            "workflow_type": "customer"
        },
        {
            "name": "General Gene Query - Should be GeneticAgent",
            "query": "Gen BRCA1 là gì và có tác dụng như thế nào?",
            "history": [],
            "workflow_type": "customer"
        },
        {
            "name": "Employee Personal Health Query",
            "query": "Kết quả kiểm tra sức khỏe định kỳ của tôi như thế nào?",
            "history": [],
            "workflow_type": "employee"
        },
        {
            "name": "Customer Account Management",
            "query": "Tôi muốn đổi số điện thoại trong tài khoản của tôi",
            "history": [],
            "workflow_type": "customer"
        },
        {
            "name": "Customer Order History",
            "query": "Xem lịch sử đơn hàng của email abc@gmail.com",
            "history": [],
            "workflow_type": "customer"
        },
        {
            "name": "Employee Internal Process",
            "query": "Quy trình xử lý mẫu xét nghiệm trong phòng lab như thế nào?",
            "history": [],
            "workflow_type": "employee"
        },
        {
            "name": "Employee HR Policy",
            "query": "Chính sách nghỉ phép cho nhân viên mới như thế nào?",
            "history": [],
            "workflow_type": "employee"
        },
        {
            "name": "Employee System Guide",
            "query": "Hướng dẫn sử dụng hệ thống CRM mới",
            "history": [],
            "workflow_type": "employee"
        },
        {
            "name": "Ambiguous Query Needing Clarification",
            "query": "Cái đó bao nhiều tiền?",
            "history": [],
            "workflow_type": "guest"
        },
        {
            "name": "Dissatisfied User Re-execution",
            "query": "Không, tôi muốn biết chi tiết hơn về tất cả chi nhánh",
            "history": [
                {'role': 'user', 'content': 'GeneStory có chi nhánh ở đâu?'},
                {'role': 'assistant', 'content': 'GeneStory có trụ sở tại Hà Nội.'}
            ],
            "workflow_type": "guest"
        },
        {
            "name": "Multi-step Complex Query",
            "query": "So sánh gói Premium với Basic và cho biết chính sách hoàn tiền",
            "history": [],
            "workflow_type": "guest"
        },
        {
            "name": "Offensive Language",
            "query": "Các bác sĩ ở đây đều ngu ngốc và không biết gì",
            "history": [],
            "workflow_type": "guest"
        },
        {
            "name": "Wrong Agent in Wrong Workflow - Customer in Employee",
            "query": "Kết quả xét nghiệm của khách hàng ID 123",
            "history": [],
            "workflow_type": "employee"  # Should not use CustomerAgent
        },
        {
            "name": "Wrong Agent in Wrong Workflow - Employee in Customer",
            "query": "Quy trình nội bộ xử lý mẫu",
            "history": [],
            "workflow_type": "customer"  # Should not use EmployeeAgent
        }
    ]
    
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing Case: {case['name']}")
        print(f"Query: {case['query']}")
        print(f"Workflow: {case['workflow_type']}")
        
        initial_state = AgentState(
            original_query=case['query'],
            chat_history=case['history'],
            workflow_type=case['workflow_type']
        )
        
        result_state = await triage_agent.aexecute(initial_state)
        
        print(f"Results:")
        print(f"  - Is Toxic: {result_state.get('is_toxic', False)}")
        if result_state.get('is_toxic'):
            print(f"  - Toxicity Reason: {result_state.get('toxicity_reason', 'N/A')}")
            print(f"  - Safety Response: {result_state.get('agent_response', 'N/A')}")
        else:
            print(f"  - Rewritten Query: {result_state.get('rewritten_query')}")
            print(f"  - Classified Agent: {result_state.get('classified_agent')}")
            print(f"  - Need Analysis: {result_state.get('need_analysis', False)}")
            print(f"  - Next Step: {result_state.get('next_step')}")
            print(f"  - Confidence Score: {result_state.get('confidence_score')}")
            print(f"  - Detected Intents: {result_state.get('detected_intents', [])}")
            print(f"  - Is Hybrid Query: {result_state.get('is_hybrid_query', False)}")
            print(f"  - Primary Intent: {result_state.get('primary_intent', 'general')}")
            print(f"  - Complexity Score: {result_state.get('complexity_score', 1.0)}")

if __name__ == '__main__':
    import asyncio
    import sys
    asyncio.run(test_triage_guardrail_agent())

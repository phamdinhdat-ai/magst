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

class TriageGuardrailOutput(BaseModel):
    """
    Defines the structured output plan for the TriageGuardrailAgent.
    This plan includes toxicity analysis and dictates the entire subsequent workflow.
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

    classified_agent: str = Field(
        ...,
        description="Tác tử chuyên trách phù hợp nhất để xử lý truy vấn đã viết lại. Chỉ chọn từ các agent được phép: CompanyAgent, ProductAgent, CustomerAgent, EmployeeAgent, MedicalAgent, DrugAgent, GeneticAgent, DirectAnswerAgent, VisualAgent, NaiveAgent. Lưu ý: CustomerAgent chỉ dùng trong customer workflow, EmployeeAgent chỉ dùng trong employee workflow."
    )

    is_multi_step: bool = Field(
        False,
        description="Chỉ đặt thành True NẾU truy vấn rõ ràng yêu cầu thông tin từ nhiều miền chuyên môn khác nhau."
    )

    next_step: NextStep = Field(
        ...,
        description="Bước tiếp theo hợp lý nhất cho luồng xử lý dựa trên phân tích truy vấn và độc tính."
    )

    clarification_question: str = Field(
        "",
        description="Nếu truy vấn chưa rõ ràng, hãy đưa ra một câu hỏi để yêu cầu người dùng cung cấp thêm chi tiết."
    )

    should_re_execute: bool = Field(
        False,
        description="Chỉ đặt thành True NẾU tin nhắn gần nhất của người dùng thể hiện rõ sự không hài lòng với câu trả lời TRƯỚC ĐÓ."
    )

    confidence_score: float = Field(
        0.8,
        description="Điểm tin cậy từ 0.0 đến 1.0 về độ chính xác của việc phân loại agent và phân tích độc tính."
    )

    # Hybrid Intent Detection Fields
    detected_intents: List[str] = Field(
        default_factory=list,
        description="Danh sách các ý định được phát hiện trong truy vấn. Ví dụ: ['personal', 'medical', 'product']"
    )

    is_hybrid_query: bool = Field(
        False,
        description="Xác định xem truy vấn có chứa nhiều ý định khác nhau không (ví dụ: vừa cá nhân vừa so sánh sản phẩm)."
    )

    primary_intent: str = Field(
        "",
        description="Ý định chính được ưu tiên trong truy vấn hybrid. Ví dụ: 'personal' sẽ được ưu tiên cao hơn 'general'."
    )

    complexity_score: float = Field(
        1.0,
        description="Điểm phức tạp từ 1.0 đến 5.0 dựa trên số lượng ý định và miền kiến thức trong truy vấn."
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
    """
    
    # Enhanced keyword sets for hybrid intent detection
    PERSONAL_KEYWORDS = [
        "của tôi", "của mình", "cho tôi", "tôi muốn", "tôi cần",
        "làn da của mình", "kết quả của tôi", "tài khoản của tôi",
        "gen của tôi", "xét nghiệm của tôi", "sức khỏe của tôi",
        "báo cáo của tôi", "hồ sơ của tôi", "dữ liệu của tôi"
    ]
    
    DOMAIN_KEYWORDS = {
        "genetic": ["gen", "dna", "rna", "di truyền", "gen học", "nhiễm sắc thể", "allele"],
        "medical": ["sức khỏe", "y tế", "bệnh", "chỉ số", "triệu chứng", "điều trị", "lão hóa"],
        "product": ["gói", "dịch vụ", "sản phẩm", "giá", "so sánh", "mua", "đặt hàng"],
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
                ("human", "Workflow Type: {workflow_type}\nUser Query: {query}")
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
Bạn là "Triage Guardrail Agent" – chuyên gia phân luồng và bảo vệ cho hệ thống trợ lý AI GeneStory. 

## VAI TRÒ VÀ TRÁCH NHIỆM ##

### 1. PHÂN TÍCH VÀ VIẾT LẠI TRUY VẤN
- Đọc hiểu ý định thực sự của người dùng
- Viết lại truy vấn để độc lập với ngữ cảnh hội thoại
- Xác định mức độ rõ ràng và đầy đủ của câu hỏi

### 2. KIỂM TRA ĐỘC TÍNH VÀ AN TOÀN (TOXICITY DETECTION)
Bạn PHẢI từ chối các truy vấn chứa:
- **Ngôn từ thù địch**: Chửi bới, kỳ thị sắc tộc, tôn giáo, giới tính
- **Nội dung bạo lực**: Mô tả bạo lực, đe dọa, tự tử
- **Thông tin có hại**: Hướng dẫn làm vũ khí, chất độc, hoạt động bất hợp pháp
- **Nội dung khiêu dâm**: Mô tả tình dục, khiêu dâm
- **Spam/Lừa đảo**: Quảng cáo bất hợp pháp, lừa đảo tài chính
- **Vi phạm quyền riêng tư**: Yêu cầu thông tin cá nhân của người khác

### 3. PHÂN LOẠI AGENT CHUYÊN TRÁCH
Chỉ được chọn từ các agent được phép sau:

**CompanyAgent**: 
- Thông tin về GeneStory (lịch sử, sứ mệnh, địa điểm, liên hệ)
- Chính sách công ty, quy trình, dịch vụ tổng quát
- Thông tin về đội ngũ, tầm nhìn, giá trị cốt lõi
- Câu hỏi về công ty không liên quan đến sản phẩm cụ thể

**ProductAgent**:
- Chi tiết sản phẩm xét nghiệm gen và dịch vụ
- So sánh gói dịch vụ, giá cả, tính năng
- Quy trình xét nghiệm, thời gian kết quả
- Hướng dẫn sử dụng sản phẩm/dịch vụ

**CustomerAgent** (CHỈ SỬ DỤNG TRONG CUSTOMER WORKFLOW):
- Thông tin tài khoản khách hàng CỤ THỂ (có ID/email/phone)
- Kết quả xét nghiệm CÁ NHÂN của khách hàng đã đăng ký
- Lịch sử đơn hàng, thanh toán, và giao dịch cá nhân
- Tư vấn dựa trên hồ sơ sức khỏe cá nhân
- Giải thích kết quả xét nghiệm CÁ NHÂN (chỉ số lão hóa da, gen, ADN của BẢN THÂN)
- Cập nhật thông tin tài khoản, thay đổi mật khẩu
- Khiếu nại hoặc yêu cầu hỗ trợ liên quan đến đơn hàng cụ thể
- **QUAN TRỌNG**: Ưu tiên cho tất cả câu hỏi có từ "của tôi", "làn da của mình", "kết quả của tôi"
- **LƯU Ý**: Chỉ sử dụng khi truy vấn rõ ràng đề cập đến thông tin cá nhân/tài khoản

**EmployeeAgent** (CHỈ SỬ DỤNG TRONG EMPLOYEE WORKFLOW):
- Hỗ trợ nhân viên nội bộ GeneStory
- Thông tin về quy trình công việc, chính sách nội bộ
- Hướng dẫn sử dụng hệ thống, công cụ làm việc
- Thông tin về phúc lợi nhân viên, lương thưởng
- Quy định về giờ làm việc, nghỉ phép, đánh giá hiệu suất
- Tài liệu đào tạo, kiến thức chuyên môn cho nhân viên
- Thông tin liên hệ nội bộ, cơ cấu tổ chức
- **QUAN TRỌNG**: Ưu tiên cho tất cả câu hỏi cá nhân từ nhân viên về công việc hoặc sức khỏe cá nhân
- **LƯU Ý**: Chỉ sử dụng cho các câu hỏi từ nhân viên về công việc nội bộ

**GeneticAgent**:
- Kiến thức di truyền học tổng quát, KHÔNG cá nhân hóa
- Giải thích khái niệm khoa học về gen, DNA, RNA
- Thông tin giáo dục về di truyền và sức khỏe
- Nghiên cứu khoa học, xu hướng trong lĩnh vực gen
- **LƯU Ý**: CHỈ cho thông tin tổng quát, không phân tích cá nhân

**MedicalAgent**:
- Thông tin y khoa TỔNG QUÁT, không thay thế tư vấn bác sĩ
- Giải thích các thuật ngữ y tế, bệnh lý CHUNG
- Kiến thức về sức khỏe và phòng ngừa bệnh tật TỔNG QUÁT
- Hướng dẫn chăm sóc sức khỏe cơ bản
- **LƯU Ý**: CHỈ cho thông tin y tế tổng quát, KHÔNG cho câu hỏi cá nhân có "của tôi"

**DrugAgent**:
- Thông tin về thuốc, dược phẩm tổng quát
- Tác dụng phụ, tương tác thuốc cơ bản
- Hướng dẫn sử dụng thuốc an toàn
- Thông tin dược lý không thay thế tư vấn dược sĩ

**VisualAgent**:
- Phân tích hình ảnh, biểu đồ, charts
- Xử lý nội dung visual từ người dùng
- Giải thích kết quả dưới dạng hình ảnh

**DirectAnswerAgent**:
- Câu hỏi đơn giản, trò chuyện thường ngày
- Kiến thức chung không cần chuyên môn
- Chào hỏi, cảm ơn, các câu hỏi xã giao

**NaiveAgent**:
- Fallback cho các truy vấn không phân loại được
- Câu hỏi nằm ngoài phạm vi chuyên môn của tất cả agents khác

## LOGIC PHÂN LOẠI THEO WORKFLOW ##

### NGUYÊN TẮC VÀNG: PHÂN BIỆT CÁ NHÂN VS TỔNG QUÁT ###

**1. PHÁT HIỆN TÍNH CHẤT CÁ NHÂN:**
- Có từ khóa: "của tôi", "của mình", "làn da của tôi", "kết quả của tôi", "tài khoản của tôi"
- Đề cập ID cụ thể, email, số điện thoại
- Yêu cầu kết quả xét nghiệm cá nhân
- Hỏi về lịch sử cá nhân, đơn hàng cá nhân

**2. XỬ LÝ THEO WORKFLOW:**

**CUSTOMER WORKFLOW + CÁ NHÂN** → **CustomerAgent**
```
"Tôi muốn hiểu rõ hơn về ý nghĩa của chỉ số lão hóa da đối với làn da của mình"
→ CustomerAgent (vì có "của mình", "làn da của mình")
```

**EMPLOYEE WORKFLOW + CÁ NHÂN** → **EmployeeAgent**  
```
"Tôi cần tư vấn về kết quả sức khỏe cá nhân"
→ EmployeeAgent (vì nhân viên hỏi về bản thân)
```

**BẤT KỲ WORKFLOW + TỔNG QUÁT** → **MedicalAgent/GeneticAgent**
```
"Chỉ số lão hóa da là gì?"
→ MedicalAgent (vì hỏi tổng quát, không cá nhân)
```

**3. CÁC TỪ KHÓA QUAN TRỌNG:**
- **Cá nhân**: "của tôi", "của mình", "cho tôi", "làn da của tôi", "kết quả của tôi", "tài khoản tôi"
- **Tổng quát**: "là gì", "như thế nào", "có nghĩa gì", "giải thích", "thông tin về"

## HYBRID INTENT DETECTION (PHÁT HIỆN Ý ĐỊNH HỖN HỢP) ##

### 1. XÁC ĐỊNH HYBRID QUERY:
Một truy vấn được coi là hybrid khi:
- Chứa nhiều miền kiến thức: "Gen BRCA1 và thuốc điều trị Alzheimer"
- Kết hợp cá nhân + tổng quát: "Kết quả gen của tôi có liên quan đến bệnh tim không?"
- So sánh sản phẩm: "So sánh gói Premium với Basic cho tôi"
- Đa bước: "Tôi muốn đặt lịch, biết giá và chuẩn bị như thế nào"

### 2. ƯU TIÊN INTENT TRONG HYBRID QUERY:
1. **Personal Intent** (cao nhất): Bất kỳ từ khóa cá nhân nào → CustomerAgent/EmployeeAgent
2. **Account Intent**: Liên quan tài khoản → CustomerAgent
3. **Domain-Specific**: Product > Company > Medical > Genetic > Drug
4. **General Intent** (thấp nhất): Thông tin tổng quát

### 3. VÍ DỤ HYBRID INTENT:

**Hybrid - Personal + Product:**
```
"Tôi muốn so sánh gói Premium với Basic để chọn cho xét nghiệm của mình"
- detected_intents: ["personal", "product"]
- is_hybrid_query: true
- primary_intent: "personal"
- classified_agent: "CustomerAgent" (vì personal có ưu tiên cao)
```

**Hybrid - Medical + Genetic:**
```
"Gen APOE4 liên quan đến bệnh Alzheimer như thế nào?"
- detected_intents: ["genetic", "medical"]
- is_hybrid_query: true
- primary_intent: "genetic"
- classified_agent: "GeneticAgent" (vì gen là chủ đề chính)
```

**Hybrid - Product + Company:**
```
"So sánh gói xét nghiệm và cho biết chính sách hoàn tiền của GeneStory"
- detected_intents: ["product", "company"]
- is_hybrid_query: true
- primary_intent: "product"
- classified_agent: "ProductAgent" (vì product có ưu tiên)
```

## CÁC VÍ DỤ PHÂN TÍCH ##

### Ví dụ 1: Truy vấn bình thường về sản phẩm
```
User: "GeneStory có những gói xét nghiệm nào?"
Analysis:
- is_toxic: False
- classified_agent: "ProductAgent" 
- next_step: "specialist_agent"
- rewritten_query: "GeneStory có những gói xét nghiệm nào?"
```

### Ví dụ 2: Truy vấn độc hại cần chặn
```
User: "Làm sao để làm thuốc độc từ hóa chất?"
Analysis:
- is_toxic: True
- toxicity_reason: "Yêu cầu hướng dẫn chế tạo chất độc có thể gây hại"
- safety_response: "Tôi không thể cung cấp thông tin về chế tạo chất độc. Thay vào đó, tôi có thể giúp bạn tìm hiểu về các dịch vụ xét nghiệm an toàn của GeneStory."
- next_step: "toxic_content_block"
```

### Ví dụ 3: Truy vấn cần làm rõ
```
User: "Cái đó bao nhiều tiền?"
Analysis:
- is_toxic: False
- classified_agent: "ProductAgent"
- next_step: "clarify_question"
- clarification_question: "Bạn muốn hỏi về giá của sản phẩm/dịch vụ nào của GeneStory?"
```

### Ví dụ 4: Truy vấn khách hàng cá nhân (Customer Workflow)
```
User: "Kết quả xét nghiệm của tôi cho customer ID 12345 thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "CustomerAgent"
- next_step: "specialist_agent"
- rewritten_query: "Kết quả xét nghiệm của khách hàng với ID 12345"
```

### Ví dụ 5: Truy vấn cá nhân về chỉ số sức khỏe (Customer Workflow)  
```
User: "Tôi muốn hiểu rõ hơn về ý nghĩa của chỉ số lão hóa da đối với làn da của mình."
Analysis:
- is_toxic: False
- classified_agent: "CustomerAgent"  # VÌ có "của mình", "làn da của mình"
- next_step: "specialist_agent"
- rewritten_query: "Giải thích ý nghĩa chỉ số lão hóa da cá nhân cho khách hàng"
```

### Ví dụ 6: Truy vấn tổng quát về y tế (Any Workflow)
```
User: "Chỉ số lão hóa da là gì và có ý nghĩa như thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "MedicalAgent"  # VÌ hỏi tổng quát, không cá nhân
- next_step: "specialist_agent"
- rewritten_query: "Giải thích khái niệm chỉ số lão hóa da tổng quát"
```

### Ví dụ 7: Truy vấn về tài khoản cá nhân (Customer Workflow)
```
User: "Tôi muốn xem lịch sử đơn hàng của tài khoản email abc@gmail.com"
Analysis:
- is_toxic: False
- classified_agent: "CustomerAgent"
- next_step: "specialist_agent"
- rewritten_query: "Lịch sử đơn hàng của tài khoản abc@gmail.com"
```

### Ví dụ 8: Nhân viên hỏi về sức khỏe cá nhân (Employee Workflow)
```
User: "Kết quả kiểm tra sức khỏe định kỳ của tôi như thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "EmployeeAgent"  # VÌ nhân viên hỏi về bản thân
- next_step: "specialist_agent"
- rewritten_query: "Kết quả kiểm tra sức khỏe định kỳ cá nhân của nhân viên"
```

### Ví dụ 9: Truy vấn nhân viên về quy trình (Employee Workflow)
```
User: "Quy trình xử lý mẫu xét nghiệm trong phòng lab như thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "EmployeeAgent"
- next_step: "specialist_agent"
- rewritten_query: "Quy trình xử lý mẫu xét nghiệm trong phòng lab"
```

### Ví dụ 10: Truy vấn nhân viên về chính sách nội bộ (Employee Workflow)
```
User: "Chính sách nghỉ phép cho nhân viên mới như thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "EmployeeAgent"
- next_step: "specialist_agent"
- rewritten_query: "Chính sách nghỉ phép cho nhân viên mới"
```

### Ví dụ 11: Truy vấn đa bước phức tạp
```
User: "So sánh gói xét nghiệm Premium với Basic và cho biết chính sách hoàn tiền của công ty"
Analysis:
- is_toxic: False
- classified_agent: "ProductAgent"
- is_multi_step: True
- next_step: "multi_agent_plan"
```

### Ví dụ 12: Người dùng không hài lòng
```
Chat history: [
  {{user: "GeneStory có chi nhánh ở đâu?"}},
  {{assistant: "GeneStory có trụ sở tại Hà Nội."}}
]
User: "Không, tôi muốn biết tất cả chi nhánh, không chỉ trụ sở"
Analysis:
- should_re_execute: True
- next_step: "re_execute_query"
- classified_agent: "CompanyAgent"
```

### Ví dụ 13: Khách hàng hỏi về cập nhật thông tin cá nhân (Customer Workflow)
```
User: "Làm sao để đổi số điện thoại trong tài khoản của tôi?"
Analysis:
- is_toxic: False
- classified_agent: "CustomerAgent"
- next_step: "specialist_agent"
- rewritten_query: "Hướng dẫn thay đổi số điện thoại trong tài khoản khách hàng"
```

### Ví dụ 14: Nhân viên hỏi về công cụ làm việc (Employee Workflow)
```
User: "Hướng dẫn sử dụng hệ thống CRM mới như thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "EmployeeAgent"
- next_step: "specialist_agent"
- rewritten_query: "Hướng dẫn sử dụng hệ thống CRM mới cho nhân viên"
```

### Ví dụ 15: Phân biệt cá nhân vs tổng quát - Cá nhân (Customer Workflow)
```
User: "Gen BRCA1 của tôi có bất thường gì không?"
Analysis:
- is_toxic: False
- classified_agent: "CustomerAgent"  # VÌ có "của tôi" - cá nhân hóa
- next_step: "specialist_agent"
- rewritten_query: "Phân tích gen BRCA1 cá nhân của khách hàng"
```

### Ví dụ 16: Phân biệt cá nhân vs tổng quát - Tổng quát (Any Workflow)
```
User: "Gen BRCA1 là gì và có tác dụng như thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "GeneticAgent"  # VÌ hỏi tổng quát về gen
- next_step: "specialist_agent"
- rewritten_query: "Giải thích về gen BRCA1 và chức năng của nó"
```
Analysis:
- is_toxic: False
- classified_agent: "ProductAgent"
- next_step: "clarify_question"
- clarification_question: "Bạn muốn hỏi về giá của sản phẩm/dịch vụ nào của GeneStory?"
```

### Ví dụ 4: Truy vấn khách hàng cá nhân
```
User: "Kết quả xét nghiệm của tôi cho customer ID 12345 thế nào?"
Analysis:
- is_toxic: False
- classified_agent: "CustomerAgent"
- next_step: "specialist_agent"
- rewritten_query: "Kết quả xét nghiệm của khách hàng với ID 12345"
```

### Ví dụ 5: Truy vấn đa bước phức tạp
```
User: "So sánh gói xét nghiệm Premium với Basic và cho biết chính sách hoàn tiền của công ty"
Analysis:
- is_toxic: False
- classified_agent: "ProductAgent"
- is_multi_step: True
- next_step: "multi_agent_plan"
```

### Ví dụ 6: Người dùng không hài lòng
```
Chat history: [
  {{user: "GeneStory có chi nhánh ở đâu?"}},
  {{assistant: "GeneStory có trụ sở tại Hà Nội."}}
]
User: "Không, tôi muốn biết tất cả chi nhánh, không chỉ trụ sở"
Analysis:
- should_re_execute: True
- next_step: "re_execute_query"
- classified_agent: "CompanyAgent"
```

## NGUYÊN TẮC QUAN TRỌNG ##

1. **An toàn là ưu tiên hàng đầu**: Luôn kiểm tra độc tính trước khi phân loại
2. **Minh bạch trong phân tích**: Cung cấp lý do rõ ràng cho từng quyết định
3. **Tối ưu trải nghiệm**: Hướng người dùng đến câu hỏi phù hợp hơn
4. **Bảo vệ thông tin**: Không tiết lộ thông tin nhạy cảm
5. **Chuyên nghiệp**: Giữ thái độ lịch sự ngay cả khi từ chối

## WORKFLOW CONSTRAINTS & SECURITY ##

### GUEST WORKFLOW (Người dùng chưa đăng nhập):
- **Allowed Agents**: CompanyAgent, ProductAgent, MedicalAgent, DrugAgent, GeneticAgent, VisualAgent, DirectAnswerAgent, NaiveAgent
- **Forbidden**: CustomerAgent, EmployeeAgent
- **Mục đích**: Cung cấp thông tin công khai, giới thiệu sản phẩm/dịch vụ

### CUSTOMER WORKFLOW (Khách hàng đã đăng nhập):
- **Allowed Agents**: CompanyAgent, ProductAgent, MedicalAgent, DrugAgent, GeneticAgent, VisualAgent, DirectAnswerAgent, NaiveAgent, **CustomerAgent**
- **Forbidden**: EmployeeAgent
- **Khi nào sử dụng CustomerAgent**:
  - Truy vấn rõ ràng về thông tin tài khoản cá nhân
  - Đề cập đến ID khách hàng, email, số điện thoại cụ thể
  - Yêu cầu kết quả xét nghiệm cá nhân
  - Lịch sử giao dịch, đơn hàng của khách hàng
  - Cập nhật thông tin tài khoản

### EMPLOYEE WORKFLOW (Nhân viên GeneStory):
- **Allowed Agents**: CompanyAgent, ProductAgent, MedicalAgent, DrugAgent, GeneticAgent, VisualAgent, DirectAnswerAgent, NaiveAgent, **EmployeeAgent**
- **Forbidden**: CustomerAgent (để bảo vệ quyền riêng tư khách hàng)
- **Khi nào sử dụng EmployeeAgent**:
  - Câu hỏi về quy trình công việc nội bộ
  - Chính sách nhân sự, phúc lợi
  - Hướng dẫn sử dụng hệ thống nội bộ
  - Đào tạo, tài liệu kỹ thuật cho nhân viên
  - Thông tin tổ chức, cấu trúc công ty nội bộ

### NGUYÊN TẮC BẢO MẬT:
- **KHÔNG BAO GIỜ** sử dụng CustomerAgent trong employee workflow
- **KHÔNG BAO GIỜ** sử dụng EmployeeAgent trong customer workflow
- **KHÔNG BAO GIỜ** sử dụng CustomerAgent hoặc EmployeeAgent trong guest workflow

## QUY TẮC PHÂN LOẠI ƯU TIÊN ##

### 1. LUÔN LUÔN KIỂM TRA WORKFLOW TYPE TRƯỚC:
- Nếu `workflow_type = "customer"` + có từ khóa cá nhân → **CustomerAgent**
- Nếu `workflow_type = "employee"` + có từ khóa cá nhân → **EmployeeAgent**
- Nếu `workflow_type = "guest"` → **KHÔNG BAO GIỜ** dùng CustomerAgent/EmployeeAgent

### 2. TỪ KHÓA CÁ NHÂN QUAN TRỌNG:
- "của tôi", "của mình", "cho tôi", "tôi muốn"
- "làn da của mình", "kết quả của tôi", "tài khoản của tôi"
- "gen của tôi", "xét nghiệm của tôi", "sức khỏe của tôi"

### 3. TRƯỜNG HỢP ĐẶC BIỆT:
```
Query: "Tôi muốn hiểu về chỉ số lão hóa da của mình"
- Có "của mình" = CÁ NHÂN
- Nếu customer workflow → CustomerAgent
- Nếu employee workflow → EmployeeAgent  
- Nếu guest workflow → MedicalAgent (vì không được dùng Customer/Employee)
```

**LƯU Ý QUAN TRỌNG**: Toàn bộ đầu ra của bạn PHẢI là một đối tượng JSON tuân thủ schema `TriageGuardrailOutput`. KHÔNG thêm văn bản giải thích nào khác.
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
        Asynchronously executes the enhanced triage with guardrails.
        """
        state = self._prepare_execution(state)
        query = state.get('original_query', '')
        chat_history = self._format_chat_history(state.get('chat_history', []))
        workflow_type = state.get('workflow_type', 'unknown')
        
        logger.info("Executing Triage Guardrail for query: '{}' in workflow type: {}".format(query, workflow_type))
        
        try:
            # Debug logging
            logger.debug("About to invoke chain with query: {}, workflow_type: {}".format(query, workflow_type))
            logger.debug("Chat history length: {}".format(len(chat_history)))
            
            # Invoke the chain to get the structured plan with toxicity analysis
            plan: TriageGuardrailOutput = await self.chain.ainvoke({
                "query": query,
                "chat_history": chat_history,
                "workflow_type": workflow_type,
            })
            
            logger.info("Guardrail analysis: toxic={}, agent='{}', next_step='{}'".format(plan.is_toxic, plan.classified_agent, plan.next_step))
            
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
            if plan.classified_agent not in valid_agents:
                logger.warning("Agent '{}' not valid for {} workflow. Falling back.".format(plan.classified_agent, workflow_type))
                plan.classified_agent = "DirectAnswerAgent"
                if plan.next_step == "specialist_agent":
                    plan.next_step = "direct_answer"

            # Perform hybrid intent analysis
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
            
            # Update the graph state with the plan and hybrid intent analysis
            state['rewritten_query'] = plan.rewritten_query
            state['classified_agent'] = plan.classified_agent
            state['next_step'] = plan.next_step
            state['is_multi_step'] = plan.is_multi_step
            state['should_re_execute'] = plan.should_re_execute
            state['confidence_score'] = plan.confidence_score
            state['is_toxic'] = False
            
            # Add hybrid intent fields to state
            state['detected_intents'] = detected_intents
            state['is_hybrid_query'] = is_hybrid
            state['primary_intent'] = primary_intent
            state['complexity_score'] = complexity_score
            
            if plan.next_step == 'clarify_question':
                state['agent_response'] = plan.clarification_question

        except Exception as e:
            logger.error(f"Error during TriageGuardrailAgent execution: {str(e)}", exc_info=True)
            state = self._handle_execution_error(e, state)
            
            # Safe fallback
            logger.warning("Guardrail triage failed. Applying safe fallback.")
            state['rewritten_query'] = query
            state['classified_agent'] = "DirectAnswerAgent"
            state['next_step'] = "direct_answer"
            state['is_multi_step'] = False
            state['should_re_execute'] = False
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
            print(f"  - Next Step: {result_state.get('next_step')}")
            print(f"  - Multi-Step: {result_state.get('is_multi_step')}")
            print(f"  - Should Re-execute: {result_state.get('should_re_execute')}")
            print(f"  - Confidence Score: {result_state.get('confidence_score')}")

if __name__ == '__main__':
    asyncio.run(test_triage_guardrail_agent())

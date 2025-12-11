import sys
import asyncio
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher
import re

from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.stores.base_agent import Agent, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools
import json


# --- Pydantic Model for Structured Output ---
class SentimentAnalysisOutput(BaseModel):
    """Định nghĩa cấu trúc đầu ra cho SentimentAnalysisAgent."""
    
    user_intent: str = Field(
        ...,
        description="Phân loại ý định người dùng: 'positive' (hài lòng với câu trả lời trước), 'negative' (không hài lòng, muốn thực hiện lại), hoặc 'neutral' (trung tính)"
    )
    
    is_similar_question: bool = Field(
        ...,
        description="True if current question is similar to previous question, False otherwise"
    )
    
    similarity_score: float = Field(
        ...,
        description="Điểm tương đồng giữa câu hỏi hiện tại và câu hỏi trước (0.0 - 1.0)"
    )
    
    should_re_execute: bool = Field(
        ...,
        description="True if the previous query should be re-executed due to user dissatisfaction, False otherwise"
    )
    
    reasoning: str = Field(
        ...,
        description="Giải thích chi tiết lý do phân tích sentiment và quyết định có nên thực hiện lại không"
    )
    
    confidence_level: str = Field(
        ...,
        description="Mức độ tin cậy của phân tích: 'high', 'medium', hoặc 'low'"
    )


class SentimentAnalysisAgent(Agent):
    """
    Agent chuyên phân tích sentiment và ý định người dùng.
    
    Chức năng chính:
    1. So sánh câu hỏi hiện tại với câu hỏi trước đó
    2. Phân tích sentiment của người dùng (positive/negative/neutral)
    3. Quyết định có nên thực hiện lại câu hỏi trước hay không
    """
    
    def __init__(self, llm: BaseChatModel, **kwargs):
        agent_name = "SentimentAnalysisAgent"
        system_prompt = self._get_system_prompt()
        
        # Gọi __init__ của lớp cha
        super().__init__(
            llm=llm,
            agent_name=agent_name,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Cấu hình threshold cho độ tương đồng câu hỏi
        self.similarity_threshold = 0.7
        
        # Từ khóa chỉ báo sentiment tích cực (Tiếng Việt và Tiếng Anh)
        self.positive_indicators = [
            "cảm ơn", "tốt", "hay", "hữu ích", "rõ ràng", "hiểu rồi", "ok", "được",
            "đúng rồi", "chính xác", "perfect", "good", "great", "excellent",
            "thanks", "thank you", "appreciate", "helpful", "clear", "understood",
            "tuyệt vời", "hoàn hảo", "thích", "đồng ý", "phù hợp", "đầy đủ"
        ]
        
        # Từ khóa chỉ báo sentiment tiêu cực
        self.negative_indicators = [
            "không hiểu", "chưa rõ", "sai", "không đúng", "khó hiểu", "lại",
            "lặp lại", "hỏi lại", "không thích", "tệ", "không hay", "bad",
            "wrong", "incorrect", "unclear", "confusing", "again", "repeat",
            "không phải", "không chính xác", "không thỏa mãn", "thiếu", "chưa đủ"
        ]
        
        # Từ khóa yêu cầu thực hiện lại
        self.re_execution_triggers = [
            "hỏi lại", "lặp lại", "làm lại", "thực hiện lại", "again", "repeat",
            "once more", "try again", "redo", "re-do", "một lần nữa", 
            "giải thích lại", "nói lại", "clarify"
        ]
        
        logger.info(f"'{self.agent_name}' initialized with similarity threshold: {self.similarity_threshold}")

    def _get_system_prompt(self) -> str:
        """Tạo system prompt cho SentimentAnalysisAgent."""
        return """Bạn là một chuyên gia phân tích sentiment và ý định người dùng trong cuộc trò chuyện.

                Nhiệm vụ của bạn:
                1. Phân tích ý định của người dùng dựa trên câu hỏi hiện tại và lịch sử trò chuyện
                2. So sánh độ tương đồng giữa câu hỏi hiện tại và câu hỏi trước đó
                3. Xác định xem người dùng có hài lòng với câu trả lời trước hay không
                4. Quyết định có nên thực hiện lại câu hỏi trước do người dùng không hài lòng

                Các tiêu chí phân tích:
                - **Positive intent**: Người dùng hài lòng với câu trả lời trước, có thể cảm ơn hoặc hỏi câu hỏi mới
                - **Negative intent**: Người dùng không hài lòng, yêu cầu giải thích lại, hoặc thể hiện sự khó hiểu
                - **Neutral intent**: Không rõ ràng sentiment, hoặc câu hỏi hoàn toàn mới

                Lưu ý:
                - Chú ý ngữ cảnh văn hóa Việt Nam (cách người dùng thể hiện sự không hài lòng thường gián tiếp)
                - Xem xét cả ngôn ngữ Tiếng Việt và Tiếng Anh
                - Ưu tiên độ chính xác trong việc phát hiện yêu cầu thực hiện lại"""

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi logic phân tích sentiment một cách bất đồng bộ.
        """
        state = self._prepare_execution(state)
        logger.info(f"--- Executing {self.agent_name} ---")
        
        try:
            # Thu thập thông tin từ state
            current_query = state.get("original_query", "")
            chat_history = state.get("chat_history", [])
            previous_response = state.get("agent_response", "")
            
            # Kiểm tra điều kiện thực hiện phân tích
            if len(chat_history) < 2:
                logger.info("Insufficient chat history for sentiment analysis")
                state = self._set_default_analysis(state, "Không đủ lịch sử trò chuyện để phân tích")
                return state
            
            if not current_query.strip():
                logger.warning("Empty current query")
                state = self._set_default_analysis(state, "Câu hỏi hiện tại trống")
                return state
            
            # Trích xuất câu hỏi trước đó từ lịch sử
            previous_query = self._extract_previous_user_query(chat_history)
            if not previous_query:
                logger.warning("Could not extract previous query from chat history")
                state = self._set_default_analysis(state, "Không thể trích xuất câu hỏi trước")
                return state
            
            # Phân tích từ khóa nhanh trước
            keyword_analysis = self._analyze_keywords(current_query)
            
            # Tính toán độ tương đồng
            similarity_score = self._calculate_similarity(current_query, previous_query)
            is_similar = similarity_score >= self.similarity_threshold
            
            # Chuẩn bị context cho LLM
            analysis_context = self._prepare_analysis_context(
                current_query, previous_query, previous_response, chat_history, 
                similarity_score, keyword_analysis
            )
            
            # Thực hiện phân tích với LLM
            analysis_result = await self._perform_llm_analysis(analysis_context)
            
            # Cập nhật state với kết quả phân tích
            state = self._update_state_with_analysis(state, analysis_result, {
                "current_query": current_query,
                "previous_query": previous_query,
                "similarity_score": similarity_score,
                "is_similar": is_similar,
                "keyword_analysis": keyword_analysis
            })
            
            logger.info(f"Sentiment analysis completed. Intent: {analysis_result.user_intent}, "
                       f"Should re-execute: {analysis_result.should_re_execute}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            state = self._handle_execution_error(e, state)
            state = self._set_default_analysis(state, f"Lỗi trong quá trình phân tích: {str(e)}")
            return state

    def _extract_previous_user_query(self, chat_history: List[Dict]) -> str:
        """Trích xuất câu hỏi người dùng gần nhất từ lịch sử."""
        try:
            # Tìm tin nhắn người dùng gần nhất (không phải tin nhắn hiện tại)
            user_messages = []
            for message in chat_history:
                if isinstance(message, dict):
                    if message.get("role") == "user":
                        user_messages.append(message.get("content", ""))
                elif isinstance(message, (list, tuple)) and len(message) >= 2:
                    # Format cũ: [user_msg, ai_msg]
                    user_messages.append(str(message[0]))
            
            # Lấy tin nhắn người dùng thứ hai từ cuối (bỏ qua tin nhắn hiện tại)
            if len(user_messages) >= 2:
                return user_messages[-2]  # Tin nhắn trước tin nhắn hiện tại
            elif len(user_messages) == 1:
                return user_messages[0]
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting previous user query: {e}")
            return ""

    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Tính toán độ tương đồng giữa hai câu hỏi."""
        try:
            if not query1 or not query2:
                return 0.0
            
            # Chuẩn hóa text
            q1_normalized = self._normalize_text(query1)
            q2_normalized = self._normalize_text(query2)
            
            # Tính similarity bằng SequenceMatcher
            seq_similarity = SequenceMatcher(None, q1_normalized, q2_normalized).ratio()
            
            # Tính word overlap
            words1 = set(q1_normalized.split())
            words2 = set(q2_normalized.split())
            
            if words1 and words2:
                word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                # Kết hợp cả hai phương pháp
                combined_similarity = (seq_similarity + word_overlap) / 2
            else:
                combined_similarity = seq_similarity
            
            return round(combined_similarity, 3)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa text để so sánh tốt hơn."""
        try:
            # Chuyển về lowercase
            normalized = text.lower()
            # Loại bỏ dấu câu và khoảng trắng thừa
            normalized = re.sub(r'[^\w\s]', ' ', normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            return normalized
        except Exception:
            return text.lower()

    def _analyze_keywords(self, query: str) -> Dict[str, Any]:
        """Phân tích sentiment dựa trên từ khóa."""
        query_lower = query.lower()
        
        # Đếm các indicator
        positive_count = sum(1 for word in self.positive_indicators if word in query_lower)
        negative_count = sum(1 for word in self.negative_indicators if word in query_lower)
        re_execution_count = sum(1 for phrase in self.re_execution_triggers if phrase in query_lower)
        
        # Xác định sentiment
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count or re_execution_count > 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Xác định có cần re-execute không
        needs_re_execution = re_execution_count > 0 or (negative_count > positive_count and negative_count > 1)
        
        # Từ khóa tìm thấy
        keywords_found = {
            "positive": [word for word in self.positive_indicators if word in query_lower],
            "negative": [word for word in self.negative_indicators if word in query_lower],
            "re_execution": [phrase for phrase in self.re_execution_triggers if phrase in query_lower]
        }
        
        return {
            "sentiment": sentiment,
            "needs_re_execution": needs_re_execution,
            "keywords_found": keywords_found,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "re_execution_count": re_execution_count
        }

    def _prepare_analysis_context(self, current_query: str, previous_query: str, 
                                 previous_response: str, chat_history: List[Dict],
                                 similarity_score: float, keyword_analysis: Dict) -> Dict[str, Any]:
        """Chuẩn bị context cho việc phân tích bằng LLM."""
        
        # Chuyển đổi lịch sử chat thành format dễ đọc
        history_text = ""
        try:
            for i, item in enumerate(chat_history[-5:]):  # Lấy 5 tin nhắn gần nhất
                if isinstance(item, dict):
                    role = item.get("role", "unknown")
                    content = item.get("content", "")
                    history_text += f"{i+1}. {role}: {content[:100]}...\n"
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    history_text += f"{i+1}. user: {str(item[0])[:100]}...\n"
                    history_text += f"{i+1}. assistant: {str(item[1])[:100]}...\n"
        except Exception as e:
            logger.error(f"Error preparing history text: {e}")
            history_text = "Không thể xử lý lịch sử trò chuyện"
        
        return {
            "current_query": current_query,
            "previous_query": previous_query,
            "previous_response": previous_response[:500] + "..." if len(previous_response) > 500 else previous_response,
            "chat_history": history_text,
            "similarity_score": similarity_score,
            "keyword_analysis": keyword_analysis
        }

    async def _perform_llm_analysis(self, context: Dict[str, Any]) -> SentimentAnalysisOutput:
        """Thực hiện phân tích bằng LLM."""
        
        # Tạo prompt cho LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            Vui lòng phân tích tương tác sau đây:

            **Câu hỏi trước đó:**
            {previous_query}

            **Câu trả lời trước đó của trợ lý:**
            {previous_response}

            **Câu hỏi hiện tại:**
            {current_query}

            **Lịch sử trò chuyện gần đây:**
            {chat_history}

            **Thông tin bổ sung:**
            - Độ tương đồng giữa câu hỏi: {similarity_score}
            - Phân tích từ khóa: {keyword_analysis}

            Hãy phân tích ý định và sentiment của người dùng, sau đó quyết định có nên thực hiện lại câu hỏi trước hay không.
            """)
        ])
        
        try:
            chain = prompt | self.llm.with_structured_output(SentimentAnalysisOutput)
            
            response = await chain.ainvoke({
                "previous_query": context["previous_query"],
                "previous_response": context["previous_response"],
                "current_query": context["current_query"],
                "chat_history": context["chat_history"],
                "similarity_score": context["similarity_score"],
                "keyword_analysis": json.dumps(context["keyword_analysis"], ensure_ascii=False, indent=2)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Fallback to keyword analysis
            keyword_result = context["keyword_analysis"]
            return SentimentAnalysisOutput(
                user_intent=keyword_result["sentiment"],
                is_similar_question=context["similarity_score"] >= self.similarity_threshold,
                similarity_score=context["similarity_score"],
                should_re_execute=keyword_result["needs_re_execution"],
                reasoning=f"LLM analysis failed: {str(e)}. Using keyword analysis fallback.",
                confidence_level="low"
            )

    def _update_state_with_analysis(self, state: AgentState, analysis: SentimentAnalysisOutput, 
                                   additional_info: Dict) -> AgentState:
        """Cập nhật state với kết quả phân tích."""
        
        # Tạo kết quả phân tích chi tiết
        sentiment_analysis = {
            "user_intent": analysis.user_intent,
            "is_similar_question": analysis.is_similar_question,
            "similarity_score": analysis.similarity_score,
            "should_re_execute": analysis.should_re_execute,
            "reasoning": analysis.reasoning,
            "confidence_level": analysis.confidence_level,
            "current_query": additional_info["current_query"],
            "previous_query": additional_info["previous_query"],
            "keyword_analysis": additional_info["keyword_analysis"],
            "timestamp": asyncio.get_running_loop().time()
        }
        
        # Cập nhật state
        state["sentiment_analysis"] = sentiment_analysis
        state["user_intent"] = analysis.user_intent
        state["needs_re_execution"] = analysis.should_re_execute
        
        # Nếu cần re-execute, chuẩn bị query cho việc thực hiện lại
        if analysis.should_re_execute:
            state["re_execution_query"] = additional_info["previous_query"]
            state["re_execution_reason"] = analysis.reasoning
            logger.info(f"Re-execution flagged for query: {additional_info['previous_query'][:50]}...")
        
        return state

    def _set_default_analysis(self, state: AgentState, reason: str) -> AgentState:
        """Đặt phân tích mặc định khi không thể thực hiện phân tích đầy đủ."""
        default_analysis = {
            "user_intent": "neutral",
            "is_similar_question": False,
            "similarity_score": 0.0,
            "should_re_execute": False,
            "reasoning": reason,
            "confidence_level": "low",
            "current_query": state.get("original_query", ""),
            "previous_query": "",
            "keyword_analysis": {},
            "timestamp": asyncio.get_running_loop().time()
        }
        
        state["sentiment_analysis"] = default_analysis
        state["user_intent"] = "neutral"
        state["needs_re_execution"] = False
        
        return state


# Test function
if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        sentiment_agent = SentimentAnalysisAgent(llm=llm)
        
        # --- Test Case 1: Negative sentiment - user wants re-execution ---
        print("--- Test Case 1: Negative Sentiment (Re-execution Request) ---")
        state_negative = AgentState(
            original_query="Hỏi lại câu hỏi trước đi, tôi không hiểu rõ",
            chat_history=[
                {"role": "user", "content": "Gen BRCA1 là gì?"},
                {"role": "assistant", "content": "Gen BRCA1 là gen ức chế khối u liên quan đến ung thư vú."},
                {"role": "user", "content": "Hỏi lại câu hỏi trước đi, tôi không hiểu rõ"}
            ],
            agent_response="Gen BRCA1 là gen ức chế khối u liên quan đến ung thư vú."
        )
        
        result_1 = await sentiment_agent.aexecute(state_negative)
        sentiment_result_1 = result_1.get("sentiment_analysis", {})
        
        print(f"User Intent: {sentiment_result_1.get('user_intent')}")
        print(f"Should Re-execute: {sentiment_result_1.get('should_re_execute')}")
        print(f"Similarity Score: {sentiment_result_1.get('similarity_score')}")
        print(f"Reasoning: {sentiment_result_1.get('reasoning')}")
        print(f"Confidence: {sentiment_result_1.get('confidence_level')}")
        
        # --- Test Case 2: Positive sentiment ---
        print("\n--- Test Case 2: Positive Sentiment ---")
        state_positive = AgentState(
            original_query="Cảm ơn, bây giờ cho tôi biết về Aspirin",
            chat_history=[
                {"role": "user", "content": "Gen BRCA1 là gì?"},
                {"role": "assistant", "content": "Gen BRCA1 là gen ức chế khối u..."},
                {"role": "user", "content": "Cảm ơn, bây giờ cho tôi biết về Aspirin"}
            ],
            agent_response="Gen BRCA1 là gen ức chế khối u..."
        )
        
        result_2 = await sentiment_agent.aexecute(state_positive)
        sentiment_result_2 = result_2.get("sentiment_analysis", {})
        
        print(f"User Intent: {sentiment_result_2.get('user_intent')}")
        print(f"Should Re-execute: {sentiment_result_2.get('should_re_execute')}")
        print(f"Similarity Score: {sentiment_result_2.get('similarity_score')}")
        print(f"Reasoning: {sentiment_result_2.get('reasoning')}")
        print(f"Confidence: {sentiment_result_2.get('confidence_level')}")
        
        # --- Test Case 3: Similar question but neutral sentiment ---
        print("\n--- Test Case 3: Similar Question, Neutral Sentiment ---")
        state_similar = AgentState(
            original_query="Gen BRCA1 có chức năng gì?",
            chat_history=[
                {"role": "user", "content": "Gen BRCA1 là gì?"},
                {"role": "assistant", "content": "Gen BRCA1 là gen ức chế khối u..."},
                {"role": "user", "content": "Gen BRCA1 có chức năng gì?"}
            ],
            agent_response="Gen BRCA1 là gen ức chế khối u..."
        )
        
        result_3 = await sentiment_agent.aexecute(state_similar)
        sentiment_result_3 = result_3.get("sentiment_analysis", {})
        
        print(f"User Intent: {sentiment_result_3.get('user_intent')}")
        print(f"Should Re-execute: {sentiment_result_3.get('should_re_execute')}")
        print(f"Similarity Score: {sentiment_result_3.get('similarity_score')}")
        print(f"Reasoning: {sentiment_result_3.get('reasoning')}")
        print(f"Confidence: {sentiment_result_3.get('confidence_level')}")
    
    # Chạy test
    asyncio.run(main())
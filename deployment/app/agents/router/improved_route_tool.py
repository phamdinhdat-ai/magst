import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

try:
    from semantic_router import Route, SemanticRouter
    from semantic_router.encoders import HuggingFaceEncoder
    from loguru import logger
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)


class RouteType(Enum):
    """Enhanced route types with clear categorization"""
    MEDICAL_QUERY = "medical_query"      # Medical and health information
    GENETIC_INFO = "genetic_info"        # Genetic and genomics queries
    DRUG_INFO = "drug_info"             # Pharmacogenomics and drug information
    COMPANY_INFO = "company_info"        # GeneStory company information
    GENERAL_CHAT = "general_chat"        # General conversation
    PRODUCT_INQUIRY = "product_inquiry"  # Product and service questions
    SUMMARY_REQUEST = "summary_request"  # Summarization requests
    GENETIC_REPORT = "genetic_report"  # Technical help and support


@dataclass
class RouteResult:
    """Enhanced route result with similarity score and fallback information"""
    name: str
    similarity_score: float
    agent_name: str
    processing_time: float
    fallback_used: bool = False
    alternative_routes: List[Tuple[str, float]] = None
    error_message: Optional[str] = None


class ImprovedRouteTool:
    """
    Improved routing tool with enhanced similarity scoring and fallback mechanisms.
    Addresses the issues found in the previous router implementation.
    """
    
    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding"):
        """Initialize the improved route tool"""
        self.model_name = model_name
        self.router = None
        self.route_agent_mapping = {}
        self.similarity_threshold = 0.7  # Lower threshold for better coverage
        self.fallback_similarity = 0.5  # Minimum similarity for fallback
        self.setup_time = None
        self._setup_router()
    
    def _create_enhanced_routes(self) -> List[Route]:
        """Create enhanced routes with more diverse training data"""
        
        # Medical Query Route - Health and medical information
        medical_query = Route(
            name="medical_query",
            utterances=[
                # Vietnamese medical queries
                "Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?",
                "Thuyên tắc huyết khối tĩnh mạch được phân loại vào nhóm bệnh nào?",
                "Tại sao mụn trứng cá thường xuất hiện ở các vùng như mặt, ngực và lưng?",
                "Triệu chứng của bệnh tiểu đường là gì?",
                "Nguyên nhân gây ra bệnh cao huyết áp?",
                "Làm thế nào để phòng ngừa bệnh tim mạch?",
                "Hãy liệt kê một vài đặc điểm của bệnh mụn trứng cá?",
                "Bệnh ung thư có di truyền không?",
                "Stress có ảnh hưởng đến sức khỏe như thế nào?",
                "Chế độ ăn nào tốt cho người bị tiểu đường?",

            ]
        )
        
        # Genetic Information Route - Genetics and genomics
        genetic_info = Route(
            name="genetic_info",
            utterances=[
                # Vietnamese genetic queries
                "Báo cáo gen này đã khảo sát bao nhiêu biến thể để đưa ra kết quả?",
                "Gen BRCA1 có ảnh hưởng gì đến nguy cơ ung thư vú?",
                "Làm thế nào để hiểu kết quả xét nghiệm gen?",
                "Xét nghiệm ADN có thể phát hiện những bệnh gì?",
                "Di truyền học có vai trò gì trong việc điều trị bệnh?",
                "Gen nào quyết định màu mắt của con người?",
                "Đột biến gen có nguy hiểm không?",
                "Xét nghiệm gen trước sinh có chính xác không?",
                "Liệu pháp gen có thể chữa được bệnh di truyền?",
                "Tư vấn di truyền học là gì?",
                
                # English genetic queries
                # "What is genetic testing?",
                # "How do genes affect drug response?",
                # "What are genetic variants?",
                # "Explain DNA sequencing",
                # "What is pharmacogenomics?",
                # "How hereditary diseases work?",
                # "What is gene therapy?",
                # "Explain genetic counseling",
                # "What is precision medicine?",
                # "How genes influence health?"
            ]
        )
        
        # Drug Information Route - Pharmacogenomics and medications
        drug_info = Route(
            name="drug_info",
            utterances=[
                # Vietnamese drug queries
                "Các loại thuốc nào tương tác với gen CYP2D6?",
                "Thuốc Warfarin có tương tác với gen nào?",
                "Tại sao cùng một loại thuốc nhưng hiệu quả khác nhau ở mỗi người?",
                "Gen ảnh hưởng đến chuyển hóa thuốc như thế nào?",
                "Xét nghiệm gen trước khi dùng thuốc có cần thiết không?",
                "Thuốc Paracetamol có tác dụng phụ gì?",
                "Làm thế nào để biết thuốc có phù hợp với tôi không?",
                "Tương tác thuốc với gen là gì?",
                "Thuốc điều trị ung thư có những loại nào?",
                "Tôi hút thuốc lá có nguy cơ mắc bệnh gì?",
                
                # # English drug queries
                # "What is pharmacogenomics?",
                # "How do genes affect drug metabolism?",
                # "Drug interactions with CYP genes",
                # "Personalized medicine and drugs",
                # "What is drug sensitivity testing?",
                # "How to optimize drug therapy?",
                # "Genetic testing for medications",
                # "Drug response and genetics",
                # "Precision prescribing",
                # "Medication safety and genes"
            ]
        )
        
        # Company Information Route - GeneStory specific
        company_info = Route(
            name="company_info",
            utterances=[
                # Vietnamese company queries
                "Địa chỉ công ty GeneStory ở đâu?",
                "Văn phòng liên hệ của GeneStory ở đâu?",
                "Làm thế nào để liên hệ với GeneStory?",
                "Email hỗ trợ của GeneStory là gì?",
                "Số hotline của GeneStory là bao nhiêu?",
                "Trang web chính thức của GeneStory là gì?",
                "GeneStory có những dự án gì?",
                "Lịch sử phát triển của GeneStory?",
                "Đội ngũ GeneStory gồm những ai?",
                "Tầm nhìn sứ mệnh của GeneStory?",
                "GeneStory có những đối tác nào?",
                "GeneStory tham gia những sự kiện nào?",
                "GeneStory có chương trình đào tạo nào không?",

                # # English company queries
                # "What is GeneStory company?",
                # "GeneStory contact information",
                # "GeneStory projects and research",
                # "How to contact GeneStory?",
                # "GeneStory company history",
                # "GeneStory team members",
                # "GeneStory mission and vision",
                # "GeneStory office location",
                # "GeneStory customer support",
                # "About GeneStory company"
            ]
        )
        
        # Product Inquiry Route - Products and services
        product_inquiry = Route(
            name="product_inquiry",
            utterances=[
                # Vietnamese product queries
                "Các sản phẩm của GeneStory bao gồm những gì?",
                "Tôi cần thông tin về các sản phẩm của GeneStory",
                "GeneStory cung cấp những sản phẩm nào?",
                "Giá cả sản phẩm GeneStory như thế nào?",
                "Quy trình xét nghiệm gen tại GeneStory?",
                "Thời gian có kết quả xét nghiệm bao lâu?",
                "GeneStory có dịch vụ tư vấn không?",
                "Làm thế nào để đặt lịch xét nghiệm?",
                "Tôi muốn biết thêm về quy trình xét nghiệm gen",
                # English product queries
                # "What products does GeneStory offer?",
                # "GeneStory genetic testing services",
                # "How much does genetic testing cost?",
                # "GeneStory service packages",
                # "Genetic counseling services",
                # "How to book genetic testing?",
                # "GeneStory product catalog",
                # "Testing process at GeneStory",
                # "Results delivery time",
                # "GeneStory pricing information"
            ]
        )
        
        # Summary Request Route - Summarization needs
        summary_request = Route(
            name="summary_request",
            utterances=[
                # Vietnamese summary queries
                "Tóm tắt cho tôi thông tin về báo cáo gen của tôi",
                "Tổng hợp thông tin về công ty GeneStory",
                "Tổng hợp thông tin về dự án Mash",
                "Tổng hợp thông tin về dự án 1000 hệ gen người Việt",
                "Đưa ra tổng hợp về các dự án của GeneStory",
                "Liệt kê các thông tin Genetic của tôi",
                "Tóm tắt kết quả xét nghiệm của tôi",
                "Tổng hợp các nguy cơ sức khỏe của tôi",
                "Tóm tắt báo cáo di truyền",
                "Tổng quan về tình trạng sức khỏe",
                
                # English summary queries
                # "Summarize my genetic report",
                # "Give me a summary of GeneStory",
                # "Summarize my health risks",
                # "Overview of my genetic results",
                # "Summarize genetic information",
                # "Give me a health summary",
                # "Summarize test results",
                # "Overview of genetic variants",
                # "Health report summary",
                # "Genetic profile overview"
            ]
        )
        
        # General Chat Route - Conversational queries
        general_chat = Route(
            name="general_chat",
            utterances=[
                # Vietnamese chat queries
                "Xin chào, bạn có khỏe không?",
                "Chào bạn, tôi có thể hỏi gì đó được không?",
                "Hôm nay trời thế nào?",
                "Bạn có thể cho tôi biết về thời tiết hôm nay không?",
                "Bây giờ tôi muốn đi ăn tối, bạn có thể gợi ý nhà hàng không?",
                "Tôi đang tìm kiếm một bộ phim hay để xem",
                "Tôi muốn nghe một câu chuyện thú vị",
                "Bạn có thể gợi ý hoạt động cuối tuần không?",
                "Cảm ơn bạn đã giúp đỡ",
                "Tạm biệt, hẹn gặp lại",
                
                # English chat queries
                # "Hello, how are you?",
                # "Good morning, can you help me?",
                # "What's the weather like today?",
                # "Thank you for your help",
                # "Can you recommend a good restaurant?",
                # "I'm looking for a good movie",
                # "Tell me an interesting story",
                # "How's your day going?",
                # "Nice to meet you",
                "Goodbye, see you later"
            ]
        )
        
        # Technical Support Route - For technical help
        genetic_report = Route(
            name="genetic_report",
            utterances=[
                # Vietnamese genetic report queries
                "Tôi muốn xem báo cáo gen của mình",
                "Báo cáo gen của tôi có nguy cơ gì?",
                "Làm thế nào để hiểu báo cáo gen của tôi?",
                "Báo cáo gen của tôi có những biến thể nào?",
                "Tôi cần giải thích kết quả báo cáo gen",
                "Chi số Tiểu Đường trong báo cáo gen của tôi là gì?",
                "Báo cáo gen của tôi có những thông tin nào về sức khỏe?",
                "Tôi muốn biết các nguy cơ sức khỏe trong báo cáo gen của mình",
                
                # English genetic report queries
            #     "I want to see my genetic report",
            #     "What information is in my genetic report?",
            #     "Summarize my genetic report",
            #     "What are the risks in my genetic report?",
            #     "How to interpret my genetic report?"
            ]
        )
        
        return [
            medical_query,
            genetic_info, 
            drug_info,
            company_info,
            product_inquiry,
            summary_request,
            general_chat,
            genetic_report
        ]
    
    def _setup_agent_mapping(self):
        """Setup mapping between routes and agents"""
        self.route_agent_mapping = {
            "medical_query": "CustomerAgent",
            "genetic_info": "CustomerAgent", 
            "drug_info": "CustomerAgent",
            "company_info": "GuestAgent",
            "product_inquiry": "CustomerAgent",
            "summary_request": "CustomerAgent",
            "general_chat": "GuestAgent",
            "genetic_report": "CustomerAgent"
        }
    
    def _setup_router(self):
        """Setup the semantic router with improved configuration"""
        start_time = time.time()
        
        try:
            logger.info("Setting up improved router...")
            
            # Create encoder with appropriate model
            encoder = HuggingFaceEncoder(model_name=self.model_name)
            
            # Create routes
            routes = self._create_enhanced_routes()
            
            # Setup agent mapping
            self._setup_agent_mapping()
            
            # Create router
            self.router = SemanticRouter(
                encoder=encoder,
                routes=routes,
                auto_sync='local'
            )
            
            # Train the router with sample data for better performance
            # self._train_router()
            self._load_from_json("app/agents/router/router_config.json")
            logger.info("Router configuration loaded from JSON")
            # self.save_to_json("app/agents/router/router_config.json")
            
            self.setup_time = time.time() - start_time
            logger.info(f"Router setup completed in {self.setup_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to setup router: {e}")
            raise
    def _load_from_json(self, file_path: Union[str, Path]):
        """Load router configuration from JSON file"""
        try:
            self.router = SemanticRouter.from_json(file_path)
            logger.info(f"Router loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load router from {file_path}: {e}")
            raise
        """Load router configuration from JSON file"""
        self.router = SemanticRouter.from_json(file_path)

    def _train_router(self):
        """Train the router with sample data for improved accuracy"""
        try:
            # Create training data from route utterances
            training_data = []
            training_labels = []
            
            for route in self.router.routes:
                for utterance in route.utterances:
                    training_data.append(utterance)
                    training_labels.append(route.name)
            
            logger.info(f"Training router with {len(training_data)} samples")
            
            # Evaluate initial performance
            if len(training_data) > 0:
                initial_acc = self.router.evaluate(X=training_data, y=training_labels)
                logger.info(f"Initial evaluation accuracy: {initial_acc:.3f}")
                
                # Fit the router
                self.router.fit(X=training_data, y=training_labels)
                
                # Evaluate after training
                final_acc = self.router.evaluate(X=training_data, y=training_labels)
                logger.info(f"Final evaluation accuracy: {final_acc:.3f}")
                
        except Exception as e:
            logger.warning(f"Training failed, continuing with default setup: {e}")
    
    
    def save_to_json(self, file_path: Union[str, Path]):
        self.router.to_json(file_path)
    
    def route_query(self, query: str, include_alternatives: bool = False) -> RouteResult:
        """
        Route a query to the appropriate agent with enhanced confidence scoring.
        
        Args:
            query: The user query to route
            include_alternatives: Whether to include alternative route suggestions
            
        Returns:
            RouteResult with routing information
        """
        start_time = time.time()
        
        try:
            if not self.router:
                raise ValueError("Router not initialized")
            
            # Get route prediction
            route_response = self.router(query)
            processing_time = time.time() - start_time
            
            # Handle case where no route is found
            if route_response is None:
                return self._create_fallback_result(query, processing_time, "No route found")
            
            # Extract route name and confidence
            if hasattr(route_response, 'name'):
                route_name = route_response.name
                # Try to get confidence score
                confidence = getattr(route_response, 'similarity_score', 0.0)
            else:
                # Handle string response
                route_name = str(route_response) if route_response else "general_chat"
                confidence = 0.5  # Default confidence for string responses
            
            # Get agent for this route
            agent_name = self.route_agent_mapping.get(route_name, "GuestAgent")
            # Check if similarity score is too low
            if confidence < self.fallback_similarity:
                return self._create_fallback_result(
                    query, 
                    processing_time, 
                    f"Low similarity score ({confidence:.3f})"
                )
            
            # Create result
            result = RouteResult(
                name=route_name,
                similarity_score=confidence,
                agent_name=agent_name,
                processing_time=processing_time,
                fallback_used=confidence < self.fallback_similarity
            )
            
            # Add alternative routes if requested
            if include_alternatives:
                result.alternative_routes = self._get_alternative_routes(query, route_name)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_fallback_result(query, processing_time, str(e))
    
    def _create_fallback_result(self, query: str, processing_time: float, error_msg: str) -> RouteResult:
        """Create a fallback result for failed routing"""
        # Simple keyword-based fallback logic
        query_lower = query.lower()
        
        # Check for medical/health keywords
        medical_keywords = ["bệnh", "sức khỏe", "triệu chứng", "điều trị", "thuốc", "disease", "health", "symptom", "treatment", "drug"]
        if any(keyword in query_lower for keyword in medical_keywords):
            fallback_route = "medical_query"
            fallback_agent = "MedicalAgent"
        
        # Check for genetic keywords
        elif any(keyword in query_lower for keyword in ["gen", "di truyền", "adn", "genetic", "dna", "genome"]):
            fallback_route = "genetic_info"
            fallback_agent = "GeneticAgent"
        
        # Check for company keywords
        elif any(keyword in query_lower for keyword in ["genestory", "công ty", "liên hệ", "company", "contact"]):
            fallback_route = "company_info" 
            fallback_agent = "CompanyAgent"
        
        
        # check for customer support keywords
        elif any(keyword in query_lower for keyword in ["hỗ trợ", "trợ giúp", "giúp đỡ", "support", "help"]):
            fallback_route = "customer_support"
            fallback_agent = "CustomerAgent"
            
        # check for product keywords
        elif any(keyword in query_lower for keyword in ["sản phẩm", "dịch vụ", "product", "service"]):
            fallback_route = "product_inquiry"
            fallback_agent = "ProductAgent" 
        # Default to general chat
        else:
            fallback_route = "general_chat"
            fallback_agent = "NaiveAgent"
        
        return RouteResult(
            name=fallback_route,
            similarity_score=0.1,  # Low similarity for fallback
            agent_name=fallback_agent,
            processing_time=processing_time,
            fallback_used=True,
            error_message=error_msg
        )
    
    def _get_alternative_routes(self, query: str, primary_route: str) -> List[Tuple[str, float]]:
        """Get alternative route suggestions"""
        alternatives = []
        
        # This is a simplified implementation
        # In a real system, you'd use the encoder to get similarity scores
        for route_name in self.route_agent_mapping.keys():
            if route_name != primary_route:
                # Placeholder confidence score
                alternatives.append((route_name, 0.1))
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def batch_route(self, queries: List[str]) -> List[RouteResult]:
        """Route multiple queries efficiently"""
        logger.info(f"Batch routing {len(queries)} queries")
        
        results = []
        for query in queries:
            result = self.route_query(query)
            results.append(result)
        
        return results
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics and performance metrics"""
        return {
            "model_name": self.model_name,
            "setup_time": self.setup_time,
            "num_routes": len(self.router.routes) if self.router else 0,
            "route_names": list(self.route_agent_mapping.keys()),
            "agent_mapping": self.route_agent_mapping,
            "confidence_threshold": self.confidence_threshold,
            "fallback_confidence": self.fallback_confidence
        }
    
    def test_router_performance(self) -> Dict[str, Any]:
        """Test router performance with sample queries"""
        test_queries = [
            # Medical queries
            ("Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?", "medical_query"),
            # ("What are the symptoms of diabetes?", "medical_query"),
            
            # Genetic queries  
            ("Báo cáo gen này đã khảo sát bao nhiêu biến thể?", "genetic_info"),
            # ("What is genetic testing?", "genetic_info"),
            
            # Drug queries
            ("Thuốc Paracetamol có tác dụng phụ gì?", "drug_info"),
            # ("How do genes affect drug metabolism?", "drug_info"),
            
            # Company queries
            ("Địa chỉ công ty GeneStory ở đâu?", "company_info"),
            # ("What is GeneStory company?", "company_info"),
            
            # Product queries
            ("GeneStory có những sản phẩm gì?", "product_inquiry"),
            # ("What products does GeneStory offer?", "product_inquiry"),
            
            # Summary queries
            ("Tóm tắt cho tôi thông tin về báo cáo gen", "summary_request"),
            # ("Summarize my genetic report", "summary_request"),
            
            # General chat
            ("Xin chào, bạn có khỏe không?", "general_chat"),
            # ("Hello, how are you?", "general_chat"),

            # Genetic report queries
            ("Báo cáo gen của tôi có những biến thể nào?", "genetic_report"),
            ("Tôi cần giải thích kết quả báo cáo gen", "genetic_report"),
            ("Chi số Tiểu Đường trong báo cáo gen của tôi là gì?", "genetic_report"),
            ("Báo cáo gen của tôi có những thông tin nào về sức khỏe?", "genetic_report"),
            ("Tôi muốn biết các nguy cơ sức khỏe trong báo cáo gen của mình", "genetic_report")
        ]
        
        results = []
        correct_predictions = 0
        total_time = 0
        
        logger.info("Testing router performance...")
        
        for query, expected_route in test_queries:
            result = self.route_query(query)
            
            is_correct = result.route_name == expected_route
            if is_correct:
                correct_predictions += 1
            
            total_time += result.processing_time
            
            results.append({
                "query": query,
                "expected": expected_route,
                "predicted": result.route_name,
                "confidence": result.confidence,
                "correct": is_correct,
                "agent": result.agent_name,
                "fallback_used": result.fallback_used,
                "processing_time": result.processing_time
            })
        
        accuracy = correct_predictions / len(test_queries)
        avg_processing_time = total_time / len(test_queries)
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_queries": len(test_queries),
            "average_processing_time": avg_processing_time,
            "total_processing_time": total_time,
            "test_results": results
        }


# Singleton instance for easy access
_route_tool_instance = None

def get_route_tool() -> ImprovedRouteTool:
    """Get singleton instance of the route tool"""
    global _route_tool_instance
    if _route_tool_instance is None:
        _route_tool_instance = ImprovedRouteTool()
    return _route_tool_instance


MAPPING_ROTER_TO_AGENT = {
    "medical_query": "MedicalAgent",
    "genetic_info": "GeneticAgent",
    "drug_info": "DrugAgent",
    "company_info": "CompanyAgent",
    "product_inquiry": "ProductAgent",
    "summary_request": "SummaryAgent",
    "general_chat": "NaiveAgent",
    "product_inquiry": "ProductAgent",
    "genetic_report": "CustomerAgent"
}

def main():
    """Main function for testing the improved route tool"""
    logger.info("=== Testing Improved Route Tool ===")
    
    # Initialize route tool
    route_tool = get_route_tool()
    
    # Get router statistics
    stats = route_tool.get_router_stats()
    logger.info(f"Router Stats: {json.dumps(stats, indent=2)}")
    
    # Test individual queries
    test_queries = [
        "Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?",
        "What is genetic testing?", 
        "Địa chỉ công ty GeneStory ở đâu?",
        "GeneStory có những sản phẩm gì?",
        "Tóm tắt cho tôi thông tin về báo cáo gen",
        "Xin chào, bạn có khỏe không?",
        "Báo cáo gen của tôi có những biến thể nào?",
        "Tôi cần giải thích kết quả báo cáo gen",
        "Chi số Tiểu Đường trong báo cáo gen của tôi là gì?",
        "Báo cáo gen của tôi có những thông tin nào về sức khỏe?",
        "Tôi muốn biết các nguy cơ sức khỏe trong báo cáo gen của mình",
    ]
    
    logger.info("\n=== Individual Query Testing ===")
    for query in test_queries:
        result = route_tool.route_query(query, include_alternatives=True)
        logger.info(f"Query: {query}")
        logger.info(f"Route: {result.route_name} | Agent: {result.agent_name}")
        logger.info(f"Confidence: {result.confidence:.3f} | Fallback: {result.fallback_used}")
        logger.info(f"Time: {result.processing_time:.3f}s")
        if result.error_message:
            logger.warning(f"Error: {result.error_message}")
        logger.info("-" * 50)
    
    # Performance testing
    logger.info("\n=== Performance Testing ===")
    perf_results = route_tool.test_router_performance()
    
    logger.info(f"Overall Accuracy: {perf_results['accuracy']:.1%}")
    logger.info(f"Correct Predictions: {perf_results['correct_predictions']}/{perf_results['total_queries']}")
    logger.info(f"Average Processing Time: {perf_results['average_processing_time']:.3f}s")
    
    # Show detailed results
    logger.info("\n=== Detailed Test Results ===")
    for result in perf_results['test_results']:
        status = "✓" if result['correct'] else "✗"
        fallback = " (FALLBACK)" if result['fallback_used'] else ""
        logger.info(f"{status} {result['query'][:50]}...")
        logger.info(f"  Expected: {result['expected']} | Got: {result['predicted']} | Conf: {result['confidence']:.3f}{fallback}")
    
    # Batch testing
    logger.info("\n=== Batch Testing ===")
    batch_queries = [
        "Gen BRCA1 có ảnh hưởng gì?",
        "Thuốc Warfarin tương tác với gen nào?", 
        "Số điện thoại GeneStory là gì?"
    ]
    
    batch_results = route_tool.batch_route(batch_queries)
    for i, result in enumerate(batch_results):
        logger.info(f"Batch {i+1}: {batch_queries[i][:30]}... -> {result.route_name} ({result.confidence:.3f})")
    
    logger.info("\n=== Testing Complete ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced Router Tools for GeneStory Multi-Agent Chatbot System
This module provides comprehensive routing functionality with improved error handling,
logging, configuration management, and performance monitoring.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import pickle
import hashlib

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder, SparseEncoder
from semantic_router import SemanticRouter
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
route_logger = logging.getLogger(__name__)


class RouteType(Enum):
    """Enumeration of available route types"""
    RETRIEVE = "retrieve"
    CHITCHAT = "chitchat"
    SUMMARY = "summary"
    SEARCHWEB = "searchweb"
    PRODUCT_SQL = "product_sql"


@dataclass
class RouteResult:
    """Result object for route classification"""
    route_name: str
    confidence: float
    processing_time: float
    fallback_used: bool = False
    error_message: Optional[str] = None


@dataclass
class RouterConfig:
    """Configuration for the router"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    auto_sync: str = "local"
    confidence_threshold: float = 0.5
    fallback_route: str = "chitchat"
    enable_evaluation: bool = True
    enable_caching: bool = True
    cache_size: int = 1000


class EnhancedRouterTools:
    """
    Enhanced Router Tools with comprehensive functionality for query routing.
    
    Features:
    - Semantic routing with confidence scoring
    - Performance monitoring and logging
    - Caching for improved performance
    - Fallback mechanisms
    - Configuration management
    - Evaluation and metrics
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        """
        Initialize the Enhanced Router Tools.
        
        Args:
            config: RouterConfig object with router settings
        """
        self.config = config or RouterConfig()
        self.router = None
        self.routes = {}
        self.training_data = []
        self.performance_metrics = {
            "total_queries": 0,
            "successful_routes": 0,
            "fallback_routes": 0,
            "errors": 0,
            "avg_response_time": 0.0
        }
        self.query_cache = {} if self.config.enable_caching else None
        
        logger.info(f"Initializing Enhanced Router Tools with config: {self.config}")
        self._initialize_routes()
        self._initialize_router()
    
    def _initialize_routes(self):
        """Initialize the route definitions"""
        logger.info("Initializing route definitions...")
        
        # Retrieve route - Medical and genetic information queries
        self.routes[RouteType.RETRIEVE] = Route(
            name="retrieve",
            utterances=[
                "Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?",
                "Tại sao tập luyện thể thao thường xuyên lại giúp tăng cường hấp thụ canxi?",
                "Thuyên tắc huyết khối tĩnh mạch được phân loại vào nhóm bệnh nào?",
                "Toi hut thuoc la co nguy co mac benh gi?",
                "Hãy kể tên một vài yếu tố nguy cơ chính gây ra đột quỵ nhồi máu não?",
                "Báo cáo gen này đã khảo sát bao nhiêu biến thể để đưa ra kết quả?",
                "Tại sao mụn trứng cá thường xuất hiện ở các vùng như mặt, ngực và lưng của tôi?",
                "Hay liet ke mot vai dac diem cua benh mun trung ca?",
                "Toi bi ung thu dai truc trang, toi can lam gi?",
                "Gen BRCA1 có ảnh hưởng gì đến nguy cơ ung thư vú?",
                "Các loại thuốc nào tương tác với gen CYP2D6?",
                "Làm thế nào để hiểu kết quả xét nghiệm gen?",
            ],
        )
        
        # Chitchat route - General conversation and company information
        self.routes[RouteType.CHITCHAT] = Route(
            name="chitchat",
            utterances=[
                "Xin chào, tôi muốn biết thêm về GeneStory",
                "Bạn có thể cho tôi biết về công ty GeneStory không?",
                "Tôi muốn tìm hiểu về các dự án của GeneStory",
                "Bạn có thể giúp tôi với thông tin về GeneStory không?",
                "Tôi cần thông tin về GeneStory",
                "Hãy kể cho tôi nghe về GeneStory",
                "Hom nay troi the nao?",
                "Ban co the cho toi biet ve thoi tiet hom nay khong?",
                "Bay gio toi muon di an toi, ban co the goi cho toi mot nha hang ngon khong?",
                "Tôi đang tìm kiếm một bộ phim hay để xem, bạn có gợi ý nào không?",
                "Tôi muốn nghe một câu chuyện thú vị, bạn có thể kể cho tôi không?",
                "Tôi muốn biết thêm về các hoạt động giải trí trong khu vực của tôi",
                "Bạn có thể gợi ý cho tôi một số hoạt động thú vị để làm trong cuối tuần này?",
                "Tôi muốn tìm hiểu về các sự kiện văn hóa sắp tới",
                "Cảm ơn bạn đã giúp đỡ",
                "Tạm biệt và hẹn gặp lại",
            ],
        )
        
        # Summary route - Information summarization requests
        self.routes[RouteType.SUMMARY] = Route(
            name="summary",
            utterances=[
                "Tổng hợp thông tin về công ty GeneStory",
                "Tổng hợp thông tin về dự án Mash",
                "Tổng hợp thông tin về dự án 1000 hệ gen người Việt",
                "Tóm tắt cho tôi thông tin về report gen của tôi",
                "Đưa ra tổng hợp về các dự án của GeneStory",
                "Liệt kê các thông tin Genetic của tôi",
                "Tổng quan về kết quả xét nghiệm gen",
                "Tóm tắt lịch sử y tế của tôi",
                "Đưa ra báo cáo tổng hợp về sức khỏe",
            ],
        )
        
        # SearchWeb route - External information and contact details
        self.routes[RouteType.SEARCHWEB] = Route(
            name="searchweb",
            utterances=[
                "Địa chỉ công ty GeneStory ở đâu?",
                "Văn phòng liên hệ của GeneStory ở đâu?",
                "Làm thế nào để liên hệ với GeneStory?",
                "Email hỗ trợ của GeneStory là gì?",
                "Số hotline của GeneStory là bao nhiêu?",
                "Trang web chính thức của GeneStory là gì?",
                "Thuốc Paracetamol là gì?",
                "Tác dụng phụ của thuốc Paracetamol là gì?",
                "Thông tin về bác sĩ chuyên khoa gen?",
                "Địa chỉ phòng khám di truyền ở Hà Nội?",
                "Giá dịch vụ xét nghiệm gen ở đâu rẻ nhất?",
            ],
        )
        
        # Product SQL route - Product and service inquiries
        # self.routes[RouteType.PRODUCT_SQL] = Route(
        #     name="product_sql",
        #     utterances=[
        #         "Tôi muốn biết về các sản phẩm của GeneStory",
        #         "GeneStory có những sản phẩm gì?",
        #         "Các sản phẩm của GeneStory bao gồm những gì?",
        #         "Tôi cần thông tin về các sản phẩm của GeneStory",
        #         "GeneStory cung cấp những sản phẩm nào?",
        #         "Tôi muốn tìm hiểu về các sản phẩm của GeneStory",
        #         "Giá cả các gói xét nghiệm gen",
        #         "So sánh các gói dịch vụ của GeneStory",
        #         "Tôi muốn đặt mua sản phẩm xét nghiệm gen",
        #     ],
        # )
        
        logger.info(f"Initialized {len(self.routes)} routes successfully")
    
    def _initialize_router(self):
        """Initialize the semantic router"""
        logger.info("Initializing semantic router...")
        
        try:
            encoder = HuggingFaceEncoder(model_name=self.config.model_name)
            route_list = list(self.routes.values())
            
            self.router = SemanticRouter(
                encoder=encoder,
                routes=route_list,
                auto_sync=self.config.auto_sync
            )
            
            # Prepare training data
            self._prepare_training_data()
            
            # Evaluate and fit the router if enabled
            if self.config.enable_evaluation and self.training_data:
                self._evaluate_and_fit_router()
            
            logger.info("Semantic router initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic router: {e}")
            raise
    
    def _prepare_training_data(self):
        """Prepare training data from route utterances"""
        self.training_data = []
        
        for route_type, route in self.routes.items():
            for utterance in route.utterances:
                self.training_data.append((utterance, route.name))
        
        logger.info(f"Prepared {len(self.training_data)} training samples")
    
    def _evaluate_and_fit_router(self):
        """Evaluate and fit the router with training data"""
        if not self.training_data:
            logger.warning("No training data available for evaluation")
            return
        
        try:
            X = [item[0] for item in self.training_data]
            y = [item[1] for item in self.training_data]
            
            # Initial evaluation
            initial_acc = self.router.evaluate(X=X, y=y)
            logger.info(f"Initial evaluation accuracy: {initial_acc:.2%}")
            
            # Fit the router
            self.router.fit(X=X, y=y)
            
            # Final evaluation
            final_acc = self.router.evaluate(X=X, y=y)
            logger.info(f"Final evaluation accuracy: {final_acc:.2%}")
            
            # Store metrics
            self.performance_metrics["initial_accuracy"] = initial_acc
            self.performance_metrics["final_accuracy"] = final_acc
            
        except Exception as e:
            logger.error(f"Error during router evaluation and fitting: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def _check_cache(self, query: str) -> Optional[RouteResult]:
        """Check if query result is cached"""
        if not self.config.enable_caching or not self.query_cache:
            return None
        
        cache_key = self._get_cache_key(query)
        return self.query_cache.get(cache_key)
    
    def _cache_result(self, query: str, result: RouteResult):
        """Cache query result"""
        if not self.config.enable_caching or not self.query_cache:
            return
        
        # Implement simple LRU cache
        if len(self.query_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        cache_key = self._get_cache_key(query)
        self.query_cache[cache_key] = result
    
    def route_query(self, query: str, return_confidence: bool = True) -> Union[str, RouteResult]:
        """
        Route a query to the appropriate agent.
        
        Args:
            query: The input query to route
            return_confidence: Whether to return detailed RouteResult or just route name
            
        Returns:
            RouteResult object or route name string
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self._check_cache(query)
        if cached_result:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_result if return_confidence else cached_result.route_name
        
        try:
            # Update metrics
            self.performance_metrics["total_queries"] += 1
            
            # Route the query
            route_response = self.router(query)
            processing_time = time.time() - start_time
            
            if route_response:
                route_name = route_response.name
                # Try to get confidence score if available
                confidence = getattr(route_response, 'similarity_score', 0.0)
                
                # Check confidence threshold
                if confidence < self.config.confidence_threshold:
                    logger.warning(f"Low confidence route for query: {query[:50]}... "
                                 f"(confidence: {confidence:.2f})")
                    route_name = self.config.fallback_route
                    fallback_used = True
                    self.performance_metrics["fallback_routes"] += 1
                else:
                    fallback_used = False
                    self.performance_metrics["successful_routes"] += 1
                
                result = RouteResult(
                    route_name=route_name,
                    confidence=confidence,
                    processing_time=processing_time,
                    fallback_used=fallback_used
                )
                
            else:
                # No route found, use fallback
                logger.warning(f"No route found for query: {query[:50]}...")
                result = RouteResult(
                    route_name=self.config.fallback_route,
                    confidence=0.0,
                    processing_time=processing_time,
                    fallback_used=True
                )
                self.performance_metrics["fallback_routes"] += 1
            
            # Cache the result
            self._cache_result(query, result)
            
            # Update average response time
            total_time = (self.performance_metrics["avg_response_time"] * 
                         (self.performance_metrics["total_queries"] - 1) + processing_time)
            self.performance_metrics["avg_response_time"] = total_time / self.performance_metrics["total_queries"]
            
            logger.debug(f"Routed query to '{result.route_name}' "
                        f"(confidence: {result.confidence:.2f}, "
                        f"time: {result.processing_time:.3f}s)")
            
            return result if return_confidence else result.route_name
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error routing query: {str(e)}"
            logger.error(error_msg)
            
            self.performance_metrics["errors"] += 1
            
            result = RouteResult(
                route_name=self.config.fallback_route,
                confidence=0.0,
                processing_time=processing_time,
                fallback_used=True,
                error_message=error_msg
            )
            
            return result if return_confidence else result.route_name
    
    def batch_route_queries(self, queries: List[str]) -> List[RouteResult]:
        """
        Route multiple queries in batch.
        
        Args:
            queries: List of queries to route
            
        Returns:
            List of RouteResult objects
        """
        logger.info(f"Batch routing {len(queries)} queries...")
        start_time = time.time()
        
        results = []
        for query in queries:
            result = self.route_query(query, return_confidence=True)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(queries) if queries else 0
        
        logger.info(f"Batch routing completed in {total_time:.3f}s "
                   f"(avg: {avg_time:.3f}s per query)")
        
        return results
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics"""
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "route_count": len(self.routes),
            "training_samples": len(self.training_data),
            "cache_size": len(self.query_cache) if self.query_cache else 0,
            "config": {
                "model_name": self.config.model_name,
                "confidence_threshold": self.config.confidence_threshold,
                "fallback_route": self.config.fallback_route,
                "enable_caching": self.config.enable_caching,
            }
        }
    
    def test_router_accuracy(self, test_data: Optional[List[Tuple[str, str]]] = None) -> Dict[str, float]:
        """
        Test router accuracy on provided test data or training data.
        
        Args:
            test_data: Optional test data as (query, expected_route) tuples
            
        Returns:
            Dictionary with accuracy metrics
        """
        if test_data is None:
            test_data = self.training_data
        
        if not test_data:
            logger.warning("No test data available for accuracy testing")
            return {}
        
        logger.info(f"Testing router accuracy on {len(test_data)} samples...")
        
        correct_predictions = 0
        total_predictions = len(test_data)
        route_accuracy = {}
        
        for query, expected_route in test_data:
            result = self.route_query(query, return_confidence=True)
            predicted_route = result.route_name
            
            if predicted_route == expected_route:
                correct_predictions += 1
            
            # Track per-route accuracy
            if expected_route not in route_accuracy:
                route_accuracy[expected_route] = {"correct": 0, "total": 0}
            
            route_accuracy[expected_route]["total"] += 1
            if predicted_route == expected_route:
                route_accuracy[expected_route]["correct"] += 1
        
        overall_accuracy = correct_predictions / total_predictions
        
        # Calculate per-route accuracy
        for route in route_accuracy:
            route_accuracy[route]["accuracy"] = (
                route_accuracy[route]["correct"] / route_accuracy[route]["total"]
            )
        
        accuracy_metrics = {
            "overall_accuracy": overall_accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "route_accuracy": route_accuracy
        }
        
        logger.info(f"Router accuracy: {overall_accuracy:.2%}")
        
        return accuracy_metrics
    
    def save_router_state(self, filepath: str):
        """Save router state to file"""
        try:
            state = {
                "config": self.config,
                "performance_metrics": self.performance_metrics,
                "query_cache": self.query_cache
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Router state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save router state: {e}")
    
    def load_router_state(self, filepath: str):
        """Load router state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.config = state.get("config", self.config)
            self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
            if self.config.enable_caching:
                self.query_cache = state.get("query_cache", {})
            
            logger.info(f"Router state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load router state: {e}")
    
    def add_custom_route(self, route_name: str, utterances: List[str]):
        """
        Add a custom route to the router.
        
        Args:
            route_name: Name of the new route
            utterances: List of example utterances for the route
        """
        try:
            custom_route = Route(name=route_name, utterances=utterances)
            
            # Add to routes dict
            self.routes[route_name] = custom_route
            
            # Reinitialize router with new route
            route_list = list(self.routes.values())
            encoder = HuggingFaceEncoder(model_name=self.config.model_name)
            
            self.router = SemanticRouter(
                encoder=encoder,
                routes=route_list,
                auto_sync=self.config.auto_sync
            )
            
            # Update training data
            for utterance in utterances:
                self.training_data.append((utterance, route_name))
            
            logger.info(f"Added custom route '{route_name}' with {len(utterances)} utterances")
            
        except Exception as e:
            logger.error(f"Failed to add custom route '{route_name}': {e}")
    
    def clear_cache(self):
        """Clear the query cache"""
        if self.query_cache:
            self.query_cache.clear()
            logger.info("Query cache cleared")


def create_default_router() -> EnhancedRouterTools:
    """Create a default router with standard configuration"""
    config = RouterConfig(
        enable_evaluation=True,
        enable_caching=True,
        confidence_threshold=0.3,
        fallback_route="chitchat"
    )
    
    return EnhancedRouterTools(config)


def main():
    """Main function for testing the router tools"""
    logger.info("=" * 60)
    logger.info("Enhanced Router Tools - Testing Suite")
    logger.info("=" * 60)
    
    # Create router
    router_tools = create_default_router()
    
    # Test queries for each route type
    test_queries = {
        "retrieve": [
            "Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?",
            "Gen BRCA1 có ảnh hưởng gì đến nguy cơ ung thư vú?",
            "Tôi bị ung thư đại trực tràng, tôi cần làm gì?",
        ],
        "chitchat": [
            "Xin chào, tôi muốn biết thêm về GeneStory",
            "Hôm nay thời tiết thế nào?",
            "Cảm ơn bạn đã giúp đỡ",
        ],
        "summary": [
            "Tổng hợp thông tin về công ty GeneStory",
            "Tóm tắt kết quả xét nghiệm gen của tôi",
            "Đưa ra báo cáo tổng hợp về sức khỏe",
        ],
        "searchweb": [
            "Địa chỉ công ty GeneStory ở đâu?",
            "Thuốc Paracetamol có tác dụng phụ gì?",
            "Số hotline hỗ trợ khách hàng là bao nhiêu?",
        ],
        "product_sql": [
            "GeneStory có những sản phẩm gì?",
            "Giá cả các gói xét nghiệm gen",
            "Tôi muốn đặt mua sản phẩm xét nghiệm",
        ]
    }
    
    logger.info("\n🧪 Testing Individual Queries")
    logger.info("-" * 40)
    
    all_results = []
    for expected_route, queries in test_queries.items():
        logger.info(f"\nTesting {expected_route.upper()} route:")
        for query in queries:
            result = router_tools.route_query(query, return_confidence=True)
            all_results.append((query, expected_route, result))
            
            status = "✅" if result.route_name == expected_route else "❌"
            logger.info(f"{status} Query: {query[:50]}...")
            logger.info(f"   Expected: {expected_route}, Got: {result.route_name}")
            logger.info(f"   Confidence: {result.confidence:.2f}, Time: {result.processing_time:.3f}s")
    
    # Test accuracy
    logger.info("\n📊 Router Accuracy Test")
    logger.info("-" * 40)
    
    accuracy_metrics = router_tools.test_router_accuracy()
    if accuracy_metrics:
        logger.info(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.2%}")
        logger.info(f"Correct: {accuracy_metrics['correct_predictions']}/{accuracy_metrics['total_predictions']}")
    
    # Performance statistics
    logger.info("\n📈 Performance Statistics")
    logger.info("-" * 40)
    
    stats = router_tools.get_route_statistics()
    metrics = stats["performance_metrics"]
    
    logger.info(f"Total Queries: {metrics['total_queries']}")
    logger.info(f"Successful Routes: {metrics['successful_routes']}")
    logger.info(f"Fallback Routes: {metrics['fallback_routes']}")
    logger.info(f"Errors: {metrics['errors']}")
    logger.info(f"Average Response Time: {metrics['avg_response_time']:.3f}s")
    logger.info(f"Cache Size: {stats['cache_size']}")
    
    # Test batch routing
    logger.info("\n🚀 Batch Routing Test")
    logger.info("-" * 40)
    
    batch_queries = [
        "Xin chào GeneStory",
        "Tôi muốn biết về sản phẩm",
        "Địa chỉ công ty ở đâu?",
        "Tóm tắt thông tin gen",
        "Triệu chứng ung thư là gì?"
    ]
    
    batch_results = router_tools.batch_route_queries(batch_queries)
    for i, result in enumerate(batch_results):
        logger.info(f"Query {i+1}: {batch_queries[i][:30]}... -> {result.route_name}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

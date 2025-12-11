#!/usr/bin/env python3
"""
Simple Route Tool - Easy-to-use routing interface for GeneStory chatbot
This module provides a simplified interface for query routing with minimal setup.
"""

import time
from typing import Optional, Union, Dict, Any
from loguru import logger

try:
    from .enhanced_router_tools import EnhancedRouterTools, RouterConfig, RouteResult, RouteType
except ImportError:
    try:
        from enhanced_router_tools import EnhancedRouterTools, RouterConfig, RouteResult, RouteType
    except ImportError:
        logger.error("Could not import enhanced router tools. Please ensure the module is available.")
        raise


class RouteTool:
    """
    Simple route tool that provides an easy interface for query routing.
    This is a wrapper around EnhancedRouterTools for easier integration.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one router instance"""
        if cls._instance is None:
            cls._instance = super(RouteTool, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the route tool"""
        if not self._initialized:
            logger.info("Initializing RouteTool...")
            self._setup_router()
            RouteTool._initialized = True
    
    def _setup_router(self):
        """Setup the enhanced router with optimized configuration"""
        try:
            # Create optimized configuration
            config = RouterConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                confidence_threshold=0.3,  # Lower threshold for better coverage
                fallback_route="chitchat",  # Safe fallback for general queries
                enable_evaluation=True,
                enable_caching=True,
                cache_size=500,  # Reasonable cache size
                auto_sync="local"
            )
            
            self.router = EnhancedRouterTools(config)
            self.is_ready = True
            logger.info("RouteTool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RouteTool: {e}")
            self.is_ready = False
            raise
    
    def route(self, query: str, detailed: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Route a query to the appropriate agent.
        
        Args:
            query (str): The user query to route
            detailed (bool): If True, return detailed routing information
            
        Returns:
            str: Route name (if detailed=False)
            dict: Detailed routing information (if detailed=True)
        """
        if not self.is_ready:
            logger.warning("RouteTool not ready, using fallback routing")
            return "chitchat" if not detailed else {
                "route": "chitchat",
                "confidence": 0.0,
                "fallback": True,
                "error": "Router not initialized"
            }
        
        try:
            result = self.router.route_query(query, return_confidence=True)
            
            if detailed:
                return {
                    "route": result.route_name,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "fallback_used": result.fallback_used,
                    "error": result.error_message
                }
            else:
                return result.route_name
                
        except Exception as e:
            logger.error(f"Error in routing query: {e}")
            if detailed:
                return {
                    "route": "chitchat",
                    "confidence": 0.0,
                    "fallback": True,
                    "error": str(e)
                }
            else:
                return "chitchat"
    
    def batch_route(self, queries: list) -> list:
        """
        Route multiple queries at once.
        
        Args:
            queries (list): List of query strings
            
        Returns:
            list: List of route names
        """
        if not self.is_ready:
            return ["chitchat"] * len(queries)
        
        try:
            results = self.router.batch_route_queries(queries)
            return [result.route_name for result in results]
            
        except Exception as e:
            logger.error(f"Error in batch routing: {e}")
            return ["chitchat"] * len(queries)
    
    def get_available_routes(self) -> list:
        """Get list of available routes"""
        return [route.value for route in RouteType]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.is_ready:
            return {"error": "Router not initialized"}
        
        try:
            return self.router.get_route_statistics()
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def test_accuracy(self) -> Dict[str, float]:
        """Test router accuracy"""
        if not self.is_ready:
            return {"error": "Router not initialized"}
        
        try:
            return self.router.test_router_accuracy()
        except Exception as e:
            logger.error(f"Error testing accuracy: {e}")
            return {"error": str(e)}


# Global router instance
_route_tool = None


def get_route_tool() -> RouteTool:
    """Get the global route tool instance"""
    global _route_tool
    if _route_tool is None:
        _route_tool = RouteTool()
    return _route_tool


def route_query(query: str) -> str:
    """
    Simple function to route a query to the appropriate agent.
    
    Args:
        query (str): The user query
        
    Returns:
        str: The route name (retrieve, chitchat, summary, searchweb, product_sql)
    """
    tool = get_route_tool()
    return tool.route(query)


def route_query_detailed(query: str) -> Dict[str, Any]:
    """
    Route a query with detailed information.
    
    Args:
        query (str): The user query
        
    Returns:
        dict: Detailed routing information including confidence, timing, etc.
    """
    tool = get_route_tool()
    return tool.route(query, detailed=True)


def batch_route_queries(queries: list) -> list:
    """
    Route multiple queries at once.
    
    Args:
        queries (list): List of query strings
        
    Returns:
        list: List of route names
    """
    tool = get_route_tool()
    return tool.batch_route(queries)


def get_routing_statistics() -> Dict[str, Any]:
    """Get routing performance statistics"""
    tool = get_route_tool()
    return tool.get_statistics()


def test_routing_accuracy() -> Dict[str, float]:
    """Test routing accuracy on training data"""
    tool = get_route_tool()
    return tool.test_accuracy()


# Route mapping for agent orchestration
ROUTE_TO_AGENT_MAPPING = {
    "retrieve": "CustomerAgent",  # Medical/genetic queries go to customer agent
    "chitchat": "GuestAgent",     # General chat goes to guest agent
    "summary": "CustomerAgent",   # Summarization goes to customer agent
    "searchweb": "GuestAgent",    # Web search goes to guest agent
    "product_sql": "ProductAgent" # Product queries go to product agent
}


def route_to_agent(query: str) -> str:
    """
    Route a query directly to an agent name.
    
    Args:
        query (str): The user query
        
    Returns:
        str: The agent name (CustomerAgent, GuestAgent, ProductAgent, etc.)
    """
    route = route_query(query)
    return ROUTE_TO_AGENT_MAPPING.get(route, "GuestAgent")


if __name__ == "__main__":
    # Test the route tool
    print("=" * 50)
    print("Route Tool - Quick Test")
    print("=" * 50)
    
    # Test basic routing
    test_queries = [
        "Xin chào, tôi muốn biết về GeneStory",
        "Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?",
        "Tôi muốn biết về các sản phẩm của GeneStory",
        "Địa chỉ công ty GeneStory ở đâu?",
        "Tóm tắt thông tin gen của tôi"
    ]
    
    print("\n🧪 Basic Routing Test:")
    for query in test_queries:
        route = route_query(query)
        agent = route_to_agent(query)
        print(f"Query: {query[:40]}...")
        print(f"Route: {route} -> Agent: {agent}\n")
    
    print("📊 Detailed Routing Test:")
    detailed = route_query_detailed("Tôi bị ung thư, cần làm gì?")
    print(f"Detailed result: {detailed}")
    
    print("\n🚀 Batch Routing Test:")
    batch_routes = batch_route_queries(test_queries[:3])
    print(f"Batch routes: {batch_routes}")
    
    print("\n📈 Statistics:")
    stats = get_routing_statistics()
    if "performance_metrics" in stats:
        metrics = stats["performance_metrics"]
        print(f"Total queries: {metrics.get('total_queries', 0)}")
        print(f"Average response time: {metrics.get('avg_response_time', 0):.3f}s")
    
    print("\n✅ Route Tool test completed!")

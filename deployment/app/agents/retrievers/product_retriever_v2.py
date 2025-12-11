"""
Enhanced ProductRetrieverTool with negative query handling and query intent analysis.
This addresses the issue where queries like "Có thông tin về các gói dịch vụ khác ngoài Thẻ Genemark không?"
are not handled correctly by standard RAG systems.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass
from loguru import logger
from app.agents.retrievers.product_retriever import ProductRetrieverTool
from functools import wraps
import time
class QueryType(Enum):
    """Types of queries the system can handle"""
    SIMPLE_SEARCH = "simple_search"  # "thông tin về genemap adult"
    EXCLUSION = "exclusion"          # "các gói khác ngoài genemap adult"
    COMPARISON = "comparison"        # "so sánh genemap adult và kid"
    LISTING = "listing"             # "tất cả các gói dịch vụ"
    CONDITIONAL = "conditional"     # "nếu tôi muốn xét nghiệm gen"

@dataclass
class QueryIntent:
    """Represents the parsed intent of a user query"""
    query_type: QueryType
    main_terms: List[str]           # Main search terms
    exclusion_terms: List[str]      # Terms to exclude
    comparison_terms: List[str]     # Terms for comparison
    context_terms: List[str]        # Additional context
    confidence: float               # Confidence in the parsing
    original_query: str

class QueryIntentAnalyzer:
    """Analyzes user queries to understand intent and handle negative queries"""
    
    def __init__(self):
        # Vietnamese patterns for different query types
        self.exclusion_patterns = [
            r'(?:khác\s+)?(?:ngoài|trừ|loại\s+trừ)\s+([^?]+)',
            r'(?:không\s+phải|không\s+bao\s+gồm)\s+([^?]+)',
            r'(?:ngoại\s+trừ|ngoại\s+lệ)\s+([^?]+)',
            r'(?:khác\s+với|khác\s+so\s+với)\s+([^?]+)',
        ]
        
        self.listing_patterns = [
            r'(?:tất\s+cả|toàn\s+bộ|danh\s+sách)\s+(?:các\s+)?(.+)',
            r'(?:có\s+)?(?:những|các)\s+(.+?)\s+(?:nào|gì)',
            r'(?:liệt\s+kê|kể\s+ra)\s+(.+)',
        ]
        
        self.comparison_patterns = [
            r'(?:so\s+sánh|đối\s+chiếu)\s+(.+?)\s+(?:và|với)\s+(.+)',
            r'(?:khác\s+biệt|khác\s+nhau)\s+(?:giữa\s+)?(.+?)\s+(?:và|với)\s+(.+)',
            r'(.+?)\s+(?:hay|hoặc)\s+(.+?)\s+(?:tốt\s+hơn|phù\s+hợp\s+hơn)',
        ]
        
        self.product_terms = {
            'gói', 'dịch vụ', 'sản phẩm', 'xét nghiệm', 'test', 'giải mã',
            'genemap', 'genemark', 'genestory', 'adult', 'kid', 'premium'
        }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """Analyze a query and return its intent"""
        query_lower = query.lower().strip()
        
        # Check for exclusion patterns first (most complex)
        exclusion_match = self._check_exclusion_patterns(query_lower)
        if exclusion_match:
            return self._build_exclusion_intent(query, exclusion_match)
        
        # Check for comparison patterns
        comparison_match = self._check_comparison_patterns(query_lower)
        if comparison_match:
            return self._build_comparison_intent(query, comparison_match)
        
        # Check for listing patterns
        listing_match = self._check_listing_patterns(query_lower)
        if listing_match:
            return self._build_listing_intent(query, listing_match)
        
        # Default to simple search
        main_terms = self._extract_product_terms(query_lower)
        return QueryIntent(
            query_type=QueryType.SIMPLE_SEARCH,
            main_terms=main_terms,
            exclusion_terms=[],
            comparison_terms=[],
            context_terms=[],
            confidence=0.8,
            original_query=query
        )
    
    def _check_exclusion_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if query matches exclusion patterns"""
        for pattern in self.exclusion_patterns:
            match = re.search(pattern, query)
            if match:
                exclusion_text = match.group(1).strip()
                # Find what user is actually looking for
                query_before = query[:match.start()].strip()
                query_after = query[match.end():].strip()
                
                main_search = query_before + " " + query_after
                main_search = re.sub(r'\s+', ' ', main_search).strip()
                
                return {
                    'exclusion_text': exclusion_text,
                    'main_search': main_search,
                    'confidence': 0.9
                }
        return None
    
    def _check_comparison_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if query matches comparison patterns"""
        for pattern in self.comparison_patterns:
            match = re.search(pattern, query)
            if match:
                return {
                    'term1': match.group(1).strip(),
                    'term2': match.group(2).strip(),
                    'confidence': 0.85
                }
        return None
    
    def _check_listing_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if query matches listing patterns"""
        for pattern in self.listing_patterns:
            match = re.search(pattern, query)
            if match:
                return {
                    'category': match.group(1).strip(),
                    'confidence': 0.8
                }
        return None
    
    def _extract_product_terms(self, query: str) -> List[str]:
        """Extract product-related terms from query"""
        words = re.findall(r'\w+', query)
        return [word for word in words if word in self.product_terms or len(word) > 3]
    
    def _build_exclusion_intent(self, query: str, match: Dict) -> QueryIntent:
        """Build intent for exclusion queries"""
        exclusion_terms = self._extract_product_terms(match['exclusion_text'])
        main_terms = self._extract_product_terms(match['main_search'])
        
        return QueryIntent(
            query_type=QueryType.EXCLUSION,
            main_terms=main_terms,
            exclusion_terms=exclusion_terms,
            comparison_terms=[],
            context_terms=[],
            confidence=match['confidence'],
            original_query=query
        )
    
    def _build_comparison_intent(self, query: str, match: Dict) -> QueryIntent:
        """Build intent for comparison queries"""
        term1_tokens = self._extract_product_terms(match['term1'])
        term2_tokens = self._extract_product_terms(match['term2'])
        
        return QueryIntent(
            query_type=QueryType.COMPARISON,
            main_terms=term1_tokens + term2_tokens,
            exclusion_terms=[],
            comparison_terms=[match['term1'], match['term2']],
            context_terms=[],
            confidence=match['confidence'],
            original_query=query
        )
    
    def _build_listing_intent(self, query: str, match: Dict) -> QueryIntent:
        """Build intent for listing queries"""
        category_terms = self._extract_product_terms(match['category'])
        
        return QueryIntent(
            query_type=QueryType.LISTING,
            main_terms=category_terms,
            exclusion_terms=[],
            comparison_terms=[],
            context_terms=[match['category']],
            confidence=match['confidence'],
            original_query=query
        )

def gpu_management_decorator(func):
    """Decorator to manage GPU resources for retrieval methods."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if object was pickled/unpickled
        if hasattr(self, '_was_pickled') and getattr(self, '_was_pickled', False):
            logger.info("Detected unpickled instance - reinitializing core components")
            try:
                # Reinitialize core components if needed
                if not hasattr(self, '_vector_store') or self._vector_store is None:
                    # Need to reinitialize everything
                    self._initialize_all()
            except Exception as e:
                logger.error(f"Failed to reinitialize after unpickling: {e}")
        
        # Try to ensure GPU components are available
        try:
            if not hasattr(self, '_ensure_gpu_components') or not self._ensure_gpu_components():
                logger.warning("Failed to ensure GPU components availability")
                # Try to fall back to CPU if possible
                if not hasattr(self, '_final_retriever') or self._final_retriever is None:
                    try:
                        # Initialize with CPU as fallback
                        self._initialize_core_components(use_gpu=False)
                        self._build_retriever_pipeline(force_gpu=False)
                    except Exception as e:
                        logger.error(f"CPU fallback initialization failed: {e}")
                        return "Retrieval tool temporarily unavailable. Please try again in a moment."
        except Exception as gpu_err:
            logger.error(f"Error ensuring GPU components: {gpu_err}")
            # Continue with potentially limited functionality
        
        try:
            # Run the retrieval function
            result = await func(self, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in retrieval function: {str(e)}")
            return f"Error retrieving information: {str(e)}"
        finally:
            # Update last used time for GPU components
            if hasattr(self, '_gpu_components_active') and getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
            
            # Check memory pressure
            try:
                if hasattr(self, '_check_memory_pressure'):
                    self._check_memory_pressure()
            except Exception as e:
                logger.error(f"Error checking memory pressure: {e}")
            
    return wrapper   
class EnhancedProductRetriever:
    """Enhanced retriever that handles negative queries and intent analysis"""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.intent_analyzer = QueryIntentAnalyzer()
        
    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        """Enhanced retrieval with intent analysis"""
        # Analyze query intent first
        intent = self.intent_analyzer.analyze_query(query)
        logger.info(f"Query intent: {intent.query_type.value}, confidence: {intent.confidence}")
        
        if intent.query_type == QueryType.EXCLUSION:
            return self._handle_exclusion_query(intent, use_cache)
        elif intent.query_type == QueryType.COMPARISON:
            return self._handle_comparison_query(intent, use_cache)
        elif intent.query_type == QueryType.LISTING:
            return self._handle_listing_query(intent, use_cache)
        else:
            # Simple search - use base retriever
            return self.base_retriever.retrieve_documents(query, use_cache)
    
    def _handle_exclusion_query(self, intent: QueryIntent, use_cache: bool = True) -> List[str]:
        """Handle exclusion queries like 'services other than genemap'"""
        logger.info(f"Handling exclusion query. Excluding: {intent.exclusion_terms}")
        
        # Step 1: Get all relevant documents for the main category
        category_query = " ".join(intent.main_terms + intent.context_terms)
        if not category_query.strip():
            category_query = "gói dịch vụ sản phẩm"  # Default broad search
        
        all_docs = self.base_retriever.retrieve_documents(category_query, use_cache)
        
        # Step 2: Filter out documents that contain exclusion terms
        filtered_docs = []
        exclusion_terms_lower = [term.lower() for term in intent.exclusion_terms]
        
        for doc in all_docs:
            doc_lower = doc.lower()
            
            # Check if any exclusion term is in the document
            should_exclude = False
            for exclusion_term in exclusion_terms_lower:
                if exclusion_term in doc_lower:
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_docs.append(doc)
        
        # Step 3: If we have results, return them. Otherwise, provide helpful response
        if filtered_docs:
            logger.info(f"Found {len(filtered_docs)} documents after exclusion filtering")
            return filtered_docs
        else:
            # Try a broader search to see what's available
            broad_docs = self.base_retriever.retrieve_documents("gói dịch vụ", use_cache)
            if broad_docs:
                return [
                    "Dựa vào tiêu chí tìm kiếm của bạn, đây là các gói dịch vụ khác có sẵn:",
                    *broad_docs[:3]  # Return top 3 alternatives
                ]
            else:
                return ["Không tìm thấy gói dịch vụ nào phù hợp với yêu cầu của bạn."]
    
    def _handle_comparison_query(self, intent: QueryIntent, use_cache: bool = True) -> List[str]:
        """Handle comparison queries"""
        logger.info(f"Handling comparison query: {intent.comparison_terms}")
        
        results = []
        
        # Get documents for each comparison term
        for term in intent.comparison_terms:
            term_docs = self.base_retriever.retrieve_documents(term, use_cache)
            if term_docs:
                results.append(f"=== Thông tin về {term.upper()} ===")
                results.extend(term_docs[:2])  # Top 2 for each term
                results.append("")  # Empty line for separation
        
        if not results:
            return ["Không tìm thấy thông tin để so sánh các sản phẩm bạn yêu cầu."]
        
        return results
    
    def _handle_listing_query(self, intent: QueryIntent, use_cache: bool = True) -> List[str]:
        """Handle listing queries like 'all services'"""
        logger.info(f"Handling listing query for category: {intent.context_terms}")
        
        # Use broad terms to get comprehensive results
        broad_query = " ".join(intent.main_terms) if intent.main_terms else "gói dịch vụ sản phẩm"
        
        all_docs = self.base_retriever.retrieve_documents(broad_query, use_cache)
        
        if all_docs:
            # Extract unique services/products from documents
            services = self._extract_service_names(all_docs)
            
            if services:
                result = ["Danh sách các gói dịch vụ có sẵn:"]
                result.extend([f"• {service}" for service in services])
                result.append("\nChi tiết:")
                result.extend(all_docs[:3])  # Add detailed info
                return result
            else:
                return all_docs
        else:
            return ["Không tìm thấy danh sách dịch vụ nào."]
    
    def _extract_service_names(self, documents: List[str]) -> List[str]:
        """Extract service names from documents"""
        services = set()
        
        # Common patterns for service names in Vietnamese
        patterns = [
            r'(?:gói|dịch vụ|sản phẩm)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:package|test|kit)',
            r'GeneMap\s+(\w+)',
            r'GeneMark\s+(\w+)',
        ]
        
        for doc in documents:
            for pattern in patterns:
                matches = re.findall(pattern, doc, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, str) and len(match.strip()) > 2:
                        services.add(match.strip().title())
        
        return sorted(list(services))


# Integration with existing ProductRetrieverTool
class ProductRetrieverToolEnhanced(ProductRetrieverTool):
    """Enhanced version of ProductRetrieverTool with negative query handling"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enhanced_retriever = None
        
    def _initialize_all(self):
        """Override to initialize enhanced retriever"""
        super()._initialize_all()
        self.enhanced_retriever = EnhancedProductRetriever(self)
        logger.info("Enhanced ProductRetrieverTool with negative query handling initialized")
    
    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        """Override to use enhanced retrieval"""
        if self.enhanced_retriever:
            return self.enhanced_retriever.retrieve_documents(query, use_cache)
        else:
            # Fallback to original method
            return super().retrieve_documents(query, use_cache)
    
    @gpu_management_decorator
    async def _arun(self, query: str) -> str:
        """Enhanced async run with intent-aware retrieval"""
        if not self.enhanced_retriever:
            return await super()._arun(query)
        
        try:
            start_time = time.time()
            
            # Use enhanced retrieval
            results = self.enhanced_retriever.retrieve_documents(query, use_cache=True)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Enhanced retrieval completed in {retrieval_time:.2f}s for query: '{query}'")
            
            if not results:
                return "Không tìm thấy kết quả phù hợp với truy vấn của bạn."
            
            # Format results
            if isinstance(results, list):
                return "\n\n".join(results)
            else:
                return str(results)
                
        except Exception as e:
            logger.error(f"Error in enhanced retrieval for query: '{query}'. Error: {e}")
            # Fallback to original method
            return await super()._arun(query)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_enhanced_retriever():
        """Test the enhanced retriever with various query types"""
        
        # Mock data directory for testing
        from pathlib import Path
        test_dir = Path("./test_product_data")
        test_dir.mkdir(exist_ok=True)
        
        # Create test documents
        with open(test_dir / "genemap_adult.txt", "w", encoding="utf-8") as f:
            f.write("""
            Gói GeneMap Adult - Giải mã gen toàn diện cho người lớn
            Giá: 5.000.000 VNĐ
            Bao gồm: Xét nghiệm 500+ chỉ số gen
            Phù hợp: Người trên 18 tuổi
            """)
        
        with open(test_dir / "genemap_kid.txt", "w", encoding="utf-8") as f:
            f.write("""
            Gói GeneMap Kid - Giải mã gen cho trẻ em
            Giá: 3.500.000 VNĐ  
            Bao gồm: Xét nghiệm 300+ chỉ số gen phù hợp trẻ em
            Phù hợp: Trẻ em từ 1-17 tuổi
            """)
        
        with open(test_dir / "premium_service.txt", "w", encoding="utf-8") as f:
            f.write("""
            Gói Premium Consultation - Tư vấn chuyên sâu
            Giá: 2.000.000 VNĐ
            Bao gồm: Tư vấn 1-1 với chuyên gia di truyền
            Thời gian: 90 phút tư vấn trực tiếp
            """)
        
        # Initialize enhanced retriever
        retriever = ProductRetrieverToolEnhanced(watch_directory=str(test_dir))
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Test cases
        test_queries = [
            "thông tin về genemap adult",  # Simple search
            "các gói dịch vụ khác ngoài genemap adult",  # Exclusion query
            "có thông tin về các gói khác ngoài thẻ genemark genestory không?",  # Complex exclusion
            "so sánh genemap adult và genemap kid",  # Comparison
            "tất cả các gói dịch vụ có sẵn",  # Listing
            "những gói nào phù hợp cho trẻ em"  # Context search
        ]
        
        print("=== Testing Enhanced Product Retriever ===\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"{i}. Query: '{query}'")
            print("-" * 50)
            
            try:
                result = await retriever._arun(query)
                print(f"Result:\n{result}\n")
            except Exception as e:
                print(f"Error: {e}\n")
            
            print("=" * 60)
        
        # Cleanup
        retriever.cleanup()
    
    # Run the test
    asyncio.run(test_enhanced_retriever())
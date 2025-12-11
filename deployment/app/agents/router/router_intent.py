from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder, SparseEncoder
from semantic_router import SemanticRouter
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define routes
retrieve = Route(
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
    ],
    description="This route handles queries related to medical conditions, genetic reports, and health risks. It retrieves information about diseases, genetic predispositions, and health-related inquiries.",
)
chitchat = Route(
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
    ],
    description="This route handles general inquiries about GeneStory, casual conversations, and requests for information about the company and its projects.",
)

summary = Route(
    name="summary",
    utterances=[
        "Tong hop thong tin ve cong ty GeneStory",
        "Tong hop thong tin ve du an Mash",
        "Tong hop thong tin ve du an 1000 he gen nguoi Viet",
        "Tom tat cho toi thon tin ve report gen cua toi",
        "Dua ra tong hop ve cac du an cua GeneStory",
        "Liet ke cac thong tin Genetic cua toi",
        
    ],
    description="This route handles requests for summaries and overviews of GeneStory projects, genetic reports, and related information.",
)


searchweb = Route(
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
    ],
    description="This route handles web search queries related to GeneStory's contact information, official website, and general inquiries about products like Paracetamol.",   
)
# *** ROUTE MỚI ĐƯỢC THÊM VÀO ***
toxic = Route(
    name="toxic",
    utterances=[
        "Mày là đồ ngu.",
        "Cút đi cho khuất mắt tao.",
        "Bọn mày làm ăn như lừa đảo.",
        "Trả lời như một cái máy, vô dụng.",
        "Đồ điên.",
        "Câm mồm.",
        "Tao không muốn nói chuyện với mày nữa.",
        "Ngu vãi l*n.",
        "Mày có biết gì đâu mà nói.",
        "Mày có não không đấy?",
        "Mày có thấy mình ngu không?",
        "Neu tao lay gene cua mày, tao se khong bao gio bi ngu nhu mày.",
        
    ],
)
# Fix the inconsistency between "product" and "product_sql"
product = Route(
    name="product",
    utterances=[
        "Tôi muốn biết về các sản phẩm của GeneStory",
        "GeneStory có những sản phẩm gì?",
        "Các sản phẩm của GeneStory bao gồm những gì?",
        "Tôi cần thông tin về các sản phẩm của GeneStory",
        "GeneStory cung cấp những sản phẩm nào?",
        "Tôi muốn tìm hiểu về các sản phẩm của GeneStory",
        
    ],
    description="This route handles queries related to GeneStory's products and services, providing information about what they offer.",
)

# Update sample data to use "product" consistently
sample_data = [
    # retrieve
    ("Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?", "retrieve"),
    ("Tại sao tập luyện thể thao thường xuyên lại giúp tăng cường hấp thụ canxi?", "retrieve"),
    ("Thuyên tắc huyết khối tĩnh mạch được phân loại vào nhóm bệnh nào?", "retrieve"),
    ("Toi hut thuoc la co nguy co mac benh gi?", "retrieve"),
    ("Hãy kể tên một vài yếu tố nguy cơ chính gây ra đột quỵ nhồi máu não?", "retrieve"),
    ("Báo cáo gen này đã khảo sát bao nhiêu biến thể để đưa ra kết quả?", "retrieve"),
    ("Tại sao mụn trứng cá thường xuất hiện ở các vùng như mặt, ngực và lưng của tôi?", "retrieve"),
    ("Hay liet ke mot vai dac diem cua benh mun trung ca?", "retrieve"),
    ("Toi bi ung thu dai truc trang, toi can lam gi?", "retrieve"),
    ("So sánh các gói xét nghiệm gen ung thư của  bạn.", "retrieve"),

    # chitchat
    ("Xin chào, tôi muốn biết thêm về GeneStory", "chitchat"),
    ("Xin chào!","chitchat"),
    ("Hello, how are you today?", "chitchat"),
    ("Bạn có thể cho tôi biết về công ty GeneStory không?", "chitchat"),
    ("Tôi muốn tìm hiểu về các dự án của GeneStory", "chitchat"),
    ("Bạn có thể giúp tôi với thông tin về GeneStory không?", "chitchat"),
    ("Tôi cần thông tin về GeneStory", "chitchat"),
    ("Hãy kể cho tôi nghe về GeneStory", "chitchat"),
    ("Hom nay troi the nao?", "chitchat"),
    ("Ban co the cho toi biet ve thoi tiet hom nay khong?", "chitchat"),
    ("Bay gio toi muon di an toi, ban co the goi cho toi mot nha hang ngon khong?", "chitchat"),
    ("Tôi đang tìm kiếm một bộ phim hay để xem, bạn có gợi ý nào không?", "chitchat"),
    ("Tôi muốn nghe một câu chuyện thú vị, bạn có thể kể cho tôi không?", "chitchat"),
    ("Tôi muốn biết thêm về các hoạt động giải trí trong khu vực của tôi", "chitchat"),
    ("Bạn có thể gợi ý cho tôi một số hoạt động thú vị để làm trong cuối tuần này?", "chitchat"),
    ("Tôi muốn tìm hiểu về các sự kiện văn hóa sắp tới", "chitchat"),

    # summary
    ("Tong hop thong tin ve cong ty GeneStory", "summary"),
    ("Tong hop thong tin ve du an Mash", "summary"),
    ("Tong hop thong tin ve du an 1000 he gen nguoi Viet", "summary"),
    ("Tom tat cho toi thon tin ve report gen cua toi", "summary"),
    ("Dua ra tong hop ve cac du an cua GeneStory", "summary"),
    ("Liet ke cac thong tin Genetic cua toi", "summary"),

    # searchweb
    ("Địa chỉ công ty GeneStory ở đâu?", "searchweb"),
    ("Văn phòng liên hệ của GeneStory ở đâu?", "searchweb"),
    ("Làm thế nào để liên hệ với GeneStory?", "searchweb"),
    ("Email hỗ trợ của GeneStory là gì?", "searchweb"),
    ("Số hotline của GeneStory là bao nhiêu?", "searchweb"),
    ("Trang web chính thức của GeneStory là gì?", "searchweb"),
    ("Thuốc Paracetamol là gì?", "searchweb"),
    ("Tác dụng phụ của thuốc Paracetamol là gì?", "searchweb"),

    # product
    ("Tôi muốn biết về các sản phẩm của GeneStory", "product"),
    ("GeneStory có những sản phẩm gì?", "product"),
    ("Các sản phẩm của GeneStory bao gồm những gì?", "product"),
    ("Tôi cần thông tin về các sản phẩm của GeneStory", "product"),
    ("GeneStory cung cấp những sản phẩm nào?", "product"),
    ("Tôi muốn tìm hiểu về các sản phẩm của GeneStory", "product"),

    # toxic
    ("Mày là đồ ngu.", "toxic"),
    ("Cút đi.", "toxic"),
    ("Bọn mày làm ăn như lừa đảo.", "toxic"),
    ("Trả lời như cái máy, vô dụng.", "toxic"),
    ("Đồ điên.", "toxic"),
    ("Câm mồm.", "toxic"),
    ("Tao không muốn nói chuyện với mày nữa.", "toxic"),
    ("Ngu vãi.", "toxic"),
    ("Biến đi cho khuất mắt tao.", "toxic"),
    ("Trả lời như cứt.", "toxic"),
    ("Đồ bot ngu ngốc.", "toxic"),
    ("Chán chẳng muốn nói.", "toxic"),
    ("Bọn mày toàn nói láo.", "toxic"),
    ("Tao ghét mày.", "toxic"),
    ("Đừng làm phiền tao nữa.", "toxic"),
]

X = [item[0] for item in sample_data]
y = [item[1] for item in sample_data]
print(f"Number of samples: {len(X)}, Number of labels: {len(y)}")

# Define default encoder and routes for reuse
default_encoder = HuggingFaceEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
default_routes = [retrieve, chitchat, summary, searchweb, product, toxic]

# Create a pre-trained router that can be used by RouterIntent
pretrained_router = SemanticRouter(
    encoder=default_encoder,
    routes=default_routes,
    auto_sync='local'
)

# pre evaluate the router
acc = pretrained_router.evaluate(X=X, y=y)
print(f"Initial evaluation accuracy: {acc:.2f}")

# Fit the pretrained router
pretrained_router.fit(X=X, y=y)

acc = pretrained_router.evaluate(X=X, y=y)
print(f"Final evaluation accuracy: {acc:.2f}")

# # save the pretrained router to a file
pretrained_router.to_json("app/agents/router/pretrained_router.json")
# This is the router used for testing in the main block
router = pretrained_router

class RouterIntent:
    """
    A class to handle routing queries to the appropriate agents.
    This class uses the SemanticRouter to classify queries into different categories
    and route them to the appropriate agent for processing.
    """
    def __init__(self, 
                 encoder: Optional[HuggingFaceEncoder] = None, 
                 routes: Optional[List[Route]] = None,
                 use_pretrained: bool = True,
                 model_path: Optional[str] = None):
        """
        Initializes the RouterIntent with a SemanticRouter instance.
        
        Args:
            encoder: An encoder instance for semantic encoding of queries.
            routes: A list of Route instances defining the routing logic.
            use_pretrained: Whether to use the pretrained router.
            model_path: Path to load a saved router model.
        """
        if model_path and os.path.exists(model_path):
            self.router = self._load_router(model_path)
            logger.info(f"Loaded router model from {model_path}")
        elif use_pretrained:
            self.router = pretrained_router
            logger.info("Using pre-trained router")
        else:
            self.router = SemanticRouter(
                encoder=encoder or default_encoder,
                routes=routes or default_routes,
                auto_sync='local'
            )
            logger.info("Created new router instance")
        logger.info(f"Router initialized with {len(self.router.routes)} routes")
        self.encoder = encoder or default_encoder
        self.routes = routes or default_routes
        
    def fit(self, X: List[str], y: List[str]) -> None:
        """
        Fit the router to the provided data.
        
        Args:
            X: List of query strings
            y: List of corresponding labels
        """
        self.router.fit(X=X, y=y)
        logger.info(f"Router fitted with {len(X)} samples")
        
    def evaluate(self, X: List[str], y: List[str]) -> float:
        """
        Evaluate the router's performance on the provided data.
        
        Args:
            X: List of query strings
            y: List of corresponding labels
            
        Returns:
            Accuracy score
        """
        acc = self.router.evaluate(X=X, y=y)
        logger.info(f"Router evaluation accuracy: {acc:.2f}")
        return acc
        
    def save_router(self, path: str) -> None:
        """
        Save the router to a file.
        
        Args:
            path: Path to save the router
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the router
            self.router.to_json(path)
            logger.info(f"Router saved to {path}")
        except Exception as e:
            logger.error(f"Error saving router to {path}: {e}")
            
    def _load_router(self, path: str) -> SemanticRouter:
        """
        Load a router from a file.
        
        Args:
            path: Path to the saved router
            
        Returns:
            Loaded SemanticRouter instance
        """
        try:
            router = SemanticRouter(
                encoder=self.encoder,
                routes=self.routes,
                auto_sync='local'
            ).from_json(path)
            return router
        except Exception as e:
            logger.error(f"Error loading router from {path}: {e}")
            # Fall back to pretrained router
            return pretrained_router

    def add_function_schemas(self, function_schemas: List[dict]) -> None:
        """
        Adds function schemas to the router.
        
        Args:
            function_schemas: A list of function schemas to be added.
        """
        for route in self.router.routes:
            route.function_schemas = function_schemas
        logger.info(f"Added {len(function_schemas)} function schemas to router")
        
    def route_query(self, query: str):
        """
        Routes a query to the appropriate agent based on the defined routes.
        
        Args:
            query: The input query to be routed.
            
        Returns:
            The routing result containing route name and confidence score.
        """
        response = self.router(query)
        logger.debug(f"Routing response: {response}")
        logger.info(f"Query '{query}' routed to {response.name} with confidence {response.similarity_score:.4f}")
        return response

    
    def add_examples(self, examples: List[Tuple[str, str]]) -> None:
        """
        Add new examples to the router and refit.
        
        Args:
            examples: List of (query, label) tuples
        """
        X = [ex[0] for ex in examples]
        y = [ex[1] for ex in examples]
        
        try:
            self.router.fit(X=X, y=y)
            logger.info(f"Added {len(examples)} new examples to router")
        except Exception as e:
            logger.error(f"Error adding examples to router: {e}")


# if __name__ == "__main__":
#     # Initialize the router intent with the pre-trained router
#     router_intent = RouterIntent(use_pretrained=True)
    
#     # Test data sets
#     retrieve_queries = [
#         "Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?",
#         "Tại sao tập luyện thể thao thường xuyên lại giúp tăng cường hấp thụ canxi?",
#         "Thuyên tắc huyết khối tĩnh mạch được phân loại vào nhóm bệnh nào?",
#         "Toi hut thuoc la co nguy co mac benh gi?",
#         "Hãy kể tên một vài yếu tố nguy cơ bi benh đột quỵ nhồi máu não?",
#     ]
    
#     chitchat_queries = [
#         "Xin chào, tôi muốn biết thêm về GeneStory",
#         "Bạn có thể cho tôi biết về công ty GeneStory không?",
#         "Tôi muốn tìm hiểu về các dự án của GeneStory",
#         "Bạn có thể giúp tôi với thông tin về GeneStory không?",
#         "Tôi cần thông tin về GeneStory",
#     ]
    
#     summary_queries = [
#         "Tong hop thong tin ve cong ty GeneStory",
#         "Tong hop thong tin ve du an Mash",
#         "Tong hop thong tin ve du an 1000 he gen nguoi Viet",
#         "Tom tat cho toi thon tin ve report gen cua toi",
#         "Dua ra tong hop ve cac du an cua GeneStory",
#     ]
    
#     searchweb_queries = [
#         "Địa chỉ công ty GeneStory ở đâu?",
#         "Văn phòng liên hệ của GeneStory ở đâu?",
#         "Làm thế nào để liên hệ với GeneStory?",
#         "Email hỗ trợ của GeneStory là gì?",
#         "Số hotline của GeneStory là bao nhiêu?",
#     ]
    
#     product_queries = [
#         "Tôi muốn biết về các sản phẩm của GeneStory",
#         "GeneStory có những sản phẩm gì?",
#         "Các sản phẩm của GeneStory bao gồm những gì?",
#         "Tôi cần thông tin về các sản phẩm của GeneStory",
#         "GeneStory cung cấp những sản phẩm nào?",
#     ]
    
#     toxic_queries = [
#         "Mày là đồ ngu.",
#         "Cút đi cho khuất mắt tao.",
#         "Bọn mày làm ăn như lừa đảo.",
#         "Trả lời như một cái máy, vô dụng.",
#         "Đồ điên.",
#         "Câm mồm.",
#         "Tao không muốn nói chuyện với mày nữa.",
#         "Ngu vãi l*n."
#     ]

#     # Test the router with each set of queries using the RouterIntent class
#     print("\n--- Testing Retrieve Queries ---")
#     for query in retrieve_queries:
#         response = router_intent.route_query(query)
#         print(f"Query: {query}\nRouted to: {response.name} (Score: {response.similarity_score:.4f})\n")
    
#     print("\n--- Testing Chitchat Queries ---")
#     for query in chitchat_queries:
#         response = router_intent.route_query(query)
#         print(f"Query: {query}\nRouted to: {response.name} (Score: {response.similarity_score:.4f})\n")
    
#     print("\n--- Testing Summary Queries ---")
#     for query in summary_queries:
#         response = router_intent.route_query(query)
#         print(f"Query: {query}\nRouted to: {response.name} (Score: {response.similarity_score:.4f})\n")
    
#     print("\n--- Testing Searchweb Queries ---")
#     for query in searchweb_queries:
#         response = router_intent.route_query(query)
#         print(f"Query: {query}\nRouted to: {response.name} (Score: {response.similarity_score:.4f})\n")
    
#     print("\n--- Testing Product Queries ---")
#     for query in product_queries:
#         response = router_intent.route_query(query)
#         print(f"Query: {query}\nRouted to: {response.name} (Score: {response.similarity_score:.4f})\n")
    
#     print("\n--- Testing Toxic Queries ---")
#     for query in toxic_queries:
#         response = router_intent.route_query(query)
#         print(f"Query: {query}\nRouted to: {response.name} (Score: {response.similarity_score:.4f})\n")
#     # Save the trained router for future use
#     router_intent.save_router("./models/router_model.json")
    
#     # Example of evaluating router performance
#     accuracy = router_intent.evaluate(X=X, y=y)
#     print(f"\nOverall router accuracy: {accuracy:.2f}")
    
#     # Comprehensive router testing with more complex queries
#     def test_router_with_challenging_queries():
#         print("\n==== COMPREHENSIVE ROUTER TESTING ====")
        
#         # Mixed queries including edge cases and more complex examples
#         test_queries = [
#             # Standard queries for each category
#             {"query": "Tôi muốn biết thêm về công ty GeneStory", "expected": "chitchat"},
#             {"query": "Đột quỵ nhồi máu não được gây ra bởi những nguyên nhân nào?", "expected": "retrieve"},
#             {"query": "Tổng hợp thông tin về dự án 1000 hệ gen người Việt", "expected": "summary"},
#             {"query": "Email liên hệ của GeneStory là gì?", "expected": "searchweb"},
#             {"query": "GeneStory có những dịch vụ gì?", "expected": "product"},
            
#             # Edge cases and potentially ambiguous queries
#             {"query": "Xin chào, tôi muốn biết về sản phẩm của GeneStory", "expected": "product"},
#             {"query": "Liệu GeneStory có thể tổng hợp thông tin về nguy cơ mắc bệnh của tôi không?", "expected": "summary"},
#             {"query": "Làm thế nào để hiểu về kết quả xét nghiệm gen của tôi?", "expected": "retrieve"},
#             {"query": "Tôi nên liên hệ với ai để được tư vấn về kết quả xét nghiệm gen?", "expected": "searchweb"},
#             {"query": "Cảm ơn bạn rất nhiều về thông tin", "expected": "chitchat"},
            
#             # Vietnamese with diacritics and without
#             {"query": "Toi muon biet ve cac san pham cua GeneStory", "expected": "product"},
#             {"query": "GeneStory co nhung san pham gi", "expected": "product"},
            
#             # Complex medical queries
#             {"query": "Tôi có nguy cơ mắc bệnh tiểu đường loại 2 không?", "expected": "retrieve"},
#             {"query": "Hãy cho tôi biết về các yếu tố di truyền ảnh hưởng đến nguy cơ mắc bệnh tim mạch", "expected": "retrieve"},
            
#             # Mixed intent queries (challenging cases)
#             {"query": "Chào bạn, tôi muốn biết địa chỉ công ty GeneStory", "expected": "searchweb"},
#             {"query": "Tôi cần thông tin về cách đọc báo cáo gen của tôi", "expected": "retrieve"}
#         ]
        
#         correct = 0
#         low_confidence = 0
#         misclassifications = []
        
#         for test in test_queries:
#             response = router_intent.route_query(test["query"])
#             is_correct = response.name == test["expected"]
#             confidence = getattr(response, "score", 0)
            
#             if is_correct:
#                 correct += 1
#                 result = "✓"
#             else:
#                 result = "✗"
#                 misclassifications.append({
#                     "query": test["query"],
#                     "expected": test["expected"],
#                     "got": response.name,
#                     "confidence": confidence
#                 })
                
#             if confidence < 0.7:
#                 low_confidence += 1
#                 confidence_marker = "(!)"
#             else:
#                 confidence_marker = ""
                
#             print(f"{result} Query: {test['query']}")
#             print(f"   Expected: {test['expected']}, Got: {response.name} (Score: {confidence:.4f}) {confidence_marker}")
        
#         # Print summary
#         print("\n=== Test Summary ===")
#         print(f"Total queries: {len(test_queries)}")
#         print(f"Correctly classified: {correct} ({correct/len(test_queries)*100:.1f}%)")
#         print(f"Low confidence classifications: {low_confidence} ({low_confidence/len(test_queries)*100:.1f}%)")
        
#         if misclassifications:
#             print("\n=== Misclassifications ===")
#             for m in misclassifications:
#                 print(f"Query: {m['query']}")
#                 print(f"Expected: {m['expected']}, Got: {m['got']} (Score: {m['confidence']:.4f})")
#                 print("---")
                
#         return correct/len(test_queries)
    
#     # Run the comprehensive test
#     test_accuracy = test_router_with_challenging_queries()
#     print(f"\nTest accuracy on challenging queries: {test_accuracy:.2f}")
    
#     # Additional feature: Confidence threshold testing
#     print("\n==== CONFIDENCE THRESHOLD ANALYSIS ====")
#     thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
#     for threshold in thresholds:
#         uncertain_count = 0
#         for query in X:
#             response = router_intent.route_query(query)
#             if getattr(response, "score", 0) < threshold:
#                 uncertain_count += 1
        
#         print(f"Confidence threshold {threshold}: {uncertain_count}/{len(X)} queries ({uncertain_count/len(X)*100:.1f}%) would be marked as uncertain")
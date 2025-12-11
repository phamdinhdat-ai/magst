#!/usr/bin/env python3
"""
Router Configuration for GeneStory Multi-Agent System
This module contains configuration settings for the routing system.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import os


@dataclass
class RouteConfig:
    """Configuration for individual routes"""
    name: str
    description: str
    utterances: List[str]
    target_agent: str
    confidence_threshold: float = 0.3


# Vietnamese Route Configurations
VIETNAMESE_ROUTES = {
    "retrieve": RouteConfig(
        name="retrieve",
        description="Truy vấn thông tin y tế, gen, và sức khỏe",
        target_agent="CustomerAgent",
        utterances=[
            # Medical conditions and symptoms
            "Đột quỵ nhồi máu não được xếp vào nhóm bệnh nào?",
            "Thuyên tắc huyết khối tĩnh mạch được phân loại vào nhóm bệnh nào?",
            "Tại sao mụn trứng cá thường xuất hiện ở các vùng như mặt, ngực và lưng?",
            "Triệu chứng của bệnh tiểu đường type 2 là gì?",
            "Nguyên nhân gây ra bệnh cao huyết áp là gì?",
            
            # Health risks and prevention
            "Tôi hút thuốc lá có nguy cơ mắc bệnh gì?",
            "Hãy kể tên một vài yếu tố nguy cơ chính gây ra đột quỵ nhồi máu não?",
            "Tại sao tập luyện thể thao thường xuyên lại giúp tăng cường hấp thụ canxi?",
            "Làm thế nào để phòng ngừa bệnh tim mạch?",
            "Chế độ ăn nào tốt cho người bị tiểu đường?",
            
            # Genetic information
            "Báo cáo gen này đã khảo sát bao nhiêu biến thể để đưa ra kết quả?",
            "Gen BRCA1 có ảnh hưởng gì đến nguy cơ ung thư vú?",
            "Làm thế nào để hiểu kết quả xét nghiệm gen?",
            "Xét nghiệm ADN có thể phát hiện những bệnh gì?",
            "Di truyền học có vai trò gì trong việc điều trị bệnh?",
            
            # Cancer and serious diseases
            "Tôi bị ung thư đại trực tràng, tôi cần làm gì?",
            "Các giai đoạn của ung thư phổi là gì?",
            "Phương pháp điều trị ung thư hiện đại nhất là gì?",
            "Hóa trị có tác dụng phụ gì?",
            "Liệu pháp miễn dịch trong điều trị ung thư",
            
            # Drug interactions and pharmacogenomics
            "Các loại thuốc nào tương tác với gen CYP2D6?",
            "Thuốc Warfarin có tương tác với gen nào?",
            "Tại sao cùng một loại thuốc nhưng hiệu quả khác nhau ở mỗi người?",
            "Gen ảnh hưởng đến chuyển hóa thuốc như thế nào?",
            "Xét nghiệm gen trước khi dùng thuốc có cần thiết không?",
            
            # Lifestyle and health
            "Hãy liệt kê một vài đặc điểm của bệnh mụn trứng cá?",
            "Stress có ảnh hưởng đến hệ miễn dịch không?",
            "Ngủ không đủ giấc có gây ra bệnh gì?",
            "Chất lượng không khí ảnh hưởng đến sức khỏe như thế nào?",
            "Vitamin D thiếu hụt có triệu chứng gì?"
        ]
    ),
    
    "chitchat": RouteConfig(
        name="chitchat",
        description="Trò chuyện thông thường và thông tin chung về GeneStory",
        target_agent="GuestAgent",
        utterances=[
            # Greetings and introductions
            "Xin chào, tôi muốn biết thêm về GeneStory",
            "Chào bạn, tôi có thể hỏi về công ty không?",
            "Tôi muốn tìm hiểu về các dự án của GeneStory",
            "Bạn có thể giúp tôi với thông tin về GeneStory không?",
            "Tôi cần thông tin về GeneStory",
            "Hãy kể cho tôi nghe về GeneStory",
            "GeneStory là công ty gì?",
            "Lĩnh vực hoạt động của GeneStory",
            
            # Casual conversation
            "Hôm nay thời tiết thế nào?",
            "Bạn có thể cho tôi biết về thời tiết hôm nay không?",
            "Bây giờ tôi muốn đi ăn tối, bạn có thể gọi cho tôi một nhà hàng ngon không?",
            "Tôi đang tìm kiếm một bộ phim hay để xem, bạn có gợi ý nào không?",
            "Tôi muốn nghe một câu chuyện thú vị, bạn có thể kể cho tôi không?",
            "Cuối tuần này có hoạt động gì thú vị không?",
            "Có sự kiện văn hóa nào đáng chú ý gần đây?",
            
            # Polite expressions
            "Cảm ơn bạn đã giúp đỡ",
            "Tạm biệt và hẹn gặp lại",
            "Xin lỗi vì đã làm phiền",
            "Bạn có khỏe không?",
            "Chúc bạn một ngày tốt lành",
            
            # General inquiries
            "Tôi muốn biết thêm về các hoạt động giải trí trong khu vực",
            "Bạn có thể gợi ý một số hoạt động thú vị để làm trong cuối tuần này?",
            "Làm thế nào để có một lối sống lành mạnh?",
            "Có lời khuyên nào cho người mới bắt đầu công việc?",
            "Cách quản lý thời gian hiệu quả"
        ]
    ),
    
    "summary": RouteConfig(
        name="summary",
        description="Yêu cầu tóm tắt thông tin",
        target_agent="CustomerAgent",
        utterances=[
            # Company and project summaries
            "Tổng hợp thông tin về công ty GeneStory",
            "Tổng hợp thông tin về dự án Mash",
            "Tổng hợp thông tin về dự án 1000 hệ gen người Việt",
            "Đưa ra tổng hợp về các dự án của GeneStory",
            "Tóm tắt lịch sử phát triển của GeneStory",
            "Báo cáo tổng quan về hoạt động của công ty",
            
            # Personal health summaries
            "Tóm tắt cho tôi thông tin về report gen của tôi",
            "Liệt kê các thông tin Genetic của tôi",
            "Tổng quan về kết quả xét nghiệm gen",
            "Tóm tắt lịch sử y tế của tôi",
            "Đưa ra báo cáo tổng hợp về sức khỏe",
            "Tổng kết các chỉ số sức khỏe quan trọng",
            "Báo cáo nguy cơ bệnh tật của tôi",
            
            # Research and findings
            "Tóm tắt các nghiên cứu mới nhất về gen",
            "Tổng hợp thông tin về liệu pháp gen",
            "Báo cáo tiến bộ trong y học cá nhân hóa",
            "Tóm tắt xu hướng phát triển của ngành",
            "Tổng quan về công nghệ xét nghiệm gen",
            
            # Recommendations summary
            "Tóm tắt các khuyến nghị sức khỏe cho tôi",
            "Liệt kê các lời khuyên dựa trên gen của tôi",
            "Đưa ra tổng hợp về chế độ ăn phù hợp",
            "Tóm tắt kế hoạch chăm sóc sức khỏe cá nhân"
        ]
    ),
    
    "searchweb": RouteConfig(
        name="searchweb",
        description="Tìm kiếm thông tin bên ngoài và liên hệ",
        target_agent="GuestAgent",
        utterances=[
            # Contact information
            "Địa chỉ công ty GeneStory ở đâu?",
            "Văn phòng liên hệ của GeneStory ở đâu?",
            "Làm thế nào để liên hệ với GeneStory?",
            "Email hỗ trợ của GeneStory là gì?",
            "Số hotline của GeneStory là bao nhiêu?",
            "Trang web chính thức của GeneStory là gì?",
            "Giờ làm việc của GeneStory",
            "Cách đặt lịch hẹn với GeneStory",
            
            # Drug information
            "Thuốc Paracetamol là gì?",
            "Tác dụng phụ của thuốc Paracetamol là gì?",
            "Thuốc Metformin có tác dụng gì?",
            "Liều dùng thuốc Aspirin cho người lớn",
            "Cách sử dụng thuốc Insulin",
            "Thuốc chống viêm không steroid có những loại nào?",
            
            # Healthcare providers
            "Thông tin về bác sĩ chuyên khoa gen?",
            "Địa chỉ phòng khám di truyền ở Hà Nội?",
            "Bệnh viện nào có chuyên khoa di truyền?",
            "Bác sĩ nào chuyên về y học cá nhân hóa?",
            "Trung tâm xét nghiệm gen uy tín",
            
            # Pricing and services
            "Giá dịch vụ xét nghiệm gen ở đâu rẻ nhất?",
            "Chi phí xét nghiệm ADN là bao nhiêu?",
            "So sánh giá các trung tâm xét nghiệm gen",
            "Bảo hiểm y tế có chi trả cho xét nghiệm gen không?",
            
            # External research
            "Nghiên cứu mới nhất về gen và ung thư",
            "Tin tức về tiến bộ y học cá nhân hóa",
            "Hội nghị khoa học về di truyền học",
            "Báo cáo thị trường xét nghiệm gen"
        ]
    ),
    
    "product_sql": RouteConfig(
        name="product_sql",
        description="Thông tin về sản phẩm và dịch vụ",
        target_agent="ProductAgent",
        utterances=[
            # General product inquiries
            "Tôi muốn biết về các sản phẩm của GeneStory",
            "GeneStory có những sản phẩm gì?",
            "Các sản phẩm của GeneStory bao gồm những gì?",
            "Tôi cần thông tin về các sản phẩm của GeneStory",
            "GeneStory cung cấp những sản phẩm nào?",
            "Tôi muốn tìm hiểu về các sản phẩm của GeneStory",
            "Danh mục sản phẩm của GeneStory",
            "Sản phẩm mới nhất của GeneStory",
            
            # Pricing and packages
            "Giá cả các gói xét nghiệm gen",
            "So sánh các gói dịch vụ của GeneStory",
            "Gói xét nghiệm gen cơ bản có giá bao nhiêu?",
            "Gói xét nghiệm toàn diện có những gì?",
            "Có chương trình khuyến mãi nào không?",
            "Giá xét nghiệm gen cho gia đình",
            
            # Service details
            "Quy trình xét nghiệm gen như thế nào?",
            "Thời gian có kết quả xét nghiệm là bao lâu?",
            "Xét nghiệm gen có đau không?",
            "Cần chuẩn bị gì trước khi xét nghiệm?",
            "Độ chính xác của xét nghiệm gen",
            
            # Purchase and booking
            "Tôi muốn đặt mua sản phẩm xét nghiệm gen",
            "Làm thế nào để đặt lịch xét nghiệm?",
            "Có thể thanh toán bằng thẻ tín dụng không?",
            "Giao hàng tận nhà có không?",
            "Chính sách hoàn trả như thế nào?",
            
            # Specific tests
            "Xét nghiệm nguy cơ ung thư có gì?",
            "Test di truyền về bệnh tim mạch",
            "Xét nghiệm gen về chuyển hóa thuốc",
            "Test DNA về nguồn gốc dân tộc",
            "Xét nghiệm gen trước khi mang thai",
            
            # Comparisons
            "So sánh với các công ty xét nghiệm gen khác",
            "Ưu điểm của sản phẩm GeneStory",
            "Điểm khác biệt của GeneStory",
            "Tại sao nên chọn GeneStory?"
        ]
    )
}

# English Route Configurations (if needed)
ENGLISH_ROUTES = {
    "retrieve": RouteConfig(
        name="retrieve",
        description="Medical, genetic, and health information queries",
        target_agent="CustomerAgent",
        utterances=[
            "What are the symptoms of diabetes type 2?",
            "How does the BRCA1 gene affect breast cancer risk?",
            "What can genetic testing reveal about my health?",
            "How do genes influence drug metabolism?",
            "What are the side effects of chemotherapy?",
        ]
    ),
    
    "chitchat": RouteConfig(
        name="chitchat",
        description="General conversation and company information",
        target_agent="GuestAgent",
        utterances=[
            "Hello, I want to know more about GeneStory",
            "How is the weather today?",
            "Can you tell me about your company?",
            "Thank you for your help",
            "Have a nice day",
        ]
    ),
    
    "summary": RouteConfig(
        name="summary",
        description="Information summarization requests",
        target_agent="CustomerAgent",
        utterances=[
            "Summarize my genetic test results",
            "Give me an overview of GeneStory projects",
            "Provide a health summary based on my DNA",
            "Summarize the latest research findings",
            "Overview of genetic health risks",
        ]
    ),
    
    "searchweb": RouteConfig(
        name="searchweb",
        description="External information search and contact details",
        target_agent="GuestAgent",
        utterances=[
            "What is the address of GeneStory company?",
            "How can I contact GeneStory?",
            "What is Paracetamol medication?",
            "Find genetic counselors near me",
            "Latest news about personalized medicine",
        ]
    ),
    
    "product_sql": RouteConfig(
        name="product_sql",
        description="Product and service information",
        target_agent="ProductAgent",
        utterances=[
            "What products does GeneStory offer?",
            "How much does genetic testing cost?",
            "I want to purchase a DNA test",
            "Compare genetic testing packages",
            "What's included in the comprehensive test?",
        ]
    )
}

# Router System Configuration
ROUTER_SYSTEM_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "confidence_threshold": 0.3,
    "fallback_route": "chitchat",
    "enable_evaluation": True,
    "enable_caching": True,
    "cache_size": 500,
    "auto_sync": "local",
    "language": "vietnamese",  # or "english" or "mixed"
    "enable_logging": True,
    "log_level": "INFO"
}

# Agent mapping for orchestration
ROUTE_TO_AGENT_MAPPING = {
    "retrieve": "CustomerAgent",
    "chitchat": "GuestAgent", 
    "summary": "CustomerAgent",
    "searchweb": "GuestAgent",
    "product_sql": "ProductAgent"
}

# Alternative agent mapping for different user roles
USER_ROLE_AGENT_MAPPING = {
    "customer": {
        "retrieve": "CustomerAgent",
        "chitchat": "CustomerAgent",
        "summary": "CustomerAgent", 
        "searchweb": "CustomerAgent",
        "product_sql": "ProductAgent"
    },
    "employee": {
        "retrieve": "EmployeeAgent",
        "chitchat": "EmployeeAgent",
        "summary": "EmployeeAgent",
        "searchweb": "EmployeeAgent", 
        "product_sql": "EmployeeAgent"
    },
    "guest": {
        "retrieve": "GuestAgent",
        "chitchat": "GuestAgent",
        "summary": "GuestAgent",
        "searchweb": "GuestAgent",
        "product_sql": "GuestAgent"
    }
}

def get_routes_for_language(language: str = "vietnamese") -> Dict[str, RouteConfig]:
    """Get route configurations for specified language"""
    if language.lower() == "english":
        return ENGLISH_ROUTES
    elif language.lower() == "vietnamese":
        return VIETNAMESE_ROUTES
    else:
        # Mix both languages
        mixed_routes = {}
        for route_name in VIETNAMESE_ROUTES:
            vn_route = VIETNAMESE_ROUTES[route_name]
            en_route = ENGLISH_ROUTES.get(route_name)
            
            if en_route:
                # Combine utterances from both languages
                mixed_utterances = vn_route.utterances + en_route.utterances
                mixed_routes[route_name] = RouteConfig(
                    name=vn_route.name,
                    description=f"{vn_route.description} / {en_route.description}",
                    target_agent=vn_route.target_agent,
                    utterances=mixed_utterances,
                    confidence_threshold=vn_route.confidence_threshold
                )
            else:
                mixed_routes[route_name] = vn_route
                
        return mixed_routes

def get_agent_for_route_and_role(route: str, user_role: str = "guest") -> str:
    """Get appropriate agent based on route and user role"""
    role_mapping = USER_ROLE_AGENT_MAPPING.get(user_role, USER_ROLE_AGENT_MAPPING["guest"])
    return role_mapping.get(route, "GuestAgent")


if __name__ == "__main__":
    # Test the configuration
    print("Route Configuration Test")
    print("=" * 50)
    
    # Test Vietnamese routes
    vn_routes = get_routes_for_language("vietnamese")
    print(f"\nVietnamese routes: {len(vn_routes)}")
    for name, config in vn_routes.items():
        print(f"- {name}: {len(config.utterances)} utterances -> {config.target_agent}")
    
    # Test English routes  
    en_routes = get_routes_for_language("english")
    print(f"\nEnglish routes: {len(en_routes)}")
    for name, config in en_routes.items():
        print(f"- {name}: {len(config.utterances)} utterances -> {config.target_agent}")
    
    # Test mixed routes
    mixed_routes = get_routes_for_language("mixed")
    print(f"\nMixed routes: {len(mixed_routes)}")
    for name, config in mixed_routes.items():
        print(f"- {name}: {len(config.utterances)} utterances -> {config.target_agent}")
    
    # Test agent mapping
    print(f"\nAgent mapping test:")
    test_routes = ["retrieve", "chitchat", "product_sql"]
    test_roles = ["customer", "employee", "guest"]
    
    for route in test_routes:
        for role in test_roles:
            agent = get_agent_for_route_and_role(route, role)
            print(f"Route: {route}, Role: {role} -> Agent: {agent}")
    
    print("\n✅ Configuration test completed!")

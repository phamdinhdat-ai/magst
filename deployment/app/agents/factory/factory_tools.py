from typing import Optional, Dict, Any
import chromadb
from langchain_chroma import Chroma
from loguru import logger
import re

import torch
# Import các lớp tool cần thiết
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.tools.search_tool import SearchTool
from app.agents.factory.tools.summary_tool import SummaryTool
from app.agents.retrievers.optimized_product_retriever import OptimizedProductRetrieverTool
from app.agents.retrievers.customer_retriever import CustomerRetrieverTool
from app.agents.retrievers.employee_retriever import EmployeeRetrieverTool
from app.agents.retrievers.genetic_retriever import GeneticRetrieverTool
from app.agents.retrievers.drug_retriever import DrugRetrieverTool
from app.agents.retrievers.company_retriever import CompanyRetrieverTool
from app.agents.retrievers.company_mcp_retriever import CompanyRetrieverMCPClient
from app.agents.retrievers.drug_mcp_retriever import DrugRetrieverMCPClient
from app.agents.retrievers.genetic_mcp_retriever import GeneticRetrieverMCPClient, create_genetic_retriever_client
from app.agents.retrievers.customer_mcp_retriever import CustomerRetrieverMCPClient
from app.agents.retrievers.employee_mcp_retriever import EmployeeRetrieverMCPClient
# from app.agents.retrievers.product_mcp_retriever import EnhancedProductRetrieverMCPClient, create_enhanced_product_retriever_client
from app.agents.retrievers.medical_mcp_retriever import MedicalRetrieverMCPClient, create_medical_retriever_client
from app.agents.retrievers.customer_mcp_retriever import CustomerRetrieverMCPClient, create_customer_retriever_client
from app.agents.retrievers.employee_mcp_retriever import EmployeeRetrieverMCPClient, create_employee_retriever_client
from app.agents.retrievers.medical_mcp_retriever import MedicalRetrieverMCPClient, create_medical_retriever_client
from app.agents.retrievers.product_mcp_retriever import EnhancedProductRetrieverMCPClient, create_enhanced_product_retriever_client
from app.agents.retrievers.drug_mcp_retriever import DrugRetrieverMCPClient, create_drug_retriever_client
from app.agents.retrievers.company_mcp_retriever import CompanyRetrieverMCPClient, create_company_retriever_client 

from app.agents.retrievers.medical_retriever import MedicalRetrieverTool
from app.agents.factory.mcp_retriever_factory import MCPRetrieverFactory
from app.agents.factory.tools.image_analysis_tool import ImageAnalysisTool
from app.agents.factory.tools.get_datetime_tool import GetDateTimeTool
from app.agents.workflow.state import  GraphState as AgentState

from app.agents.workflow.initalize import llm_instance
from langchain_community.embeddings.ollama import OllamaEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from app.core.config import get_settings
from pathlib import Path
settings = get_settings()
WATCH_DIR_COMPANY = "app/agents/retrievers/storages/companies"
WATCH_DIR_PRODUCT = "app/agents/retrievers/storages/products"
WATCH_DIR_GENETIC = "app/agents/retrievers/storages/genetics"
WATCH_DIR_DRUGS = "app/agents/retrievers/storages/drugs"
WATCH_DIR_MEDICAL = "app/agents/retrievers/storages/medical_docs"
class ToolFactory:
    """
    Một nhà máy tập trung để quản lý, khởi tạo và cung cấp các tool.
    Nó giúp giảm sự lặp lại code và làm cho việc thêm/bỏ tool trở nên dễ dàng.
    """
    def __init__(self):
        # Đăng ký các "bản thiết kế" (blueprint) cho việc tạo tool.
        # Chỉ những tool tĩnh, có thể tái sử dụng mới được khởi tạo sẵn.
        self._tool_blueprints = {
            # Tên tool (key) và hàm tạo (lambda)
            "searchweb_tool": lambda: SearchTool(n_results=3),
            "product_retriever_tool": lambda: OptimizedProductRetrieverTool(),
            "drug_retriever_tool": lambda: DrugRetrieverTool(collection_name=settings.DRUGS_DB, watch_directory='app/agents/retrievers/storages/drugs'),
            "genetic_retriever_tool": lambda: GeneticRetrieverTool(collection_name=settings.GENETIC_DB, watch_directory='app/agents/retrievers/storages/genetics'),
            # "customer_retriever_tool": lambda: CustomerRetrieverTool(watch_directory='app/agents/retrievers/storages/customers'),
            # "employee_retriever_tool": lambda: EmployeeRetrieverTool(watch_directory='app/agents/retrievers/storages/employees'),
            "company_retriever_tool": lambda: CompanyRetrieverTool(collection_name=settings.COMPANY_DB, watch_directory='app/agents/retrievers/storages/companies'),
            "medical_retriever_tool": lambda: MedicalRetrieverTool(collection_name=settings.MEDICAL_DB, watch_directory='app/agents/retrievers/storages/medical_docs'),
            "image_analysis_tool": lambda: ImageAnalysisTool(),
            "get_datetime_tool": lambda: GetDateTimeTool(),
            
            # MCP Retrievers - more advanced versions using remote MCP servers
            "company_retriever_mcp_tool": lambda: create_company_retriever_client(
                                                    mcp_server_url=settings.MCP_SERVER_URL,
                                                    watch_directory=WATCH_DIR_COMPANY,
                                                    collection_name=settings.COMPANY_DB
                                                ),
            "drug_retriever_mcp_tool": lambda: create_drug_retriever_client(
                                                    mcp_server_url=settings.MCP_SERVER_URL,
                                                    watch_directory=WATCH_DIR_DRUGS,
                                                    collection_name=settings.DRUGS_DB
                                                ),
            "genetic_retriever_mcp_tool": lambda: create_genetic_retriever_client(
                                                    mcp_server_url=settings.MCP_SERVER_URL,
                                                    watch_directory=WATCH_DIR_GENETIC,
                                                    collection_name=settings.GENETIC_DB
                                                ),
            "product_retriever_mcp_tool": lambda: create_enhanced_product_retriever_client(
                                                    mcp_server_url=settings.MCP_SERVER_URL,
                                                    watch_directory=WATCH_DIR_PRODUCT,
                                                    collection_name=settings.PRODUCTS_DB    
                                                ),
            "medical_retriever_mcp_tool": lambda: create_medical_retriever_client(
                                                    mcp_server_url=settings.MCP_SERVER_URL,
                                                    watch_directory=WATCH_DIR_MEDICAL,
                                                    collection_name=settings.MEDICAL_DB
                                                ),
            "company_retriever_mcp_tool": lambda: create_company_retriever_client(
                                                    mcp_server_url=settings.MCP_SERVER_URL,
                                                    watch_directory=WATCH_DIR_COMPANY,
                                                    collection_name=settings.COMPANY_DB  
                                                ),
            # Các tool động sẽ được tạo theo yêu cầu
        }
        # Cache để lưu trữ các instance của tool tĩnh đã được tạo
        self.tools_cache: Dict[str, BaseAgentTool] = {}
        logger.info("ToolFactory initialized with static tool blueprints.")

    def get_static_tool(self, tool_name: str) -> Optional[BaseAgentTool]:
        """Lấy một instance của tool tĩnh. Tái sử dụng từ cache nếu có."""
        if tool_name in self.tools_cache:
            return self.tools_cache[tool_name]

        if tool_name in self._tool_blueprints:
            logger.info(f"Creating new instance of static tool: '{tool_name}'")
            tool_instance = self._tool_blueprints[tool_name]()
            self.tools_cache[tool_name] = tool_instance
            return tool_instance
            
        logger.warning(f"Static tool '{tool_name}' not found in factory blueprints.")
        return None

    def get_dynamic_tool(self, tool_name: str, state: AgentState) -> Optional[BaseAgentTool]:
        """Tạo và trả về một instance của tool động dựa trên state."""
        logger.info(f"Requesting dynamic tool '{tool_name}' with current state.")
        if tool_name == "customer_retriever_tool":
            customer_id = state.get('customer_id')
            if not customer_id:
                logger.error("Cannot create 'customer_retriever_tool': missing 'customer_id' in state.")
                return None
            # Lưu ý: Tool này không nên được cache trong factory vì nó phụ thuộc vào customer_id
            return CustomerRetrieverTool(customer_id=customer_id, watch_directory=f'app/uploaded_files/documents/customer_{customer_id}')
        
        if tool_name == "customer_retriever_mcp_tool":
            customer_id = state.get('customer_id')
            if not customer_id:
                logger.error("Cannot create 'customer_retriever_mcp_tool': missing 'customer_id' in state.")
                return None
            # MCP version for customer-specific documents
            return   create_customer_retriever_client(
                        mcp_server_url=settings.MCP_SERVER_URL,
                        watch_directory=f'app/uploaded_files/documents/customer_{customer_id}',
                        customer_id=str(customer_id)
                    )
            
            # return MCPRetrieverFactory.create_retriever('customer', customer_id=str(customer_id))
            
        if tool_name == "employee_retriever_tool":
            employee_id = state.get('employee_id')
            
            if not employee_id:
                logger.error("Cannot create 'employee_retriever_tool': missing 'employee_id' in state.")
                return None
            return EmployeeRetrieverTool(employee_id=employee_id, watch_directory=f'app/uploaded_files/documents/employee_{employee_id}')
        
        if tool_name == "employee_retriever_mcp_tool":
            employee_id = state.get('employee_id')
            
            if not employee_id:
                logger.error("Cannot create 'employee_retriever_mcp_tool': missing 'employee_id' in state.")
                return None
            # MCP version for employee-specific documents
            return MCPRetrieverFactory.create_retriever('employee', employee_id=str(employee_id))
        
        if tool_name == "summary_tool":
            # Trả về một instance mới của SummaryTool với LLM hiện tại
            user_id = state.get('customer_id') or state.get('employee_id')
           
            logger.info(f"Creating 'summary_tool' for user_id: {user_id}")
            if user_id is not None:
                user_role = state.get("user_role", "guest").lower()
                safe_user_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', user_id)
                
                collection_name = f"{user_role}_{safe_user_id}_data"
                # Sử dụng PersistentClient để kết nối với vector store
                try:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    
                    logger.info(f"Connecting to persistent vector store: {collection_name}")
                    persistent_client = chromadb.PersistentClient(path=str(Path(settings.VECTOR_STORE_BASE_DIR)))
                    model_kwargs = {'device': device}
                    encode_kwargs = {'normalize_embeddings': True}
                    embeddings = HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDING_MODEL, 
                                               model_kwargs=model_kwargs, 
                                               encode_kwargs=encode_kwargs)
                    logger.info(f"Using embeddings model: {settings.HF_EMBEDDING_MODEL}")
                    vector_store = Chroma(client=persistent_client, collection_name=collection_name, embedding_function=embeddings)
                    logger.info(f"Type of vector store: {type(vector_store)}")
                except Exception as e:
                    logger.error(f"Failed to connect to persistent vector store: {e}")
                    return None    
                return SummaryTool(llm=llm_instance, collection_name=collection_name)

            return None
        logger.warning(f"Dynamic tool '{tool_name}' is not recognized by the factory.")
        return None

    def get_mcp_tool(self, tool_type: str, **kwargs) -> Optional[BaseAgentTool]:
        """
        Get an MCP tool with custom configuration.
        
        Args:
            tool_type: Type of MCP tool ('drug', 'genetic', 'company', 'product', 'customer')
            **kwargs: Custom configuration parameters
            
        Returns:
            MCP tool instance or None if creation fails
        """
        logger.info(f"Creating MCP tool of type '{tool_type}' with custom config")
        return MCPRetrieverFactory.create_retriever(tool_type, **kwargs)
    
    def get_available_mcp_types(self) -> Dict[str, str]:
        """Get available MCP retriever types with descriptions."""
        return MCPRetrieverFactory.get_available_types()
    
    def create_mcp_tool_batch(self, tool_specs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseAgentTool]:
        """
        Create multiple MCP tools at once.
        
        Args:
            tool_specs: Dict of {name: {type: str, **kwargs}} specifications
            
        Returns:
            Dict of {name: tool_instance} for successfully created tools
        """
        return MCPRetrieverFactory.create_batch_retrievers(tool_specs)

# Khởi tạo một instance toàn cục của factory để toàn bộ ứng dụng có thể sử dụng

TOOL_FACTORY = ToolFactory()
from typing import Dict, Any, Optional, Union
from pathlib import Path
from loguru import logger

from app.agents.retrievers.drug_mcp_retriever import DrugRetrieverMCPClient
from app.agents.retrievers.genetic_mcp_retriever import GeneticRetrieverMCPClient
from app.agents.retrievers.customer_mcp_retriever import CustomerRetrieverMCPClient
from app.agents.retrievers.employee_mcp_retriever import EmployeeRetrieverMCPClient
from app.agents.retrievers.product_mcp_retriever import EnhancedProductRetrieverMCPClient
from app.agents.retrievers.company_mcp_retriever import CompanyRetrieverMCPClient
from app.agents.retrievers.medical_mcp_retriever import MedicalRetrieverMCPClient
from app.core.config import get_settings

settings = get_settings()


class MCPRetrieverFactory:
    """
    Centralized factory for creating and managing MCP retriever instances.
    Provides configuration management and standardized creation patterns.
    """
    
    # Default configurations for each retriever type
    _default_configs = {
        'drug': {
            'default_collection': 'drug_knowledge',
            'watch_directory': 'app/agents/retrievers/storages/drugs',
            'description': 'Drug and pharmacological information retriever'
        },
        'genetic': {
            'default_collection': 'genetic_knowledge', 
            'watch_directory': 'app/agents/retrievers/storages/genetics',
            'description': 'Genetic and biomedical information retriever'
        },
        'company': {
            'default_collection': 'company_knowledge',
            'watch_directory': 'app/agents/retrievers/storages/companies', 
            'description': 'Company documents and information retriever'
        },
        'product': {
            'default_collection': 'product_knowledge',
            'watch_directory': 'app/agents/retrievers/storages/products',
            'description': 'Product catalog and information retriever'
        },
        'customer': {
            'default_collection': 'customer_docs',
            'watch_directory': 'app/uploaded_files/documents',
            'description': 'Customer-specific documents retriever'
        },
        'employee': {
            'default_collection': 'employee_docs',
            'watch_directory': 'app/uploaded_files/documents',
            'description': 'Employee-specific documents retriever'
        },
        'medical': {
            'default_collection': 'medical_docs',
            'watch_directory': 'app/agents/retrievers/storages/medical_docs',
            'description': 'Medical and healthcare information retriever'
        }
    }
    
    @classmethod
    def create_retriever(cls, retriever_type: str, **kwargs) -> Optional[Any]:
        """
        Create an MCP retriever instance of the specified type.
        
        Args:
            retriever_type: Type of retriever ('drug', 'genetic', 'company', 'product', 'customer')
            **kwargs: Override parameters for the retriever
            
        Returns:
            Configured MCP retriever instance or None if creation fails
        """
        if retriever_type not in cls._default_configs:
            logger.error(f"Unknown MCP retriever type: {retriever_type}")
            return None
            
        # Get default config and merge with overrides
        config = cls._default_configs[retriever_type].copy()
        config.update(kwargs)
        
        # Ensure MCP server URL is set
        if 'mcp_server_url' not in config:
            config['mcp_server_url'] = settings.MCP_SERVER_URL
            
        try:
            if retriever_type == 'drug':
                return cls._create_drug_retriever(**config)
            elif retriever_type == 'genetic':
                return cls._create_genetic_retriever(**config)
            elif retriever_type == 'company':
                return cls._create_company_retriever(**config)
            elif retriever_type == 'product':
                return cls._create_product_retriever(**config)
            elif retriever_type == 'customer':
                return cls._create_customer_retriever(**config)
            elif retriever_type == 'employee':
                return cls._create_employee_retriever(**config)
            elif retriever_type == 'medical':
                return cls._create_medical_retriever(**config)
            else:
                logger.error(f"No creation method for retriever type: {retriever_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {retriever_type} MCP retriever: {e}")
            return None
    
    @classmethod
    def _create_drug_retriever(cls, **kwargs) -> DrugRetrieverMCPClient:
        """Create a drug MCP retriever."""
        logger.info("Creating DrugRetrieverMCPClient")
        return DrugRetrieverMCPClient(**kwargs)
    
    @classmethod
    def _create_genetic_retriever(cls, **kwargs) -> GeneticRetrieverMCPClient:
        """Create a genetic MCP retriever."""
        logger.info("Creating GeneticRetrieverMCPClient")
        return GeneticRetrieverMCPClient(**kwargs)
    
    @classmethod
    def _create_company_retriever(cls, **kwargs) -> CompanyRetrieverMCPClient:
        """Create a company MCP retriever."""
        logger.info("Creating CompanyRetrieverMCPClient")
        return CompanyRetrieverMCPClient(**kwargs)
    
    @classmethod
    def _create_product_retriever(cls, **kwargs) -> EnhancedProductRetrieverMCPClient:
        """Create a product MCP retriever."""
        logger.info("Creating EnhancedProductRetrieverMCPClient")
        return EnhancedProductRetrieverMCPClient(**kwargs)
    
    @classmethod
    def _create_customer_retriever(cls, **kwargs) -> CustomerRetrieverMCPClient:
        """Create a customer-specific MCP retriever."""
        logger.info("Creating CustomerRetrieverMCPClient")
        
        # Handle customer-specific directory structure
        customer_id = kwargs.get('customer_id')
        if customer_id:
            base_dir = kwargs.get('watch_directory', 'app/uploaded_files/documents')
            kwargs['watch_directory'] = f"{base_dir}/customer_{customer_id}"
            
        return CustomerRetrieverMCPClient(**kwargs)
    
    @classmethod
    def _create_employee_retriever(cls, **kwargs) -> EmployeeRetrieverMCPClient:
        """Create an employee-specific MCP retriever."""
        logger.info("Creating EmployeeRetrieverMCPClient")
        
        # Handle employee-specific directory structure
        employee_id = kwargs.get('employee_id')
        if employee_id:
            base_dir = kwargs.get('watch_directory', 'app/uploaded_files/documents')
            kwargs['watch_directory'] = f"{base_dir}/employee_{employee_id}"
            
        return EmployeeRetrieverMCPClient(**kwargs)
    
    @classmethod
    def _create_medical_retriever(cls, **kwargs) -> MedicalRetrieverMCPClient:
        """Create a medical MCP retriever."""
        logger.info("Creating MedicalRetrieverMCPClient")
        return MedicalRetrieverMCPClient(**kwargs)
    
    @classmethod
    def get_available_types(cls) -> Dict[str, str]:
        """Get available retriever types with descriptions."""
        return {
            ret_type: config['description'] 
            for ret_type, config in cls._default_configs.items()
        }
    
    @classmethod
    def get_default_config(cls, retriever_type: str) -> Optional[Dict[str, Any]]:
        """Get default configuration for a retriever type."""
        return cls._default_configs.get(retriever_type)
    
    @classmethod
    def create_batch_retrievers(cls, retriever_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create multiple retrievers at once.
        
        Args:
            retriever_specs: Dict of {name: {type: str, **kwargs}} specifications
            
        Returns:
            Dict of {name: retriever_instance} for successfully created retrievers
        """
        retrievers = {}
        
        for name, spec in retriever_specs.items():
            if 'type' not in spec:
                logger.error(f"Missing 'type' in spec for retriever '{name}'")
                continue
                
            retriever_type = spec.pop('type')
            retriever = cls.create_retriever(retriever_type, **spec)
            
            if retriever:
                retrievers[name] = retriever
                logger.info(f"Created MCP retriever '{name}' of type '{retriever_type}'")
            else:
                logger.error(f"Failed to create MCP retriever '{name}' of type '{retriever_type}'")
        
        return retrievers


class MCPRetrieverManager:
    """
    Manager class for handling lifecycle of MCP retrievers.
    """
    
    def __init__(self):
        self._active_retrievers: Dict[str, Any] = {}
        self._factory = MCPRetrieverFactory()
    
    def create_and_register(self, name: str, retriever_type: str, **kwargs) -> Optional[Any]:
        """Create and register an MCP retriever."""
        retriever = self._factory.create_retriever(retriever_type, **kwargs)
        if retriever:
            self._active_retrievers[name] = retriever
            logger.info(f"Registered MCP retriever '{name}' of type '{retriever_type}'")
        return retriever
    
    def get_retriever(self, name: str) -> Optional[Any]:
        """Get a registered retriever by name."""
        return self._active_retrievers.get(name)
    
    def list_retrievers(self) -> Dict[str, str]:
        """List all registered retrievers with their types."""
        return {
            name: type(retriever).__name__ 
            for name, retriever in self._active_retrievers.items()
        }
    
    def cleanup_retriever(self, name: str) -> bool:
        """Cleanup and remove a specific retriever."""
        if name in self._active_retrievers:
            retriever = self._active_retrievers[name]
            try:
                if hasattr(retriever, 'cleanup'):
                    retriever.cleanup()
                del self._active_retrievers[name]
                logger.info(f"Cleaned up MCP retriever '{name}'")
                return True
            except Exception as e:
                logger.error(f"Error cleaning up MCP retriever '{name}': {e}")
                return False
        return False
    
    def cleanup_all(self):
        """Cleanup all registered retrievers."""
        for name in list(self._active_retrievers.keys()):
            self.cleanup_retriever(name)


# Global manager instance
MCP_RETRIEVER_MANAGER = MCPRetrieverManager()


# Convenience functions for direct usage
def create_drug_mcp_retriever(**kwargs) -> Optional[DrugRetrieverMCPClient]:
    """Create a drug MCP retriever with default settings."""
    return MCPRetrieverFactory.create_retriever('drug', **kwargs)


def create_genetic_mcp_retriever(**kwargs) -> Optional[GeneticRetrieverMCPClient]:
    """Create a genetic MCP retriever with default settings."""
    return MCPRetrieverFactory.create_retriever('genetic', **kwargs)


def create_company_mcp_retriever(**kwargs) -> Optional[CompanyRetrieverMCPClient]:
    """Create a company MCP retriever with default settings."""
    return MCPRetrieverFactory.create_retriever('company', **kwargs)


def create_product_mcp_retriever(**kwargs) -> Optional[EnhancedProductRetrieverMCPClient]:
    """Create a product MCP retriever with default settings."""
    return MCPRetrieverFactory.create_retriever('product', **kwargs)


def create_customer_mcp_retriever(customer_id: str, **kwargs) -> Optional[CustomerRetrieverMCPClient]:
    """Create a customer-specific MCP retriever."""
    return MCPRetrieverFactory.create_retriever('customer', customer_id=customer_id, **kwargs)


def create_employee_mcp_retriever(employee_id: str, **kwargs) -> Optional[EmployeeRetrieverMCPClient]:
    """Create an employee-specific MCP retriever."""
    return MCPRetrieverFactory.create_retriever('employee', employee_id=employee_id, **kwargs)


def create_medical_mcp_retriever(**kwargs) -> Optional[MedicalRetrieverMCPClient]:
    """Create a medical MCP retriever with default settings."""
    return MCPRetrieverFactory.create_retriever('medical', **kwargs)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_factory():
        # Test creating different types of retrievers
        drug_retriever = create_drug_mcp_retriever()
        genetic_retriever = create_genetic_mcp_retriever()
        company_retriever = create_company_mcp_retriever()
        
        print("Available retriever types:")
        for ret_type, description in MCPRetrieverFactory.get_available_types().items():
            print(f"  {ret_type}: {description}")
        
        # Test batch creation
        batch_specs = {
            'main_drug': {'type': 'drug'},
            'main_genetic': {'type': 'genetic'},
            'test_customer': {'type': 'customer', 'customer_id': '123'}
        }
        
        batch_retrievers = MCPRetrieverFactory.create_batch_retrievers(batch_specs)
        print(f"Created {len(batch_retrievers)} retrievers in batch")
        
        # Test manager
        manager = MCPRetrieverManager()
        manager.create_and_register('test_drug', 'drug')
        print(f"Active retrievers: {manager.list_retrievers()}")
        
        manager.cleanup_all()
    
    # asyncio.run(test_factory())

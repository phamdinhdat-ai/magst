# MCP Retrievers Documentation

## Overview

The MCP (Model Context Protocol) Retrievers are advanced document retrieval tools that provide intelligent, context-aware search capabilities across various specialized domains. These tools connect to remote MCP servers for high-performance vector database operations while maintaining local file watching and processing capabilities.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Available MCP Retrievers](#available-mcp-retrievers)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Architecture Overview

### Core Components

1. **MCP Client Tools**: Individual retriever clients for different domains
2. **MCP Retriever Factory**: Centralized creation and management system
3. **Document Watchers**: Real-time file system monitoring
4. **Vector Database Integration**: Remote MCP server connectivity
5. **Document Processing Pipeline**: Intelligent text chunking and preprocessing

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   MCP Factory   │    │   MCP Server    │
│     Agents      │◄──►│     Manager     │◄──►│  (Remote DB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local File    │    │   Document      │    │   Vector Store  │
│   Watchers      │    │   Processing    │    │   Collections   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Available MCP Retrievers

### 1. Drug MCP Retriever (`drug_mcp_retriever.py`)

**Purpose**: Pharmacological and drug information retrieval with enhanced medical safety.

**Key Features**:
- Drug interaction analysis
- Dosage and safety information
- Pharmacokinetic data
- Side effect profiles
- Vietnamese pharmaceutical database support

**Collection**: `drug_knowledge`
**Watch Directory**: `app/agents/retrievers/storages/drugs`

**Specialized Capabilities**:
- Medical terminology processing
- Drug name standardization
- Safety contraindication alerts
- Multi-language support (EN/VI)

### 2. Genetic MCP Retriever (`genetic_mcp_retriever.py`)

**Purpose**: Genetic and genomic information retrieval for personalized medicine.

**Key Features**:
- Gene variant analysis
- Genomic annotation lookup
- Hereditary disease information
- Pharmacogenomics data
- Clinical genetics protocols

**Collection**: `genetic_knowledge`
**Watch Directory**: `app/agents/retrievers/storages/genetics`

**Specialized Capabilities**:
- HGVS notation processing
- RefSeq and Ensembl ID mapping
- Population frequency data
- Clinical significance scoring

### 3. Medical MCP Retriever (`medical_mcp_retriever.py`)

**Purpose**: General medical and healthcare information with clinical decision support.

**Key Features**:
- Disease information and symptoms
- Treatment protocols and guidelines
- Medical case studies
- Healthcare best practices
- Vietnamese medical terminology

**Collection**: `medical_docs`
**Watch Directory**: `app/agents/retrievers/storages/medical_docs`

**Specialized Capabilities**:
- ICD-10 classification support
- Medical imaging integration
- Clinical decision trees
- Evidence-based medicine rankings

### 4. Product MCP Retriever (`product_mcp_retriever.py`)

**Purpose**: Enhanced product information with intelligent query analysis.

**Key Features**:
- Product specifications and catalogs
- Price and availability information
- Feature comparison matrices
- Customer reviews and ratings
- Multi-intent query processing

**Collection**: `product_knowledge`
**Watch Directory**: `app/agents/retrievers/storages/products`

**Advanced Features**:
- **Query Intent Analysis**: Automatically classifies queries into types:
  - Simple search: "thông tin về genemap adult"
  - Exclusion queries: "các gói khác ngoài genemap adult"
  - Comparison: "so sánh genemap adult và kid"
  - Listing: "tất cả các gói dịch vụ"
  - Price queries: "giá của genemap adult"
- **Database Synchronization**: Real-time sync with product database
- **Performance Caching**: High-speed query result caching
- **Relevance Scoring**: Advanced scoring algorithm for result ranking

### 5. Company MCP Retriever (`company_mcp_retriever.py`)

**Purpose**: Corporate information and policy document management.

**Key Features**:
- Company policies and procedures
- Organizational structure
- Contact information
- Service offerings
- Compliance documentation

**Collection**: `company_knowledge`
**Watch Directory**: `app/agents/retrievers/storages/companies`

**Specialized Capabilities**:
- Multi-format document support (PDF, DOC, JSON)
- Hierarchical information organization
- Access control integration
- Version tracking for policy updates

### 6. Customer MCP Retriever (`customer_mcp_retriever.py`)

**Purpose**: Customer-specific document management with privacy isolation.

**Key Features**:
- Customer-specific document isolation
- Personal health records (PHRs)
- Service history and interactions
- Privacy-compliant data handling
- Dynamic collection management

**Collection**: `customer_{customer_id}_data`
**Watch Directory**: `app/uploaded_files/documents/customer_{id}`

**Privacy & Security**:
- **Data Isolation**: Each customer has separate vector collections
- **Access Control**: Customer ID-based access restrictions
- **GDPR Compliance**: Automatic data cleanup and right-to-be-forgotten
- **Encryption**: End-to-end encrypted document processing

### 7. Employee MCP Retriever (`employee_mcp_retriever.py`)

**Purpose**: Employee-specific information and HR document management.

**Key Features**:
- Employee records and documentation
- Training materials and certifications
- Performance evaluations
- Internal communications
- Role-specific access controls

**Collection**: `employee_{employee_id}_data`
**Watch Directory**: `app/uploaded_files/documents/employee_{id}`

**HR Integration**:
- **Role-Based Access**: Department and position-based filtering
- **Training Tracking**: Certification and skill development monitoring
- **Performance Analytics**: Document-based performance insights
- **Compliance Monitoring**: Regulatory requirement tracking

## Installation & Setup

### Prerequisites

```bash
# Install required dependencies
pip install -r requirements.txt

# Core dependencies
pip install langchain-community
pip install chromadb
pip install watchdog
pip install pydantic
pip install loguru
```

### MCP Server Setup

1. **Configure MCP Server URL**:
```python
# In app/core/config.py
MCP_SERVER_URL = "http://localhost:50051/sse"
```

2. **Initialize MCP Server** (if running locally):
```bash
# Start MCP server instance
python -m mcp.server --port 50051
```

### Directory Structure Setup

```bash
# Create required directories
mkdir -p app/agents/retrievers/storages/{drugs,genetics,companies,products,medical_docs}
mkdir -p app/uploaded_files/documents
mkdir -p vector_stores_data
```

## Configuration

### Environment Variables

```env
# MCP Server Configuration
MCP_SERVER_URL=http://localhost:50051/sse
MCP_TIMEOUT=30
MCP_MAX_RETRIES=3

# Document Processing
CHUNK_SIZE=1000
OVERLAP_SIZE=100
MAX_WORKERS=4

# Vector Database
VECTOR_STORE_PATH=./vector_stores_data
EMBEDDING_MODEL=ollama/nomic-embed-text

# Performance
ENABLE_CACHE=true
CACHE_TTL=300
MAX_CACHE_SIZE=1000
```

### Factory Configuration

```python
from app.agents.factory.mcp_retriever_factory import MCPRetrieverFactory

# Default configurations are pre-set for each retriever type
# Override as needed:
drug_retriever = MCPRetrieverFactory.create_retriever(
    'drug',
    mcp_server_url="custom://server",
    watch_directory="/custom/path",
    default_collection="custom_drugs"
)
```

## Usage Examples

### Basic Retrieval

```python
from app.agents.factory.mcp_retriever_factory import MCPRetrieverFactory

# Create a drug retriever
drug_tool = MCPRetrieverFactory.create_retriever('drug')

# Perform search
result = await drug_tool._arun(
    query="What are the side effects of metformin?",
    collection_name="drug_knowledge",
    max_results=5
)
print(result)
```

### Advanced Product Search

```python
# Create enhanced product retriever
product_tool = MCPRetrieverFactory.create_retriever(
    'product',
    enable_database_sync=True,
    enable_cache=True
)

# Complex query with intent analysis
result = await product_tool._arun(
    query="So sánh GenAI Adult và GenAI Kids về giá cả và tính năng",
    max_results=10
)
```

### Customer-Specific Retrieval

```python
# Create customer-specific retriever
customer_tool = MCPRetrieverFactory.create_retriever(
    'customer',
    customer_id="customer_123"
)

# Search customer documents
result = await customer_tool._arun(
    query="Previous test results for genetic screening",
    max_results=3
)
```

### Document Upload

```python
# Upload new documents
await drug_tool.upload_file(
    file_path="/path/to/new_drug_info.pdf",
    collection_name="drug_knowledge"
)

# Upload text documents
await drug_tool.upload_text_documents(
    texts=["New drug information...", "Additional details..."],
    source_name="manual_update_2024"
)
```

### Batch Operations

```python
# Upload multiple files
import os
from pathlib import Path

watch_dir = Path("app/agents/retrievers/storages/drugs")
for file_path in watch_dir.glob("*.pdf"):
    await drug_tool.upload_file(str(file_path))
```

## Advanced Features

### 1. Real-Time Document Watching

All MCP retrievers include automatic file system monitoring:

```python
# Documents are automatically processed when:
# - New files are added to watch directories
# - Existing files are modified
# - Files are deleted (removed from index)

# Watch directory structure:
app/agents/retrievers/storages/
├── drugs/          # Pharmaceutical documents
├── genetics/       # Genomic data files
├── companies/      # Corporate information
├── products/       # Product specifications
└── medical_docs/   # Medical literature
```

### 2. Intelligent Query Processing

**Product Retriever** includes advanced query analysis:

```python
# Automatic query intent detection:
queries = [
    "thông tin về genemap adult",      # → SIMPLE_SEARCH
    "các gói khác ngoài genemap",      # → EXCLUSION  
    "so sánh genemap adult và kid",    # → COMPARISON
    "tất cả các gói dịch vụ",         # → LISTING
    "giá của genemap adult"            # → PRICE_QUERY
]

# Each intent triggers specialized search strategies
```

### 3. Enhanced Relevance Scoring

```python
class RetrievedDocument(BaseModel):
    def calculate_advanced_relevance(self, query: str) -> float:
        # Multi-factor scoring:
        # - Exact phrase matching (30%)
        # - Drug/entity name matching (25%) 
        # - Term overlap with position weighting (20%)
        # - Retrieval score (15%)
        # - Content length penalty (5%)
        # - Source reliability (5%)
        pass
```

### 4. Performance Optimization

```python
# Built-in caching system
cache = PerformanceCache(max_size=1000, ttl_seconds=300)

# Parallel document processing
thread_pool = ThreadPoolExecutor(max_workers=4)

# Optimized text splitting for different domains
medical_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Larger chunks for medical content
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)
```

### 5. Error Handling & Reliability

```python
# Automatic retry mechanisms
@retry(max_attempts=3, backoff_factor=1.5)
async def _retrieve_from_server(self, query: str, collection: str):
    # Robust error handling and connection recovery
    pass

# Graceful degradation
if not await self._test_server_connection():
    logger.warning("MCP server unavailable, using local fallback")
    return await self._local_fallback_search(query)
```

## Performance Optimization

### 1. Caching Strategy

```python
# Query result caching
cache_key = f"{query}:{collection_name}:{max_results}"
cached_result = self._query_cache.get(cache_key)
if cached_result:
    return cached_result

# Document hash caching for deduplication
def get_content_hash(self) -> str:
    if self._content_hash is None:
        self._content_hash = hashlib.md5(self.content.encode()).hexdigest()
    return self._content_hash
```

### 2. Batch Processing

```python
# Process multiple files concurrently
async def _scan_and_process_all_files(self):
    tasks = []
    for file_path in self.watch_directory.rglob("*"):
        if file_path.is_file():
            tasks.append(self._process_file_if_needed(file_path))
    
    await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Memory Management

```python
# Efficient document processing
def get_processed_tokens(self) -> List[str]:
    if self._processed_tokens is None:
        self._processed_tokens = re.findall(r'\b\w+\b', self.content.lower())
    return self._processed_tokens

# Cleanup resources
def cleanup(self):
    if self._observer:
        self._observer.stop()
        self._observer.join()
    if self._thread_pool:
        self._thread_pool.shutdown(wait=True)
```

## Troubleshooting

### Common Issues

#### 1. MCP Server Connection Issues

```python
# Check server connectivity
async def _test_server_connection(self) -> bool:
    try:
        async with sse_client(self.mcp_server_url) as (read, write):
            await write({"method": "ping"})
            response = await asyncio.wait_for(read(), timeout=5.0)
            return response.get("status") == "ok"
    except Exception as e:
        logger.error(f"MCP server connection failed: {e}")
        return False
```

**Solutions**:
- Verify MCP server URL is correct
- Check firewall and network connectivity
- Ensure MCP server is running and accessible
- Verify SSL/TLS configuration if using HTTPS

#### 2. Document Processing Failures

```python
# Common file processing issues
def _load_and_split_file(self, file_path: Path) -> List[Document]:
    try:
        # Check file size limits
        if file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB limit
            logger.warning(f"File too large: {file_path}")
            return []
        
        # Verify file extension support
        if file_path.suffix.lower() not in ['.pdf', '.txt', '.md', '.json', '.csv']:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
            
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []
```

**Solutions**:
- Check file permissions and accessibility
- Verify supported file formats
- Monitor disk space and memory usage
- Review file encoding (UTF-8 recommended)

#### 3. Performance Issues

**Symptoms**: Slow retrieval, high memory usage, timeouts

**Diagnostic Commands**:
```python
# Check retriever status
status = await retriever.get_status()
print(f"Documents processed: {status['document_count']}")
print(f"Cache hit rate: {status['cache_stats']}")
print(f"Average query time: {status['performance_metrics']}")
```

**Solutions**:
- Increase cache size for frequently accessed data
- Optimize chunk size for your content type
- Use parallel processing for large document sets
- Monitor and tune MCP server resources

#### 4. Memory Leaks

```python
# Proper cleanup in destructor
def __del__(self):
    try:
        self.cleanup()
    except Exception:
        pass  # Ignore cleanup errors in destructor
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger('app.agents.retrievers').setLevel(logging.DEBUG)

# Enable MCP client debugging
import os
os.environ['MCP_DEBUG'] = '1'
```

### Health Checks

```python
async def health_check():
    """Comprehensive health check for MCP retrievers."""
    checks = {}
    
    # Test each retriever type
    for retriever_type in ['drug', 'genetic', 'medical', 'product', 'company']:
        try:
            retriever = MCPRetrieverFactory.create_retriever(retriever_type)
            status = await retriever.get_status()
            checks[retriever_type] = {
                'status': 'healthy' if status['server_connected'] else 'degraded',
                'details': status
            }
        except Exception as e:
            checks[retriever_type] = {
                'status': 'error',
                'error': str(e)
            }
    
    return checks
```

## API Reference

### MCPRetrieverFactory

```python
class MCPRetrieverFactory:
    @classmethod
    def create_retriever(cls, retriever_type: str, **kwargs) -> Optional[Any]:
        """
        Create an MCP retriever instance.
        
        Args:
            retriever_type: One of 'drug', 'genetic', 'medical', 'product', 
                          'company', 'customer', 'employee'
            **kwargs: Configuration overrides
            
        Returns:
            Configured MCP retriever instance
        """
```

### Base MCP Retriever Methods

```python
class BaseMCPRetriever:
    async def _arun(self, query: str, collection_name: str = None, 
                   max_results: int = 5) -> str:
        """Async retrieval method."""
        
    def _run(self, query: str, collection_name: str = None, 
            max_results: int = 5) -> str:
        """Sync retrieval method."""
        
    async def upload_file(self, file_path: str, 
                         collection_name: str = None) -> str:
        """Upload a file to the vector database."""
        
    async def upload_text_documents(self, texts: List[str], 
                                   collection_name: str = None,
                                   source_name: str = "manual_upload") -> str:
        """Upload text documents."""
        
    async def get_status(self) -> Dict[str, Any]:
        """Get retriever status and health information."""
        
    def cleanup(self):
        """Clean up resources and stop background processes."""
```

### RetrievedDocument Model

```python
class RetrievedDocument(BaseModel):
    content: str
    source: str
    retrieval_score: float
    relevance_score: float = 0.0
    drug_name: str = ""
    category: str = ""
    
    def calculate_advanced_relevance(self, query: str) -> float:
        """Calculate advanced relevance score."""
        
    def get_content_hash(self) -> str:
        """Get content hash for deduplication."""
        
    def calculate_fast_relevance(self, query_tokens: List[str]) -> float:
        """Fast relevance calculation using pre-processed tokens."""
```

## Best Practices

### 1. Retriever Selection

- **Drug Retriever**: Pharmaceutical queries, drug interactions, safety information
- **Genetic Retriever**: Genomic analysis, variant interpretation, hereditary conditions  
- **Medical Retriever**: General medical questions, symptoms, treatment protocols
- **Product Retriever**: Product comparisons, specifications, pricing information
- **Company Retriever**: Corporate policies, contact information, service descriptions
- **Customer Retriever**: Personal health records, service history, private documents
- **Employee Retriever**: HR documents, training materials, internal communications

### 2. Query Optimization

```python
# Good queries are specific and contextual
good_queries = [
    "Side effects of metformin in elderly patients",
    "BRCA1 variants associated with breast cancer risk",
    "GenAI Adult package pricing and features",
    "Company privacy policy for genetic data"
]

# Avoid overly broad or vague queries
avoid_queries = [
    "medicine",
    "genes", 
    "products",
    "information"
]
```

### 3. Document Organization

```
# Recommended directory structure
app/agents/retrievers/storages/
├── drugs/
│   ├── fda_approvals/
│   ├── drug_interactions/
│   └── vietnamese_drugs/
├── genetics/
│   ├── clinical_variants/
│   ├── population_data/
│   └── gene_annotations/
└── medical_docs/
    ├── guidelines/
    ├── research_papers/
    └── clinical_protocols/
```

### 4. Error Handling

```python
try:
    result = await retriever._arun(query)
except ConnectionError:
    # Handle MCP server connectivity issues
    result = await fallback_search(query)
except TimeoutError:
    # Handle slow queries
    result = await quick_search(query, max_results=3)
except Exception as e:
    logger.error(f"Retrieval failed: {e}")
    result = "I apologize, but I'm unable to retrieve that information right now."
```

### 5. Resource Management

```python
# Always clean up resources
async def use_retriever():
    retriever = MCPRetrieverFactory.create_retriever('drug')
    try:
        result = await retriever._arun("query")
        return result
    finally:
        retriever.cleanup()

# Or use context managers (if implemented)
async with MCPRetrieverFactory.create_retriever('drug') as retriever:
    result = await retriever._arun("query")
```

## Security Considerations

### 1. Data Privacy

- Customer and employee retrievers implement strict data isolation
- Each customer/employee has separate vector collections
- Access controls based on authentication and authorization
- Automatic cleanup of expired or deleted user data

### 2. Network Security

- Use HTTPS for MCP server connections in production
- Implement proper authentication for MCP server access
- Network segmentation for sensitive data processing
- Regular security audits of retriever components

### 3. Document Security

- Sanitize uploaded documents for malicious content
- Implement file type validation and size limits
- Use encrypted storage for sensitive documents
- Audit trail for document access and modifications

## Future Enhancements

### Planned Features

1. **Multi-Modal Support**: Image and audio document processing
2. **Real-Time Sync**: Bi-directional synchronization with external systems
3. **Advanced Analytics**: Query pattern analysis and optimization suggestions
4. **Federated Search**: Cross-retriever query capabilities
5. **Machine Learning**: Adaptive relevance scoring and query understanding
6. **API Gateway**: RESTful API for external system integration

### Roadmap

- **Q1 2025**: Multi-modal document support, enhanced caching
- **Q2 2025**: Federated search and cross-retriever queries  
- **Q3 2025**: ML-powered relevance scoring and query optimization
- **Q4 2025**: Full API gateway and external system integration

---

## Support & Contributing

### Getting Help

- **Documentation**: This README and inline code documentation
- **Logs**: Check application logs for detailed error information
- **Health Checks**: Use built-in status endpoints for diagnostics
- **Community**: Contribute to the project on GitHub

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

### License

This project is licensed under the MIT License. See LICENSE file for details.

---

*Last updated: September 2025*
*Version: 2.0.0*

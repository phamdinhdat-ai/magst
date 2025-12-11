# GuestWorkflow: Secure, Optimized Guest User Workflow

## Overview
The `GuestWorkflow` is a robust, secure, and privacy-focused workflow designed for anonymous guest users of the platform. It leverages a modular, agent-based architecture and a LangGraph-powered state machine to provide intelligent, adaptive responses to guest queries, while strictly sandboxing access to only public-facing information.

---

## Key Features
- **Triage and Guardrails:** Uses a TriageGuardrailAgent to classify, route, and apply safety checks (e.g., toxicity filtering) to all guest queries.
- **Specialist Agents:** Supports a curated set of specialist agents (Company, Product, Medical, Drug, Genetic, DirectAnswer) for public information retrieval.
- **MCP Integration:** Advanced Model Context Protocol support for efficient, standardized data retrieval from public knowledge bases.
- **Adaptive Routing:** Dynamically routes queries based on complexity, required analysis, and safety checks.
- **Streaming & Synthesis:** Supports streaming responses and multi-agent synthesis for complex queries.
- **No Personalization:** Strictly prohibits access to customer or employee data, ensuring guest privacy.
- **Factory Tool Pattern:** Utilizes a centralized factory for tool and retriever instantiation, supporting both static and dynamic tool creation.
- **Enhanced Security:** Multiple layers of security ensure guests can only access public information.

---

## Workflow Graph Structure

### Execution Paths

The GuestWorkflow supports multiple execution paths optimized for different query types:

#### 1. Direct Answer Path (Fast Path)
- **Purpose**: Simple queries, greetings, general knowledge
- **Flow**: `Triage → Direct Answer → Final Answer → Question Generator → END`
- **Examples**: "Hello", "What time is it?", "Thank you"
- **Agent**: `DirectAnswerAgent` (NaiveAgent)

#### 2. Specialist Path (Standard Path)
- **Purpose**: Domain-specific queries requiring public information
- **Flow**: `Triage → Specialist Agent → Verification → Final Answer → Question Generator → END`
- **Examples**: "What products do you sell?", "Tell me about aspirin side effects"
- **Agents**: `CompanyAgent`, `ProductAgent`, `MedicalAgent`, `DrugAgent`, `GeneticAgent`

#### 3. Multi-Agent Path (Complex Queries)
- **Purpose**: Queries requiring coordination between multiple domains
- **Flow**: `Triage → Plan Executor → Synthesizer → Final Answer → Question Generator → END`
- **Examples**: "Compare your genetic testing products with their medical applications"
- **Agents**: Multiple specialists coordinated by `SynthesizerAgent`

#### 4. Fallback Path (Error Recovery)
- **Purpose**: Error recovery when other paths fail
- **Flow**: `Triage/Specialist/Verification → Fallback Agent → Final Answer → Question Generator → END`
- **Trigger**: Verification failures, toxic content, agent errors
- **Agent**: `FallbackAgent` (NaiveAgent)

### Detailed Graph Flow

```
┌─────────────┐
│   START     │
└─────┬───────┘
      │
      v
┌─────────────┐
│ triage_node │ ← TriageGuardrailAgent
└─────┬───────┘   (Classification, Safety, Routing)
      │
      v
┌─────────────────────────────────────────────────────┐
│              _route_from_triage                     │
│  ┌─────────────────────────────────────────────────┐│
│  │ Decision Logic:                                 ││
│  │ • is_toxic? → END                              ││
│  │ • next_step = "direct_answer" → direct_answer   ││
│  │ • next_step = "specialist_agent" → specialist   ││
│  │ • next_step = "multi_agent_plan" → plan_exec   ││
│  │ • next_step = "need_analysis" → fallback       ││
│  │ • Default fallback → END                       ││
│  └─────────────────────────────────────────────────┘│
└─────┬────────┬─────────────┬─────────────┬─────────┘
      │        │             │             │
      v        v             v             v
┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────┐
│direct_  │ │specialist_  │ │plan_        │ │ END │
│answer   │ │agent        │ │executor     │ └─────┘
└─────┬───┘ └─────┬───────┘ └─────┬───────┘
      │           │               │
      │           v               │
      │     ┌─────────────┐       │
      │     │verification_│       │
      │     │node         │       │
      │     └─────┬───────┘       │
      │           │               │
      │           v               │
      │     ┌─────────────────────┐│
      │     │_route_after_        ││
      │     │verification         ││
      │     │ • is_final_answer?  ││
      │     │   Yes → final_answer││
      │     │   No → fallback     ││
      │     └─────┬─────────┬─────┘│
      │           │         │      │
      │           v         v      │
      │     ┌─────────┐ ┌─────────┐│
      │     │fallback_│ │         ││
      │     │agent    │ │         ││
      │     └─────┬───┘ │         ││
      │           │     │         ││
      └───────────┼─────┘         │
                  │               │
                  v               v
            ┌─────────────┐ ┌─────────────┐
            │final_answer │ │synthesizer_ │
            │             │ │node         │
            └─────┬───────┘ └─────┬───────┘
                  │               │
                  │               v
                  │         ┌─────────────┐
                  │         │final_answer │
                  │         └─────┬───────┘
                  │               │
                  └───────────────┼───────┐
                                  │       │
                                  v       v
                            ┌─────────────┐
                            │question_    │
                            │generator    │
                            └─────┬───────┘
                                  │
                                  v
                                ┌─────┐
                                │ END │
                                └─────┘
```

### Node Descriptions

| Node | Agent | Purpose | Input | Output |
|------|-------|---------|-------|---------|
| `triage_node` | `TriageGuardrailAgent` | Query classification, safety checks, routing decisions | `original_query`, `chat_history` | `classified_agent`, `next_step`, `is_toxic`, `need_analysis` |
| `direct_answer` | `DirectAnswerAgent` | Simple, direct responses for basic queries | `original_query` | `agent_response` |
| `specialist_agent` | Domain specialists | Execute domain-specific knowledge retrieval | `original_query`, `classified_agent` | `agent_response`, `confidence_score` |
| `plan_executor` | Various specialists | Multi-agent coordination (currently simulated) | `original_query`, `classified_agent` | `agent_response` |
| `verification_node` | Built-in logic | Verify response quality, safety, confidence | `agent_response`, `error_message` | `is_final_answer`, verification status |
| `fallback_agent` | `FallbackAgent` | Safe fallback responses for failures | `original_query`, `error_message` | `agent_response` |
| `synthesizer_node` | `SynthesizerAgent` | Combine multi-agent results | Multiple agent responses | `agent_response` |
| `final_answer` | `FinalAnswerAgent` | Polish and format final response | `agent_response` | `agent_response`, streaming events |
| `question_generator` | `QuestionGeneratorAgent` | Generate follow-up questions | `agent_response`, `original_query` | `suggested_questions` |

### Security Checkpoints

The workflow includes multiple security checkpoints:

1. **Entry Triage**: Initial toxicity and safety screening
2. **Agent Allowlist**: Only guest-safe agents are accessible
3. **Verification Gate**: Post-processing safety and quality checks
4. **Fallback Safety**: Safe responses for any failure scenario

### MCP (Model Context Protocol) Integration

The GuestWorkflow leverages advanced MCP integration for superior public data retrieval while maintaining strict security boundaries:

#### MCP Retriever Factory for Guests

The workflow uses `MCPRetrieverFactory` for centralized MCP tool management with guest-specific restrictions:

```python
from app.agents.factory.mcp_retriever_factory import MCPRetrieverFactory

# Available MCP retriever types for guests (public-only data)
guest_mcp_types = {
    'company': 'Company documents and public information retriever',
    'product': 'Product catalog and public information retriever',
    'medical': 'Medical and healthcare public information retriever',
    'drug': 'Drug and pharmacological public information retriever',
    'genetic': 'Genetic and biomedical public information retriever'
}
# Note: 'customer' and 'employee' MCP types are explicitly excluded for guests
```

#### Guest MCP Tool Configuration

Each MCP tool is configured for public access only:

| MCP Tool | Collection | Watch Directory | Description | Guest Access Level |
|----------|------------|-----------------|-------------|-------------------|
| `company_retriever_mcp_tool` | `company_knowledge` | `app/agents/retrievers/storages/companies` | Company public documents and information | Public only |
| `product_retriever_mcp_tool` | `product_knowledge` | `app/agents/retrievers/storages/products` | Product specifications and public catalogs | Public only |
| `medical_retriever_mcp_tool` | `medical_docs` | `app/agents/retrievers/storages/medical_docs` | Medical research and public guidelines | Public only |
| `drug_retriever_mcp_tool` | `drug_knowledge` | `app/agents/retrievers/storages/drugs` | Drug information and public interactions | Public only |
| `genetic_retriever_mcp_tool` | `genetic_knowledge` | `app/agents/retrievers/storages/genetics` | Genetic and genomic public data | Public only |

**Security Note:** Customer and employee-specific MCP tools are **explicitly excluded** from the GuestWorkflow to ensure data privacy and security.

#### MCP Server Configuration for Guests

```python
# Guest MCP server configuration (public data only)
MCP_SERVER_URL = "http://localhost:50051/sse"

# MCP retriever initialization for guest agents (secure configuration)
{
    "CompanyAgent": CompanyAgent(llm=llm, default_tool_names=["company_retriever_mcp_tool"]),
    "ProductAgent": ProductAgent(llm=llm, default_tool_names=["product_retriever_mcp_tool"]),
    "MedicalAgent": MedicalAgent(llm=llm, default_tool_names=["medical_retriever_mcp_tool"]),
    "DrugAgent": DrugAgent(llm=llm, default_tool_names=["drug_retriever_mcp_tool"]),
    "GeneticAgent": GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_mcp_tool"]),
    "DirectAnswerAgent": NaiveAgent(llm=llm, default_tool_names=["searchweb_tool"]),
    "FallbackAgent": NaiveAgent(llm=llm, default_tool_names=["searchweb_tool", "company_retriever_mcp_tool"]),
    # Note: Customer and Employee agents are explicitly excluded
}
```

#### Guest MCP Security Features

- **Public Data Only**: MCP tools configured to access only public information
- **No Personal Data**: Customer and employee MCP retrievers are not available
- **Privacy Protection**: No personalization or user-specific data retrieval
- **Safe Fallbacks**: Traditional tools available as backup
- **Audit Logging**: All MCP access logged for security monitoring

## State Management

The GuestWorkflow uses a comprehensive `GraphState` (AgentState) to track workflow execution:

```python
class GraphState(TypedDict):
    # Core query fields
    original_query: str
    rewritten_query: Optional[str]
    classified_agent: Optional[str]
    next_step: Optional[str]
    agent_response: str
    
    # Guest context
    guest_id: Optional[str]
    interaction_id: Optional[str]
    session_id: Optional[str]
    user_role: str  # Always "guest"
    
    # Workflow control
    is_toxic: Optional[bool]
    need_analysis: Optional[bool]
    is_final_answer: bool
    error_message: Optional[str]
    confidence_score: Optional[float]
    
    # Content and suggestions
    chat_history: List[Dict[str, str]]
    suggested_questions: List[str]
    
    # Metadata
    timestamp: Optional[str]
    iteration_count: int
    workflow_type: str  # "guest"
    session_context: Dict[str, Any]
```

### State Flow Through Nodes

1. **Initial State**: Created with `original_query`, `guest_id`, `user_role="guest"`
2. **After Triage**: Enriched with `classified_agent`, `next_step`, safety flags
3. **After Agent Execution**: Contains `agent_response`, `confidence_score`
4. **After Verification**: Updated with `is_final_answer`, validation results
5. **Final State**: Complete with `suggested_questions`, final response

---

### ToolFactory
- Centralizes creation and management of all tools and retrievers.
- Supports both static (reusable) and dynamic (per-user or per-query) tool instantiation.
- Integrates with MCPRetrieverFactory for advanced, remote retrievers (MCP clients).
- Ensures only guest-safe tools are available in the guest workflow.

### MCPRetrieverFactory
- Provides standardized creation of MCP retrievers for public data domains (drug, genetic, company, product, medical, etc.).
- Handles configuration, directory management, and server URL injection.
- Used by ToolFactory to instantiate MCP-based tools for agents.

---

## Agents Used in GuestWorkflow

### Core Workflow Agents
- **TriageGuardrailAgent:** Classifies and routes queries, applies safety/guardrails.
- **SynthesizerAgent:** Combines results for complex/multi-agent queries.
- **FinalAnswerAgent:** Produces the final answer, supports streaming.
- **QuestionGeneratorAgent:** Suggests follow-up questions.
- **NaiveAgent:** Used for fallback and direct answer (simple queries).

### Specialist Agents with MCP Tools
| Agent | Purpose | Traditional Tools | MCP Tools | Guest Access Level |
|-------|---------|------------------|-----------|-------------------|
| `CompanyAgent` | Company information retrieval | `company_retriever_tool` | `company_retriever_mcp_tool` | Public information only |
| `ProductAgent` | Product specifications and details | `product_retriever_tool` | `product_retriever_mcp_tool` | Public catalogs only |
| `MedicalAgent` | Medical information and research | `medical_retriever_tool` | `medical_retriever_mcp_tool` | Public medical data only |
| `DrugAgent` | Drug information and interactions | `drug_retriever_tool` | `drug_retriever_mcp_tool` | Public drug data only |
| `GeneticAgent` | Genetic and genomic information | `genetic_retriever_tool` | `genetic_retriever_mcp_tool` | Public genetic data only |
| `DirectAnswerAgent` | Simple responses | `searchweb_tool` | None | Public web search |
| `FallbackAgent` | Error recovery | `searchweb_tool`, `company_retriever_mcp_tool` | Public MCP tools | Safe public fallback |

**Security Note:** Customer and Employee agents are **explicitly excluded** from GuestWorkflow to ensure data privacy and prevent unauthorized access to personal or internal information.

---

## Security & Privacy
- **No Customer/Employee Data:** GuestWorkflow does NOT instantiate or route to CustomerAgent or EmployeeAgent.
- **Strict Agent Allowlist:** Only a specific set of agents are available to guests.
- **MCP Security:** Only public-facing MCP tools are accessible; customer and employee MCP retrievers are blocked.
- **Toxicity & Error Handling:** All responses are checked for toxicity, errors, and confidence before being returned.
- **No Caching:** Caching is disabled for guests to avoid state confusion and ensure privacy.
- **Public Data Only:** All MCP tools configured for public information access exclusively.
- **Privacy Protection:** No personalization or user-specific data collection for guest users.

---

## Extending the Workflow
- To add a new public-facing agent, update the `_initialize_agents` method and ensure it is included in the allowed_guest_agents list.
- To add new tools or retrievers, register them in the ToolFactory and, if needed, in MCPRetrieverFactory.
- For new routing logic, update the `_route_from_triage` and related methods.

---

## Usage Examples

### Basic Streaming Usage

```python
from app.agents.workflow.guest_workflow import GuestWorkflow

# Initialize workflow
workflow = GuestWorkflow()

# Simple streaming execution
async for event in workflow.arun_streaming(
    query="What are the side effects of aspirin?",
    config={"configurable": {"thread_id": "guest_123"}}
):
    if event['event'] == 'node_start':
        print(f"Starting: {event['data']['node']}")
    elif event['event'] == 'answer_chunk':
        print(event['data'], end="", flush=True)
    elif event['event'] == 'final_result':
        print(f"\nSuggestions: {event['data']['suggested_questions']}")
```

### Guest MCP Usage Examples

```python
# Example of guest workflow with MCP tool monitoring
async for event in workflow.arun_streaming(
    query="What genetic testing products do you offer?",
    config={"configurable": {"thread_id": "guest_mcp_123"}}
):
    if event['event'] == 'node_start':
        print(f"Starting node: {event['data']['node']}")
    elif event['event'] == 'mcp_tool_start':
        print(f"Using public MCP tool: {event['data']['tool_name']}")
    elif event['event'] == 'mcp_tool_complete':
        print(f"MCP tool completed: {event['data']['result_count']} results")
    elif event['event'] == 'security_check':
        print(f"Security validation: {event['data']['status']}")
    elif event['event'] == 'answer_chunk':
        print(event['data'], end="", flush=True)
    elif event['event'] == 'final_result':
        print(f"\nSuggestions: {event['data']['suggested_questions']}")
```

### Guest MCP Tool Management

```python
from app.agents.factory.factory_tools import TOOL_FACTORY

# Get available public MCP tools for guests
public_mcp_tools = [
    "company_retriever_mcp_tool",
    "product_retriever_mcp_tool", 
    "medical_retriever_mcp_tool",
    "drug_retriever_mcp_tool",
    "genetic_retriever_mcp_tool"
]

# Verify guest-safe MCP tools are available
for tool_name in public_mcp_tools:
    tool = TOOL_FACTORY.get_static_tool(tool_name)
    if tool:
        print(f"Guest MCP tool {tool_name} is available")
    else:
        print(f"Warning: {tool_name} not available")

# Note: Customer and employee MCP tools should NOT be accessible
restricted_tools = ["customer_retriever_mcp_tool", "employee_retriever_mcp_tool"]
for tool_name in restricted_tools:
    tool = TOOL_FACTORY.get_static_tool(tool_name)
    if tool is None:
        print(f"✓ Security: {tool_name} properly restricted for guests")
    else:
        print(f"⚠️ Security Warning: {tool_name} accessible to guests")
```

### Direct Graph Streaming

```python
# Using the direct graph streaming method
async for event in workflow.astreaming_workflow(
    query="Tell me about your company",
    config={"configurable": {}},
    guest_id="guest_456"
):
    if event['event'] == 'on_chat_model_stream':
        # Handle streaming tokens from final answer
        content = event['data'].get('content', '')
        if content:
            print(content, end='', flush=True)
    elif event['event'] == 'final_result':
        result = event['data']
        print(f"\nFinal Answer: {result['agent_response']}")
        print(f"Suggested Questions: {result['suggested_questions']}")
```

### Error Handling

```python
try:
    async for event in workflow.arun_streaming(query="inappropriate content"):
        if event['event'] == 'error':
            error_msg = event['data']['error']
            if 'toxic content' in error_msg.lower():
                print("Content blocked for safety reasons")
            else:
                print(f"Error: {error_msg}")
        elif event['event'] == 'final_result':
            # Handle successful completion
            pass
except Exception as e:
    print(f"Workflow error: {e}")
```

### Session Management with Chat History

```python
# Example with chat history
chat_history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help you today?"}
]

async for event in workflow.arun_streaming(
    query="What products do you recommend?",
    config={"configurable": {"thread_id": "guest_session_789"}},
    guest_id="guest_789",
    chat_history=chat_history
):
    # Process events...
    pass
```

## Event Types

### Streaming Events

| Event Type | Description | Data Structure |
|------------|-------------|----------------|
| `workflow_start` | Workflow execution begins | `{message: str, guest_id: str}` |
| `node_start` | Node execution starts | `{node: str, timestamp: str}` |
| `node_complete` | Node execution completes | `{node: str, duration: float}` |
| `answer_chunk` | Streaming response content | `{content: str, chunk_id: int}` |
| `triage_decision` | Triage routing decision | `{classified_agent: str, next_step: str, confidence: float}` |
| `mcp_tool_start` | Guest MCP tool execution begins | `{tool_name: str, tool_type: str, access_level: "public"}` |
| `mcp_tool_complete` | Guest MCP tool execution complete | `{tool_name: str, result_count: number, access_level: "public"}` |
| `safety_check` | Security/safety validation | `{is_safe: bool, reason?: str}` |
| `security_check` | MCP security validation | `{status: str, access_level: str, tool_restricted?: bool}` |
| `agent_execution` | Agent processing update | `{agent: str, status: str}` |
| `verification_result` | Response verification outcome | `{passed: bool, confidence: float}` |
| `final_result` | Complete workflow result | `{agent_response: str, suggested_questions: List[str], metadata: Dict}` |
| `error` | Error occurred | `{error: str, node?: str, recoverable: bool, security_related?: bool}` |

## Configuration

### Guest Access Controls

The GuestWorkflow enforces strict access controls:

```python
# Allowed guest agents (hardcoded security list)
ALLOWED_GUEST_AGENTS = [
    "CompanyAgent",      # Public company information
    "ProductAgent",      # Product catalog and specifications
    "GeneticAgent",      # General genetic information
    "MedicalAgent",      # General medical knowledge
    "DrugAgent",         # Public drug information
    "DirectAnswerAgent", # Simple Q&A
    "NaiveAgent",        # Fallback responses
]

# Prohibited agents for guests
PROHIBITED_AGENTS = [
    "CustomerAgent",     # Customer-specific data
    "EmployeeAgent",     # Internal employee data
    "AdminAgent",        # Administrative functions
]
```

### Environment Variables

```bash
# Core LLM Configuration
LLM_PROVIDER=vllm
LLM_MODEL=your-model-name
VLLM_API_URL=http://localhost:6622/v1

# MCP Server Configuration
MCP_SERVER_URL=http://localhost:8000

# Vector Store Settings
VECTOR_STORE_BASE_DIR=./vector_stores_data
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Database Collections for Public Data
COMPANY_DB=company_knowledge
PRODUCTS_DB=product_knowledge
GENETIC_DB=genetic_knowledge
DRUGS_DB=drug_knowledge
MEDICAL_DB=medical_docs

# Security Settings
ENABLE_TOXICITY_FILTER=true
ENABLE_CONTENT_MODERATION=true
MAX_QUERY_LENGTH=1000
```

## Performance & Monitoring

### Performance Optimizations

1. **Caching Disabled**: No caching for guests to ensure privacy
2. **Fast Path Routing**: Simple queries bypass complex processing
3. **Resource Cleanup**: Automatic connection management
4. **Streaming Responses**: Progressive content delivery
5. **Early Termination**: Toxic content blocked at triage

### Monitoring Features

```python
# Automatic logging to:
# app/logs/log_workflows/guest_workflow_optimized.log

# Metrics tracked:
# - Query processing time
# - Triage decision accuracy
# - Agent execution duration
# - Safety filter effectiveness
# - Error rates and types
```

### Debug Information

Enable debug logging to see detailed workflow execution:

```python
import logging
from loguru import logger

# Enable debug level logging
logger.add("guest_workflow_debug.log", level="DEBUG")

# Debug information includes:
# - Full state transitions
# - Triage decision reasoning
# - Agent selection logic
# - Security checkpoint results
# - Performance timings
```

---

## Testing

### Example Test Queries

```python
# Test queries for different workflow paths
test_queries = [
    # Direct Answer Path (Fast)
    "Hello, how are you?",
    "Thank you for your help",
    "What time is it?",
    
    # Company Agent Path
    "What is your company's mission?",
    "Tell me about your contact information",
    "What are your business hours?",
    
    # Product Agent Path
    "What products do you sell?",
    "Show me your product catalog",
    "What are the specifications of your genetic tests?",
    
    # Medical Agent Path
    "What is diabetes?",
    "Tell me about common symptoms of flu",
    "What are the types of cancer?",
    
    # Drug Agent Path
    "What is aspirin used for?",
    "Tell me about paracetamol side effects",
    "What are antibiotics?",
    
    # Genetic Agent Path
    "What is DNA sequencing?",
    "Explain genetic mutations",
    "What is pharmacogenomics?",
    
    # Toxic Content (Should be blocked)
    "How to make harmful substances",
    "Inappropriate content here",
    
    # Complex Queries (Multi-agent potential)
    "Compare your genetic testing products with their medical applications",
    "How do your company's genetic tests help with drug selection?",
    
    # Clarification Needed
    "Tell me about that thing",
    "What about the stuff we discussed?",
]
```

### Comprehensive Test Suite

```python
import asyncio
from datetime import datetime

async def test_guest_workflow():
    """Comprehensive test suite for GuestWorkflow"""
    workflow = GuestWorkflow()
    
    test_results = {
        "direct_answer": [],
        "specialist_routing": [],
        "safety_filtering": [],
        "error_handling": [],
        "streaming_quality": []
    }
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing: {query}")
        print(f"{'='*60}")
        
        try:
            events = []
            async for event in workflow.arun_streaming(
                query=query,
                config={"configurable": {"thread_id": f"test_{datetime.utcnow().timestamp()}"}}
            ):
                events.append(event)
                
                if event['event'] == 'triage_decision':
                    print(f"Routed to: {event['data']['classified_agent']}")
                elif event['event'] == 'safety_check':
                    print(f"Safety: {event['data']['is_safe']}")
                elif event['event'] == 'final_result':
                    result = event['data']
                    print(f"Response length: {len(result['agent_response'])}")
                    print(f"Suggestions: {len(result['suggested_questions'])}")
                elif event['event'] == 'error':
                    print(f"Error: {event['data']['error']}")
            
            # Analyze test results
            # ... analysis logic here
            
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    return test_results

# Run tests
if __name__ == '__main__':
    asyncio.run(test_guest_workflow())
```

### Performance Testing

```python
import time
import statistics

async def performance_test():
    """Test workflow performance metrics"""
    workflow = GuestWorkflow()
    
    queries = [
        "Hello",  # Fast path
        "What products do you sell?",  # Specialist path
        "Tell me about aspirin",  # Drug specialist
    ]
    
    for query in queries:
        times = []
        
        for i in range(10):  # 10 iterations per query
            start_time = time.time()
            
            async for event in workflow.arun_streaming(query):
                if event['event'] == 'final_result':
                    end_time = time.time()
                    times.append(end_time - start_time)
                    break
        
        avg_time = statistics.mean(times)
        print(f"Query: {query}")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Min/Max: {min(times):.3f}s / {max(times):.3f}s")
        print()

asyncio.run(performance_test())
```

## Error Handling & Recovery

### Error Categories

1. **Triage Failures**
   - Invalid query format
   - Classification errors
   - Missing required fields

2. **Agent Execution Errors**
   - Tool failures
   - MCP server unavailable
   - Response generation issues

3. **Verification Failures**
   - Low confidence responses
   - Safety violations
   - Content quality issues

4. **System Errors**
   - Network timeouts
   - Resource exhaustion
   - Configuration issues

### Recovery Strategies

```python
# Built-in recovery mechanisms:

# 1. Triage Recovery
if not classified_agent or classified_agent not in allowed_agents:
    # Fallback to DirectAnswerAgent
    state['classified_agent'] = 'DirectAnswerAgent'
    state['next_step'] = 'direct_answer'

# 2. Agent Execution Recovery
try:
    result = await agent.aexecute(state)
except Exception as e:
    # Route to fallback agent
    state['error_message'] = str(e)
    return await fallback_agent.aexecute(state)

# 3. Verification Recovery
if not verification_passed:
    # Second chance with fallback
    state['is_final_answer'] = False
    return route_to_fallback()

# 4. Final Safety Net
if all_else_fails:
    return {
        'agent_response': 'I apologize, but I cannot process your request right now. Please try again later.',
        'suggested_questions': ['How can I help you?', 'What would you like to know?']
    }
```

## Migration & Compatibility

### From Previous Versions

When migrating from older guest workflow versions:

1. **State Structure**: New fields in GraphState - ensure compatibility
2. **Agent Names**: Some agents may have been renamed
3. **Security Model**: Enhanced safety checks and agent restrictions
4. **Event Types**: New streaming events for better UX

### Integration Points

```python
# Integration with other systems:

# 1. Frontend Integration
# - Handles streaming events
# - Processes safety notifications
# - Manages session state

# 2. Backend Services
# - MCP server integration
# - Vector store connections
# - Logging and monitoring

# 3. Security Layer
# - Content moderation API
# - Toxicity detection service
# - Access control validation
```

## Deployment Considerations

### Resource Requirements

- **Memory**: 2-4GB RAM for LLM inference
- **CPU**: Multi-core for parallel processing
- **Storage**: Vector stores and logs
- **Network**: MCP server connectivity

### Scaling

- **Horizontal**: Multiple workflow instances
- **Vertical**: Resource allocation per instance
- **Caching**: Disabled for privacy (guests)
- **Load Balancing**: Session-aware routing

### Security Checklist

- [ ] Guest agent allowlist properly configured
- [ ] Toxicity filtering enabled
- [ ] Content moderation active
- [ ] No customer/employee data access
- [ ] Logging excludes sensitive information
- [ ] Rate limiting configured
- [ ] Input validation active

---

## File Structure

```
workflow/
├── guest_workflow.py           # Main workflow implementation
├── state.py                    # State definitions (GraphState)
├── initalize.py               # LLM and global configuration
└── README_GuestWorkflow.md    # This documentation

factory/
├── factory_tools.py           # Tool factory and management
├── mcp_retriever_factory.py   # MCP retriever factory
└── tools/                     # Individual tool implementations

agents/
├── stores/                    # Agent implementations
│   ├── triage_guardrail_agent.py
│   ├── synthesizer_agent.py
│   ├── final_answer_agent.py
│   ├── question_generator_agent.py
│   ├── naive_agent.py
│   ├── company_agent.py
│   ├── product_agent.py
│   ├── medical_agent.py
│   ├── drug_agent.py
│   └── genetic_agent.py
└── retrievers/               # MCP retriever implementations
    ├── company_mcp_retriever.py
    ├── product_mcp_retriever.py
    ├── medical_mcp_retriever.py
    ├── drug_mcp_retriever.py
    └── genetic_mcp_retriever.py
```

---

## References

- **LangGraph Documentation**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **Model Context Protocol (MCP)**: For advanced retriever implementations
- **Security Best Practices**: Guest workflow security guidelines
- **Performance Optimization**: Workflow tuning and monitoring guides

---

**Last Updated**: September 3, 2025  
**Version**: 1.0.0  
**Compatibility**: Python 3.8+, LangChain 0.1+, LangGraph 0.1+

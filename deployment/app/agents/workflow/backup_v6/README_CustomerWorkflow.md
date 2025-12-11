# Customer Workflow - Optimized Agentic System

## Overview

The **CustomerWorkflow** is an advanced, optimized agentic workflow system designed to provide intelligent, adaptive responses to customer queries. It uses a sophisticated triage-based routing system that dynamically selects the most appropriate processing path based on query complexity and requirements.

## Architecture

### Core Design Principles

1. **Intelligent Triage**: Uses a `TriageRouterAgent` for smart query classification and routing
2. **Adaptive Paths**: Multiple processing paths optimized for different query types
3. **Robust Fallbacks**: Comprehensive error handling and fallback mechanisms
4. **Streaming Support**: Real-time response streaming for better user experience
5. **Enhanced State Management**: Comprehensive state tracking with sentiment analysis support

### Workflow Paths

The system supports three main execution paths:

#### 1. Fast Path (Direct Answer)
- **Purpose**: Simple queries that don't require specialist knowledge
- **Flow**: `Triage → Direct Answer → Final Answer`
- **Examples**: Greetings, basic questions, general chitchat
- **Agents**: `DirectAnswerAgent` (NaiveAgent)

#### 2. Standard Path (Specialist Agent)
- **Purpose**: Complex queries requiring domain expertise
- **Flow**: `Triage → Specialist Agent → Verification → Final Answer`
- **Examples**: Medical questions, product inquiries, company information
- **Agents**: `CompanyAgent`, `MedicalAgent`, `DrugAgent`, `GeneticAgent`, `ProductAgent`, `CustomerAgent`

#### 3. Multi-Agent Path (Complex Queries)
- **Purpose**: Queries requiring multiple domain specialists
- **Flow**: `Triage → Plan Executor → Synthesizer → Final Answer`
- **Examples**: Comparative analysis, cross-domain questions
- **Agents**: Multiple specialists coordinated by `SynthesizerAgent`

## Key Components

### Core Agents

| Agent | Purpose | Tools |
|-------|---------|-------|
| `TriageRouterAgent` | Query classification and routing decisions | None |
| `DirectAnswerAgent` | Simple, direct responses | None |
| `CompanyAgent` | Company information and policies | `company_retriever_tool` |
| `MedicalAgent` | Medical information and advice | `medical_retriever_tool` |
| `DrugAgent` | Pharmaceutical information | `drug_retriever_tool` |
| `GeneticAgent` | Genetic analysis and information | `genetic_retriever_tool` |
| `ProductAgent` | Product specifications and details | `product_retriever_tool` |
| `CustomerAgent` | Customer service and support | Various customer tools |
| `FinalAnswerAgent` | Response polishing and formatting | None |
| `FallbackAgent` | Error recovery and safe responses | None |
| `SynthesizerAgent` | Multi-agent result aggregation | None |

### State Management

The workflow uses a comprehensive `GraphState` that tracks:

```python
class GraphState(TypedDict):
    # Core fields
    original_query: str
    rewritten_query: str
    classified_agent: Literal[...] 
    next_step: Optional[str]
    agent_response: str
    
    # Customer context
    customer_id: Optional[str]
    customer_role: Optional[str]
    interaction_id: Optional[str]
    
    # Workflow control
    is_multi_step: Optional[bool]
    should_re_execute: Optional[bool]
    is_final_answer: bool
    
    # Sentiment analysis (Enhanced version)
    sentiment_analysis: Dict[str, Any]
    needs_re_execution: bool
    was_re_executed: bool
    
    # Metadata
    chat_history: List[Tuple[str, str]]
    suggested_questions: List[str]
    agent_thinks: Dict[str, Any]
    timestamp: Optional[str]
```

## Features

### 1. Intelligent Routing

The `TriageRouterAgent` analyzes each query and determines:
- **Query Classification**: Which specialist agent should handle the query
- **Complexity Assessment**: Whether multi-agent coordination is needed
- **Processing Path**: Fast path, standard path, or multi-agent path
- **Re-execution Needs**: Whether to re-process previous queries

### 2. Robust Error Handling

- **Triage Failures**: Automatic fallback to safe defaults
- **Agent Errors**: Verification step catches poor responses
- **Routing Issues**: Recovery based on available state information
- **Graceful Degradation**: Fallback agent provides basic responses

### 3. Enhanced Streaming

Real-time response streaming with support for:
- **Answer Chunks**: Progressive response building
- **Node Progress**: Workflow step notifications
- **Sentiment Events**: User satisfaction analysis
- **Error Events**: Graceful error communication
- **Metadata**: Rich contextual information

### 4. Customer Context Awareness

- **Role-based Access**: Different capabilities for customer types
- **Session Management**: Persistent conversation context
- **Interaction Tracking**: Unique interaction IDs
- **Privacy Controls**: Customer data protection

## Usage

### Basic Usage

```python
from app.agents.workflow.customer_workflow import CustomerWorkflow

# Initialize workflow
workflow = CustomerWorkflow()

# Simple streaming execution
async for event in workflow.arun_streaming("What are the side effects of aspirin?"):
    if event['event'] == 'node_start':
        print(f"Starting: {event['data']['node']}")
    elif event['event'] == 'answer_chunk':
        print(event['data'], end="", flush=True)
    elif event['event'] == 'final_result':
        print(f"\nSuggested: {event['data']['suggested_questions']}")
```

### Authenticated Customer Usage

```python
import uuid
from datetime import datetime

# Enhanced authenticated streaming
config = {
    "configurable": {
        "thread_id": f"customer_session_{datetime.utcnow().timestamp()}"
    }
}

async for event in workflow.arun_streaming_authenticated(
    query="I need help with my order",
    config=config,
    customer_id=12345,
    customer_role="premium_customer",
    interaction_id=uuid.uuid4(),
    chat_history=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
):
    # Handle streaming events
    if event['event'] == 'sentiment_analysis_result':
        print(f"User sentiment: {event['data']['user_intent']}")
    elif event['event'] == 'final_result':
        result = event['data']
        print(f"Response: {result['agent_response']}")
        print(f"Was re-executed: {result['was_re_executed']}")
```

### Session Management

```python
# Create customer session
session_id = await create_customer_workflow_session(
    customer_id=12345,
    customer_role="customer"
)

# Validate customer access
has_access = validate_customer_access(
    customer_role="premium_customer",
    requested_feature="advanced_search"
)
```

## Configuration

### Customer Role Access Matrix

| Role | Features |
|------|----------|
| `customer` | Basic search, company info, product info, sentiment analysis |
| `premium_customer` | + Advanced search, priority support, query re-execution |
| `vip_customer` | + Personal consultant, priority re-execution |
| `admin` | All features |

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=vllm  # or ollama
LLM_MODEL=your-model-name
VLLM_API_URL=http://localhost:6622/v1

# Reasoning LLM (for complex analysis)
REASONING_LLM_PROVIDER=vllm
LLM_REASONING_MODEL=your-reasoning-model

# Database and Caching
ENABLE_CACHING=true
CACHE_TTL=1800  # 30 minutes
```

## Event Types

### Streaming Events

| Event Type | Description | Data |
|------------|-------------|------|
| `node_start` | Workflow node begins execution | `{node: string}` |
| `answer_chunk` | Partial response content | `{response_text: string}` |
| `sentiment_analysis_start` | Sentiment analysis begins | `{message: string}` |
| `sentiment_analysis_result` | Sentiment analysis complete | `{user_intent, confidence, etc.}` |
| `re_execution_start` | Query re-processing begins | `{message: string}` |
| `re_execution_complete` | Query re-processing done | `{message: string}` |
| `final_result` | Complete workflow result | `{agent_response, suggested_questions, metadata}` |
| `error` | Error occurred | `{error: string, node?: string}` |

## Monitoring and Debugging

### Logging

The workflow provides comprehensive logging at multiple levels:

```python
# Log files are automatically created in:
# app/logs/log_workflows/customer_workflow_optimized.log

# Debug level logging includes:
# - Triage decisions and reasoning
# - State transitions and routing
# - Agent execution details
# - Performance metrics
# - Error stack traces
```

### Debug Information

Enable debug logging to see:
- Full state keys at each step
- Triage agent decision reasoning
- Routing logic evaluation
- Agent selection process
- Error recovery attempts

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Performance Considerations

### Optimization Features

1. **Fast Path Routing**: Simple queries bypass complex processing
2. **Intelligent Caching**: Frequently requested information cached
3. **Streaming Responses**: Progressive content delivery
4. **Connection Management**: Automatic database connection cleanup
5. **Resource Monitoring**: Database pool and memory tracking

### Scalability

- **Stateless Design**: Each request is independent
- **Async Processing**: Non-blocking I/O operations
- **Resource Cleanup**: Automatic connection and memory management
- **Error Isolation**: Agent failures don't cascade

## Error Handling

### Error Recovery Strategies

1. **Triage Failures**: Default to direct answer path
2. **Agent Errors**: Route to fallback agent
3. **Routing Issues**: Attempt recovery based on available state
4. **Stream Errors**: Graceful error events to client

### Common Error Scenarios

```python
# Example error handling in client code
async for event in workflow.arun_streaming_authenticated(...):
    if event['event'] == 'error':
        error_type = event['data'].get('error')
        if 'workflow complexity limit exceeded' in error_type:
            # Handle recursion limit
            print("Query too complex, please simplify")
        else:
            # Handle general errors
            print(f"Error: {error_type}")
```

## Testing

### Example Test Queries

```python
test_queries = [
    # Fast path
    "Hello, how are you today?",
    
    # Specialist path - Medical
    "What are the main side effects of Paracetamol?",
    
    # Clarification needed
    "Tell me about that thing I asked about before.",
    
    # Multi-agent path
    "Compare your company's stock performance to its return policy."
]
```

### Running Tests

```python
if __name__ == '__main__':
    async def main():
        workflow = CustomerWorkflow()
        
        for query in test_queries:
            print(f"\n{'='*20} TESTING QUERY: {query} {'='*20}")
            async for event in workflow.arun_streaming(query):
                # Process events...
                pass
    
    asyncio.run(main())
```

## Migration Notes

### From Previous Versions

If migrating from older workflow versions:

1. **State Fields**: New fields added to `GraphState` - ensure compatibility
2. **Agent Names**: Some agents renamed for clarity
3. **Event Types**: New streaming events for enhanced features
4. **Error Handling**: More granular error types and recovery

### Backward Compatibility

The workflow maintains backward compatibility for:
- Basic streaming interface
- Core agent execution
- Essential state fields
- Error event structure

## Future Enhancements

### Planned Features

1. **Advanced Sentiment Analysis**: Deeper emotion and intent understanding
2. **Predictive Routing**: ML-based routing optimization
3. **Custom Agent Chains**: User-defined agent sequences
4. **Performance Analytics**: Detailed workflow metrics
5. **A/B Testing**: Workflow variant testing
6. **Multi-language Support**: Internationalization features

### Extension Points

The workflow is designed for extensibility:
- **Custom Agents**: Add domain-specific agents
- **New Routing Logic**: Implement custom routing strategies
- **Enhanced State**: Add application-specific state fields
- **Event Handlers**: Custom event processing
- **Middleware**: Request/response processing layers

## Contributing

### Adding New Agents

1. Create agent class inheriting from `BaseAgentNode`
2. Add to agent initialization in `_initialize_agents()`
3. Update `GraphState` if new state fields needed
4. Add routing logic if custom routing required
5. Update documentation and tests

### Code Structure

```
workflow/
├── customer_workflow.py      # Main workflow implementation
├── state.py                 # State definitions
├── initalize.py            # LLM and global setup
└── README_CustomerWorkflow.md # This documentation
```

## License

This workflow is part of the Agentic GeneStory Platform and follows the project's licensing terms.

---

**Last Updated**: August 7, 2025  
**Version**: 2.0.0  
**Compatibility**: Python 3.8+, LangChain 0.1+, LangGraph 0.1+

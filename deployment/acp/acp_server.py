from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart 
from acp_sdk.server import RunYield, RunYieldResume, Server
from typing import Dict, Any

from loguru import logger
import yaml
from initialize_llm import (
    AgentFactory, AgentManager, LLMProvider, AgentType,
    initialize_global_agents, create_agents_from_config, load_config
)
from config import get_settings
from dotenv import load_dotenv

# Load environment and settings
load_dotenv()
settings = get_settings()

# ========================================
# METHOD 1: Create from Settings
# ========================================
def create_agents_from_settings():
    """Create agents using your application settings."""
    
    # Create different types of agents from settings
    standard_agent = AgentFactory.create_from_settings(settings, AgentType.STANDARD)
    reasoning_agent = AgentFactory.create_from_settings(settings, AgentType.REASONING)
    chat_agent = AgentFactory.create_from_settings(settings, AgentType.CHAT)
    analysis_agent = AgentFactory.create_from_settings(settings, AgentType.ANALYSIS)
    
    # Get the ChatModel instances for direct use
    standard_llm = standard_agent.get_chat_model()
    reasoning_llm = reasoning_agent.get_chat_model()
    chat_llm = chat_agent.get_chat_model()
    analysis_llm = analysis_agent.get_chat_model()
    
    return {
        'standard': standard_llm,
        'reasoning': reasoning_llm,
        'chat': chat_llm,
        'analysis': analysis_llm
    }

server = Server()



agent_config = load_config("agent.yaml")
print("Loaded agent configuration:", agent_config)

agents = create_agents_from_config(agent_config)
print("Created agents from config:", agents)
analysis_agent = agents.get("analysis_agent")

@server.agent()
async def research_drafter(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Agent that creates a general research summary on a given topic."""
    if not analysis_agent:
        yield Message(parts=[MessagePart(content="Error: Analysis agent not initialized.")])
        return
    print(f"Received input for research_drafter: {input}")
    query = input[0].parts[0].content
    task_output = await analysis_agent.ainvoke(query)
    yield Message(parts=[MessagePart(content=str(task_output))])

if __name__ == "__main__":
    server.run(port=6666)
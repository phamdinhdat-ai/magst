import re
import os
import sys
import json
import time
import chromadb
from typing import Optional, TypedDict, Literal, List, Tuple, Dict, Any, Callable
from loguru import logger
import asyncio
from typing import List
import torch
from abc import ABC, abstractmethod
from enum import Enum

# --- LangChain Core & Community Imports ---
from langchain_ollama import ChatOllama
from langchain_community.llms.vllm import VLLM, VLLMOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

# --- Tool Imports ---
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
from app.core.config import get_settings
import yaml
import atexit

class LLMProvider(Enum):
    """Enumeration for supported LLM providers."""
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"

class AgentType(Enum):
    """Enumeration for different agent types."""
    STANDARD = "standard"
    REASONING = "reasoning"
    CHAT = "chat"
    ANALYSIS = "analysis"

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, model_name: str, provider: LLMProvider, **kwargs):
        self.model_name = model_name
        self.provider = provider
        self.config = kwargs
        self.chat_model: BaseChatModel = self._create_chat_model()
    
    @abstractmethod
    def _create_chat_model(self) -> BaseChatModel:
        """Create the LangChain ChatModel instance based on provider."""
        pass
    
    def get_chat_model(self) -> BaseChatModel:
        """Get the LangChain ChatModel instance."""
        return self.chat_model
    
    def __repr__(self):
        return f"{self.__class__.__name__}(provider={self.provider.value}, model={self.model_name})"

class VLLMAgent(BaseAgent):
    """Agent using VLLM provider - returns ChatOpenAI instance."""
    
    def _create_chat_model(self) -> ChatOpenAI:
        try:
            base_url = self.config.get('base_url', "http://0.0.0.0:6622/v1")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            
            os.environ["OPENAI_API_KEY"] = "EMPTY"
            
            chat_model = ChatOpenAI(
                model=self.model_name,
                base_url=base_url,
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.95),
                max_tokens=self.config.get('max_tokens', 1024),
                streaming=self.config.get('streaming', True),
                verbose=self.config.get('verbose', True),
                extra_body=self.config.get('extra_body', {}),
                api_key=self.config.get('api_key', 'EMPTY')
            )
            logger.info(f"VLLM ChatModel initialized with model '{self.model_name}' at {base_url}")
            return chat_model
        except Exception as e:
            logger.error(f"Failed to initialize VLLM ChatModel: {e}")
            raise

class OllamaAgent(BaseAgent):
    """Agent using Ollama provider - returns ChatOllama instance."""
    
    def _create_chat_model(self) -> ChatOllama:
        try:
            chat_model = ChatOllama(
                model=self.model_name,
                base_url=self.config.get('base_url', 'http://localhost:11434'),
                temperature=self.config.get('temperature', 0.7),
                request_timeout=self.config.get('request_timeout', 30),
                reasoning=self.config.get('reasoning', False),
                num_thread=self.config.get('num_thread', 1),
                repeat_penalty=self.config.get('repeat_penalty', 1.1),
                mirostat_tau=self.config.get('mirostat_tau', 5.0),
                mirostat_eta=self.config.get('mirostat_eta', 0.1)
            )
            logger.info(f"Ollama ChatModel initialized with model '{self.model_name}'")
            return chat_model
        except Exception as e:
            logger.error(f"Failed to initialize Ollama ChatModel: {e}")
            raise

class OpenAIAgent(BaseAgent):
    """Agent using OpenAI provider - returns ChatOpenAI instance."""
    
    def _create_chat_model(self) -> ChatOpenAI:
        try:
            chat_model = ChatOpenAI(
                model=self.model_name,
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 1024),
                streaming=self.config.get('streaming', False),
                api_key=self.config.get('api_key', os.getenv('OPENAI_API_KEY'))
            )
            logger.info(f"OpenAI ChatModel initialized with model '{self.model_name}'")
            return chat_model
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI ChatModel: {e}")
            raise

class AgentFactory:
    """Factory class to create different types of agents with various LLM providers."""
    
    # Default configurations for different agent types
    AGENT_CONFIGS = {
        AgentType.STANDARD: {
            'temperature': 0.7,
            'max_tokens': 1024,
            'streaming': True
        },
        AgentType.REASONING: {
            'temperature': 0.7,
            'max_tokens': 2048,
            'streaming': True,
            'reasoning': True,
            'extra_body': {
                "enable_thinking": True,
                "thinking_budget": 50
            }
        },
        AgentType.CHAT: {
            'temperature': 0.8,
            'max_tokens': 512,
            'streaming': True
        },
        AgentType.ANALYSIS: {
            'temperature': 0.3,
            'max_tokens': 2048,
            'streaming': False
        }
    }
    
    @staticmethod
    def create_agent(
        provider: LLMProvider,
        model_name: str,
        agent_type: AgentType = AgentType.STANDARD,
        **custom_config
    ) -> BaseAgent:
        """
        Create an agent based on provider and type.
        
        Args:
            provider: The LLM provider to use
            model_name: The model name to use
            agent_type: The type of agent to create
            **custom_config: Custom configuration overrides
            
        Returns:
            BaseAgent: Initialized agent instance with ChatModel
        """
        # Get default config for agent type
        base_config = AgentFactory.AGENT_CONFIGS[agent_type].copy()
        
        # Override with custom config
        base_config.update(custom_config)
        
        # Create agent based on provider
        if provider == LLMProvider.VLLM:
            return VLLMAgent(model_name, provider, **base_config)
        elif provider == LLMProvider.OLLAMA:
            return OllamaAgent(model_name, provider, **base_config)
        elif provider == LLMProvider.OPENAI:
            return OpenAIAgent(model_name, provider, **base_config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def create_chat_model(
        provider: LLMProvider,
        model_name: str,
        agent_type: AgentType = AgentType.STANDARD,
        **custom_config
    ) -> BaseChatModel:
        """
        Create a ChatModel directly without agent wrapper.
        
        Args:
            provider: The LLM provider to use
            model_name: The model name to use
            agent_type: The type of agent to create
            **custom_config: Custom configuration overrides
            
        Returns:
            BaseChatModel: LangChain ChatModel instance
        """
        agent = AgentFactory.create_agent(provider, model_name, agent_type, **custom_config)
        return agent.get_chat_model()
    
    @staticmethod
    def create_from_settings(settings, agent_type: AgentType = AgentType.STANDARD) -> BaseAgent:
        """
        Create an agent from settings configuration.
        
        Args:
            settings: Settings object containing configuration
            agent_type: Type of agent to create
            
        Returns:
            BaseAgent: Initialized agent instance
        """
        provider_map = {
            "vllm": LLMProvider.VLLM,
            "ollama": LLMProvider.OLLAMA,
            "openai": LLMProvider.OPENAI
        }
        
        provider_str = getattr(settings, 'LLM_PROVIDER', 'ollama').lower()
        provider = provider_map.get(provider_str, LLMProvider.OLLAMA)
        
        if provider == LLMProvider.VLLM:
            model_name = getattr(settings, 'LLM_MODEL', 'default-model')
            base_url = getattr(settings, 'VLLM_API_URL', "http://0.0.0.0:6622/v1")
            api_key = getattr(settings, 'API_KEY', 'EMPTY')
            
            return AgentFactory.create_agent(
                provider=provider,
                model_name=model_name,
                agent_type=agent_type,
                base_url=base_url,
                api_key=api_key
            )
        
        elif provider == LLMProvider.OLLAMA:
            model_name = getattr(settings, 'OLLAMA_MODEL', 'llama2')
            base_url = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            
            return AgentFactory.create_agent(
                provider=provider,
                model_name=model_name,
                agent_type=agent_type,
                base_url=base_url
            )
        
        else:  # OpenAI
            model_name = getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo')
            api_key = getattr(settings, 'OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
            
            return AgentFactory.create_agent(
                provider=provider,
                model_name=model_name,
                agent_type=agent_type,
                api_key=api_key
            )
    
    @staticmethod
    def create_chat_model_from_settings(settings, agent_type: AgentType = AgentType.STANDARD) -> BaseChatModel:
        """
        Create a ChatModel directly from settings without agent wrapper.
        
        Args:
            settings: Settings object containing configuration
            agent_type: Type of agent to create
            
        Returns:
            BaseChatModel: LangChain ChatModel instance
        """
        agent = AgentFactory.create_from_settings(settings, agent_type)
        return agent.get_chat_model()

class AgentManager:
    """Manager class to handle multiple agents and their lifecycle."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.chat_models: Dict[str, BaseChatModel] = {}
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with a given name."""
        self.agents[name] = agent
        self.chat_models[name] = agent.get_chat_model()
        logger.info(f"Agent '{name}' registered: {agent}")
    
    def register_chat_model(self, name: str, chat_model: BaseChatModel):
        """Register a ChatModel directly with a given name."""
        self.chat_models[name] = chat_model
        logger.info(f"ChatModel '{name}' registered: {type(chat_model).__name__}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def get_chat_model(self, name: str) -> Optional[BaseChatModel]:
        """Get a ChatModel by name."""
        return self.chat_models.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agents.keys())
    
    def list_chat_models(self) -> List[str]:
        """List all registered ChatModel names."""
        return list(self.chat_models.keys())
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent by name."""
        removed = False
        if name in self.agents:
            del self.agents[name]
            removed = True
        if name in self.chat_models:
            del self.chat_models[name]
            removed = True
        if removed:
            logger.info(f"Agent/ChatModel '{name}' removed")
        return removed

# Utility functions
def load_config(file_path: str) -> Dict[str, Any]:

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise

def extract_clean_json(text: str) -> Optional[str]:
    """Extract and clean JSON from text response."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove thinking tags and markdown code blocks
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<think>.*', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'</think>', '', cleaned_text)
    cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)

    cleaned_text = cleaned_text.strip()
    brace_count = 0
    start_idx = -1
    end_idx = -1
    
    for i, char in enumerate(cleaned_text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i
                break
    
    if start_idx != -1 and end_idx != -1:
        json_candidate = cleaned_text[start_idx:end_idx + 1]
        
        try:
            json.loads(json_candidate)
            return json_candidate.strip()
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed_json = json_candidate
            
            fixed_json = re.sub(r'(\w+)(\s*:)', r'"\1"\2', fixed_json)
            fixed_json = re.sub(r',\s*}', '}', fixed_json)
            fixed_json = re.sub(r',\s*]', ']', fixed_json)
            fixed_json = re.sub(r'}\s*{', '},{', fixed_json)
            
            try:
                json.loads(fixed_json)
                return fixed_json.strip()
            except json.JSONDecodeError as e:
                logger.warning(f"Could not fix malformed JSON: {json_candidate}. Error: {e}")
    
    return ""

def cleanup_distributed():
    """Cleanup PyTorch distributed group on shutdown if initialized."""
    if torch.distributed.is_initialized():
        logger.info("Destroying PyTorch distributed process group")
        try:
            torch.distributed.destroy_process_group()
            logger.info("PyTorch distributed process group cleanup completed")
        except Exception as e:
            logger.error(f"Error during PyTorch cleanup: {e}")

# Register cleanup handler
atexit.register(cleanup_distributed)

# Example usage and initialization
def main():
    """Example usage of the agent factory system."""
    load_dotenv()
    settings = get_settings()
    
    # Initialize agent manager
    agent_manager = AgentManager()
    
    try:
        # Method 1: Create agents and get ChatModels from them
        standard_agent = AgentFactory.create_from_settings(settings, AgentType.STANDARD)
        reasoning_agent = AgentFactory.create_from_settings(settings, AgentType.REASONING)
        
        # Register agents (which also registers their ChatModels)
        agent_manager.register_agent("standard", standard_agent)
        agent_manager.register_agent("reasoning", reasoning_agent)
        
        # Method 2: Create ChatModels directly
        chat_model = AgentFactory.create_chat_model_from_settings(settings, AgentType.CHAT)
        agent_manager.register_chat_model("direct_chat", chat_model)
        
        # Method 3: Create custom ChatModel
        custom_chat_model = AgentFactory.create_chat_model(
            provider=LLMProvider.VLLM,
            model_name="custom-model",
            agent_type=AgentType.ANALYSIS,
            temperature=0.1,
            max_tokens=4096
        )
        agent_manager.register_chat_model("analysis", custom_chat_model)
        
        logger.info(f"Initialized agents: {agent_manager.list_agents()}")
        logger.info(f"Available ChatModels: {agent_manager.list_chat_models()}")
        
        # Example usage - Get ChatModel and use directly with LangChain
        chat_model = agent_manager.get_chat_model("standard")
        if chat_model:
            # Use directly with LangChain - invoke, stream, etc.
            from langchain_core.messages import HumanMessage
            response = chat_model.invoke([HumanMessage(content="Hello, how are you?")])
            logger.info(f"ChatModel response: {response.content}")
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise
def create_agents_from_config(config: Dict[str, Any]):
    """Create agents from a configuration dictionary."""
    
    agents = {}
    
    for agent_name, agent_config in config.get('agents', {}).items():
        provider_str = agent_config.get('provider', 'ollama').lower()
        provider_map = {
            'vllm': LLMProvider.VLLM,
            'ollama': LLMProvider.OLLAMA,
            'openai': LLMProvider.OPENAI
        }
        
        provider = provider_map.get(provider_str, LLMProvider.OLLAMA)
        agent_type_str = agent_config.get('type', 'standard').lower()
        agent_type_map = {
            'standard': AgentType.STANDARD,
            'reasoning': AgentType.REASONING,
            'chat': AgentType.CHAT,
            'analysis': AgentType.ANALYSIS
        }
        agent_type = agent_type_map.get(agent_type_str, AgentType.STANDARD)
        
        model_name = agent_config.get('model_name')
        
        # Extract other configuration
        extra_config = {k: v for k, v in agent_config.items() 
                       if k not in ['provider', 'type', 'model_name']}
        
        agents[agent_name] = AgentFactory.create_chat_model(
            provider=provider,
            model_name=model_name,
            agent_type=agent_type,
            **extra_config
        )
    
    return agents

def load_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
# Simplified global initialization for backward compatibility
def initialize_global_agents(settings):
    """Initialize global agent instances for backward compatibility."""
    global llm_instance, llm_reasoning
    
    try:
        # Create standard LLM instance
        standard_agent = AgentFactory.create_from_settings(settings, AgentType.STANDARD)
        llm_instance = standard_agent.get_chat_model()
        
        # Create reasoning LLM instance
        # Check if we should use reasoning-specific settings
        reasoning_provider_str = getattr(settings, 'REASONING_LLM_PROVIDER', settings.LLM_PROVIDER).lower()
        reasoning_provider_map = {
            "vllm": LLMProvider.VLLM,
            "ollama": LLMProvider.OLLAMA,
            "openai": LLMProvider.OPENAI
        }
        reasoning_provider = reasoning_provider_map.get(reasoning_provider_str, LLMProvider.OLLAMA)
        
        if reasoning_provider == LLMProvider.VLLM:
            reasoning_model = getattr(settings, 'LLM_REASONING_MODEL', settings.LLM_MODEL)
            reasoning_agent = AgentFactory.create_agent(
                provider=reasoning_provider,
                model_name=reasoning_model,
                agent_type=AgentType.REASONING,
                base_url=getattr(settings, 'VLLM_API_URL', "http://0.0.0.0:6622/v1"),
                api_key=getattr(settings, 'API_KEY', 'EMPTY')
            )
        else:  # Ollama
            reasoning_model = getattr(settings, 'LLM_REASONING_MODEL', settings.OLLAMA_MODEL)
            reasoning_agent = AgentFactory.create_agent(
                provider=reasoning_provider,
                model_name=reasoning_model,
                agent_type=AgentType.REASONING,
                base_url=getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            )
        
        llm_reasoning = reasoning_agent.get_chat_model()
        
        logger.info("Global agent instances initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize global agents: {e}")
        raise

    # main()
agent_cfg_path = "app/agents/config/agent.yaml"
agent_config = load_config(agent_cfg_path)
print("Loaded agent configuration:", agent_config)

agents = create_agents_from_config(agent_config)
# print("Created agents from config:", agents)
llm_reasoning = agents.get("llm_reasoning")
llm_instance = agents.get("llm_chat")
print("llm_reasoning:", llm_reasoning)
print("llm_chat:", llm_instance)

def cleanup_distributed():
    if torch.distributed.is_initialized():
        logger.info("Destroying PyTorch distributed process group")
        try:
            torch.distributed.destroy_process_group()
            logger.info("PyTorch distributed process group cleanup completed")
        except Exception as e:
            logger.error(f"Error during PyTorch cleanup: {e}")

# Register cleanup handler
import atexit
atexit.register(cleanup_distributed)
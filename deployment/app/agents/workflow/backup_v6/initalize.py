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
# --- LangChain Core & Community Imports ---
# from langchain_community.chat_models.ollama import ChatOllama
from langchain_ollama import ChatOllama
from langchain_community.llms.vllm import VLLM, VLLMOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import re
# --- Tool Imports ---
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
from app.core.config import get_settings
import yaml

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary.
    """
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

import os
# Use a path that's relative to the current file's location
current_dir = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(current_dir, "config_vie", "agent.yaml")
# # Fallback to the hardcoded path if the file doesn't exist
# if not os.path.exists(config_path):
#     config_path = "/home/datpd1/genstory/multi-agent-app/agentic-gst-chatbot/backend/app/agents/config_vie/agent.yaml"

config_path ="/home/datpd1/genstory/agentic-genestory-platform/agentic-gst-chatbot/backend/app/agents/config/vie/agent.yaml"
agent_config = load_config(config_path)

# logger.add("logs/agentic_GST_Chatbot.log", rotation="4 MB", retention="5 days", level="DEBUG")
# logger.info("Starting (LangChain/LangGraph Style)")

settings = get_settings()
load_dotenv()
def extract_clean_json(text: str) -> Optional[str]:
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
    
    # Strategy 2: Try to find JSON even if it starts after some text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
    if json_match:
        json_candidate = json_match.group(0).strip()
        try:
            json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            pass
    json_pattern = re.search(r'\{[\s\S]*?"rewritten_query"[\s\S]*?\}', cleaned_text)
    if json_pattern:
        json_candidate = json_pattern.group(0).strip()
        try:
            json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            if not json_candidate.endswith('}'):
                json_candidate += '}'
                try:
                    json.loads(json_candidate)
                    return json_candidate
                except json.JSONDecodeError:
                    pass
    
    if '"rewritten_query"' in cleaned_text and not cleaned_text.strip().startswith('{'):
        # Try to reconstruct the JSON by adding opening brace
        json_candidate = '{' + cleaned_text.strip()
        if not json_candidate.endswith('}'):
            json_candidate += '}'
        
        try:
            json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            fixed_json = json_candidate
            fixed_json = re.sub(r'(\w+)(\s*:)', r'"\1"\2', fixed_json)
            fixed_json = re.sub(r',\s*}', '}', fixed_json)
            fixed_json = re.sub(r',\s*]', ']', fixed_json)
            
            try:
                json.loads(fixed_json)
                return fixed_json
            except json.JSONDecodeError:
                pass
    logger.warning(f"Could not extract valid JSON from text: {cleaned_text[:500]}...")
    return ""

# Initialize LLM based on settings
llm_provider = settings.LLM_PROVIDER.lower()  # Get from settings

try:
    if llm_provider == "vllm":
        # Use VLLM API integration
        base_url = settings.VLLM_API_URL if hasattr(settings, 'VLLM_API_URL') else "http://0.0.0.0:6622/v1"
        # base_url = settings.VLLM_API_URL if hasattr(settings, 'VLLM_API_URL') else "http://0.0.0.0:8000/v1" # user server vast ai
        
        # Make sure the base_url ends with /v1
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        os.environ["OPENAI_API_KEY"] = "EMPTY"  # Required for VLLMOpenAI
        
        llm_instance = ChatOpenAI(
            model=settings.LLM_MODEL,
            base_url=base_url,
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,  # Set maximum tokens
            streaming=True,   # Enable streaming for better performance
            verbose=True,
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 50
                },
            api_key=settings.API_KEY  # Verbose logging for debugging
        )
        logger.info(f"Global LLM initialized using VLLM API integration with model '{settings.LLM_MODEL}' at {base_url}")
        
        # Create a prompt template for use with this LLM if needed
        
    else:
        # Default to Ollama
        llm_instance = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL, 
            temperature=0.7,  # Reduced for faster responses
            request_timeout=30,  # Add timeout
            reasoning=False
        )
        logger.info(f"Global LLM '{settings.OLLAMA_MODEL}' loaded from '{settings.OLLAMA_BASE_URL}'.")
except Exception as e:
    logger.error(f"Failed to load global LLM with provider '{llm_provider}': {e}", exc_info=True)
    raise



# Initialize Reasoning LLM based on settings
reasoning_provider = settings.REASONING_LLM_PROVIDER.lower()  # Get from settings

try:
    if reasoning_provider == "vllm":
        # Use VLLM API integration for reasoning
        base_url = settings.VLLM_API_URL if hasattr(settings, 'VLLM_API_URL') else "http://0.0.0.0:6622/v1"
        # Make sure the base_url ends with /v1
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        os.environ["OPENAI_API_KEY"] = "EMPTY"  # Required for VLLMOpenAI
        
        # Use ChatOpenAI instead of VLLMOpenAI for better serialization support
        llm_reasoning = ChatOpenAI(
            model=settings.LLM_REASONING_MODEL,
            base_url=base_url,
            temperature=0.7,
            top_p=0.8,      # Slightly lower top_p for more focused reasoning
            max_tokens=512, # Larger context for reasoning
            streaming=True, # Enable streaming for better handling of responses
            verbose=True    # Verbose logging for debugging
        )
        logger.info(f"Reasoning LLM initialized using VLLM API integration with model '{settings.LLM_REASONING_MODEL}' at {base_url}")
        
        # # Create a prompt template for reasoning
        # reasoning_prompt_template = PromptTemplate(
        #     input_variables=["input"],
        #     template="{input}\n\nPlease reason step by step:"
        # )
        
        # # Create an LLM chain for reasoning
        # reasoning_chain = LLMChain(llm=llm_reasoning, prompt=reasoning_prompt_template)
    else:
        # Default to Ollama for reasoning
        llm_reasoning = ChatOllama(
            model=settings.LLM_REASONING_MODEL,
            base_url=settings.OLLAMA_BASE_URL, 
            temperature=0.7,  # Reduced for faster responses
            reasoning=True,  # Enable reasoning extraction
            num_thread= 2,  # Use dual thread for reasoning
            repeat_penalty = 1.3,  # Increase penalty for repeated tokens
            mirostat_tau= 3.0,  # Mirostat parameter for adaptive sampling
            mirostat_eta= 0.1,  # Mirostat parameter for adaptive sampling
        )
        logger.info(f"Reasoning LLM '{settings.LLM_REASONING_MODEL}' loaded from '{settings.OLLAMA_BASE_URL}'.")
except Exception as e:
    logger.error(f"Failed to load reasoning LLM with provider '{reasoning_provider}': {e}", exc_info=True)
    raise






# Cleanup PyTorch distributed group on shutdown if initialized
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

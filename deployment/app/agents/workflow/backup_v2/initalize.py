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
# --- LangChain Core & Community Imports ---
# from langchain_community.chat_models.ollama import ChatOllama
from langchain_ollama import ChatOllama
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
config_path = os.path.join(current_dir, "config_vie", "agent.yaml")
# Fallback to the hardcoded path if the file doesn't exist
if not os.path.exists(config_path):
    config_path = "/home/datpd1/genstory/multi-agent-app/agentic-gst-chatbot/backend/app/agents/config_vie/agent.yaml"

config_path ="/home/datpd1/genstory/multi-agent-app/agentic-gst-chatbot/backend/app/agents/config/vie/agent.yaml"
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

try:
    llm_instance = ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL, 
        temperature=0.7,  # Reduced for faster responses
        request_timeout=30,  # Add timeout
        
    )
    logger.info(f"Global LLM '{settings.LLM_MODEL}' loaded from '{settings.OLLAMA_BASE_URL}'.")
except Exception as e:
    logger.error(f"Failed to load global LLM '{settings.LLM_MODEL}' from '{settings.OLLAMA_BASE_URL}': {e}", exc_info=True)
    raise

import sys
import json
import time
import chromadb
from typing import Optional, TypedDict, Literal, List, Tuple, Dict, Any, Callable
from loguru import logger
import asyncio
from typing import List






class GraphState(TypedDict):
    session_id: Optional[str]
    original_query: str
    employee_id: Optional[int]
    guest_id: Optional[str]
    customer_id: Optional[str]
    user_role: str  # Added user role field
    rewritten_query: str
    intents: str
    contexts: Optional[List[Dict[str, Any]]]
    classified_agent: Literal["CompanyAgent", "CustomerAgent", "MedicalAgent", "ProductAgent", "DrugAgent", "NaiveAgent", "GeneticAgent", "QuestionGeneratorAgent", "RewriterAgent", "VisualAgent", "EmployeeAgent", "GuestAgent"]
    task_assigned: List[Literal["searchweb", "retrieve", "chitchat", "summary"]] # Changed to List[Literal[...]]
    agent_response: str
    agent_thinks: Dict[str, Any]
    reflection_feedback: str
    is_final_answer: bool
    error_message: Optional[str]
    needs_rewrite: bool
    retry_count: int
    suggested_questions: List[str]
    chat_history: List[Tuple[str, str]]
    suggest_agent_followups: Literal["CompanyAgent", "CustomerAgent", "MedicalAgent", "ProductAgent", "DrugAgent", "NaiveAgent", "GeneticAgent", "QuestionGeneratorAgent", "RewriterAgent", "VisualAgent", "SummaryAgent", "EmployeeAgent", "GuestAgent"]
    # Added fields for context summarization
    quality_score: Optional[float]  # Score from reflection on response quality
    tool_summaries: Optional[Dict[str, str]]  # Tool name to summarized output mapping
    context_summaries: Optional[Dict[str, str]]  # Context type to summary mapping
    visual_context: Optional[Dict[str, Any]]  # For visual data and chart information
    entity_summaries: Optional[Dict[str, Any]]  # For entity-specific summarized information
    tool_usage_history: Optional[List[Dict[str, Any]]]  # Track tool usage across conversation
    combined_context: Optional[str]  # Synthesized context from multiple sources
    image_path: Optional[str]  # Path to any image used in the conversation
logger.info("====== LangGraph State Defined. =======")
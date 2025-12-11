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
    customer_role: Optional[str]  # Added field for customer role
    interaction_id: Optional[str]  # Added field for interaction tracking
    timestamp: Optional[str]  # Added field for timestamp tracking
    user_role: str  # Added user role field
    rewritten_query: str
    intents: str
    contexts: Optional[List[Dict[str, Any]]]
    classified_agent: Literal["CompanyAgent", "CustomerAgent", "MedicalAgent", "ProductAgent", "DrugAgent", "NaiveAgent", "GeneticAgent", "QuestionGeneratorAgent", "RewriterAgent", "VisualAgent", "EmployeeAgent", "GuestAgent", "DirectAnswerAgent", "FallbackAgent", "SynthesizerAgent", "FinalAnswerAgent"]
    next_step: Optional[str]  # Added field for routing decisions
    is_multi_step: Optional[bool]  # Added field for multi-step workflows
    should_re_execute: Optional[bool]  # Added field for re-execution logic
    clarification_question: Optional[str]  # Added field for clarification questions
    task_assigned: List[Literal["searchweb", "retrieve", "chitchat", "summary"]] # Changed to List[Literal[...]]
    agent_response: str
    agent_thinks: Dict[str, Any]
    reflection_feedback: str
    is_final_answer: bool
    error_message: Optional[str]
    needs_rewrite: bool
    retry_count: int
    iteration_count: Optional[int]  # Added field for iteration tracking
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
    golden_context: Optional[str]  # Synthesized context from multiple sources
    image_path: Optional[str]  # Path to any image used in the conversation
    needs_re_execution: bool = False
    sentiment_analysis: Dict[str, Any] = {}
    is_re_execution: bool = False
    was_re_executed: bool = False
logger.info("====== LangGraph State Defined. =======")
import hashlib
import json
import os
import uuid
from typing import Any, Optional, List, Dict
from datetime import datetime

import redis.asyncio as redis
import asyncio
from loguru import logger

# Import HistoryCache for guest chat history management
from app.agents.stores.history_cache import HistoryCache, ChatMessage, get_history_cache

# Import LLM for intelligent summarization
try:
    from app.agents.workflow.initalize import llm_instance
    LLM_AVAILABLE = True
except ImportError:
    logger.warning("LLM not available, falling back to simple summarization")
    LLM_AVAILABLE = False

class CacheManager:
    """
    An asynchronous cache manager using Redis for storing and retrieving AI workflow results.
    Implements an "Exact Cache" strategy.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure __init__ is run only once for the singleton
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.redis_pool = None
        try:
            # Use environment variables for configuration (best practice)
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            self.redis_pool = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            logger.info(f"CacheManager initialized and connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Could not connect to Redis. Caching will be disabled. Error: {e}")
            self.redis_pool = None
            
        # Initialize the guest history cache with persistence enabled
        enable_persistence = os.environ.get("ENABLE_HISTORY_PERSISTENCE", "true").lower() == "true"
        persistence_path = os.environ.get("HISTORY_PERSISTENCE_PATH", "app/cache/guest_history")
        
        self.history_cache = get_history_cache(
            max_sessions=2000,  # Support more guest sessions
            session_ttl_hours=24,  # Keep guest history for 24 hours
            max_messages_per_session=100,  # Up to 100 messages per session
            enable_persistence=enable_persistence,  # Enable file-based persistence
            persistence_path=persistence_path
        )
        logger.info(f"Guest history cache initialized with persistence={enable_persistence}")
        
        # Create a lock for async operations
        self._lock = asyncio.Lock()

    def is_active(self) -> bool:
        """Check if the cache is connected and active."""
        return self.redis_pool is not None

    def create_cache_key(self, query: str, chat_history: Optional[List[Dict]] = None, context: Optional[Dict] = None) -> str:
        """
        Creates a stable, hash-based key from the core request components.
        We exclude session_id and guest_id to maximize cache hits across users for common questions.
        """
        # Create a dictionary of the data that defines the request's uniqueness
        payload = {
            "query": query,
            # Sort chat history to ensure consistent serialization
            # "chat_history": sorted(chat_history, key=lambda x: x.get('type', '')) if chat_history else []
        }
        
        # Serialize the dictionary to a string. `sort_keys=True` is critical.
        serialized_payload = json.dumps(payload, sort_keys=True)
        
        # Use SHA-256 to create a unique and fixed-length key
        hash_object = hashlib.sha256(serialized_payload.encode('utf-8'))
        
        # Prepend with a namespace for clarity in Redis
        # Determine workflow type from query prefix if present
        if context == "employee_workflow":
            namespace = "employee_workflow"
        elif context == "customer_workflow":
            namespace = "customer_workflow"
        else:
            namespace = "guest_workflow"
        
        return f"cache:{namespace}:{hash_object.hexdigest()}"

    def create_session_cache_key(self, session_id: str, key_type: str = "summary") -> str:
        """
        Creates a cache key for session-specific data like summaries and context.
        
        Args:
            session_id (str): Session identifier
            key_type (str): Type of cached data ('summary', 'context', 'metadata')
            
        Returns:
            str: Session-specific cache key
        """
        return f"session:{key_type}:{session_id}"

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Asynchronously get a value from the cache."""
        if not self.is_active():
            return None
        try:
            cached_data = await self.redis_pool.get(key)
            if cached_data:
                logger.success(f"CACHE HIT for key: {key}")
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """
        Asynchronously set a value in the cache with a Time-To-Live (TTL).
        
        Args:
            key (str): The cache key.
            value (Dict[str, Any]): The dictionary object to cache.
            ttl (int): Time-to-live in seconds. Default is 1 hour.
        """
        if not self.is_active():
            return
        try:
            serialized_value = json.dumps(value)
            await self.redis_pool.set(key, serialized_value, ex=ttl)
            logger.info(f"CACHE SET for key: {key} with TTL: {ttl}s")
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            
    # ===== Session Memory Management Methods =====
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a fast summary of the chat session for context.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            Dict containing session summary or None if not found
        """
        cache_key = self.create_session_cache_key(session_id, "summary")
        
        # Try to get from Redis cache first
        cached_summary = await self.get(cache_key)
        if cached_summary:
            logger.info(f"Retrieved cached session summary for {session_id}")
            return cached_summary
        
        # Generate summary from history cache
        chat_history = await self.get_chat_history(session_id, limit=50)
        if not chat_history:
            return None
            
        summary = await self._generate_session_summary(chat_history, session_id)
        
        # Cache the summary for 30 minutes
        if summary:
            await self.set(cache_key, summary, ttl=1800)
            
        return summary
    
    async def get_session_context(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Get relevant context from session history for the current query.
        
        Args:
            session_id (str): Session identifier
            query (str): Current user query
            
        Returns:
            Dict containing relevant context from previous conversations
        """
        cache_key = self.create_session_cache_key(session_id, f"context_{hashlib.md5(query.encode()).hexdigest()[:8]}")
        
        # Try cache first
        cached_context = await self.get(cache_key)
        if cached_context:
            logger.info(f"Retrieved cached session context for query: {query[:50]}...")
            return cached_context
        
        # Generate context from history
        context = await self._extract_relevant_context(session_id, query)
        
        # Cache context for 15 minutes
        if context:
            await self.set(cache_key, context, ttl=900)
            
        return context
    
    async def update_session_memory(self, session_id: str, user_message: str, assistant_response: str) -> bool:
        """
        Update session memory with new conversation turn and refresh summaries.
        
        Args:
            session_id (str): Session identifier
            user_message (str): User's message
            assistant_response (str): Assistant's response
            
        Returns:
            bool: True if memory was updated successfully
        """
        # Add to chat history
        success = await self.add_conversation_turn(
            session_id, user_message, assistant_response
        )
        
        if success:
            # Invalidate cached summaries to force regeneration
            await self._invalidate_session_cache(session_id)
            logger.info(f"Updated session memory and invalidated cache for {session_id}")
            
        return success
    
    async def get_session_insights(self, session_id: str) -> Dict[str, Any]:
        """
        Get analytical insights about the session for better context understanding.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            Dict containing session insights and patterns
        """
        cache_key = self.create_session_cache_key(session_id, "insights")
        
        # Try cache first
        cached_insights = await self.get(cache_key)
        if cached_insights:
            return cached_insights
        
        # Generate insights
        insights = await self._analyze_session_patterns(session_id)
        
        # Cache for 1 hour
        if insights:
            await self.set(cache_key, insights, ttl=3600)
            
        return insights
    
    async def _generate_session_summary(self, chat_history: List[Dict], session_id: str) -> Dict[str, Any]:
        """
        Generate an intelligent summary of the chat session using LLM when available.
        Falls back to simple analysis if LLM is not available.
        """
        if not chat_history:
            return {}
        
        # Basic metrics
        total_messages = len(chat_history)
        user_messages = [msg for msg in chat_history if msg.get('role') == 'user']
        assistant_messages = [msg for msg in chat_history if msg.get('role') == 'assistant']
        
        # Calculate session duration
        duration = "0 minutes"
        if chat_history:
            start_time = chat_history[0].get('timestamp', '')
            end_time = chat_history[-1].get('timestamp', '')
            duration = self._calculate_duration(start_time, end_time)
        
        # Use LLM for intelligent summarization if available
        if LLM_AVAILABLE and total_messages > 2:
            try:
                llm_summary = await self._generate_llm_summary(chat_history)
                
                # Combine LLM insights with basic metrics
                summary = {
                    "session_id": session_id,
                    "total_messages": total_messages,
                    "user_messages_count": len(user_messages),
                    "assistant_messages_count": len(assistant_messages),
                    "session_duration": duration,
                    "last_updated": datetime.utcnow().isoformat(),
                    
                    # LLM-enhanced fields
                    "intelligent_summary": llm_summary.get("summary", ""),
                    "key_topics": llm_summary.get("topics", []),
                    "user_intent": llm_summary.get("user_intent", ""),
                    "conversation_stage": llm_summary.get("conversation_stage", ""),
                    "user_sentiment": llm_summary.get("sentiment", "neutral"),
                    "key_insights": llm_summary.get("insights", []),
                    "next_likely_questions": llm_summary.get("next_questions", []),
                    "domain_focus": llm_summary.get("domain", "general"),
                    
                    # Fallback to simple analysis for compatibility
                    "recent_topics": llm_summary.get("topics", [])[:5],
                    "key_themes": llm_summary.get("themes", []),
                    "conversation_flow": llm_summary.get("flow", "linear")
                }
                
                logger.info(f"Generated LLM-based summary for session {session_id}")
                return summary
                
            except Exception as llm_error:
                logger.warning(f"LLM summarization failed, using fallback: {llm_error}")
        
        # Fallback to simple analysis
        recent_messages = chat_history[-10:] if len(chat_history) > 10 else chat_history
        topics = self._extract_topics(recent_messages)
        
        summary = {
            "session_id": session_id,
            "total_messages": total_messages,
            "user_messages_count": len(user_messages),
            "assistant_messages_count": len(assistant_messages),
            "recent_topics": topics,
            "session_duration": duration,
            "last_updated": datetime.utcnow().isoformat(),
            "key_themes": self._identify_key_themes(chat_history),
            "conversation_flow": self._analyze_conversation_flow(chat_history),
            
            # Default values for LLM fields
            "intelligent_summary": f"Conversation with {len(user_messages)} user messages covering topics: {', '.join(topics[:3])}",
            "user_intent": "information_seeking",
            "conversation_stage": "ongoing" if total_messages > 6 else "initial",
            "user_sentiment": "neutral",
            "domain_focus": self._guess_domain_simple(chat_history)
        }
        
        return summary

    async def _generate_llm_summary(self, chat_history: List[Dict]) -> Dict[str, Any]:
        """
        Use LLM to generate intelligent conversation summary and insights.
        """
        # Format conversation for LLM analysis
        conversation_text = self._format_conversation_for_llm(chat_history)
        
        # Create a comprehensive prompt for conversation analysis
        prompt = f"""
Analyze this guest conversation and provide a comprehensive summary with insights:

=== CONVERSATION ===
{conversation_text}

=== ANALYSIS REQUIRED ===
Please provide a JSON response with the following structure:
{{
    "summary": "A 2-3 sentence summary of the conversation",
    "topics": ["list", "of", "main", "topics", "discussed"],
    "user_intent": "primary user goal (e.g., information_seeking, product_interest, support_request)",
    "conversation_stage": "stage of conversation (initial, exploring, deciding, concluding)",
    "sentiment": "user sentiment (positive, neutral, negative, concerned, excited)",
    "insights": ["key", "insights", "about", "user", "needs"],
    "next_questions": ["likely", "follow-up", "questions", "user", "might", "ask"],
    "domain": "primary domain (genetics, health, product, company, pricing)",
    "themes": ["broader", "themes", "in", "conversation"],
    "flow": "conversation pattern (linear, exploratory, focused, scattered)",
    "user_knowledge_level": "beginner, intermediate, or advanced",
    "pain_points": ["any", "concerns", "or", "obstacles", "mentioned"],
    "opportunities": ["opportunities", "to", "help", "or", "upsell"]
}}

Focus on being accurate and concise. If the conversation is short, adjust the depth accordingly.
"""

        try:
            # Use the LLM to analyze the conversation
            response = await llm_instance.ainvoke(prompt)
            
            # Parse the LLM response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
                
                # Validate and clean the analysis
                analysis = self._validate_llm_analysis(analysis)
                
                return analysis
            else:
                raise ValueError("No valid JSON found in LLM response")
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Return a basic analysis if LLM fails
            return {
                "summary": "Conversation analysis failed, using basic summary",
                "topics": self._extract_topics(chat_history),
                "user_intent": "information_seeking",
                "sentiment": "neutral",
                "domain": "general"
            }

    def _format_conversation_for_llm(self, chat_history: List[Dict]) -> str:
        """Format conversation history for LLM analysis."""
        formatted = []
        for i, msg in enumerate(chat_history[-20:], 1):  # Last 20 messages
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')[:500]  # Limit message length
            timestamp = msg.get('timestamp', '')[:19]  # Remove microseconds
            formatted.append(f"{i}. [{timestamp}] {role}: {content}")
        
        return "\n".join(formatted)

    def _validate_llm_analysis(self, analysis: Dict) -> Dict:
        """Validate and clean LLM analysis results."""
        # Ensure required fields exist with defaults
        defaults = {
            "summary": "Conversation summary not available",
            "topics": [],
            "user_intent": "information_seeking",
            "conversation_stage": "ongoing",
            "sentiment": "neutral",
            "insights": [],
            "next_questions": [],
            "domain": "general",
            "themes": [],
            "flow": "linear"
        }
        
        # Fill in missing fields
        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
        
        # Limit list lengths to prevent overly long responses
        for list_field in ["topics", "insights", "next_questions", "themes"]:
            if isinstance(analysis.get(list_field), list):
                analysis[list_field] = analysis[list_field][:5]  # Max 5 items
        
        # Ensure strings are reasonable length
        if isinstance(analysis.get("summary"), str):
            analysis["summary"] = analysis["summary"][:500]  # Max 500 chars
        
        return analysis

    def _guess_domain_simple(self, chat_history: List[Dict]) -> str:
        """Simple domain guessing for fallback."""
        all_text = " ".join([msg.get('content', '').lower() for msg in chat_history])
        
        domain_keywords = {
            'genetics': ['gene', 'genetic', 'dna', 'hereditary', 'mutation'],
            'health': ['health', 'medical', 'disease', 'symptom', 'treatment'],
            'product': ['product', 'test', 'kit', 'order', 'buy', 'price'],
            'company': ['company', 'about', 'contact', 'policy']
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            scores[domain] = sum(1 for keyword in keywords if keyword in all_text)
        
        return max(scores, key=scores.get) if scores else 'general'
    
    async def _extract_relevant_context(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Extract relevant context from session history based on current query.
        Uses LLM for semantic relevance when available, falls back to keyword matching.
        """
        chat_history = await self.get_chat_history(session_id, limit=20)
        if not chat_history:
            return {}
        
        # Always get recent context for immediate conversation flow
        recent_context = chat_history[-3:] if len(chat_history) >= 3 else chat_history
        
        # Use LLM for intelligent context extraction if available
        if LLM_AVAILABLE and len(chat_history) > 3:
            try:
                llm_context = await self._extract_llm_context(chat_history, query)
                
                context = {
                    "relevant_messages": llm_context.get("relevant_messages", []),
                    "recent_context": recent_context,
                    "query_keywords": llm_context.get("query_keywords", []),
                    "context_score": llm_context.get("relevance_score", 0.0),
                    "semantic_analysis": llm_context.get("semantic_analysis", {}),
                    "extracted_at": datetime.utcnow().isoformat(),
                    "method": "llm_enhanced"
                }
                
                logger.info(f"LLM-based context extraction for query: {query[:50]}...")
                return context
                
            except Exception as llm_error:
                logger.warning(f"LLM context extraction failed, using keyword fallback: {llm_error}")
        
        # Fallback to keyword matching
        query_lower = query.lower()
        relevant_messages = []
        
        # Find messages related to current query
        for msg in chat_history:
            content = msg.get('content', '').lower()
            # Simple keyword matching
            if any(word in content for word in query_lower.split() if len(word) > 3):
                relevant_messages.append(msg)
        
        context = {
            "relevant_messages": relevant_messages[-5:],  # Last 5 relevant messages
            "recent_context": recent_context,
            "query_keywords": [word for word in query_lower.split() if len(word) > 3],
            "context_score": len(relevant_messages) / len(chat_history) if chat_history else 0,
            "extracted_at": datetime.utcnow().isoformat(),
            "method": "keyword_matching"
        }
        
        return context

    async def _extract_llm_context(self, chat_history: List[Dict], query: str) -> Dict[str, Any]:
        """
        Use LLM to intelligently extract relevant context from conversation history.
        """
        # Format conversation for LLM
        conversation_text = self._format_conversation_for_llm(chat_history)
        
        prompt = f"""
Analyze this conversation history and identify which parts are most relevant to the user's current query.

=== CONVERSATION HISTORY ===
{conversation_text}

=== CURRENT USER QUERY ===
{query}

=== ANALYSIS REQUIRED ===
Please provide a JSON response with:
{{
    "relevant_messages": [
        {{
            "message_number": 1,
            "relevance_score": 0.85,
            "reason": "why this message is relevant",
            "content": "message content"
        }}
    ],
    "query_keywords": ["key", "concepts", "from", "query"],
    "semantic_analysis": {{
        "query_intent": "what the user is trying to accomplish",
        "information_needed": "what specific information they need",
        "context_connection": "how this relates to previous conversation"
    }},
    "relevance_score": 0.75,
    "confidence": "high|medium|low"
}}

Focus on messages that:
1. Directly relate to the query topic
2. Provide background context for the query
3. Show the user's journey leading to this question
4. Contain information that might help answer the query

Be selective - only include truly relevant messages (score > 0.6).
"""

        try:
            response = await llm_instance.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                context_analysis = json.loads(json_str)
                
                # Extract the actual message content for relevant messages
                relevant_messages = []
                for rel_msg in context_analysis.get("relevant_messages", []):
                    msg_num = rel_msg.get("message_number", 0)
                    if 1 <= msg_num <= len(chat_history):
                        original_msg = chat_history[msg_num - 1]
                        relevant_messages.append({
                            "message": original_msg,
                            "relevance_score": rel_msg.get("relevance_score", 0.5),
                            "reason": rel_msg.get("reason", "")
                        })
                
                context_analysis["relevant_messages"] = relevant_messages
                return self._validate_context_analysis(context_analysis)
            else:
                raise ValueError("No valid JSON in LLM response")
                
        except Exception as e:
            logger.error(f"LLM context extraction error: {e}")
            return {
                "relevant_messages": [],
                "query_keywords": query.lower().split(),
                "relevance_score": 0.0,
                "semantic_analysis": {}
            }

    def _validate_context_analysis(self, analysis: Dict) -> Dict:
        """Validate LLM context analysis results."""
        defaults = {
            "relevant_messages": [],
            "query_keywords": [],
            "semantic_analysis": {},
            "relevance_score": 0.0,
            "confidence": "low"
        }
        
        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
        
        # Ensure relevance score is valid
        try:
            analysis["relevance_score"] = max(0.0, min(1.0, float(analysis["relevance_score"])))
        except:
            analysis["relevance_score"] = 0.0
        
        # Limit relevant messages to top 5
        if isinstance(analysis.get("relevant_messages"), list):
            analysis["relevant_messages"] = analysis["relevant_messages"][:5]
        
        return analysis
    
    async def _analyze_session_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze patterns in the session for insights.
        """
        chat_history = await self.get_chat_history(session_id)
        if not chat_history:
            return {}
        
        # Analyze message patterns
        message_lengths = [len(msg.get('content', '')) for msg in chat_history]
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        
        # Identify question patterns
        questions = [msg for msg in chat_history if msg.get('role') == 'user' and '?' in msg.get('content', '')]
        
        # Extract domain focus
        domain_keywords = {
            'medical': ['health', 'medical', 'disease', 'symptom', 'treatment', 'doctor'],
            'genetic': ['gene', 'DNA', 'genetic', 'hereditary', 'mutation', 'genome'],
            'product': ['product', 'test', 'kit', 'price', 'order', 'buy'],
            'company': ['company', 'about', 'contact', 'service', 'policy']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = 0
            for msg in chat_history:
                content = msg.get('content', '').lower()
                score += sum(1 for keyword in keywords if keyword in content)
            domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        
        insights = {
            "session_id": session_id,
            "total_messages": len(chat_history),
            "avg_message_length": round(avg_message_length, 2),
            "question_count": len(questions),
            "primary_domain": primary_domain,
            "domain_scores": domain_scores,
            "conversation_depth": "deep" if len(chat_history) > 10 else "shallow",
            "user_engagement": "high" if len(questions) > 3 else "moderate" if len(questions) > 1 else "low",
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        return insights
    
    def _extract_topics(self, messages: List[Dict]) -> List[str]:
        """Extract main topics from recent messages."""
        topics = set()
        for msg in messages:
            content = msg.get('content', '').lower()
            # Simple topic extraction - can be enhanced with NLP
            words = content.split()
            # Extract potential topics (nouns, important keywords)
            for word in words:
                if len(word) > 4 and word.isalpha():
                    topics.add(word)
        return list(topics)[:10]  # Top 10 topics
    
    def _identify_key_themes(self, chat_history: List[Dict]) -> List[str]:
        """Identify key themes across the entire conversation."""
        themes = []
        all_content = ' '.join([msg.get('content', '') for msg in chat_history]).lower()
        
        # Define theme patterns
        theme_patterns = {
            'health_concern': ['worried', 'concern', 'risk', 'afraid', 'anxiety'],
            'information_seeking': ['what', 'how', 'when', 'where', 'why', 'explain'],
            'product_interest': ['price', 'cost', 'buy', 'order', 'purchase'],
            'technical_detail': ['process', 'procedure', 'method', 'technique'],
            'comparison': ['compare', 'difference', 'better', 'versus', 'vs']
        }
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in all_content for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _analyze_conversation_flow(self, chat_history: List[Dict]) -> str:
        """Analyze the flow pattern of the conversation."""
        if len(chat_history) < 4:
            return "brief"
        elif len(chat_history) < 10:
            return "focused"
        else:
            # Check for follow-up patterns
            user_messages = [msg for msg in chat_history if msg.get('role') == 'user']
            follow_ups = sum(1 for msg in user_messages if any(word in msg.get('content', '').lower() 
                            for word in ['also', 'additionally', 'furthermore', 'and']))
            
            if follow_ups > 2:
                return "exploratory"
            else:
                return "linear"
    
    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """Calculate session duration."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration = end - start
            minutes = int(duration.total_seconds() / 60)
            return f"{minutes} minutes"
        except:
            return "unknown"
    
    async def _invalidate_session_cache(self, session_id: str):
        """Invalidate all cached data for a session."""
        if not self.is_active():
            return
            
        try:
            # Remove summary, context, and insights caches
            cache_patterns = [
                self.create_session_cache_key(session_id, "summary"),
                self.create_session_cache_key(session_id, "insights")
            ]
            
            for pattern in cache_patterns:
                await self.redis_pool.delete(pattern)
            
            # Remove context caches (they have dynamic keys)
            # In a production environment, you might want to use SCAN to find and delete pattern-matched keys
            logger.debug(f"Invalidated session cache for {session_id}")
            
        except Exception as e:
            logger.error(f"Error invalidating session cache: {e}")
            
    # ===== Guest Chat History Methods =====
    
    async def create_guest_session(self, guest_id: Optional[str] = None, session_id: Optional[str] = None) -> tuple:
        """
        Create a new guest chat session.
        
        Args:
            guest_id (str, optional): Guest identifier. If None, a new one is generated.
            session_id (str, optional): Session identifier. If None, a new one is generated.
            
        Returns:
            tuple: (guest_id, session_id) of the created session
        """
        if not guest_id:
            guest_id = f"guest_{uuid.uuid4().hex[:10]}"
            
        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex}"
            
        # Create metadata to track session information
        metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "client_info": {
                "user_agent": "unknown",
                "ip_address": "unknown"
            }
        }
        
        # Create session in history cache
        guest_id, session_id = await self.history_cache.create_session(
            guest_id=guest_id,
            session_id=session_id,
            metadata=metadata
        )
        
        logger.info(f"Created new guest session: {session_id} for guest: {guest_id}")
        return guest_id, session_id
    
    async def add_chat_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a single message to the guest chat history.
        
        Args:
            session_id (str): Session identifier
            role (str): Message role ('user' or 'assistant')
            content (str): Message content
            metadata (dict, optional): Additional message metadata
            
        Returns:
            bool: True if message was added successfully
        """
        if not metadata:
            metadata = {}
            
        # Add timestamp to metadata
        metadata["timestamp"] = datetime.utcnow().isoformat()
        
        # Add message to history cache
        success = await self.history_cache.add_message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata
        )
        
        return success
    
    async def add_conversation_turn(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str,
        user_metadata: Optional[Dict] = None, 
        assistant_metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a complete conversation turn (user message + assistant response) to history.
        
        Args:
            session_id (str): Session identifier
            user_message (str): User's message
            assistant_response (str): Assistant's response
            user_metadata (dict, optional): Metadata for user message
            assistant_metadata (dict, optional): Metadata for assistant response
            
        Returns:
            bool: True if conversation turn was added successfully
        """
        return await self.history_cache.add_conversation_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata
        )
    
    async def get_chat_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        format_for_workflow: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a guest session.
        
        Args:
            session_id (str): Session identifier
            limit (int, optional): Maximum number of messages to retrieve
            format_for_workflow (bool): Format messages for workflow compatibility
            
        Returns:
            List[Dict]: Chat history messages
        """
        return await self.history_cache.get_chat_history(
            session_id=session_id,
            limit=limit,
            format_for_workflow=format_for_workflow
        )
    
    async def get_session(self, session_id: str):
        """
        Get a guest session by its ID.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            GuestSession: Session object or None if not found
        """
        return await self.history_cache.get_session(session_id)
    
    async def get_guest_sessions(self, guest_id: str) -> List[str]:
        """
        Get all session IDs for a specific guest.
        
        Args:
            guest_id (str): Guest identifier
            
        Returns:
            List[str]: List of session IDs belonging to the guest
        """
        return await self.history_cache.get_guest_sessions(guest_id)
    
    async def delete_guest_session(self, session_id: str) -> bool:
        """
        Delete a guest session and its chat history.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            bool: True if session was deleted successfully
        """
        return await self.history_cache.delete_session(session_id)
    
    async def clear_guest_sessions(self, guest_id: str) -> int:
        """
        Clear all sessions for a specific guest.
        
        Args:
            guest_id (str): Guest identifier
            
        Returns:
            int: Number of deleted sessions
        """
        return await self.history_cache.clear_guest_sessions(guest_id)
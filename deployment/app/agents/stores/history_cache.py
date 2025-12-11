import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ChatMessage:
    """Represents a single chat message in the history."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    message_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class GuestSession:
    """Represents a guest session with chat history."""
    session_id: str
    guest_id: str
    created_at: str
    last_accessed: str
    chat_history: List[ChatMessage]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.last_accessed:
            self.last_accessed = datetime.utcnow().isoformat()
        if not self.metadata:
            self.metadata = {}


class HistoryCache:
    """
    Manages chat history for guest users using in-memory storage with optional persistence.
    Provides session-based chat history management for guest workflows.
    """
    
    def __init__(
        self, 
        max_sessions: int = 100,
        session_ttl_hours: int = 2,  # Reduced TTL for session-only storage
        max_messages_per_session: int = 100,
        enable_persistence: bool = False,  # Default to no persistence
        persistence_path: Optional[str] = None
    ):
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.max_messages_per_session = max_messages_per_session
        self.enable_persistence = False  # Force disable persistence for guest sessions
        
        # In-memory storage only
        self._sessions: Dict[str, GuestSession] = {}
        self._guest_sessions: Dict[str, List[str]] = {}  # guest_id -> [session_ids]
        self._lock = None  # Initialize lock lazily
        
        logger.info(f"HistoryCache initialized with {max_sessions} max sessions, {session_ttl_hours}h TTL (session-only mode)")
        
        # Start cleanup task for session management (only if event loop is running)
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self._periodic_cleanup())
            logger.info("Periodic cleanup task started")
        except RuntimeError:
            # No event loop is running, cleanup will be handled manually
            logger.info("No event loop running, periodic cleanup will be handled manually")

    def _get_lock(self):
        """Get or create the async lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def create_session(
        self, 
        guest_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Create a new guest session.
        
        Args:
            guest_id: Optional guest identifier. If None, generates a new one.
            session_id: Optional session identifier. If None, generates a new one.
            metadata: Optional session metadata.
            
        Returns:
            Tuple of (guest_id, session_id)
        """
        async with self._get_lock():
            if not guest_id:
                guest_id = f"guest_{uuid.uuid4().hex[:8]}"
            
            if not session_id:
                session_id = f"sess_{guest_id}_{int(datetime.utcnow().timestamp())}"
            
            # Check if session already exists
            if session_id in self._sessions:
                logger.info(f"Session {session_id} already exists, returning existing session")
                return guest_id, session_id
            
            # Create new session
            session = GuestSession(
                session_id=session_id,
                guest_id=guest_id,
                created_at=datetime.utcnow().isoformat(),
                last_accessed=datetime.utcnow().isoformat(),
                chat_history=[],
                metadata=metadata or {}
            )
            
            # Store session
            self._sessions[session_id] = session
            
            # Track sessions by guest_id
            if guest_id not in self._guest_sessions:
                self._guest_sessions[guest_id] = []
            self._guest_sessions[guest_id].append(session_id)
            
            # Clean up old sessions if we exceed the limit
            await self._cleanup_old_sessions()
            
            logger.info(f"Created new guest session: {session_id} for guest: {guest_id}")
            return guest_id, session_id

    async def get_session(self, session_id: str) -> Optional[GuestSession]:
        """Get a session by session_id."""
        async with self._get_lock():
            session = self._sessions.get(session_id)
            if session:
                # Update last accessed time
                session.last_accessed = datetime.utcnow().isoformat()
                return session
            
            # Session not found in memory (we don't load from persistence for guest sessions)
            logger.debug(f"Session {session_id} not found in memory cache")
            return None

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to the session's chat history.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            True if message was added successfully, False otherwise
        """
        async with self._get_lock():
            # Get session directly without calling get_session to avoid nested locking
            session = self._sessions.get(session_id)
            if session:
                # Update last accessed time
                session.last_accessed = datetime.utcnow().isoformat()
            else:
                # Create new session if it doesn't exist
                logger.info(f"Creating new session {session_id} for first message")
                session = GuestSession(
                    session_id=session_id,
                    guest_id=session_id.split('_')[-1] if '_' in session_id else 'unknown',
                    created_at=datetime.utcnow().isoformat(),
                    last_accessed=datetime.utcnow().isoformat(),
                    chat_history=[]
                )
                self._sessions[session_id] = session            # Create message
            message = ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.utcnow().isoformat(),
                metadata=metadata
            )
            
            # Add to history
            session.chat_history.append(message)
            
            # Trim history if it exceeds max length
            if len(session.chat_history) > self.max_messages_per_session:
                # Keep the most recent messages
                session.chat_history = session.chat_history[-self.max_messages_per_session:]
                logger.info(f"Trimmed chat history for session {session_id} to {self.max_messages_per_session} messages")
            
            # Update last accessed
            session.last_accessed = datetime.utcnow().isoformat()
            
            logger.debug(f"Added {role} message to session {session_id}: {content[:50]}...")
            return True

    async def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a complete conversation turn (user message + assistant response).
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            user_metadata: Optional metadata for user message
            assistant_metadata: Optional metadata for assistant response
            
        Returns:
            True if both messages were added successfully
        """
        success_user = await self.add_message(
            session_id, 'user', user_message, user_metadata
        )
        success_assistant = await self.add_message(
            session_id, 'assistant', assistant_response, assistant_metadata
        )
        
        return success_user and success_assistant

    async def get_chat_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        format_for_workflow: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages to return
            format_for_workflow: If True, format for workflow compatibility
            
        Returns:
            List of chat messages
        """
        session = await self.get_session(session_id)
        if not session:
            return []
        
        messages = session.chat_history
        if limit:
            messages = messages[-limit:]
        
        if format_for_workflow:
            # Format for workflow compatibility
            formatted_history = []
            for msg in messages:
                formatted_msg = {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                if msg.metadata:
                    formatted_msg.update(msg.metadata)
                formatted_history.append(formatted_msg)
            return formatted_history
        else:
            # Return raw message objects as dicts
            return [asdict(msg) for msg in messages]

    async def get_chat_summary(self, session_id: str, max_turns: int = 5) -> Dict[str, Any]:
        """
        Get a fast summary of recent chat turns for context.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of conversation turns to include in summary
            
        Returns:
            Dict containing chat summary with key information
        """
        session = await self.get_session(session_id)
        if not session or not session.chat_history:
            return {
                "session_id": session_id,
                "summary": "No conversation history",
                "turn_count": 0,
                "topics": [],
                "last_topic": None
            }
        
        # Get recent messages (user-assistant pairs)
        messages = session.chat_history
        recent_turns = []
        
        # Group messages into conversation turns
        i = 0
        turn_count = 0
        while i < len(messages) and turn_count < max_turns:
            if messages[i].role == "user":
                turn = {"user": messages[i].content, "timestamp": messages[i].timestamp}
                if i + 1 < len(messages) and messages[i + 1].role == "assistant":
                    turn["assistant"] = messages[i + 1].content
                    i += 2
                else:
                    i += 1
                recent_turns.append(turn)
                turn_count += 1
            else:
                i += 1
        
        # Extract topics and keywords
        all_content = " ".join([msg.content for msg in messages[-10:]])  # Last 10 messages
        topics = self._extract_topics_from_text(all_content)
        last_user_message = next((msg.content for msg in reversed(messages) if msg.role == "user"), None)
        
        summary = {
            "session_id": session_id,
            "summary": f"Conversation with {len(recent_turns)} recent turns covering {', '.join(topics[:3])}",
            "recent_turns": recent_turns,
            "turn_count": len(recent_turns),
            "total_messages": len(messages),
            "topics": topics,
            "last_user_message": last_user_message,
            "last_topic": topics[0] if topics else None,
            "session_duration_minutes": self._calculate_session_duration(session),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return summary

    async def get_contextual_history(
        self, 
        session_id: str, 
        current_query: str, 
        max_relevant_messages: int = 6
    ) -> Dict[str, Any]:
        """
        Get contextually relevant history based on the current query.
        
        Args:
            session_id: Session identifier
            current_query: Current user query to find relevant context for
            max_relevant_messages: Maximum number of relevant messages to return
            
        Returns:
            Dict containing relevant historical context
        """
        session = await self.get_session(session_id)
        if not session or not session.chat_history:
            return {
                "relevant_context": [],
                "query_keywords": [],
                "context_strength": 0.0
            }
        
        query_keywords = self._extract_keywords(current_query)
        relevant_messages = []
        
        # Score messages based on keyword overlap and recency
        for i, msg in enumerate(session.chat_history):
            content_keywords = self._extract_keywords(msg.content)
            
            # Calculate relevance score
            keyword_overlap = len(set(query_keywords) & set(content_keywords))
            recency_score = (i + 1) / len(session.chat_history)  # More recent = higher score
            relevance_score = keyword_overlap * 0.7 + recency_score * 0.3
            
            if relevance_score > 0.1:  # Minimum relevance threshold
                relevant_messages.append({
                    "message": asdict(msg),
                    "relevance_score": relevance_score,
                    "keyword_overlap": keyword_overlap,
                    "matched_keywords": list(set(query_keywords) & set(content_keywords))
                })
        
        # Sort by relevance and take top messages
        relevant_messages.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_relevant = relevant_messages[:max_relevant_messages]
        
        # Calculate overall context strength
        context_strength = min(1.0, sum(msg["relevance_score"] for msg in top_relevant) / max_relevant_messages)
        
        return {
            "relevant_context": top_relevant,
            "query_keywords": query_keywords,
            "context_strength": context_strength,
            "total_relevant_found": len(relevant_messages),
            "session_id": session_id,
            "extracted_for_query": current_query[:100],
            "extracted_at": datetime.utcnow().isoformat()
        }

    async def get_conversation_insights(self, session_id: str) -> Dict[str, Any]:
        """
        Get analytical insights about the conversation patterns and user behavior.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing conversation insights
        """
        session = await self.get_session(session_id)
        if not session or not session.chat_history:
            return {"session_id": session_id, "insights": "No conversation data"}
        
        messages = session.chat_history
        user_messages = [msg for msg in messages if msg.role == "user"]
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]
        
        # Analyze conversation patterns
        avg_user_msg_length = sum(len(msg.content) for msg in user_messages) / len(user_messages) if user_messages else 0
        avg_assistant_msg_length = sum(len(msg.content) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        
        # Identify question patterns
        questions = [msg for msg in user_messages if "?" in msg.content]
        follow_up_indicators = ["also", "additionally", "furthermore", "and what about", "what else"]
        follow_ups = [msg for msg in user_messages if any(indicator in msg.content.lower() for indicator in follow_up_indicators)]
        
        # Topic progression analysis
        topics_over_time = []
        for i in range(0, len(messages), 4):  # Sample every 4 messages
            batch = messages[i:i+4]
            batch_content = " ".join([msg.content for msg in batch])
            batch_topics = self._extract_topics_from_text(batch_content)
            topics_over_time.append({
                "message_range": f"{i+1}-{min(i+4, len(messages))}",
                "topics": batch_topics[:3]
            })
        
        # Engagement analysis
        engagement_level = "high" if len(questions) > 3 and len(follow_ups) > 1 else \
                          "medium" if len(questions) > 1 else "low"
        
        insights = {
            "session_id": session_id,
            "message_stats": {
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "avg_user_message_length": round(avg_user_msg_length, 1),
                "avg_assistant_message_length": round(avg_assistant_msg_length, 1)
            },
            "interaction_patterns": {
                "questions_asked": len(questions),
                "follow_up_questions": len(follow_ups),
                "engagement_level": engagement_level,
                "conversation_depth": "deep" if len(messages) > 15 else "moderate" if len(messages) > 5 else "shallow"
            },
            "topic_progression": topics_over_time,
            "session_duration_minutes": self._calculate_session_duration(session),
            "dominant_topics": self._get_dominant_topics(messages),
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        return insights

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract main topics from text content."""
        # Simple topic extraction - can be enhanced with NLP libraries
        words = text.lower().split()
        
        # Filter out common words and extract meaningful terms
        stop_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        meaningful_words = [word for word in words if len(word) > 3 and word not in stop_words and word.isalpha()]
        
        # Count frequency and return top terms
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top 5
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5]]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for relevance matching."""
        words = text.lower().split()
        # Focus on longer words that are likely to be meaningful
        keywords = [word.strip('.,!?;:"()[]') for word in words if len(word) > 3 and word.isalpha()]
        return list(set(keywords))  # Remove duplicates

    def _calculate_session_duration(self, session: GuestSession) -> float:
        """Calculate session duration in minutes."""
        try:
            if not session.chat_history:
                return 0.0
            
            start_time = datetime.fromisoformat(session.created_at.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(session.last_accessed.replace('Z', '+00:00'))
            duration = end_time - start_time
            return round(duration.total_seconds() / 60, 1)
        except:
            return 0.0

    def _get_dominant_topics(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Get dominant topics across the entire conversation."""
        all_content = " ".join([msg.content for msg in messages])
        topics = self._extract_topics_from_text(all_content)
        
        # Categorize topics into domains
        domain_keywords = {
            'health': ['health', 'medical', 'disease', 'symptom', 'treatment', 'doctor', 'medicine'],
            'genetics': ['gene', 'genetic', 'dna', 'hereditary', 'mutation', 'genome', 'chromosome'],
            'testing': ['test', 'testing', 'analysis', 'result', 'report', 'sample'],
            'product': ['product', 'kit', 'service', 'price', 'order', 'purchase'],
            'company': ['company', 'about', 'contact', 'support', 'policy', 'information']
        }
        
        topic_domains = []
        for topic in topics:
            for domain, keywords in domain_keywords.items():
                if topic in keywords:
                    topic_domains.append({"topic": topic, "domain": domain})
                    break
            else:
                topic_domains.append({"topic": topic, "domain": "general"})
        
        return topic_domains

    async def get_guest_sessions(self, guest_id: str) -> List[str]:
        """Get all session IDs for a guest."""
        async with self._get_lock():
            return self._guest_sessions.get(guest_id, []).copy()

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._get_lock():
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            guest_id = session.guest_id
            
            # Remove from sessions
            del self._sessions[session_id]
            
            # Remove from guest sessions
            if guest_id in self._guest_sessions:
                if session_id in self._guest_sessions[guest_id]:
                    self._guest_sessions[guest_id].remove(session_id)
                if not self._guest_sessions[guest_id]:
                    del self._guest_sessions[guest_id]
            
            logger.info(f"Deleted session: {session_id}")
            return True

    async def clear_guest_sessions(self, guest_id: str) -> int:
        """Clear all sessions for a guest."""
        sessions = await self.get_guest_sessions(guest_id)
        deleted_count = 0
        
        for session_id in sessions:
            if await self.delete_session(session_id):
                deleted_count += 1
        
        logger.info(f"Cleared {deleted_count} sessions for guest: {guest_id}")
        return deleted_count

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._get_lock():
            total_sessions = len(self._sessions)
            total_guests = len(self._guest_sessions)
            total_messages = sum(len(s.chat_history) for s in self._sessions.values())
            
            # Calculate memory usage (approximate)
            memory_usage = sum(
                len(json.dumps(asdict(s)).encode('utf-8'))
                for s in self._sessions.values()
            )
            
            return {
                "total_sessions": total_sessions,
                "total_guests": total_guests,
                "total_messages": total_messages,
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "max_sessions": self.max_sessions,
                "session_ttl_hours": self.session_ttl.total_seconds() / 3600
            }

    async def _cleanup_old_sessions(self):
        """
        Clean up old sessions that exceed TTL or when we have too many sessions.
        For guest workflow, we're more aggressive with cleanup to keep memory usage low.
        """
        current_time = datetime.utcnow()
        sessions_to_delete = []
        
        # Find expired sessions with lower TTL threshold for guest workflow
        for session_id, session in self._sessions.items():
            last_accessed = datetime.fromisoformat(session.last_accessed)
            # Use a more aggressive TTL for idle sessions (half the configured TTL)
            aggressive_ttl = self.session_ttl / 2
            if current_time - last_accessed > aggressive_ttl:
                sessions_to_delete.append(session_id)
                logger.debug(f"Session {session_id} marked for deletion due to inactivity")
        
        # If we still have too many sessions, delete the oldest ones
        if len(self._sessions) - len(sessions_to_delete) > self.max_sessions:
            # Sort sessions by last accessed time
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Add oldest sessions to deletion list
            sessions_needed_to_delete = len(self._sessions) - self.max_sessions
            for session_id, _ in sorted_sessions[:sessions_needed_to_delete]:
                if session_id not in sessions_to_delete:
                    sessions_to_delete.append(session_id)
                    logger.debug(f"Session {session_id} marked for deletion due to session limit")
        
        # Delete sessions
        for session_id in sessions_to_delete:
            await self.delete_session(session_id)
        
        if sessions_to_delete:
            logger.info(f"Cleaned up {len(sessions_to_delete)} old guest sessions")

    async def _periodic_cleanup(self):
        """Periodic cleanup task that runs more frequently for session-only mode."""
        while True:
            try:
                await asyncio.sleep(900)  # Run every 15 minutes for more aggressive cleanup
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def _persist_session(self, session: GuestSession):
        """
        Persist session to disk - no-op for guest workflow.
        Session data is kept in-memory only.
        """
        # No-op for guest workflow (session-only mode)
        logger.debug(f"Session persistence disabled for guest workflow: {session.session_id}")
        return

    async def _load_session(self, session_id: str) -> Optional[GuestSession]:
        """
        Load session from disk - no-op for guest workflow.
        Session data is kept in-memory only.
        """
        # No-op for guest workflow (session-only mode)
        logger.debug(f"Session loading from disk disabled for guest workflow: {session_id}")
        return None

    async def _delete_persisted_session(self, session_id: str):
        """
        Delete persisted session file - no-op for guest workflow.
        Session data is kept in-memory only.
        """
        # No-op for guest workflow (session-only mode)
        logger.debug(f"Session deletion from disk disabled for guest workflow: {session_id}")
        return


# Singleton instance for global access
_history_cache_instance: Optional[HistoryCache] = None


def get_history_cache(
    max_sessions: int = 100,
    session_ttl_hours: int = 2,  # Short TTL for guest sessions
    max_messages_per_session: int = 100,
    enable_persistence: bool = False,  # Not used for guest sessions
    persistence_path: Optional[str] = None  # Not used for guest sessions
) -> HistoryCache:
    """
    Get or create the global HistoryCache instance.
    For guest workflow, this creates a session-only cache (no long-term persistence).
    """
    global _history_cache_instance
    
    if _history_cache_instance is None:
        _history_cache_instance = HistoryCache(
            max_sessions=max_sessions,
            session_ttl_hours=session_ttl_hours,
            max_messages_per_session=max_messages_per_session,
            enable_persistence=False,  # Force disable persistence for guest sessions
            persistence_path=None
        )
        logger.info("Created global HistoryCache instance (session-only mode)")
    
    return _history_cache_instance


def reset_history_cache():
    """Reset the global HistoryCache instance (mainly for testing)."""
    global _history_cache_instance
    _history_cache_instance = None


# Usage example and testing
if __name__ == "__main__":
    async def test_history_cache():
        """Test the HistoryCache functionality."""
        print("=== Testing HistoryCache ===")
        
        # Initialize cache in session-only mode
        cache = get_history_cache(
            max_sessions=10,
            session_ttl_hours=1,
            max_messages_per_session=20,
            enable_persistence=False  # Session-only mode for guest workflow
        )
        
        # Create a session
        guest_id, session_id = await cache.create_session()
        print(f"Created session: {session_id} for guest: {guest_id}")
        
        # Add some messages
        await cache.add_message(session_id, "user", "Hello, what is GeneStory?")
        await cache.add_message(session_id, "assistant", "GeneStory is a genetic testing company...")
        
        # Add a conversation turn
        await cache.add_conversation_turn(
            session_id,
            "How does genetic testing work?",
            "Genetic testing analyzes your DNA to identify changes in genes..."
        )
        
        # Get chat history
        history = await cache.get_chat_history(session_id)
        print(f"Chat history length: {len(history)}")
        for msg in history:
            print(f"  {msg['role']}: {msg['content'][:50]}...")
        
        # Get stats
        stats = await cache.get_session_stats()
        print(f"Cache stats: {stats}")
        
        # Test session retrieval
        retrieved_session = await cache.get_session(session_id)
        print(f"Retrieved session: {retrieved_session.session_id if retrieved_session else 'None'}")
        
        print("=== Test completed ===")
    
    # Run the test
    asyncio.run(test_history_cache())
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.orm import selectinload
from loguru import logger

from app.db.models.guest import (
    GuestSession, GuestInteraction, GuestChatThread, GuestChatMessage,
    GuestSessionStatus, GuestInteractionType, Guest
)

# Import Pydantic models separately to avoid circular imports
try:
    from app.db.models.guest import (
        GuestSessionCreate, GuestSessionUpdate, GuestInteractionCreate, 
        GuestInteractionUpdate, GuestChatThreadCreate, GuestChatThreadUpdate,
        GuestChatMessageCreate, GuestChatMessageUpdate
    )
except ImportError:
    # Define minimal classes if imports fail
    from pydantic import BaseModel
    
    class GuestSessionCreate(BaseModel):
        session_id: str
        ip_address: Optional[str] = None
        user_agent: Optional[str] = None
        preferred_language: Optional[str] = "vi"
        session_metadata: Optional[Dict[str, Any]] = None
    
    class GuestSessionUpdate(BaseModel):
        preferred_language: Optional[str] = None
        session_metadata: Optional[Dict[str, Any]] = None
        last_activity_at: Optional[datetime] = None
    
    class GuestInteractionCreate(BaseModel):
        session_id: uuid.UUID
        original_query: str
        interaction_type: Optional[str] = None
    
    class GuestInteractionUpdate(BaseModel):
        rewritten_query: Optional[str] = None
        classified_agent: Optional[str] = None
        agent_response: Optional[str] = None
        suggested_questions: Optional[List[str]] = None
        processing_time_ms: Optional[int] = None
        iteration_count: Optional[int] = None
        workflow_metadata: Optional[Dict[str, Any]] = None
        completed_at: Optional[datetime] = None
        user_feedback_rating: Optional[int] = None
        user_feedback_text: Optional[str] = None
        was_helpful: Optional[bool] = None
    
    class GuestChatThreadCreate(BaseModel):
        session_id: uuid.UUID
        title: Optional[str] = None
        summary: Optional[str] = None
        primary_topic: Optional[str] = None
    
    class GuestChatThreadUpdate(BaseModel):
        title: Optional[str] = None
        summary: Optional[str] = None
        primary_topic: Optional[str] = None
        last_message_at: Optional[datetime] = None
        message_count: Optional[int] = None
        conversation_metadata: Optional[Dict[str, Any]] = None
    
    class GuestChatMessageCreate(BaseModel):
        thread_id: uuid.UUID
        sequence_number: int
        speaker: str
        content: str
    
    class GuestChatMessageUpdate(BaseModel):
        agent_used: Optional[str] = None
        processing_time_ms: Optional[int] = None
        message_metadata: Optional[Dict[str, Any]] = None

# =====================================================================================
# Guest Session CRUD Operations
# =====================================================================================

async def create_guest_session(
    db: AsyncSession, 
    *, 
    session_in: GuestSessionCreate,
    expires_in_hours: int = 24
) -> GuestSession:
    """Create a new guest session with automatic expiry and conflict handling"""
    expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
    
    db_session = GuestSession(
        session_id=session_in.session_id,
        ip_address=session_in.ip_address,
        user_agent=session_in.user_agent,
        preferred_language=session_in.preferred_language,
        session_metadata=session_in.session_metadata,
        expires_at=expires_at,
        status=GuestSessionStatus.ACTIVE
    )
    
    try:
        db.add(db_session)
        await db.commit()
        await db.refresh(db_session)
        return db_session
    except Exception as e:
        await db.rollback()
        # Check if it's a unique constraint violation
        if "duplicate key value violates unique constraint" in str(e) and "session_id" in str(e):
            # Session already exists, fetch and return it
            logger.warning(f"Session {session_in.session_id} already exists, returning existing session")
            existing_session = await get_guest_session(db, session_in.session_id)
            if existing_session:
                return existing_session
        # Re-raise if it's a different error or session not found
        raise

async def get_guest_session(db: AsyncSession, session_id: str) -> Optional[GuestSession]:
    """Get guest session by session_id"""
    result = await db.execute(
        select(GuestSession)
        .filter(GuestSession.session_id == session_id)
        .options(selectinload(GuestSession.interactions))
    )
    return result.scalars().first()

async def get_guest_session_by_id(db: AsyncSession, id: uuid.UUID) -> Optional[GuestSession]:
    """Get guest session by UUID"""
    result = await db.execute(
        select(GuestSession)
        .filter(GuestSession.id == id)
    )
    return result.scalars().first()

async def get_or_create_guest_session(
    db: AsyncSession,
    *,
    session_id: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    preferred_language: Optional[str] = "vi",
    session_metadata: Optional[Dict[str, Any]] = None,
    expires_in_hours: int = 24
) -> GuestSession:
    """Get existing session or create new one atomically"""
    # First try to get existing session
    existing_session = await get_guest_session(db, session_id)
    if existing_session:
        # Update last activity
        await update_guest_session_activity(db, session_id)
        return existing_session
    
    # Create new session if it doesn't exist
    session_create = GuestSessionCreate(
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        preferred_language=preferred_language,
        session_metadata=session_metadata
    )
    
    return await create_guest_session(db, session_in=session_create, expires_in_hours=expires_in_hours)

async def update_guest_session_activity(
    db: AsyncSession, 
    session_id: str
) -> Optional[GuestSession]:
    """Update last activity timestamp for a session"""
    session = await get_guest_session(db, session_id)
    if session:
        session.last_activity_at = datetime.utcnow()
        await db.commit()
        await db.refresh(session)
    return session

async def update_guest_session(
    db: AsyncSession,
    session_id: str,
    session_update: GuestSessionUpdate
) -> Optional[GuestSession]:
    """Update guest session details"""
    session = await get_guest_session(db, session_id)
    if not session:
        return None
    
    update_data = session_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(session, field, value)
    
    await db.commit()
    await db.refresh(session)
    return session

async def expire_guest_session(db: AsyncSession, session_id: str) -> Optional[GuestSession]:
    """Mark a guest session as expired"""
    session = await get_guest_session(db, session_id)
    if session:
        session.status = GuestSessionStatus.EXPIRED
        await db.commit()
        await db.refresh(session)
    return session

async def cleanup_expired_sessions(db: AsyncSession) -> int:
    """Clean up expired guest sessions and return count of deleted sessions"""
    current_time = datetime.utcnow()
    
    # Find expired sessions
    result = await db.execute(
        select(GuestSession)
        .filter(
            or_(
                GuestSession.expires_at < current_time,
                and_(
                    GuestSession.last_activity_at < current_time - timedelta(hours=48),
                    GuestSession.status == GuestSessionStatus.ACTIVE
                )
            )
        )
    )
    expired_sessions = result.scalars().all()
    
    # Delete expired sessions (cascade will handle related records)
    for session in expired_sessions:
        await db.delete(session)
    
    await db.commit()
    return len(expired_sessions)

# =====================================================================================
# Guest Interaction CRUD Operations
# =====================================================================================

async def create_guest_interaction(
    db: AsyncSession,
    *,
    interaction_in: GuestInteractionCreate
) -> GuestInteraction:
    """Create a new guest interaction"""
    db_interaction = GuestInteraction(
        session_id=interaction_in.session_id,
        original_query=interaction_in.original_query,
        interaction_type=interaction_in.interaction_type or GuestInteractionType.GENERAL_SEARCH
    )
    
    db.add(db_interaction)
    
    # Update session interaction count
    session = await get_guest_session_by_id(db, interaction_in.session_id)
    if session:
        session.total_interactions += 1
        session.last_activity_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(db_interaction)
    return db_interaction

async def get_guest_interaction(db: AsyncSession, interaction_id: uuid.UUID) -> Optional[GuestInteraction]:
    """Get guest interaction by ID"""
    result = await db.execute(
        select(GuestInteraction)
        .filter(GuestInteraction.id == interaction_id)
        .options(selectinload(GuestInteraction.session))
    )
    return result.scalars().first()

async def update_guest_interaction(
    db: AsyncSession,
    interaction_id: uuid.UUID,
    interaction_update: GuestInteractionUpdate
) -> Optional[GuestInteraction]:
    """Update guest interaction with workflow results"""
    interaction = await get_guest_interaction(db, interaction_id)
    if not interaction:
        return None
    
    update_data = interaction_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(interaction, field, value)
    
    # Set completion time if agent response is provided
    if interaction_update.agent_response and not interaction.completed_at:
        interaction.completed_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(interaction)
    return interaction

async def get_session_interactions(
    db: AsyncSession,
    session_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0
) -> List[GuestInteraction]:
    """Get interactions for a specific session"""
    result = await db.execute(
        select(GuestInteraction)
        .filter(GuestInteraction.session_id == session_id)
        .order_by(desc(GuestInteraction.created_at))
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()

async def add_interaction_feedback(
    db: AsyncSession,
    interaction_id: uuid.UUID,
    rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    was_helpful: Optional[bool] = None
) -> Optional[GuestInteraction]:
    """Add user feedback to an interaction"""
    interaction = await get_guest_interaction(db, interaction_id)
    if not interaction:
        return None
    
    if rating is not None:
        interaction.user_feedback_rating = rating
    if feedback_text is not None:
        interaction.user_feedback_text = feedback_text
    if was_helpful is not None:
        interaction.was_helpful = was_helpful
    
    await db.commit()
    await db.refresh(interaction)
    return interaction

# =====================================================================================
# Guest Chat Thread CRUD Operations
# =====================================================================================

async def create_guest_chat_thread(
    db: AsyncSession,
    *,
    thread_in: GuestChatThreadCreate
) -> GuestChatThread:
    """Create a new guest chat thread"""
    db_thread = GuestChatThread(
        session_id=thread_in.session_id,
        title=thread_in.title,
        summary=thread_in.summary,
        primary_topic=thread_in.primary_topic
    )
    
    db.add(db_thread)
    await db.commit()
    await db.refresh(db_thread)
    return db_thread

async def get_guest_chat_thread(db: AsyncSession, thread_id: uuid.UUID) -> Optional[GuestChatThread]:
    """Get guest chat thread by ID"""
    result = await db.execute(
        select(GuestChatThread)
        .filter(GuestChatThread.id == thread_id)
        .options(selectinload(GuestChatThread.messages))
    )
    return result.scalars().first()

async def get_session_chat_threads(
    db: AsyncSession,
    session_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0
) -> List[GuestChatThread]:
    """Get chat threads for a specific session"""
    result = await db.execute(
        select(GuestChatThread)
        .filter(GuestChatThread.session_id == session_id)
        .order_by(desc(GuestChatThread.last_message_at))
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()

async def update_guest_chat_thread(
    db: AsyncSession,
    thread_id: uuid.UUID,
    thread_update: GuestChatThreadUpdate
) -> Optional[GuestChatThread]:
    """Update guest chat thread"""
    thread = await get_guest_chat_thread(db, thread_id)
    if not thread:
        return None
    
    update_data = thread_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(thread, field, value)
    
    await db.commit()
    await db.refresh(thread)
    return thread

# =====================================================================================
# Guest Chat Message CRUD Operations
# =====================================================================================

async def create_guest_chat_message(
    db: AsyncSession,
    *,
    message_in: GuestChatMessageCreate
) -> GuestChatMessage:
    """Create a new guest chat message"""
    db_message = GuestChatMessage(
        thread_id=message_in.thread_id,
        sequence_number=message_in.sequence_number,
        speaker=message_in.speaker,
        content=message_in.content
    )
    
    db.add(db_message)
    
    # Update thread message count and last message time
    thread = await get_guest_chat_thread(db, message_in.thread_id)
    if thread:
        thread.message_count += 1
        thread.last_message_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(db_message)
    return db_message

async def get_thread_messages(
    db: AsyncSession,
    thread_id: uuid.UUID,
    limit: int = 100,
    offset: int = 0
) -> List[GuestChatMessage]:
    """Get messages for a specific thread"""
    result = await db.execute(
        select(GuestChatMessage)
        .filter(GuestChatMessage.thread_id == thread_id)
        .order_by(GuestChatMessage.sequence_number)
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()

async def update_guest_chat_message(
    db: AsyncSession,
    message_id: uuid.UUID,
    message_update: GuestChatMessageUpdate
) -> Optional[GuestChatMessage]:
    """Update guest chat message"""
    result = await db.execute(
        select(GuestChatMessage)
        .filter(GuestChatMessage.id == message_id)
    )
    message = result.scalars().first()
    
    if not message:
        return None
    
    update_data = message_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(message, field, value)
    
    await db.commit()
    await db.refresh(message)
    return message

# =====================================================================================
# Analytics and Reporting
# =====================================================================================

async def get_guest_analytics(
    db: AsyncSession,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Get analytics data for guest usage"""
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()
    
    # Total sessions
    total_sessions_result = await db.execute(
        select(func.count(GuestSession.id))
        .filter(GuestSession.created_at.between(start_date, end_date))
    )
    total_sessions = total_sessions_result.scalar()
    
    # Active sessions
    active_sessions_result = await db.execute(
        select(func.count(GuestSession.id))
        .filter(
            and_(
                GuestSession.status == GuestSessionStatus.ACTIVE,
                GuestSession.last_activity_at > datetime.utcnow() - timedelta(hours=1)
            )
        )
    )
    active_sessions = active_sessions_result.scalar()
    
    # Total interactions
    total_interactions_result = await db.execute(
        select(func.count(GuestInteraction.id))
        .filter(GuestInteraction.created_at.between(start_date, end_date))
    )
    total_interactions = total_interactions_result.scalar()
    
    # Average interactions per session
    avg_interactions = total_interactions / max(total_sessions, 1)
    
    # Most common interaction types
    interaction_types_result = await db.execute(
        select(
            GuestInteraction.interaction_type,
            func.count(GuestInteraction.id).label('count')
        )
        .filter(GuestInteraction.created_at.between(start_date, end_date))
        .group_by(GuestInteraction.interaction_type)
        .order_by(desc(func.count(GuestInteraction.id)))
        .limit(10)
    )
    interaction_types = [
        {"type": row.interaction_type, "count": row.count}
        for row in interaction_types_result.fetchall()
    ]
    
    # Most used agents
    agents_result = await db.execute(
        select(
            GuestInteraction.classified_agent,
            func.count(GuestInteraction.id).label('count')
        )
        .filter(
            and_(
                GuestInteraction.created_at.between(start_date, end_date),
                GuestInteraction.classified_agent.isnot(None)
            )
        )
        .group_by(GuestInteraction.classified_agent)
        .order_by(desc(func.count(GuestInteraction.id)))
        .limit(10)
    )
    agents = [
        {"agent": row.classified_agent, "count": row.count}
        for row in agents_result.fetchall()
    ]
    
    # Average processing time
    avg_time_result = await db.execute(
        select(func.avg(GuestInteraction.processing_time_ms))
        .filter(
            and_(
                GuestInteraction.created_at.between(start_date, end_date),
                GuestInteraction.processing_time_ms.isnot(None)
            )
        )
    )
    avg_processing_time = avg_time_result.scalar()
    
    # User satisfaction rating
    satisfaction_result = await db.execute(
        select(func.avg(GuestInteraction.user_feedback_rating))
        .filter(
            and_(
                GuestInteraction.created_at.between(start_date, end_date),
                GuestInteraction.user_feedback_rating.isnot(None)
            )
        )
    )
    user_satisfaction = satisfaction_result.scalar()
    
    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "total_interactions": total_interactions,
        "avg_interactions_per_session": round(avg_interactions, 2),
        "most_common_interaction_types": interaction_types,
        "most_used_agents": agents,
        "avg_processing_time_ms": round(avg_processing_time, 2) if avg_processing_time else None,
        "user_satisfaction_rating": round(user_satisfaction, 2) if user_satisfaction else None,
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    }


class GuestAnalytics:
    def __init__(self, data: dict):
        self.data = data

    def get_total_sessions(self) -> int:
        return self.data.get("total_sessions", 0)

    def get_active_sessions(self) -> int:
        return self.data.get("active_sessions", 0)

    def get_total_interactions(self) -> int:
        return self.data.get("total_interactions", 0)

    def get_avg_interactions_per_session(self) -> float:
        return self.data.get("avg_interactions_per_session", 0.0)

    def get_most_common_interaction_types(self) -> list:
        return self.data.get("most_common_interaction_types", [])

    def get_most_used_agents(self) -> list:
        return self.data.get("most_used_agents", [])

    def get_avg_processing_time(self) -> float:
        return self.data.get("avg_processing_time_ms", 0.0)

    def get_user_satisfaction_rating(self) -> float:
        return self.data.get("user_satisfaction_rating", 0.0)

    def get_period(self) -> dict:
        return self.data.get("period", {})


# =====================================================================================
# CRUD Classes for structured access (following customer pattern)
# =====================================================================================

class GuestCrud:
    """CRUD operations for Guest Sessions"""
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: GuestSessionCreate,
        expires_in_hours: int = 24
    ) -> GuestSession:
        """Create a new guest session"""
        return await create_guest_session(db, session_in=obj_in, expires_in_hours=expires_in_hours)
    
    async def get(self, db: AsyncSession, session_id: str) -> Optional[GuestSession]:
        """Get guest session by session_id"""
        return await get_guest_session(db, session_id)
    
    async def get_by_id(self, db: AsyncSession, id: uuid.UUID) -> Optional[GuestSession]:
        """Get guest session by UUID"""
        return await get_guest_session_by_id(db, id)
    
    async def update(
        self, 
        db: AsyncSession, 
        session_id: str, 
        obj_in: GuestSessionUpdate
    ) -> Optional[GuestSession]:
        """Update guest session"""
        return await update_guest_session(db, session_id, obj_in)
    
    async def update_activity(
        self, 
        db: AsyncSession, 
        session_id: str
    ) -> Optional[GuestSession]:
        """Update session last activity"""
        return await update_guest_session_activity(db, session_id)
    
    async def expire(self, db: AsyncSession, session_id: str) -> Optional[GuestSession]:
        """Mark session as expired"""
        return await expire_guest_session(db, session_id)
    
    async def cleanup_expired(self, db: AsyncSession) -> int:
        """Clean up expired sessions"""
        return await cleanup_expired_sessions(db)
    
    async def get_analytics(
        self,
        db: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get guest analytics"""
        return await get_guest_analytics(db, start_date, end_date)
    
    async def get_active_sessions_count(self, db: AsyncSession) -> int:
        """Get count of currently active sessions"""
        result = await db.execute(
            select(func.count(GuestSession.id))
            .filter(
                and_(
                    GuestSession.status == GuestSessionStatus.ACTIVE,
                    GuestSession.last_activity_at > datetime.utcnow() - timedelta(hours=1)
                )
            )
        )
        return result.scalar() or 0
    
    async def get_sessions_by_date_range(
        self,
        db: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
        offset: int = 0
    ) -> List[GuestSession]:
        """Get sessions within a date range"""
        result = await db.execute(
            select(GuestSession)
            .filter(GuestSession.created_at.between(start_date, end_date))
            .order_by(desc(GuestSession.created_at))
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
    
    async def get_session_stats(self, db: AsyncSession, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        session = await self.get(db, session_id)
        if not session:
            return {}
        
        # Count interactions
        interaction_count = await db.scalar(
            select(func.count(GuestInteraction.id))
            .filter(GuestInteraction.session_id == session.id)
        )
        
        # Count chat threads
        thread_count = await db.scalar(
            select(func.count(GuestChatThread.id))
            .filter(GuestChatThread.session_id == session.id)
        )
        
        # Average satisfaction rating
        avg_rating = await db.scalar(
            select(func.avg(GuestInteraction.user_feedback_rating))
            .filter(
                and_(
                    GuestInteraction.session_id == session.id,
                    GuestInteraction.user_feedback_rating.isnot(None)
                )
            )
        )
        
        return {
            "session_id": session.session_id,
            "total_interactions": interaction_count or 0,
            "total_chat_threads": thread_count or 0,
            "avg_satisfaction_rating": float(avg_rating) if avg_rating else None,
            "session_duration_hours": (
                (datetime.utcnow() - session.created_at).total_seconds() / 3600
                if session.status == GuestSessionStatus.ACTIVE
                else (session.last_activity_at - session.created_at).total_seconds() / 3600
            )
        }


class GuestInteractionCrud:
    """CRUD operations for Guest Interactions"""
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: GuestInteractionCreate
    ) -> GuestInteraction:
        """Create a new guest interaction"""
        return await create_guest_interaction(db, interaction_in=obj_in)
    
    async def get(self, db: AsyncSession, interaction_id: uuid.UUID) -> Optional[GuestInteraction]:
        """Get guest interaction by ID"""
        return await get_guest_interaction(db, interaction_id)
    
    async def update(
        self, 
        db: AsyncSession, 
        interaction_id: uuid.UUID, 
        obj_in: GuestInteractionUpdate
    ) -> Optional[GuestInteraction]:
        """Update guest interaction"""
        return await update_guest_interaction(db, interaction_id, obj_in)
    
    async def get_by_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0
    ) -> List[GuestInteraction]:
        """Get interactions for a session"""
        return await get_session_interactions(db, session_id, limit, offset)
    
    async def add_feedback(
        self,
        db: AsyncSession,
        interaction_id: uuid.UUID,
        rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        was_helpful: Optional[bool] = None
    ) -> Optional[GuestInteraction]:
        """Add user feedback to interaction"""
        return await add_interaction_feedback(db, interaction_id, rating, feedback_text, was_helpful)
    
    async def get_recent_interactions(
        self,
        db: AsyncSession,
        hours: int = 24,
        limit: int = 100
    ) -> List[GuestInteraction]:
        """Get recent interactions within specified hours"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        result = await db.execute(
            select(GuestInteraction)
            .filter(GuestInteraction.created_at >= since_time)
            .order_by(desc(GuestInteraction.created_at))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_interactions_by_type(
        self,
        db: AsyncSession,
        interaction_type: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[GuestInteraction]:
        """Get interactions by type"""
        result = await db.execute(
            select(GuestInteraction)
            .filter(GuestInteraction.interaction_type == interaction_type)
            .order_by(desc(GuestInteraction.created_at))
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
    
    async def get_interaction_stats(
        self,
        db: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get interaction statistics"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Total interactions
        total_count = await db.scalar(
            select(func.count(GuestInteraction.id))
            .filter(GuestInteraction.created_at.between(start_date, end_date))
        )
        
        # Completed interactions
        completed_count = await db.scalar(
            select(func.count(GuestInteraction.id))
            .filter(
                and_(
                    GuestInteraction.created_at.between(start_date, end_date),
                    GuestInteraction.completed_at.isnot(None)
                )
            )
        )
        
        # Average processing time
        avg_processing_time = await db.scalar(
            select(func.avg(GuestInteraction.processing_time_ms))
            .filter(
                and_(
                    GuestInteraction.created_at.between(start_date, end_date),
                    GuestInteraction.processing_time_ms.isnot(None)
                )
            )
        )
        
        return {
            "total_interactions": total_count or 0,
            "completed_interactions": completed_count or 0,
            "completion_rate": (completed_count / max(total_count, 1)) * 100 if total_count else 0,
            "avg_processing_time_ms": float(avg_processing_time) if avg_processing_time else None
        }


class GuestChatThreadCrud:
    """CRUD operations for Guest Chat Threads"""
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: GuestChatThreadCreate
    ) -> GuestChatThread:
        """Create a new guest chat thread"""
        return await create_guest_chat_thread(db, thread_in=obj_in)
    
    async def get(self, db: AsyncSession, thread_id: uuid.UUID) -> Optional[GuestChatThread]:
        """Get guest chat thread by ID"""
        return await get_guest_chat_thread(db, thread_id)
    
    async def get_by_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        limit: int = 20,
        offset: int = 0
    ) -> List[GuestChatThread]:
        """Get chat threads for a session"""
        return await get_session_chat_threads(db, session_id, limit, offset)
    
    async def update(
        self, 
        db: AsyncSession, 
        thread_id: uuid.UUID, 
        obj_in: GuestChatThreadUpdate
    ) -> Optional[GuestChatThread]:
        """Update guest chat thread"""
        return await update_guest_chat_thread(db, thread_id, obj_in)
    
    async def get_active_threads(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        hours: int = 24
    ) -> List[GuestChatThread]:
        """Get active threads with recent activity"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        result = await db.execute(
            select(GuestChatThread)
            .filter(
                and_(
                    GuestChatThread.session_id == session_id,
                    GuestChatThread.last_message_at >= since_time
                )
            )
            .order_by(desc(GuestChatThread.last_message_at))
        )
        return result.scalars().all()
    
    async def get_thread_summary(
        self,
        db: AsyncSession,
        thread_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Get thread summary with message count and latest activity"""
        thread = await self.get(db, thread_id)
        if not thread:
            return None
        
        message_count = await db.scalar(
            select(func.count(GuestChatMessage.id))
            .filter(GuestChatMessage.thread_id == thread_id)
        )
        
        latest_message = await db.execute(
            select(GuestChatMessage)
            .filter(GuestChatMessage.thread_id == thread_id)
            .order_by(desc(GuestChatMessage.sequence_number))
            .limit(1)
        )
        latest = latest_message.scalar_one_or_none()
        
        return {
            "thread_id": str(thread.id),
            "title": thread.title,
            "message_count": message_count or 0,
            "created_at": thread.created_at,
            "last_message_at": thread.last_message_at,
            "latest_message_content": latest.content[:100] + "..." if latest and len(latest.content) > 100 else latest.content if latest else None,
            "primary_topic": thread.primary_topic
        }


class GuestChatMessageCrud:
    """CRUD operations for Guest Chat Messages"""
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: GuestChatMessageCreate
    ) -> GuestChatMessage:
        """Create a new guest chat message"""
        return await create_guest_chat_message(db, message_in=obj_in)
    
    async def get_by_thread(
        self,
        db: AsyncSession,
        thread_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[GuestChatMessage]:
        """Get messages for a thread"""
        return await get_thread_messages(db, thread_id, limit, offset)
    
    async def update(
        self, 
        db: AsyncSession, 
        message_id: uuid.UUID, 
        obj_in: GuestChatMessageUpdate
    ) -> Optional[GuestChatMessage]:
        """Update guest chat message"""
        return await update_guest_chat_message(db, message_id, obj_in)
    
    async def get_next_sequence_number(
        self,
        db: AsyncSession,
        thread_id: uuid.UUID
    ) -> int:
        """Get the next sequence number for a thread"""
        result = await db.scalar(
            select(func.max(GuestChatMessage.sequence_number))
            .filter(GuestChatMessage.thread_id == thread_id)
        )
        return (result or 0) + 1
    
    async def get_message_by_sequence(
        self,
        db: AsyncSession,
        thread_id: uuid.UUID,
        sequence_number: int
    ) -> Optional[GuestChatMessage]:
        """Get message by thread and sequence number"""
        result = await db.execute(
            select(GuestChatMessage)
            .filter(
                and_(
                    GuestChatMessage.thread_id == thread_id,
                    GuestChatMessage.sequence_number == sequence_number
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_latest_messages(
        self,
        db: AsyncSession,
        thread_id: uuid.UUID,
        limit: int = 10
    ) -> List[GuestChatMessage]:
        """Get latest messages from a thread"""
        result = await db.execute(
            select(GuestChatMessage)
            .filter(GuestChatMessage.thread_id == thread_id)
            .order_by(desc(GuestChatMessage.sequence_number))
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))  # Return in chronological order
    
    async def search_messages(
        self,
        db: AsyncSession,
        thread_id: uuid.UUID,
        search_term: str,
        limit: int = 50
    ) -> List[GuestChatMessage]:
        """Search messages in a thread by content"""
        result = await db.execute(
            select(GuestChatMessage)
            .filter(
                and_(
                    GuestChatMessage.thread_id == thread_id,
                    GuestChatMessage.content.ilike(f"%{search_term}%")
                )
            )
            .order_by(GuestChatMessage.sequence_number)
            .limit(limit)
        )
        return result.scalars().all()


# =====================================================================================
# CRUD object instances for easy import (following customer pattern)
# =====================================================================================

# Create instances for easy import
guest_crud = GuestCrud()
guest_interaction_crud = GuestInteractionCrud()
guest_chat_thread_crud = GuestChatThreadCrud()
guest_chat_message_crud = GuestChatMessageCrud()

# =====================================================================================
# Guest CRUD Operations (for Guest model, not GuestSession)
# =====================================================================================

async def create_guest(
    db: AsyncSession,
    *,
    session_id: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    preferred_language: Optional[str] = "vi"
) -> Guest:
    """Create a new guest user"""
    guest = Guest(
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        preferred_language=preferred_language
    )
    
    db.add(guest)
    await db.commit()
    await db.refresh(guest)
    return guest

async def get_guest_by_session_id(db: AsyncSession, session_id: str) -> Optional[Guest]:
    """Get guest by session ID"""
    result = await db.execute(
        select(Guest).filter(Guest.session_id == session_id)
    )
    return result.scalars().first()

async def get_guest_by_id(db: AsyncSession, guest_id: int) -> Optional[Guest]:
    """Get guest by ID"""
    result = await db.execute(
        select(Guest).filter(Guest.id == guest_id)
    )
    return result.scalars().first()

async def update_guest_activity(db: AsyncSession, guest: Guest) -> Guest:
    """Update guest last activity timestamp"""
    guest.last_activity_at = datetime.utcnow()
    await db.commit()
    await db.refresh(guest)
    return guest

async def get_or_create_guest(
    db: AsyncSession,
    *,
    session_id: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    preferred_language: Optional[str] = "vi"
) -> Guest:
    """Get existing guest by session ID or create new one"""
    guest = await get_guest_by_session_id(db, session_id)
    if not guest:
        guest = await create_guest(
            db,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            preferred_language=preferred_language
        )
    else:
        await update_guest_activity(db, guest)
    
    return guest

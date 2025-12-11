# from asyncio.log import logger
from loguru import logger
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, text, update
from sqlalchemy.orm import selectinload, joinedload
import uuid
import hashlib

from app.db.models.customer import (
    Customer, CustomerSession, CustomerInteraction, CustomerChatThread, 
    CustomerChatMessage, CustomerRefreshToken, CustomerRole, CustomerStatus,
    CustomerSessionStatus, CustomerInteractionType
)
from app.db.models.customer import (
    CustomerCreate, CustomerUpdate, CustomerAdminUpdate,
    CustomerSessionCreate, CustomerSessionUpdate,
    CustomerInteractionCreate, CustomerInteractionUpdate,
    CustomerChatThreadCreate, CustomerChatThreadUpdate,
    CustomerChatMessageCreate, CustomerChatMessageUpdate
)
from app.core.security import get_password_hash, verify_password, generate_session_id
from app.core.config import settings

# Customer CRUD operations

async def get_customer(db: AsyncSession, customer_id: int) -> Optional[Customer]:
    """Get customer by ID"""
    result = await db.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    return result.scalar_one_or_none()

async def get_customer_by_email(db: AsyncSession, email: str) -> Optional[Customer]:
    """Get customer by email"""
    result = await db.execute(
        select(Customer).where(Customer.email == email)
    )
    return result.scalar_one_or_none()

async def get_customer_by_username(db: AsyncSession, username: int) -> Optional[Customer]:
    """Get customer by username (integer ID)"""
    result = await db.execute(
        select(Customer).where(Customer.username == username)
    )
    return result.scalar_one_or_none()

async def create_customer(db: AsyncSession, customer_in: CustomerCreate) -> Customer:
    """Create new customer with password hashing"""
    obj_in_data = customer_in.model_dump()
    password = obj_in_data.pop("password")
    
    # Generate unique username if not provided
    if obj_in_data.get("username") is None:
        # Find the next available username (integer)
        result = await db.execute(
            select(func.max(Customer.username)).where(Customer.username.isnot(None))
        )
        max_username = result.scalar()
        obj_in_data["username"] = (max_username or 0) + 1
    
    db_obj = Customer(**obj_in_data)
    db_obj.set_password(password)
    
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def authenticate_customer(db: AsyncSession, username: int, password: str) -> Optional[Customer]:
    """Authenticate customer by email and password"""
    customer = await get_customer_by_username(db,  username)
    if not customer:
        return None
    if not customer.verify_password(password):
        return None
    return customer

async def update_customer_last_login(
    db: AsyncSession, customer: Customer, ip_address: str = None
) -> Customer:
    """Update customer's last login timestamp"""
    customer.last_login_at = datetime.utcnow()
    customer.last_activity_at = datetime.utcnow()
    customer.failed_login_attempts = 0  # Reset on successful login
    customer.locked_until = None  # Remove lock on successful login
    
    await db.commit()
    await db.refresh(customer)
    return customer

async def increment_customer_failed_login(db: AsyncSession, customer: Customer) -> Customer:
    """Increment failed login attempts and lock if necessary"""
    customer.failed_login_attempts += 1
    
    # Lock account if too many failed attempts
    if customer.failed_login_attempts >= settings.MAX_LOGIN_ATTEMPTS:
        customer.locked_until = datetime.utcnow() + timedelta(
            minutes=settings.ACCOUNT_LOCKOUT_DURATION_MINUTES
        )
    
    await db.commit()
    await db.refresh(customer)
    return customer

async def update_customer(
    db: AsyncSession, customer: Customer, customer_update: CustomerUpdate
) -> Customer:
    """Update customer profile"""
    update_data = customer_update.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(customer, field, value)
    
    customer.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(customer)
    return customer

async def change_customer_password(
    db: AsyncSession, customer: Customer, new_password: str
) -> Customer:
    """Change customer password"""
    customer.set_password(new_password)
    customer.must_change_password = False
    
    await db.commit()
    await db.refresh(customer)
    return customer

async def verify_customer_email(db: AsyncSession, customer: Customer) -> Customer:
    """Mark customer email as verified"""
    customer.email_verified = True
    if customer.status == CustomerStatus.PENDING_VERIFICATION:
        customer.status = CustomerStatus.ACTIVE
        customer.is_verified = True
    
    await db.commit()
    await db.refresh(customer)
    return customer

async def admin_update_customer(
    db: AsyncSession, customer: Customer, obj_in: CustomerAdminUpdate
) -> Customer:
    """Admin update with additional fields"""
    update_data = obj_in.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(customer, field, value)
    
    customer.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(customer)
    return customer

async def get_customers_with_filters(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    role: Optional[CustomerRole] = None,
    status: Optional[CustomerStatus] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> List[Customer]:
    """Get customers with filtering and pagination"""
    query = select(Customer)
    
    # Apply filters
    if role:
        query = query.where(Customer.role == role)
    if status:
        query = query.where(Customer.status == status)
    if search:
        # Try to convert search to integer for username comparison if it's numeric
        username_int = None
        try:
            if search.isdigit():
                username_int = int(search)
        except (ValueError, AttributeError):
            pass
        
        if username_int is not None:
            search_filter = or_(
                Customer.email.ilike(f"%{search}%"),
                Customer.username == username_int,
                Customer.first_name.ilike(f"%{search}%"),
                Customer.last_name.ilike(f"%{search}%")
            )
        else:
            search_filter = or_(
                Customer.email.ilike(f"%{search}%"),
                Customer.first_name.ilike(f"%{search}%"),
                Customer.last_name.ilike(f"%{search}%")
            )
        query = query.where(search_filter)
    
    # Apply sorting
    if hasattr(Customer, sort_by):
        order_column = getattr(Customer, sort_by)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(order_column))
        else:
            query = query.order_by(asc(order_column))
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_customer_stats(db: AsyncSession, customer_id: int) -> Dict[str, Any]:
    """Get customer statistics"""
    # Get basic counts
    session_count = await db.scalar(
        select(func.count(CustomerSession.id)).where(
            CustomerSession.customer_id == customer_id
        )
    )
    
    interaction_count = await db.scalar(
        select(func.count(CustomerInteraction.id)).where(
            CustomerInteraction.customer_id == customer_id
        )
    )
    
    chat_count = await db.scalar(
        select(func.count(CustomerChatThread.id)).where(
            CustomerChatThread.customer_id == customer_id
        )
    )
    
    # Get average satisfaction rating
    avg_rating = await db.scalar(
        select(func.avg(CustomerInteraction.user_feedback_rating)).where(
            and_(
                CustomerInteraction.customer_id == customer_id,
                CustomerInteraction.user_feedback_rating.isnot(None)
            )
        )
    )
    
    # Get most used interaction type
    most_used_type_result = await db.execute(
        select(
            CustomerInteraction.interaction_type,
            func.count(CustomerInteraction.id).label('count')
        ).where(
            CustomerInteraction.customer_id == customer_id
        ).group_by(
            CustomerInteraction.interaction_type
        ).order_by(
            desc('count')
        ).limit(1)
    )
    most_used_type = most_used_type_result.first()
    
    return {
        "total_sessions": session_count or 0,
        "total_interactions": interaction_count or 0,
        "total_chats": chat_count or 0,
        "avg_satisfaction_rating": float(avg_rating) if avg_rating else None,
        "most_used_interaction_type": most_used_type[0] if most_used_type else None
    }

# Customer Session CRUD operations

async def create_customer_session(
    db: AsyncSession, customer_id: int, **kwargs
) -> CustomerSession:
    """Create new customer session"""
    session_data = {
        "customer_id": customer_id,
        "session_id": generate_session_id(),
        "expires_at": datetime.utcnow() + timedelta(hours=settings.SESSION_EXPIRE_HOURS),
        **kwargs
    }
    
    db_obj = CustomerSession(**session_data)
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_customer_session(db: AsyncSession, session_id: uuid.UUID) -> Optional[CustomerSession]:
    """Get customer session by ID"""
    result = await db.execute(
        select(CustomerSession).where(CustomerSession.id == session_id)
    )
    return result.scalar_one_or_none()

async def get_active_customer_sessions(
    db: AsyncSession, customer_id: int
) -> List[CustomerSession]:
    """Get active sessions for customer"""
    result = await db.execute(
        select(CustomerSession).where(
            and_(
                CustomerSession.customer_id == customer_id,
                CustomerSession.status == CustomerSessionStatus.ACTIVE,
                CustomerSession.expires_at > datetime.utcnow()
            )
        ).order_by(desc(CustomerSession.last_activity_at))
    )
    return result.scalars().all()

async def update_customer_session_activity(
    db: AsyncSession, session: CustomerSession
) -> CustomerSession:
    """Update session last activity"""
    session.last_activity_at = datetime.utcnow()
    await db.commit()
    await db.refresh(session)
    return session

async def expire_old_customer_sessions(
    db: AsyncSession, customer_id: int, keep_latest: int = 5
) -> int:
    """Expire old sessions for customer, keeping the latest N"""
    # Get sessions to expire
    result = await db.execute(
        select(CustomerSession.id).where(
            and_(
                CustomerSession.customer_id == customer_id,
                CustomerSession.status == CustomerSessionStatus.ACTIVE
            )
        ).order_by(desc(CustomerSession.last_activity_at)).offset(keep_latest)
    )
    session_ids_to_expire = result.scalars().all()
    
    if session_ids_to_expire:
        await db.execute(
            update(CustomerSession).where(
                CustomerSession.id.in_(session_ids_to_expire)
            ).values(
                status=CustomerSessionStatus.EXPIRED,
                ended_at=datetime.utcnow()
            )
        )
        await db.commit()
    
    return len(session_ids_to_expire)

# Customer Interaction CRUD operations

async def create_customer_interaction(
    db: AsyncSession, interaction_in: CustomerInteractionCreate
) -> CustomerInteraction:
    """Create customer interaction"""
    db_obj = CustomerInteraction(**interaction_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_customer_interaction(
    db: AsyncSession, interaction_id: uuid.UUID
) -> Optional[CustomerInteraction]:
    """Get customer interaction by ID"""
    result = await db.execute(
        select(CustomerInteraction).where(CustomerInteraction.id == interaction_id)
    )
    return result.scalar_one_or_none()

async def get_customer_interactions(
    db: AsyncSession,
    customer_id: int,
    skip: int = 0,
    limit: int = 100,
    interaction_type: Optional[CustomerInteractionType] = None
) -> List[CustomerInteraction]:
    """Get customer interactions with pagination"""
    query = select(CustomerInteraction).where(
        CustomerInteraction.customer_id == customer_id
    )
    
    if interaction_type:
        query = query.where(CustomerInteraction.interaction_type == interaction_type)
    
    query = query.order_by(desc(CustomerInteraction.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def update_customer_interaction(
    db: AsyncSession, interaction: CustomerInteraction, update_data: CustomerInteractionUpdate
) -> CustomerInteraction:
    """Update customer interaction"""
    update_dict = update_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        setattr(interaction, field, value)
    
    await db.commit()
    await db.refresh(interaction)
    return interaction

async def update_customer_interaction_feedback(
    db: AsyncSession,
    interaction_id: uuid.UUID,
    rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    was_helpful: Optional[bool] = None
) -> Optional[CustomerInteraction]:
    """Update interaction with user feedback"""
    interaction = await get_customer_interaction(db, interaction_id)
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

# Customer Chat Thread CRUD operations

async def create_customer_chat_thread(
    db: AsyncSession, thread_in: CustomerChatThreadCreate
) -> CustomerChatThread:
    """Create customer chat thread"""
    db_obj = CustomerChatThread(**thread_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_customer_chat_thread(
    db: AsyncSession, thread_id: uuid.UUID
) -> Optional[CustomerChatThread]:
    """Get customer chat thread by ID"""
    result = await db.execute(
        select(CustomerChatThread).where(CustomerChatThread.id == thread_id)
    )
    # logger.info(f"Retrieved chat thread: {result.scalar_one_or_none()}")
    return result.scalar_one_or_none()

async def get_customer_chat_threads(
    db: AsyncSession,
    customer_id: int,
    skip: int = 0,
    limit: int = 100,
    include_archived: bool = False
) -> List[CustomerChatThread]:
    """Get customer chat threads"""
    query = select(CustomerChatThread).where(
        and_(
            CustomerChatThread.customer_id == customer_id,
            CustomerChatThread.is_deleted == False
        )
    )
    
    if not include_archived:
        query = query.where(CustomerChatThread.is_archived == False)
    
    query = query.order_by(desc(CustomerChatThread.last_message_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def update_customer_chat_thread(
    db: AsyncSession, thread: CustomerChatThread, update_data: CustomerChatThreadUpdate
) -> CustomerChatThread:
    """Update customer chat thread"""
    update_dict = update_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        setattr(thread, field, value)
    
    await db.commit()
    await db.refresh(thread)
    return thread

async def archive_customer_chat_thread(
    db: AsyncSession, thread_id: uuid.UUID
) -> Optional[CustomerChatThread]:
    """Archive a chat thread"""
    thread = await get_customer_chat_thread(db, thread_id)
    if thread:
        thread.is_archived = True
        await db.commit()
        await db.refresh(thread)
    return thread

async def delete_customer_chat_thread(
    db: AsyncSession, thread_id: uuid.UUID
) -> Optional[CustomerChatThread]:
    """Soft delete a chat thread"""
    thread = await get_customer_chat_thread(db, thread_id)
    if thread:
        thread.is_deleted = True
        await db.commit()
        await db.refresh(thread)
    return thread

# Customer Chat Message CRUD operations

async def create_customer_chat_message(
    db: AsyncSession, message_in: CustomerChatMessageCreate
) -> CustomerChatMessage:
    """Create customer chat message"""
    db_obj = CustomerChatMessage(**message_in.model_dump())
    db.add(db_obj)
    
    # Update thread message count and last message time
    thread = await get_customer_chat_thread(db, message_in.thread_id)
    if thread:
        thread.message_count += 1
        thread.last_message_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_customer_chat_message(
    db: AsyncSession, message_id: uuid.UUID
) -> Optional[CustomerChatMessage]:
    """Get customer chat message by ID"""
    result = await db.execute(
        select(CustomerChatMessage).where(CustomerChatMessage.id == message_id)
    )
    return result.scalar_one_or_none()



async def get_customer_chat_thread_messages(
    db: AsyncSession,
    thread_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100
) -> List[CustomerChatMessage]:
    """Get messages for a chat thread"""
    
    result = await db.execute(
        select(CustomerChatMessage).where(
            and_(
                CustomerChatMessage.thread_id == thread_id,
                CustomerChatMessage.is_deleted == False
            )
        ).order_by(desc(CustomerChatMessage.sequence_number)).offset(skip).limit(limit)
    )
    return result.scalars().all()


# async def get_chat_thread_messages(
#     db: AsyncSession,
#     thread_id: uuid.UUID,
#     skip: int = 0,
#     limit: int = 100
# ) -> List[CustomerChatMessage]:
#     """Get messages for a chat thread"""
#     result = await db.execute(
#         select(CustomerChatMessage).where(
#             and_(
#                 CustomerChatMessage.thread_id == thread_id,
#                 CustomerChatMessage.is_deleted == False
#             )
#         ).order_by(desc(CustomerChatMessage.sequence_number)).offset(skip).limit(limit)
#     )
#     return result.scalars().all()





async def get_next_message_sequence_number(
    db: AsyncSession, thread_id: uuid.UUID
) -> int:
    """Get the next sequence number for a thread"""
    result = await db.scalar(
        select(func.max(CustomerChatMessage.sequence_number)).where(
            CustomerChatMessage.thread_id == thread_id
        )
    )
    return (result or 0) + 1

async def update_customer_chat_message(
    db: AsyncSession, message: CustomerChatMessage, update_data: CustomerChatMessageUpdate
) -> CustomerChatMessage:
    """Update customer chat message"""
    update_dict = update_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        setattr(message, field, value)
    
    await db.commit()
    await db.refresh(message)
    return message

# Customer Refresh Token CRUD operations

async def create_customer_refresh_token(
    db: AsyncSession,
    customer_id: int,  # Changed from uuid.UUID to int to match the model
    token_hash: str,
    expires_at: datetime,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> CustomerRefreshToken:
    """Create refresh token"""
    db_obj = CustomerRefreshToken(
        customer_id=customer_id,  # Now using int instead of uuid.UUID
        token_hash=token_hash,
        expires_at=expires_at,
        device_info=device_info,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_customer_refresh_token_by_hash(
    db: AsyncSession, token_hash: str
) -> Optional[CustomerRefreshToken]:
    """Get refresh token by hash"""
    result = await db.execute(
        select(CustomerRefreshToken).where(
            and_(
                CustomerRefreshToken.token_hash == token_hash,
                CustomerRefreshToken.is_revoked == False,
                CustomerRefreshToken.expires_at > datetime.utcnow()
            )
        )
    )
    return result.scalar_one_or_none()

async def revoke_customer_refresh_token(
    db: AsyncSession, token_hash: str
) -> Optional[CustomerRefreshToken]:
    """Revoke refresh token"""
    token = await get_customer_refresh_token_by_hash(db, token_hash)
    if token:
        token.is_revoked = True
        token.revoked_at = datetime.utcnow()
        await db.commit()
        await db.refresh(token)
    return token

async def revoke_all_customer_tokens(
    db: AsyncSession, customer_id: int
) -> int:
    """Revoke all tokens for a customer"""
    result = await db.execute(
        select(CustomerRefreshToken).where(
            and_(
                CustomerRefreshToken.customer_id == customer_id,
                CustomerRefreshToken.is_revoked == False
            )
        )
    )
    tokens = result.scalars().all()
    
    for token in tokens:
        token.is_revoked = True
        token.revoked_at = datetime.utcnow()
    
    await db.commit()
    return len(tokens)

async def cleanup_expired_customer_tokens(db: AsyncSession) -> int:
    """Clean up expired tokens"""
    result = await db.execute(
        select(CustomerRefreshToken).where(
            CustomerRefreshToken.expires_at <= datetime.utcnow()
        )
    )
    expired_tokens = result.scalars().all()
    
    for token in expired_tokens:
        await db.delete(token)
    
    await db.commit()
    return len(expired_tokens)


# Additional CRUD functions for completeness

async def delete_customer(db: AsyncSession, customer_id: int) -> bool:
    """Delete customer (soft delete by setting inactive)"""
    result = await db.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    customer = result.scalar_one_or_none()
    if not customer:
        return False
    
    # Soft delete by setting status to inactive
    customer.status = CustomerStatus.INACTIVE
    customer.is_active = False
    customer.updated_at = datetime.utcnow()
    
    await db.commit()
    return True

async def get_customers(
    db: AsyncSession, 
    skip: int = 0, 
    limit: int = 100, 
    filters: Optional[Dict[str, Any]] = None
) -> List[Customer]:
    """Get customers with optional filters"""
    return await get_customers_with_filters(db, filters, skip, limit)

async def set_customer_active(db: AsyncSession, customer_id: int, is_active: bool) -> bool:
    """Set customer active status"""
    result = await db.execute(
        select(Customer).where(Customer.id == customer_id)
    )
    customer = result.scalar_one_or_none()
    if not customer:
        return False

    customer.status = CustomerStatus.ACTIVE if is_active else CustomerStatus.INACTIVE
    customer.updated_at = datetime.utcnow()
    
    await db.commit()
    return True

async def update_customer_session(
    db: AsyncSession,
    session_id: uuid.UUID, 
    obj_in: CustomerSessionUpdate
) -> Optional[CustomerSession]:
    """Update customer session"""
    result = await db.execute(
        select(CustomerSession).where(CustomerSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return None
    
    update_data = obj_in.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()
    
    for field, value in update_data.items():
        setattr(session, field, value)
    
    await db.commit()
    await db.refresh(session)
    return session

async def end_customer_session(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """End customer session"""
    result = await db.execute(
        select(CustomerSession).where(CustomerSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return False
    
    session.status = CustomerSessionStatus.ENDED
    session.ended_at = datetime.utcnow()
    session.updated_at = datetime.utcnow()
    
    await db.commit()
    return True

async def get_customer_sessions(
    db: AsyncSession, 
    customer_id: int, 
    skip: int = 0, 
    limit: int = 100
) -> List[CustomerSession]:
    """Get customer sessions"""
    result = await db.execute(
        select(CustomerSession)
        .where(CustomerSession.customer_id == customer_id)
        .order_by(desc(CustomerSession.created_at))
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())

async def get_session_interactions(
    db: AsyncSession, 
    session_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100
) -> List[CustomerInteraction]:
    """Get interactions for a session"""
    result = await db.execute(
        select(CustomerInteraction)
        .where(CustomerInteraction.session_id == session_id)
        .order_by(desc(CustomerInteraction.created_at))
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())

async def get_thread_messages(
    db: AsyncSession, 
    thread_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100
) -> List[CustomerChatMessage]:
    """Get messages for a chat thread"""
    result = await db.execute(
        select(CustomerChatMessage)
        .where(CustomerChatMessage.thread_id == thread_id)
        .order_by(asc(CustomerChatMessage.created_at))
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())

async def get_customer_refresh_token(db: AsyncSession, token: str) -> Optional[CustomerRefreshToken]:
    """Get refresh token by token string"""
    result = await db.execute(
        select(CustomerRefreshToken).where(
            and_(
                CustomerRefreshToken.token == token,
                CustomerRefreshToken.is_revoked == False,
                CustomerRefreshToken.expires_at > datetime.utcnow()
            )
        )
    )
    return result.scalar_one_or_none()


# CRUD object instances for easy import
# These provide a structured way to access all CRUD operations

class CustomerCrud:
    """Customer CRUD operations"""
    
    async def get(self, db: AsyncSession, id: int) -> Optional[Customer]:
        return await get_customer(db, id)
    
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[Customer]:
        return await get_customer_by_email(db, email)
    
    async def get_by_username(self, db: AsyncSession, username: int) -> Optional[Customer]:
        return await get_customer_by_username(db, username)
    
    async def create(self, db: AsyncSession, obj_in: CustomerCreate) -> Customer:
        return await create_customer(db, obj_in)
    
    async def update(self, db: AsyncSession, customer_id: uuid.UUID, obj_in: CustomerUpdate) -> Optional[Customer]:
        return await update_customer(db, customer_id, obj_in)
    
    async def delete(self, db: AsyncSession, customer_id: uuid.UUID) -> bool:
        return await delete_customer(db, customer_id)
    
    async def get_multi(self, db: AsyncSession, skip: int = 0, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Customer]:
        return await get_customers(db, skip, limit, filters)
    
    async def authenticate(self, db: AsyncSession, email: str, password: str) -> Optional[Customer]:
        return await authenticate_customer(db, email, password)
    
    async def verify_email(self, db: AsyncSession, customer_id: int) -> bool:
        return await verify_customer_email(db, customer_id)
    
    async def set_active(self, db: AsyncSession, customer_id: int, is_active: bool) -> bool:
        return await set_customer_active(db, customer_id, is_active)

class CustomerSessionCrud:
    """Customer session CRUD operations"""
    
    async def create(self, db: AsyncSession, obj_in: CustomerSessionCreate) -> CustomerSession:
        return await create_customer_session(db, obj_in)
    
    async def get(self, db: AsyncSession, session_id: uuid.UUID) -> Optional[CustomerSession]:
        return await get_customer_session(db, session_id)
    
    async def update(self, db: AsyncSession, session_id: uuid.UUID, obj_in: CustomerSessionUpdate) -> Optional[CustomerSession]:
        return await update_customer_session(db, session_id, obj_in)
    
    async def end(self, db: AsyncSession, session_id: uuid.UUID) -> bool:
        return await end_customer_session(db, session_id)
    
    async def get_customer_sessions(self, db: AsyncSession, customer_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[CustomerSession]:
        return await get_customer_sessions(db, customer_id, skip, limit)

class CustomerInteractionCrud:
    """Customer interaction CRUD operations"""
    
    async def create(self, db: AsyncSession, obj_in: CustomerInteractionCreate) -> CustomerInteraction:
        return await create_customer_interaction(db, obj_in)
    
    async def get(self, db: AsyncSession, interaction_id: uuid.UUID) -> Optional[CustomerInteraction]:
        return await get_customer_interaction(db, interaction_id)
    
    async def update(self, db: AsyncSession, interaction_id: uuid.UUID, obj_in: CustomerInteractionUpdate) -> Optional[CustomerInteraction]:
        return await update_customer_interaction(db, interaction_id, obj_in)
    
    async def get_session_interactions(self, db: AsyncSession, session_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[CustomerInteraction]:
        return await get_session_interactions(db, session_id, skip, limit)

class CustomerChatThreadCrud:
    """Customer chat thread CRUD operations"""
    
    async def create(self, db: AsyncSession, obj_in: CustomerChatThreadCreate) -> CustomerChatThread:
        return await create_customer_chat_thread(db, obj_in)
    
    async def get(self, db: AsyncSession, thread_id: uuid.UUID) -> Optional[CustomerChatThread]:
        return await get_customer_chat_thread(db, thread_id)
    
    async def update(self, db: AsyncSession, thread_id: uuid.UUID, obj_in: CustomerChatThreadUpdate) -> Optional[CustomerChatThread]:
        return await update_customer_chat_thread(db, thread_id, obj_in)
    
    async def get_customer_threads(self, db: AsyncSession, customer_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[CustomerChatThread]:
        return await get_customer_chat_threads(db, customer_id, skip, limit)

class CustomerChatMessageCrud:
    """Customer chat message CRUD operations"""
    
    async def create(self, db: AsyncSession, obj_in: CustomerChatMessageCreate) -> CustomerChatMessage:
        return await create_customer_chat_message(db, obj_in)
    
    async def get_thread_messages(self, db: AsyncSession, thread_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[CustomerChatMessage]:
        return await get_thread_messages(db, thread_id, skip, limit)

class CustomerRefreshTokenCrud:
    """Customer refresh token CRUD operations"""
    
    async def create(self, db: AsyncSession, customer_id: uuid.UUID, token: str, expires_at: datetime) -> CustomerRefreshToken:
        return await create_customer_refresh_token(db, customer_id, token, expires_at)
    
    async def get_by_token(self, db: AsyncSession, token: str) -> Optional[CustomerRefreshToken]:
        return await get_customer_refresh_token(db, token)
    
    async def revoke(self, db: AsyncSession, token: str) -> bool:
        return await revoke_customer_refresh_token(db, token)
    
    async def revoke_all_customer_tokens(self, db: AsyncSession, customer_id: uuid.UUID) -> int:
        return await revoke_all_customer_tokens(db, customer_id)
    
    async def cleanup_expired(self, db: AsyncSession) -> int:
        return await cleanup_expired_customer_tokens(db)

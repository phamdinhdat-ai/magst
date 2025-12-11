from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from typing import AsyncGenerator
from loguru import logger
import os
import asyncio


# Database configuration with fallback to SQLite for development
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://datpd:datpd@localhost:5432/gst_ai")
DATABASE_URL = "postgresql+asyncpg://datpd:datpd@localhost:5432/gst_agents"
logger.info(f"Using configured DATABASE_URL: {DATABASE_URL}...") if DATABASE_URL else logger.warning("No DATABASE_URL provided, using SQLite fallback")

# Async engine with connection pooling
async_engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,  # Reduced pool size
    max_overflow=10,  # Increased overflow
    pool_timeout=30,
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,  # Set to True for SQL debugging
    # Additional connection management
    pool_reset_on_return='commit',
    # PostgreSQL specific settings
    connect_args={
        "server_settings": {"application_name": "genstory_workflow"},
        "command_timeout": 60
    } if "postgresql" in DATABASE_URL else {}
)

# Async session factory
AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get an async database session.
    Ensures the session is closed after the request.
    """
    session = None
    try:
        session = AsyncSessionLocal()
        yield session
        
    except asyncio.CancelledError:
        # Ensure rollback on cancellation
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise


def get_engine():
    """Return the current async engine instance."""
    return async_engine

async def close_db_connections():
    """
    Force close any lingering database connections.
    Call this function after processing is complete in long-running tasks.
    """
    try:
        engine = get_engine()
        if engine is None:
            return
        if hasattr(engine, 'dispose'):
            await engine.dispose()
        # If using asyncpg directly, clean up its connections too
        try:
            import asyncpg
            # There is no public API for asyncpg pool cleanup here, so skip
        except ImportError:
            pass
    except Exception as e:
        logger.warning(f"Error cleaning up database connections: {e}")

class ManagedAsyncSession:
    """
    Context manager to ensure database sessions are properly managed and connections returned to pool.
    Usage:
        async with ManagedAsyncSession() as db:
            # use db session here
    """
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = AsyncSession(bind=get_engine())
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                if exc_type:
                    await self.session.rollback()
                else:
                    await self.session.commit()
            finally:
                await self.session.close()
                self.session = None

# Utility function to get a managed session
async def get_managed_db_session():
    """
    Get a database session that's automatically managed.
    """
    async with ManagedAsyncSession() as session:
        yield session

async def get_db_engine():
    """Get the database engine for manual connection management"""
    return async_engine

# Create all tables (use this in startup)
async def create_tables():
    """Create all database tables"""
    try:
        # Import all models to ensure they are registered with SQLAlchemy
        from app.db.models.guest import Base as GuestBase
        from app.db.models.customer import Base as CustomerBase
        from app.db.models.document import Base as DocumentBase
        
        # Try to import employee model if it exists
        try:
            from app.db.models.employee import Base as EmployeeBase
            has_employee_model = True
        except ImportError:
            has_employee_model = False
        
        async with async_engine.begin() as conn:
            # Create employee tables first if they exist (to satisfy foreign key constraints)
            if has_employee_model:
                logger.info("Creating employee tables...")
                await conn.run_sync(EmployeeBase.metadata.create_all)
            
            # Create customer tables
            logger.info("Creating customer tables...")
            await conn.run_sync(CustomerBase.metadata.create_all)
            
            # Create guest tables
            logger.info("Creating guest tables...")
            await conn.run_sync(GuestBase.metadata.create_all)
            
            # Create document tables last (since they may reference other tables)
            logger.info("Creating document tables...")
            await conn.run_sync(DocumentBase.metadata.create_all)
            
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise
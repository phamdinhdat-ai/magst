"""
Application lifecycle management for the GeneStory Chatbot
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

# Import async resources that need lifecycle management
try:
    from app.api.v1.guest_api import request_queue as guest_request_queue
    GUEST_QUEUE_AVAILABLE = True
except ImportError:
    GUEST_QUEUE_AVAILABLE = False
    logger.warning("Guest request queue not available for lifecycle management")

@asynccontextmanager
async def lifecycle_context(app: FastAPI):
    """
    Lifecycle context manager for the FastAPI application
    Handles startup and shutdown of various async resources
    """
    # STARTUP PHASE
    logger.info("Application startup beginning...")
    
    # Initialize guest request queue if available
    if GUEST_QUEUE_AVAILABLE:
        logger.info("Starting guest request queue workers...")
        try:
            await guest_request_queue.start_workers()
            logger.info("Guest request queue workers started successfully")
        except Exception as e:
            logger.error(f"Failed to start guest request queue workers: {str(e)}", exc_info=True)
    
    logger.info("Application startup complete")
    
    # RUNTIME PHASE
    yield  # Application runs here
    
    # SHUTDOWN PHASE
    logger.info("Application shutdown beginning...")
    
    # Shutdown guest request queue if available
    if GUEST_QUEUE_AVAILABLE:
        logger.info("Shutting down guest request queue...")
        try:
            await guest_request_queue.shutdown()
            logger.info("Guest request queue shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down guest request queue: {str(e)}", exc_info=True)
    
    # Other cleanup tasks can go here
    logger.info("Application shutdown complete")


def configure_lifecycle(app: FastAPI):
    """Configure the FastAPI app with our lifecycle context"""
    app.router.lifespan_context = lifecycle_context
    return app

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
from typing import Optional

# Import API routers
from app.api.v1.guest_api import router as guest_router
from app.api.v1.customer_api import router as customer_router
from app.api.v1.employee_api import router as employee_router
from app.api.v1.document_api import router as document_router
from app.api.v1.rag_api import router as rag_router
from app.api.v1.system import router as system_router
from app.api.v1.endpoints.employee_queue_endpoints import router as employee_queue_router
from app.api.v1.endpoints.customer_queue_endpoints import router as customer_queue_router
from app.api.v1.endpoints.document_queue_endpoints import router as document_queue_router
from app.api.v1.endpoints.products import router as products_router
from app.api.health import get_health_router
from app.db.session import create_tables, close_db_connections

# System monitoring components
from app.core.db_health_checker import init_db_health_checker, get_db_health_checker
from app.core.load_balancer import init_load_balancer, get_load_balancer
from app.core.queue_manager import init_queue_manager, get_queue_manager
from app.agents.workflow.customer_workflow import CustomerWorkflow
from app.agents.workflow.employee_workflow import EmployeeWorkflow
from app.agents.workflow.guest_workflow import GuestWorkflow
from app.api.v1.document_request_queue import document_request_queue
from app.tasks.workflow_tasks import process_query_task
from app.lifecycle import configure_lifecycle
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class WorkflowManager:
#     customer_workflow: Optional[CustomerWorkflow] = CustomerWorkflow()
#     employee_workflow: Optional[EmployeeWorkflow] = EmployeeWorkflow()
#     guest_workflow: Optional[GuestWorkflow] = GuestWorkflow()


# workflow_manager = WorkflowManager()

# def initialize_workflow_manager() -> WorkflowManager:
#     """Initialize the application's workflow manager instance"""
#     global workflow_manager
#     if workflow_manager is None:
#         workflow_manager = WorkflowManager()
#         logger.info("WorkflowManager initialized")
#     return workflow_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    # Startup
    logger.info("Starting up GeneStory Workflow Application...")
    try:
        # Create database tables
        await create_tables()
        
        # Initialize system monitoring components
        logger.info("Initializing system monitoring components...")
        
        # 1. Database health checker
        db_health = init_db_health_checker(check_interval=60)
        await db_health.start()
        logger.info("Database health checker initialized")
        
        # 2. Load balancer (non-coordinator mode)
        load_balancer = init_load_balancer(app=app, coordinator_mode=False)
        await load_balancer.start()
        logger.info("Load balancer initialized")
        
        # 3. Queue manager
        queue_manager = init_queue_manager()
        
        # Register workflow task processors
        queue_manager.register_task("app.tasks.workflow_tasks.process_query_task", process_query_task)
        logger.info("Registered workflow task processors with queue manager")
        
        await queue_manager.start()
        logger.info("Queue manager initialized")
        
        # Initialize all workflows at startup
        # initialize_workflow_manager()
        logger.info("All workflows initialized")
        
        # Initialize document processing queue
        asyncio.create_task(document_request_queue.start_workers())
        logger.info("Document processing queue initialized")
        
        # Import and initialize customer request queue
        from app.api.v1.customer_request_queue import customer_request_queue
        from app.agents.workflow.customer_workflow import CustomerWorkflow
        from app.core.workflow_manager import init_customer_workflow, get_customer_workflow
        
        # Initialize and store the customer workflow
        # Set max_iterations to 3 - balances thoroughness with preventing recursion errors
        customer_workflow = CustomerWorkflow(max_iterations=3)
        init_customer_workflow(customer_workflow)
        
        # Set the workflow manager for the customer queue
        customer_request_queue.set_workflow(customer_workflow)
        
        # Start customer queue workers and ensure they're running
        worker_task = asyncio.create_task(customer_request_queue.start_workers())
        logger.info("Customer processing queue workers started")
        
        # Add this task to the application state for tracking
        app.state.worker_tasks = getattr(app.state, "worker_tasks", []) + [worker_task]
        
        # Mark workers as started to avoid initialization issues
        customer_request_queue._workers_started = True
        logger.info("Customer processing queue initialized and registered with application state")
        
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        logger.info("Application will continue but functionality may be limited")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GeneStory Workflow Application...")
    try:
        # Close all database connections
        await close_db_connections()
        
        # Stop system monitoring components
        db_health = get_db_health_checker()
        if db_health:
            await db_health.stop()
            
        load_balancer = get_load_balancer()
        if load_balancer:
            await load_balancer.stop()
            
        queue_manager = get_queue_manager()
        if queue_manager:
            await queue_manager.stop()
            
        # Stop document queue
        try:
            await document_request_queue.shutdown()
            logger.info("Document processing queue stopped")
        except Exception as e:
            logger.error(f"Error stopping document queue: {e}")
            
        # Stop customer queue
        try:
            from app.api.v1.customer_request_queue import customer_request_queue
            await customer_request_queue.shutdown()
            logger.info("Customer processing queue stopped")
        except Exception as e:
            logger.error(f"Error stopping customer queue: {e}")
            
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="GeneStory Workflow API",
    description="API for guest, customer, and employee interactions with the GeneStory workflow system",
    version="1.0.0",
    lifespan=lifespan,
)
configure_lifecycle(app)
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(guest_router)
app.include_router(customer_router)
app.include_router(employee_router)
app.include_router(document_router)
app.include_router(rag_router)
app.include_router(system_router)
app.include_router(employee_queue_router)
app.include_router(customer_queue_router)
app.include_router(document_queue_router)
app.include_router(products_router)
app.include_router(products_router)

# Include the universal feedback router
from app.api.v1.feedback_api import router as feedback_router
app.include_router(feedback_router)

# Include comprehensive health router
health_router = get_health_router()
app.include_router(health_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

"""
Workflow Manager module for maintaining singleton instances of workflow classes
across the application.
"""
from typing import Optional
from app.agents.workflow.customer_workflow import CustomerWorkflow

# Module-level variables to store workflow instances
_customer_workflow: Optional[CustomerWorkflow] = None

def init_customer_workflow(workflow: CustomerWorkflow) -> None:
    """
    Initialize the global customer workflow instance
    
    Args:
        workflow: The CustomerWorkflow instance to store
    """
    global _customer_workflow
    _customer_workflow = workflow
    
def get_customer_workflow() -> Optional[CustomerWorkflow]:
    """
    Get the global customer workflow instance
    
    Returns:
        The CustomerWorkflow instance or None if not initialized
    """
    return _customer_workflow

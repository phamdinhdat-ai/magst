"""
Schema definitions for user types.
This module provides type definitions for use with FastAPI dependency injection.
"""
from typing import Protocol, Union, Optional, Any
from datetime import datetime

class UserLike(Protocol):
    """Protocol defining common attributes across user types."""
    id: Any
    
    @property
    def is_active(self) -> bool:
        """Check if user account is active."""
        ...

# Import at this level to avoid circular dependencies
from app.db.models.customer import Customer
from app.db.models.employee import Employee
from app.db.models.guest import Guest

# Define a type alias for use in dependencies
UserType = Union[Customer, Employee, Guest]

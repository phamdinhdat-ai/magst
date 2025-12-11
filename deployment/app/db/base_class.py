from sqlalchemy.ext.declarative import as_declarative, declared_attr
from typing import Any
import re

def camel_to_snake(name):
    """Converts CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

@as_declarative()
class Base:
    """
    Base class for SQLAlchemy models.
    It provides a default __tablename__ generation mechanism.
    """
    id: Any  # All tables will have an id column (type defined in subclasses)
    __name__: str

    # Generate __tablename__ automatically from class name
    # e.g., GuestSession -> guest_sessions
    @declared_attr
    def __tablename__(cls) -> str:
        return camel_to_snake(cls.__name__) + "s"

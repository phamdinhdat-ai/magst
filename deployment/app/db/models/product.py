"""
Product model for storing genetic testing product configurations
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, func, JSON, Numeric
from sqlalchemy.dialects.postgresql import UUID

from app.db.base_class import Base


class ProductModel(Base):
    """
    Database model for storing product configurations from ProductConfig2024.csv
    """
    __tablename__ = "products"
    
    # Primary key - using the ID from CSV
    id = Column(Integer, primary_key=True)
    
    # Basic product information
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(100), nullable=False, index=True)
    price = Column(String(50), nullable=True)  # Keep as string to preserve formatting like "4,800,000 "
    price_numeric = Column(Numeric(12, 2), nullable=True)  # Parsed numeric value for calculations
    subject = Column(String(255), nullable=True, index=True)  # Target audience
    working_time = Column(Integer, nullable=True)  # Processing time in days
    technology = Column(String(100), nullable=True, index=True)
    
    # Product descriptions
    summary = Column(Text, nullable=True)  # Detailed Vietnamese description
    feature = Column(Text, nullable=True)  # Vietnamese features
    feature_en = Column(Text, nullable=True)  # English features
    product_index = Column(String(100), nullable=True, unique=True, index=True)  # Unique product identifier
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(String(10), default="true")  # For soft deletion
    
    # Additional metadata for future use
    product_metadata = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<ProductModel(id={self.id}, name='{self.name}', type='{self.type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "price": self.price,
            "price_numeric": float(self.price_numeric) if self.price_numeric else None,
            "subject": self.subject,
            "working_time": self.working_time,
            "technology": self.technology,
            "summary": self.summary,
            "feature": self.feature,
            "feature_en": self.feature_en,
            "product_index": self.product_index,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "product_metadata": self.product_metadata
        }

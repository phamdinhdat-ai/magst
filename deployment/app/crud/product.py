"""
CRUD operations for Product model
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from app.db.models.product import ProductModel


class ProductCRUD:
    """CRUD operations for Product model"""
    
    async def create(self, db: AsyncSession, product_data: Dict[str, Any]) -> ProductModel:
        """Create a new product"""
        product = ProductModel(**product_data)
        db.add(product)
        await db.commit()
        await db.refresh(product)
        return product
    
    async def get_by_id(self, db: AsyncSession, product_id: int) -> Optional[ProductModel]:
        """Get product by ID"""
        result = await db.execute(
            select(ProductModel).where(ProductModel.id == product_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_product_index(self, db: AsyncSession, product_index: str) -> Optional[ProductModel]:
        """Get product by product index"""
        result = await db.execute(
            select(ProductModel).where(ProductModel.product_index == product_index)
        )
        return result.scalar_one_or_none()
    
    async def get_by_name(self, db: AsyncSession, name: str) -> Optional[ProductModel]:
        """Get product by name"""
        result = await db.execute(
            select(ProductModel).where(ProductModel.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> List[ProductModel]:
        """Get all products with pagination"""
        query = select(ProductModel)
        
        if active_only:
            query = query.where(ProductModel.is_active == "true")
        
        query = query.offset(skip).limit(limit).order_by(ProductModel.id)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_by_type(
        self, 
        db: AsyncSession, 
        product_type: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ProductModel]:
        """Get products by type"""
        result = await db.execute(
            select(ProductModel)
            .where(and_(
                ProductModel.type == product_type,
                ProductModel.is_active == "true"
            ))
            .offset(skip)
            .limit(limit)
            .order_by(ProductModel.id)
        )
        return result.scalars().all()
    
    async def get_by_subject(
        self, 
        db: AsyncSession, 
        subject: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ProductModel]:
        """Get products by subject (target audience)"""
        result = await db.execute(
            select(ProductModel)
            .where(and_(
                ProductModel.subject.ilike(f"%{subject}%"),
                ProductModel.is_active == "true"
            ))
            .offset(skip)
            .limit(limit)
            .order_by(ProductModel.id)
        )
        return result.scalars().all()
    
    async def get_by_technology(
        self, 
        db: AsyncSession, 
        technology: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ProductModel]:
        """Get products by technology"""
        result = await db.execute(
            select(ProductModel)
            .where(and_(
                ProductModel.technology == technology,
                ProductModel.is_active == "true"
            ))
            .offset(skip)
            .limit(limit)
            .order_by(ProductModel.id)
        )
        return result.scalars().all()
    
    async def search_products(
        self,
        db: AsyncSession,
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ProductModel]:
        """Search products by name, feature, or summary"""
        search_pattern = f"%{search_term}%"
        
        result = await db.execute(
            select(ProductModel)
            .where(and_(
                or_(
                    ProductModel.name.ilike(search_pattern),
                    ProductModel.feature.ilike(search_pattern),
                    ProductModel.feature_en.ilike(search_pattern),
                    ProductModel.summary.ilike(search_pattern)
                ),
                ProductModel.is_active == "true"
            ))
            .offset(skip)
            .limit(limit)
            .order_by(ProductModel.id)
        )
        return result.scalars().all()
    
    async def get_price_range(
        self,
        db: AsyncSession,
        min_price: float = None,
        max_price: float = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ProductModel]:
        """Get products within price range"""
        query = select(ProductModel).where(ProductModel.is_active == "true")
        
        if min_price is not None:
            query = query.where(ProductModel.price_numeric >= min_price)
        
        if max_price is not None:
            query = query.where(ProductModel.price_numeric <= max_price)
        
        query = query.offset(skip).limit(limit).order_by(ProductModel.price_numeric)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def update(
        self, 
        db: AsyncSession, 
        product_id: int, 
        update_data: Dict[str, Any]
    ) -> Optional[ProductModel]:
        """Update a product"""
        product = await self.get_by_id(db, product_id)
        if not product:
            return None
        
        for field, value in update_data.items():
            setattr(product, field, value)
        
        await db.commit()
        await db.refresh(product)
        return product
    
    async def soft_delete(self, db: AsyncSession, product_id: int) -> bool:
        """Soft delete a product (set is_active to false)"""
        product = await self.get_by_id(db, product_id)
        if not product:
            return False
        
        product.is_active = "false"
        await db.commit()
        return True
    
    async def hard_delete(self, db: AsyncSession, product_id: int) -> bool:
        """Hard delete a product"""
        product = await self.get_by_id(db, product_id)
        if not product:
            return False
        
        await db.delete(product)
        await db.commit()
        return True
    
    async def get_product_types(self, db: AsyncSession) -> List[str]:
        """Get all unique product types"""
        result = await db.execute(
            select(ProductModel.type)
            .where(ProductModel.is_active == "true")
            .distinct()
            .order_by(ProductModel.type)
        )
        return [row[0] for row in result.fetchall()]
    
    async def get_technologies(self, db: AsyncSession) -> List[str]:
        """Get all unique technologies"""
        result = await db.execute(
            select(ProductModel.technology)
            .where(and_(
                ProductModel.is_active == "true",
                ProductModel.technology.is_not(None)
            ))
            .distinct()
            .order_by(ProductModel.technology)
        )
        return [row[0] for row in result.fetchall()]
    
    async def get_count(self, db: AsyncSession, active_only: bool = True) -> int:
        """Get total count of products"""
        query = select(func.count(ProductModel.id))
        
        if active_only:
            query = query.where(ProductModel.is_active == "true")
        
        result = await db.execute(query)
        return result.scalar()


# Create instance for use in API endpoints
product_crud = ProductCRUD()

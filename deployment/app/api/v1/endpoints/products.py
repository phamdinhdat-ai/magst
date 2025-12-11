"""
API endpoints for Product management
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.session import get_db_session
from app.crud.product import product_crud
from app.db.models.product import ProductModel

router = APIRouter()

# Pydantic models for API
class ProductBase(BaseModel):
    name: str = Field(..., description="Product name")
    type: str = Field(..., description="Product type")
    price: Optional[str] = Field(None, description="Price as string")
    price_numeric: Optional[float] = Field(None, description="Numeric price value")
    subject: Optional[str] = Field(None, description="Target audience")
    working_time: Optional[int] = Field(None, description="Processing time in days")
    technology: Optional[str] = Field(None, description="Technology used")
    summary: Optional[str] = Field(None, description="Product summary")
    feature: Optional[str] = Field(None, description="Vietnamese features")
    feature_en: Optional[str] = Field(None, description="English features")
    product_index: Optional[str] = Field(None, description="Product index")

class ProductCreate(ProductBase):
    id: int = Field(..., description="Product ID")

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    price: Optional[str] = None
    price_numeric: Optional[float] = None
    subject: Optional[str] = None
    working_time: Optional[int] = None
    technology: Optional[str] = None
    summary: Optional[str] = None
    feature: Optional[str] = None
    feature_en: Optional[str] = None
    product_index: Optional[str] = None
    is_active: Optional[str] = None

class ProductResponse(ProductBase):
    id: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_active: str
    product_metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class ProductListResponse(BaseModel):
    products: List[ProductResponse]
    total: int
    skip: int
    limit: int

@router.get("/products", response_model=ProductListResponse)
async def get_products(
    skip: int = Query(0, ge=0, description="Number of products to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of products to return"),
    active_only: bool = Query(True, description="Filter only active products"),
    db: AsyncSession = Depends(get_db_session)
):
    """Get all products with pagination"""
    products = await product_crud.get_all(db, skip=skip, limit=limit, active_only=active_only)
    total = await product_crud.get_count(db, active_only=active_only)
    
    return ProductListResponse(
        products=[ProductResponse(**product.to_dict()) for product in products],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Get a specific product by ID"""
    product = await product_crud.get_by_id(db, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return ProductResponse(**product.to_dict())

@router.get("/products/index/{product_index}", response_model=ProductResponse)
async def get_product_by_index(
    product_index: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get a specific product by product index"""
    product = await product_crud.get_by_product_index(db, product_index)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return ProductResponse(**product.to_dict())

@router.get("/products/type/{product_type}", response_model=ProductListResponse)
async def get_products_by_type(
    product_type: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db_session)
):
    """Get products by type"""
    products = await product_crud.get_by_type(db, product_type, skip=skip, limit=limit)
    total = len(products)  # Simple count for filtered results
    
    return ProductListResponse(
        products=[ProductResponse(**product.to_dict()) for product in products],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/products/subject/{subject}", response_model=ProductListResponse)
async def get_products_by_subject(
    subject: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db_session)
):
    """Get products by subject (target audience)"""
    products = await product_crud.get_by_subject(db, subject, skip=skip, limit=limit)
    total = len(products)
    
    return ProductListResponse(
        products=[ProductResponse(**product.to_dict()) for product in products],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/products/technology/{technology}", response_model=ProductListResponse)
async def get_products_by_technology(
    technology: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db_session)
):
    """Get products by technology"""
    products = await product_crud.get_by_technology(db, technology, skip=skip, limit=limit)
    total = len(products)
    
    return ProductListResponse(
        products=[ProductResponse(**product.to_dict()) for product in products],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/products/search", response_model=ProductListResponse)
async def search_products(
    q: str = Query(..., description="Search term"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db_session)
):
    """Search products by name, feature, or summary"""
    products = await product_crud.search_products(db, q, skip=skip, limit=limit)
    total = len(products)
    
    return ProductListResponse(
        products=[ProductResponse(**product.to_dict()) for product in products],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/products/price-range", response_model=ProductListResponse)
async def get_products_by_price_range(
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db_session)
):
    """Get products within price range"""
    products = await product_crud.get_price_range(
        db, min_price=min_price, max_price=max_price, skip=skip, limit=limit
    )
    total = len(products)
    
    return ProductListResponse(
        products=[ProductResponse(**product.to_dict()) for product in products],
        total=total,
        skip=skip,
        limit=limit
    )

@router.post("/products", response_model=ProductResponse)
async def create_product(
    product_data: ProductCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new product"""
    # Check if product with same ID already exists
    existing = await product_crud.get_by_id(db, product_data.id)
    if existing:
        raise HTTPException(status_code=400, detail="Product with this ID already exists")
    
    # Check if product with same index already exists
    if product_data.product_index:
        existing_index = await product_crud.get_by_product_index(db, product_data.product_index)
        if existing_index:
            raise HTTPException(status_code=400, detail="Product with this index already exists")
    
    product = await product_crud.create(db, product_data.dict())
    return ProductResponse(**product.to_dict())

@router.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product_data: ProductUpdate,
    db: AsyncSession = Depends(get_db_session)
):
    """Update a product"""
    # Remove None values from update data
    update_data = {k: v for k, v in product_data.dict().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    product = await product_crud.update(db, product_id, update_data)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return ProductResponse(**product.to_dict())

@router.delete("/products/{product_id}")
async def delete_product(
    product_id: int,
    hard_delete: bool = Query(False, description="Perform hard delete instead of soft delete"),
    db: AsyncSession = Depends(get_db_session)
):
    """Delete a product (soft delete by default)"""
    if hard_delete:
        success = await product_crud.hard_delete(db, product_id)
    else:
        success = await product_crud.soft_delete(db, product_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {"message": "Product deleted successfully"}

@router.get("/products/meta/types", response_model=List[str])
async def get_product_types(db: AsyncSession = Depends(get_db_session)):
    """Get all unique product types"""
    return await product_crud.get_product_types(db)

@router.get("/products/meta/technologies", response_model=List[str])
async def get_technologies(db: AsyncSession = Depends(get_db_session)):
    """Get all unique technologies"""
    return await product_crud.get_technologies(db)

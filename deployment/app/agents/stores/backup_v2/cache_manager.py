import hashlib
import json
import os
from typing import Any, Optional, List, Dict

import redis.asyncio as redis
from loguru import logger

class CacheManager:
    """
    An asynchronous cache manager using Redis for storing and retrieving AI workflow results.
    Implements an "Exact Cache" strategy.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure __init__ is run only once for the singleton
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.redis_pool = None
        try:
            # Use environment variables for configuration (best practice)
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            self.redis_pool = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            logger.info(f"CacheManager initialized and connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Could not connect to Redis. Caching will be disabled. Error: {e}")
            self.redis_pool = None

    def is_active(self) -> bool:
        """Check if the cache is connected and active."""
        return self.redis_pool is not None

    def create_cache_key(self, query: str, chat_history: Optional[List[Dict]] = None) -> str:
        """
        Creates a stable, hash-based key from the core request components.
        We exclude session_id and guest_id to maximize cache hits across users for common questions.
        """
        # Create a dictionary of the data that defines the request's uniqueness
        payload = {
            "query": query,
            # Sort chat history to ensure consistent serialization
            # "chat_history": sorted(chat_history, key=lambda x: x.get('type', '')) if chat_history else []
        }
        
        # Serialize the dictionary to a string. `sort_keys=True` is critical.
        serialized_payload = json.dumps(payload, sort_keys=True)
        
        # Use SHA-256 to create a unique and fixed-length key
        hash_object = hashlib.sha256(serialized_payload.encode('utf-8'))
        
        # Prepend with a namespace for clarity in Redis
        # Determine workflow type from query prefix if present
        if query.startswith("[EMPLOYEE:"):
            namespace = "employee_workflow"
        elif query.startswith("[CUSTOMER:"):
            namespace = "customer_workflow"
        else:
            namespace = "guest_workflow"
        
        return f"cache:{namespace}:{hash_object.hexdigest()}"

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Asynchronously get a value from the cache."""
        if not self.is_active():
            return None
        try:
            cached_data = await self.redis_pool.get(key)
            if cached_data:
                logger.success(f"CACHE HIT for key: {key}")
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """
        Asynchronously set a value in the cache with a Time-To-Live (TTL).
        
        Args:
            key (str): The cache key.
            value (Dict[str, Any]): The dictionary object to cache.
            ttl (int): Time-to-live in seconds. Default is 1 hour.
        """
        if not self.is_active():
            return
        try:
            serialized_value = json.dumps(value)
            await self.redis_pool.set(key, serialized_value, ex=ttl)
            logger.info(f"CACHE SET for key: {key} with TTL: {ttl}s")
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
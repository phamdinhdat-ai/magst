"""
Load Balancing Module for distributing workload across API instances.
This module provides functionality to coordinate multiple running instances of the API.
"""

import asyncio
import time
import os
import socket
import json
import aiohttp
import hashlib
from typing import Dict, List, Any, Optional, Set, Callable, Awaitable
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Depends
from sqlalchemy import text
import random
from loguru import logger
from pydantic import BaseModel
from app.db.session import get_db_session, close_db_connections
import uuid

class APINode(BaseModel):
    """Represents an API server instance in the cluster"""
    id: str
    hostname: str
    ip_address: str
    port: int
    health_endpoint: str
    last_seen: datetime
    status: str = "active"  # active, degraded, unavailable
    load: float = 0.0  # 0.0-1.0 load factor
    metrics: Dict[str, Any] = {}

class LoadBalancer:
    """
    Manages a set of API nodes and distributes requests among them.
    Can operate in standalone mode or as part of a coordinated cluster.
    """
    def __init__(self, 
                 app: FastAPI = None,
                 health_check_interval: int = 30,
                 node_timeout: int = 90,
                 coordinator_mode: bool = False):
        self.app = app
        self.health_check_interval = health_check_interval
        self.node_timeout = node_timeout
        self.coordinator_mode = coordinator_mode
        
        # Node tracking
        self.nodes: Dict[str, APINode] = {}
        self.nodes_by_service: Dict[str, Set[str]] = {}
        
        # Health checking
        self.health_check_task = None
        self.is_running = False
        
        # Node identity
        self.node_id = str(uuid.uuid4())
        self.hostname = socket.gethostname()
        try:
            self.ip_address = socket.gethostbyname(self.hostname)
        except:
            self.ip_address = "127.0.0.1"
        self.port = int(os.environ.get("PORT", 8000))
        
        # Stats
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = datetime.now()
        self.avg_response_time = 0.0
        
        logger.info(f"Load Balancer initialized on {self.hostname} ({self.ip_address}:{self.port})")

    async def start(self):
        """Start load balancer operations"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Register with FastAPI if provided
        if self.app:
            @self.app.middleware("http")
            async def load_balancing_middleware(request: Request, call_next):
                """Track request metrics and potentially redirect requests"""
                self.request_count += 1
                self.last_request_time = datetime.now()
                start_time = time.time()
                
                try:
                    # If we're in coordinator mode, we might redirect the request
                    if self.coordinator_mode and self._should_redirect(request):
                        target_node = self._select_node_for_request(request)
                        if target_node and target_node.id != self.node_id:
                            # Redirect to another node
                            return await self._redirect_request(request, target_node)
                    
                    # Otherwise, process the request locally
                    response = await call_next(request)
                    
                    # Update metrics
                    elapsed = time.time() - start_time
                    self.avg_response_time = 0.95 * self.avg_response_time + 0.05 * elapsed
                    
                    return response
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Request error: {e}")
                    raise
        
        # Start health checking
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Load Balancer started")

    async def stop(self):
        """Stop load balancer operations"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel health checking
        if self.health_check_task:
            self.health_check_task.cancel()
            
        logger.info("Load Balancer stopped")
        
    async def register_node(self, 
                          hostname: str, 
                          ip_address: str, 
                          port: int,
                          services: List[str] = None,
                          health_endpoint: str = "/api/health") -> str:
        """Register a new API node with the load balancer"""
        node_id = self._generate_node_id(hostname, ip_address, port)
        
        node = APINode(
            id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=port,
            health_endpoint=health_endpoint,
            last_seen=datetime.now(),
            status="active",
            load=0.0,
            metrics={}
        )
        
        self.nodes[node_id] = node
        
        # Register services
        services = services or ["default"]
        for service in services:
            if service not in self.nodes_by_service:
                self.nodes_by_service[service] = set()
            self.nodes_by_service[service].add(node_id)
        
        logger.info(f"Registered node {hostname} ({ip_address}:{port}) with ID {node_id}")
        
        # Check node health immediately
        asyncio.create_task(self._check_node_health(node))
        
        return node_id

    async def unregister_node(self, node_id: str) -> bool:
        """Remove a node from the load balancer"""
        if node_id not in self.nodes:
            return False
            
        node = self.nodes.pop(node_id)
        
        # Remove from services
        for service, nodes in self.nodes_by_service.items():
            if node_id in nodes:
                nodes.remove(node_id)
        
        logger.info(f"Unregistered node {node.hostname} with ID {node_id}")
        return True

    async def update_node_status(self, node_id: str, status: str, metrics: Dict[str, Any] = None) -> bool:
        """Update a node's status and metrics"""
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        node.status = status
        node.last_seen = datetime.now()
        
        if metrics:
            node.metrics.update(metrics)
            
            # Calculate load factor (0.0-1.0)
            cpu_load = metrics.get("cpu_percent", 0) / 100
            mem_load = metrics.get("memory_percent", 0) / 100
            req_load = min(1.0, metrics.get("requests_per_second", 0) / 100)
            
            # Weighted average
            node.load = 0.4 * cpu_load + 0.3 * mem_load + 0.3 * req_load
        
        return True

    def is_active(self) -> bool:
        """Check if the load balancer is currently running"""
        return self.is_running
        
    def get_node_for_service(self, service: str = "default") -> Optional[APINode]:
        """Get a node to handle a specific service using load-based routing"""
        if service not in self.nodes_by_service:
            service = "default"
            
        if service not in self.nodes_by_service or not self.nodes_by_service[service]:
            return None
            
        # Get all active nodes for this service
        active_nodes = []
        for node_id in self.nodes_by_service[service]:
            node = self.nodes.get(node_id)
            if node and node.status == "active":
                active_nodes.append(node)
                
        if not active_nodes:
            # Try degraded nodes if no active ones
            for node_id in self.nodes_by_service[service]:
                node = self.nodes.get(node_id)
                if node and node.status == "degraded":
                    active_nodes.append(node)
        
        if not active_nodes:
            return None
            
        # Select node based on lowest load
        active_nodes.sort(key=lambda n: n.load)
        return active_nodes[0]

    def get_all_nodes(self) -> List[APINode]:
        """Get all registered nodes"""
        return list(self.nodes.values())

    def get_node_count(self) -> int:
        """Get the number of active nodes"""
        return len([n for n in self.nodes.values() if n.status != "unavailable"])

    async def _health_check_loop(self):
        """Periodically check the health of registered nodes"""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check all nodes
                for node_id, node in list(self.nodes.items()):
                    if node.id == self.node_id:
                        # Skip self
                        continue
                        
                    # Check if node is timed out
                    if (datetime.now() - node.last_seen).total_seconds() > self.node_timeout:
                        # Mark as unavailable
                        node.status = "unavailable"
                        logger.warning(f"Node {node.hostname} (ID: {node_id}) marked unavailable due to timeout")
                        continue
                    
                    # Check node health
                    await self._check_node_health(node)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_node_health(self, node: APINode):
        """Check the health of a specific node"""
        try:
            url = f"http://{node.ip_address}:{node.port}{node.health_endpoint}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Update node status and metrics
                        await self.update_node_status(
                            node.id, 
                            data.get("status", "active"),
                            data.get("metrics", {})
                        )
                    else:
                        # Node is degraded
                        node.status = "degraded"
        except Exception as e:
            # Node is potentially unavailable
            logger.warning(f"Failed to check health of node {node.hostname}: {e}")
            
            # Don't mark as unavailable immediately, let the timeout handle that
            if node.status == "active":
                node.status = "degraded"

    def _generate_node_id(self, hostname: str, ip_address: str, port: int) -> str:
        """Generate a unique ID for a node"""
        key = f"{hostname}:{ip_address}:{port}:{time.time()}"
        return hashlib.md5(key.encode()).hexdigest()

    def _should_redirect(self, request: Request) -> bool:
        """Determine if a request should be redirected to another node"""
        # Don't redirect health checks
        if request.url.path in ["/health", "/api/health"]:
            return False
            
        # If we're the only node, don't redirect
        if len(self.nodes) <= 1:
            return False
            
        # If we're overloaded, consider redirecting
        current_load = self._get_current_load()
        if current_load > 0.8:  # 80% load threshold
            # Find a node with significantly lower load
            for node in self.nodes.values():
                if node.id != self.node_id and node.status == "active" and node.load < current_load * 0.7:
                    return True
        
        return False

    def _select_node_for_request(self, request: Request) -> Optional[APINode]:
        """Select an appropriate node for a request"""
        # Determine the service from the path
        path = request.url.path
        service = "default"
        
        if path.startswith("/api/v1/guest"):
            service = "guest"
        elif path.startswith("/api/v1/customer"):
            service = "customer"
        elif path.startswith("/api/v1/employee"):
            service = "employee"
        elif path.startswith("/api/v1/document"):
            service = "document"
            
        return self.get_node_for_service(service)

    async def _redirect_request(self, request: Request, target_node: APINode):
        """Redirect a request to another node"""
        # TODO: Implement request forwarding
        # For now, just return a redirect response
        from fastapi.responses import RedirectResponse
        target_url = f"http://{target_node.ip_address}:{target_node.port}{request.url.path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
            
        return RedirectResponse(url=target_url)

    def _get_current_load(self) -> float:
        """Get the current load factor for this node"""
        import psutil
        
        # Calculate load factor (0.0-1.0)
        cpu_load = psutil.cpu_percent() / 100
        mem_load = psutil.virtual_memory().percent / 100
        
        # Calculate requests per second
        rps = 0
        if (datetime.now() - self.last_request_time).total_seconds() < 60:
            # Only calculate if we've had requests in the last minute
            rps = min(1.0, self.request_count / 100)
            
        # Weighted average
        return 0.4 * cpu_load + 0.3 * mem_load + 0.3 * rps

    async def get_health_data(self) -> Dict[str, Any]:
        """Get health and metrics data for this node"""
        import psutil
        
        # Calculate basic system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Calculate requests per second
        time_diff = (datetime.now() - self.last_request_time).total_seconds()
        requests_per_second = 0
        if time_diff > 0 and time_diff < 60:
            requests_per_second = self.request_count / min(time_diff, 60)
            
        # Database connection check
        db_status = "unknown"
        try:
            from app.db.session import get_db_engine
            engine = await get_db_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"
            
        return {
            "status": "active",
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "port": self.port,
            "uptime": "unknown",  # TODO: Track uptime
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_total": memory.total,
                "memory_used": memory.used,
                "memory_percent": memory.percent,
                "disk_total": disk.total,
                "disk_used": disk.used,
                "disk_percent": disk.percent,
                "requests_total": self.request_count,
                "errors_total": self.error_count,
                "requests_per_second": requests_per_second,
                "avg_response_time": self.avg_response_time
            },
            "database": {
                "status": db_status,
                "connections": "unknown"  # TODO: Get connection count
            },
            "cluster": {
                "nodes": len(self.nodes),
                "active_nodes": len([n for n in self.nodes.values() if n.status == "active"])
            }
        }

    async def get_local_load(self) -> Dict[str, Any]:
        """
        Get the current system load metrics for this node
        
        Returns:
            Dict containing cpu_percent, memory_percent, and other metrics
        """
        try:
            import psutil
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to get connection info if available
            try:
                from app.core.db_health_checker import get_db_health_checker
                db_health = get_db_health_checker()
                if db_health:
                    conn_stats = await db_health.get_connection_stats()
                    metrics["db_connections"] = conn_stats
            except Exception as e:
                logger.debug(f"Could not get DB connection stats: {e}")
                
            return metrics
        except ImportError:
            logger.warning("psutil not available for system metrics")
            return {"cpu_percent": 0, "memory_percent": 0}
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"cpu_percent": 0, "memory_percent": 0, "error": str(e)}

    async def get_best_node(self, request_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Get the best node for handling a request based on load and request type
        
        Args:
            request_type: Optional type of request (e.g., "standard_query", "premium_query")
            
        Returns:
            Dictionary with node details or None if no suitable node found
        """
        service = "customer_workflow"
        if request_type:
            if request_type == "premium_query":
                service = "premium_customer_workflow"
            elif request_type == "document_processing":
                service = "document_processing"
        
        # Find best node based on load
        node = self.get_node_for_service(service)
        
        # If no specialized service node found, try generic customer_workflow
        if not node and service != "customer_workflow":
            node = self.get_node_for_service("customer_workflow")
        
        # If still no node, try default
        if not node:
            node = self.get_node_for_service("default")
            
        if not node:
            return None
            
        # Return node details as dict
        return {
            "id": node.id,
            "hostname": node.hostname,
            "ip_address": node.ip_address,
            "port": node.port,
            "load": node.load,
            "status": node.status
        }

    def get_node_id(self) -> str:
        """Get the ID of this load balancer node"""
        return self.node_id
        
    async def forward_request(self, node: Dict[str, Any], endpoint: str, method: str = "GET", payload: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Forward a request to another node
        
        Args:
            node: Node details dictionary (from get_best_node)
            endpoint: API endpoint to call
            method: HTTP method to use
            payload: Request payload data
            
        Returns:
            Response data or None if request failed
        """
        import aiohttp
        
        try:
            url = f"http://{node['ip_address']}:{node['port']}{endpoint}"
            logger.debug(f"Forwarding request to {url}")
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.error(f"Error forwarding request: {response.status}")
                            return None
                elif method.upper() == "POST":
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.error(f"Error forwarding request: {response.status}")
                            return None
                else:
                    logger.error(f"Unsupported method: {method}")
                    return None
        except Exception as e:
            logger.exception(f"Error forwarding request to node {node['hostname']}: {e}")
            return None

# Global load balancer instance
load_balancer = None

def init_load_balancer(app: FastAPI = None, coordinator_mode: bool = False) -> LoadBalancer:
    """Initialize the global load balancer"""
    global load_balancer
    if load_balancer is None:
        load_balancer = LoadBalancer(app=app, coordinator_mode=coordinator_mode)
    return load_balancer

def get_load_balancer() -> LoadBalancer:
    """Get the global load balancer instance"""
    if load_balancer is None:
        raise RuntimeError("Load balancer not initialized")
    return load_balancer

# FastAPI dependency
async def get_load_balancer_dep() -> LoadBalancer:
    """FastAPI dependency for getting the load balancer instance"""
    if load_balancer is None:
        raise RuntimeError("Load balancer not initialized")
    return load_balancer

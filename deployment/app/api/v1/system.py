"""
System Monitoring API Routes

This module provides API endpoints for monitoring system health, database connections,
load balancing, and queue processing.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import time
import json
import psutil
import os
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime

from app.core.db_health_checker import get_db_health_checker, DatabaseHealthChecker
from app.core.load_balancer import get_load_balancer, LoadBalancer
from app.core.queue_manager import get_queue_manager, QueueManager
from app.db.session import get_db_session, close_db_connections, AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

router = APIRouter(prefix="/api/system", tags=["system"])

@router.get("/health")
async def health_check(
    detailed: bool = False,
    db_health: DatabaseHealthChecker = Depends(get_db_health_checker),
    load_balancer: Optional[LoadBalancer] = Depends(get_load_balancer),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Check system health including database, API, and resources.
    Set detailed=true for comprehensive metrics.
    """
    try:
        # Basic system info
        system_info = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname().nodename,
        }
        
        # Resource usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent
        }
        
        # Overall status based on resource usage
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
            system_info["status"] = "critical"
        elif cpu_percent > 75 or memory.percent > 80 or disk.percent > 85:
            system_info["status"] = "warning"
            
        # Database check - simple ping
        db_status = "ok"
        try:
            await db.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception as e:
            db_status = "error"
            system_info["status"] = "warning"
            resources["db_error"] = str(e)
            
        resources["database_status"] = db_status
        
        # Prepare response
        response = {
            "info": system_info,
            "resources": resources
        }
        
        # Add load balancer info if available
        if load_balancer:
            lb_info = {
                "nodes": load_balancer.get_node_count(),
                "request_count": load_balancer.request_count,
                "status": "ok"
            }
            response["load_balancer"] = lb_info
        
        # Add detailed metrics if requested
        if detailed:
            # Detailed system info
            detailed_resources = {
                "cpu": {
                    "percent": cpu_percent,
                    "cores": psutil.cpu_count(),
                    "load": os.getloadavg()
                },
                "memory": {
                    "total": memory.total,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "percent": disk.percent
                }
            }
            
            # Process info
            process = psutil.Process()
            process_info = {
                "pid": process.pid,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "threads": len(process.threads()),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
            
            # Database detailed info from health checker
            db_health_report = await db_health.get_health_report()
            
            response["detailed"] = {
                "system": detailed_resources,
                "process": process_info,
                "database": db_health_report
            }
            
            # Add load balancer detailed info if available
            if load_balancer:
                response["detailed"]["load_balancer"] = await load_balancer.get_health_data()
        
        return response
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/database")
async def database_status(
    db_health: DatabaseHealthChecker = Depends(get_db_health_checker),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get detailed database status and connection information.
    """
    try:
        # Get health report
        health_report = await db_health.get_health_report()
        
        # Run integrity check
        integrity = await db_health.run_integrity_check()
        
        # Get table usage
        usage = await db_health.analyze_table_usage()
        
        return {
            "status": health_report["status"],
            "health": health_report,
            "integrity": integrity,
            "usage": usage,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Database status error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

@router.post("/database/cleanup")
async def database_cleanup(
    db_health: DatabaseHealthChecker = Depends(get_db_health_checker)
):
    """
    Clean up idle database connections and perform maintenance.
    """
    try:
        # Close idle connections
        closed = await db_health.cleanup_idle_connections()
        
        # Force cleanup via engine
        await close_db_connections()
        
        return {
            "status": "success",
            "connections_closed": closed,
            "message": "Database connections cleaned up",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Database cleanup error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

@router.get("/database/performance")
async def database_performance(
    db_health: DatabaseHealthChecker = Depends(get_db_health_checker)
):
    """
    Get database performance metrics and graph.
    """
    try:
        # Get performance graph
        graph_data = await db_health.get_performance_graph()
        
        if graph_data:
            return StreamingResponse(content=graph_data, media_type="image/png")
        else:
            return JSONResponse(
                content={"status": "error", "message": "No performance data available"}
            )
    except Exception as e:
        logger.error(f"Database performance error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

@router.get("/load-balancer")
async def load_balancer_status(
    load_balancer: LoadBalancer = Depends(get_load_balancer)
):
    """
    Get load balancer status and node information.
    """
    try:
        health_data = await load_balancer.get_health_data()
        nodes = load_balancer.get_all_nodes()
        
        return {
            "status": "ok",
            "health": health_data,
            "nodes": [
                {
                    "id": node.id,
                    "hostname": node.hostname,
                    "ip_address": node.ip_address,
                    "port": node.port,
                    "status": node.status,
                    "load": node.load,
                    "last_seen": node.last_seen.isoformat()
                } for node in nodes
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Load balancer status error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

@router.get("/queue")
async def queue_status(
    queue_manager: Optional[QueueManager] = Depends(get_queue_manager)
):
    """
    Get queue status and statistics.
    """
    if not queue_manager:
        return JSONResponse(
            content={"status": "not_available", "message": "Queue manager not initialized"}
        )
        
    try:
        stats = queue_manager.get_stats()
        
        return {
            "status": "ok",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

@router.post("/queue/task/{task_id}/cancel")
async def cancel_queue_task(
    task_id: str,
    queue_manager: QueueManager = Depends(get_queue_manager)
):
    """
    Cancel a task in the queue.
    """
    try:
        success = await queue_manager.cancel_task(task_id)
        
        if success:
            return {"status": "success", "message": f"Task {task_id} cancelled"}
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": f"Task {task_id} could not be cancelled"}
            )
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Cancel task error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

@router.get("/resources")
async def system_resources():
    """
    Get detailed system resource usage.
    """
    try:
        # CPU info
        cpu_info = {
            "percent": psutil.cpu_percent(interval=1),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(),
            "load_avg": os.getloadavg(),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # Swap info
        swap = psutil.swap_memory()
        swap_info = {
            "total": swap.total,
            "used": swap.used,
            "percent": swap.percent
        }
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
        
        # Network info
        net_io = psutil.net_io_counters()
        net_info = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "err_in": net_io.errin,
            "err_out": net_io.errout
        }
        
        # Process info
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "memory_info": process.memory_info()._asdict(),
            "num_threads": len(process.threads()),
            "num_fds": len(process.open_files()),
            "num_connections": len(process.connections()),
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
        }
        
        return {
            "status": "ok",
            "cpu": cpu_info,
            "memory": memory_info,
            "swap": swap_info,
            "disk": disk_info,
            "network": net_info,
            "process": process_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System resources error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

@router.post("/log-level/{level}")
async def set_log_level(level: str):
    """
    Change the application log level at runtime.
    Valid levels: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    valid_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level = level.upper()
    
    if level not in valid_levels:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": f"Invalid log level. Valid levels are: {', '.join(valid_levels)}"}
        )
        
    try:
        # Update loguru level
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=level)
        
        return {
            "status": "success",
            "message": f"Log level set to {level}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": str(e)}
        )

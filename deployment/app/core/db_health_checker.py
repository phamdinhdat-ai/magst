
import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from app.db.session import get_db_engine, AsyncSessionLocal, close_db_connections

class DatabaseHealthChecker:
    """
    Monitors database health and performance, provides metrics and cleanup functions.
    Fixed version that properly manages database connections.
    """
    def __init__(self, 
                 check_interval: int = 60,
                 connection_timeout: int = 30,
                 max_idle_connections: int = 5,
                 performance_threshold_ms: int = 500):
        self.check_interval = check_interval
        self.connection_timeout = connection_timeout
        self.max_idle_connections = max_idle_connections
        self.performance_threshold_ms = performance_threshold_ms
        
        # Health check state
        self.is_running = False
        self.health_check_task = None
        
        # Performance metrics
        self.query_history: List[Dict[str, Any]] = []
        self.slow_queries: List[Dict[str, Any]] = []
        self.connection_metrics = {
            "total": 0,
            "active": 0,
            "idle": 0,
            "max_connections": 20,  # Default, will be updated
            "connection_errors": 0,
            "timeout_errors": 0
        }
        
        # Status
        self.last_check_time = None
        self.status = "unknown"
        
        logger.info(f"Database Health Checker initialized with {check_interval}s interval")

    async def start(self):
        """Start health check monitoring"""
        if self.is_running:
            return
            
        self.is_running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Database Health Checker started")

    async def stop(self):
        """Stop health check monitoring"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Database Health Checker stopped")

    async def get_health_report(self) -> Dict[str, Any]:
        """Get a complete database health report"""
        await self._run_health_check()
        
        # Connection pool metrics
        try:
            engine = await get_db_engine()
            pool_status = {
                "size": engine.pool.size(),
                "checkedout": engine.pool.checkedout(),
                "overflow": engine.pool.overflow(),
                "checkedin": engine.pool.checkedin(),
            }
        except Exception as e:
            logger.error(f"Error getting pool status: {e}")
            pool_status = {"error": str(e)}
        
        # Create health report
        report = {
            "status": self.status,
            "last_checked": self.last_check_time,
            "connection_metrics": {
                **self.connection_metrics,
                "pool_status": pool_status
            },
            "performance_metrics": {
                "avg_query_time_ms": self._calculate_avg_query_time(),
                "slow_query_count": len(self.slow_queries),
                "queries_per_minute": self._calculate_queries_per_minute()
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report

    async def cleanup_idle_connections(self) -> int:
        """Close idle database connections"""
        try:
            engine = await get_db_engine()
            
            # Get database type
            db_url = str(engine.url)
            if "postgresql" in db_url:
                return await self._cleanup_postgres_connections()
            elif "sqlite" in db_url:
                # SQLite doesn't need connection cleanup
                return 0
            else:
                logger.warning(f"Connection cleanup not implemented for this database type: {db_url}")
                return 0
        except Exception as e:
            logger.error(f"Error in cleanup_idle_connections: {e}")
            return 0

    async def run_integrity_check(self) -> Dict[str, Any]:
        """Run database integrity checks - FIXED VERSION"""
        results = {"status": "ok", "details": {}}
        
        try:
            engine = await get_db_engine()
            db_url = str(engine.url)
            
            # Use proper async context manager
            async with AsyncSessionLocal() as session:
                try:
                    if "postgresql" in db_url:
                        results = await self._check_postgres_integrity(session)
                    elif "sqlite" in db_url:
                        results = await self._check_sqlite_integrity(session)
                    
                    # Ensure session is properly committed/closed
                    if session.in_transaction():
                        await session.commit()
                        
                except Exception as e:
                    logger.error(f"Error during integrity check: {e}")
                    await session.rollback()
                    results = {"status": "error", "error": str(e)}
                    
        except Exception as e:
            logger.error(f"Error creating session for integrity check: {e}")
            results = {"status": "error", "error": str(e)}
            
        return results

    async def _check_postgres_integrity(self, session) -> Dict[str, Any]:
        """Check PostgreSQL integrity - extracted method"""
        results = {"status": "ok", "details": {}}
        
        try:
            # Check for bloated tables
            query = text("""
            SELECT schemaname, relname, n_dead_tup, n_live_tup,
            round(n_dead_tup * 100.0 / (n_live_tup + n_dead_tup + 0.000001), 2) AS dead_percentage
            FROM pg_stat_user_tables
            WHERE n_dead_tup > 10000
            OR (round(n_dead_tup * 100.0 / (n_live_tup + n_dead_tup + 0.000001), 2) > 20)
            ORDER BY dead_percentage DESC, n_dead_tup DESC;
            """)
            result = await session.execute(query)
            bloated_tables = [dict(r._mapping) for r in result]
            
            # Check for long running queries
            query = text("""
            SELECT pid, now() - query_start as duration, state, query
            FROM pg_stat_activity
            WHERE state = 'active' AND now() - query_start > interval '5 seconds'
            AND query NOT LIKE '%pg_stat_activity%'
            ORDER BY duration DESC;
            """)
            result = await session.execute(query)
            long_queries = [dict(r._mapping) for r in result]
            
            # Check for locks
            query = text("""
            SELECT relation::regclass, mode, pid, pg_blocking_pids(pid) as blocked_by
            FROM pg_locks
            WHERE NOT granted;
            """)
            result = await session.execute(query)
            locks = [dict(r._mapping) for r in result]
            
            results["details"] = {
                "bloated_tables": bloated_tables,
                "long_running_queries": long_queries,
                "locks": locks
            }
            
            recommendations = []
            if bloated_tables:
                results["status"] = "warning"
                recommendations.append("Consider running VACUUM on bloated tables")
            
            if long_queries:
                results["status"] = "warning" 
                recommendations.append("Check for slow queries")
                
            if locks:
                results["status"] = "warning"
                recommendations.append("Check for locks")
                
            results["recommendations"] = recommendations
            
        except Exception as e:
            logger.error(f"Error in PostgreSQL integrity check: {e}")
            results = {"status": "error", "error": str(e)}
            
        return results

    async def _check_sqlite_integrity(self, session) -> Dict[str, Any]:
        """Check SQLite integrity - extracted method"""
        results = {"status": "ok", "details": {}}
        
        try:
            # SQLite integrity check
            result = await session.execute(text("PRAGMA integrity_check;"))
            integrity = result.scalar()
            
            # Check for indexes
            result = await session.execute(text("PRAGMA index_list;"))
            indexes = [dict(r._mapping) for r in result]
            
            results["details"] = {
                "integrity_check": integrity,
                "indexes": indexes
            }
            
            if integrity != "ok":
                results["status"] = "error"
                results["recommendations"] = ["Database integrity check failed"]
                
        except Exception as e:
            logger.error(f"Error in SQLite integrity check: {e}")
            results = {"status": "error", "error": str(e)}
            
        return results

    async def analyze_table_usage(self, tables: List[str] = None) -> Dict[str, Any]:
        """Analyze usage statistics for database tables - FIXED VERSION"""
        results = {}
        
        try:
            engine = await get_db_engine()
            db_url = str(engine.url)
            
            async with AsyncSessionLocal() as session:
                try:
                    if "postgresql" in db_url:
                        results = await self._analyze_postgres_tables(session, tables)
                    elif "sqlite" in db_url:
                        results = await self._analyze_sqlite_tables(session, tables)
                    
                    # Ensure proper session cleanup
                    if session.in_transaction():
                        await session.commit()
                        
                except Exception as e:
                    logger.error(f"Error during table analysis: {e}")
                    await session.rollback()
                    results = {"error": str(e)}
                    
        except Exception as e:
            logger.error(f"Error creating session for table analysis: {e}")
            results = {"error": str(e)}
            
        return results

    async def _analyze_postgres_tables(self, session, tables: List[str] = None) -> Dict[str, Any]:
        """Analyze PostgreSQL tables"""
        # If no tables specified, get all tables
        if not tables:
            query = text("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY tablename;
            """)
            result = await session.execute(query)
            tables = [r[0] for r in result]
        
        # Get statistics for each table
        stats = {}
        for table in tables:
            try:
                # Get table size
                query = text("""
                SELECT pg_size_pretty(pg_total_relation_size(:table)) as size,
                pg_total_relation_size(:table) as size_bytes
                """)
                result = await session.execute(query, {"table": table})
                size_info = dict(result.fetchone()._mapping)
                
                # Get row count - use safe approach
                query = text(f"SELECT COUNT(*) FROM {table}")
                result = await session.execute(query)
                row_count = result.scalar()
                
                # Get table statistics
                query = text("""
                SELECT seq_scan, seq_tup_read, idx_scan, idx_tup_fetch, 
                n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup
                FROM pg_stat_user_tables WHERE relname = :table
                """)
                result = await session.execute(query, {"table": table})
                row = result.fetchone()
                table_stats = dict(row._mapping) if row else {}
                
                stats[table] = {
                    "size": size_info.get("size"),
                    "size_bytes": size_info.get("size_bytes"),
                    "row_count": row_count,
                    "stats": table_stats
                }
            except Exception as e:
                logger.error(f"Error analyzing table {table}: {e}")
                stats[table] = {"error": str(e)}
        
        # Generate recommendations
        recommendations = []
        for table, data in stats.items():
            if "error" in data:
                continue
                
            table_stats = data.get("stats", {})
            
            # Check for tables with many dead tuples
            if table_stats.get("n_dead_tup", 0) > 10000 and table_stats.get("n_live_tup", 0) > 0:
                dead_percent = table_stats["n_dead_tup"] * 100.0 / (table_stats["n_live_tup"] + table_stats["n_dead_tup"])
                if dead_percent > 20:
                    recommendations.append(f"Table '{table}' has {dead_percent:.1f}% dead tuples - consider running VACUUM")
            
            # Check for tables with many sequential scans
            if table_stats.get("seq_scan", 0) > 100 and table_stats.get("idx_scan", 0) < table_stats.get("seq_scan", 0) / 10:
                recommendations.append(f"Table '{table}' has many sequential scans - consider adding indexes")
        
        return {
            "tables": stats,
            "recommendations": recommendations
        }

    async def _analyze_sqlite_tables(self, session, tables: List[str] = None) -> Dict[str, Any]:
        """Analyze SQLite tables"""
        if not tables:
            query = text("SELECT name FROM sqlite_master WHERE type='table';")
            result = await session.execute(query)
            tables = [r[0] for r in result]
        
        stats = {}
        for table in tables:
            try:
                # Get row count
                query = text(f"SELECT COUNT(*) FROM {table}")
                result = await session.execute(query)
                row_count = result.scalar()
                
                stats[table] = {"row_count": row_count}
            except Exception as e:
                logger.error(f"Error analyzing SQLite table {table}: {e}")
                stats[table] = {"error": str(e)}
        
        return {"tables": stats}

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get current database connection statistics - FIXED VERSION"""
        try:
            engine = await get_db_engine()
            pool_metrics = {
                "total": engine.pool.size(),
                "checked_out": engine.pool.checkedout(),
                "overflow": engine.pool.overflow(),
                "checked_in": engine.pool.checkedin(),
            }
            
            # Update our stored metrics
            self.connection_metrics.update({
                "total": pool_metrics["total"],
                "active": pool_metrics["checked_out"],
                "idle": pool_metrics["checked_in"]
            })
            
            return {
                **self.connection_metrics,
                "pool_status": pool_metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"error": str(e), "total": 0, "active": 0, "idle": 0}

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get current database pool statistics (alias for get_connection_stats)"""
        return await self.get_connection_stats()

    async def _health_check_loop(self):
        """Periodically run database health checks - FIXED VERSION"""
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Use timeout to prevent hanging
                try:
                    await asyncio.wait_for(self._run_health_check(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.error("Health check timed out")
                    self.status = "error"
                    self.connection_metrics["timeout_errors"] += 1
                
                # Log status
                logger.info(f"Database health: {self.status} - "
                           f"{self.connection_metrics['active']} active, "
                           f"{self.connection_metrics['idle']} idle connections")
                
                # If we have too many idle connections, clean them up
                if self.connection_metrics['idle'] > self.max_idle_connections:
                    try:
                        closed = await self.cleanup_idle_connections()
                        if closed > 0:
                            logger.info(f"Closed {closed} idle database connections")
                    except Exception as e:
                        logger.error(f"Error cleaning up connections: {e}")
                        
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                self.status = "error"

    async def _run_health_check(self):
        """Run a single health check - FIXED VERSION"""
        self.last_check_time = datetime.now()
        
        try:
            engine = await get_db_engine()
            db_url = str(engine.url)
            
            # Run a simple query to check connectivity with proper session management
            start_time = time.time()
            
            async with AsyncSessionLocal() as session:
                try:
                    await session.execute(text("SELECT 1"))
                    query_time = (time.time() - start_time) * 1000  # in ms
                    
                    # Update status
                    self.status = "healthy" if query_time < self.connection_timeout * 1000 else "degraded"
                    
                    # Check connection metrics
                    if "postgresql" in db_url:
                        await self._check_postgres_metrics(session)
                    elif "sqlite" in db_url:
                        # SQLite has limited connection metrics
                        self.connection_metrics.update({
                            "total": 1,
                            "active": 1 if engine.pool.checkedout() > 0 else 0,
                            "idle": engine.pool.checkedin()
                        })
                    
                    # Ensure proper session cleanup
                    if session.in_transaction():
                        await session.commit()
                        
                except Exception as e:
                    logger.error(f"Error in health check query: {e}")
                    await session.rollback()
                    raise
            
            # Record this health check
            self.query_history.append({
                "timestamp": datetime.now(),
                "query": "SELECT 1",
                "duration_ms": query_time
            })
            
            # Keep history limited
            if len(self.query_history) > 10000:
                self.query_history = self.query_history[-5000:]
                
            # Check if this was a slow query
            if query_time > self.performance_threshold_ms:
                self.slow_queries.append({
                    "timestamp": datetime.now(),
                    "query": "SELECT 1 (health check)",
                    "duration_ms": query_time
                })
                logger.warning(f"Slow health check query: {query_time:.2f}ms")
            
        except Exception as e:
            self.status = "error"
            self.connection_metrics["connection_errors"] += 1
            logger.error(f"Database health check failed: {e}")

    async def _check_postgres_metrics(self, session):
        """Check PostgreSQL specific metrics - FIXED VERSION"""
        try:
            # Get connection counts
            query = text("""
            SELECT
                count(*) as total,
                sum(CASE WHEN state = 'active' THEN 1 ELSE 0 END) as active,
                sum(CASE WHEN state = 'idle' THEN 1 ELSE 0 END) as idle,
                setting::int as max_connections
            FROM pg_stat_activity, pg_settings
            WHERE pg_settings.name = 'max_connections'
            GROUP BY setting
            """)
            
            result = await session.execute(query)
            row = result.fetchone()
            if row:
                metrics = dict(row._mapping)
                self.connection_metrics.update({
                    "total": metrics.get("total", 0),
                    "active": metrics.get("active", 0),
                    "idle": metrics.get("idle", 0),
                    "max_connections": metrics.get("max_connections", 20)
                })
        except Exception as e:
            logger.error(f"Error checking PostgreSQL metrics: {e}")

    async def _cleanup_postgres_connections(self) -> int:
        """Close idle PostgreSQL connections - FIXED VERSION"""
        closed = 0
        
        try:
            async with AsyncSessionLocal() as session:
                try:
                    # Find idle connections older than the timeout
                    query = text("""
                    SELECT pid FROM pg_stat_activity
                    WHERE state = 'idle'
                    AND application_name = 'guest_workflow_api'
                    AND now() - state_change > :timeout_interval
                    """)
                    
                    result = await session.execute(query, {
                        "timeout_interval": f"{self.connection_timeout} seconds"
                    })
                    
                    idle_pids = [r[0] for r in result]
                    
                    # Terminate each connection
                    for pid in idle_pids:
                        try:
                            await session.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
                            closed += 1
                        except Exception as e:
                            logger.error(f"Failed to terminate connection {pid}: {e}")
                    
                    await session.commit()
                    
                except Exception as e:
                    logger.error(f"Error in cleanup query: {e}")
                    await session.rollback()
                    raise
                
        except Exception as e:
            logger.error(f"Error cleaning up connections: {e}")
            
        return closed

    def _calculate_avg_query_time(self) -> float:
        """Calculate average query execution time"""
        if not self.query_history:
            return 0.0
            
        # Calculate average of the last 100 queries
        recent = self.query_history[-100:]
        return sum(q["duration_ms"] for q in recent) / len(recent)

    def _calculate_queries_per_minute(self) -> float:
        """Calculate queries per minute based on history"""
        if len(self.query_history) < 2:
            return 0.0
            
        # Get time range
        start_time = self.query_history[0]["timestamp"]
        end_time = self.query_history[-1]["timestamp"]
        
        # Calculate time difference in minutes
        time_diff = (end_time - start_time).total_seconds() / 60
        
        if time_diff <= 0:
            return 0.0
            
        return len(self.query_history) / time_diff

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        # Check connection usage
        if self.connection_metrics["total"] > self.connection_metrics["max_connections"] * 0.8:
            recommendations.append(f"High connection usage: {self.connection_metrics['total']} of {self.connection_metrics['max_connections']} connections used")
            
        # Check for many idle connections
        if self.connection_metrics["idle"] > 10:
            recommendations.append(f"Many idle connections: {self.connection_metrics['idle']} idle connections")
            
        # Check for slow queries
        if len(self.slow_queries) > 5:
            recent_slow = len([q for q in self.slow_queries if (datetime.now() - q["timestamp"]).total_seconds() < 3600])
            if recent_slow > 0:
                recommendations.append(f"Performance issue: {recent_slow} slow queries in the last hour")
                
        # Check for connection errors
        if self.connection_metrics["connection_errors"] > 0:
            recommendations.append(f"Connection issues: {self.connection_metrics['connection_errors']} connection errors")
            
        return recommendations

    async def get_performance_graph(self) -> Optional[bytes]:
        """Generate a performance graph of query execution times"""
        if not self.query_history:
            return None
        
        try:
            # Convert to pandas for visualization
            df = pd.DataFrame(self.query_history)
            
            # Only keep the last 1000 queries for visualization
            if len(df) > 1000:
                df = df.tail(1000)
            
            # Plot query times
            plt.figure(figsize=(10, 6))
            plt.plot(df['timestamp'], df['duration_ms'], 'b-', alpha=0.5)
            plt.axhline(y=self.performance_threshold_ms, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Time')
            plt.ylabel('Query Duration (ms)')
            plt.title('Database Query Performance')
            plt.grid(True)
            plt.legend()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Cleanup matplotlib resources
            plt.close()
            
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error generating performance graph: {e}")
            return None

# Global database health checker instance
db_health_checker = None

def init_db_health_checker(check_interval: int = 60) -> DatabaseHealthChecker:
    """Initialize the global database health checker"""
    global db_health_checker
    if db_health_checker is None:
        db_health_checker = DatabaseHealthChecker(check_interval=check_interval)
    return db_health_checker

def get_db_health_checker() -> DatabaseHealthChecker:
    """Get the global database health checker instance"""
    if db_health_checker is None:
        raise RuntimeError("Database health checker not initialized")
    return db_health_checker

# FastAPI dependency
async def get_db_health_checker_dep() -> DatabaseHealthChecker:
    """FastAPI dependency for getting the database health checker instance"""
    if db_health_checker is None:
        raise RuntimeError("Database health checker not initialized")
    return db_health_checker
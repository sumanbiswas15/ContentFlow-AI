"""
Health check endpoints for ContentFlow AI.

This module provides health check and system status endpoints
for monitoring and diagnostics.
"""

from datetime import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    database_connected: bool
    uptime_seconds: float


class DetailedHealthResponse(HealthResponse):
    """Detailed health response with component status."""
    components: dict


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        # Test database connection
        db = get_database()
        await db.command("ping")
        db_connected = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_connected = False
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.utcnow(),
        database_connected=db_connected,
        uptime_seconds=0.0  # TODO: Implement actual uptime tracking
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with component status."""
    components = {}
    
    # Check database
    try:
        db = get_database()
        await db.command("ping")
        components["database"] = {"status": "healthy", "message": "Connected"}
    except Exception as e:
        components["database"] = {"status": "unhealthy", "message": str(e)}
    
    # TODO: Add checks for other components
    # - Redis connection
    # - AI service availability
    # - Storage backend
    # - Job queue status
    
    overall_status = "healthy" if all(
        comp["status"] == "healthy" for comp in components.values()
    ) else "degraded"
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        database_connected=components.get("database", {}).get("status") == "healthy",
        uptime_seconds=0.0,  # TODO: Implement actual uptime tracking
        components=components
    )


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        db = get_database()
        await db.command("ping")
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}, 503


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}
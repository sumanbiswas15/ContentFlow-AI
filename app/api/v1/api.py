"""
Main API router for ContentFlow AI v1.

This module aggregates all API endpoints and provides the main router
for the FastAPI application.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, orchestrator, jobs, auth, auth_basic, content, engines, engagement

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth_basic.router, prefix="/auth", tags=["authentication"])  # Login/Register (must be first)
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])  # API Key management
api_router.include_router(orchestrator.router, prefix="/orchestrator", tags=["orchestration"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(content.router, prefix="/content", tags=["content"])
api_router.include_router(engines.router, prefix="/engines", tags=["engines"])
api_router.include_router(engagement.router, prefix="/engagement", tags=["engagement"])
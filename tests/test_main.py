"""
Tests for the main application module.

This module tests the FastAPI application initialization,
health endpoints, and basic functionality.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test the root endpoint returns correct information."""
    response = await client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["message"] == "ContentFlow AI Platform"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_api_health_endpoint(client):
    """Test the API health check endpoint."""
    response = await client.get("/api/v1/health/")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "database_connected" in data


@pytest.mark.asyncio
async def test_detailed_health_endpoint(client):
    """Test the detailed health check endpoint."""
    response = await client.get("/api/v1/health/detailed")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "database" in data["components"]


@pytest.mark.asyncio
async def test_readiness_endpoint(client):
    """Test the readiness probe endpoint."""
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ready"


@pytest.mark.asyncio
async def test_liveness_endpoint(client: AsyncClient):
    """Test the liveness probe endpoint."""
    response = await client.get("/api/v1/health/live")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"
"""
Pytest configuration and fixtures for ContentFlow AI tests.

This module provides common test fixtures and configuration
for the test suite.
"""

import asyncio
import pytest
from typing import AsyncGenerator
from httpx import AsyncClient
from motor.motor_asyncio import AsyncIOMotorClient

from app.main import app
from app.core.config import settings
from app.core.database import db


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db():
    """Create a test database connection."""
    # Use a test database
    test_db_name = f"{settings.MONGODB_DATABASE}_test"
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    database = client[test_db_name]
    
    # Set up test database
    db.client = client
    db.database = database
    
    yield database
    
    # Clean up test database
    await client.drop_database(test_db_name)
    client.close()


@pytest.fixture
async def client(test_db):
    """Create a test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_content_data():
    """Sample content data for testing."""
    return {
        "type": "text",
        "title": "Test Content",
        "content": "This is test content for validation.",
        "user_id": "test_user_123",
        "tags": ["test", "sample"]
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "hashed_password": "hashed_password_here"
    }


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "job_type": "content_generation",
        "engine": "text_intelligence",
        "operation": "generate_blog",
        "parameters": {
            "topic": "AI in content creation",
            "length": "medium",
            "tone": "professional"
        },
        "user_id": "test_user_123"
    }
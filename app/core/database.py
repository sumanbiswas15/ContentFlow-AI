"""
Database connection and management for ContentFlow AI.

This module handles MongoDB connection using Motor async driver
and provides database access patterns for the application.
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings

logger = logging.getLogger(__name__)


class Database:
    """Database connection manager."""
    
    client: Optional[AsyncIOMotorClient] = None
    database: Optional[AsyncIOMotorDatabase] = None


db = Database()


async def connect_to_mongo():
    """Create database connection."""
    try:
        logger.info("Connecting to MongoDB...")
        db.client = AsyncIOMotorClient(settings.MONGODB_URL)
        db.database = db.client[settings.MONGODB_DATABASE]
        
        # Test the connection
        await db.client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DATABASE}")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close database connection."""
    if db.client:
        logger.info("Closing MongoDB connection...")
        db.client.close()
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance."""
    if db.database is None:
        raise RuntimeError("Database not initialized. Call connect_to_mongo() first.")
    return db.database


async def create_indexes():
    """Create database indexes for optimal performance."""
    database = get_database()
    
    # Content items indexes
    await database.content_items.create_index("id", unique=True)
    await database.content_items.create_index("type")
    await database.content_items.create_index("workflow_state")
    await database.content_items.create_index("created_at")
    await database.content_items.create_index("tags")
    await database.content_items.create_index([("title", "text"), ("content", "text")])
    
    # Async jobs indexes
    await database.async_jobs.create_index("id", unique=True)
    await database.async_jobs.create_index("status")
    await database.async_jobs.create_index("content_id")
    await database.async_jobs.create_index("created_at")
    await database.async_jobs.create_index("engine")
    
    # User sessions indexes (for creative assistant)
    await database.creative_sessions.create_index("session_id", unique=True)
    await database.creative_sessions.create_index("user_id")
    await database.creative_sessions.create_index("created_at")
    
    # Cost tracking indexes
    await database.cost_tracking.create_index("user_id")
    await database.cost_tracking.create_index("date")
    await database.cost_tracking.create_index("operation_type")
    
    logger.info("Database indexes created successfully")
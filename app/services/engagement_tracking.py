"""
Engagement tracking service for real-time analytics.

This module provides real-time engagement tracking for content items.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
from app.core.database import get_database
from app.models.content import EngagementMetrics

logger = logging.getLogger(__name__)


class EngagementTracker:
    """Service for tracking content engagement in real-time."""
    
    @staticmethod
    async def track_view(content_id: str, user_id: Optional[str] = None) -> bool:
        """Track a content view."""
        try:
            db = get_database()
            
            # Update content engagement metrics
            await db.content_items.update_one(
                {"_id": content_id},
                {
                    "$inc": {"engagement_metrics.views": 1},
                    "$set": {"engagement_metrics.last_viewed_at": datetime.utcnow()}
                }
            )
            
            # Log view event
            await db.engagement_events.insert_one({
                "content_id": content_id,
                "user_id": user_id or "anonymous",
                "event_type": "view",
                "timestamp": datetime.utcnow()
            })
            
            logger.info(f"Tracked view for content {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track view: {e}")
            return False
    
    @staticmethod
    async def track_like(content_id: str, user_id: str) -> bool:
        """Track a content like."""
        try:
            db = get_database()
            
            # Check if already liked
            existing = await db.engagement_events.find_one({
                "content_id": content_id,
                "user_id": user_id,
                "event_type": "like"
            })
            
            if existing:
                # Unlike
                await db.content_items.update_one(
                    {"_id": content_id},
                    {"$inc": {"engagement_metrics.likes": -1}}
                )
                await db.engagement_events.delete_one({"_id": existing["_id"]})
                logger.info(f"Removed like for content {content_id}")
                return False
            else:
                # Like
                await db.content_items.update_one(
                    {"_id": content_id},
                    {"$inc": {"engagement_metrics.likes": 1}}
                )
                await db.engagement_events.insert_one({
                    "content_id": content_id,
                    "user_id": user_id,
                    "event_type": "like",
                    "timestamp": datetime.utcnow()
                })
                logger.info(f"Tracked like for content {content_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to track like: {e}")
            return False
    
    @staticmethod
    async def track_share(content_id: str, user_id: Optional[str] = None, platform: Optional[str] = None) -> bool:
        """Track a content share."""
        try:
            db = get_database()
            
            await db.content_items.update_one(
                {"_id": content_id},
                {"$inc": {"engagement_metrics.shares": 1}}
            )
            
            await db.engagement_events.insert_one({
                "content_id": content_id,
                "user_id": user_id or "anonymous",
                "event_type": "share",
                "platform": platform,
                "timestamp": datetime.utcnow()
            })
            
            logger.info(f"Tracked share for content {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track share: {e}")
            return False
    
    @staticmethod
    async def track_comment(content_id: str, user_id: str, comment_text: str) -> bool:
        """Track a content comment."""
        try:
            db = get_database()
            
            await db.content_items.update_one(
                {"_id": content_id},
                {"$inc": {"engagement_metrics.comments": 1}}
            )
            
            await db.engagement_events.insert_one({
                "content_id": content_id,
                "user_id": user_id,
                "event_type": "comment",
                "comment_text": comment_text,
                "timestamp": datetime.utcnow()
            })
            
            logger.info(f"Tracked comment for content {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track comment: {e}")
            return False
    
    @staticmethod
    async def get_engagement_metrics(content_id: str) -> Optional[Dict]:
        """Get current engagement metrics for content."""
        try:
            db = get_database()
            
            content = await db.content_items.find_one(
                {"_id": content_id},
                {"engagement_metrics": 1}
            )
            
            if content and "engagement_metrics" in content:
                return content["engagement_metrics"]
            
            return {
                "views": 0,
                "likes": 0,
                "comments": 0,
                "shares": 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get engagement metrics: {e}")
            return None
    
    @staticmethod
    async def get_user_engagement(content_id: str, user_id: str) -> Dict[str, bool]:
        """Check if user has engaged with content."""
        try:
            db = get_database()
            
            liked = await db.engagement_events.find_one({
                "content_id": content_id,
                "user_id": user_id,
                "event_type": "like"
            })
            
            return {
                "liked": liked is not None,
                "bookmarked": False  # TODO: Implement bookmarking
            }
            
        except Exception as e:
            logger.error(f"Failed to get user engagement: {e}")
            return {"liked": False, "bookmarked": False}


# Singleton instance
engagement_tracker = EngagementTracker()

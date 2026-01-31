"""
Engagement tracking API endpoints.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from app.models.users import User
from app.api.dependencies import get_current_user, get_current_user_optional
from app.services.engagement_tracking import engagement_tracker

logger = logging.getLogger(__name__)
router = APIRouter()


class TrackViewRequest(BaseModel):
    """Request model for tracking views."""
    content_id: str


class TrackLikeRequest(BaseModel):
    """Request model for tracking likes."""
    content_id: str


class TrackShareRequest(BaseModel):
    """Request model for tracking shares."""
    content_id: str
    platform: Optional[str] = None


class TrackCommentRequest(BaseModel):
    """Request model for tracking comments."""
    content_id: str
    comment_text: str


@router.post("/track/view")
async def track_view(
    request: TrackViewRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Track a content view."""
    try:
        user_id = user.username if user else None
        success = await engagement_tracker.track_view(request.content_id, user_id)
        
        if success:
            metrics = await engagement_tracker.get_engagement_metrics(request.content_id)
            return {
                "success": True,
                "metrics": metrics
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to track view"
            )
            
    except Exception as e:
        logger.error(f"Error tracking view: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/track/like")
async def track_like(
    request: TrackLikeRequest,
    user: User = Depends(get_current_user)
):
    """Track a content like/unlike."""
    try:
        liked = await engagement_tracker.track_like(request.content_id, user.username)
        metrics = await engagement_tracker.get_engagement_metrics(request.content_id)
        
        return {
            "success": True,
            "liked": liked,
            "metrics": metrics
        }
            
    except Exception as e:
        logger.error(f"Error tracking like: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/track/share")
async def track_share(
    request: TrackShareRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Track a content share."""
    try:
        user_id = user.username if user else None
        success = await engagement_tracker.track_share(
            request.content_id, 
            user_id, 
            request.platform
        )
        
        if success:
            metrics = await engagement_tracker.get_engagement_metrics(request.content_id)
            return {
                "success": True,
                "metrics": metrics
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to track share"
            )
            
    except Exception as e:
        logger.error(f"Error tracking share: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/track/comment")
async def track_comment(
    request: TrackCommentRequest,
    user: User = Depends(get_current_user)
):
    """Track a content comment."""
    try:
        success = await engagement_tracker.track_comment(
            request.content_id,
            user.username,
            request.comment_text
        )
        
        if success:
            metrics = await engagement_tracker.get_engagement_metrics(request.content_id)
            return {
                "success": True,
                "metrics": metrics
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to track comment"
            )
            
    except Exception as e:
        logger.error(f"Error tracking comment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/metrics/{content_id}")
async def get_metrics(
    content_id: str,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Get engagement metrics for content."""
    try:
        metrics = await engagement_tracker.get_engagement_metrics(content_id)
        
        user_engagement = {}
        if user:
            user_engagement = await engagement_tracker.get_user_engagement(
                content_id,
                user.username
            )
        
        return {
            "success": True,
            "metrics": metrics,
            "user_engagement": user_engagement
        }
            
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

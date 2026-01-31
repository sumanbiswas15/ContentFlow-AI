"""
Authentication and security management endpoints.

This module provides API endpoints for authentication, API key management,
and security monitoring.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field

from app.models.users import User, APIKey
from app.api.dependencies import (
    get_current_user,
    get_current_api_key,
    require_admin,
    get_rate_limit_info
)
from app.services.auth_service import auth_service
from app.utils.security import generate_api_key, hash_api_key
from app.core.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class CreateAPIKeyRequest(BaseModel):
    """Request model for creating API key."""
    name: str = Field(..., description="Name for the API key")
    permissions: List[str] = Field(default_factory=list, description="List of permissions")
    expires_in_days: Optional[int] = Field(None, description="Days until expiration")


class APIKeyResponse(BaseModel):
    """Response model for API key."""
    key_id: str
    name: str
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    usage_count: int
    
    # Only returned on creation
    api_key: Optional[str] = Field(None, description="Full API key (only shown once)")


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int
    remaining: int
    reset: int
    window_seconds: int


class SecurityReport(BaseModel):
    """Security monitoring report."""
    timestamp: str
    suspicious_activity: dict
    rate_limiter_stats: dict


class UnblockIPRequest(BaseModel):
    """Request to unblock an IP address."""
    ip_address: str = Field(..., description="IP address to unblock")


# Endpoints

@router.get("/me", response_model=dict)
async def get_current_user_info(
    user: User = Depends(get_current_user),
    api_key: APIKey = Depends(get_current_api_key),
    rate_limit: dict = Depends(get_rate_limit_info)
):
    """
    Get current authenticated user information.
    
    Returns user details, API key info, and rate limit status.
    """
    return {
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "is_active": user.is_active,
            "is_verified": user.is_verified
        },
        "api_key": {
            "key_id": api_key.key_id,
            "name": api_key.name,
            "permissions": api_key.permissions,
            "usage_count": api_key.usage_count,
            "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None
        },
        "rate_limit": rate_limit,
        "usage_limits": user.check_usage_limits()
    }


@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateAPIKeyRequest,
    user: User = Depends(get_current_user)
):
    """
    Create a new API key for the current user.
    
    The full API key is only returned once during creation.
    Store it securely as it cannot be retrieved later.
    """
    # Generate new API key
    new_api_key = generate_api_key()
    
    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
    
    # Create API key object
    api_key_obj = APIKey(
        key_id=f"key_{datetime.utcnow().timestamp()}",
        key_hash=hash_api_key(new_api_key),
        name=request.name,
        permissions=request.permissions,
        expires_at=expires_at
    )
    
    # Add to user
    user.add_api_key(api_key_obj)
    
    # Save to database
    db = get_database()
    await db.users.update_one(
        {"_id": user.id},
        {"$set": {"api_keys": [key.dict() for key in user.api_keys], "updated_at": user.updated_at}}
    )
    
    # Log security event
    await auth_service.security_monitor.record_security_event(
        "api_key_created",
        user.username,
        {"key_id": api_key_obj.key_id, "permissions": request.permissions},
        severity="info"
    )
    
    # Return response with full API key
    return APIKeyResponse(
        key_id=api_key_obj.key_id,
        name=api_key_obj.name,
        permissions=api_key_obj.permissions,
        is_active=api_key_obj.is_active,
        created_at=api_key_obj.created_at,
        last_used_at=api_key_obj.last_used_at,
        expires_at=api_key_obj.expires_at,
        usage_count=api_key_obj.usage_count,
        api_key=new_api_key  # Only returned on creation
    )


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(user: User = Depends(get_current_user)):
    """
    List all API keys for the current user.
    
    Note: Full API keys are never returned, only metadata.
    """
    return [
        APIKeyResponse(
            key_id=key.key_id,
            name=key.name,
            permissions=key.permissions,
            is_active=key.is_active,
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            expires_at=key.expires_at,
            usage_count=key.usage_count
        )
        for key in user.api_keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_user)
):
    """
    Revoke (deactivate) an API key.
    
    The key will be marked as inactive and can no longer be used.
    """
    # Find the key
    key_found = False
    for key in user.api_keys:
        if key.key_id == key_id:
            key.is_active = False
            key_found = True
            break
    
    if not key_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )
    
    # Update in database
    db = get_database()
    await db.users.update_one(
        {"_id": user.id},
        {"$set": {"api_keys": [key.dict() for key in user.api_keys], "updated_at": datetime.utcnow()}}
    )
    
    # Log security event
    await auth_service.security_monitor.record_security_event(
        "api_key_revoked",
        user.username,
        {"key_id": key_id},
        severity="warning"
    )
    
    return None


@router.get("/rate-limit", response_model=RateLimitInfo)
async def get_rate_limit_status(
    api_key: APIKey = Depends(get_current_api_key),
    rate_limit: dict = Depends(get_rate_limit_info)
):
    """
    Get current rate limit status for the authenticated API key.
    """
    return RateLimitInfo(**rate_limit)


@router.get("/security/report", response_model=SecurityReport, dependencies=[Depends(require_admin)])
async def get_security_report():
    """
    Get comprehensive security report.
    
    Requires admin permission.
    Includes suspicious activity, blocked IPs, and rate limiter statistics.
    """
    report = await auth_service.get_security_report()
    return SecurityReport(**report)


@router.post("/security/unblock-ip", status_code=status.HTTP_200_OK, dependencies=[Depends(require_admin)])
async def unblock_ip_address(request: UnblockIPRequest):
    """
    Manually unblock an IP address.
    
    Requires admin permission.
    Use this to remove IP blocks for legitimate users who were blocked.
    """
    await auth_service.security_monitor.unblock_ip(request.ip_address)
    
    logger.info(f"IP address unblocked: {request.ip_address}")
    
    return {
        "message": f"IP address {request.ip_address} has been unblocked",
        "ip_address": request.ip_address
    }


@router.get("/security/suspicious-activity", dependencies=[Depends(require_admin)])
async def get_suspicious_activity():
    """
    Get detailed suspicious activity report.
    
    Requires admin permission.
    Returns information about blocked IPs, suspicious IPs, and failed authentication attempts.
    """
    report = await auth_service.security_monitor.get_suspicious_activity_report()
    return report


@router.post("/security/reset-rate-limit/{identifier}", dependencies=[Depends(require_admin)])
async def reset_rate_limit(identifier: str):
    """
    Reset rate limit for a specific identifier.
    
    Requires admin permission.
    Use this to manually reset rate limits for users or API keys.
    """
    await auth_service.rate_limiter.reset_limit(identifier)
    
    logger.info(f"Rate limit reset for: {identifier}")
    
    return {
        "message": f"Rate limit reset for {identifier}",
        "identifier": identifier
    }


@router.get("/usage", response_model=dict)
async def get_usage_stats(user: User = Depends(get_current_user)):
    """
    Get current usage statistics and limits for the authenticated user.
    """
    return {
        "usage_stats": {
            "tokens_used_today": user.usage_stats.tokens_used_today,
            "tokens_used_this_month": user.usage_stats.tokens_used_this_month,
            "cost_this_month": user.usage_stats.cost_this_month,
            "content_items_created": user.usage_stats.content_items_created,
            "storage_used_mb": user.usage_stats.storage_used_mb,
            "creative_sessions_today": user.usage_stats.creative_sessions_today,
            "content_generations_this_hour": user.usage_stats.content_generations_this_hour,
            "transformations_this_hour": user.usage_stats.transformations_this_hour
        },
        "usage_limits": {
            "daily_token_limit": user.usage_limits.daily_token_limit,
            "monthly_cost_limit": user.usage_limits.monthly_cost_limit,
            "max_concurrent_jobs": user.usage_limits.max_concurrent_jobs,
            "max_content_items": user.usage_limits.max_content_items,
            "max_storage_mb": user.usage_limits.max_storage_mb,
            "max_creative_sessions_per_day": user.usage_limits.max_creative_sessions_per_day,
            "max_content_generations_per_hour": user.usage_limits.max_content_generations_per_hour,
            "max_transformations_per_hour": user.usage_limits.max_transformations_per_hour
        },
        "limits_status": user.check_usage_limits()
    }

"""
FastAPI dependencies for authentication and authorization.

This module provides dependency injection functions for
authentication, authorization, and user context.

Requirements: 9.1, 9.2, 9.4
"""

from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.models.users import User, APIKey
from app.core.exceptions import AuthorizationError
from app.services.auth_service import auth_service

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(request: Request) -> User:
    """
    Get current authenticated user from request state.
    
    This dependency assumes the authentication middleware has already
    validated the request and attached user info to request.state.
    """
    if not hasattr(request.state, "user"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return request.state.user


async def get_current_user_optional(request: Request) -> Optional[User]:
    """
    Get current authenticated user from request state (optional).
    
    Returns None if no user is authenticated.
    """
    if hasattr(request.state, "user"):
        return request.state.user
    return None


async def get_current_api_key(request: Request) -> APIKey:
    """Get current API key from request state."""
    if not hasattr(request.state, "api_key"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return request.state.api_key


async def require_permission(permission: str):
    """
    Dependency factory for requiring specific permissions.
    
    Usage:
        @app.get("/admin/users", dependencies=[Depends(require_permission("admin"))])
    """
    async def check_permission(api_key: APIKey = Depends(get_current_api_key)):
        if not api_key.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return api_key
    
    return check_permission


async def get_rate_limit_info(request: Request) -> dict:
    """Get rate limit information from request state."""
    if hasattr(request.state, "rate_limit_info"):
        return request.state.rate_limit_info
    
    return {
        "limit": 0,
        "remaining": 0,
        "reset": 0
    }


class PermissionChecker:
    """Dependency class for checking permissions."""
    
    def __init__(self, required_permissions: list[str]):
        """
        Initialize permission checker.
        
        Args:
            required_permissions: List of required permissions (any match grants access)
        """
        self.required_permissions = required_permissions
    
    async def __call__(self, api_key: APIKey = Depends(get_current_api_key)) -> APIKey:
        """Check if API key has any of the required permissions."""
        has_permission = any(
            api_key.has_permission(perm)
            for perm in self.required_permissions
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these permissions required: {', '.join(self.required_permissions)}"
            )
        
        return api_key


class UsageLimitChecker:
    """Dependency class for checking usage limits."""
    
    def __init__(self, limit_type: str):
        """
        Initialize usage limit checker.
        
        Args:
            limit_type: Type of limit to check (e.g., 'daily_tokens', 'monthly_cost')
        """
        self.limit_type = limit_type
    
    async def __call__(self, user: User = Depends(get_current_user)) -> User:
        """Check if user is within usage limits."""
        limits = user.check_usage_limits()
        
        if self.limit_type in limits and not limits[self.limit_type]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Usage limit exceeded: {self.limit_type}"
            )
        
        return user


# Common permission dependencies
require_admin = PermissionChecker(["admin"])
require_content_create = PermissionChecker(["content:create", "admin"])
require_content_read = PermissionChecker(["content:read", "content:create", "admin"])
require_content_update = PermissionChecker(["content:update", "admin"])
require_content_delete = PermissionChecker(["content:delete", "admin"])
require_analytics_read = PermissionChecker(["analytics:read", "admin"])

# Common usage limit dependencies
check_daily_token_limit = UsageLimitChecker("daily_tokens")
check_monthly_cost_limit = UsageLimitChecker("monthly_cost")
check_content_generation_limit = UsageLimitChecker("content_generations")
check_transformation_limit = UsageLimitChecker("transformations")

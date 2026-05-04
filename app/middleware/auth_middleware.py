"""
Authentication middleware for ContentFlow AI.

This middleware handles API key validation, rate limiting,
and security monitoring for all API requests.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError
)
from app.services.auth_service import auth_service
from app.core.database import get_database

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for authentication and authorization."""
    
    # Paths that don't require authentication
    PUBLIC_PATHS = [
        "/",
        "/health",
        "/api/v1/health",
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/engines",  # Temporarily allow engines without auth for testing
        "/api/v1/content",  # TEMPORARY: Allow content endpoints without auth for debugging
        "/api/v1/engagement",  # Allow engagement tracking without auth
        "/storage",  # Allow access to static files (images, audio, video)
        "/docs",
        "/redoc",
        "/openapi.json"
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication middleware."""
        
        # Log all requests for debugging
        logger.info(f"Auth middleware processing: {request.method} {request.url.path}")
        logger.info(f"Authorization header: {request.headers.get('Authorization', 'NOT PROVIDED')[:50]}")
        
        # Skip authentication for public paths
        if any(request.url.path.startswith(path) for path in self.PUBLIC_PATHS):
            return await call_next(request)
        
        # Try JWT token first (Authorization: Bearer <token>)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            jwt_token = auth_header.split(" ")[1]
            logger.info(f"Attempting JWT authentication for path: {request.url.path}")
            logger.info(f"JWT token (first 50 chars): {jwt_token[:50]}")
            
            try:
                user = await self._get_user_by_jwt(jwt_token)
                if user:
                    logger.info(f"JWT authentication successful for user: {user.username}")
                    # Create a default API key object for JWT users
                    from app.models.users import APIKey
                    from datetime import datetime
                    
                    default_api_key = APIKey(
                        key_id="jwt_token",
                        key_hash="",
                        name="JWT Token",
                        permissions=["content:read", "content:write", "content:create", "content:update", "content:delete"]
                    )
                    
                    request.state.user = user
                    request.state.api_key = default_api_key
                    request.state.rate_limit_info = {
                        "limit": 100,
                        "remaining": 100,
                        "reset": 0,
                        "window_seconds": 60
                    }
                    
                    return await call_next(request)
                else:
                    logger.warning(f"JWT authentication failed for path: {request.url.path} - user not found or token invalid")
            except Exception as e:
                logger.error(f"Exception during JWT authentication: {e}", exc_info=True)
        
        # Fall back to API key authentication
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "AUTHENTICATION_ERROR",
                    "message": "Authentication required. Provide Authorization header with Bearer token or X-API-Key header."
                }
            )
        
        # Get client IP address
        ip_address = self._get_client_ip(request)
        
        try:
            # Check if IP is blocked
            if await auth_service.security_monitor.is_blocked(ip_address):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "BLOCKED",
                        "message": "Access denied from this IP address due to suspicious activity."
                    }
                )
            
            # Get user from database (simplified - in production, cache this)
            user = await self._get_user_by_api_key(api_key)
            if not user:
                await auth_service.security_monitor.record_failed_auth(
                    api_key[:10],
                    ip_address,
                    "user_not_found"
                )
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "AUTHENTICATION_ERROR",
                        "message": "Invalid API key"
                    }
                )
            
            # Authenticate request
            api_key_obj, rate_limit_info = await auth_service.authenticate_request(
                api_key,
                user,
                ip_address=ip_address
            )
            
            # Add authentication info to request state
            request.state.user = user
            request.state.api_key = api_key_obj
            request.state.rate_limit_info = rate_limit_info
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_limit_info["reset"])
            
            return response
            
        except AuthenticationError as e:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "AUTHENTICATION_ERROR",
                    "message": str(e)
                }
            )
        
        except AuthorizationError as e:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "AUTHORIZATION_ERROR",
                    "message": str(e)
                }
            )
        
        except RateLimitError as e:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": str(e)
                }
            )
        
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": "An error occurred during authentication"
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded IP (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def _get_user_by_api_key(self, api_key: str):
        """
        Get user by API key.
        
        This is a simplified implementation. In production, you would:
        1. Hash the API key
        2. Query the database for matching user
        3. Cache the result for performance
        """
        try:
            from app.utils.security import hash_api_key
            from app.models.users import User
            
            db = get_database()
            api_key_hash = hash_api_key(api_key)
            
            # Find user with matching API key hash
            user_doc = await db.users.find_one({
                "api_keys.key_hash": api_key_hash
            })
            
            if user_doc:
                return User(**user_doc)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching user by API key: {e}")
            return None
    
    async def _get_user_by_jwt(self, token: str):
        """Get user by JWT token."""
        try:
            from jose import jwt, JWTError
            from app.models.users import User
            from app.core.config import settings
            
            logger.info(f"Decoding JWT token with SECRET_KEY: {settings.SECRET_KEY[:20]}...")
            
            # Decode JWT token
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            email = payload.get("sub")
            
            logger.info(f"JWT decoded successfully, email: {email}")
            
            if not email:
                logger.warning("No email found in JWT payload")
                return None
            
            # Get user from database
            db = get_database()
            user_doc = await db.users.find_one({"email": email})
            
            if user_doc:
                logger.info(f"User found in database: {user_doc.get('username')}")
                return User(**user_doc)
            else:
                logger.warning(f"No user found with email: {email}")
            
            return None
            
        except JWTError as e:
            logger.error(f"JWT validation error: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error fetching user by JWT: {e}", exc_info=True)
            return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting based on IP address."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limit middleware."""
        
        # Skip for public paths
        if any(request.url.path.startswith(path) for path in AuthenticationMiddleware.PUBLIC_PATHS):
            return await call_next(request)
        
        # Get client IP
        ip_address = self._get_client_ip(request)
        
        try:
            # Check IP-based rate limit (more permissive than API key limit)
            await auth_service.check_rate_limit(
                f"ip:{ip_address}",
                limit=settings.RATE_LIMIT_PER_MINUTE * 2,  # Double limit for IP
                window_seconds=60
            )
            
            return await call_next(request)
            
        except RateLimitError as e:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": str(e)
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"


# Import settings after class definitions to avoid circular imports
from app.core.config import settings

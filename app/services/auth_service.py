"""
Authentication and authorization service for ContentFlow AI.

This module provides authentication, authorization, API key validation,
rate limiting, security monitoring, and suspicious activity detection.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import asyncio

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ValidationError
)
from app.utils.security import (
    verify_api_key,
    verify_token,
    hash_api_key,
    generate_request_id
)
from app.models.users import User, APIKey

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiting implementation with sliding window algorithm."""
    
    def __init__(self):
        """Initialize rate limiter with in-memory storage."""
        self._requests: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_interval = 300  # Clean up every 5 minutes
        self._last_cleanup = datetime.utcnow()
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (user_id, api_key, IP address)
            limit: Maximum number of requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=window_seconds)
            
            # Clean up old requests
            if identifier in self._requests:
                self._requests[identifier] = [
                    req_time for req_time in self._requests[identifier]
                    if req_time > cutoff
                ]
            
            # Count requests in window
            request_count = len(self._requests[identifier])
            
            # Check if limit exceeded
            is_allowed = request_count < limit
            
            if is_allowed:
                self._requests[identifier].append(now)
            
            # Calculate reset time
            if self._requests[identifier]:
                oldest_request = min(self._requests[identifier])
                reset_time = int((oldest_request + timedelta(seconds=window_seconds) - now).total_seconds())
            else:
                reset_time = window_seconds
            
            rate_limit_info = {
                "limit": limit,
                "remaining": max(0, limit - request_count - (1 if is_allowed else 0)),
                "reset": reset_time,
                "window_seconds": window_seconds
            }
            
            # Periodic cleanup
            if (now - self._last_cleanup).total_seconds() > self._cleanup_interval:
                await self._cleanup_old_entries()
            
            return is_allowed, rate_limit_info
    
    async def _cleanup_old_entries(self):
        """Clean up old rate limit entries to prevent memory bloat."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=1)  # Keep last hour of data
        
        identifiers_to_remove = []
        for identifier, requests in self._requests.items():
            # Remove old requests
            self._requests[identifier] = [
                req_time for req_time in requests
                if req_time > cutoff
            ]
            # Mark empty entries for removal
            if not self._requests[identifier]:
                identifiers_to_remove.append(identifier)
        
        # Remove empty entries
        for identifier in identifiers_to_remove:
            del self._requests[identifier]
        
        self._last_cleanup = now
        logger.debug(f"Rate limiter cleanup: removed {len(identifiers_to_remove)} empty entries")
    
    async def reset_limit(self, identifier: str):
        """Reset rate limit for a specific identifier."""
        async with self._lock:
            if identifier in self._requests:
                del self._requests[identifier]
    
    async def get_current_usage(self, identifier: str, window_seconds: int = 60) -> int:
        """Get current request count for an identifier."""
        async with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=window_seconds)
            
            if identifier not in self._requests:
                return 0
            
            # Count requests in window
            return len([
                req_time for req_time in self._requests[identifier]
                if req_time > cutoff
            ])


class SecurityMonitor:
    """Security monitoring and suspicious activity detection."""
    
    def __init__(self):
        """Initialize security monitor."""
        self._failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._suspicious_ips: Dict[str, Dict] = {}
        self._blocked_ips: set = set()
        self._lock = asyncio.Lock()
    
    async def record_failed_auth(
        self,
        identifier: str,
        ip_address: Optional[str] = None,
        reason: str = "invalid_credentials"
    ):
        """
        Record failed authentication attempt.
        
        Args:
            identifier: User identifier (username, email, api_key)
            ip_address: IP address of the request
            reason: Reason for failure
        """
        async with self._lock:
            now = datetime.utcnow()
            self._failed_attempts[identifier].append(now)
            
            # Log security event
            logger.warning(
                f"Failed authentication attempt",
                extra={
                    "identifier": identifier,
                    "ip_address": ip_address,
                    "reason": reason,
                    "timestamp": now.isoformat()
                }
            )
            
            # Check for suspicious activity
            await self._check_suspicious_activity(identifier, ip_address)
    
    async def _check_suspicious_activity(
        self,
        identifier: str,
        ip_address: Optional[str] = None
    ):
        """Check for suspicious activity patterns."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=15)
        
        # Clean up old attempts
        self._failed_attempts[identifier] = [
            attempt for attempt in self._failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        # Count recent failures
        recent_failures = len(self._failed_attempts[identifier])
        
        # Detect brute force attempts (5+ failures in 15 minutes)
        if recent_failures >= 5:
            logger.error(
                f"Suspicious activity detected: {recent_failures} failed attempts",
                extra={
                    "identifier": identifier,
                    "ip_address": ip_address,
                    "failure_count": recent_failures
                }
            )
            
            if ip_address:
                self._suspicious_ips[ip_address] = {
                    "identifier": identifier,
                    "failure_count": recent_failures,
                    "first_seen": self._failed_attempts[identifier][0],
                    "last_seen": now
                }
                
                # Block IP after 10 failures
                if recent_failures >= 10:
                    self._blocked_ips.add(ip_address)
                    logger.critical(
                        f"IP address blocked due to excessive failed attempts",
                        extra={
                            "ip_address": ip_address,
                            "identifier": identifier,
                            "failure_count": recent_failures
                        }
                    )
    
    async def is_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        async with self._lock:
            return ip_address in self._blocked_ips
    
    async def is_suspicious(self, identifier: str) -> Tuple[bool, int]:
        """
        Check if identifier has suspicious activity.
        
        Returns:
            Tuple of (is_suspicious, failure_count)
        """
        async with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=15)
            
            # Clean up old attempts
            if identifier in self._failed_attempts:
                self._failed_attempts[identifier] = [
                    attempt for attempt in self._failed_attempts[identifier]
                    if attempt > cutoff
                ]
                
                failure_count = len(self._failed_attempts[identifier])
                return failure_count >= 3, failure_count
            
            return False, 0
    
    async def record_security_event(
        self,
        event_type: str,
        identifier: str,
        details: Dict,
        severity: str = "info"
    ):
        """
        Record security event for audit logging.
        
        Args:
            event_type: Type of security event
            identifier: User/API key identifier
            details: Event details
            severity: Event severity (info, warning, error, critical)
        """
        log_data = {
            "event_type": event_type,
            "identifier": identifier,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        }
        
        if severity == "critical":
            logger.critical(f"Security event: {event_type}", extra=log_data)
        elif severity == "error":
            logger.error(f"Security event: {event_type}", extra=log_data)
        elif severity == "warning":
            logger.warning(f"Security event: {event_type}", extra=log_data)
        else:
            logger.info(f"Security event: {event_type}", extra=log_data)
    
    async def get_suspicious_activity_report(self) -> Dict:
        """Get report of suspicious activity."""
        async with self._lock:
            return {
                "blocked_ips": list(self._blocked_ips),
                "suspicious_ips": dict(self._suspicious_ips),
                "identifiers_with_failures": {
                    identifier: len(attempts)
                    for identifier, attempts in self._failed_attempts.items()
                    if attempts
                }
            }
    
    async def unblock_ip(self, ip_address: str):
        """Manually unblock an IP address."""
        async with self._lock:
            if ip_address in self._blocked_ips:
                self._blocked_ips.remove(ip_address)
                logger.info(f"IP address unblocked: {ip_address}")


class AuthService:
    """Main authentication and authorization service."""
    
    def __init__(self):
        """Initialize authentication service."""
        self.rate_limiter = RateLimiter()
        self.security_monitor = SecurityMonitor()
    
    async def validate_api_key(
        self,
        api_key: str,
        user: User,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[APIKey]]:
        """
        Validate API key and check permissions.
        
        Args:
            api_key: API key to validate
            user: User object containing API keys
            ip_address: IP address of the request
        
        Returns:
            Tuple of (is_valid, api_key_object)
        
        Raises:
            AuthenticationError: If API key is invalid
        """
        # Check if IP is blocked
        if ip_address and await self.security_monitor.is_blocked(ip_address):
            await self.security_monitor.record_security_event(
                "blocked_ip_attempt",
                api_key[:10],
                {"ip_address": ip_address},
                severity="warning"
            )
            raise AuthenticationError("Access denied from this IP address")
        
        # Find matching API key
        api_key_hash = hash_api_key(api_key)
        matching_key = None
        
        for key in user.api_keys:
            if verify_api_key(api_key, key.key_hash):
                matching_key = key
                break
        
        if not matching_key:
            await self.security_monitor.record_failed_auth(
                user.username,
                ip_address,
                "invalid_api_key"
            )
            return False, None
        
        # Check if key is active
        if not matching_key.is_active:
            await self.security_monitor.record_security_event(
                "inactive_api_key_used",
                user.username,
                {"key_id": matching_key.key_id},
                severity="warning"
            )
            return False, None
        
        # Check if key is expired
        if matching_key.is_expired():
            await self.security_monitor.record_security_event(
                "expired_api_key_used",
                user.username,
                {"key_id": matching_key.key_id},
                severity="warning"
            )
            return False, None
        
        # Update last used timestamp
        matching_key.last_used_at = datetime.utcnow()
        matching_key.usage_count += 1
        
        # Record successful authentication
        await self.security_monitor.record_security_event(
            "api_key_authenticated",
            user.username,
            {"key_id": matching_key.key_id, "ip_address": ip_address},
            severity="info"
        )
        
        return True, matching_key
    
    async def check_permissions(
        self,
        api_key: APIKey,
        required_permission: str
    ) -> bool:
        """
        Check if API key has required permission.
        
        Args:
            api_key: API key object
            required_permission: Permission to check
        
        Returns:
            True if permission granted
        
        Raises:
            AuthorizationError: If permission denied
        """
        if not api_key.has_permission(required_permission):
            await self.security_monitor.record_security_event(
                "permission_denied",
                api_key.key_id,
                {"required_permission": required_permission},
                severity="warning"
            )
            raise AuthorizationError(
                f"API key does not have required permission: {required_permission}"
            )
        
        return True
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: Optional[int] = None,
        window_seconds: int = 60
    ) -> Dict[str, int]:
        """
        Check and enforce rate limits.
        
        Args:
            identifier: Unique identifier for rate limiting
            limit: Maximum requests allowed (defaults to config)
            window_seconds: Time window in seconds
        
        Returns:
            Rate limit information
        
        Raises:
            RateLimitError: If rate limit exceeded
        """
        if limit is None:
            limit = settings.RATE_LIMIT_PER_MINUTE
        
        is_allowed, rate_limit_info = await self.rate_limiter.check_rate_limit(
            identifier,
            limit,
            window_seconds
        )
        
        if not is_allowed:
            await self.security_monitor.record_security_event(
                "rate_limit_exceeded",
                identifier,
                rate_limit_info,
                severity="warning"
            )
            raise RateLimitError(
                f"Rate limit exceeded. Try again in {rate_limit_info['reset']} seconds."
            )
        
        return rate_limit_info
    
    async def authenticate_request(
        self,
        api_key: str,
        user: User,
        required_permission: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[APIKey, Dict[str, int]]:
        """
        Authenticate and authorize a request.
        
        Args:
            api_key: API key from request
            user: User object
            required_permission: Required permission (optional)
            ip_address: IP address of request
        
        Returns:
            Tuple of (api_key_object, rate_limit_info)
        
        Raises:
            AuthenticationError: If authentication fails
            AuthorizationError: If authorization fails
            RateLimitError: If rate limit exceeded
        """
        # Validate API key
        is_valid, api_key_obj = await self.validate_api_key(api_key, user, ip_address)
        
        if not is_valid or not api_key_obj:
            raise AuthenticationError("Invalid API key")
        
        # Check permissions if required
        if required_permission:
            await self.check_permissions(api_key_obj, required_permission)
        
        # Check rate limits
        rate_limit_info = await self.check_rate_limit(
            f"api_key:{api_key_obj.key_id}",
            settings.RATE_LIMIT_PER_MINUTE
        )
        
        return api_key_obj, rate_limit_info
    
    async def get_security_report(self) -> Dict:
        """Get comprehensive security report."""
        suspicious_activity = await self.security_monitor.get_suspicious_activity_report()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "suspicious_activity": suspicious_activity,
            "rate_limiter_stats": {
                "tracked_identifiers": len(self.rate_limiter._requests)
            }
        }


# Global auth service instance
auth_service = AuthService()

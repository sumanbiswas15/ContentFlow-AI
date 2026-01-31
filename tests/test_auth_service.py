"""
Unit tests for authentication and authorization service.

Tests cover API key validation, rate limiting, security monitoring,
and suspicious activity detection.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services.auth_service import (
    RateLimiter,
    SecurityMonitor,
    AuthService,
    auth_service
)
from app.models.users import User, APIKey, UsageLimits, UsageStats
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError
)
from app.utils.security import generate_api_key, hash_api_key


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    api_key = generate_api_key()
    api_key_obj = APIKey(
        key_id="test_key_1",
        key_hash=hash_api_key(api_key),
        name="Test API Key",
        permissions=["content:read", "content:create"],
        is_active=True
    )
    
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed_password",
        api_keys=[api_key_obj]
    )
    
    return user, api_key


@pytest.fixture
def expired_api_key_user():
    """Create a user with an expired API key."""
    api_key = generate_api_key()
    api_key_obj = APIKey(
        key_id="expired_key",
        key_hash=hash_api_key(api_key),
        name="Expired API Key",
        permissions=["content:read"],
        is_active=True,
        expires_at=datetime.utcnow() - timedelta(days=1)
    )
    
    user = User(
        email="expired@example.com",
        username="expireduser",
        hashed_password="hashed_password",
        api_keys=[api_key_obj]
    )
    
    return user, api_key


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter()
        identifier = "test_user_1"
        
        # Make requests within limit
        for i in range(5):
            is_allowed, info = await limiter.check_rate_limit(identifier, limit=10)
            assert is_allowed
            assert info["remaining"] == 10 - i - 1
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter()
        identifier = "test_user_2"
        limit = 3
        
        # Make requests up to limit
        for i in range(limit):
            is_allowed, info = await limiter.check_rate_limit(identifier, limit=limit)
            assert is_allowed
        
        # Next request should be blocked
        is_allowed, info = await limiter.check_rate_limit(identifier, limit=limit)
        assert not is_allowed
        assert info["remaining"] == 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_sliding_window(self):
        """Test that rate limit uses sliding window."""
        limiter = RateLimiter()
        identifier = "test_user_3"
        limit = 2
        window = 2  # 2 seconds
        
        # Make requests up to limit
        for i in range(limit):
            is_allowed, _ = await limiter.check_rate_limit(
                identifier, limit=limit, window_seconds=window
            )
            assert is_allowed
        
        # Should be blocked
        is_allowed, _ = await limiter.check_rate_limit(
            identifier, limit=limit, window_seconds=window
        )
        assert not is_allowed
        
        # Wait for window to pass
        await asyncio.sleep(window + 0.1)
        
        # Should be allowed again
        is_allowed, _ = await limiter.check_rate_limit(
            identifier, limit=limit, window_seconds=window
        )
        assert is_allowed
    
    @pytest.mark.asyncio
    async def test_rate_limit_reset(self):
        """Test rate limit reset functionality."""
        limiter = RateLimiter()
        identifier = "test_user_4"
        
        # Make requests up to limit
        for i in range(3):
            await limiter.check_rate_limit(identifier, limit=3)
        
        # Should be blocked
        is_allowed, _ = await limiter.check_rate_limit(identifier, limit=3)
        assert not is_allowed
        
        # Reset limit
        await limiter.reset_limit(identifier)
        
        # Should be allowed again
        is_allowed, _ = await limiter.check_rate_limit(identifier, limit=3)
        assert is_allowed
    
    @pytest.mark.asyncio
    async def test_rate_limit_different_identifiers(self):
        """Test that different identifiers have separate limits."""
        limiter = RateLimiter()
        
        # User 1 makes requests
        for i in range(3):
            is_allowed, _ = await limiter.check_rate_limit("user1", limit=3)
            assert is_allowed
        
        # User 1 should be blocked
        is_allowed, _ = await limiter.check_rate_limit("user1", limit=3)
        assert not is_allowed
        
        # User 2 should still be allowed
        is_allowed, _ = await limiter.check_rate_limit("user2", limit=3)
        assert is_allowed
    
    @pytest.mark.asyncio
    async def test_get_current_usage(self):
        """Test getting current usage count."""
        limiter = RateLimiter()
        identifier = "test_user_5"
        
        # Make some requests
        for i in range(3):
            await limiter.check_rate_limit(identifier, limit=10)
        
        # Check usage
        usage = await limiter.get_current_usage(identifier)
        assert usage == 3


class TestSecurityMonitor:
    """Test security monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_record_failed_auth(self):
        """Test recording failed authentication attempts."""
        monitor = SecurityMonitor()
        
        await monitor.record_failed_auth("user1", "192.168.1.1", "invalid_password")
        
        is_suspicious, count = await monitor.is_suspicious("user1")
        assert count == 1
        assert not is_suspicious  # Not suspicious yet (threshold is 3)
    
    @pytest.mark.asyncio
    async def test_detect_suspicious_activity(self):
        """Test detection of suspicious activity."""
        monitor = SecurityMonitor()
        identifier = "suspicious_user"
        ip = "192.168.1.100"
        
        # Record multiple failed attempts
        for i in range(4):
            await monitor.record_failed_auth(identifier, ip, "brute_force")
        
        # Should be marked as suspicious
        is_suspicious, count = await monitor.is_suspicious(identifier)
        assert is_suspicious
        assert count >= 3
    
    @pytest.mark.asyncio
    async def test_ip_blocking_after_excessive_failures(self):
        """Test that IPs are blocked after excessive failures."""
        monitor = SecurityMonitor()
        identifier = "attacker"
        ip = "192.168.1.200"
        
        # Record many failed attempts
        for i in range(11):
            await monitor.record_failed_auth(identifier, ip, "brute_force")
        
        # IP should be blocked
        is_blocked = await monitor.is_blocked(ip)
        assert is_blocked
    
    @pytest.mark.asyncio
    async def test_unblock_ip(self):
        """Test manual IP unblocking."""
        monitor = SecurityMonitor()
        identifier = "user"
        ip = "192.168.1.250"
        
        # Block IP
        for i in range(11):
            await monitor.record_failed_auth(identifier, ip, "test")
        
        assert await monitor.is_blocked(ip)
        
        # Unblock IP
        await monitor.unblock_ip(ip)
        
        assert not await monitor.is_blocked(ip)
    
    @pytest.mark.asyncio
    async def test_security_event_recording(self):
        """Test security event recording."""
        monitor = SecurityMonitor()
        
        # Should not raise exception
        await monitor.record_security_event(
            "test_event",
            "test_user",
            {"detail": "test"},
            severity="info"
        )
    
    @pytest.mark.asyncio
    async def test_suspicious_activity_report(self):
        """Test getting suspicious activity report."""
        monitor = SecurityMonitor()
        
        # Create some activity
        await monitor.record_failed_auth("user1", "192.168.1.1")
        await monitor.record_failed_auth("user2", "192.168.1.2")
        
        report = await monitor.get_suspicious_activity_report()
        
        assert "blocked_ips" in report
        assert "suspicious_ips" in report
        assert "identifiers_with_failures" in report
        assert len(report["identifiers_with_failures"]) >= 2


class TestAuthService:
    """Test main authentication service."""
    
    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, sample_user):
        """Test successful API key validation."""
        user, api_key = sample_user
        service = AuthService()
        
        is_valid, api_key_obj = await service.validate_api_key(
            api_key, user, "192.168.1.1"
        )
        
        assert is_valid
        assert api_key_obj is not None
        assert api_key_obj.key_id == "test_key_1"
        assert api_key_obj.usage_count == 1
    
    @pytest.mark.asyncio
    async def test_validate_api_key_invalid(self, sample_user):
        """Test validation with invalid API key."""
        user, _ = sample_user
        service = AuthService()
        
        is_valid, api_key_obj = await service.validate_api_key(
            "invalid_key", user, "192.168.1.1"
        )
        
        assert not is_valid
        assert api_key_obj is None
    
    @pytest.mark.asyncio
    async def test_validate_expired_api_key(self, expired_api_key_user):
        """Test validation with expired API key."""
        user, api_key = expired_api_key_user
        service = AuthService()
        
        is_valid, api_key_obj = await service.validate_api_key(
            api_key, user, "192.168.1.1"
        )
        
        assert not is_valid
        assert api_key_obj is None
    
    @pytest.mark.asyncio
    async def test_validate_inactive_api_key(self, sample_user):
        """Test validation with inactive API key."""
        user, api_key = sample_user
        user.api_keys[0].is_active = False
        service = AuthService()
        
        is_valid, api_key_obj = await service.validate_api_key(
            api_key, user, "192.168.1.1"
        )
        
        assert not is_valid
        assert api_key_obj is None
    
    @pytest.mark.asyncio
    async def test_check_permissions_success(self, sample_user):
        """Test successful permission check."""
        user, api_key = sample_user
        service = AuthService()
        
        _, api_key_obj = await service.validate_api_key(api_key, user)
        
        # Should not raise exception
        result = await service.check_permissions(api_key_obj, "content:read")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_permissions_failure(self, sample_user):
        """Test permission check failure."""
        user, api_key = sample_user
        service = AuthService()
        
        _, api_key_obj = await service.validate_api_key(api_key, user)
        
        # Should raise AuthorizationError
        with pytest.raises(AuthorizationError):
            await service.check_permissions(api_key_obj, "admin")
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_success(self):
        """Test rate limit check within limits."""
        service = AuthService()
        
        # Should not raise exception
        info = await service.check_rate_limit("test_user", limit=10)
        assert info["remaining"] > 0
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self):
        """Test rate limit exceeded."""
        service = AuthService()
        identifier = "rate_limited_user"
        
        # Exhaust rate limit
        for i in range(5):
            await service.check_rate_limit(identifier, limit=5)
        
        # Should raise RateLimitError
        with pytest.raises(RateLimitError):
            await service.check_rate_limit(identifier, limit=5)
    
    @pytest.mark.asyncio
    async def test_authenticate_request_success(self, sample_user):
        """Test successful request authentication."""
        user, api_key = sample_user
        service = AuthService()
        
        api_key_obj, rate_limit_info = await service.authenticate_request(
            api_key, user, "content:read", "192.168.1.1"
        )
        
        assert api_key_obj is not None
        assert api_key_obj.key_id == "test_key_1"
        assert "limit" in rate_limit_info
        assert "remaining" in rate_limit_info
    
    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_key(self, sample_user):
        """Test authentication with invalid key."""
        user, _ = sample_user
        service = AuthService()
        
        with pytest.raises(AuthenticationError):
            await service.authenticate_request(
                "invalid_key", user, None, "192.168.1.1"
            )
    
    @pytest.mark.asyncio
    async def test_authenticate_request_insufficient_permissions(self, sample_user):
        """Test authentication with insufficient permissions."""
        user, api_key = sample_user
        service = AuthService()
        
        with pytest.raises(AuthorizationError):
            await service.authenticate_request(
                api_key, user, "admin", "192.168.1.1"
            )
    
    @pytest.mark.asyncio
    async def test_blocked_ip_authentication(self, sample_user):
        """Test that blocked IPs cannot authenticate."""
        user, api_key = sample_user
        service = AuthService()
        ip = "192.168.1.99"
        
        # Block the IP
        for i in range(11):
            await service.security_monitor.record_failed_auth("attacker", ip)
        
        # Authentication should fail
        with pytest.raises(AuthenticationError):
            await service.validate_api_key(api_key, user, ip)
    
    @pytest.mark.asyncio
    async def test_get_security_report(self):
        """Test getting security report."""
        service = AuthService()
        
        # Generate some activity
        await service.security_monitor.record_failed_auth("user1", "192.168.1.1")
        await service.check_rate_limit("user2", limit=10)
        
        report = await service.get_security_report()
        
        assert "timestamp" in report
        assert "suspicious_activity" in report
        assert "rate_limiter_stats" in report


class TestAPIKeyModel:
    """Test APIKey model methods."""
    
    def test_api_key_is_expired_false(self):
        """Test that non-expired key returns False."""
        api_key = APIKey(
            key_id="test",
            key_hash="hash",
            name="Test",
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        assert not api_key.is_expired()
    
    def test_api_key_is_expired_true(self):
        """Test that expired key returns True."""
        api_key = APIKey(
            key_id="test",
            key_hash="hash",
            name="Test",
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
        
        assert api_key.is_expired()
    
    def test_api_key_is_expired_none(self):
        """Test that key with no expiration returns False."""
        api_key = APIKey(
            key_id="test",
            key_hash="hash",
            name="Test",
            expires_at=None
        )
        
        assert not api_key.is_expired()
    
    def test_api_key_has_permission_true(self):
        """Test permission check returns True."""
        api_key = APIKey(
            key_id="test",
            key_hash="hash",
            name="Test",
            permissions=["content:read", "content:create"]
        )
        
        assert api_key.has_permission("content:read")
        assert api_key.has_permission("content:create")
    
    def test_api_key_has_permission_false(self):
        """Test permission check returns False."""
        api_key = APIKey(
            key_id="test",
            key_hash="hash",
            name="Test",
            permissions=["content:read"]
        )
        
        assert not api_key.has_permission("admin")
    
    def test_api_key_admin_permission(self):
        """Test that admin permission grants all access."""
        api_key = APIKey(
            key_id="test",
            key_hash="hash",
            name="Test",
            permissions=["admin"]
        )
        
        assert api_key.has_permission("content:read")
        assert api_key.has_permission("content:create")
        assert api_key.has_permission("any_permission")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_checks(self):
        """Test concurrent rate limit checks."""
        limiter = RateLimiter()
        identifier = "concurrent_user"
        
        # Make concurrent requests
        tasks = [
            limiter.check_rate_limit(identifier, limit=10)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should be allowed
        assert all(is_allowed for is_allowed, _ in results)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup(self):
        """Test that rate limiter cleans up old entries."""
        limiter = RateLimiter()
        
        # Add entries for many identifiers
        for i in range(100):
            await limiter.check_rate_limit(f"user_{i}", limit=10)
        
        # Force cleanup
        await limiter._cleanup_old_entries()
        
        # Should not raise exception
        assert True
    
    @pytest.mark.asyncio
    async def test_empty_permissions_list(self):
        """Test API key with empty permissions."""
        api_key = APIKey(
            key_id="test",
            key_hash="hash",
            name="Test",
            permissions=[]
        )
        
        assert not api_key.has_permission("any_permission")
    
    @pytest.mark.asyncio
    async def test_security_monitor_with_none_ip(self):
        """Test security monitor handles None IP address."""
        monitor = SecurityMonitor()
        
        # Should not raise exception
        await monitor.record_failed_auth("user", None, "test")
        
        is_suspicious, count = await monitor.is_suspicious("user")
        assert count == 1

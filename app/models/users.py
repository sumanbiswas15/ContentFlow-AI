"""
User and authentication models for ContentFlow AI.

This module defines user-related models including user profiles,
authentication, and usage tracking.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr, validator

from app.models.base import BaseDocument, TimestampMixin


class UsageLimits(BaseModel):
    """Model for user usage limits and quotas."""
    daily_token_limit: int = 10000
    monthly_cost_limit: float = 50.0
    max_concurrent_jobs: int = 3
    max_content_items: int = 1000
    max_storage_mb: int = 1000
    
    # Feature-specific limits
    max_creative_sessions_per_day: int = 10
    max_content_generations_per_hour: int = 20
    max_transformations_per_hour: int = 50


class UsageStats(BaseModel):
    """Model for tracking user usage statistics."""
    tokens_used_today: int = 0
    tokens_used_this_month: int = 0
    cost_this_month: float = 0.0
    content_items_created: int = 0
    storage_used_mb: float = 0.0
    
    # Daily counters (reset daily)
    creative_sessions_today: int = 0
    
    # Hourly counters (reset hourly)
    content_generations_this_hour: int = 0
    transformations_this_hour: int = 0
    
    last_reset_date: datetime = Field(default_factory=datetime.utcnow)
    last_hourly_reset: datetime = Field(default_factory=datetime.utcnow)
    
    def reset_daily_counters(self):
        """Reset daily usage counters."""
        self.tokens_used_today = 0
        self.creative_sessions_today = 0
        self.last_reset_date = datetime.utcnow()
    
    def reset_hourly_counters(self):
        """Reset hourly usage counters."""
        self.content_generations_this_hour = 0
        self.transformations_this_hour = 0
        self.last_hourly_reset = datetime.utcnow()
    
    def should_reset_daily(self) -> bool:
        """Check if daily counters should be reset."""
        now = datetime.utcnow()
        return now.date() > self.last_reset_date.date()
    
    def should_reset_hourly(self) -> bool:
        """Check if hourly counters should be reset."""
        now = datetime.utcnow()
        return (now - self.last_hourly_reset).total_seconds() >= 3600


class APIKey(BaseModel):
    """Model for API key management."""
    key_id: str
    key_hash: str  # Hashed version of the actual key
    name: str
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission."""
        return permission in self.permissions or "admin" in self.permissions


class UserPreferences(BaseModel):
    """Model for user preferences and settings."""
    default_language: str = "en"
    preferred_ai_model: str = "gemini-pro"
    notification_settings: Dict[str, bool] = Field(default_factory=lambda: {
        "job_completion": True,
        "cost_warnings": True,
        "security_alerts": True,
        "feature_updates": False
    })
    ui_theme: str = "light"  # light, dark, auto
    timezone: str = "UTC"
    
    # Content preferences
    default_content_type: str = "text"
    default_tone: str = "professional"
    preferred_platforms: List[str] = Field(default_factory=list)


class User(BaseDocument):
    """Main user model."""
    
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    
    # Authentication
    hashed_password: str
    api_keys: List[APIKey] = Field(default_factory=list)
    
    # Usage tracking
    usage_limits: UsageLimits = Field(default_factory=UsageLimits)
    usage_stats: UsageStats = Field(default_factory=UsageStats)
    
    # User preferences
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    
    # Profile information
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    website: Optional[str] = None
    
    # Account status
    last_login_at: Optional[datetime] = None
    email_verified_at: Optional[datetime] = None
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not v or len(v) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if len(v) > 50:
            raise ValueError("Username cannot exceed 50 characters")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
        return v.lower()
    
    @validator('bio')
    def validate_bio(cls, v):
        """Validate bio length."""
        if v and len(v) > 500:
            raise ValueError("Bio cannot exceed 500 characters")
        return v
    
    def add_api_key(self, api_key: APIKey):
        """Add API key to user."""
        self.api_keys.append(api_key)
        self.updated_at = datetime.utcnow()
    
    def get_active_api_keys(self) -> List[APIKey]:
        """Get all active, non-expired API keys."""
        return [
            key for key in self.api_keys
            if key.is_active and not key.is_expired()
        ]
    
    def update_last_login(self):
        """Update last login timestamp."""
        self.last_login_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def check_usage_limits(self) -> Dict[str, bool]:
        """Check if user is within usage limits."""
        # Reset counters if needed
        if self.usage_stats.should_reset_daily():
            self.usage_stats.reset_daily_counters()
        
        if self.usage_stats.should_reset_hourly():
            self.usage_stats.reset_hourly_counters()
        
        return {
            "daily_tokens": self.usage_stats.tokens_used_today < self.usage_limits.daily_token_limit,
            "monthly_cost": self.usage_stats.cost_this_month < self.usage_limits.monthly_cost_limit,
            "storage": self.usage_stats.storage_used_mb < self.usage_limits.max_storage_mb,
            "content_items": self.usage_stats.content_items_created < self.usage_limits.max_content_items,
            "creative_sessions": self.usage_stats.creative_sessions_today < self.usage_limits.max_creative_sessions_per_day,
            "content_generations": self.usage_stats.content_generations_this_hour < self.usage_limits.max_content_generations_per_hour,
            "transformations": self.usage_stats.transformations_this_hour < self.usage_limits.max_transformations_per_hour,
        }
    
    def increment_usage(self, usage_type: str, amount: int = 1, cost: float = 0.0):
        """Increment usage statistics."""
        if usage_type == "tokens":
            self.usage_stats.tokens_used_today += amount
            self.usage_stats.tokens_used_this_month += amount
            self.usage_stats.cost_this_month += cost
        elif usage_type == "creative_sessions":
            self.usage_stats.creative_sessions_today += amount
        elif usage_type == "content_generations":
            self.usage_stats.content_generations_this_hour += amount
        elif usage_type == "transformations":
            self.usage_stats.transformations_this_hour += amount
        elif usage_type == "content_items":
            self.usage_stats.content_items_created += amount
        elif usage_type == "storage":
            self.usage_stats.storage_used_mb += amount
        
        self.updated_at = datetime.utcnow()


class UserSession(BaseDocument):
    """Model for user session tracking."""
    
    user_id: str
    session_token: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    expires_at: datetime
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def invalidate(self):
        """Invalidate the session."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
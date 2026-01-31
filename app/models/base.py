"""
Base models and common types for ContentFlow AI.

This module defines the foundational data structures and enums
used throughout the application.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, handler=None):
        """Validate ObjectId for Pydantic V2 compatibility."""
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            if not ObjectId.is_valid(v):
                raise ValueError("Invalid ObjectId")
            return ObjectId(v)
        raise ValueError(f"Invalid ObjectId type: {type(v)}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Pydantic V2 core schema."""
        from pydantic_core import core_schema
        
        def validate_from_any(value):
            if isinstance(value, ObjectId):
                return value
            if isinstance(value, str):
                if not ObjectId.is_valid(value):
                    raise ValueError("Invalid ObjectId")
                return ObjectId(value)
            raise ValueError(f"Invalid ObjectId type: {type(value)}")
        
        return core_schema.no_info_plain_validator_function(
            validate_from_any,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x),
                return_schema=core_schema.str_schema(),
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")
        return field_schema


class ContentType(str, Enum):
    """Enumeration of supported content types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class WorkflowState(str, Enum):
    """Enumeration of workflow states in the content lifecycle."""
    DISCOVER = "discover"
    CREATE = "create"
    TRANSFORM = "transform"
    PLAN = "plan"
    PUBLISH = "publish"
    ANALYZE = "analyze"
    IMPROVE = "improve"


class JobStatus(str, Enum):
    """Enumeration of async job statuses."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Enumeration of job types."""
    CONTENT_GENERATION = "content_generation"
    CONTENT_TRANSFORMATION = "content_transformation"
    CREATIVE_ASSISTANCE = "creative_assistance"
    SOCIAL_MEDIA_OPTIMIZATION = "social_media_optimization"
    ANALYTICS_PROCESSING = "analytics_processing"
    MEDIA_GENERATION = "media_generation"


class Platform(str, Enum):
    """Enumeration of social media platforms."""
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    GENERIC = "generic"


class ContentFormat(str, Enum):
    """Enumeration of content formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    JSON = "json"
    XML = "xml"


class EngineType(str, Enum):
    """Enumeration of AI engine types."""
    TEXT_INTELLIGENCE = "text_intelligence"
    IMAGE_GENERATION = "image_generation"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_PIPELINE = "video_pipeline"
    CREATIVE_ASSISTANT = "creative_assistant"
    SOCIAL_MEDIA_PLANNER = "social_media_planner"
    DISCOVERY_ANALYTICS = "discovery_analytics"


class TransformationType(str, Enum):
    """Enumeration of content transformation types."""
    SUMMARIZE = "summarize"
    TONE_CHANGE = "tone_change"
    TRANSLATE = "translate"
    PLATFORM_ADAPT = "platform_adapt"
    EXPAND = "expand"
    SIMPLIFY = "simplify"


class CreativeSessionType(str, Enum):
    """Enumeration of creative assistance session types."""
    IDEATION = "ideation"
    REWRITE = "rewrite"
    DESIGN_ASSISTANCE = "design_assistance"
    MARKETING_SUPPORT = "marketing_support"
    HOOK_GENERATION = "hook_generation"


class ErrorCode(str, Enum):
    """Enumeration of system error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    USAGE_LIMIT_EXCEEDED = "USAGE_LIMIT_EXCEEDED"
    ENGINE_ERROR = "ENGINE_ERROR"
    TIMEOUT = "TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PARSING_ERROR = "PARSING_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"


class BaseDocument(BaseModel):
    """Base class for all database documents."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        
    def dict(self, **kwargs):
        """Override dict method to handle ObjectId serialization."""
        d = super().dict(**kwargs)
        if "_id" in d:
            d["_id"] = str(d["_id"])
        return d


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp tracking."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserMixin(BaseModel):
    """Mixin for models that need user tracking."""
    user_id: str
    created_by: Optional[str] = None
    updated_by: Optional[str] = None


class MetadataMixin(BaseModel):
    """Mixin for models that need metadata tracking."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class PaginationParams(BaseModel):
    """Parameters for paginated queries."""
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(10, ge=1, le=100, description="Number of items to return")


class PaginatedResponse(BaseModel):
    """Generic paginated response model."""
    items: List[Any]
    total: int
    skip: int
    limit: int
    has_more: bool
    
    @classmethod
    def create(cls, items: List[Any], total: int, skip: int, limit: int):
        """Create a paginated response."""
        return cls(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total
        )


class ValidationResult(BaseModel):
    """Model for validation results."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @classmethod
    def success(cls):
        """Create a successful validation result."""
        return cls(is_valid=True)
    
    @classmethod
    def failure(cls, errors: List[str], warnings: List[str] = None):
        """Create a failed validation result."""
        return cls(
            is_valid=False,
            errors=errors,
            warnings=warnings or []
        )
    
    def add_error(self, error: str):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)


class SystemHealth(BaseModel):
    """Model for system health status."""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)  # service_name -> status
    metrics: Dict[str, Union[int, float]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == "healthy"
    
    def add_service_status(self, service: str, status: str):
        """Add service status."""
        self.services[service] = status
        if status != "healthy" and self.status == "healthy":
            self.status = "degraded"
    
    def add_metric(self, name: str, value: Union[int, float]):
        """Add system metric."""
        self.metrics[name] = value
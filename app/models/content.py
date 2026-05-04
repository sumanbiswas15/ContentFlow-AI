"""
Content-related data models for ContentFlow AI.

This module defines the core content models including ContentItem,
ContentMetadata, and related structures for content management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator

from app.models.base import (
    BaseDocument, ContentType, WorkflowState, Platform,
    TimestampMixin, UserMixin, MetadataMixin
)


class EngagementMetrics(BaseModel):
    """Model for content engagement metrics."""
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    click_through_rate: float = 0.0
    engagement_rate: float = 0.0
    reach: int = 0
    impressions: int = 0
    
    def calculate_engagement_rate(self) -> float:
        """Calculate engagement rate based on interactions and reach."""
        if self.reach == 0:
            return 0.0
        total_interactions = self.likes + self.shares + self.comments
        return (total_interactions / self.reach) * 100


class OptimizationData(BaseModel):
    """Model for platform-specific optimization data."""
    platform: Platform
    optimized_content: str
    hashtags: List[str] = Field(default_factory=list)
    call_to_action: Optional[str] = None
    optimal_posting_times: List[datetime] = Field(default_factory=list)
    character_count: int = 0
    estimated_reach: int = 0
    engagement_prediction: float = 0.0


class ProcessingStep(BaseModel):
    """Model for tracking content processing history."""
    step_id: str
    engine: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[int] = None
    status: str = "completed"  # completed, failed, skipped
    error_message: Optional[str] = None


class CostData(BaseModel):
    """Model for tracking content-related costs."""
    total_tokens_used: int = 0
    cost_per_operation: Dict[str, float] = Field(default_factory=dict)
    total_cost: float = 0.0
    currency: str = "USD"
    
    def add_operation_cost(self, operation: str, tokens: int, cost_per_token: float):
        """Add cost for a specific operation."""
        operation_cost = tokens * cost_per_token
        self.cost_per_operation[operation] = operation_cost
        self.total_tokens_used += tokens
        self.total_cost += operation_cost


class ContentMetadata(BaseModel):
    """Model for content metadata and tracking information."""
    author: str
    title: Optional[str] = None
    description: Optional[str] = None
    language: str = "en"
    platform_optimizations: Dict[Platform, OptimizationData] = Field(default_factory=dict)
    engagement_metrics: EngagementMetrics = Field(default_factory=EngagementMetrics)
    cost_tracking: CostData = Field(default_factory=CostData)
    processing_history: List[ProcessingStep] = Field(default_factory=list)
    source_url: Optional[str] = None
    content_length: int = 0
    word_count: int = 0
    reading_time_minutes: int = 0
    
    def calculate_reading_time(self, words_per_minute: int = 200) -> int:
        """Calculate estimated reading time in minutes."""
        if self.word_count == 0:
            return 0
        return max(1, round(self.word_count / words_per_minute))


class ContentItem(BaseDocument, UserMixin, MetadataMixin):
    """Main content item model representing any piece of content in the system."""
    
    type: ContentType
    title: str
    content: Union[str, bytes, Dict[str, Any]]  # Text content, binary data, or structured data
    content_metadata: ContentMetadata
    workflow_state: WorkflowState = WorkflowState.CREATE
    version: int = 1
    parent_id: Optional[str] = None  # For content versions/derivatives
    is_published: bool = False
    published_at: Optional[datetime] = None
    
    @validator('content')
    def validate_content(cls, v, values):
        """Validate content based on content type."""
        content_type = values.get('type')
        
        if content_type == ContentType.TEXT:
            if not isinstance(v, str):
                raise ValueError("Text content must be a string")
        elif content_type in [ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO]:
            # For media content, we store references or metadata, not raw bytes
            if not isinstance(v, (str, dict)):
                raise ValueError("Media content must be a string reference or metadata dict")
        
        return v
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title length and content."""
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        if len(v) > 200:
            raise ValueError("Title cannot exceed 200 characters")
        return v.strip()
    
    def update_metadata(self):
        """Update content metadata based on current content."""
        if self.type == ContentType.TEXT and isinstance(self.content, str):
            self.content_metadata.content_length = len(self.content)
            self.content_metadata.word_count = len(self.content.split())
            self.content_metadata.reading_time_minutes = self.content_metadata.calculate_reading_time()
        
        self.updated_at = datetime.utcnow()
    
    def add_processing_step(self, engine: str, operation: str, **kwargs):
        """Add a processing step to the history."""
        step = ProcessingStep(
            step_id=f"{engine}_{operation}_{len(self.content_metadata.processing_history)}",
            engine=engine,
            operation=operation,
            parameters=kwargs
        )
        self.content_metadata.processing_history.append(step)
    
    def get_latest_version_for_platform(self, platform: Platform) -> Optional[OptimizationData]:
        """Get the latest optimization data for a specific platform."""
        return self.content_metadata.platform_optimizations.get(platform)
    
    def is_optimized_for_platform(self, platform: Platform) -> bool:
        """Check if content is optimized for a specific platform."""
        return platform in self.content_metadata.platform_optimizations


class ContentVersion(BaseModel):
    """Model for tracking content versions."""
    version_number: int
    content_id: str
    changes_summary: str
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    diff_data: Optional[Dict[str, Any]] = None


class ContentCollection(BaseDocument, UserMixin):
    """Model for grouping related content items."""
    name: str
    description: Optional[str] = None
    content_ids: List[str] = Field(default_factory=list)
    collection_type: str = "general"  # campaign, series, project, etc.
    is_public: bool = False
    
    def add_content(self, content_id: str):
        """Add content to the collection."""
        if content_id not in self.content_ids:
            self.content_ids.append(content_id)
            self.updated_at = datetime.utcnow()
    
    def remove_content(self, content_id: str):
        """Remove content from the collection."""
        if content_id in self.content_ids:
            self.content_ids.remove(content_id)
            self.updated_at = datetime.utcnow()
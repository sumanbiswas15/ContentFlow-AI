"""
Unit tests for core data models and validation functions.

This module tests the core data models including ContentItem, AsyncJob,
CostData, and their validation functions.
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from pydantic import ValidationError

from app.models.content import ContentItem, ContentMetadata, CostData, EngagementMetrics
from app.models.jobs import AsyncJob, JobResult, RetryConfig
from app.models.base import (
    ContentType, WorkflowState, JobType, JobStatus, Platform,
    ValidationResult
)
from app.utils.data_integrity import DataIntegrityValidator, validate_model_integrity
from app.utils.validators import (
    validate_content_type, validate_workflow_state, validate_platform,
    validate_workflow_transition, validate_content_metadata,
    validate_job_parameters, validate_cost_data
)


class TestContentItem:
    """Test cases for ContentItem model."""
    
    def test_valid_content_item_creation(self):
        """Test creating a valid ContentItem."""
        metadata = ContentMetadata(author="test_user")
        content_item = ContentItem(
            type=ContentType.TEXT,
            title="Test Content",
            content="This is test content.",
            content_metadata=metadata,
            user_id="user123",
            tags=["test", "validation"]
        )
        
        assert content_item.type == ContentType.TEXT
        assert content_item.title == "Test Content"
        assert content_item.workflow_state == WorkflowState.CREATE
        assert content_item.version == 1
        assert len(content_item.tags) == 2
    
    def test_content_item_validation(self):
        """Test ContentItem validation."""
        metadata = ContentMetadata(author="test_user")
        content_item = ContentItem(
            type=ContentType.TEXT,
            title="Valid Title",
            content="Valid content with sufficient length.",
            content_metadata=metadata,
            user_id="user123"
        )
        
        result = DataIntegrityValidator.validate_content_item(content_item)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_invalid_content_item_validation(self):
        """Test validation of invalid ContentItem."""
        metadata = ContentMetadata(author="test_user")
        
        # Test that invalid data raises validation error during creation
        with pytest.raises(ValidationError):
            ContentItem(
                type=ContentType.TEXT,
                title="x" * 250,  # Too long title
                content="",  # Empty content
                content_metadata=metadata,
                user_id="user123",
                version=0  # Invalid version
            )
    
    def test_content_metadata_update(self):
        """Test content metadata update functionality."""
        metadata = ContentMetadata(author="test_user")
        content_item = ContentItem(
            type=ContentType.TEXT,
            title="Test Content",
            content="This is test content with multiple words for counting.",
            content_metadata=metadata,
            user_id="user123"
        )
        
        content_item.update_metadata()
        
        assert content_item.content_metadata.content_length > 0
        assert content_item.content_metadata.word_count > 0
        assert content_item.content_metadata.reading_time_minutes >= 1


class TestAsyncJob:
    """Test cases for AsyncJob model."""
    
    def test_valid_job_creation(self):
        """Test creating a valid AsyncJob."""
        job = AsyncJob(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Write about AI", "content_type": "text"},
            user_id="user123"
        )
        
        assert job.job_type == JobType.CONTENT_GENERATION
        assert job.status == JobStatus.QUEUED
        assert job.priority == 5  # Default priority
        assert job.retry_count == 0
    
    def test_job_execution_lifecycle(self):
        """Test job execution lifecycle methods."""
        job = AsyncJob(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Write about AI"},
            user_id="user123"
        )
        
        # Start execution
        job.start_execution()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        
        # Complete with success
        job.complete_with_success("Generated content", tokens_used=100, cost=0.05)
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.result.success
        assert job.result.tokens_used == 100
        assert job.result.cost == 0.05
    
    def test_job_failure_handling(self):
        """Test job failure handling."""
        job = AsyncJob(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Write about AI"},
            user_id="user123"
        )
        
        job.start_execution()
        job.complete_with_error("Service unavailable", "SERVICE_UNAVAILABLE")
        
        assert job.status == JobStatus.FAILED
        assert job.result.success is False
        assert job.result.error_code == "SERVICE_UNAVAILABLE"
        assert job.last_error == "Service unavailable"
    
    def test_job_retry_logic(self):
        """Test job retry logic."""
        retry_config = RetryConfig(
            max_retries=2,
            retry_on_errors=["TIMEOUT", "SERVICE_UNAVAILABLE"]
        )
        
        job = AsyncJob(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Write about AI"},
            user_id="user123",
            retry_config=retry_config
        )
        
        job.complete_with_error("Timeout", "TIMEOUT")
        
        # Should be eligible for retry
        assert job.should_retry()
        
        job.increment_retry()
        assert job.retry_count == 1
        assert job.status == JobStatus.QUEUED
        
        # After max retries, should not retry
        job.complete_with_error("Timeout", "TIMEOUT")
        job.increment_retry()
        job.complete_with_error("Timeout", "TIMEOUT")
        
        assert not job.should_retry()


class TestCostData:
    """Test cases for CostData model."""
    
    def test_cost_data_creation(self):
        """Test CostData creation and methods."""
        cost_data = CostData()
        
        # Add operation costs
        cost_data.add_operation_cost("generation", 1000, 0.00005)
        cost_data.add_operation_cost("transformation", 500, 0.00003)
        
        assert cost_data.total_tokens_used == 1500
        assert cost_data.total_cost == 0.065
        assert len(cost_data.cost_per_operation) == 2
    
    def test_cost_data_validation(self):
        """Test CostData validation."""
        cost_data = CostData(
            total_tokens_used=1000,
            cost_per_operation={"generation": 0.05},
            total_cost=0.05
        )
        
        result = DataIntegrityValidator.validate_cost_data_integrity(cost_data)
        assert result.is_valid


class TestValidationFunctions:
    """Test cases for validation functions."""
    
    def test_content_type_validation(self):
        """Test content type validation."""
        assert validate_content_type("text")
        assert validate_content_type("image")
        assert not validate_content_type("invalid_type")
    
    def test_workflow_state_validation(self):
        """Test workflow state validation."""
        assert validate_workflow_state("create")
        assert validate_workflow_state("publish")
        assert not validate_workflow_state("invalid_state")
    
    def test_platform_validation(self):
        """Test platform validation."""
        assert validate_platform("twitter")
        assert validate_platform("instagram")
        assert not validate_platform("invalid_platform")
    
    def test_workflow_transition_validation(self):
        """Test workflow state transition validation."""
        # Valid transitions
        is_valid, error = validate_workflow_transition(
            WorkflowState.CREATE, WorkflowState.PUBLISH
        )
        assert is_valid
        assert error is None
        
        # Invalid transitions
        is_valid, error = validate_workflow_transition(
            WorkflowState.DISCOVER, WorkflowState.PUBLISH
        )
        assert not is_valid
        assert error is not None
    
    def test_content_metadata_validation(self):
        """Test content metadata validation."""
        valid_metadata = {
            "author": "test_user",
            "language": "en",
            "content_length": 100,
            "word_count": 20
        }
        
        is_valid, errors = validate_content_metadata(valid_metadata)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid metadata
        invalid_metadata = {
            "language": "invalid_lang",  # Should be 2 characters
            "content_length": -1,  # Should be non-negative
            "word_count": "not_a_number"  # Should be integer
        }
        
        is_valid, errors = validate_content_metadata(invalid_metadata)
        assert not is_valid
        assert len(errors) > 0
    
    def test_job_parameters_validation(self):
        """Test job parameters validation."""
        # Valid content generation parameters
        valid_params = {
            "prompt": "Write about AI",
            "content_type": "text"
        }
        
        is_valid, errors = validate_job_parameters("content_generation", valid_params)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid parameters (missing required)
        invalid_params = {
            "prompt": "Write about AI"
            # Missing content_type
        }
        
        is_valid, errors = validate_job_parameters("content_generation", invalid_params)
        assert not is_valid
        assert len(errors) > 0


class TestDataIntegrityValidator:
    """Test cases for DataIntegrityValidator."""
    
    def test_system_consistency_validation(self):
        """Test system-wide consistency validation."""
        # Create test data
        metadata = ContentMetadata(author="test_user")
        content_item = ContentItem(
            type=ContentType.TEXT,
            title="Test Content",
            content="Test content",
            content_metadata=metadata,
            user_id="user123"
        )
        
        job = AsyncJob(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Write about AI"},
            user_id="user123",
            content_id=str(content_item.id)  # Use the actual content item ID
        )
        
        result = DataIntegrityValidator.validate_system_consistency([content_item], [job])
        assert result.is_valid
    
    def test_business_rules_validation(self):
        """Test business rules validation."""
        metadata = ContentMetadata(author="test_user")
        content_item = ContentItem(
            type=ContentType.TEXT,
            title="Test Content",
            content="Test content",
            content_metadata=metadata,
            user_id="user123",
            tags=["test", "validation", "business"]
        )
        
        user_limits = {
            "max_content_length": 1000,
            "max_tags": 5,
            "max_versions": 10
        }
        
        result = DataIntegrityValidator.validate_business_rules(content_item, user_limits)
        assert result.is_valid


class TestEngagementMetrics:
    """Test cases for EngagementMetrics model."""
    
    def test_engagement_rate_calculation(self):
        """Test engagement rate calculation."""
        metrics = EngagementMetrics(
            views=1000,
            likes=50,
            shares=10,
            comments=5,
            reach=800
        )
        
        engagement_rate = metrics.calculate_engagement_rate()
        expected_rate = ((50 + 10 + 5) / 800) * 100  # 8.125%
        
        assert abs(engagement_rate - expected_rate) < 0.01
    
    def test_zero_reach_engagement_rate(self):
        """Test engagement rate calculation with zero reach."""
        metrics = EngagementMetrics(
            likes=50,
            shares=10,
            comments=5,
            reach=0
        )
        
        engagement_rate = metrics.calculate_engagement_rate()
        assert engagement_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
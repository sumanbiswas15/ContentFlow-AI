"""
Data integrity validation module for ContentFlow AI.

This module provides comprehensive validation functions for ensuring
data integrity across all models and operations.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from app.models.base import (
    ContentType, WorkflowState, JobStatus, JobType, Platform,
    ValidationResult, ErrorCode
)
from app.models.content import ContentItem, ContentMetadata, CostData
from app.models.jobs import AsyncJob, JobResult
from app.utils.validators import (
    validate_content_type, validate_workflow_state, validate_platform,
    validate_content_length, validate_tags, validate_priority,
    validate_workflow_transition, validate_content_metadata,
    validate_job_parameters, validate_cost_data, validate_engagement_metrics
)


class DataIntegrityValidator:
    """Comprehensive data integrity validator for ContentFlow AI models."""
    
    @staticmethod
    def validate_content_item(content_item: ContentItem) -> ValidationResult:
        """Validate a ContentItem for data integrity."""
        result = ValidationResult.success()
        
        # Validate basic fields
        if not content_item.title or not content_item.title.strip():
            result.add_error("Content title cannot be empty")
        
        if len(content_item.title) > 200:
            result.add_error("Content title cannot exceed 200 characters")
        
        # Validate content type
        if not validate_content_type(content_item.type.value):
            result.add_error(f"Invalid content type: {content_item.type}")
        
        # Validate workflow state
        if not validate_workflow_state(content_item.workflow_state.value):
            result.add_error(f"Invalid workflow state: {content_item.workflow_state}")
        
        # Validate content based on type
        if content_item.type == ContentType.TEXT:
            if not isinstance(content_item.content, str):
                result.add_error("Text content must be a string")
            else:
                is_valid, error = validate_content_length(content_item.content)
                if not is_valid:
                    result.add_error(error)
        
        # Validate version
        if content_item.version < 1:
            result.add_error("Content version must be at least 1")
        
        # Validate metadata
        metadata_valid, metadata_errors = validate_content_metadata(
            content_item.content_metadata.dict()
        )
        if not metadata_valid:
            result.errors.extend(metadata_errors)
        
        # Validate tags
        tags_valid, tag_errors = validate_tags(content_item.tags)
        if not tags_valid:
            result.errors.extend(tag_errors)
        
        return result
    
    @staticmethod
    def validate_async_job(job: AsyncJob) -> ValidationResult:
        """Validate an AsyncJob for data integrity."""
        result = ValidationResult.success()
        
        # Validate job type
        try:
            JobType(job.job_type)
        except ValueError:
            result.add_error(f"Invalid job type: {job.job_type}")
        
        # Validate job status
        try:
            JobStatus(job.status)
        except ValueError:
            result.add_error(f"Invalid job status: {job.status}")
        
        # Validate engine
        if not job.engine or not job.engine.strip():
            result.add_error("Job engine cannot be empty")
        
        # Validate operation
        if not job.operation or not job.operation.strip():
            result.add_error("Job operation cannot be empty")
        
        # Validate priority
        if not validate_priority(job.priority):
            result.add_error("Job priority must be between 1 and 10")
        
        # Validate parameters based on job type
        params_valid, param_errors = validate_job_parameters(
            job.job_type.value, job.parameters
        )
        if not params_valid:
            result.errors.extend(param_errors)
        
        # Validate retry count
        if job.retry_count < 0:
            result.add_error("Retry count cannot be negative")
        
        # Validate execution timing
        if job.started_at and job.completed_at:
            if job.started_at > job.completed_at:
                result.add_error("Job start time cannot be after completion time")
        
        # Validate status consistency
        if job.status == JobStatus.RUNNING and not job.started_at:
            result.add_warning("Running job should have a start time")
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and not job.completed_at:
            result.add_warning("Completed/failed job should have a completion time")
        
        return result
    
    @staticmethod
    def validate_cost_data_integrity(cost_data: CostData) -> ValidationResult:
        """Validate CostData for integrity."""
        result = ValidationResult.success()
        
        # Validate using existing function
        cost_valid, cost_errors = validate_cost_data(cost_data.dict())
        if not cost_valid:
            result.errors.extend(cost_errors)
        
        # Additional integrity checks
        calculated_total = sum(cost_data.cost_per_operation.values())
        if abs(calculated_total - cost_data.total_cost) > 0.01:  # Allow for small floating point errors
            result.add_warning(
                f"Total cost ({cost_data.total_cost}) doesn't match sum of operation costs ({calculated_total})"
            )
        
        return result
    
    @staticmethod
    def validate_workflow_state_transition(
        current_state: WorkflowState, 
        target_state: WorkflowState,
        content_item: Optional[ContentItem] = None
    ) -> ValidationResult:
        """Validate workflow state transition."""
        result = ValidationResult.success()
        
        # Basic transition validation
        is_valid, error = validate_workflow_transition(current_state, target_state)
        if not is_valid:
            result.add_error(error)
        
        # Additional business logic validation
        if content_item:
            # Check if content is ready for publication
            if target_state == WorkflowState.PUBLISH:
                if not content_item.content or (
                    isinstance(content_item.content, str) and not content_item.content.strip()
                ):
                    result.add_error("Cannot publish empty content")
                
                if content_item.type == ContentType.TEXT and len(content_item.content) < 10:
                    result.add_warning("Publishing very short content")
            
            # Check if content can be analyzed
            if target_state == WorkflowState.ANALYZE:
                if not content_item.is_published:
                    result.add_warning("Analyzing unpublished content")
        
        return result
    
    @staticmethod
    def validate_data_serialization_roundtrip(
        original_data: Dict[str, Any],
        serialized_data: str,
        deserialized_data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate data serialization round-trip integrity."""
        result = ValidationResult.success()
        
        # Check if deserialized data matches original
        if original_data != deserialized_data:
            result.add_error("Serialization round-trip failed: data mismatch")
            
            # Identify specific differences
            original_keys = set(original_data.keys())
            deserialized_keys = set(deserialized_data.keys())
            
            missing_keys = original_keys - deserialized_keys
            extra_keys = deserialized_keys - original_keys
            
            if missing_keys:
                result.add_error(f"Missing keys after deserialization: {missing_keys}")
            
            if extra_keys:
                result.add_error(f"Extra keys after deserialization: {extra_keys}")
            
            # Check value differences for common keys
            common_keys = original_keys & deserialized_keys
            for key in common_keys:
                if original_data[key] != deserialized_data[key]:
                    result.add_error(f"Value mismatch for key '{key}': {original_data[key]} != {deserialized_data[key]}")
        
        # Validate serialized data is not empty
        if not serialized_data or not serialized_data.strip():
            result.add_error("Serialized data is empty")
        
        return result
    
    @staticmethod
    def validate_system_consistency(
        content_items: List[ContentItem],
        jobs: List[AsyncJob]
    ) -> ValidationResult:
        """Validate system-wide data consistency."""
        result = ValidationResult.success()
        
        # Check for orphaned jobs (jobs referencing non-existent content)
        content_ids = {str(item.id) for item in content_items}  # Convert ObjectIds to strings
        for job in jobs:
            if job.content_id and job.content_id not in content_ids:
                result.add_error(f"Job {job.id} references non-existent content {job.content_id}")
        
        # Check for duplicate content titles within the same user
        user_titles = {}
        for item in content_items:
            user_id = item.user_id
            title = item.title.lower().strip()
            
            if user_id not in user_titles:
                user_titles[user_id] = set()
            
            if title in user_titles[user_id]:
                result.add_warning(f"Duplicate content title '{item.title}' for user {user_id}")
            else:
                user_titles[user_id].add(title)
        
        # Check for stale running jobs (running for too long)
        now = datetime.utcnow()
        for job in jobs:
            if job.status == JobStatus.RUNNING and job.started_at:
                runtime_hours = (now - job.started_at).total_seconds() / 3600
                if runtime_hours > 24:  # Jobs running for more than 24 hours
                    result.add_warning(f"Job {job.id} has been running for {runtime_hours:.1f} hours")
        
        return result
    
    @staticmethod
    def validate_business_rules(content_item: ContentItem, user_limits: Dict[str, Any]) -> ValidationResult:
        """Validate business rules and constraints."""
        result = ValidationResult.success()
        
        # Check content limits
        if "max_content_length" in user_limits:
            if isinstance(content_item.content, str):
                if len(content_item.content) > user_limits["max_content_length"]:
                    result.add_error(f"Content exceeds maximum length of {user_limits['max_content_length']} characters")
        
        # Check tag limits
        if "max_tags" in user_limits:
            if len(content_item.tags) > user_limits["max_tags"]:
                result.add_error(f"Content has too many tags (max: {user_limits['max_tags']})")
        
        # Check version limits
        if "max_versions" in user_limits:
            if content_item.version > user_limits["max_versions"]:
                result.add_error(f"Content has too many versions (max: {user_limits['max_versions']})")
        
        # Business logic: Published content should have engagement metrics
        if content_item.is_published and content_item.published_at:
            days_since_publish = (datetime.utcnow() - content_item.published_at).days
            if days_since_publish > 1:  # Published more than a day ago
                metrics = content_item.content_metadata.engagement_metrics
                if metrics.views == 0 and metrics.impressions == 0:
                    result.add_warning("Published content has no engagement metrics")
        
        return result


def validate_model_integrity(model: Any) -> ValidationResult:
    """Generic model integrity validator."""
    if isinstance(model, ContentItem):
        return DataIntegrityValidator.validate_content_item(model)
    elif isinstance(model, AsyncJob):
        return DataIntegrityValidator.validate_async_job(model)
    elif isinstance(model, CostData):
        return DataIntegrityValidator.validate_cost_data_integrity(model)
    else:
        result = ValidationResult.success()
        result.add_warning(f"No specific validator for model type: {type(model).__name__}")
        return result
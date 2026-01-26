"""
Job processing models for ContentFlow AI.

This module defines models for asynchronous job processing,
including job definitions, status tracking, and result management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

from app.models.base import BaseDocument, JobStatus, JobType, UserMixin


class JobResult(BaseModel):
    """Model for job execution results."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time_ms: Optional[int] = None
    tokens_used: int = 0
    cost: float = 0.0
    
    @classmethod
    def success_result(cls, data: Any, execution_time_ms: int = None, tokens_used: int = 0, cost: float = 0.0):
        """Create a successful job result."""
        return cls(
            success=True,
            data=data,
            execution_time_ms=execution_time_ms,
            tokens_used=tokens_used,
            cost=cost
        )
    
    @classmethod
    def error_result(cls, error_message: str, error_code: str = None, execution_time_ms: int = None):
        """Create an error job result."""
        return cls(
            success=False,
            error_message=error_message,
            error_code=error_code,
            execution_time_ms=execution_time_ms
        )


class RetryConfig(BaseModel):
    """Configuration for job retry behavior."""
    max_retries: int = 3
    retry_delay_seconds: int = 60
    exponential_backoff: bool = True
    retry_on_errors: List[str] = Field(default_factory=lambda: ["TIMEOUT", "SERVICE_UNAVAILABLE"])


class AsyncJob(BaseDocument, UserMixin):
    """Model for asynchronous job processing."""
    
    job_type: JobType
    status: JobStatus = JobStatus.QUEUED
    content_id: Optional[str] = None
    engine: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)  # 1 = highest, 10 = lowest
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    
    # Results and errors
    result: Optional[JobResult] = None
    
    # Retry handling
    retry_count: int = 0
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    last_error: Optional[str] = None
    
    # Dependencies and workflow
    depends_on: List[str] = Field(default_factory=list)  # Job IDs this job depends on
    blocks: List[str] = Field(default_factory=list)  # Job IDs that depend on this job
    workflow_id: Optional[str] = None
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority is within acceptable range."""
        if not 1 <= v <= 10:
            raise ValueError("Priority must be between 1 (highest) and 10 (lowest)")
        return v
    
    def start_execution(self):
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def complete_with_success(self, result_data: Any, tokens_used: int = 0, cost: float = 0.0):
        """Mark job as completed successfully."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        if self.started_at:
            execution_time = (self.completed_at - self.started_at).total_seconds() * 1000
            self.execution_time_ms = int(execution_time)
        
        self.result = JobResult.success_result(
            data=result_data,
            execution_time_ms=self.execution_time_ms,
            tokens_used=tokens_used,
            cost=cost
        )
        self.updated_at = datetime.utcnow()
    
    def complete_with_error(self, error_message: str, error_code: str = None):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.last_error = error_message
        
        if self.started_at:
            execution_time = (self.completed_at - self.started_at).total_seconds() * 1000
            self.execution_time_ms = int(execution_time)
        
        self.result = JobResult.error_result(
            error_message=error_message,
            error_code=error_code,
            execution_time_ms=self.execution_time_ms
        )
        self.updated_at = datetime.utcnow()
    
    def should_retry(self) -> bool:
        """Check if job should be retried based on retry configuration."""
        if self.retry_count >= self.retry_config.max_retries:
            return False
        
        if self.result and self.result.error_code:
            return self.result.error_code in self.retry_config.retry_on_errors
        
        return True
    
    def increment_retry(self):
        """Increment retry count and reset status for retry."""
        self.retry_count += 1
        self.status = JobStatus.QUEUED
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.updated_at = datetime.utcnow()
    
    def cancel(self):
        """Cancel the job."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def is_ready_to_execute(self, completed_job_ids: List[str]) -> bool:
        """Check if job is ready to execute based on dependencies."""
        if self.status != JobStatus.QUEUED:
            return False
        
        # Check if all dependencies are completed
        for dep_id in self.depends_on:
            if dep_id not in completed_job_ids:
                return False
        
        return True


class JobQueue(BaseModel):
    """Model for job queue management."""
    name: str
    max_concurrent_jobs: int = 5
    active_jobs: List[str] = Field(default_factory=list)  # Job IDs currently running
    paused: bool = False
    
    def can_accept_job(self) -> bool:
        """Check if queue can accept a new job."""
        return not self.paused and len(self.active_jobs) < self.max_concurrent_jobs
    
    def add_active_job(self, job_id: str):
        """Add job to active jobs list."""
        if job_id not in self.active_jobs:
            self.active_jobs.append(job_id)
    
    def remove_active_job(self, job_id: str):
        """Remove job from active jobs list."""
        if job_id in self.active_jobs:
            self.active_jobs.remove(job_id)


class WorkflowExecution(BaseDocument, UserMixin):
    """Model for tracking workflow execution across multiple jobs."""
    
    workflow_name: str
    description: Optional[str] = None
    job_ids: List[str] = Field(default_factory=list)
    status: str = "running"  # running, completed, failed, cancelled
    
    # Execution tracking
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Results
    final_result: Optional[Any] = None
    error_message: Optional[str] = None
    
    def add_job(self, job_id: str):
        """Add job to workflow."""
        if job_id not in self.job_ids:
            self.job_ids.append(job_id)
            self.updated_at = datetime.utcnow()
    
    def complete_workflow(self, result: Any = None):
        """Mark workflow as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.final_result = result
        self.updated_at = datetime.utcnow()
    
    def fail_workflow(self, error_message: str):
        """Mark workflow as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
"""
Job processing API endpoints for ContentFlow AI.

This module provides REST API endpoints for managing async jobs,
including job submission, status checking, and queue management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.models.base import JobType, JobStatus
from app.services.job_service import get_job_service
from app.services.notifications import get_notification_service
from app.core.exceptions import JobProcessingError, ValidationError

router = APIRouter()


# Request/Response Models
class JobSubmissionRequest(BaseModel):
    """Request model for job submission."""
    job_type: JobType
    engine: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    content_id: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    depends_on: List[str] = Field(default_factory=list)
    workflow_id: Optional[str] = None


class JobSubmissionResponse(BaseModel):
    """Response model for job submission."""
    job_id: str
    status: str
    message: str
    estimated_wait_time_seconds: Optional[float] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    job_type: JobType
    engine: str
    operation: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    retry_count: int
    priority: int
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    queue_info: Optional[Dict[str, Any]] = None


class QueueStatusResponse(BaseModel):
    """Response model for queue status."""
    processor: Dict[str, Any]
    queues: Dict[str, Any]


class WorkflowRequest(BaseModel):
    """Request model for workflow creation."""
    workflow_name: str
    description: Optional[str] = None


class WorkflowJobRequest(BaseModel):
    """Request model for adding job to workflow."""
    job_type: JobType
    engine: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    content_id: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    depends_on: List[str] = Field(default_factory=list)


# Dependency to get current user (placeholder)
async def get_current_user() -> str:
    """Get current user ID. This is a placeholder implementation."""
    # In a real implementation, this would extract user ID from JWT token
    return "user123"


@router.post("/submit", response_model=JobSubmissionResponse)
async def submit_job(
    request: JobSubmissionRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Submit a new job for async processing.
    
    Args:
        request: Job submission details
        user_id: Current user ID
        
    Returns:
        Job submission response with job ID and status
    """
    try:
        job_service = get_job_service()
        job_id = await job_service.submit_job(
            job_type=request.job_type,
            engine=request.engine,
            operation=request.operation,
            parameters=request.parameters,
            user_id=user_id,
            content_id=request.content_id,
            priority=request.priority,
            depends_on=request.depends_on,
            workflow_id=request.workflow_id
        )
        
        # Get estimated wait time
        job_status = await job_service.get_job_status(job_id)
        wait_time = None
        if job_status and job_status.get("queue_info"):
            wait_time = job_status["queue_info"].get("estimated_wait_time_seconds")
        
        return JobSubmissionResponse(
            job_id=job_id,
            status="queued",
            message="Job submitted successfully",
            estimated_wait_time_seconds=wait_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get the status of a specific job.
    
    Args:
        job_id: Job ID to check
        user_id: Current user ID
        
    Returns:
        Job status details
    """
    job_service = get_job_service()
    job_status = await job_service.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns this job
    if job_status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JobStatusResponse(**job_status)


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Cancel a queued or running job.
    
    Args:
        job_id: Job ID to cancel
        user_id: Current user ID
        
    Returns:
        Cancellation result
    """
    # Check if job exists and user owns it
    job_service = get_job_service()
    job_status = await job_service.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = await job_service.cancel_job(job_id)
    
    if success:
        return {"message": "Job cancelled successfully"}
    else:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")


@router.get("/", response_model=List[JobStatusResponse])
async def get_user_jobs(
    user_id: str = Depends(get_current_user),
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Number of jobs to return"),
    skip: int = Query(0, ge=0, description="Number of jobs to skip")
):
    """
    Get jobs for the current user.
    
    Args:
        user_id: Current user ID
        status: Optional status filter
        limit: Maximum number of jobs to return
        skip: Number of jobs to skip
        
    Returns:
        List of user's jobs
    """
    job_service = get_job_service()
    jobs = await job_service.get_user_jobs(
        user_id=user_id,
        status=status,
        limit=limit,
        skip=skip
    )
    
    return [JobStatusResponse(**job) for job in jobs]


@router.get("/history")
async def get_job_history(
    user_id: str = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to include")
):
    """
    Get job history and statistics for the current user.
    
    Args:
        user_id: Current user ID
        days: Number of days to include in history
        
    Returns:
        Job history and statistics
    """
    job_service = get_job_service()
    history = await job_service.get_job_history(user_id=user_id, days=days)
    return history


@router.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """
    Get current queue status and statistics.
    
    Returns:
        Queue status information
    """
    job_service = get_job_service()
    status = await job_service.get_queue_status()
    return QueueStatusResponse(**status)


@router.post("/queue/{queue_name}/pause")
async def pause_queue(queue_name: str):
    """
    Pause a specific queue.
    
    Args:
        queue_name: Name of the queue to pause
        
    Returns:
        Operation result
    """
    job_service = get_job_service()
    success = await job_service.pause_queue(queue_name)
    
    if success:
        return {"message": f"Queue '{queue_name}' paused successfully"}
    else:
        raise HTTPException(status_code=404, detail="Queue not found")


@router.post("/queue/{queue_name}/resume")
async def resume_queue(queue_name: str):
    """
    Resume a paused queue.
    
    Args:
        queue_name: Name of the queue to resume
        
    Returns:
        Operation result
    """
    job_service = get_job_service()
    success = await job_service.resume_queue(queue_name)
    
    if success:
        return {"message": f"Queue '{queue_name}' resumed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Queue not found")


@router.put("/queue/{queue_name}/limits")
async def update_queue_limits(
    queue_name: str,
    max_concurrent: int = Query(..., ge=1, le=20, description="Maximum concurrent jobs")
):
    """
    Update the concurrent job limit for a queue.
    
    Args:
        queue_name: Name of the queue to update
        max_concurrent: New maximum concurrent jobs limit
        
    Returns:
        Operation result
    """
    job_service = get_job_service()
    success = await job_service.update_queue_limits(queue_name, max_concurrent)
    
    if success:
        return {"message": f"Queue '{queue_name}' limit updated to {max_concurrent}"}
    else:
        raise HTTPException(status_code=404, detail="Queue not found")


# Workflow endpoints
@router.post("/workflows", response_model=Dict[str, str])
async def create_workflow(
    request: WorkflowRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Create a new workflow.
    
    Args:
        request: Workflow creation details
        user_id: Current user ID
        
    Returns:
        Workflow ID
    """
    job_service = get_job_service()
    workflow_id = await job_service.workflow_manager.create_workflow(
        workflow_name=request.workflow_name,
        user_id=user_id,
        description=request.description
    )
    
    return {"workflow_id": workflow_id}


@router.post("/workflows/{workflow_id}/jobs", response_model=Dict[str, str])
async def add_job_to_workflow(
    workflow_id: str,
    request: WorkflowJobRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Add a job to an existing workflow.
    
    Args:
        workflow_id: Workflow ID
        request: Job details
        user_id: Current user ID
        
    Returns:
        Job ID
    """
    job_service = get_job_service()
    job_id = await job_service.workflow_manager.add_job_to_workflow(
        workflow_id=workflow_id,
        job_type=request.job_type,
        engine=request.engine,
        operation=request.operation,
        parameters=request.parameters,
        user_id=user_id,
        content_id=request.content_id,
        priority=request.priority,
        depends_on=request.depends_on
    )
    
    return {"job_id": job_id}


@router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get workflow status with job details.
    
    Args:
        workflow_id: Workflow ID
        user_id: Current user ID
        
    Returns:
        Workflow status and job details
    """
    job_service = get_job_service()
    status = await job_service.workflow_manager.get_workflow_status(workflow_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Check if user owns this workflow
    if status["workflow"].get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return status


# Notification endpoints
@router.get("/notifications")
async def get_notifications(
    user_id: str = Depends(get_current_user),
    unread_only: bool = Query(False, description="Return only unread notifications"),
    limit: int = Query(50, ge=1, le=100, description="Number of notifications to return"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip")
):
    """
    Get notifications for the current user.
    
    Args:
        user_id: Current user ID
        unread_only: Whether to return only unread notifications
        limit: Maximum number of notifications to return
        skip: Number of notifications to skip
        
    Returns:
        List of notifications
    """
    notification_service = get_notification_service()
    notifications = await notification_service.get_user_notifications(
        user_id=user_id,
        unread_only=unread_only,
        limit=limit,
        skip=skip
    )
    
    return notifications


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Mark a notification as read.
    
    Args:
        notification_id: Notification ID
        user_id: Current user ID
        
    Returns:
        Operation result
    """
    notification_service = get_notification_service()
    success = await notification_service.mark_notification_read(notification_id, user_id)
    
    if success:
        return {"message": "Notification marked as read"}
    else:
        raise HTTPException(status_code=404, detail="Notification not found")


@router.post("/notifications/read-all")
async def mark_all_notifications_read(user_id: str = Depends(get_current_user)):
    """
    Mark all notifications as read for the current user.
    
    Args:
        user_id: Current user ID
        
    Returns:
        Number of notifications marked as read
    """
    notification_service = get_notification_service()
    count = await notification_service.mark_all_notifications_read(user_id)
    return {"message": f"Marked {count} notifications as read"}
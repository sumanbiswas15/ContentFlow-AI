"""
Main job service that integrates all job processing components.

This module provides a unified interface for job processing,
combining the job processor, queue manager, and notification system.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.models.jobs import AsyncJob
from app.models.base import JobType, JobStatus
from app.services.job_processor import get_job_processor, get_workflow_manager
from app.services.queue_manager import get_queue_manager
from app.services.notifications import get_notification_service, job_completion_handler

logger = logging.getLogger(__name__)


class JobService:
    """
    Unified job service that coordinates all job processing components.
    
    This service provides a single interface for:
    - Submitting jobs
    - Managing queues
    - Handling notifications
    - Monitoring job status
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        self.database = database or get_database()
        
        # Initialize components (lazy-loaded)
        self.queue_manager = get_queue_manager()
        self.job_processor = get_job_processor()
        self.notification_service = get_notification_service()
        self.workflow_manager = get_workflow_manager()
        
        # Override database for all components if provided
        if database:
            self.queue_manager.database = database
            self.job_processor.database = database
            self.notification_service.database = database
            self.workflow_manager.database = database
        
        # Register notification handler with job processor
        self.job_processor.register_notification_handler(job_completion_handler)
        
        # Override job processor's get_next_job method to use queue manager
        self.job_processor._get_next_job = self._get_next_job_with_queue_manager
        
        self._started = False
    
    async def start(self):
        """Start the job service."""
        if self._started:
            logger.warning("Job service is already started")
            return
        
        logger.info("Starting job service...")
        
        # Start job processor
        await self.job_processor.start()
        
        self._started = True
        logger.info("Job service started successfully")
    
    async def stop(self):
        """Stop the job service."""
        if not self._started:
            return
        
        logger.info("Stopping job service...")
        
        # Stop job processor
        await self.job_processor.stop()
        
        self._started = False
        logger.info("Job service stopped")
    
    async def submit_job(
        self,
        job_type: JobType,
        engine: str,
        operation: str,
        parameters: Dict[str, Any],
        user_id: str,
        content_id: Optional[str] = None,
        priority: int = 5,
        depends_on: List[str] = None,
        workflow_id: Optional[str] = None
    ) -> str:
        """Submit a new job for processing."""
        job_id = await self.job_processor.submit_job(
            job_type=job_type,
            engine=engine,
            operation=operation,
            parameters=parameters,
            user_id=user_id,
            content_id=content_id,
            priority=priority,
            depends_on=depends_on,
            workflow_id=workflow_id
        )
        
        # Get the job and send notification
        job = await self.job_processor.get_job_status(job_id)
        if job:
            await self.notification_service.notify_job_started(job)
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive job status including queue position."""
        job = await self.job_processor.get_job_status(job_id)
        if not job:
            return None
        
        result = job.dict()
        
        # Add queue information if job is queued
        if job.status == JobStatus.QUEUED:
            position = await self.queue_manager.get_job_position(job_id)
            wait_time = await self.queue_manager.get_estimated_wait_time(job_id)
            
            result["queue_info"] = {
                "position": position,
                "estimated_wait_time_seconds": wait_time.total_seconds() if wait_time else None
            }
        
        return result
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        success = await self.job_processor.cancel_job(job_id)
        
        if success:
            # Update queue manager
            await self.queue_manager.mark_job_completed(job_id)
            
            # Send notification
            job = await self.job_processor.get_job_status(job_id)
            if job:
                await self.notification_service.notify_job_cancelled(job)
        
        return success
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status."""
        processor_status = await self.job_processor.get_queue_status()
        queue_stats = await self.queue_manager.get_queue_statistics()
        
        return {
            "processor": processor_status,
            "queues": queue_stats
        }
    
    async def pause_queue(self, queue_name: str) -> bool:
        """Pause a specific queue."""
        return await self.queue_manager.pause_queue(queue_name)
    
    async def resume_queue(self, queue_name: str) -> bool:
        """Resume a paused queue."""
        return await self.queue_manager.resume_queue(queue_name)
    
    async def update_queue_limits(self, queue_name: str, max_concurrent: int) -> bool:
        """Update queue limits."""
        return await self.queue_manager.update_queue_limits(queue_name, max_concurrent)
    
    async def register_job_handler(self, job_type: JobType, handler: Callable):
        """Register a handler for a specific job type."""
        self.job_processor.register_job_handler(job_type, handler)
    
    async def get_user_jobs(
        self,
        user_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get jobs for a specific user."""
        query = {"user_id": user_id}
        if status:
            query["status"] = status
        
        cursor = self.database.async_jobs.find(query).sort("created_at", -1).skip(skip).limit(limit)
        jobs = []
        
        async for job_doc in cursor:
            job = AsyncJob(**job_doc)
            job_dict = job.dict()
            
            # Add queue info for queued jobs
            if job.status == JobStatus.QUEUED:
                position = await self.queue_manager.get_job_position(str(job.id))
                wait_time = await self.queue_manager.get_estimated_wait_time(str(job.id))
                
                job_dict["queue_info"] = {
                    "position": position,
                    "estimated_wait_time_seconds": wait_time.total_seconds() if wait_time else None
                }
            
            jobs.append(job_dict)
        
        return jobs
    
    async def get_job_history(
        self,
        user_id: str,
        days: int = 30,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get job history and statistics for a user."""
        from datetime import datetime, timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get job statistics
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "created_at": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "total_execution_time": {"$sum": "$execution_time_ms"},
                    "total_tokens": {"$sum": "$result.tokens_used"},
                    "total_cost": {"$sum": "$result.cost"}
                }
            }
        ]
        
        stats = {}
        async for doc in self.database.async_jobs.aggregate(pipeline):
            stats[doc["_id"]] = {
                "count": doc["count"],
                "total_execution_time_ms": doc["total_execution_time"] or 0,
                "total_tokens_used": doc["total_tokens"] or 0,
                "total_cost": doc["total_cost"] or 0.0
            }
        
        # Get recent jobs
        recent_jobs = await self.get_user_jobs(user_id, limit=limit)
        
        return {
            "period_days": days,
            "statistics": stats,
            "recent_jobs": recent_jobs
        }
    
    async def _get_next_job_with_queue_manager(self) -> Optional[AsyncJob]:
        """Get next job using queue manager."""
        job = await self.queue_manager.get_next_job()
        
        if job:
            # Send notification that job is starting
            await self.notification_service.notify_job_started(job)
        
        return job
    
    async def _on_job_completed(self, job: AsyncJob, success: bool):
        """Handle job completion."""
        # Update queue manager
        if success:
            await self.queue_manager.mark_job_completed(str(job.id))
        else:
            await self.queue_manager.mark_job_failed(str(job.id))
        
        # Check workflow completion if job is part of a workflow
        if job.workflow_id:
            await self.workflow_manager.check_workflow_completion(job.workflow_id)


# Global job service instance (lazy-loaded)
job_service = None


def get_job_service():
    """Get or create the global job service instance."""
    global job_service
    if job_service is None:
        job_service = JobService()
    return job_service


# Startup and shutdown functions for FastAPI
async def startup_job_service():
    """Start the job service on application startup."""
    service = get_job_service()
    await service.start()


async def shutdown_job_service():
    """Stop the job service on application shutdown."""
    service = get_job_service()
    await service.stop()
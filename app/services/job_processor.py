"""
Async job processing system for ContentFlow AI.

This module implements background job processing with status updates,
retry logic with exponential backoff, and job completion notifications.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.core.exceptions import JobProcessingError, OrchestrationError
from app.models.jobs import AsyncJob, JobQueue, JobResult, WorkflowExecution
from app.models.base import JobStatus, JobType

logger = logging.getLogger(__name__)


class JobProcessor:
    """
    Async job processor with retry logic and status tracking.
    
    Handles background execution of AI engine tasks with:
    - Queue management and priority handling
    - Exponential backoff retry logic
    - Real-time status updates
    - Job completion and failure notifications
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        self.database = database or get_database()
        self.job_handlers: Dict[JobType, Callable] = {}
        self.notification_handlers: List[Callable] = []
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.max_workers = 5
        self.poll_interval = 1.0  # seconds
        
    def register_job_handler(self, job_type: JobType, handler: Callable):
        """Register a handler function for a specific job type."""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    def register_notification_handler(self, handler: Callable):
        """Register a notification handler for job completion/failure."""
        self.notification_handlers.append(handler)
        logger.info("Registered notification handler")
    
    async def start(self):
        """Start the job processor with worker tasks."""
        if self.running:
            logger.warning("Job processor is already running")
            return
        
        self.running = True
        logger.info(f"Starting job processor with {self.max_workers} workers")
        
        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info("Job processor started successfully")
    
    async def stop(self):
        """Stop the job processor and cancel all worker tasks."""
        if not self.running:
            return
        
        logger.info("Stopping job processor...")
        self.running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Job processor stopped")
    
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
        """
        Submit a new job for processing.
        
        Args:
            job_type: Type of job to execute
            engine: Target engine for the job
            operation: Specific operation to perform
            parameters: Job parameters
            user_id: User submitting the job
            content_id: Optional content ID for the job
            priority: Job priority (1=highest, 10=lowest)
            depends_on: List of job IDs this job depends on
            workflow_id: Optional workflow ID for grouping jobs
            
        Returns:
            Job ID of the submitted job
        """
        job = AsyncJob(
            job_type=job_type,
            engine=engine,
            operation=operation,
            parameters=parameters,
            user_id=user_id,
            content_id=content_id,
            priority=priority,
            depends_on=depends_on or [],
            workflow_id=workflow_id
        )
        
        # Insert job into database
        result = await self.database.async_jobs.insert_one(job.dict())
        job_id = str(result.inserted_id)
        
        logger.info(f"Submitted job {job_id} of type {job_type} for user {user_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[AsyncJob]:
        """Get current status of a job."""
        job_data = await self.database.async_jobs.find_one({"_id": job_id})
        if job_data:
            return AsyncJob(**job_data)
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        job = await self.get_job_status(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        job.cancel()
        await self._update_job(job)
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        pipeline = [
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1}
            }}
        ]
        
        status_counts = {}
        async for doc in self.database.async_jobs.aggregate(pipeline):
            status_counts[doc["_id"]] = doc["count"]
        
        return {
            "queued": status_counts.get(JobStatus.QUEUED, 0),
            "running": status_counts.get(JobStatus.RUNNING, 0),
            "completed": status_counts.get(JobStatus.COMPLETED, 0),
            "failed": status_counts.get(JobStatus.FAILED, 0),
            "cancelled": status_counts.get(JobStatus.CANCELLED, 0),
            "workers_active": len([t for t in self.worker_tasks if not t.done()]),
            "processor_running": self.running
        }
    
    async def _worker(self, worker_name: str):
        """Worker task that processes jobs from the queue."""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get next job to process
                job = await self._get_next_job()
                
                if job:
                    logger.info(f"Worker {worker_name} processing job {job.id}")
                    await self._process_job(job)
                else:
                    # No jobs available, wait before checking again
                    await asyncio.sleep(self.poll_interval)
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _get_next_job(self) -> Optional[AsyncJob]:
        """Get the next job to process based on priority and dependencies."""
        # Get completed job IDs for dependency checking
        completed_jobs = []
        async for job_doc in self.database.async_jobs.find(
            {"status": JobStatus.COMPLETED},
            {"_id": 1}
        ):
            completed_jobs.append(str(job_doc["_id"]))
        
        # Find queued jobs ordered by priority and creation time
        async for job_doc in self.database.async_jobs.find(
            {"status": JobStatus.QUEUED}
        ).sort([("priority", 1), ("created_at", 1)]):
            
            job = AsyncJob(**job_doc)
            
            # Check if job is ready to execute (dependencies satisfied)
            if job.is_ready_to_execute(completed_jobs):
                # Mark job as running and return it
                job.start_execution()
                await self._update_job(job)
                return job
        
        return None
    
    async def _process_job(self, job: AsyncJob):
        """Process a single job with error handling and retry logic."""
        start_time = time.time()
        
        try:
            # Get job handler
            handler = self.job_handlers.get(job.job_type)
            if not handler:
                raise JobProcessingError(
                    job_id=str(job.id),
                    message=f"No handler registered for job type: {job.job_type}"
                )
            
            # Execute the job
            result = await handler(job)
            
            # Calculate execution time and complete job
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract cost information from result if available
            tokens_used = getattr(result, 'tokens_used', 0)
            cost = getattr(result, 'cost', 0.0)
            
            job.complete_with_success(
                result_data=result,
                tokens_used=tokens_used,
                cost=cost
            )
            
            await self._update_job(job)
            await self._notify_job_completion(job, success=True)
            
            logger.info(f"Job {job.id} completed successfully in {execution_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)
            
            # Determine error code based on exception type
            error_code = self._get_error_code(e)
            
            job.complete_with_error(
                error_message=str(e),
                error_code=error_code
            )
            
            # Check if job should be retried
            if job.should_retry():
                await self._schedule_retry(job)
            else:
                await self._update_job(job)
                await self._notify_job_completion(job, success=False)
    
    async def _schedule_retry(self, job: AsyncJob):
        """Schedule a job for retry with exponential backoff."""
        job.increment_retry()
        
        # Calculate retry delay with exponential backoff
        if job.retry_config.exponential_backoff:
            delay = job.retry_config.retry_delay_seconds * (2 ** (job.retry_count - 1))
        else:
            delay = job.retry_config.retry_delay_seconds
        
        # Cap maximum delay at 1 hour
        delay = min(delay, 3600)
        
        logger.info(f"Scheduling retry for job {job.id} in {delay} seconds (attempt {job.retry_count})")
        
        # Update job in database
        await self._update_job(job)
        
        # Schedule retry (in a real implementation, you might use a task scheduler like Celery)
        asyncio.create_task(self._delayed_retry(job, delay))
    
    async def _delayed_retry(self, job: AsyncJob, delay: int):
        """Execute delayed retry of a job."""
        await asyncio.sleep(delay)
        
        # Re-fetch job to ensure it hasn't been cancelled
        current_job = await self.get_job_status(str(job.id))
        if current_job and current_job.status == JobStatus.QUEUED:
            logger.info(f"Retrying job {job.id} (attempt {current_job.retry_count})")
    
    async def _update_job(self, job: AsyncJob):
        """Update job in database."""
        job.updated_at = datetime.utcnow()
        await self.database.async_jobs.replace_one(
            {"_id": job.id},
            job.dict()
        )
    
    async def _notify_job_completion(self, job: AsyncJob, success: bool):
        """Send notifications for job completion or failure."""
        for handler in self.notification_handlers:
            try:
                await handler(job, success)
            except Exception as e:
                logger.error(f"Notification handler error: {e}", exc_info=True)
    
    def _get_error_code(self, exception: Exception) -> str:
        """Determine error code based on exception type."""
        if isinstance(exception, asyncio.TimeoutError):
            return "TIMEOUT"
        elif isinstance(exception, ConnectionError):
            return "SERVICE_UNAVAILABLE"
        elif isinstance(exception, JobProcessingError):
            return "JOB_PROCESSING_ERROR"
        elif isinstance(exception, OrchestrationError):
            return "ORCHESTRATION_ERROR"
        else:
            return "GENERAL_ERROR"


class WorkflowManager:
    """
    Manages workflow execution across multiple jobs.
    
    Coordinates job dependencies and tracks workflow completion.
    """
    
    def __init__(self, job_processor: JobProcessor, database: AsyncIOMotorDatabase = None):
        self.job_processor = job_processor
        self.database = database or get_database()
    
    async def create_workflow(
        self,
        workflow_name: str,
        user_id: str,
        description: Optional[str] = None
    ) -> str:
        """Create a new workflow execution."""
        workflow = WorkflowExecution(
            workflow_name=workflow_name,
            user_id=user_id,
            description=description
        )
        
        result = await self.database.workflow_executions.insert_one(workflow.dict())
        workflow_id = str(result.inserted_id)
        
        logger.info(f"Created workflow {workflow_id}: {workflow_name}")
        return workflow_id
    
    async def add_job_to_workflow(
        self,
        workflow_id: str,
        job_type: JobType,
        engine: str,
        operation: str,
        parameters: Dict[str, Any],
        user_id: str,
        content_id: Optional[str] = None,
        priority: int = 5,
        depends_on: List[str] = None
    ) -> str:
        """Add a job to an existing workflow."""
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
        
        # Update workflow with new job
        await self.database.workflow_executions.update_one(
            {"_id": workflow_id},
            {
                "$push": {"job_ids": job_id},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return job_id
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status with job details."""
        workflow_data = await self.database.workflow_executions.find_one({"_id": workflow_id})
        if not workflow_data:
            return None
        
        workflow = WorkflowExecution(**workflow_data)
        
        # Get status of all jobs in workflow
        job_statuses = {}
        for job_id in workflow.job_ids:
            job = await self.job_processor.get_job_status(job_id)
            if job:
                job_statuses[job_id] = {
                    "status": job.status,
                    "engine": job.engine,
                    "operation": job.operation,
                    "created_at": job.created_at,
                    "completed_at": job.completed_at,
                    "retry_count": job.retry_count
                }
        
        return {
            "workflow": workflow.dict(),
            "jobs": job_statuses
        }
    
    async def check_workflow_completion(self, workflow_id: str):
        """Check if workflow is complete and update status."""
        workflow_data = await self.database.workflow_executions.find_one({"_id": workflow_id})
        if not workflow_data:
            return
        
        workflow = WorkflowExecution(**workflow_data)
        
        if workflow.status != "running":
            return
        
        # Check status of all jobs
        all_completed = True
        any_failed = False
        
        for job_id in workflow.job_ids:
            job = await self.job_processor.get_job_status(job_id)
            if job:
                if job.status == JobStatus.FAILED:
                    any_failed = True
                elif job.status not in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
                    all_completed = False
        
        # Update workflow status if needed
        if any_failed:
            workflow.fail_workflow("One or more jobs failed")
            await self.database.workflow_executions.replace_one(
                {"_id": workflow_id},
                workflow.dict()
            )
        elif all_completed:
            workflow.complete_workflow()
            await self.database.workflow_executions.replace_one(
                {"_id": workflow_id},
                workflow.dict()
            )


# Global job processor instance (lazy-loaded)
job_processor = None
workflow_manager = None


def get_job_processor():
    """Get or create the global job processor instance."""
    global job_processor
    if job_processor is None:
        job_processor = JobProcessor()
    return job_processor


def get_workflow_manager():
    """Get or create the global workflow manager instance."""
    global workflow_manager
    if workflow_manager is None:
        workflow_manager = WorkflowManager(get_job_processor())
    return workflow_manager
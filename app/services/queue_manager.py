"""
Job queue management for ContentFlow AI.

This module handles job queue management, priority scheduling,
and resource allocation for async job processing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.core.exceptions import JobProcessingError
from app.models.jobs import AsyncJob, JobQueue
from app.models.base import JobStatus, JobType, EngineType

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Manages job queues with priority scheduling and resource allocation.
    
    Features:
    - Multiple queues for different job types
    - Priority-based scheduling
    - Resource allocation and limits
    - Queue monitoring and statistics
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        self.database = database or get_database()
        self.queues: Dict[str, JobQueue] = {}
        self.engine_limits: Dict[str, int] = {}
        self.global_limit = 10  # Maximum concurrent jobs across all queues
        self.running_jobs: Set[str] = set()
        
        # Initialize default queues
        self._initialize_default_queues()
    
    def _initialize_default_queues(self):
        """Initialize default job queues for different engines."""
        default_queues = {
            "text_intelligence": JobQueue(name="text_intelligence", max_concurrent_jobs=3),
            "image_generation": JobQueue(name="image_generation", max_concurrent_jobs=2),
            "audio_generation": JobQueue(name="audio_generation", max_concurrent_jobs=2),
            "video_pipeline": JobQueue(name="video_pipeline", max_concurrent_jobs=1),
            "creative_assistant": JobQueue(name="creative_assistant", max_concurrent_jobs=3),
            "social_media_planner": JobQueue(name="social_media_planner", max_concurrent_jobs=2),
            "discovery_analytics": JobQueue(name="discovery_analytics", max_concurrent_jobs=2),
            "general": JobQueue(name="general", max_concurrent_jobs=2)
        }
        
        self.queues.update(default_queues)
        
        # Set engine-specific limits
        self.engine_limits = {
            "text_intelligence": 3,
            "image_generation": 2,
            "audio_generation": 2,
            "video_pipeline": 1,
            "creative_assistant": 3,
            "social_media_planner": 2,
            "discovery_analytics": 2
        }
        
        logger.info("Initialized default job queues")
    
    async def get_next_job(self) -> Optional[AsyncJob]:
        """
        Get the next job to process based on priority and resource availability.
        
        Returns:
            Next job to process or None if no jobs are available
        """
        # Check global limit
        if len(self.running_jobs) >= self.global_limit:
            return None
        
        # Get completed job IDs for dependency checking
        completed_jobs = await self._get_completed_job_ids()
        
        # Find the highest priority job that can be executed
        best_job = None
        best_priority = float('inf')
        
        # Query jobs ordered by priority and creation time
        async for job_doc in self.database.async_jobs.find(
            {"status": JobStatus.QUEUED}
        ).sort([("priority", 1), ("created_at", 1)]):
            
            job = AsyncJob(**job_doc)
            
            # Check if job is ready to execute (dependencies satisfied)
            if not job.is_ready_to_execute(completed_jobs):
                continue
            
            # Check queue capacity
            queue_name = self._get_queue_name(job.engine)
            queue = self.queues.get(queue_name)
            
            if not queue or not queue.can_accept_job():
                continue
            
            # Check engine-specific limits
            engine_limit = self.engine_limits.get(job.engine, 1)
            current_engine_jobs = await self._count_running_jobs_for_engine(job.engine)
            
            if current_engine_jobs >= engine_limit:
                continue
            
            # This job can be executed - check if it's the best priority
            if job.priority < best_priority:
                best_job = job
                best_priority = job.priority
        
        if best_job:
            # Mark job as running and update queue
            await self._mark_job_running(best_job)
        
        return best_job
    
    async def mark_job_completed(self, job_id: str):
        """Mark a job as completed and update queue status."""
        job = await self._get_job(job_id)
        if not job:
            return
        
        # Remove from running jobs
        self.running_jobs.discard(job_id)
        
        # Update queue
        queue_name = self._get_queue_name(job.engine)
        queue = self.queues.get(queue_name)
        if queue:
            queue.remove_active_job(job_id)
        
        logger.debug(f"Marked job {job_id} as completed in queue")
    
    async def mark_job_failed(self, job_id: str):
        """Mark a job as failed and update queue status."""
        await self.mark_job_completed(job_id)  # Same cleanup process
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        stats = {
            "global": {
                "running_jobs": len(self.running_jobs),
                "max_concurrent": self.global_limit,
                "utilization": len(self.running_jobs) / self.global_limit
            },
            "queues": {},
            "engines": {}
        }
        
        # Get job counts by status
        pipeline = [
            {"$group": {
                "_id": {"status": "$status", "engine": "$engine"},
                "count": {"$sum": 1}
            }}
        ]
        
        job_counts = {}
        async for doc in self.database.async_jobs.aggregate(pipeline):
            engine = doc["_id"]["engine"]
            status = doc["_id"]["status"]
            
            if engine not in job_counts:
                job_counts[engine] = {}
            job_counts[engine][status] = doc["count"]
        
        # Queue statistics
        for queue_name, queue in self.queues.items():
            engine_counts = job_counts.get(queue_name, {})
            
            stats["queues"][queue_name] = {
                "active_jobs": len(queue.active_jobs),
                "max_concurrent": queue.max_concurrent_jobs,
                "paused": queue.paused,
                "utilization": len(queue.active_jobs) / queue.max_concurrent_jobs,
                "queued": engine_counts.get(JobStatus.QUEUED, 0),
                "running": engine_counts.get(JobStatus.RUNNING, 0),
                "completed": engine_counts.get(JobStatus.COMPLETED, 0),
                "failed": engine_counts.get(JobStatus.FAILED, 0)
            }
        
        # Engine statistics
        for engine, limit in self.engine_limits.items():
            engine_counts = job_counts.get(engine, {})
            running_count = engine_counts.get(JobStatus.RUNNING, 0)
            
            stats["engines"][engine] = {
                "running_jobs": running_count,
                "max_concurrent": limit,
                "utilization": running_count / limit if limit > 0 else 0,
                "queued": engine_counts.get(JobStatus.QUEUED, 0),
                "completed": engine_counts.get(JobStatus.COMPLETED, 0),
                "failed": engine_counts.get(JobStatus.FAILED, 0)
            }
        
        return stats
    
    async def pause_queue(self, queue_name: str) -> bool:
        """Pause a specific queue."""
        if queue_name not in self.queues:
            return False
        
        self.queues[queue_name].paused = True
        logger.info(f"Paused queue: {queue_name}")
        return True
    
    async def resume_queue(self, queue_name: str) -> bool:
        """Resume a paused queue."""
        if queue_name not in self.queues:
            return False
        
        self.queues[queue_name].paused = False
        logger.info(f"Resumed queue: {queue_name}")
        return True
    
    async def update_queue_limits(self, queue_name: str, max_concurrent: int) -> bool:
        """Update the concurrent job limit for a queue."""
        if queue_name not in self.queues:
            return False
        
        self.queues[queue_name].max_concurrent_jobs = max_concurrent
        logger.info(f"Updated queue {queue_name} limit to {max_concurrent}")
        return True
    
    async def update_engine_limits(self, engine: str, max_concurrent: int) -> bool:
        """Update the concurrent job limit for an engine."""
        self.engine_limits[engine] = max_concurrent
        logger.info(f"Updated engine {engine} limit to {max_concurrent}")
        return True
    
    async def get_job_position(self, job_id: str) -> Optional[int]:
        """Get the position of a job in the queue (1-based)."""
        job = await self._get_job(job_id)
        if not job or job.status != JobStatus.QUEUED:
            return None
        
        # Count jobs with higher priority or same priority but earlier creation time
        position = await self.database.async_jobs.count_documents({
            "status": JobStatus.QUEUED,
            "$or": [
                {"priority": {"$lt": job.priority}},
                {
                    "priority": job.priority,
                    "created_at": {"$lt": job.created_at}
                }
            ]
        })
        
        return position + 1
    
    async def get_estimated_wait_time(self, job_id: str) -> Optional[timedelta]:
        """Estimate wait time for a queued job."""
        position = await self.get_job_position(job_id)
        if position is None:
            return None
        
        job = await self._get_job(job_id)
        if not job:
            return None
        
        # Get average processing time for this engine
        avg_time = await self._get_average_processing_time(job.engine)
        
        # Estimate based on position and average processing time
        queue_name = self._get_queue_name(job.engine)
        queue = self.queues.get(queue_name)
        
        if not queue:
            return timedelta(minutes=5)  # Default estimate
        
        # Calculate estimated wait time
        jobs_ahead = max(0, position - queue.max_concurrent_jobs)
        estimated_seconds = jobs_ahead * avg_time
        
        return timedelta(seconds=estimated_seconds)
    
    def _get_queue_name(self, engine: str) -> str:
        """Get queue name for an engine."""
        # Map engine names to queue names
        engine_to_queue = {
            "text_intelligence": "text_intelligence",
            "image_generation": "image_generation",
            "audio_generation": "audio_generation",
            "video_pipeline": "video_pipeline",
            "creative_assistant": "creative_assistant",
            "social_media_planner": "social_media_planner",
            "discovery_analytics": "discovery_analytics"
        }
        
        return engine_to_queue.get(engine, "general")
    
    async def _get_completed_job_ids(self) -> List[str]:
        """Get list of completed job IDs for dependency checking."""
        completed_jobs = []
        async for job_doc in self.database.async_jobs.find(
            {"status": JobStatus.COMPLETED},
            {"_id": 1}
        ):
            completed_jobs.append(str(job_doc["_id"]))
        
        return completed_jobs
    
    async def _count_running_jobs_for_engine(self, engine: str) -> int:
        """Count currently running jobs for a specific engine."""
        count = await self.database.async_jobs.count_documents({
            "status": JobStatus.RUNNING,
            "engine": engine
        })
        return count
    
    async def _mark_job_running(self, job: AsyncJob):
        """Mark a job as running and update queue status."""
        job.start_execution()
        
        # Update job in database
        await self.database.async_jobs.replace_one(
            {"_id": job.id},
            job.dict()
        )
        
        # Add to running jobs set
        self.running_jobs.add(str(job.id))
        
        # Update queue
        queue_name = self._get_queue_name(job.engine)
        queue = self.queues.get(queue_name)
        if queue:
            queue.add_active_job(str(job.id))
        
        logger.debug(f"Marked job {job.id} as running")
    
    async def _get_job(self, job_id: str) -> Optional[AsyncJob]:
        """Get job by ID."""
        job_data = await self.database.async_jobs.find_one({"_id": job_id})
        if job_data:
            return AsyncJob(**job_data)
        return None
    
    async def _get_average_processing_time(self, engine: str) -> float:
        """Get average processing time for an engine in seconds."""
        pipeline = [
            {
                "$match": {
                    "engine": engine,
                    "status": JobStatus.COMPLETED,
                    "execution_time_ms": {"$exists": True, "$ne": None}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_time_ms": {"$avg": "$execution_time_ms"}
                }
            }
        ]
        
        async for doc in self.database.async_jobs.aggregate(pipeline):
            return doc["avg_time_ms"] / 1000.0  # Convert to seconds
        
        # Default estimate if no historical data
        return 30.0  # 30 seconds


# Global queue manager instance (lazy-loaded)
queue_manager = None


def get_queue_manager():
    """Get or create the global queue manager instance."""
    global queue_manager
    if queue_manager is None:
        queue_manager = QueueManager()
    return queue_manager
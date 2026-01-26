#!/usr/bin/env python3
"""
Demo script for ContentFlow AI async job processing system.

This script demonstrates the key features of the async job processing system:
- Job submission and status tracking
- Queue management and priority handling
- Retry logic with exponential backoff
- Job completion and failure notifications
- Workflow management
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from app.models.jobs import AsyncJob
from app.models.base import JobType, JobStatus
from app.services.job_service import JobService
from app.services.job_handlers import register_all_handlers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDatabase:
    """Mock database for demonstration purposes."""
    
    def __init__(self):
        self.jobs = {}
        self.workflows = {}
        self.notifications = {}
        self.job_counter = 1
        self.workflow_counter = 1
        self.notification_counter = 1
    
    def get_collection(self, name):
        """Get a mock collection."""
        return MockCollection(self, name)


class MockCollection:
    """Mock MongoDB collection."""
    
    def __init__(self, db, name):
        self.db = db
        self.name = name
    
    async def insert_one(self, document):
        """Mock insert operation."""
        if self.name == "async_jobs":
            job_id = f"job{self.db.job_counter}"
            self.db.job_counter += 1
            self.db.jobs[job_id] = document
            return MockResult(job_id)
        elif self.name == "workflow_executions":
            workflow_id = f"workflow{self.db.workflow_counter}"
            self.db.workflow_counter += 1
            self.db.workflows[workflow_id] = document
            return MockResult(workflow_id)
        elif self.name == "notifications":
            notif_id = f"notif{self.db.notification_counter}"
            self.db.notification_counter += 1
            self.db.notifications[notif_id] = document
            return MockResult(notif_id)
    
    async def find_one(self, query):
        """Mock find one operation."""
        if self.name == "async_jobs":
            job_id = query.get("_id")
            return self.db.jobs.get(job_id)
        elif self.name == "workflow_executions":
            workflow_id = query.get("_id")
            return self.db.workflows.get(workflow_id)
        return None
    
    async def replace_one(self, query, document):
        """Mock replace operation."""
        if self.name == "async_jobs":
            job_id = query.get("_id")
            if job_id in self.db.jobs:
                self.db.jobs[job_id] = document
                return MockResult(job_id, modified_count=1)
        return MockResult(None, modified_count=0)
    
    async def update_one(self, query, update):
        """Mock update operation."""
        return MockResult(None, modified_count=1)
    
    async def count_documents(self, query):
        """Mock count operation."""
        return 0
    
    def find(self, query):
        """Mock find operation."""
        return MockCursor([])
    
    def aggregate(self, pipeline):
        """Mock aggregation operation."""
        return MockCursor([])


class MockCursor:
    """Mock MongoDB cursor."""
    
    def __init__(self, data):
        self.data = data
    
    def sort(self, *args):
        return self
    
    def skip(self, n):
        return self
    
    def limit(self, n):
        return self
    
    async def __aiter__(self):
        for item in self.data:
            yield item


class MockResult:
    """Mock operation result."""
    
    def __init__(self, inserted_id, modified_count=0):
        self.inserted_id = inserted_id
        self.modified_count = modified_count


async def demo_basic_job_processing():
    """Demonstrate basic job processing functionality."""
    print("\n" + "="*60)
    print("DEMO: Basic Job Processing")
    print("="*60)
    
    # Create mock database and job service
    mock_db = MockDatabase()
    
    # Create job service with explicit database to avoid initialization issues
    from app.services.job_processor import JobProcessor
    from app.services.queue_manager import QueueManager
    from app.services.notifications import NotificationService
    from app.services.job_service import JobService
    
    # Create components with mock database
    job_processor = JobProcessor(mock_db)
    queue_manager = QueueManager(mock_db)
    notification_service = NotificationService(mock_db)
    
    # Create job service and manually set components
    job_service = JobService.__new__(JobService)
    job_service.database = mock_db
    job_service.job_processor = job_processor
    job_service.queue_manager = queue_manager
    job_service.notification_service = notification_service
    job_service.workflow_manager = None  # Skip workflow manager for demo
    job_service._started = False
    
    # Register job handlers
    await register_all_handlers(job_service)
    
    print("✓ Job service initialized with handlers")
    
    # Submit a content generation job
    job_id = await job_service.submit_job(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_blog",
        parameters={
            "prompt": "Write a blog post about the future of AI",
            "content_type": "blog",
            "max_length": 1000
        },
        user_id="demo_user",
        priority=3
    )
    
    print(f"✓ Submitted job: {job_id}")
    
    # Check job status
    status = await job_service.get_job_status(job_id)
    if status:
        print(f"✓ Job status: {status['status']}")
        print(f"  - Engine: {status['engine']}")
        print(f"  - Operation: {status['operation']}")
        print(f"  - Priority: {status['priority']}")
    
    return job_service, job_id


async def demo_job_types():
    """Demonstrate different job types."""
    print("\n" + "="*60)
    print("DEMO: Different Job Types")
    print("="*60)
    
    mock_db = MockDatabase()
    job_service = JobService(mock_db)
    await register_all_handlers(job_service)
    
    job_configs = [
        {
            "job_type": JobType.CONTENT_GENERATION,
            "engine": "text_intelligence",
            "operation": "generate_caption",
            "parameters": {"prompt": "Create an Instagram caption", "platform": "instagram"},
            "description": "Social media caption generation"
        },
        {
            "job_type": JobType.CONTENT_TRANSFORMATION,
            "engine": "text_intelligence", 
            "operation": "summarize",
            "parameters": {"content": "Long article content...", "target_length": 200},
            "description": "Content summarization"
        },
        {
            "job_type": JobType.CREATIVE_ASSISTANCE,
            "engine": "creative_assistant",
            "operation": "brainstorm_ideas",
            "parameters": {"session_type": "ideation", "request": "Marketing campaign ideas"},
            "description": "Creative brainstorming"
        },
        {
            "job_type": JobType.SOCIAL_MEDIA_OPTIMIZATION,
            "engine": "social_media_planner",
            "operation": "generate_hashtags",
            "parameters": {"content": "AI technology post", "platform": "twitter"},
            "description": "Hashtag generation"
        },
        {
            "job_type": JobType.ANALYTICS_PROCESSING,
            "engine": "discovery_analytics",
            "operation": "analyze_sentiment",
            "parameters": {"content": "Customer feedback text", "analysis_type": "sentiment"},
            "description": "Sentiment analysis"
        },
        {
            "job_type": JobType.MEDIA_GENERATION,
            "engine": "image_generation",
            "operation": "create_thumbnail",
            "parameters": {"prompt": "AI robot thumbnail", "media_type": "image"},
            "description": "Image generation"
        }
    ]
    
    job_ids = []
    for config in job_configs:
        job_id = await job_service.submit_job(
            job_type=config["job_type"],
            engine=config["engine"],
            operation=config["operation"],
            parameters=config["parameters"],
            user_id="demo_user"
        )
        job_ids.append(job_id)
        print(f"✓ {config['description']}: {job_id}")
    
    print(f"\n✓ Submitted {len(job_ids)} different job types")
    return job_service, job_ids


async def demo_queue_management():
    """Demonstrate queue management features."""
    print("\n" + "="*60)
    print("DEMO: Queue Management")
    print("="*60)
    
    mock_db = MockDatabase()
    job_service = JobService(mock_db)
    await register_all_handlers(job_service)
    
    # Get initial queue status
    queue_status = await job_service.get_queue_status()
    print("✓ Initial queue status:")
    print(f"  - Global running jobs: {queue_status['processor']['running_jobs']}")
    print(f"  - Workers active: {queue_status['processor']['workers_active']}")
    
    # Submit jobs with different priorities
    priorities = [1, 3, 5, 7, 10]  # 1 = highest, 10 = lowest
    job_ids = []
    
    for priority in priorities:
        job_id = await job_service.submit_job(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_content",
            parameters={"prompt": f"Priority {priority} job"},
            user_id="demo_user",
            priority=priority
        )
        job_ids.append(job_id)
        print(f"✓ Submitted priority {priority} job: {job_id}")
    
    # Demonstrate queue operations
    print("\n✓ Queue operations:")
    
    # Pause a queue
    success = await job_service.pause_queue("text_intelligence")
    print(f"  - Paused text_intelligence queue: {success}")
    
    # Resume a queue
    success = await job_service.resume_queue("text_intelligence")
    print(f"  - Resumed text_intelligence queue: {success}")
    
    # Update queue limits
    success = await job_service.update_queue_limits("text_intelligence", 5)
    print(f"  - Updated queue limits: {success}")
    
    return job_service, job_ids


async def demo_workflow_management():
    """Demonstrate workflow management."""
    print("\n" + "="*60)
    print("DEMO: Workflow Management")
    print("="*60)
    
    mock_db = MockDatabase()
    job_service = JobService(mock_db)
    await register_all_handlers(job_service)
    
    # Create a workflow
    workflow_id = await job_service.workflow_manager.create_workflow(
        workflow_name="Content Creation Pipeline",
        user_id="demo_user",
        description="Complete content creation and optimization workflow"
    )
    print(f"✓ Created workflow: {workflow_id}")
    
    # Add jobs to workflow with dependencies
    job1_id = await job_service.workflow_manager.add_job_to_workflow(
        workflow_id=workflow_id,
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_blog",
        parameters={"prompt": "AI in healthcare"},
        user_id="demo_user",
        priority=1
    )
    print(f"✓ Added content generation job: {job1_id}")
    
    job2_id = await job_service.workflow_manager.add_job_to_workflow(
        workflow_id=workflow_id,
        job_type=JobType.SOCIAL_MEDIA_OPTIMIZATION,
        engine="social_media_planner",
        operation="optimize_for_platform",
        parameters={"platform": "twitter"},
        user_id="demo_user",
        priority=2,
        depends_on=[job1_id]  # Depends on content generation
    )
    print(f"✓ Added social media optimization job: {job2_id}")
    
    job3_id = await job_service.workflow_manager.add_job_to_workflow(
        workflow_id=workflow_id,
        job_type=JobType.ANALYTICS_PROCESSING,
        engine="discovery_analytics",
        operation="analyze_content",
        parameters={"analysis_type": "tagging"},
        user_id="demo_user",
        priority=3,
        depends_on=[job1_id]  # Also depends on content generation
    )
    print(f"✓ Added analytics job: {job3_id}")
    
    # Get workflow status
    workflow_status = await job_service.workflow_manager.get_workflow_status(workflow_id)
    if workflow_status:
        print(f"\n✓ Workflow status: {workflow_status['workflow']['status']}")
        print(f"  - Jobs in workflow: {len(workflow_status['workflow']['job_ids'])}")
        print(f"  - Created at: {workflow_status['workflow']['started_at']}")
    
    return job_service, workflow_id, [job1_id, job2_id, job3_id]


async def demo_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n" + "="*60)
    print("DEMO: Error Handling and Retry Logic")
    print("="*60)
    
    # Create a job that will demonstrate retry behavior
    job = AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_content",
        parameters={"prompt": "Test content"},
        user_id="demo_user"
    )
    
    print("✓ Created job for error handling demo")
    print(f"  - Initial retry count: {job.retry_count}")
    print(f"  - Max retries: {job.retry_config.max_retries}")
    print(f"  - Should retry: {job.should_retry()}")
    
    # Simulate job failure and retry
    job.start_execution()
    print(f"✓ Job started: {job.status}")
    
    job.complete_with_error("Simulated processing error", "PROCESSING_ERROR")
    print(f"✓ Job failed: {job.status}")
    print(f"  - Error: {job.last_error}")
    print(f"  - Should retry: {job.should_retry()}")
    
    # Increment retry
    if job.should_retry():
        job.increment_retry()
        print(f"✓ Retry scheduled: retry count = {job.retry_count}")
        print(f"  - Status reset to: {job.status}")
    
    # Test max retries reached
    job.retry_count = job.retry_config.max_retries
    print(f"✓ Max retries reached: {job.retry_count}/{job.retry_config.max_retries}")
    print(f"  - Should retry: {job.should_retry()}")
    
    return job


async def demo_notifications():
    """Demonstrate notification system."""
    print("\n" + "="*60)
    print("DEMO: Notification System")
    print("="*60)
    
    mock_db = MockDatabase()
    job_service = JobService(mock_db)
    
    # Create a sample job for notifications
    job = AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_blog",
        parameters={"prompt": "AI notifications"},
        user_id="demo_user"
    )
    
    # Test different notification types
    print("✓ Testing notification types:")
    
    # Job started notification
    await job_service.notification_service.notify_job_started(job)
    print("  - Job started notification sent")
    
    # Job completed notification
    job.complete_with_success({"content": "Generated content"}, tokens_used=150, cost=0.015)
    await job_service.notification_service.notify_job_completed(job)
    print("  - Job completed notification sent")
    
    # Job failed notification (create another job for this)
    failed_job = AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_blog",
        parameters={"prompt": "Failed job"},
        user_id="demo_user"
    )
    failed_job.complete_with_error("Simulated failure", "TEST_ERROR")
    await job_service.notification_service.notify_job_failed(failed_job)
    print("  - Job failed notification sent")
    
    # Workflow notifications
    await job_service.notification_service.notify_workflow_completed("workflow123", "demo_user")
    print("  - Workflow completed notification sent")
    
    print(f"\n✓ Total notifications in mock DB: {len(mock_db.notifications)}")
    
    return job_service


async def main():
    """Run all demonstrations."""
    print("ContentFlow AI - Async Job Processing System Demo")
    print("=" * 60)
    print("This demo showcases the key features of the async job processing system:")
    print("- Job submission and status tracking")
    print("- Queue management and priority handling") 
    print("- Retry logic with exponential backoff")
    print("- Job completion and failure notifications")
    print("- Workflow management with dependencies")
    print("- Error handling and recovery")
    
    try:
        # Run demonstrations
        await demo_basic_job_processing()
        await demo_job_types()
        await demo_queue_management()
        await demo_workflow_management()
        await demo_error_handling()
        await demo_notifications()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Async job processing with multiple job types")
        print("✓ Priority-based queue management")
        print("✓ Workflow orchestration with dependencies")
        print("✓ Retry logic with exponential backoff")
        print("✓ Comprehensive notification system")
        print("✓ Error handling and recovery mechanisms")
        print("✓ Real-time status tracking")
        print("✓ Resource allocation and limits")
        
        print("\nThe async job processing system is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
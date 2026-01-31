"""
Unit tests for the async job processing system.

This module tests the core job processing functionality including
job submission, execution, retry logic, and notifications.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.jobs import AsyncJob, JobResult
from app.models.base import JobType, JobStatus
from app.services.job_processor import JobProcessor, WorkflowManager
from app.services.queue_manager import QueueManager
from app.services.notifications import NotificationService
from app.services.job_service import JobService
from app.core.exceptions import JobProcessingError


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    db = MagicMock()
    
    # Mock collections
    db.async_jobs = AsyncMock()
    db.workflow_executions = AsyncMock()
    db.notifications = AsyncMock()
    
    # Mock insert operations
    db.async_jobs.insert_one = AsyncMock(return_value=MagicMock(inserted_id="job123"))
    db.workflow_executions.insert_one = AsyncMock(return_value=MagicMock(inserted_id="workflow123"))
    db.notifications.insert_one = AsyncMock()
    
    # Mock find operations
    db.async_jobs.find_one = AsyncMock(return_value=None)
    db.async_jobs.find = AsyncMock(return_value=AsyncMock())
    db.async_jobs.replace_one = AsyncMock()
    db.async_jobs.count_documents = AsyncMock(return_value=0)
    db.async_jobs.aggregate = AsyncMock(return_value=AsyncMock())
    
    return db


@pytest.fixture
def job_processor(mock_database):
    """Create job processor for testing."""
    processor = JobProcessor(mock_database)
    return processor


@pytest.fixture
def queue_manager(mock_database):
    """Create queue manager for testing."""
    manager = QueueManager(mock_database)
    return manager


@pytest.fixture
def notification_service(mock_database):
    """Create notification service for testing."""
    service = NotificationService(mock_database)
    return service


@pytest.fixture
def job_service(mock_database):
    """Create job service for testing."""
    with patch('app.services.job_service.get_queue_manager') as mock_qm, \
         patch('app.services.job_service.get_job_processor') as mock_jp, \
         patch('app.services.job_service.get_notification_service') as mock_ns, \
         patch('app.services.job_service.get_workflow_manager') as mock_wm:
        
        # Create mock instances
        job_proc = JobProcessor(mock_database)
        mock_qm.return_value = QueueManager(mock_database)
        mock_jp.return_value = job_proc
        mock_ns.return_value = NotificationService(mock_database)
        mock_wm.return_value = WorkflowManager(job_proc, mock_database)
        
        service = JobService(mock_database)
        yield service


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    return AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_blog",
        parameters={"prompt": "Write about AI", "max_length": 1000},
        user_id="user123",
        priority=5
    )


class TestJobProcessor:
    """Test cases for JobProcessor."""
    
    async def test_submit_job(self, job_processor, mock_database):
        """Test job submission."""
        job_id = await job_processor.submit_job(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Test prompt"},
            user_id="user123"
        )
        
        assert job_id == "job123"
        mock_database.async_jobs.insert_one.assert_called_once()
    
    async def test_get_job_status(self, job_processor, mock_database, sample_job):
        """Test getting job status."""
        # Mock database return
        mock_database.async_jobs.find_one.return_value = sample_job.dict()
        
        job = await job_processor.get_job_status("job123")
        
        assert job is not None
        assert job.job_type == JobType.CONTENT_GENERATION
        assert job.engine == "text_intelligence"
    
    async def test_cancel_job(self, job_processor, mock_database, sample_job):
        """Test job cancellation."""
        # Mock database return
        mock_database.async_jobs.find_one.return_value = sample_job.dict()
        
        success = await job_processor.cancel_job("job123")
        
        assert success is True
        mock_database.async_jobs.replace_one.assert_called_once()
    
    async def test_job_retry_logic(self, sample_job):
        """Test job retry logic."""
        # Test should retry
        assert sample_job.should_retry() is True
        
        # Test max retries reached
        sample_job.retry_count = 3
        assert sample_job.should_retry() is False
        
        # Test increment retry
        sample_job.retry_count = 1
        sample_job.increment_retry()
        assert sample_job.retry_count == 2
        assert sample_job.status == JobStatus.QUEUED
    
    async def test_job_completion(self, sample_job):
        """Test job completion methods."""
        # Test successful completion
        sample_job.start_execution()
        assert sample_job.status == JobStatus.RUNNING
        assert sample_job.started_at is not None
        
        # Complete with success
        sample_job.complete_with_success(
            result_data={"content": "Generated content"},
            tokens_used=100,
            cost=0.01
        )
        
        assert sample_job.status == JobStatus.COMPLETED
        assert sample_job.completed_at is not None
        assert sample_job.result.success is True
        assert sample_job.result.tokens_used == 100
        assert sample_job.result.cost == 0.01
    
    async def test_job_failure(self, sample_job):
        """Test job failure handling."""
        sample_job.start_execution()
        
        # Complete with error
        sample_job.complete_with_error(
            error_message="Processing failed",
            error_code="PROCESSING_ERROR"
        )
        
        assert sample_job.status == JobStatus.FAILED
        assert sample_job.completed_at is not None
        assert sample_job.result.success is False
        assert sample_job.result.error_message == "Processing failed"
        assert sample_job.result.error_code == "PROCESSING_ERROR"
    
    async def test_job_dependencies(self, sample_job):
        """Test job dependency checking."""
        # Job with no dependencies should be ready
        assert sample_job.is_ready_to_execute([]) is True
        
        # Job with dependencies
        sample_job.depends_on = ["job1", "job2"]
        
        # Not ready if dependencies not completed
        assert sample_job.is_ready_to_execute(["job1"]) is False
        
        # Ready if all dependencies completed
        assert sample_job.is_ready_to_execute(["job1", "job2"]) is True


class TestQueueManager:
    """Test cases for QueueManager."""
    
    async def test_queue_initialization(self, queue_manager):
        """Test queue manager initialization."""
        assert "text_intelligence" in queue_manager.queues
        assert "image_generation" in queue_manager.queues
        assert queue_manager.global_limit == 10
    
    async def test_queue_statistics(self, queue_manager, mock_database):
        """Test queue statistics generation."""
        # Mock aggregation result
        mock_database.async_jobs.aggregate.return_value = AsyncMock()
        mock_database.async_jobs.aggregate.return_value.__aiter__ = AsyncMock(return_value=iter([]))
        
        stats = await queue_manager.get_queue_statistics()
        
        assert "global" in stats
        assert "queues" in stats
        assert "engines" in stats
        assert stats["global"]["max_concurrent"] == 10
    
    async def test_queue_pause_resume(self, queue_manager):
        """Test queue pause and resume functionality."""
        # Test pause
        success = await queue_manager.pause_queue("text_intelligence")
        assert success is True
        assert queue_manager.queues["text_intelligence"].paused is True
        
        # Test resume
        success = await queue_manager.resume_queue("text_intelligence")
        assert success is True
        assert queue_manager.queues["text_intelligence"].paused is False
        
        # Test invalid queue
        success = await queue_manager.pause_queue("invalid_queue")
        assert success is False
    
    async def test_queue_limits_update(self, queue_manager):
        """Test updating queue limits."""
        # Test update queue limits
        success = await queue_manager.update_queue_limits("text_intelligence", 5)
        assert success is True
        assert queue_manager.queues["text_intelligence"].max_concurrent_jobs == 5
        
        # Test update engine limits
        success = await queue_manager.update_engine_limits("text_intelligence", 3)
        assert success is True
        assert queue_manager.engine_limits["text_intelligence"] == 3


class TestNotificationService:
    """Test cases for NotificationService."""
    
    async def test_job_notifications(self, notification_service, sample_job, mock_database):
        """Test job notification methods."""
        # Test job started notification
        await notification_service.notify_job_started(sample_job)
        
        # Test job completed notification
        sample_job.complete_with_success({"content": "test"}, tokens_used=50, cost=0.005)
        await notification_service.notify_job_completed(sample_job)
        
        # Test job failed notification
        sample_job.complete_with_error("Test error", "TEST_ERROR")
        await notification_service.notify_job_failed(sample_job)
        
        # Verify database calls
        assert mock_database.notifications.insert_one.call_count >= 3
    
    async def test_get_user_notifications(self, notification_service, mock_database):
        """Test getting user notifications."""
        # Mock database response
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter([
            {
                "_id": "notif1",
                "event_type": "job_completed",
                "job_id": "job123",
                "user_id": "user123",
                "job_type": "content_generation",
                "engine": "text_intelligence",
                "operation": "generate_blog",
                "status": "completed",
                "timestamp": datetime.utcnow(),
                "metadata": {},
                "read": False
            }
        ]))
        
        mock_database.notifications.find.return_value = mock_cursor
        
        notifications = await notification_service.get_user_notifications("user123")
        
        assert len(notifications) == 1
        assert notifications[0]["event_type"] == "job_completed"
        assert notifications[0]["user_id"] == "user123"
    
    async def test_mark_notifications_read(self, notification_service, mock_database):
        """Test marking notifications as read."""
        # Mock successful update
        mock_database.notifications.update_one.return_value = MagicMock(modified_count=1)
        mock_database.notifications.update_many.return_value = MagicMock(modified_count=5)
        
        # Test mark single notification read
        success = await notification_service.mark_notification_read("notif1", "user123")
        assert success is True
        
        # Test mark all notifications read
        count = await notification_service.mark_all_notifications_read("user123")
        assert count == 5


class TestJobService:
    """Test cases for JobService integration."""
    
    async def test_job_service_integration(self, job_service, mock_database):
        """Test job service integration."""
        # Test job submission
        job_id = await job_service.submit_job(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Test"},
            user_id="user123"
        )
        
        assert job_id == "job123"
        
        # Verify database interaction
        mock_database.async_jobs.insert_one.assert_called_once()
    
    async def test_job_service_status(self, job_service, mock_database, sample_job):
        """Test job service status retrieval."""
        # Mock database response
        mock_database.async_jobs.find_one.return_value = sample_job.dict()
        
        status = await job_service.get_job_status("job123")
        
        assert status is not None
        assert status["job_type"] == JobType.CONTENT_GENERATION
    
    async def test_queue_status_integration(self, job_service, mock_database):
        """Test queue status integration."""
        # Mock aggregation for queue statistics
        mock_database.async_jobs.aggregate.return_value = AsyncMock()
        mock_database.async_jobs.aggregate.return_value.__aiter__ = AsyncMock(return_value=iter([]))
        
        status = await job_service.get_queue_status()
        
        assert "processor" in status
        assert "queues" in status


class TestWorkflowManager:
    """Test cases for WorkflowManager."""
    
    async def test_create_workflow(self, mock_database):
        """Test workflow creation."""
        job_processor = JobProcessor(mock_database)
        workflow_manager = WorkflowManager(job_processor, mock_database)
        
        workflow_id = await workflow_manager.create_workflow(
            workflow_name="test_workflow",
            user_id="user123",
            description="Test workflow"
        )
        
        assert workflow_id == "workflow123"
        mock_database.workflow_executions.insert_one.assert_called_once()
    
    async def test_add_job_to_workflow(self, mock_database):
        """Test adding job to workflow."""
        job_processor = JobProcessor(mock_database)
        workflow_manager = WorkflowManager(job_processor, mock_database)
        
        job_id = await workflow_manager.add_job_to_workflow(
            workflow_id="workflow123",
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Test"},
            user_id="user123"
        )
        
        assert job_id == "job123"
        mock_database.workflow_executions.update_one.assert_called_once()


# Integration tests
class TestJobProcessingIntegration:
    """Integration tests for the complete job processing system."""
    
    @pytest.mark.asyncio
    @patch('app.services.job_service.get_queue_manager')
    @patch('app.services.job_service.get_job_processor')
    @patch('app.services.job_service.get_notification_service')
    @patch('app.services.job_service.get_workflow_manager')
    async def test_end_to_end_job_processing(self, mock_workflow_manager, mock_notification_service, 
                                           mock_job_processor, mock_queue_manager, mock_database):
        """Test end-to-end job processing flow."""
        # Create mock instances
        job_proc = JobProcessor(mock_database)
        
        # Mock the service dependencies
        mock_queue_manager.return_value = QueueManager(mock_database)
        mock_job_processor.return_value = job_proc
        mock_notification_service.return_value = NotificationService(mock_database)
        mock_workflow_manager.return_value = WorkflowManager(job_proc, mock_database)
        
        # Create job service
        job_service = JobService(mock_database)
        
        # Mock successful job handler
        async def mock_handler(job):
            return JobResult(data={"content": "Generated content"}, tokens_used=100, cost=0.01)
        
        # Register handler
        await job_service.register_job_handler(JobType.CONTENT_GENERATION, mock_handler)
        
        # Submit job
        job_id = await job_service.submit_job(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_blog",
            parameters={"prompt": "Test prompt"},
            user_id="user123"
        )
        
        assert job_id == "job123"
        
        # Verify job was submitted
        mock_database.async_jobs.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_job_retry_flow(self, mock_database, sample_job):
        """Test job retry flow."""
        # Configure job to retry on JOB_PROCESSING_ERROR
        from app.models.jobs import RetryConfig
        sample_job.retry_config = RetryConfig(
            max_retries=3,
            retry_delay_seconds=1,
            exponential_backoff=False,
            retry_on_errors=["JOB_PROCESSING_ERROR", "TIMEOUT", "SERVICE_UNAVAILABLE"]
        )
        
        job_processor = JobProcessor(mock_database)
        
        # Mock failing then succeeding handler
        call_count = 0
        
        async def mock_failing_handler(job):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise JobProcessingError(job_id=str(job.id), message="First attempt failed")
            return JobResult(data={"content": "Success on retry"})
        
        job_processor.register_job_handler(JobType.CONTENT_GENERATION, mock_failing_handler)
        
        # Mock database to return our sample job
        mock_database.async_jobs.find_one.return_value = sample_job.model_dump()
        
        # Process job (would normally be done by worker)
        await job_processor._process_job(sample_job)
        
        # Verify retry was scheduled (retry_count should be incremented)
        assert sample_job.retry_count > 0 or sample_job.status == JobStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__])
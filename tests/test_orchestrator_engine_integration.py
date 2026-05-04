"""
Tests for AI Orchestrator engine integration.

This module tests the integration between the AI Orchestrator and all
specialized engines, ensuring proper workflow coordination and data flow.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from app.ai.orchestrator import AIOrchestrator, WorkflowRequest, TaskPriority
from app.models.base import EngineType, WorkflowState, Platform
from app.core.exceptions import EngineError, ValidationError


@pytest.fixture
def orchestrator():
    """Create an orchestrator instance for testing."""
    return AIOrchestrator()


@pytest.mark.asyncio
async def test_orchestrator_initializes_all_engines(orchestrator):
    """Test that orchestrator initializes all specialized engines."""
    # Check that all engine types are initialized
    expected_engines = [
        EngineType.TEXT_INTELLIGENCE,
        EngineType.CREATIVE_ASSISTANT,
        EngineType.SOCIAL_MEDIA_PLANNER,
        EngineType.DISCOVERY_ANALYTICS,
        EngineType.IMAGE_GENERATION,
        EngineType.AUDIO_GENERATION,
        EngineType.VIDEO_PIPELINE
    ]
    
    for engine_type in expected_engines:
        engine = orchestrator.get_engine(engine_type)
        assert engine is not None, f"Engine {engine_type} not initialized"
        
        # Check engine capability is registered
        capability = orchestrator.engine_capabilities.get(engine_type)
        assert capability is not None, f"Capability for {engine_type} not registered"


@pytest.mark.asyncio
async def test_get_engine_returns_correct_instance(orchestrator):
    """Test that get_engine returns the correct engine instance."""
    text_engine = orchestrator.get_engine(EngineType.TEXT_INTELLIGENCE)
    assert text_engine is not None
    assert hasattr(text_engine, 'generate_content')
    
    creative_engine = orchestrator.get_engine(EngineType.CREATIVE_ASSISTANT)
    assert creative_engine is not None
    assert hasattr(creative_engine, 'start_creative_session')
    
    social_engine = orchestrator.get_engine(EngineType.SOCIAL_MEDIA_PLANNER)
    assert social_engine is not None
    assert hasattr(social_engine, 'optimize_for_platform')


@pytest.mark.asyncio
async def test_execute_engine_operation_text_intelligence(orchestrator):
    """Test executing operations on Text Intelligence Engine."""
    # Mock the engine to avoid actual API calls
    mock_result = Mock()
    mock_result.content = "Generated content"
    mock_result.metadata = {"tokens_used": 100}
    mock_result.cost = 0.01
    
    text_engine = orchestrator.get_engine(EngineType.TEXT_INTELLIGENCE)
    text_engine.generate_content = AsyncMock(return_value=mock_result)
    
    # Execute operation
    result = await orchestrator.execute_engine_operation(
        engine_type=EngineType.TEXT_INTELLIGENCE,
        operation="generate",
        parameters={
            "content_type": "blog",
            "prompt": "Write about AI",
            "tone": "professional"
        }
    )
    
    assert result is not None
    assert "content" in result
    assert result["content"] == "Generated content"
    assert result["cost"] == 0.01


@pytest.mark.asyncio
async def test_execute_engine_operation_creative_assistant(orchestrator):
    """Test executing operations on Creative Assistant Engine."""
    # Mock the engine
    creative_engine = orchestrator.get_engine(EngineType.CREATIVE_ASSISTANT)
    creative_engine.start_creative_session = AsyncMock(return_value="session_123")
    
    # Execute operation
    result = await orchestrator.execute_engine_operation(
        engine_type=EngineType.CREATIVE_ASSISTANT,
        operation="start_session",
        parameters={
            "session_type": "ideation",
            "topic": "Product launch"
        }
    )
    
    assert result is not None
    assert "session_id" in result
    assert result["session_id"] == "session_123"


@pytest.mark.asyncio
async def test_execute_engine_operation_social_media(orchestrator):
    """Test executing operations on Social Media Planner."""
    # Mock the engine
    mock_result = Mock()
    mock_result.dict = Mock(return_value={
        "optimized_content": "Optimized post",
        "hashtags": ["#test", "#content"],
        "platform": "twitter"
    })
    
    social_engine = orchestrator.get_engine(EngineType.SOCIAL_MEDIA_PLANNER)
    social_engine.optimize_for_platform = AsyncMock(return_value=mock_result)
    
    # Execute operation
    result = await orchestrator.execute_engine_operation(
        engine_type=EngineType.SOCIAL_MEDIA_PLANNER,
        operation="optimize",
        parameters={
            "content": "Test content",
            "platform": Platform.TWITTER
        }
    )
    
    assert result is not None
    assert "optimized_content" in result
    assert "hashtags" in result


@pytest.mark.asyncio
async def test_execute_engine_operation_unavailable_engine(orchestrator):
    """Test that executing operation on unavailable engine raises error."""
    # Mark engine as unavailable
    orchestrator.engine_capabilities[EngineType.TEXT_INTELLIGENCE].availability = False
    
    # Attempt to execute operation
    with pytest.raises(EngineError) as exc_info:
        await orchestrator.execute_engine_operation(
            engine_type=EngineType.TEXT_INTELLIGENCE,
            operation="generate",
            parameters={"prompt": "test"}
        )
    
    assert "unavailable" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_execute_engine_operation_unsupported_operation(orchestrator):
    """Test that unsupported operation raises error."""
    with pytest.raises(EngineError) as exc_info:
        await orchestrator.execute_engine_operation(
            engine_type=EngineType.TEXT_INTELLIGENCE,
            operation="unsupported_operation",
            parameters={}
        )
    
    assert "not supported" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_complete_workflow(orchestrator):
    """Test workflow completion with cleanup."""
    from bson import ObjectId
    
    # Create a mock workflow with proper ObjectId
    workflow_id = ObjectId()
    job_id = ObjectId()
    
    with patch('app.ai.orchestrator.get_database') as mock_db:
        # Mock database operations
        mock_collection = AsyncMock()
        mock_db.return_value.workflow_executions = mock_collection
        mock_db.return_value.async_jobs = AsyncMock()
        
        mock_collection.find_one = AsyncMock(return_value={
            "_id": workflow_id,
            "workflow_name": "Test Workflow",
            "user_id": "user_123",
            "job_ids": [str(job_id)],
            "status": "running",
            "created_at": datetime.utcnow()
        })
        
        mock_collection.update_one = AsyncMock()
        
        # Mock job results with all required fields
        mock_db.return_value.async_jobs.find_one = AsyncMock(return_value={
            "_id": job_id,
            "user_id": "user_123",
            "job_type": "content_generation",
            "engine": "text_intelligence",
            "operation": "generate",
            "status": "completed",
            "result": {"success": True, "cost": 0.05, "tokens_used": 100}
        })
        
        # Add workflow to active workflows
        from app.models.jobs import WorkflowExecution
        orchestrator.active_workflows[str(workflow_id)] = WorkflowExecution(
            workflow_name="Test Workflow",
            user_id="user_123"
        )
        
        # Complete workflow
        summary = await orchestrator.complete_workflow(
            workflow_id=str(workflow_id),
            final_status="completed",
            results={"output": "test"}
        )
        
        assert summary is not None
        assert summary["workflow_id"] == str(workflow_id)
        assert summary["status"] == "completed"
        assert workflow_id not in orchestrator.active_workflows


@pytest.mark.asyncio
async def test_transition_workflow_state(orchestrator):
    """Test workflow state transitions."""
    from bson import ObjectId
    
    workflow_id = ObjectId()
    content_id = ObjectId()
    
    with patch('app.ai.orchestrator.get_database') as mock_db:
        # Mock database operations
        mock_content_collection = AsyncMock()
        mock_workflow_collection = AsyncMock()
        
        mock_db.return_value.content_items = mock_content_collection
        mock_db.return_value.workflow_executions = mock_workflow_collection
        
        # Mock content item with all required fields
        mock_content_collection.find_one = AsyncMock(return_value={
            "_id": content_id,
            "user_id": "user_123",
            "type": "text",
            "title": "Test Content",
            "content": "Test content body",
            "workflow_state": WorkflowState.DISCOVER,
            "content_metadata": {
                "author": "Test Author",
                "processing_history": []
            }
        })
        
        mock_content_collection.update_one = AsyncMock()
        mock_workflow_collection.update_one = AsyncMock()
        
        # Transition state
        await orchestrator.transition_workflow_state(
            workflow_id=str(workflow_id),
            content_id=str(content_id),
            new_state=WorkflowState.CREATE,
            metadata={"reason": "test"}
        )
        
        # Verify database was updated
        assert mock_content_collection.update_one.called
        assert mock_workflow_collection.update_one.called


@pytest.mark.asyncio
async def test_monitor_workflow_progress(orchestrator):
    """Test workflow progress monitoring."""
    from bson import ObjectId
    from app.models.base import JobStatus
    
    workflow_id = ObjectId()
    job_id_1 = ObjectId()
    job_id_2 = ObjectId()
    job_id_3 = ObjectId()
    
    with patch('app.ai.orchestrator.get_database') as mock_db:
        # Mock workflow with jobs using ObjectIds
        mock_db.return_value.workflow_executions.find_one = AsyncMock(return_value={
            "_id": workflow_id,
            "workflow_name": "Test Workflow",
            "user_id": "user_123",
            "job_ids": [str(job_id_1), str(job_id_2), str(job_id_3)],
            "status": "running",
            "created_at": datetime.utcnow()
        })
        
        # Mock job statuses with all required fields using JobStatus enum values
        job_statuses = [
            {
                "_id": job_id_1,
                "user_id": "user_123",
                "job_type": "content_generation",
                "engine": "text_intelligence",
                "operation": "generate",
                "status": JobStatus.COMPLETED
            },
            {
                "_id": job_id_2,
                "user_id": "user_123",
                "job_type": "social_media_optimization",
                "engine": "social_media_planner",
                "operation": "optimize",
                "status": JobStatus.RUNNING
            },
            {
                "_id": job_id_3,
                "user_id": "user_123",
                "job_type": "analytics_processing",
                "engine": "discovery_analytics",
                "operation": "analyze",
                "status": JobStatus.QUEUED
            }
        ]
        
        async def mock_find_one(query):
            # The query will have _id as a string, so we need to compare strings
            job_id_str = query["_id"]
            for job in job_statuses:
                if str(job["_id"]) == job_id_str:
                    return job
            return None
        
        mock_db.return_value.async_jobs.find_one = mock_find_one
        
        # Monitor progress
        progress = await orchestrator.monitor_workflow_progress(str(workflow_id))
        
        assert progress is not None
        assert progress["workflow_id"] == str(workflow_id)
        assert progress["jobs"]["total"] == 3
        assert progress["jobs"]["completed"] == 1
        assert progress["jobs"]["running"] == 1
        assert progress["jobs"]["queued"] == 1
        assert 0 <= progress["progress_percentage"] <= 100


@pytest.mark.asyncio
async def test_get_workflow_analytics(orchestrator):
    """Test workflow analytics retrieval."""
    from bson import ObjectId
    
    workflow_id = ObjectId()
    job_id_1 = ObjectId()
    job_id_2 = ObjectId()
    
    with patch('app.ai.orchestrator.get_database') as mock_db:
        # Mock workflow
        created_at = datetime.utcnow()
        completed_at = datetime.utcnow()
        
        mock_db.return_value.workflow_executions.find_one = AsyncMock(return_value={
            "_id": workflow_id,
            "workflow_name": "Test Workflow",
            "user_id": "user_123",
            "job_ids": [str(job_id_1), str(job_id_2)],
            "status": "completed",
            "created_at": created_at,
            "completed_at": completed_at
        })
        
        # Mock jobs with analytics and all required fields
        mock_db.return_value.async_jobs.find_one = AsyncMock(return_value={
            "_id": job_id_1,
            "user_id": "user_123",
            "job_type": "content_generation",
            "engine": "text_intelligence",
            "operation": "generate",
            "status": "completed",
            "started_at": created_at,
            "completed_at": completed_at,
            "result": {"success": True, "cost": 0.05, "tokens_used": 100},
            "retry_count": 0
        })
        
        # Get analytics
        analytics = await orchestrator.get_workflow_analytics(str(workflow_id))
        
        assert analytics is not None
        assert analytics["workflow_id"] == str(workflow_id)
        assert "total_cost" in analytics
        assert "total_tokens" in analytics
        assert "engine_usage" in analytics
        assert "job_analytics" in analytics


@pytest.mark.asyncio
async def test_engine_error_handling(orchestrator):
    """Test that engine errors are handled properly."""
    # Mock engine to raise error
    text_engine = orchestrator.get_engine(EngineType.TEXT_INTELLIGENCE)
    original_method = text_engine.generate_content
    text_engine.generate_content = AsyncMock(side_effect=Exception("Test error"))
    
    # Execute operation and expect error
    with pytest.raises(EngineError) as exc_info:
        await orchestrator.execute_engine_operation(
            engine_type=EngineType.TEXT_INTELLIGENCE,
            operation="generate",
            parameters={
                "prompt": "test",
                "content_type": "blog",  # Add required parameter
                "target_length": 500
            }
        )
    
    # Check that an EngineError was raised (message may vary based on validation)
    assert exc_info.value is not None
    assert isinstance(exc_info.value, EngineError)
    
    # Restore original method
    text_engine.generate_content = original_method
    
    # Check that error rate was updated
    capability = orchestrator.engine_capabilities[EngineType.TEXT_INTELLIGENCE]
    assert capability.error_rate > 0


@pytest.mark.asyncio
async def test_system_health_tracking(orchestrator):
    """Test that system health is tracked correctly."""
    # Get initial health
    health = await orchestrator.get_system_health()
    
    assert health is not None
    assert health.status in ["healthy", "degraded", "unhealthy"]
    assert "engine_availability_ratio" in health.metrics
    assert "active_workflows" in health.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

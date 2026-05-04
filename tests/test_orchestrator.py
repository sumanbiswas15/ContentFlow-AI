"""
Tests for AI Orchestrator functionality.

This module contains unit tests for the AI orchestration layer,
including workflow planning, task routing, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.ai.orchestrator import (
    AIOrchestrator, WorkflowRequest, WorkflowResponse, TaskPriority
)
from app.models.base import EngineType, WorkflowState
from app.core.exceptions import ValidationError, OrchestrationError


class TestAIOrchestrator:
    """Test cases for AI Orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        with patch('app.ai.orchestrator.genai') as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel.return_value = MagicMock()
            
            orchestrator = AIOrchestrator()
            return orchestrator
    
    @pytest.fixture
    def sample_workflow_request(self):
        """Create sample workflow request for testing."""
        return WorkflowRequest(
            operation="generate_blog_post",
            parameters={"topic": "AI in content creation", "length": 1000},
            user_id="test_user_123",
            priority=TaskPriority.NORMAL
        )
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator is not None
        assert len(orchestrator.engine_capabilities) > 0
        assert orchestrator.system_health.status == "healthy"
    
    def test_engine_capabilities_loaded(self, orchestrator):
        """Test that engine capabilities are properly loaded."""
        expected_engines = [
            EngineType.TEXT_INTELLIGENCE,
            EngineType.CREATIVE_ASSISTANT,
            EngineType.SOCIAL_MEDIA_PLANNER,
            EngineType.DISCOVERY_ANALYTICS,
            EngineType.IMAGE_GENERATION,
            EngineType.AUDIO_GENERATION,
            EngineType.VIDEO_PIPELINE
        ]
        
        for engine in expected_engines:
            assert engine in orchestrator.engine_capabilities
            capability = orchestrator.engine_capabilities[engine]
            assert len(capability.operations) > 0
            assert capability.cost_per_operation >= 0
            assert capability.average_processing_time > 0
    
    @pytest.mark.asyncio
    async def test_validate_workflow_request_valid(self, orchestrator, sample_workflow_request):
        """Test validation of valid workflow request."""
        with patch('app.ai.orchestrator.get_database') as mock_db:
            mock_db.return_value.content_items.find_one = AsyncMock(return_value=None)
            
            result = await orchestrator._validate_workflow_request(sample_workflow_request)
            assert result.is_valid
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_workflow_request_missing_operation(self, orchestrator):
        """Test validation fails for missing operation."""
        invalid_request = WorkflowRequest(
            operation="",  # Empty operation
            user_id="test_user_123"
        )
        
        with patch('app.ai.orchestrator.get_database') as mock_db:
            mock_db.return_value.content_items.find_one = AsyncMock(return_value=None)
            
            result = await orchestrator._validate_workflow_request(invalid_request)
            assert not result.is_valid
            assert "Operation is required" in result.errors
    
    @pytest.mark.asyncio
    async def test_validate_workflow_request_missing_user_id(self, orchestrator):
        """Test validation fails for missing user ID."""
        invalid_request = WorkflowRequest(
            operation="test_operation",
            user_id=""  # Empty user ID
        )
        
        with patch('app.ai.orchestrator.get_database') as mock_db:
            mock_db.return_value.content_items.find_one = AsyncMock(return_value=None)
            
            result = await orchestrator._validate_workflow_request(invalid_request)
            assert not result.is_valid
            assert "User ID is required" in result.errors
    
    def test_fallback_route_task_text_generation(self, orchestrator):
        """Test fallback routing for text generation tasks."""
        task = {"operation": "generate_content", "parameters": {}}
        
        engine = orchestrator._fallback_route_task(task)
        assert engine == EngineType.TEXT_INTELLIGENCE
    
    def test_fallback_route_task_creative_assistance(self, orchestrator):
        """Test fallback routing for creative assistance tasks."""
        task = {"operation": "suggest_ideas", "parameters": {}}
        
        engine = orchestrator._fallback_route_task(task)
        assert engine == EngineType.CREATIVE_ASSISTANT
    
    def test_fallback_route_task_social_media(self, orchestrator):
        """Test fallback routing for social media tasks."""
        task = {"operation": "optimize_for_twitter", "parameters": {}}
        
        engine = orchestrator._fallback_route_task(task)
        assert engine == EngineType.SOCIAL_MEDIA_PLANNER
    
    def test_fallback_route_task_analytics(self, orchestrator):
        """Test fallback routing for analytics tasks."""
        task = {"operation": "analyze_engagement", "parameters": {}}
        
        engine = orchestrator._fallback_route_task(task)
        assert engine == EngineType.DISCOVERY_ANALYTICS
    
    def test_fallback_route_task_default(self, orchestrator):
        """Test fallback routing defaults to text intelligence."""
        task = {"operation": "unknown_operation", "parameters": {}}
        
        engine = orchestrator._fallback_route_task(task)
        assert engine == EngineType.TEXT_INTELLIGENCE
    
    def test_is_valid_state_transition_valid(self, orchestrator):
        """Test valid workflow state transitions."""
        valid_transitions = [
            (WorkflowState.DISCOVER, WorkflowState.CREATE),
            (WorkflowState.CREATE, WorkflowState.TRANSFORM),
            (WorkflowState.CREATE, WorkflowState.PLAN),
            (WorkflowState.TRANSFORM, WorkflowState.PLAN),
            (WorkflowState.PLAN, WorkflowState.PUBLISH),
            (WorkflowState.PUBLISH, WorkflowState.ANALYZE),
            (WorkflowState.ANALYZE, WorkflowState.IMPROVE),
            (WorkflowState.IMPROVE, WorkflowState.CREATE)
        ]
        
        for current, new in valid_transitions:
            assert orchestrator._is_valid_state_transition(current, new)
    
    def test_is_valid_state_transition_invalid(self, orchestrator):
        """Test invalid workflow state transitions."""
        invalid_transitions = [
            (WorkflowState.CREATE, WorkflowState.DISCOVER),
            (WorkflowState.PUBLISH, WorkflowState.CREATE),
            (WorkflowState.ANALYZE, WorkflowState.PLAN),
            (WorkflowState.IMPROVE, WorkflowState.PUBLISH)
        ]
        
        for current, new in invalid_transitions:
            assert not orchestrator._is_valid_state_transition(current, new)
    
    def test_classify_error_timeout(self, orchestrator):
        """Test error classification for timeout errors."""
        from app.models.base import ErrorCode
        
        timeout_error = Exception("Request timeout occurred")
        error_code = orchestrator._classify_error(timeout_error)
        assert error_code == ErrorCode.TIMEOUT
    
    def test_classify_error_rate_limit(self, orchestrator):
        """Test error classification for rate limit errors."""
        from app.models.base import ErrorCode
        
        rate_limit_error = Exception("Rate limit exceeded")
        error_code = orchestrator._classify_error(rate_limit_error)
        assert error_code == ErrorCode.RATE_LIMIT_EXCEEDED
    
    def test_classify_error_validation(self, orchestrator):
        """Test error classification for validation errors."""
        from app.models.base import ErrorCode
        
        validation_error = Exception("Validation failed")
        error_code = orchestrator._classify_error(validation_error)
        assert error_code == ErrorCode.VALIDATION_ERROR
    
    def test_classify_error_default(self, orchestrator):
        """Test error classification defaults to engine error."""
        from app.models.base import ErrorCode
        
        unknown_error = Exception("Unknown error occurred")
        error_code = orchestrator._classify_error(unknown_error)
        assert error_code == ErrorCode.ENGINE_ERROR
    
    @pytest.mark.asyncio
    async def test_fallback_workflow_plan(self, orchestrator, sample_workflow_request):
        """Test fallback workflow planning when LLM is unavailable."""
        workflow_plan = await orchestrator._fallback_workflow_plan(sample_workflow_request)
        
        assert "tasks" in workflow_plan
        assert "total_estimated_time" in workflow_plan
        assert "total_estimated_cost" in workflow_plan
        assert "workflow_description" in workflow_plan
        
        assert len(workflow_plan["tasks"]) == 1
        task = workflow_plan["tasks"][0]
        assert "id" in task
        assert "engine" in task
        assert "operation" in task
        assert "parameters" in task
        assert "estimated_time" in task
        assert "estimated_cost" in task
    
    def test_map_engine_to_job_type(self, orchestrator):
        """Test mapping of engine names to job types."""
        from app.models.base import JobType
        
        mappings = [
            ("text_intelligence", JobType.CONTENT_GENERATION),
            ("creative_assistant", JobType.CREATIVE_ASSISTANCE),
            ("social_media_planner", JobType.SOCIAL_MEDIA_OPTIMIZATION),
            ("discovery_analytics", JobType.ANALYTICS_PROCESSING),
            ("image_generation", JobType.MEDIA_GENERATION),
            ("audio_generation", JobType.MEDIA_GENERATION),
            ("video_pipeline", JobType.MEDIA_GENERATION)
        ]
        
        for engine, expected_job_type in mappings:
            job_type = orchestrator._map_engine_to_job_type(engine)
            assert job_type == expected_job_type
    
    def test_map_engine_to_job_type_unknown(self, orchestrator):
        """Test mapping unknown engine defaults to content generation."""
        from app.models.base import JobType
        
        job_type = orchestrator._map_engine_to_job_type("unknown_engine")
        assert job_type == JobType.CONTENT_GENERATION
    
    @pytest.mark.asyncio
    async def test_determine_recovery_strategy(self, orchestrator):
        """Test recovery strategy determination for different error types."""
        from app.models.base import ErrorCode
        
        strategies = [
            (ErrorCode.TIMEOUT, "retry_with_backoff"),
            (ErrorCode.RATE_LIMIT_EXCEEDED, "delay_and_retry"),
            (ErrorCode.USAGE_LIMIT_EXCEEDED, "queue_for_later"),
            (ErrorCode.NETWORK_ERROR, "retry_with_backoff"),
            (ErrorCode.SERVICE_UNAVAILABLE, "fallback_engine"),
            (ErrorCode.ENGINE_ERROR, "fallback_engine"),
            (ErrorCode.VALIDATION_ERROR, "fix_and_retry"),
            (ErrorCode.AUTHENTICATION_ERROR, "refresh_credentials"),
            (ErrorCode.AUTHORIZATION_ERROR, "escalate_permissions")
        ]
        
        for error_code, expected_strategy in strategies:
            strategy = await orchestrator._determine_recovery_strategy("test_engine", error_code)
            assert strategy == expected_strategy
    
    @pytest.mark.asyncio
    async def test_find_fallback_engine(self, orchestrator):
        """Test finding fallback engines."""
        fallbacks = [
            ("text_intelligence", "creative_assistant"),
            ("creative_assistant", "text_intelligence"),
            ("image_generation", None),
            ("audio_generation", None),
            ("video_pipeline", None)
        ]
        
        for engine, expected_fallback in fallbacks:
            fallback = await orchestrator._find_fallback_engine(engine)
            assert fallback == expected_fallback
    
    @pytest.mark.asyncio
    async def test_get_system_health(self, orchestrator):
        """Test system health reporting."""
        health = await orchestrator.get_system_health()
        
        assert health.status in ["healthy", "degraded", "unhealthy"]
        assert "engine_availability_ratio" in health.metrics
        assert "active_workflows" in health.metrics
        assert isinstance(health.services, dict)
    
    @pytest.mark.asyncio
    async def test_orchestrate_workflow_validation_error(self, orchestrator):
        """Test workflow orchestration with validation error."""
        invalid_request = WorkflowRequest(
            operation="",  # Invalid empty operation
            user_id="test_user"
        )
        
        with patch('app.ai.orchestrator.get_database') as mock_db:
            mock_db.return_value.content_items.find_one = AsyncMock(return_value=None)
            
            with pytest.raises(ValidationError):
                await orchestrator.orchestrate_workflow(invalid_request)


class TestWorkflowModels:
    """Test cases for workflow-related models."""
    
    def test_workflow_request_creation(self):
        """Test WorkflowRequest model creation."""
        request = WorkflowRequest(
            operation="test_operation",
            parameters={"key": "value"},
            user_id="test_user",
            priority=TaskPriority.HIGH
        )
        
        assert request.operation == "test_operation"
        assert request.parameters == {"key": "value"}
        assert request.user_id == "test_user"
        assert request.priority == TaskPriority.HIGH
        assert request.timeout_seconds == 300  # default
    
    def test_workflow_response_creation(self):
        """Test WorkflowResponse model creation."""
        response = WorkflowResponse(
            workflow_id="test_workflow_123",
            status="queued",
            job_ids=["job1", "job2"],
            cost_estimate=0.05,
            message="Workflow created successfully"
        )
        
        assert response.workflow_id == "test_workflow_123"
        assert response.status == "queued"
        assert response.job_ids == ["job1", "job2"]
        assert response.cost_estimate == 0.05
        assert response.message == "Workflow created successfully"
    
    def test_task_priority_enum(self):
        """Test TaskPriority enum values."""
        assert TaskPriority.CRITICAL == 1
        assert TaskPriority.HIGH == 2
        assert TaskPriority.NORMAL == 3
        assert TaskPriority.LOW == 4
        assert TaskPriority.BACKGROUND == 5
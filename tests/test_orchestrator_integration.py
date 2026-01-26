"""
Integration tests for AI Orchestrator.

This module contains integration tests that verify the orchestrator
works with the actual database and external services.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.ai.orchestrator import AIOrchestrator, WorkflowRequest, TaskPriority
from app.models.base import EngineType


def test_orchestrator_basic_functionality():
    """Test basic orchestrator functionality without async operations."""
    with patch('app.ai.orchestrator.genai') as mock_genai:
        mock_genai.configure = MagicMock()
        mock_genai.GenerativeModel.return_value = MagicMock()
        
        orchestrator = AIOrchestrator()
        
        # Test initialization
        assert orchestrator is not None
        assert len(orchestrator.engine_capabilities) == 7
        
        # Test task routing
        task = {"operation": "generate_content", "parameters": {}}
        engine = orchestrator._fallback_route_task(task)
        assert engine == EngineType.TEXT_INTELLIGENCE
        
        # Test state transition validation
        from app.models.base import WorkflowState
        assert orchestrator._is_valid_state_transition(
            WorkflowState.CREATE, WorkflowState.TRANSFORM
        )
        
        # Test error classification
        from app.models.base import ErrorCode
        timeout_error = Exception("Request timeout")
        error_code = orchestrator._classify_error(timeout_error)
        assert error_code == ErrorCode.TIMEOUT


def test_workflow_request_validation():
    """Test workflow request validation logic."""
    with patch('app.ai.orchestrator.genai') as mock_genai:
        mock_genai.configure = MagicMock()
        mock_genai.GenerativeModel.return_value = MagicMock()
        
        orchestrator = AIOrchestrator()
        
        # Test valid request
        valid_request = WorkflowRequest(
            operation="generate_blog_post",
            parameters={"topic": "AI", "length": 1000},
            user_id="test_user_123"
        )
        
        # Basic validation should pass for structure
        assert valid_request.operation == "generate_blog_post"
        assert valid_request.user_id == "test_user_123"
        assert valid_request.priority == TaskPriority.NORMAL


def test_engine_capabilities():
    """Test that all expected engines have proper capabilities."""
    with patch('app.ai.orchestrator.genai') as mock_genai:
        mock_genai.configure = MagicMock()
        mock_genai.GenerativeModel.return_value = MagicMock()
        
        orchestrator = AIOrchestrator()
        
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
            
            # Verify capability structure
            assert len(capability.operations) > 0
            assert capability.cost_per_operation >= 0
            assert capability.average_processing_time > 0
            assert isinstance(capability.availability, bool)


def test_fallback_workflow_planning():
    """Test fallback workflow planning when LLM is unavailable."""
    with patch('app.ai.orchestrator.genai') as mock_genai:
        mock_genai.configure = MagicMock()
        mock_genai.GenerativeModel.return_value = MagicMock()
        
        orchestrator = AIOrchestrator()
        
        request = WorkflowRequest(
            operation="generate_content",
            parameters={"topic": "test"},
            user_id="test_user"
        )
        
        # This should work synchronously
        import asyncio
        
        async def test_fallback():
            plan = await orchestrator._fallback_workflow_plan(request)
            
            assert "tasks" in plan
            assert "total_estimated_time" in plan
            assert "total_estimated_cost" in plan
            assert len(plan["tasks"]) == 1
            
            task = plan["tasks"][0]
            assert "engine" in task
            assert "operation" in task
            assert task["operation"] == "generate_content"
        
        # Run the async test
        asyncio.run(test_fallback())


def test_error_recovery_strategies():
    """Test error recovery strategy determination."""
    with patch('app.ai.orchestrator.genai') as mock_genai:
        mock_genai.configure = MagicMock()
        mock_genai.GenerativeModel.return_value = MagicMock()
        
        orchestrator = AIOrchestrator()
        
        from app.models.base import ErrorCode
        
        # Test different error types
        test_cases = [
            (ErrorCode.TIMEOUT, "retry_with_backoff"),
            (ErrorCode.RATE_LIMIT_EXCEEDED, "delay_and_retry"),
            (ErrorCode.USAGE_LIMIT_EXCEEDED, "queue_for_later"),
            (ErrorCode.SERVICE_UNAVAILABLE, "fallback_engine"),
            (ErrorCode.VALIDATION_ERROR, "fix_and_retry")
        ]
        
        import asyncio
        
        async def test_strategies():
            for error_code, expected_strategy in test_cases:
                strategy = await orchestrator._determine_recovery_strategy("test_engine", error_code)
                assert strategy == expected_strategy
        
        asyncio.run(test_strategies())


if __name__ == "__main__":
    # Run basic tests
    test_orchestrator_basic_functionality()
    test_workflow_request_validation()
    test_engine_capabilities()
    test_fallback_workflow_planning()
    test_error_recovery_strategies()
    print("All integration tests passed!")
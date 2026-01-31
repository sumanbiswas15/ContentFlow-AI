"""
Unit tests for cost tracking and usage management service.

This module tests token usage tracking, cost calculation, usage limit
monitoring, warnings, cap enforcement, and detailed analytics.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.cost_tracking import (
    CostCalculator,
    UsageTracker,
    UsageLimitMonitor,
    CostTrackingService,
    cost_tracking_service
)
from app.models.users import User, UsageLimits, UsageStats
from app.models.base import EngineType


class TestCostCalculator:
    """Test cases for CostCalculator."""
    
    def test_calculate_cost_gemini_pro(self):
        """Test cost calculation for Gemini Pro model."""
        calculator = CostCalculator()
        
        # Test with 1000 input tokens and 500 output tokens
        cost = calculator.calculate_cost(1000, 500, "gemini-pro")
        
        # Expected: (1000/1000 * 0.00025) + (500/1000 * 0.0005) = 0.00025 + 0.00025 = 0.0005
        assert cost == 0.0005
    
    def test_calculate_cost_gpt4(self):
        """Test cost calculation for GPT-4 model."""
        calculator = CostCalculator()
        
        # Test with 1000 input tokens and 1000 output tokens
        cost = calculator.calculate_cost(1000, 1000, "gpt-4")
        
        # Expected: (1000/1000 * 0.03) + (1000/1000 * 0.06) = 0.03 + 0.06 = 0.09
        assert cost == 0.09
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation with unknown model defaults to gemini-pro."""
        calculator = CostCalculator()
        
        cost = calculator.calculate_cost(1000, 500, "unknown-model")
        
        # Should use gemini-pro pricing
        assert cost == 0.0005
    
    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        calculator = CostCalculator()
        
        cost = calculator.calculate_cost(0, 0, "gemini-pro")
        
        assert cost == 0.0
    
    def test_calculate_cost_large_numbers(self):
        """Test cost calculation with large token counts."""
        calculator = CostCalculator()
        
        # 1 million input tokens, 500k output tokens
        cost = calculator.calculate_cost(1_000_000, 500_000, "gemini-pro")
        
        # Expected: (1000000/1000 * 0.00025) + (500000/1000 * 0.0005) = 0.25 + 0.25 = 0.5
        assert cost == 0.5
    
    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        calculator = CostCalculator()
        
        text = "Hello world"
        tokens = calculator.estimate_tokens(text)
        
        # "Hello world" is 11 characters, ~2-3 tokens
        assert tokens >= 1
        assert tokens <= 5
    
    def test_estimate_tokens_long_text(self):
        """Test token estimation for long text."""
        calculator = CostCalculator()
        
        text = "This is a longer piece of text that should result in more tokens. " * 10
        tokens = calculator.estimate_tokens(text)
        
        # Should be proportional to text length
        assert tokens > 100
    
    def test_estimate_tokens_empty_text(self):
        """Test token estimation for empty text."""
        calculator = CostCalculator()
        
        tokens = calculator.estimate_tokens("")
        
        # Should return at least 1 token
        assert tokens == 1
    
    def test_get_model_for_engine(self):
        """Test getting default model for engine types."""
        calculator = CostCalculator()
        
        assert calculator.get_model_for_engine(EngineType.TEXT_INTELLIGENCE) == "gemini-pro"
        assert calculator.get_model_for_engine(EngineType.IMAGE_GENERATION) == "gemini-pro-vision"
        assert calculator.get_model_for_engine(EngineType.CREATIVE_ASSISTANT) == "gemini-pro"


class TestUsageTracker:
    """Test cases for UsageTracker."""
    
    @pytest.fixture
    def mock_database(self):
        """Create mock database."""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock()
        mock_collection.aggregate = MagicMock()
        mock_db.cost_tracking = mock_collection
        return mock_db
    
    @pytest.mark.asyncio
    async def test_record_usage(self, mock_database):
        """Test recording usage."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            tracker = UsageTracker()
            
            result = await tracker.record_usage(
                user_id="user123",
                operation_type="content_generation",
                engine_type=EngineType.TEXT_INTELLIGENCE,
                input_tokens=1000,
                output_tokens=500,
                model="gemini-pro"
            )
            
            # Verify record structure
            assert result["user_id"] == "user123"
            assert result["operation_type"] == "content_generation"
            assert result["engine_type"] == "text_intelligence"
            assert result["input_tokens"] == 1000
            assert result["output_tokens"] == 500
            assert result["total_tokens"] == 1500
            assert result["cost"] == 0.0005  # (1000/1000 * 0.00025) + (500/1000 * 0.0005) = 0.00025 + 0.00025 = 0.0005
            assert "timestamp" in result
            assert "date" in result
            
            # Verify database insert was called
            mock_database.cost_tracking.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_record_usage_with_metadata(self, mock_database):
        """Test recording usage with metadata."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            tracker = UsageTracker()
            
            metadata = {"content_id": "content123", "workflow": "create"}
            
            result = await tracker.record_usage(
                user_id="user123",
                operation_type="transformation",
                engine_type=EngineType.TEXT_INTELLIGENCE,
                input_tokens=500,
                output_tokens=300,
                model="gemini-pro",
                metadata=metadata
            )
            
            assert result["metadata"] == metadata
    
    @pytest.mark.asyncio
    async def test_get_user_usage_today_no_data(self, mock_database):
        """Test getting today's usage when no data exists."""
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_database.cost_tracking.aggregate.return_value = mock_cursor
        
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            tracker = UsageTracker()
            
            result = await tracker.get_user_usage_today("user123")
            
            assert result["total_tokens"] == 0
            assert result["total_cost"] == 0.0
            assert result["operation_count"] == 0
    
    @pytest.mark.asyncio
    async def test_get_user_usage_today_with_data(self, mock_database):
        """Test getting today's usage with data."""
        mock_data = [{
            "_id": None,
            "total_tokens": 5000,
            "total_cost": 0.025,
            "operation_count": 10,
            "operations_by_type": []
        }]
        
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_data)
        mock_database.cost_tracking.aggregate.return_value = mock_cursor
        
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            tracker = UsageTracker()
            
            result = await tracker.get_user_usage_today("user123")
            
            assert result["total_tokens"] == 5000
            assert result["total_cost"] == 0.025
            assert result["operation_count"] == 10
    
    @pytest.mark.asyncio
    async def test_get_user_usage_this_month(self, mock_database):
        """Test getting this month's usage."""
        mock_data = [{
            "_id": None,
            "total_tokens": 50000,
            "total_cost": 0.25,
            "operation_count": 100,
            "by_engine": []
        }]
        
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_data)
        mock_database.cost_tracking.aggregate.return_value = mock_cursor
        
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            tracker = UsageTracker()
            
            result = await tracker.get_user_usage_this_month("user123")
            
            assert result["total_tokens"] == 50000
            assert result["total_cost"] == 0.25
            assert result["operation_count"] == 100
    
    @pytest.mark.asyncio
    async def test_get_usage_analytics(self, mock_database):
        """Test getting detailed usage analytics."""
        # Mock overall stats
        overall_data = [{
            "_id": None,
            "total_tokens": 100000,
            "total_cost": 0.5,
            "operation_count": 200,
            "avg_tokens_per_operation": 500.0,
            "avg_cost_per_operation": 0.0025
        }]
        
        # Mock engine breakdown
        engine_data = [
            {"_id": "text_intelligence", "tokens": 60000, "cost": 0.3, "count": 120},
            {"_id": "creative_assistant", "tokens": 40000, "cost": 0.2, "count": 80}
        ]
        
        # Mock operation breakdown
        operation_data = [
            {"_id": "generation", "tokens": 70000, "cost": 0.35, "count": 140},
            {"_id": "transformation", "tokens": 30000, "cost": 0.15, "count": 60}
        ]
        
        # Mock daily trend
        daily_data = [
            {"_id": "2024-01-01", "tokens": 50000, "cost": 0.25, "count": 100},
            {"_id": "2024-01-02", "tokens": 50000, "cost": 0.25, "count": 100}
        ]
        
        # Setup mock to return different data for different pipelines
        call_count = 0
        async def mock_to_list(length):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return overall_data
            elif call_count == 2:
                return engine_data
            elif call_count == 3:
                return operation_data
            else:
                return daily_data
        
        mock_cursor = MagicMock()
        mock_cursor.to_list = mock_to_list
        mock_database.cost_tracking.aggregate.return_value = mock_cursor
        
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            tracker = UsageTracker()
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 2)
            
            result = await tracker.get_usage_analytics("user123", start_date, end_date)
            
            assert "period" in result
            assert "overall" in result
            assert "by_engine" in result
            assert "by_operation" in result
            assert "daily_trend" in result
            
            assert result["overall"]["total_tokens"] == 100000
            assert len(result["by_engine"]) == 2
            assert len(result["by_operation"]) == 2
            assert len(result["daily_trend"]) == 2


class TestUsageLimitMonitor:
    """Test cases for UsageLimitMonitor."""
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user with usage limits."""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed"
        )
        user.usage_limits = UsageLimits(
            daily_token_limit=10000,
            monthly_cost_limit=10.0
        )
        return user
    
    @pytest.fixture
    def mock_tracker(self):
        """Create mock usage tracker."""
        tracker = MagicMock()
        tracker.get_user_usage_today = AsyncMock(return_value={
            "total_tokens": 5000,
            "total_cost": 0.025,
            "operation_count": 10
        })
        tracker.get_user_usage_this_month = AsyncMock(return_value={
            "total_tokens": 50000,
            "total_cost": 5.0,
            "operation_count": 100
        })
        return tracker
    
    @pytest.mark.asyncio
    async def test_check_limits_within_limits(self, mock_user, mock_tracker):
        """Test checking limits when within limits."""
        monitor = UsageLimitMonitor()
        
        result = await monitor.check_limits(mock_user, mock_tracker)
        
        assert "limits" in result
        assert "warnings" in result
        assert "any_exceeded" in result
        
        # Should be within limits
        assert not result["limits"]["daily_tokens"]["exceeded"]
        assert not result["limits"]["monthly_cost"]["exceeded"]
        assert not result["any_exceeded"]
        
        # Check percentages
        assert result["limits"]["daily_tokens"]["percentage"] == 50.0  # 5000/10000
        assert result["limits"]["monthly_cost"]["percentage"] == 50.0  # 5.0/10.0
    
    @pytest.mark.asyncio
    async def test_check_limits_approaching_limit(self, mock_user, mock_tracker):
        """Test checking limits when approaching limit (75% threshold)."""
        # Set usage to 75% of limit
        mock_tracker.get_user_usage_today = AsyncMock(return_value={
            "total_tokens": 7500,
            "total_cost": 0.0375,
            "operation_count": 15
        })
        
        monitor = UsageLimitMonitor()
        
        result = await monitor.check_limits(mock_user, mock_tracker)
        
        # Should have warning
        assert len(result["warnings"]) > 0
        assert any(w["type"] == "daily_tokens" for w in result["warnings"])
    
    @pytest.mark.asyncio
    async def test_check_limits_exceeded(self, mock_user, mock_tracker):
        """Test checking limits when limit is exceeded."""
        # Set usage to exceed limit
        mock_tracker.get_user_usage_today = AsyncMock(return_value={
            "total_tokens": 11000,
            "total_cost": 0.055,
            "operation_count": 22
        })
        
        monitor = UsageLimitMonitor()
        
        result = await monitor.check_limits(mock_user, mock_tracker)
        
        # Should be exceeded
        assert result["limits"]["daily_tokens"]["exceeded"]
        assert result["any_exceeded"]
    
    @pytest.mark.asyncio
    async def test_check_limits_multiple_warnings(self, mock_user, mock_tracker):
        """Test that multiple warning thresholds generate warnings."""
        # Set usage to 95% of limit
        mock_tracker.get_user_usage_today = AsyncMock(return_value={
            "total_tokens": 9500,
            "total_cost": 0.0475,
            "operation_count": 19
        })
        
        monitor = UsageLimitMonitor()
        
        result = await monitor.check_limits(mock_user, mock_tracker)
        
        # Should have multiple warnings (50%, 75%, 90%, 95%)
        daily_warnings = [w for w in result["warnings"] if w["type"] == "daily_tokens"]
        assert len(daily_warnings) >= 1
        
        # Should have critical severity at 95%
        assert any(w["severity"] == "critical" for w in daily_warnings)
    
    @pytest.mark.asyncio
    async def test_can_perform_operation_allowed(self, mock_user, mock_tracker):
        """Test checking if operation can be performed when allowed."""
        monitor = UsageLimitMonitor()
        
        can_perform, reason = await monitor.can_perform_operation(
            mock_user,
            1000,  # Estimated tokens
            mock_tracker
        )
        
        assert can_perform is True
        assert reason is None
    
    @pytest.mark.asyncio
    async def test_can_perform_operation_exceeds_daily_limit(self, mock_user, mock_tracker):
        """Test checking if operation would exceed daily limit."""
        monitor = UsageLimitMonitor()
        
        can_perform, reason = await monitor.can_perform_operation(
            mock_user,
            6000,  # Would exceed 10000 limit (5000 + 6000 = 11000)
            mock_tracker
        )
        
        assert can_perform is False
        assert reason is not None
        assert "Daily token limit" in reason
    
    @pytest.mark.asyncio
    async def test_can_perform_operation_exceeds_monthly_cost(self, mock_user, mock_tracker):
        """Test checking if operation would exceed monthly cost limit."""
        # Increase daily token limit so it doesn't trigger first
        mock_user.usage_limits.daily_token_limit = 200000
        
        # Set monthly usage close to limit, but daily usage low
        mock_tracker.get_user_usage_today = AsyncMock(return_value={
            "total_tokens": 1000,  # Low daily usage
            "total_cost": 0.005,
            "operation_count": 2
        })
        mock_tracker.get_user_usage_this_month = AsyncMock(return_value={
            "total_tokens": 95000,
            "total_cost": 9.99,  # Very close to $10 limit
            "operation_count": 190
        })
        
        monitor = UsageLimitMonitor()
        
        # Use a very large operation that would definitely exceed cost limit
        # 100,000 tokens would cost approximately $0.025 with gemini-pro
        can_perform, reason = await monitor.can_perform_operation(
            mock_user,
            100000,  # Large operation that would exceed cost limit
            mock_tracker
        )
        
        assert can_perform is False
        assert reason is not None
        assert "Monthly cost limit" in reason
    
    @pytest.mark.asyncio
    async def test_reset_warnings(self, mock_user):
        """Test resetting warning flags."""
        monitor = UsageLimitMonitor()
        
        # Add some warnings
        monitor._warned_users["user123"].add("daily_tokens_50")
        monitor._warned_users["user123"].add("monthly_cost_75")
        
        await monitor.reset_warnings("user123")
        
        assert len(monitor._warned_users["user123"]) == 0


class TestCostTrackingService:
    """Test cases for CostTrackingService."""
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed"
        )
        user.usage_limits = UsageLimits(
            daily_token_limit=10000,
            monthly_cost_limit=10.0
        )
        return user
    
    @pytest.fixture
    def mock_database(self):
        """Create mock database."""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock()
        mock_collection.aggregate = MagicMock()
        
        # Setup default aggregate responses
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{
            "_id": None,
            "total_tokens": 5000,
            "total_cost": 0.025,
            "operation_count": 10,
            "operations_by_type": []
        }])
        mock_collection.aggregate.return_value = mock_cursor
        
        mock_db.cost_tracking = mock_collection
        return mock_db
    
    @pytest.mark.asyncio
    async def test_track_operation(self, mock_user, mock_database):
        """Test tracking an operation."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            result = await service.track_operation(
                user=mock_user,
                operation_type="content_generation",
                engine_type=EngineType.TEXT_INTELLIGENCE,
                input_tokens=1000,
                output_tokens=500
            )
            
            assert "user_id" in result
            assert "total_tokens" in result
            assert "cost" in result
            assert "limit_status" in result
            
            # Verify database insert was called
            mock_database.cost_tracking.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_track_operation_with_custom_model(self, mock_user, mock_database):
        """Test tracking operation with custom model."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            result = await service.track_operation(
                user=mock_user,
                operation_type="content_generation",
                engine_type=EngineType.TEXT_INTELLIGENCE,
                input_tokens=1000,
                output_tokens=500,
                model="gpt-4"
            )
            
            assert result["model"] == "gpt-4"
            # GPT-4 is more expensive
            assert result["cost"] > 0.001
    
    @pytest.mark.asyncio
    async def test_check_can_perform_operation_allowed(self, mock_user, mock_database):
        """Test checking if operation can be performed."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            can_perform, reason, limit_status = await service.check_can_perform_operation(
                mock_user,
                "This is a short test text"
            )
            
            assert can_perform is True
            assert reason is None
            assert "limits" in limit_status
    
    @pytest.mark.asyncio
    async def test_check_can_perform_operation_denied(self, mock_user, mock_database):
        """Test checking operation when limit would be exceeded."""
        # Mock usage at limit
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{
            "_id": None,
            "total_tokens": 9900,
            "total_cost": 0.0495,
            "operation_count": 20,
            "operations_by_type": []
        }])
        mock_database.cost_tracking.aggregate.return_value = mock_cursor
        
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            # Large text that would exceed limit
            large_text = "This is a test. " * 1000
            
            can_perform, reason, limit_status = await service.check_can_perform_operation(
                mock_user,
                large_text
            )
            
            assert can_perform is False
            assert reason is not None
    
    @pytest.mark.asyncio
    async def test_get_usage_report_today(self, mock_user, mock_database):
        """Test getting today's usage report."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            report = await service.get_usage_report("user123", "today")
            
            assert "total_tokens" in report
            assert "total_cost" in report
            assert "operation_count" in report
    
    @pytest.mark.asyncio
    async def test_get_usage_report_month(self, mock_user, mock_database):
        """Test getting monthly usage report."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            report = await service.get_usage_report("user123", "month")
            
            assert "total_tokens" in report
            assert "total_cost" in report
    
    @pytest.mark.asyncio
    async def test_estimate_operation_cost(self, mock_database):
        """Test estimating operation cost."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            estimate = await service.estimate_operation_cost(
                "This is a test text for cost estimation",
                EngineType.TEXT_INTELLIGENCE
            )
            
            assert "estimated_input_tokens" in estimate
            assert "estimated_output_tokens" in estimate
            assert "estimated_total_tokens" in estimate
            assert "estimated_cost" in estimate
            assert "model" in estimate
            assert estimate["model"] == "gemini-pro"
    
    @pytest.mark.asyncio
    async def test_estimate_operation_cost_custom_model(self, mock_database):
        """Test estimating cost with custom model."""
        with patch('app.services.cost_tracking.get_database', return_value=mock_database):
            service = CostTrackingService()
            
            estimate = await service.estimate_operation_cost(
                "Test text",
                EngineType.TEXT_INTELLIGENCE,
                model="gpt-4"
            )
            
            assert estimate["model"] == "gpt-4"
            # GPT-4 should be more expensive
            assert estimate["estimated_cost"] > 0


class TestCostTrackingIntegration:
    """Integration tests for cost tracking service."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete cost tracking workflow."""
        # Create mock database
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock()
        
        # Setup aggregate responses
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{
            "_id": None,
            "total_tokens": 1000,
            "total_cost": 0.005,
            "operation_count": 2,
            "operations_by_type": []
        }])
        mock_collection.aggregate.return_value = mock_cursor
        mock_db.cost_tracking = mock_collection
        
        with patch('app.services.cost_tracking.get_database', return_value=mock_db):
            service = CostTrackingService()
            
            # Create user
            user = User(
                email="test@example.com",
                username="testuser",
                hashed_password="hashed"
            )
            
            # 1. Check if operation can be performed
            can_perform, reason, _ = await service.check_can_perform_operation(
                user,
                "Generate some content"
            )
            assert can_perform is True
            
            # 2. Track the operation
            result = await service.track_operation(
                user=user,
                operation_type="generation",
                engine_type=EngineType.TEXT_INTELLIGENCE,
                input_tokens=500,
                output_tokens=300
            )
            assert result["total_tokens"] == 800
            
            # 3. Get usage report
            report = await service.get_usage_report("user123", "today")
            assert "total_tokens" in report
    
    @pytest.mark.asyncio
    async def test_limit_enforcement(self):
        """Test that limits are properly enforced."""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock()
        
        # Setup usage at 95% of limit
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{
            "_id": None,
            "total_tokens": 9500,
            "total_cost": 0.0475,
            "operation_count": 19,
            "operations_by_type": []
        }])
        mock_collection.aggregate.return_value = mock_cursor
        mock_db.cost_tracking = mock_collection
        
        with patch('app.services.cost_tracking.get_database', return_value=mock_db):
            service = CostTrackingService()
            
            user = User(
                email="test@example.com",
                username="testuser",
                hashed_password="hashed"
            )
            user.usage_limits.daily_token_limit = 10000
            
            # Try to perform operation that would exceed limit
            large_text = "Test " * 1000
            can_perform, reason, limit_status = await service.check_can_perform_operation(
                user,
                large_text
            )
            
            # Should be denied
            assert can_perform is False
            assert reason is not None
            
            # Should have warnings
            assert len(limit_status["warnings"]) > 0


def test_global_service_instance():
    """Test that global service instance is available."""
    assert cost_tracking_service is not None
    assert isinstance(cost_tracking_service, CostTrackingService)

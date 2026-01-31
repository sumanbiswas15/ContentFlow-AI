"""
Cost tracking and usage management service for ContentFlow AI.

This module provides comprehensive token usage tracking, cost calculation,
usage limit monitoring, warnings, cap enforcement, and detailed analytics.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import asyncio

from app.core.config import settings
from app.core.database import get_database
from app.core.exceptions import ValidationError
from app.models.users import User, UsageLimits, UsageStats
from app.models.base import EngineType

logger = logging.getLogger(__name__)


class CostCalculator:
    """Calculate costs for different AI operations and models."""
    
    # Cost per 1000 tokens for different models (in USD)
    MODEL_COSTS = {
        "gemini-pro": {
            "input": 0.00025,  # $0.25 per 1M tokens
            "output": 0.0005   # $0.50 per 1M tokens
        },
        "gemini-pro-vision": {
            "input": 0.00025,
            "output": 0.0005
        },
        "gpt-4": {
            "input": 0.03,     # $30 per 1M tokens
            "output": 0.06     # $60 per 1M tokens
        },
        "gpt-3.5-turbo": {
            "input": 0.0005,   # $0.50 per 1M tokens
            "output": 0.0015   # $1.50 per 1M tokens
        },
        "claude-3-opus": {
            "input": 0.015,    # $15 per 1M tokens
            "output": 0.075    # $75 per 1M tokens
        },
        "claude-3-sonnet": {
            "input": 0.003,    # $3 per 1M tokens
            "output": 0.015    # $15 per 1M tokens
        }
    }
    
    # Default model for each engine type
    ENGINE_DEFAULT_MODELS = {
        EngineType.TEXT_INTELLIGENCE: "gemini-pro",
        EngineType.CREATIVE_ASSISTANT: "gemini-pro",
        EngineType.SOCIAL_MEDIA_PLANNER: "gemini-pro",
        EngineType.DISCOVERY_ANALYTICS: "gemini-pro",
        EngineType.IMAGE_GENERATION: "gemini-pro-vision",
        EngineType.AUDIO_GENERATION: "gemini-pro",
        EngineType.VIDEO_PIPELINE: "gemini-pro"
    }
    
    @classmethod
    def calculate_cost(
        cls,
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini-pro"
    ) -> float:
        """
        Calculate cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
        
        Returns:
            Total cost in USD
        """
        if model not in cls.MODEL_COSTS:
            logger.warning(f"Unknown model {model}, using gemini-pro pricing")
            model = "gemini-pro"
        
        costs = cls.MODEL_COSTS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return round(input_cost + output_cost, 6)
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Estimate token count from text.
        
        Args:
            text: Input text
        
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return max(1, len(text) // 4)
    
    @classmethod
    def get_model_for_engine(cls, engine_type: EngineType) -> str:
        """Get default model for an engine type."""
        return cls.ENGINE_DEFAULT_MODELS.get(engine_type, "gemini-pro")


class UsageTracker:
    """Track and record usage statistics."""
    
    def __init__(self):
        """Initialize usage tracker."""
        self._lock = asyncio.Lock()
    
    async def record_usage(
        self,
        user_id: str,
        operation_type: str,
        engine_type: EngineType,
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini-pro",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Record usage for a user operation.
        
        Args:
            user_id: User identifier
            operation_type: Type of operation (generation, transformation, etc.)
            engine_type: Engine that performed the operation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model used
            metadata: Additional metadata
        
        Returns:
            Usage record with cost information
        """
        async with self._lock:
            # Calculate cost
            cost = CostCalculator.calculate_cost(input_tokens, output_tokens, model)
            total_tokens = input_tokens + output_tokens
            
            # Create usage record
            usage_record = {
                "user_id": user_id,
                "operation_type": operation_type,
                "engine_type": engine_type.value,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "timestamp": datetime.utcnow(),
                "date": datetime.utcnow().date().isoformat(),
                "metadata": metadata or {}
            }
            
            # Store in database
            database = get_database()
            await database.cost_tracking.insert_one(usage_record)
            
            logger.info(
                f"Usage recorded: user={user_id}, operation={operation_type}, "
                f"tokens={total_tokens}, cost=${cost:.6f}"
            )
            
            return usage_record
    
    async def get_user_usage_today(self, user_id: str) -> Dict:
        """
        Get user's usage statistics for today.
        
        Args:
            user_id: User identifier
        
        Returns:
            Usage statistics for today
        """
        database = get_database()
        today = datetime.utcnow().date().isoformat()
        
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "date": today
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$cost"},
                    "operation_count": {"$sum": 1},
                    "operations_by_type": {
                        "$push": {
                            "type": "$operation_type",
                            "tokens": "$total_tokens",
                            "cost": "$cost"
                        }
                    }
                }
            }
        ]
        
        result = await database.cost_tracking.aggregate(pipeline).to_list(1)
        
        if not result:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "operation_count": 0,
                "operations_by_type": []
            }
        
        return result[0]
    
    async def get_user_usage_this_month(self, user_id: str) -> Dict:
        """
        Get user's usage statistics for this month.
        
        Args:
            user_id: User identifier
        
        Returns:
            Usage statistics for this month
        """
        database = get_database()
        
        # Get first day of current month
        now = datetime.utcnow()
        first_day = datetime(now.year, now.month, 1)
        
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$gte": first_day}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$cost"},
                    "operation_count": {"$sum": 1},
                    "by_engine": {
                        "$push": {
                            "engine": "$engine_type",
                            "tokens": "$total_tokens",
                            "cost": "$cost"
                        }
                    }
                }
            }
        ]
        
        result = await database.cost_tracking.aggregate(pipeline).to_list(1)
        
        if not result:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "operation_count": 0,
                "by_engine": []
            }
        
        return result[0]
    
    async def get_usage_analytics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get detailed usage analytics for a user.
        
        Args:
            user_id: User identifier
            start_date: Start date for analytics (optional)
            end_date: End date for analytics (optional)
        
        Returns:
            Detailed usage analytics
        """
        database = get_database()
        
        # Default to last 30 days if not specified
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        match_filter = {
            "user_id": user_id,
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        
        # Get overall statistics
        overall_pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$cost"},
                    "operation_count": {"$sum": 1},
                    "avg_tokens_per_operation": {"$avg": "$total_tokens"},
                    "avg_cost_per_operation": {"$avg": "$cost"}
                }
            }
        ]
        
        overall_result = await database.cost_tracking.aggregate(overall_pipeline).to_list(1)
        overall_stats = overall_result[0] if overall_result else {
            "total_tokens": 0,
            "total_cost": 0.0,
            "operation_count": 0,
            "avg_tokens_per_operation": 0.0,
            "avg_cost_per_operation": 0.0
        }
        
        # Get breakdown by engine
        engine_pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": "$engine_type",
                    "tokens": {"$sum": "$total_tokens"},
                    "cost": {"$sum": "$cost"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"cost": -1}}
        ]
        
        engine_breakdown = await database.cost_tracking.aggregate(engine_pipeline).to_list(None)
        
        # Get breakdown by operation type
        operation_pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": "$operation_type",
                    "tokens": {"$sum": "$total_tokens"},
                    "cost": {"$sum": "$cost"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"cost": -1}}
        ]
        
        operation_breakdown = await database.cost_tracking.aggregate(operation_pipeline).to_list(None)
        
        # Get daily usage trend
        daily_pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": "$date",
                    "tokens": {"$sum": "$total_tokens"},
                    "cost": {"$sum": "$cost"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        daily_trend = await database.cost_tracking.aggregate(daily_pipeline).to_list(None)
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "overall": overall_stats,
            "by_engine": [
                {
                    "engine": item["_id"],
                    "tokens": item["tokens"],
                    "cost": round(item["cost"], 6),
                    "operations": item["count"]
                }
                for item in engine_breakdown
            ],
            "by_operation": [
                {
                    "operation": item["_id"],
                    "tokens": item["tokens"],
                    "cost": round(item["cost"], 6),
                    "operations": item["count"]
                }
                for item in operation_breakdown
            ],
            "daily_trend": [
                {
                    "date": item["_id"],
                    "tokens": item["tokens"],
                    "cost": round(item["cost"], 6),
                    "operations": item["count"]
                }
                for item in daily_trend
            ]
        }


class UsageLimitMonitor:
    """Monitor usage limits and provide warnings."""
    
    # Warning thresholds (percentage of limit)
    WARNING_THRESHOLDS = [0.5, 0.75, 0.9, 0.95]
    
    def __init__(self):
        """Initialize usage limit monitor."""
        self._warned_users: Dict[str, set] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def check_limits(
        self,
        user: User,
        usage_tracker: UsageTracker
    ) -> Dict[str, any]:
        """
        Check user's usage against limits.
        
        Args:
            user: User object
            usage_tracker: Usage tracker instance
        
        Returns:
            Dictionary with limit status and warnings
        """
        async with self._lock:
            # Get current usage
            user_id = str(user.id) if hasattr(user, 'id') else user.user_id
            usage_today = await usage_tracker.get_user_usage_today(user_id)
            usage_month = await usage_tracker.get_user_usage_this_month(user_id)
            
            # Check limits
            limits_status = {
                "daily_tokens": {
                    "limit": user.usage_limits.daily_token_limit,
                    "used": usage_today["total_tokens"],
                    "remaining": max(0, user.usage_limits.daily_token_limit - usage_today["total_tokens"]),
                    "percentage": (usage_today["total_tokens"] / user.usage_limits.daily_token_limit * 100) if user.usage_limits.daily_token_limit > 0 else 0,
                    "exceeded": usage_today["total_tokens"] >= user.usage_limits.daily_token_limit
                },
                "monthly_cost": {
                    "limit": user.usage_limits.monthly_cost_limit,
                    "used": usage_month["total_cost"],
                    "remaining": max(0, user.usage_limits.monthly_cost_limit - usage_month["total_cost"]),
                    "percentage": (usage_month["total_cost"] / user.usage_limits.monthly_cost_limit * 100) if user.usage_limits.monthly_cost_limit > 0 else 0,
                    "exceeded": usage_month["total_cost"] >= user.usage_limits.monthly_cost_limit
                }
            }
            
            # Generate warnings
            warnings = []
            
            # Check daily token limit
            daily_pct = limits_status["daily_tokens"]["percentage"]
            for threshold in self.WARNING_THRESHOLDS:
                threshold_key = f"daily_tokens_{int(threshold * 100)}"
                if daily_pct >= threshold * 100 and threshold_key not in self._warned_users[user_id]:
                    warnings.append({
                        "type": "daily_tokens",
                        "severity": "critical" if threshold >= 0.9 else "warning",
                        "message": f"Daily token limit {int(threshold * 100)}% reached: {usage_today['total_tokens']:,} / {user.usage_limits.daily_token_limit:,} tokens",
                        "threshold": threshold
                    })
                    self._warned_users[user_id].add(threshold_key)
            
            # Check monthly cost limit
            monthly_pct = limits_status["monthly_cost"]["percentage"]
            for threshold in self.WARNING_THRESHOLDS:
                threshold_key = f"monthly_cost_{int(threshold * 100)}"
                if monthly_pct >= threshold * 100 and threshold_key not in self._warned_users[user_id]:
                    warnings.append({
                        "type": "monthly_cost",
                        "severity": "critical" if threshold >= 0.9 else "warning",
                        "message": f"Monthly cost limit {int(threshold * 100)}% reached: ${usage_month['total_cost']:.2f} / ${user.usage_limits.monthly_cost_limit:.2f}",
                        "threshold": threshold
                    })
                    self._warned_users[user_id].add(threshold_key)
            
            # Log warnings
            for warning in warnings:
                if warning["severity"] == "critical":
                    logger.warning(
                        f"Usage limit warning for user {user_id}: {warning['message']}"
                    )
            
            return {
                "limits": limits_status,
                "warnings": warnings,
                "any_exceeded": any(limit["exceeded"] for limit in limits_status.values())
            }
    
    async def reset_warnings(self, user_id: str):
        """Reset warning flags for a user (e.g., at day/month boundary)."""
        async with self._lock:
            if user_id in self._warned_users:
                self._warned_users[user_id].clear()
    
    async def can_perform_operation(
        self,
        user: User,
        estimated_tokens: int,
        usage_tracker: UsageTracker
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user can perform an operation based on estimated token usage.
        
        Args:
            user: User object
            estimated_tokens: Estimated tokens for the operation
            usage_tracker: Usage tracker instance
        
        Returns:
            Tuple of (can_perform, reason_if_not)
        """
        # Get current usage
        user_id = str(user.id) if hasattr(user, 'id') else user.user_id
        usage_today = await usage_tracker.get_user_usage_today(user_id)
        usage_month = await usage_tracker.get_user_usage_this_month(user_id)
        
        # Check daily token limit
        if usage_today["total_tokens"] + estimated_tokens > user.usage_limits.daily_token_limit:
            return False, f"Daily token limit would be exceeded. Used: {usage_today['total_tokens']:,}, Limit: {user.usage_limits.daily_token_limit:,}"
        
        # Estimate cost for the operation
        estimated_cost = CostCalculator.calculate_cost(
            estimated_tokens // 2,  # Rough split between input/output
            estimated_tokens // 2,
            "gemini-pro"
        )
        
        # Check monthly cost limit
        if usage_month["total_cost"] + estimated_cost > user.usage_limits.monthly_cost_limit:
            return False, f"Monthly cost limit would be exceeded. Used: ${usage_month['total_cost']:.2f}, Limit: ${user.usage_limits.monthly_cost_limit:.2f}"
        
        return True, None


class CostTrackingService:
    """Main cost tracking and usage management service."""
    
    def __init__(self):
        """Initialize cost tracking service."""
        self.calculator = CostCalculator()
        self.tracker = UsageTracker()
        self.monitor = UsageLimitMonitor()
    
    async def track_operation(
        self,
        user: User,
        operation_type: str,
        engine_type: EngineType,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Track an AI operation with cost calculation.
        
        Args:
            user: User object
            operation_type: Type of operation
            engine_type: Engine that performed the operation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model used (optional, defaults to engine default)
            metadata: Additional metadata
        
        Returns:
            Usage record with cost information
        """
        # Use default model for engine if not specified
        if not model:
            model = self.calculator.get_model_for_engine(engine_type)
        
        # Get user_id
        user_id = str(user.id) if hasattr(user, 'id') else user.user_id
        
        # Record usage
        usage_record = await self.tracker.record_usage(
            user_id,
            operation_type,
            engine_type,
            input_tokens,
            output_tokens,
            model,
            metadata
        )
        
        # Check limits and generate warnings
        limit_status = await self.monitor.check_limits(user, self.tracker)
        
        # Add limit status to response
        usage_record["limit_status"] = limit_status
        
        return usage_record
    
    async def check_can_perform_operation(
        self,
        user: User,
        estimated_text: str
    ) -> Tuple[bool, Optional[str], Dict]:
        """
        Check if user can perform an operation.
        
        Args:
            user: User object
            estimated_text: Text to estimate token usage from
        
        Returns:
            Tuple of (can_perform, reason_if_not, limit_status)
        """
        # Estimate tokens
        estimated_tokens = self.calculator.estimate_tokens(estimated_text)
        
        # Check if operation can be performed
        can_perform, reason = await self.monitor.can_perform_operation(
            user,
            estimated_tokens,
            self.tracker
        )
        
        # Get current limit status
        limit_status = await self.monitor.check_limits(user, self.tracker)
        
        return can_perform, reason, limit_status
    
    async def get_usage_report(
        self,
        user_id: str,
        report_type: str = "today"
    ) -> Dict:
        """
        Get usage report for a user.
        
        Args:
            user_id: User identifier
            report_type: Type of report (today, month, custom)
        
        Returns:
            Usage report
        """
        if report_type == "today":
            return await self.tracker.get_user_usage_today(user_id)
        elif report_type == "month":
            return await self.tracker.get_user_usage_this_month(user_id)
        else:
            # Default to analytics for custom reports
            return await self.tracker.get_usage_analytics(user_id)
    
    async def get_detailed_analytics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get detailed usage analytics.
        
        Args:
            user_id: User identifier
            start_date: Start date (optional)
            end_date: End date (optional)
        
        Returns:
            Detailed analytics report
        """
        return await self.tracker.get_usage_analytics(user_id, start_date, end_date)
    
    async def estimate_operation_cost(
        self,
        text: str,
        engine_type: EngineType,
        model: Optional[str] = None
    ) -> Dict:
        """
        Estimate cost for an operation.
        
        Args:
            text: Input text
            engine_type: Engine type
            model: Model to use (optional)
        
        Returns:
            Cost estimation
        """
        if not model:
            model = self.calculator.get_model_for_engine(engine_type)
        
        estimated_tokens = self.calculator.estimate_tokens(text)
        # Assume 50/50 split between input and output
        input_tokens = estimated_tokens // 2
        output_tokens = estimated_tokens // 2
        
        cost = self.calculator.calculate_cost(input_tokens, output_tokens, model)
        
        return {
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_total_tokens": estimated_tokens,
            "estimated_cost": cost,
            "model": model,
            "currency": "USD"
        }


# Global cost tracking service instance
cost_tracking_service = CostTrackingService()

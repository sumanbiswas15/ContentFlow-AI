"""
Job handlers for different types of async jobs.

This module contains handler functions that execute specific job types
in the background job processing system using the AI Orchestrator and
specialized engines.
"""

import asyncio
import logging
from typing import Any, Dict
from datetime import datetime

from app.models.jobs import AsyncJob
from app.models.base import JobType, EngineType
from app.core.exceptions import JobProcessingError, AIServiceError
from app.ai.orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)

# Global orchestrator instance (will be initialized by the application)
_orchestrator: AIOrchestrator = None


def set_orchestrator(orchestrator: AIOrchestrator):
    """Set the global orchestrator instance for job handlers."""
    global _orchestrator
    _orchestrator = orchestrator
    logger.info("Orchestrator set for job handlers")


def get_orchestrator() -> AIOrchestrator:
    """Get the global orchestrator instance."""
    if _orchestrator is None:
        # Initialize orchestrator if not set
        return AIOrchestrator()
    return _orchestrator


class JobResult:
    """Simple job result class for job processing."""
    
    def __init__(self, data: Any, tokens_used: int = 0, cost: float = 0.0):
        self.data = data
        self.tokens_used = tokens_used
        self.cost = cost


async def handle_content_generation(job: AsyncJob) -> JobResult:
    """
    Handle content generation jobs using Text Intelligence Engine.
    """
    logger.info(f"Processing content generation job {job.id}")
    
    try:
        orchestrator = get_orchestrator()
        
        # Extract parameters
        content_type = job.parameters.get("content_type", "blog")
        prompt = job.parameters.get("prompt", "")
        tone = job.parameters.get("tone", "professional")
        target_length = job.parameters.get("target_length")
        
        if not prompt:
            raise JobProcessingError(
                job_id=str(job.id),
                message="Prompt is required for content generation"
            )
        
        # Execute through orchestrator
        result = await orchestrator.execute_engine_operation(
            engine_type=EngineType.TEXT_INTELLIGENCE,
            operation="generate",
            parameters={
                "content_type": content_type,
                "prompt": prompt,
                "tone": tone,
                "target_length": target_length
            }
        )
        
        logger.info(f"Content generation job {job.id} completed successfully")
        
        return JobResult(
            data=result,
            tokens_used=result.get("metadata", {}).get("tokens_used", 0),
            cost=result.get("cost", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Content generation job {job.id} failed: {e}")
        raise JobProcessingError(
            job_id=str(job.id),
            message=f"Content generation failed: {str(e)}"
        )


async def handle_content_transformation(job: AsyncJob) -> JobResult:
    """
    Handle content transformation jobs using Text Intelligence Engine.
    """
    logger.info(f"Processing content transformation job {job.id}")
    
    try:
        orchestrator = get_orchestrator()
        
        # Extract parameters
        content = job.parameters.get("content", "")
        transformation_type = job.parameters.get("transformation_type", "summarize")
        
        if not content:
            raise JobProcessingError(
                job_id=str(job.id),
                message="Content is required for transformation"
            )
        
        # Route to appropriate operation
        if transformation_type == "summarize":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.TEXT_INTELLIGENCE,
                operation="summarize",
                parameters={
                    "content": content,
                    "target_length": job.parameters.get("target_length", 100)
                }
            )
        elif transformation_type == "tone_change":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.TEXT_INTELLIGENCE,
                operation="transform_tone",
                parameters={
                    "content": content,
                    "target_tone": job.parameters.get("target_tone", "professional")
                }
            )
        elif transformation_type == "translate":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.TEXT_INTELLIGENCE,
                operation="translate",
                parameters={
                    "content": content,
                    "target_language": job.parameters.get("target_language", "es")
                }
            )
        else:
            raise JobProcessingError(
                job_id=str(job.id),
                message=f"Unknown transformation type: {transformation_type}"
            )
        
        logger.info(f"Content transformation job {job.id} completed successfully")
        
        return JobResult(
            data=result,
            tokens_used=result.get("metadata", {}).get("tokens_used", 0),
            cost=result.get("cost", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Content transformation job {job.id} failed: {e}")
        raise JobProcessingError(
            job_id=str(job.id),
            message=f"Content transformation failed: {str(e)}"
        )


async def handle_creative_assistance(job: AsyncJob) -> JobResult:
    """
    Handle creative assistance jobs using Creative Assistant Engine.
    """
    logger.info(f"Processing creative assistance job {job.id}")
    
    try:
        orchestrator = get_orchestrator()
        
        # Extract parameters
        session_id = job.parameters.get("session_id")
        operation_type = job.parameters.get("operation_type", "suggest")
        
        if not session_id and operation_type != "start_session":
            raise JobProcessingError(
                job_id=str(job.id),
                message="Session ID is required for creative assistance"
            )
        
        # Route to appropriate operation
        if operation_type == "start_session":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.CREATIVE_ASSISTANT,
                operation="start_session",
                parameters=job.parameters.get("context", {})
            )
        elif operation_type == "suggest":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.CREATIVE_ASSISTANT,
                operation="suggest",
                parameters={
                    "session_id": session_id,
                    **job.parameters.get("request", {})
                }
            )
        elif operation_type == "refine":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.CREATIVE_ASSISTANT,
                operation="refine",
                parameters={
                    "session_id": session_id,
                    **job.parameters.get("feedback", {})
                }
            )
        else:
            raise JobProcessingError(
                job_id=str(job.id),
                message=f"Unknown operation type: {operation_type}"
            )
        
        logger.info(f"Creative assistance job {job.id} completed successfully")
        
        return JobResult(
            data=result,
            tokens_used=0,  # Tokens tracked at engine level
            cost=0.0
        )
        
    except Exception as e:
        logger.error(f"Creative assistance job {job.id} failed: {e}")
        raise JobProcessingError(
            job_id=str(job.id),
            message=f"Creative assistance failed: {str(e)}"
        )


async def handle_social_media_optimization(job: AsyncJob) -> JobResult:
    """
    Handle social media optimization jobs using Social Media Planner.
    """
    logger.info(f"Processing social media optimization job {job.id}")
    
    try:
        orchestrator = get_orchestrator()
        
        # Extract parameters
        content = job.parameters.get("content", "")
        platform = job.parameters.get("platform", "generic")
        optimization_type = job.parameters.get("optimization_type", "optimize")
        
        if not content and optimization_type != "suggest_times":
            raise JobProcessingError(
                job_id=str(job.id),
                message="Content is required for social media optimization"
            )
        
        # Route to appropriate operation
        if optimization_type == "optimize":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.SOCIAL_MEDIA_PLANNER,
                operation="optimize",
                parameters={
                    "content": content,
                    "platform": platform,
                    **job.parameters
                }
            )
        elif optimization_type == "hashtags":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.SOCIAL_MEDIA_PLANNER,
                operation="generate_hashtags",
                parameters={
                    "content": content,
                    "platform": platform,
                    "count": job.parameters.get("count", 5)
                }
            )
        elif optimization_type == "suggest_times":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.SOCIAL_MEDIA_PLANNER,
                operation="suggest_times",
                parameters={
                    "platform": platform,
                    "target_audience": job.parameters.get("target_audience", "general"),
                    "days_ahead": job.parameters.get("days_ahead", 7)
                }
            )
        elif optimization_type == "predict_engagement":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.SOCIAL_MEDIA_PLANNER,
                operation="predict_engagement",
                parameters={
                    "content": content,
                    "platform": platform,
                    **job.parameters
                }
            )
        else:
            raise JobProcessingError(
                job_id=str(job.id),
                message=f"Unknown optimization type: {optimization_type}"
            )
        
        logger.info(f"Social media optimization job {job.id} completed successfully")
        
        return JobResult(
            data=result,
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Social media optimization job {job.id} failed: {e}")
        raise JobProcessingError(
            job_id=str(job.id),
            message=f"Social media optimization failed: {str(e)}"
        )


async def handle_analytics_processing(job: AsyncJob) -> JobResult:
    """
    Handle analytics processing jobs using Discovery Analytics Engine.
    """
    logger.info(f"Processing analytics job {job.id}")
    
    try:
        orchestrator = get_orchestrator()
        
        # Extract parameters
        content = job.parameters.get("content", "")
        analysis_type = job.parameters.get("analysis_type", "tag")
        
        if not content and analysis_type not in ["analyze_trends"]:
            raise JobProcessingError(
                job_id=str(job.id),
                message="Content is required for analytics processing"
            )
        
        # Route to appropriate operation
        if analysis_type == "tag":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.DISCOVERY_ANALYTICS,
                operation="tag",
                parameters={
                    "content": content,
                    "content_type": job.parameters.get("content_type", "text"),
                    "max_tags": job.parameters.get("max_tags", 10)
                }
            )
        elif analysis_type == "analyze_trends":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.DISCOVERY_ANALYTICS,
                operation="analyze_trends",
                parameters={
                    "time_period": job.parameters.get("time_period", "7d"),
                    "content_data": job.parameters.get("content_data", [])
                }
            )
        elif analysis_type == "suggest_improvements":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.DISCOVERY_ANALYTICS,
                operation="suggest_improvements",
                parameters={
                    "content": content,
                    **job.parameters
                }
            )
        else:
            raise JobProcessingError(
                job_id=str(job.id),
                message=f"Unknown analysis type: {analysis_type}"
            )
        
        logger.info(f"Analytics processing job {job.id} completed successfully")
        
        return JobResult(
            data=result,
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Analytics processing job {job.id} failed: {e}")
        raise JobProcessingError(
            job_id=str(job.id),
            message=f"Analytics processing failed: {str(e)}"
        )


async def handle_media_generation(job: AsyncJob) -> JobResult:
    """
    Handle media generation jobs (images, audio, video) using appropriate engines.
    """
    logger.info(f"Processing media generation job {job.id}")
    
    try:
        orchestrator = get_orchestrator()
        
        # Extract parameters
        media_type = job.parameters.get("media_type", "image")
        prompt = job.parameters.get("prompt") or job.parameters.get("text") or job.parameters.get("script")
        
        if not prompt:
            raise JobProcessingError(
                job_id=str(job.id),
                message="Prompt/text/script is required for media generation"
            )
        
        # Route to appropriate engine based on media type
        if media_type == "image":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.IMAGE_GENERATION,
                operation="generate",
                parameters=job.parameters
            )
        elif media_type == "audio":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.AUDIO_GENERATION,
                operation="generate",
                parameters=job.parameters
            )
        elif media_type == "video":
            result = await orchestrator.execute_engine_operation(
                engine_type=EngineType.VIDEO_PIPELINE,
                operation="generate",
                parameters=job.parameters
            )
        else:
            raise JobProcessingError(
                job_id=str(job.id),
                message=f"Unknown media type: {media_type}"
            )
        
        logger.info(f"Media generation job {job.id} completed successfully")
        
        return JobResult(
            data=result,
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Media generation job {job.id} failed: {e}")
        raise JobProcessingError(
            job_id=str(job.id),
            message=f"Media generation failed: {str(e)}"
        )


# Job handler registry
JOB_HANDLERS = {
    JobType.CONTENT_GENERATION: handle_content_generation,
    JobType.CONTENT_TRANSFORMATION: handle_content_transformation,
    JobType.CREATIVE_ASSISTANCE: handle_creative_assistance,
    JobType.SOCIAL_MEDIA_OPTIMIZATION: handle_social_media_optimization,
    JobType.ANALYTICS_PROCESSING: handle_analytics_processing,
    JobType.MEDIA_GENERATION: handle_media_generation,
}


async def register_all_handlers(job_service):
    """Register all job handlers with the job service."""
    for job_type, handler in JOB_HANDLERS.items():
        await job_service.register_job_handler(job_type, handler)
    
    logger.info(f"Registered {len(JOB_HANDLERS)} job handlers")
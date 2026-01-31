"""
AI Orchestrator for ContentFlow AI.

This module implements the central AI orchestration layer using Google Gemini LLM
for intelligent task decomposition, routing, and workflow coordination.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import google.generativeai as genai
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.database import get_database
from app.models.base import (
    WorkflowState, EngineType, JobType, JobStatus, ErrorCode,
    ValidationResult, SystemHealth
)
from app.models.content import ContentItem
from app.models.jobs import AsyncJob, JobResult, WorkflowExecution
from app.core.exceptions import (
    OrchestrationError, EngineError, ValidationError,
    RateLimitError, UsageLimitError
)

# Import all specialized engines
from app.ai.text_intelligence_engine import TextIntelligenceEngine
from app.ai.creative_assistant_engine import CreativeAssistantEngine
from app.ai.social_media_planner import SocialMediaPlanner
from app.ai.discovery_analytics_engine import DiscoveryAnalyticsEngine
from app.ai.image_generation_engine import ImageGenerationEngine
from app.ai.audio_generation_engine import AudioGenerationEngine
from app.ai.video_pipeline_engine import VideoPipelineEngine

logger = logging.getLogger(__name__)


class TaskPriority(int, Enum):
    """Task priority levels for orchestration."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class WorkflowRequest(BaseModel):
    """Model for workflow orchestration requests."""
    content_id: Optional[str] = None
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_engines: List[EngineType] = Field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    user_id: str
    workflow_context: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    timeout_seconds: int = 300


class WorkflowResponse(BaseModel):
    """Model for workflow orchestration responses."""
    workflow_id: str
    status: str
    job_ids: List[str] = Field(default_factory=list)
    estimated_completion_time: Optional[datetime] = None
    cost_estimate: float = 0.0
    message: str = ""


class EngineCapability(BaseModel):
    """Model for engine capability description."""
    engine_type: EngineType
    operations: List[str]
    input_types: List[str]
    output_types: List[str]
    cost_per_operation: float = 0.0
    average_processing_time: int = 0  # seconds
    availability: bool = True
    error_rate: float = 0.0


class AIOrchestrator:
    """
    Central AI orchestration layer using Google Gemini LLM for intelligent
    task decomposition, routing, and workflow coordination.
    """
    
    def __init__(self):
        """Initialize the AI Orchestrator."""
        self.gemini_client = None
        self.engine_capabilities: Dict[EngineType, EngineCapability] = {}
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.system_health = SystemHealth(status="healthy")
        
        # Initialize engine instances
        self.engines: Dict[EngineType, Any] = {}
        
        self._initialize_gemini()
        self._initialize_engine_capabilities()
        self._initialize_engines()
    
    def _initialize_gemini(self):
        """Initialize Google Gemini client."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. AI orchestration will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.system_health.add_service_status("gemini", "unhealthy")
            raise OrchestrationError(f"Failed to initialize AI orchestrator: {e}")
    
    def _initialize_engine_capabilities(self):
        """Initialize engine capability mappings."""
        self.engine_capabilities = {
            EngineType.TEXT_INTELLIGENCE: EngineCapability(
                engine_type=EngineType.TEXT_INTELLIGENCE,
                operations=["generate", "summarize", "transform_tone", "translate", "parse"],
                input_types=["text", "url"],
                output_types=["text"],
                cost_per_operation=0.01,
                average_processing_time=5
            ),
            EngineType.CREATIVE_ASSISTANT: EngineCapability(
                engine_type=EngineType.CREATIVE_ASSISTANT,
                operations=["suggest", "refine", "ideate", "design_assist"],
                input_types=["text", "context"],
                output_types=["suggestions", "text"],
                cost_per_operation=0.02,
                average_processing_time=8
            ),
            EngineType.SOCIAL_MEDIA_PLANNER: EngineCapability(
                engine_type=EngineType.SOCIAL_MEDIA_PLANNER,
                operations=["optimize", "generate_hashtags", "schedule", "predict_engagement"],
                input_types=["text", "image", "video"],
                output_types=["optimized_content", "hashtags", "schedule"],
                cost_per_operation=0.005,
                average_processing_time=3
            ),
            EngineType.DISCOVERY_ANALYTICS: EngineCapability(
                engine_type=EngineType.DISCOVERY_ANALYTICS,
                operations=["tag", "analyze_trends", "calculate_metrics", "suggest_improvements"],
                input_types=["text", "image", "video", "metrics"],
                output_types=["tags", "analytics", "suggestions"],
                cost_per_operation=0.008,
                average_processing_time=10
            ),
            EngineType.IMAGE_GENERATION: EngineCapability(
                engine_type=EngineType.IMAGE_GENERATION,
                operations=["generate_thumbnail", "generate_poster", "edit_image"],
                input_types=["text", "image"],
                output_types=["image"],
                cost_per_operation=0.05,
                average_processing_time=15
            ),
            EngineType.AUDIO_GENERATION: EngineCapability(
                engine_type=EngineType.AUDIO_GENERATION,
                operations=["generate_voiceover", "generate_music", "edit_audio"],
                input_types=["text", "audio"],
                output_types=["audio"],
                cost_per_operation=0.03,
                average_processing_time=20
            ),
            EngineType.VIDEO_PIPELINE: EngineCapability(
                engine_type=EngineType.VIDEO_PIPELINE,
                operations=["create_video", "edit_video", "add_effects"],
                input_types=["text", "image", "audio", "video"],
                output_types=["video"],
                cost_per_operation=0.10,
                average_processing_time=60
            )
        }
    
    def _initialize_engines(self):
        """Initialize all specialized engine instances."""
        try:
            logger.info("Initializing specialized engines...")
            
            # Initialize Text Intelligence Engine
            self.engines[EngineType.TEXT_INTELLIGENCE] = TextIntelligenceEngine()
            logger.info("Text Intelligence Engine initialized")
            
            # Initialize Creative Assistant Engine
            self.engines[EngineType.CREATIVE_ASSISTANT] = CreativeAssistantEngine()
            logger.info("Creative Assistant Engine initialized")
            
            # Initialize Social Media Planner
            self.engines[EngineType.SOCIAL_MEDIA_PLANNER] = SocialMediaPlanner()
            logger.info("Social Media Planner initialized")
            
            # Initialize Discovery Analytics Engine
            self.engines[EngineType.DISCOVERY_ANALYTICS] = DiscoveryAnalyticsEngine()
            logger.info("Discovery Analytics Engine initialized")
            
            # Initialize Image Generation Engine
            self.engines[EngineType.IMAGE_GENERATION] = ImageGenerationEngine()
            logger.info("Image Generation Engine initialized")
            
            # Initialize Audio Generation Engine
            self.engines[EngineType.AUDIO_GENERATION] = AudioGenerationEngine()
            logger.info("Audio Generation Engine initialized")
            
            # Initialize Video Pipeline Engine
            self.engines[EngineType.VIDEO_PIPELINE] = VideoPipelineEngine()
            logger.info("Video Pipeline Engine initialized")
            
            logger.info(f"Successfully initialized {len(self.engines)} specialized engines")
            
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}")
            # Mark engines as unavailable but don't fail orchestrator initialization
            for engine_type in EngineType:
                if engine_type not in self.engines:
                    self.engine_capabilities[engine_type].availability = False
                    self.system_health.add_service_status(engine_type.value, "unavailable")
    
    def get_engine(self, engine_type: EngineType) -> Optional[Any]:
        """
        Get a specialized engine instance by type.
        
        Args:
            engine_type: Type of engine to retrieve
            
        Returns:
            Engine instance or None if not available
        """
        return self.engines.get(engine_type)
    
    async def execute_engine_operation(
        self,
        engine_type: EngineType,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an operation on a specialized engine.
        
        Args:
            engine_type: Type of engine to use
            operation: Operation to execute
            parameters: Operation parameters
            
        Returns:
            Operation result dictionary
            
        Raises:
            EngineError: If engine is unavailable or operation fails
        """
        try:
            # Get engine instance
            engine = self.get_engine(engine_type)
            if not engine:
                raise EngineError(
                    engine_type.value,
                    f"Engine {engine_type.value} is not available"
                )
            
            # Check engine capability
            capability = self.engine_capabilities.get(engine_type)
            if not capability or not capability.availability:
                raise EngineError(
                    engine_type.value,
                    f"Engine {engine_type.value} is currently unavailable"
                )
            
            # Route to appropriate engine method based on operation
            result = await self._route_engine_operation(engine, engine_type, operation, parameters)
            
            # Update engine metrics
            capability.error_rate = max(0, capability.error_rate - 0.01)  # Decrease error rate on success
            
            return result
            
        except Exception as e:
            logger.error(f"Engine operation failed: {engine_type.value}.{operation} - {e}")
            
            # Update engine metrics
            if engine_type in self.engine_capabilities:
                self.engine_capabilities[engine_type].error_rate = min(
                    1.0,
                    self.engine_capabilities[engine_type].error_rate + 0.05
                )
            
            # Handle error with recovery strategy
            error_response = await self.handle_engine_error(engine_type.value, e)
            raise EngineError(engine_type.value, str(e), details=error_response)
    
    async def _route_engine_operation(
        self,
        engine: Any,
        engine_type: EngineType,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route operation to the appropriate engine method.
        
        Args:
            engine: Engine instance
            engine_type: Type of engine
            operation: Operation name
            parameters: Operation parameters
            
        Returns:
            Operation result
        """
        # Text Intelligence Engine operations
        if engine_type == EngineType.TEXT_INTELLIGENCE:
            if operation == "generate":
                from app.ai.text_intelligence_engine import GenerationRequest
                request = GenerationRequest(**parameters)
                result = await engine.generate_content(request)
                return {"content": result.content, "metadata": result.metadata, "cost": result.cost}
            
            elif operation == "summarize":
                from app.ai.text_intelligence_engine import SummarizationRequest
                request = SummarizationRequest(**parameters)
                result = await engine.summarize_content(request)
                return {"content": result.content, "metadata": result.metadata, "cost": result.cost}
            
            elif operation == "transform_tone":
                from app.ai.text_intelligence_engine import ToneTransformationRequest
                request = ToneTransformationRequest(**parameters)
                result = await engine.transform_tone(request)
                return {"content": result.content, "metadata": result.metadata, "cost": result.cost}
            
            elif operation == "translate":
                from app.ai.text_intelligence_engine import TranslationRequest
                request = TranslationRequest(**parameters)
                result = await engine.translate_content(request)
                return {"content": result.content, "metadata": result.metadata, "cost": result.cost}
            
            elif operation == "adapt_platform":
                from app.ai.text_intelligence_engine import PlatformAdaptationRequest
                request = PlatformAdaptationRequest(**parameters)
                result = await engine.adapt_for_platform(request)
                return {"content": result.content, "metadata": result.metadata, "cost": result.cost}
        
        # Creative Assistant Engine operations
        elif engine_type == EngineType.CREATIVE_ASSISTANT:
            if operation == "start_session":
                from app.ai.creative_assistant_engine import CreativeContext
                context = CreativeContext(**parameters)
                session_id = await engine.start_creative_session(context)
                return {"session_id": session_id}
            
            elif operation == "suggest":
                from app.ai.creative_assistant_engine import SuggestionRequest
                session_id = parameters.pop("session_id")
                request = SuggestionRequest(**parameters)
                suggestions = await engine.provide_suggestions(session_id, request)
                return {"suggestions": [s.dict() for s in suggestions]}
            
            elif operation == "refine":
                from app.ai.creative_assistant_engine import Feedback
                session_id = parameters.pop("session_id")
                feedback = Feedback(**parameters)
                suggestions = await engine.refine_suggestions(session_id, feedback)
                return {"suggestions": [s.dict() for s in suggestions]}
            
            elif operation == "design_assist":
                from app.ai.creative_assistant_engine import DesignAssistanceRequest
                session_id = parameters.pop("session_id")
                request = DesignAssistanceRequest(**parameters)
                suggestions = await engine.provide_design_assistance(session_id, request)
                return {"suggestions": [s.dict() for s in suggestions]}
            
            elif operation == "marketing_assist":
                from app.ai.creative_assistant_engine import MarketingAssistanceRequest
                session_id = parameters.pop("session_id")
                request = MarketingAssistanceRequest(**parameters)
                suggestions = await engine.provide_marketing_assistance(session_id, request)
                return {"suggestions": [s.dict() for s in suggestions]}
        
        # Social Media Planner operations
        elif engine_type == EngineType.SOCIAL_MEDIA_PLANNER:
            if operation == "optimize":
                from app.ai.social_media_planner import OptimizationRequest
                request = OptimizationRequest(**parameters)
                result = await engine.optimize_for_platform(request)
                return result.dict()
            
            elif operation == "generate_hashtags":
                from app.ai.social_media_planner import HashtagRequest
                request = HashtagRequest(**parameters)
                hashtags = await engine.generate_hashtags(request)
                return {"hashtags": hashtags}
            
            elif operation == "suggest_times":
                from app.ai.social_media_planner import PostingTimeRequest
                request = PostingTimeRequest(**parameters)
                suggestions = await engine.suggest_posting_times(request)
                return {"suggestions": [s.dict() for s in suggestions]}
            
            elif operation == "predict_engagement":
                from app.ai.social_media_planner import EngagementPredictionRequest
                request = EngagementPredictionRequest(**parameters)
                score = await engine.predict_engagement(request)
                return score.dict()
        
        # Discovery Analytics Engine operations
        elif engine_type == EngineType.DISCOVERY_ANALYTICS:
            if operation == "tag":
                from app.ai.discovery_analytics_engine import ContentTaggingRequest
                request = ContentTaggingRequest(**parameters)
                result = await engine.auto_tag_content(request)
                return result.dict()
            
            elif operation == "analyze_trends":
                from app.ai.discovery_analytics_engine import TrendAnalysisRequest
                request = TrendAnalysisRequest(**parameters)
                content_data = parameters.get("content_data", [])
                result = await engine.analyze_trends(request, content_data)
                return result.dict()
            
            elif operation == "calculate_metrics":
                from app.ai.discovery_analytics_engine import EngagementAnalysisRequest
                from app.models.content import EngagementMetrics
                request = EngagementAnalysisRequest(**parameters)
                raw_metrics = EngagementMetrics(**parameters.get("raw_metrics", {}))
                result = await engine.calculate_engagement_metrics(request, raw_metrics)
                return result.dict()
            
            elif operation == "suggest_improvements":
                from app.ai.discovery_analytics_engine import ImprovementSuggestionsRequest
                request = ImprovementSuggestionsRequest(**parameters)
                result = await engine.generate_improvement_suggestions(request)
                return result.dict()
        
        # Image Generation Engine operations
        elif engine_type == EngineType.IMAGE_GENERATION:
            if operation == "generate":
                from app.ai.image_generation_engine import ImageGenerationRequest
                request = ImageGenerationRequest(**parameters)
                result = await engine.generate_image(request)
                return result.dict()
            
            elif operation == "generate_thumbnail":
                result = await engine.generate_thumbnail(
                    prompt=parameters.get("prompt"),
                    width=parameters.get("width", 320),
                    height=parameters.get("height", 180),
                    style=parameters.get("style")
                )
                return result.dict()
            
            elif operation == "generate_poster":
                result = await engine.generate_poster(
                    prompt=parameters.get("prompt"),
                    width=parameters.get("width", 1920),
                    height=parameters.get("height", 1080),
                    style=parameters.get("style")
                )
                return result.dict()
        
        # Audio Generation Engine operations
        elif engine_type == EngineType.AUDIO_GENERATION:
            if operation == "generate":
                from app.ai.audio_generation_engine import AudioGenerationRequest
                request = AudioGenerationRequest(**parameters)
                result = await engine.generate_audio(request)
                return result.dict()
            
            elif operation == "generate_voiceover":
                result = await engine.generate_voiceover(
                    text=parameters.get("text"),
                    voice_style=parameters.get("voice_style"),
                    format=parameters.get("format")
                )
                return result.dict()
            
            elif operation == "generate_music":
                result = await engine.generate_background_music(
                    genre=parameters.get("genre"),
                    duration_seconds=parameters.get("duration_seconds", 60),
                    mood=parameters.get("mood", "uplifting"),
                    tempo=parameters.get("tempo", "medium"),
                    format=parameters.get("format")
                )
                return result.dict()
        
        # Video Pipeline Engine operations
        elif engine_type == EngineType.VIDEO_PIPELINE:
            if operation == "generate":
                from app.ai.video_pipeline_engine import VideoGenerationRequest
                request = VideoGenerationRequest(**parameters)
                result = await engine.generate_video(request)
                return result.dict()
            
            elif operation == "generate_short_form":
                result = await engine.generate_short_form_video(
                    script=parameters.get("script"),
                    duration_seconds=parameters.get("duration_seconds", 60),
                    style=parameters.get("style"),
                    quality=parameters.get("quality")
                )
                return result.dict()
            
            elif operation == "generate_explainer":
                result = await engine.generate_explainer_video(
                    script=parameters.get("script"),
                    duration_seconds=parameters.get("duration_seconds", 120),
                    style=parameters.get("style"),
                    include_subtitles=parameters.get("include_subtitles", True)
                )
                return result.dict()
        
        # Default: operation not supported
        raise EngineError(
            engine_type.value,
            f"Operation '{operation}' not supported for engine {engine_type.value}"
        )

    
    async def orchestrate_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """
        Main orchestration method that coordinates complex workflows using LLM reasoning.
        
        Args:
            request: Workflow orchestration request
            
        Returns:
            WorkflowResponse with workflow details and job IDs
        """
        try:
            logger.info(f"Starting workflow orchestration for operation: {request.operation}")
            
            # Validate request
            validation_result = await self._validate_workflow_request(request)
            if not validation_result.is_valid:
                raise ValidationError(f"Invalid workflow request: {validation_result.errors}")
            
            # Use LLM to decompose and plan the workflow
            workflow_plan = await self._plan_workflow_with_llm(request)
            
            # Create workflow execution record
            workflow_execution = WorkflowExecution(
                workflow_name=f"{request.operation}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                description=f"Orchestrated workflow for {request.operation}",
                user_id=request.user_id
            )
            
            # Create and queue jobs based on the plan
            job_ids = await self._create_workflow_jobs(workflow_plan, request, str(workflow_execution.id))
            workflow_execution.job_ids = job_ids
            
            # Store workflow execution
            database = get_database()
            await database.workflow_executions.insert_one(workflow_execution.dict())
            
            # Track active workflow
            self.active_workflows[str(workflow_execution.id)] = workflow_execution
            
            # Calculate estimates
            cost_estimate = await self._calculate_cost_estimate(workflow_plan)
            completion_estimate = await self._estimate_completion_time(workflow_plan)
            
            response = WorkflowResponse(
                workflow_id=str(workflow_execution.id),
                status="queued",
                job_ids=job_ids,
                estimated_completion_time=completion_estimate,
                cost_estimate=cost_estimate,
                message=f"Workflow created with {len(job_ids)} jobs"
            )
            
            logger.info(f"Workflow orchestration completed: {response.workflow_id}")
            return response
            
        except Exception as e:
            logger.error(f"Workflow orchestration failed: {e}")
            await self._handle_orchestration_error(request, e)
            raise
    
    async def route_task(self, task: Dict[str, Any]) -> EngineType:
        """
        Route a task to the appropriate engine using LLM reasoning.
        
        Args:
            task: Task description and parameters
            
        Returns:
            EngineType for the most appropriate engine
        """
        try:
            # Use LLM to analyze task and determine best engine
            routing_prompt = self._build_routing_prompt(task)
            
            if self.gemini_client:
                response = await self._query_gemini(routing_prompt)
                engine_type = self._parse_routing_response(response)
            else:
                # Fallback to rule-based routing
                engine_type = self._fallback_route_task(task)
            
            logger.info(f"Task routed to engine: {engine_type}")
            return engine_type
            
        except Exception as e:
            logger.error(f"Task routing failed: {e}")
            # Fallback to default routing
            return self._fallback_route_task(task)
    
    async def manage_workflow_state(self, content_id: str, new_state: WorkflowState) -> None:
        """
        Manage workflow state transitions with validation and tracking.
        
        Args:
            content_id: ID of the content item
            new_state: New workflow state to transition to
        """
        try:
            database = get_database()
            
            # Get current content item
            content_doc = await database.content_items.find_one({"_id": content_id})
            if not content_doc:
                raise ValidationError(f"Content item not found: {content_id}")
            
            content_item = ContentItem(**content_doc)
            current_state = content_item.workflow_state
            
            # Validate state transition
            if not self._is_valid_state_transition(current_state, new_state):
                raise ValidationError(f"Invalid state transition: {current_state} -> {new_state}")
            
            # Update content item state
            await database.content_items.update_one(
                {"_id": content_id},
                {
                    "$set": {
                        "workflow_state": new_state,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Add processing step to history
            content_item.add_processing_step(
                engine="orchestrator",
                operation="state_transition",
                from_state=current_state,
                to_state=new_state
            )
            
            # Update processing history
            await database.content_items.update_one(
                {"_id": content_id},
                {
                    "$set": {
                        "content_metadata.processing_history": [
                            step.dict() for step in content_item.content_metadata.processing_history
                        ]
                    }
                }
            )
            
            logger.info(f"Workflow state updated: {content_id} {current_state} -> {new_state}")
            
        except Exception as e:
            logger.error(f"Workflow state management failed: {e}")
            raise
    
    async def handle_engine_error(self, engine: str, error: Exception) -> Dict[str, Any]:
        """
        Handle engine errors with graceful degradation and recovery strategies.
        
        Args:
            engine: Name of the engine that failed
            error: The exception that occurred
            
        Returns:
            Error response with recovery actions
        """
        try:
            logger.error(f"Engine error in {engine}: {error}")
            
            # Update system health
            self.system_health.add_service_status(engine, "unhealthy")
            
            # Determine error type and recovery strategy
            error_code = self._classify_error(error)
            recovery_strategy = await self._determine_recovery_strategy(engine, error_code)
            
            # Execute recovery actions
            recovery_result = await self._execute_recovery_strategy(engine, recovery_strategy)
            
            # Create error response
            error_response = {
                "engine": engine,
                "error_code": error_code,
                "error_message": str(error),
                "recovery_strategy": recovery_strategy,
                "recovery_result": recovery_result,
                "timestamp": datetime.utcnow(),
                "system_status": self.system_health.status
            }
            
            return error_response
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return {
                "engine": engine,
                "error_code": ErrorCode.ENGINE_ERROR,
                "error_message": f"Critical error handling failure: {e}",
                "recovery_strategy": "manual_intervention_required",
                "timestamp": datetime.utcnow()
            }
    
    async def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        # Update health metrics
        await self._update_health_metrics()
        return self.system_health
    
    # Private helper methods
    
    async def _validate_workflow_request(self, request: WorkflowRequest) -> ValidationResult:
        """Validate workflow request parameters."""
        errors = []
        warnings = []
        
        # Check required fields
        if not request.operation:
            errors.append("Operation is required")
        
        if not request.user_id:
            errors.append("User ID is required")
        
        # Validate content_id if provided
        if request.content_id:
            database = get_database()
            content_doc = await database.content_items.find_one({"_id": request.content_id})
            if not content_doc:
                errors.append(f"Content item not found: {request.content_id}")
        
        # Check engine availability
        for engine_type in request.target_engines:
            if engine_type not in self.engine_capabilities:
                errors.append(f"Unknown engine type: {engine_type}")
            elif not self.engine_capabilities[engine_type].availability:
                warnings.append(f"Engine {engine_type} is currently unavailable")
        
        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult.success()
    
    async def _plan_workflow_with_llm(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Use LLM to create an intelligent workflow plan."""
        if not self.gemini_client:
            return await self._fallback_workflow_plan(request)
        
        try:
            planning_prompt = self._build_planning_prompt(request)
            response = await self._query_gemini(planning_prompt)
            workflow_plan = self._parse_planning_response(response)
            
            # Validate and enhance the plan
            validated_plan = await self._validate_workflow_plan(workflow_plan, request)
            return validated_plan
            
        except Exception as e:
            logger.warning(f"LLM workflow planning failed, using fallback: {e}")
            return await self._fallback_workflow_plan(request)
    
    def _build_planning_prompt(self, request: WorkflowRequest) -> str:
        """Build prompt for LLM workflow planning."""
        capabilities_desc = "\n".join([
            f"- {engine.value}: {', '.join(cap.operations)}"
            for engine, cap in self.engine_capabilities.items()
            if cap.availability
        ])
        
        prompt = f"""
        You are an AI workflow orchestrator for a content management platform. 
        Plan an optimal workflow for the following request:
        
        Operation: {request.operation}
        Parameters: {json.dumps(request.parameters, indent=2)}
        Priority: {request.priority}
        
        Available engines and their capabilities:
        {capabilities_desc}
        
        Create a workflow plan that:
        1. Breaks down the operation into specific tasks
        2. Assigns each task to the most appropriate engine
        3. Defines dependencies between tasks
        4. Estimates resource requirements
        
        Respond with a JSON object containing:
        {{
            "tasks": [
                {{
                    "id": "task_1",
                    "engine": "text_intelligence",
                    "operation": "generate",
                    "parameters": {{}},
                    "depends_on": [],
                    "estimated_time": 5,
                    "estimated_cost": 0.01
                }}
            ],
            "total_estimated_time": 10,
            "total_estimated_cost": 0.05,
            "workflow_description": "Brief description of the workflow"
        }}
        """
        
        return prompt
    
    async def _query_gemini(self, prompt: str) -> str:
        """Query Gemini LLM with error handling and retries."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.gemini_client.generate_content, prompt
                )
                return response.text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Gemini query attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"All Gemini query attempts failed: {e}")
                    raise
    
    def _parse_planning_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM planning response into structured workflow plan."""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            workflow_plan = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['tasks', 'total_estimated_time', 'total_estimated_cost']
            for field in required_fields:
                if field not in workflow_plan:
                    raise ValueError(f"Missing required field: {field}")
            
            return workflow_plan
            
        except Exception as e:
            logger.error(f"Failed to parse planning response: {e}")
            raise ValidationError(f"Invalid workflow plan format: {e}")
    
    async def _fallback_workflow_plan(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Create a simple workflow plan when LLM is unavailable."""
        # Simple rule-based planning
        engine_type = await self.route_task({
            "operation": request.operation,
            "parameters": request.parameters
        })
        
        capability = self.engine_capabilities.get(engine_type)
        if not capability:
            raise ValidationError(f"No capability found for engine: {engine_type}")
        
        task = {
            "id": f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "engine": engine_type.value,
            "operation": request.operation,
            "parameters": request.parameters,
            "depends_on": [],
            "estimated_time": capability.average_processing_time,
            "estimated_cost": capability.cost_per_operation
        }
        
        return {
            "tasks": [task],
            "total_estimated_time": capability.average_processing_time,
            "total_estimated_cost": capability.cost_per_operation,
            "workflow_description": f"Simple {request.operation} workflow"
        }
    
    async def _create_workflow_jobs(
        self, 
        workflow_plan: Dict[str, Any], 
        request: WorkflowRequest,
        workflow_id: str
    ) -> List[str]:
        """Create async jobs based on workflow plan."""
        database = get_database()
        job_ids = []
        
        for task in workflow_plan["tasks"]:
            # Map engine name to JobType
            job_type = self._map_engine_to_job_type(task["engine"])
            
            job = AsyncJob(
                job_type=job_type,
                content_id=request.content_id,
                engine=task["engine"],
                operation=task["operation"],
                parameters=task["parameters"],
                priority=int(request.priority),
                depends_on=task.get("depends_on", []),
                workflow_id=workflow_id,
                user_id=request.user_id
            )
            
            # Insert job into database
            result = await database.async_jobs.insert_one(job.dict())
            job_id = str(result.inserted_id)
            job_ids.append(job_id)
        
        return job_ids
    
    def _map_engine_to_job_type(self, engine: str) -> JobType:
        """Map engine name to JobType enum."""
        mapping = {
            "text_intelligence": JobType.CONTENT_GENERATION,
            "creative_assistant": JobType.CREATIVE_ASSISTANCE,
            "social_media_planner": JobType.SOCIAL_MEDIA_OPTIMIZATION,
            "discovery_analytics": JobType.ANALYTICS_PROCESSING,
            "image_generation": JobType.MEDIA_GENERATION,
            "audio_generation": JobType.MEDIA_GENERATION,
            "video_pipeline": JobType.MEDIA_GENERATION
        }
        return mapping.get(engine, JobType.CONTENT_GENERATION)
    
    def _build_routing_prompt(self, task: Dict[str, Any]) -> str:
        """Build prompt for task routing."""
        return f"""
        Route this task to the most appropriate engine:
        
        Task: {json.dumps(task, indent=2)}
        
        Available engines:
        {list(self.engine_capabilities.keys())}
        
        Respond with just the engine name.
        """
    
    def _parse_routing_response(self, response: str) -> EngineType:
        """Parse routing response to engine type."""
        response = response.strip().lower()
        
        for engine_type in EngineType:
            if engine_type.value in response:
                return engine_type
        
        # Default fallback
        return EngineType.TEXT_INTELLIGENCE
    
    def _fallback_route_task(self, task: Dict[str, Any]) -> EngineType:
        """Fallback task routing using simple rules."""
        operation = task.get("operation", "").lower()
        
        if any(word in operation for word in ["generate", "create", "write"]):
            return EngineType.TEXT_INTELLIGENCE
        elif any(word in operation for word in ["suggest", "idea", "creative"]):
            return EngineType.CREATIVE_ASSISTANT
        elif any(word in operation for word in ["social", "hashtag", "optimize"]):
            return EngineType.SOCIAL_MEDIA_PLANNER
        elif any(word in operation for word in ["analyze", "tag", "metric"]):
            return EngineType.DISCOVERY_ANALYTICS
        elif any(word in operation for word in ["image", "thumbnail", "poster"]):
            return EngineType.IMAGE_GENERATION
        elif any(word in operation for word in ["audio", "voice", "music"]):
            return EngineType.AUDIO_GENERATION
        elif any(word in operation for word in ["video", "movie"]):
            return EngineType.VIDEO_PIPELINE
        
        return EngineType.TEXT_INTELLIGENCE
    
    def _is_valid_state_transition(self, current: WorkflowState, new: WorkflowState) -> bool:
        """Validate workflow state transitions."""
        valid_transitions = {
            WorkflowState.DISCOVER: [WorkflowState.CREATE],
            WorkflowState.CREATE: [WorkflowState.TRANSFORM, WorkflowState.PLAN],
            WorkflowState.TRANSFORM: [WorkflowState.PLAN, WorkflowState.CREATE],
            WorkflowState.PLAN: [WorkflowState.PUBLISH, WorkflowState.TRANSFORM],
            WorkflowState.PUBLISH: [WorkflowState.ANALYZE],
            WorkflowState.ANALYZE: [WorkflowState.IMPROVE, WorkflowState.DISCOVER],
            WorkflowState.IMPROVE: [WorkflowState.TRANSFORM, WorkflowState.CREATE]
        }
        
        return new in valid_transitions.get(current, [])
    
    def _classify_error(self, error: Exception) -> ErrorCode:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return ErrorCode.TIMEOUT
        elif "rate limit" in error_str:
            return ErrorCode.RATE_LIMIT_EXCEEDED
        elif "usage limit" in error_str:
            return ErrorCode.USAGE_LIMIT_EXCEEDED
        elif "authentication" in error_str:
            return ErrorCode.AUTHENTICATION_ERROR
        elif "authorization" in error_str:
            return ErrorCode.AUTHORIZATION_ERROR
        elif "validation" in error_str:
            return ErrorCode.VALIDATION_ERROR
        elif "network" in error_str or "connection" in error_str:
            return ErrorCode.NETWORK_ERROR
        elif "service unavailable" in error_str:
            return ErrorCode.SERVICE_UNAVAILABLE
        else:
            return ErrorCode.ENGINE_ERROR
    
    async def _determine_recovery_strategy(self, engine: str, error_code: ErrorCode) -> str:
        """Determine recovery strategy based on error type."""
        strategies = {
            ErrorCode.TIMEOUT: "retry_with_backoff",
            ErrorCode.RATE_LIMIT_EXCEEDED: "delay_and_retry",
            ErrorCode.USAGE_LIMIT_EXCEEDED: "queue_for_later",
            ErrorCode.NETWORK_ERROR: "retry_with_backoff",
            ErrorCode.SERVICE_UNAVAILABLE: "fallback_engine",
            ErrorCode.ENGINE_ERROR: "fallback_engine",
            ErrorCode.VALIDATION_ERROR: "fix_and_retry",
            ErrorCode.AUTHENTICATION_ERROR: "refresh_credentials",
            ErrorCode.AUTHORIZATION_ERROR: "escalate_permissions"
        }
        
        return strategies.get(error_code, "manual_intervention")
    
    async def _execute_recovery_strategy(self, engine: str, strategy: str) -> Dict[str, Any]:
        """Execute the determined recovery strategy."""
        try:
            if strategy == "retry_with_backoff":
                return {"action": "scheduled_retry", "delay_seconds": 60}
            elif strategy == "delay_and_retry":
                return {"action": "delayed_retry", "delay_seconds": 300}
            elif strategy == "queue_for_later":
                return {"action": "queued", "retry_after": "usage_reset"}
            elif strategy == "fallback_engine":
                fallback = await self._find_fallback_engine(engine)
                return {"action": "fallback", "fallback_engine": fallback}
            elif strategy == "fix_and_retry":
                return {"action": "validation_fix_required", "manual_review": True}
            else:
                return {"action": "manual_intervention_required"}
                
        except Exception as e:
            logger.error(f"Recovery strategy execution failed: {e}")
            return {"action": "recovery_failed", "error": str(e)}
    
    async def _find_fallback_engine(self, failed_engine: str) -> Optional[str]:
        """Find a suitable fallback engine."""
        # Simple fallback mapping
        fallbacks = {
            "text_intelligence": "creative_assistant",
            "creative_assistant": "text_intelligence",
            "image_generation": None,  # No fallback for media generation
            "audio_generation": None,
            "video_pipeline": None
        }
        
        return fallbacks.get(failed_engine)
    
    async def _calculate_cost_estimate(self, workflow_plan: Dict[str, Any]) -> float:
        """Calculate estimated cost for workflow execution."""
        return workflow_plan.get("total_estimated_cost", 0.0)
    
    async def _estimate_completion_time(self, workflow_plan: Dict[str, Any]) -> datetime:
        """Estimate workflow completion time."""
        estimated_seconds = workflow_plan.get("total_estimated_time", 60)
        return datetime.utcnow().timestamp() + estimated_seconds
    
    async def _validate_workflow_plan(
        self, 
        workflow_plan: Dict[str, Any], 
        request: WorkflowRequest
    ) -> Dict[str, Any]:
        """Validate and enhance workflow plan."""
        # Add validation logic here
        # For now, return the plan as-is
        return workflow_plan
    
    async def _update_health_metrics(self):
        """Update system health metrics."""
        try:
            # Check engine availability
            healthy_engines = 0
            total_engines = len(self.engine_capabilities)
            
            for engine_type, capability in self.engine_capabilities.items():
                if capability.availability:
                    healthy_engines += 1
                    self.system_health.add_service_status(engine_type.value, "healthy")
                else:
                    self.system_health.add_service_status(engine_type.value, "unhealthy")
            
            # Update overall health
            health_ratio = healthy_engines / total_engines if total_engines > 0 else 0
            
            if health_ratio >= 0.8:
                self.system_health.status = "healthy"
            elif health_ratio >= 0.5:
                self.system_health.status = "degraded"
            else:
                self.system_health.status = "unhealthy"
            
            # Add metrics
            self.system_health.add_metric("engine_availability_ratio", health_ratio)
            self.system_health.add_metric("active_workflows", len(self.active_workflows))
            
        except Exception as e:
            logger.error(f"Health metrics update failed: {e}")
            self.system_health.status = "unhealthy"
            self.system_health.errors.append(str(e))
    
    async def _handle_orchestration_error(self, request: WorkflowRequest, error: Exception):
        """Handle orchestration errors with logging and cleanup."""
        logger.error(f"Orchestration error for {request.operation}: {error}")
        
        # Update system health
        self.system_health.add_service_status("orchestrator", "degraded")
        
        # Could add more error handling logic here
        # such as notifying administrators, cleaning up resources, etc.
    
    async def complete_workflow(
        self,
        workflow_id: str,
        final_status: str = "completed",
        results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete a workflow and perform cleanup.
        
        Args:
            workflow_id: ID of the workflow to complete
            final_status: Final status (completed, failed, cancelled)
            results: Optional workflow results
            
        Returns:
            Workflow completion summary
        """
        try:
            logger.info(f"Completing workflow {workflow_id} with status: {final_status}")
            
            # Get workflow execution
            database = get_database()
            workflow_doc = await database.workflow_executions.find_one({"_id": workflow_id})
            
            if not workflow_doc:
                raise ValidationError(f"Workflow {workflow_id} not found")
            
            workflow = WorkflowExecution(**workflow_doc)
            
            # Update workflow status
            workflow.status = final_status
            workflow.completed_at = datetime.utcnow()
            
            # Calculate total cost and tokens
            total_cost = 0.0
            total_tokens = 0
            
            # Aggregate results from all jobs
            job_results = []
            for job_id in workflow.job_ids:
                job_doc = await database.async_jobs.find_one({"_id": job_id})
                if job_doc:
                    job = AsyncJob(**job_doc)
                    if job.result:
                        job_results.append(job.result)
                        if isinstance(job.result, dict):
                            total_cost += job.result.get("cost", 0.0)
                            total_tokens += job.result.get("tokens_used", 0)
            
            # Store aggregated results
            workflow.final_result = {
                "status": final_status,
                "job_results": job_results,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "results": results or {}
            }
            
            # Update workflow in database
            await database.workflow_executions.update_one(
                {"_id": workflow_id},
                {"$set": workflow.dict()}
            )
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Send completion notification
            await self._send_workflow_notification(workflow, final_status)
            
            # Create completion summary
            summary = {
                "workflow_id": workflow_id,
                "status": final_status,
                "duration_seconds": (
                    workflow.completed_at - workflow.created_at
                ).total_seconds() if workflow.completed_at else 0,
                "jobs_completed": len([j for j in job_results if j]),
                "total_jobs": len(workflow.job_ids),
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "results": results
            }
            
            logger.info(f"Workflow {workflow_id} completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to complete workflow {workflow_id}: {e}")
            raise OrchestrationError(f"Workflow completion failed: {e}")
    
    async def transition_workflow_state(
        self,
        workflow_id: str,
        content_id: str,
        new_state: WorkflowState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Transition workflow state with validation and tracking.
        
        Args:
            workflow_id: ID of the workflow
            content_id: ID of the content item
            new_state: New workflow state to transition to
            metadata: Optional metadata about the transition
        """
        try:
            logger.info(f"Transitioning workflow {workflow_id} to state: {new_state}")
            
            # Validate and update content item state
            await self.manage_workflow_state(content_id, new_state)
            
            # Update workflow execution record
            database = get_database()
            await database.workflow_executions.update_one(
                {"_id": workflow_id},
                {
                    "$set": {
                        "current_state": new_state,
                        "updated_at": datetime.utcnow()
                    },
                    "$push": {
                        "state_history": {
                            "state": new_state,
                            "timestamp": datetime.utcnow(),
                            "metadata": metadata or {}
                        }
                    }
                }
            )
            
            # Update active workflow tracking
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].current_state = new_state
            
            logger.info(f"Workflow state transition completed: {workflow_id} -> {new_state}")
            
        except Exception as e:
            logger.error(f"Workflow state transition failed: {e}")
            raise OrchestrationError(f"State transition failed: {e}")
    
    async def monitor_workflow_progress(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Monitor workflow progress and return current status.
        
        Args:
            workflow_id: ID of the workflow to monitor
            
        Returns:
            Dictionary with workflow progress information
        """
        try:
            database = get_database()
            
            # Get workflow execution
            workflow_doc = await database.workflow_executions.find_one({"_id": workflow_id})
            if not workflow_doc:
                raise ValidationError(f"Workflow {workflow_id} not found")
            
            workflow = WorkflowExecution(**workflow_doc)
            
            # Get job statuses
            job_statuses = {}
            completed_jobs = 0
            failed_jobs = 0
            running_jobs = 0
            
            for job_id in workflow.job_ids:
                job_doc = await database.async_jobs.find_one({"_id": job_id})
                if job_doc:
                    job = AsyncJob(**job_doc)
                    job_statuses[job_id] = job.status
                    
                    if job.status == JobStatus.COMPLETED:
                        completed_jobs += 1
                    elif job.status == JobStatus.FAILED:
                        failed_jobs += 1
                    elif job.status == JobStatus.RUNNING:
                        running_jobs += 1
            
            # Calculate progress percentage
            total_jobs = len(workflow.job_ids)
            progress_percentage = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
            
            # Determine overall status
            if failed_jobs > 0 and completed_jobs + failed_jobs == total_jobs:
                overall_status = "failed"
            elif completed_jobs == total_jobs:
                overall_status = "completed"
            elif running_jobs > 0:
                overall_status = "running"
            else:
                overall_status = "queued"
            
            # Calculate estimated time remaining
            if workflow.created_at and completed_jobs > 0:
                elapsed_time = (datetime.utcnow() - workflow.created_at).total_seconds()
                avg_time_per_job = elapsed_time / completed_jobs
                remaining_jobs = total_jobs - completed_jobs
                estimated_remaining = avg_time_per_job * remaining_jobs
            else:
                estimated_remaining = None
            
            progress_info = {
                "workflow_id": workflow_id,
                "status": overall_status,
                "progress_percentage": round(progress_percentage, 2),
                "jobs": {
                    "total": total_jobs,
                    "completed": completed_jobs,
                    "failed": failed_jobs,
                    "running": running_jobs,
                    "queued": total_jobs - completed_jobs - failed_jobs - running_jobs
                },
                "job_statuses": job_statuses,
                "estimated_remaining_seconds": estimated_remaining,
                "created_at": workflow.created_at,
                "updated_at": workflow.updated_at
            }
            
            return progress_info
            
        except Exception as e:
            logger.error(f"Failed to monitor workflow progress: {e}")
            raise OrchestrationError(f"Progress monitoring failed: {e}")
    
    async def get_workflow_analytics(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Get analytics and metrics for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Dictionary with workflow analytics
        """
        try:
            database = get_database()
            
            # Get workflow execution
            workflow_doc = await database.workflow_executions.find_one({"_id": workflow_id})
            if not workflow_doc:
                raise ValidationError(f"Workflow {workflow_id} not found")
            
            workflow = WorkflowExecution(**workflow_doc)
            
            # Collect job analytics
            job_analytics = []
            total_cost = 0.0
            total_tokens = 0
            engine_usage = {}
            
            for job_id in workflow.job_ids:
                job_doc = await database.async_jobs.find_one({"_id": job_id})
                if job_doc:
                    job = AsyncJob(**job_doc)
                    
                    # Calculate job duration
                    if job.started_at and job.completed_at:
                        duration = (job.completed_at - job.started_at).total_seconds()
                    else:
                        duration = None
                    
                    # Extract cost and tokens from result
                    cost = 0.0
                    tokens = 0
                    if job.result and isinstance(job.result, dict):
                        cost = job.result.get("cost", 0.0)
                        tokens = job.result.get("tokens_used", 0)
                    
                    total_cost += cost
                    total_tokens += tokens
                    
                    # Track engine usage
                    engine = job.engine
                    if engine not in engine_usage:
                        engine_usage[engine] = {"count": 0, "cost": 0.0, "tokens": 0}
                    engine_usage[engine]["count"] += 1
                    engine_usage[engine]["cost"] += cost
                    engine_usage[engine]["tokens"] += tokens
                    
                    job_analytics.append({
                        "job_id": job_id,
                        "engine": engine,
                        "operation": job.operation,
                        "status": job.status,
                        "duration_seconds": duration,
                        "cost": cost,
                        "tokens_used": tokens,
                        "retry_count": job.retry_count
                    })
            
            # Calculate workflow duration
            if workflow.completed_at:
                workflow_duration = (workflow.completed_at - workflow.created_at).total_seconds()
            else:
                workflow_duration = (datetime.utcnow() - workflow.created_at).total_seconds()
            
            analytics = {
                "workflow_id": workflow_id,
                "workflow_name": workflow.workflow_name,
                "status": workflow.status,
                "duration_seconds": workflow_duration,
                "total_jobs": len(workflow.job_ids),
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "engine_usage": engine_usage,
                "job_analytics": job_analytics,
                "created_at": workflow.created_at,
                "completed_at": workflow.completed_at,
                "user_id": workflow.user_id
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get workflow analytics: {e}")
            raise OrchestrationError(f"Analytics retrieval failed: {e}")
    
    async def _send_workflow_notification(
        self,
        workflow: WorkflowExecution,
        status: str
    ) -> None:
        """
        Send notification about workflow completion.
        
        Args:
            workflow: Workflow execution object
            status: Final workflow status
        """
        try:
            # Import notification service
            from app.services.notifications import NotificationService
            
            notification_service = NotificationService()
            
            # Create notification message
            if status == "completed":
                message = f"Workflow '{workflow.workflow_name}' completed successfully"
                notification_type = "success"
            elif status == "failed":
                message = f"Workflow '{workflow.workflow_name}' failed"
                notification_type = "error"
            else:
                message = f"Workflow '{workflow.workflow_name}' status: {status}"
                notification_type = "info"
            
            # Send notification
            await notification_service.send_notification(
                user_id=workflow.user_id,
                notification_type=notification_type,
                title="Workflow Update",
                message=message,
                data={
                    "workflow_id": str(workflow.id),
                    "workflow_name": workflow.workflow_name,
                    "status": status,
                    "job_count": len(workflow.job_ids)
                }
            )
            
            logger.info(f"Notification sent for workflow {workflow.id}")
            
        except Exception as e:
            # Don't fail workflow completion if notification fails
            logger.warning(f"Failed to send workflow notification: {e}")

"""
API endpoints for AI Orchestrator functionality.

This module provides REST API endpoints for workflow orchestration,
task routing, and system health monitoring.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from app.ai.orchestrator import (
    AIOrchestrator, WorkflowRequest, WorkflowResponse, TaskPriority
)
from app.models.base import WorkflowState, EngineType, SystemHealth
from app.core.exceptions import (
    ValidationError, OrchestrationError, EngineError
)

router = APIRouter()

# Global orchestrator instance (in production, this would be dependency injected)
orchestrator = AIOrchestrator()


@router.post("/workflow", response_model=WorkflowResponse)
async def orchestrate_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """
    Orchestrate a complex workflow using AI reasoning.
    
    This endpoint accepts a workflow request and uses the AI orchestrator
    to decompose it into tasks, route them to appropriate engines, and
    manage the execution lifecycle.
    """
    try:
        response = await orchestrator.orchestrate_workflow(request)
        return response
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Validation Error",
                "message": e.message,
                "details": e.details
            }
        )
    except OrchestrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Orchestration Error",
                "message": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal Server Error",
                "message": str(e)
            }
        )


@router.post("/route-task")
async def route_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route a single task to the most appropriate engine.
    
    This endpoint analyzes a task description and determines
    which engine should handle it based on AI reasoning.
    """
    try:
        engine_type = await orchestrator.route_task(task)
        
        return {
            "task": task,
            "recommended_engine": engine_type.value,
            "engine_capabilities": orchestrator.engine_capabilities[engine_type].dict()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Task Routing Error",
                "message": str(e)
            }
        )


@router.put("/workflow-state/{content_id}")
async def update_workflow_state(
    content_id: str, 
    new_state: WorkflowState
) -> Dict[str, Any]:
    """
    Update the workflow state of a content item.
    
    This endpoint manages workflow state transitions with validation
    to ensure only valid state changes are allowed.
    """
    try:
        await orchestrator.manage_workflow_state(content_id, new_state)
        
        return {
            "content_id": content_id,
            "new_state": new_state.value,
            "message": f"Workflow state updated to {new_state.value}"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid State Transition",
                "message": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "State Management Error",
                "message": str(e)
            }
        )


@router.get("/health", response_model=SystemHealth)
async def get_system_health() -> SystemHealth:
    """
    Get current system health status.
    
    This endpoint provides information about the health of all engines,
    system metrics, and overall orchestrator status.
    """
    try:
        health = await orchestrator.get_system_health()
        return health
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Health Check Error",
                "message": str(e)
            }
        )


@router.get("/engines")
async def get_engine_capabilities() -> Dict[str, Any]:
    """
    Get information about all available engines and their capabilities.
    
    This endpoint returns detailed information about each engine,
    including supported operations, costs, and availability.
    """
    try:
        capabilities = {}
        
        for engine_type, capability in orchestrator.engine_capabilities.items():
            capabilities[engine_type.value] = {
                "operations": capability.operations,
                "input_types": capability.input_types,
                "output_types": capability.output_types,
                "cost_per_operation": capability.cost_per_operation,
                "average_processing_time": capability.average_processing_time,
                "availability": capability.availability,
                "error_rate": capability.error_rate
            }
        
        return {
            "engines": capabilities,
            "total_engines": len(capabilities),
            "available_engines": sum(1 for cap in orchestrator.engine_capabilities.values() if cap.availability)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Engine Information Error",
                "message": str(e)
            }
        )


@router.post("/handle-error")
async def handle_engine_error(
    engine: str, 
    error_details: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle an engine error with recovery strategies.
    
    This endpoint processes engine errors and determines appropriate
    recovery actions based on error type and system state.
    """
    try:
        # Create a mock exception from the error details
        error_message = error_details.get("message", "Unknown error")
        mock_error = Exception(error_message)
        
        error_response = await orchestrator.handle_engine_error(engine, mock_error)
        
        return error_response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Error Handling Failed",
                "message": str(e)
            }
        )


@router.get("/workflows/active")
async def get_active_workflows() -> Dict[str, Any]:
    """
    Get information about currently active workflows.
    
    This endpoint returns details about workflows that are currently
    being processed by the orchestrator.
    """
    try:
        active_workflows = {}
        
        for workflow_id, workflow in orchestrator.active_workflows.items():
            active_workflows[workflow_id] = {
                "workflow_name": workflow.workflow_name,
                "description": workflow.description,
                "status": workflow.status,
                "job_ids": workflow.job_ids,
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "user_id": workflow.user_id
            }
        
        return {
            "active_workflows": active_workflows,
            "total_active": len(active_workflows)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Workflow Information Error",
                "message": str(e)
            }
        )


# Example request models for documentation
class TaskRoutingRequest(dict):
    """Example task routing request."""
    pass


class ErrorHandlingRequest(dict):
    """Example error handling request."""
    pass


# Add example schemas for OpenAPI documentation
@router.post("/workflow", 
    response_model=WorkflowResponse,
    summary="Orchestrate AI Workflow",
    description="""
    Orchestrate a complex AI workflow by decomposing it into tasks and routing them to appropriate engines.
    
    The orchestrator uses LLM reasoning to:
    - Analyze the workflow request
    - Break it down into specific tasks
    - Route tasks to the most suitable engines
    - Manage dependencies and execution order
    - Track costs and completion estimates
    
    Example request:
    ```json
    {
        "operation": "create_blog_post_with_social_media",
        "parameters": {
            "topic": "AI in Content Creation",
            "length": 1500,
            "target_platforms": ["twitter", "linkedin"]
        },
        "user_id": "user_123",
        "priority": 2
    }
    ```
    """)
async def orchestrate_workflow_documented(request: WorkflowRequest) -> WorkflowResponse:
    """This is the same as orchestrate_workflow but with enhanced documentation."""
    return await orchestrate_workflow(request)
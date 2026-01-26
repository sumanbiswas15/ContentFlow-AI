"""
Demonstration script for the AI Orchestrator.

This script shows how to use the AI Orchestrator for workflow management,
task routing, and system health monitoring.
"""

import asyncio
from app.ai.orchestrator import AIOrchestrator, WorkflowRequest, TaskPriority
from app.models.base import WorkflowState, EngineType


async def demo_orchestrator():
    """Demonstrate AI Orchestrator functionality."""
    print("🤖 ContentFlow AI Orchestrator Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    print("\n1. Initializing AI Orchestrator...")
    orchestrator = AIOrchestrator()
    print(f"✅ Orchestrator initialized with {len(orchestrator.engine_capabilities)} engines")
    
    # Show engine capabilities
    print("\n2. Available Engines:")
    for engine_type, capability in orchestrator.engine_capabilities.items():
        print(f"   • {engine_type.value}: {', '.join(capability.operations[:3])}...")
        print(f"     Cost: ${capability.cost_per_operation}, Time: {capability.average_processing_time}s")
    
    # Demonstrate task routing
    print("\n3. Task Routing Examples:")
    test_tasks = [
        {"operation": "generate_blog_post", "parameters": {"topic": "AI"}},
        {"operation": "suggest_creative_ideas", "parameters": {"theme": "marketing"}},
        {"operation": "optimize_for_twitter", "parameters": {"content": "Hello world"}},
        {"operation": "analyze_engagement", "parameters": {"content_id": "123"}},
        {"operation": "create_thumbnail", "parameters": {"style": "modern"}}
    ]
    
    for task in test_tasks:
        engine = await orchestrator.route_task(task)
        print(f"   • '{task['operation']}' → {engine.value}")
    
    # Demonstrate workflow state transitions
    print("\n4. Workflow State Transitions:")
    valid_transitions = [
        (WorkflowState.DISCOVER, WorkflowState.CREATE),
        (WorkflowState.CREATE, WorkflowState.TRANSFORM),
        (WorkflowState.TRANSFORM, WorkflowState.PLAN),
        (WorkflowState.PLAN, WorkflowState.PUBLISH),
        (WorkflowState.PUBLISH, WorkflowState.ANALYZE)
    ]
    
    for current, new in valid_transitions:
        is_valid = orchestrator._is_valid_state_transition(current, new)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {current.value} → {new.value}")
    
    # Demonstrate fallback workflow planning
    print("\n5. Fallback Workflow Planning:")
    sample_request = WorkflowRequest(
        operation="create_social_media_campaign",
        parameters={
            "topic": "Product Launch",
            "platforms": ["twitter", "instagram", "linkedin"],
            "content_types": ["text", "image"]
        },
        user_id="demo_user",
        priority=TaskPriority.HIGH
    )
    
    workflow_plan = await orchestrator._fallback_workflow_plan(sample_request)
    print(f"   • Generated workflow with {len(workflow_plan['tasks'])} tasks")
    print(f"   • Estimated time: {workflow_plan['total_estimated_time']}s")
    print(f"   • Estimated cost: ${workflow_plan['total_estimated_cost']}")
    
    for i, task in enumerate(workflow_plan['tasks'], 1):
        print(f"     Task {i}: {task['operation']} via {task['engine']}")
    
    # Demonstrate error handling
    print("\n6. Error Handling Examples:")
    test_errors = [
        Exception("Request timeout occurred"),
        Exception("Rate limit exceeded"),
        Exception("Validation failed for input"),
        Exception("Service unavailable")
    ]
    
    for error in test_errors:
        error_code = orchestrator._classify_error(error)
        strategy = await orchestrator._determine_recovery_strategy("test_engine", error_code)
        print(f"   • '{str(error)[:30]}...' → {error_code.value} → {strategy}")
    
    # Show system health
    print("\n7. System Health:")
    health = await orchestrator.get_system_health()
    print(f"   • Overall Status: {health.status}")
    print(f"   • Active Workflows: {health.metrics.get('active_workflows', 0)}")
    print(f"   • Engine Availability: {health.metrics.get('engine_availability_ratio', 0):.1%}")
    
    healthy_services = sum(1 for status in health.services.values() if status == "healthy")
    total_services = len(health.services)
    print(f"   • Healthy Services: {healthy_services}/{total_services}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nThe AI Orchestrator is ready to coordinate complex workflows")
    print("across multiple specialized engines with intelligent routing,")
    print("state management, and error recovery capabilities.")


if __name__ == "__main__":
    asyncio.run(demo_orchestrator())
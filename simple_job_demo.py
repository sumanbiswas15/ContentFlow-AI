#!/usr/bin/env python3
"""
Simple demo of ContentFlow AI async job processing system.

This script demonstrates the core job processing functionality
without complex service integration.
"""

import asyncio
import logging
from datetime import datetime

from app.models.jobs import AsyncJob, JobResult
from app.models.base import JobType, JobStatus
from app.services.job_handlers import handle_content_generation, handle_content_transformation, handle_creative_assistance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_job_models():
    """Demonstrate job model functionality."""
    print("\n" + "="*60)
    print("DEMO: Job Models and State Management")
    print("="*60)
    
    # Create a sample job
    job = AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_blog",
        parameters={
            "prompt": "Write about the future of AI in healthcare",
            "content_type": "blog",
            "max_length": 1000
        },
        user_id="demo_user",
        priority=3
    )
    
    print(f"✓ Created job: {job.id}")
    print(f"  - Type: {job.job_type}")
    print(f"  - Engine: {job.engine}")
    print(f"  - Operation: {job.operation}")
    print(f"  - Status: {job.status}")
    print(f"  - Priority: {job.priority}")
    print(f"  - Created: {job.created_at}")
    
    # Demonstrate job state transitions
    print("\n✓ Job state transitions:")
    
    # Start execution
    job.start_execution()
    print(f"  - Started: {job.status} at {job.started_at}")
    
    # Complete successfully
    job.complete_with_success(
        result_data={"content": "Generated blog content about AI in healthcare..."},
        tokens_used=250,
        cost=0.025
    )
    print(f"  - Completed: {job.status} at {job.completed_at}")
    print(f"  - Execution time: {job.execution_time_ms}ms")
    print(f"  - Tokens used: {job.result.tokens_used}")
    print(f"  - Cost: ${job.result.cost}")
    
    return job


async def demo_job_handlers():
    """Demonstrate job handler functionality."""
    print("\n" + "="*60)
    print("DEMO: Job Handlers")
    print("="*60)
    
    # Test content generation handler
    print("✓ Testing content generation handler:")
    gen_job = AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_blog",
        parameters={
            "prompt": "The impact of AI on modern education",
            "content_type": "blog",
            "max_length": 800
        },
        user_id="demo_user"
    )
    
    result = await handle_content_generation(gen_job)
    print(f"  - Generated content: {result.data['content'][:100]}...")
    print(f"  - Tokens used: {result.tokens_used}")
    print(f"  - Cost: ${result.cost}")
    
    # Test content transformation handler
    print("\n✓ Testing content transformation handler:")
    transform_job = AsyncJob(
        job_type=JobType.CONTENT_TRANSFORMATION,
        engine="text_intelligence",
        operation="summarize",
        parameters={
            "content": "This is a long article about artificial intelligence and its applications in various industries. AI has revolutionized healthcare, finance, education, and many other sectors. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions that were previously impossible for humans to achieve.",
            "transformation_type": "summarize",
            "target_length": 100
        },
        user_id="demo_user"
    )
    
    result = await handle_content_transformation(transform_job)
    print(f"  - Original length: {result.data['original_length']}")
    print(f"  - Transformed length: {result.data['transformed_length']}")
    print(f"  - Summary: {result.data['transformed_content'][:100]}...")
    
    # Test creative assistance handler
    print("\n✓ Testing creative assistance handler:")
    creative_job = AsyncJob(
        job_type=JobType.CREATIVE_ASSISTANCE,
        engine="creative_assistant",
        operation="brainstorm",
        parameters={
            "session_type": "ideation",
            "context": "Marketing campaign for a new AI product",
            "request": "Generate creative campaign ideas for launching an AI-powered writing assistant"
        },
        user_id="demo_user"
    )
    
    result = await handle_creative_assistance(creative_job)
    print(f"  - Session type: {result.data['session_type']}")
    print(f"  - Generated {result.data['suggestion_count']} suggestions:")
    for i, suggestion in enumerate(result.data['suggestions'], 1):
        print(f"    {i}. {suggestion}")


async def demo_retry_logic():
    """Demonstrate retry logic and error handling."""
    print("\n" + "="*60)
    print("DEMO: Retry Logic and Error Handling")
    print("="*60)
    
    # Create a job for retry demonstration
    job = AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_content",
        parameters={"prompt": "Test content for retry demo"},
        user_id="demo_user"
    )
    
    print(f"✓ Created job for retry demo: {job.id}")
    print(f"  - Initial retry count: {job.retry_count}")
    print(f"  - Max retries: {job.retry_config.max_retries}")
    print(f"  - Retry delay: {job.retry_config.retry_delay_seconds}s")
    print(f"  - Exponential backoff: {job.retry_config.exponential_backoff}")
    
    # Simulate multiple failures and retries
    for attempt in range(1, job.retry_config.max_retries + 2):
        print(f"\n  Attempt {attempt}:")
        
        job.start_execution()
        print(f"    - Started: {job.status}")
        
        # Simulate failure
        job.complete_with_error(f"Simulated error on attempt {attempt}", "PROCESSING_ERROR")
        print(f"    - Failed: {job.status}")
        print(f"    - Error: {job.last_error}")
        print(f"    - Should retry: {job.should_retry()}")
        
        if job.should_retry():
            # Calculate retry delay
            if job.retry_config.exponential_backoff:
                delay = job.retry_config.retry_delay_seconds * (2 ** job.retry_count)
            else:
                delay = job.retry_config.retry_delay_seconds
            
            print(f"    - Next retry delay: {delay}s")
            
            job.increment_retry()
            print(f"    - Retry count: {job.retry_count}")
            print(f"    - Status reset to: {job.status}")
        else:
            print(f"    - Max retries reached, giving up")
            break
    
    return job


async def demo_job_dependencies():
    """Demonstrate job dependencies."""
    print("\n" + "="*60)
    print("DEMO: Job Dependencies")
    print("="*60)
    
    # Create jobs with dependencies
    job1 = AsyncJob(
        job_type=JobType.CONTENT_GENERATION,
        engine="text_intelligence",
        operation="generate_article",
        parameters={"prompt": "AI in healthcare"},
        user_id="demo_user",
        priority=1
    )
    
    job2 = AsyncJob(
        job_type=JobType.CONTENT_TRANSFORMATION,
        engine="text_intelligence",
        operation="summarize",
        parameters={"content": "Article content"},
        user_id="demo_user",
        priority=2,
        depends_on=[str(job1.id)]  # Depends on job1
    )
    
    job3 = AsyncJob(
        job_type=JobType.SOCIAL_MEDIA_OPTIMIZATION,
        engine="social_media_planner",
        operation="optimize",
        parameters={"platform": "twitter"},
        user_id="demo_user",
        priority=3,
        depends_on=[str(job1.id), str(job2.id)]  # Depends on both job1 and job2
    )
    
    print(f"✓ Created job dependency chain:")
    print(f"  - Job 1 (content generation): {job1.id}")
    print(f"    Dependencies: {job1.depends_on}")
    print(f"  - Job 2 (summarization): {job2.id}")
    print(f"    Dependencies: {job2.depends_on}")
    print(f"  - Job 3 (social media): {job3.id}")
    print(f"    Dependencies: {job3.depends_on}")
    
    # Test dependency checking
    completed_jobs = []
    
    print(f"\n✓ Testing dependency resolution:")
    print(f"  - Job 1 ready (no deps): {job1.is_ready_to_execute(completed_jobs)}")
    print(f"  - Job 2 ready (needs job1): {job2.is_ready_to_execute(completed_jobs)}")
    print(f"  - Job 3 ready (needs job1,2): {job3.is_ready_to_execute(completed_jobs)}")
    
    # Complete job1
    completed_jobs.append(str(job1.id))
    print(f"\n  After completing job1:")
    print(f"  - Job 2 ready: {job2.is_ready_to_execute(completed_jobs)}")
    print(f"  - Job 3 ready: {job3.is_ready_to_execute(completed_jobs)}")
    
    # Complete job2
    completed_jobs.append(str(job2.id))
    print(f"\n  After completing job2:")
    print(f"  - Job 3 ready: {job3.is_ready_to_execute(completed_jobs)}")
    
    return [job1, job2, job3]


async def demo_job_priorities():
    """Demonstrate job priority handling."""
    print("\n" + "="*60)
    print("DEMO: Job Priority Handling")
    print("="*60)
    
    # Create jobs with different priorities
    jobs = []
    priorities = [10, 1, 5, 3, 7]  # 1 = highest, 10 = lowest
    
    for i, priority in enumerate(priorities):
        job = AsyncJob(
            job_type=JobType.CONTENT_GENERATION,
            engine="text_intelligence",
            operation="generate_content",
            parameters={"prompt": f"Priority {priority} content"},
            user_id="demo_user",
            priority=priority
        )
        jobs.append(job)
        print(f"✓ Created job {i+1} with priority {priority}: {job.id}")
    
    # Sort jobs by priority (as queue would do)
    sorted_jobs = sorted(jobs, key=lambda j: (j.priority, j.created_at))
    
    print(f"\n✓ Jobs sorted by priority (execution order):")
    for i, job in enumerate(sorted_jobs, 1):
        print(f"  {i}. Priority {job.priority}: {job.id}")
    
    return jobs


async def main():
    """Run all demonstrations."""
    print("ContentFlow AI - Simple Job Processing Demo")
    print("=" * 60)
    print("This demo showcases the core job processing functionality:")
    print("- Job models and state management")
    print("- Job handlers for different operations")
    print("- Retry logic and error handling")
    print("- Job dependencies and execution order")
    print("- Priority-based scheduling")
    
    try:
        # Run demonstrations
        await demo_job_models()
        await demo_job_handlers()
        await demo_retry_logic()
        await demo_job_dependencies()
        await demo_job_priorities()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Job model with state transitions")
        print("✓ Multiple job handlers (content generation, transformation, creative assistance)")
        print("✓ Retry logic with exponential backoff")
        print("✓ Job dependency resolution")
        print("✓ Priority-based job scheduling")
        print("✓ Comprehensive error handling")
        print("✓ Token usage and cost tracking")
        
        print("\nThe async job processing system core functionality is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
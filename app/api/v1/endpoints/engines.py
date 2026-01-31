"""
Specialized AI Engine API endpoints for ContentFlow AI.

This module provides REST API endpoints for all specialized AI engines:
- Text Intelligence Engine
- Creative Assistant Engine
- Social Media Planner
- Discovery Analytics Engine
- Image Generation Engine
- Audio Generation Engine
- Video Pipeline Engine

Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 4.1, 5.1
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.models.users import User
from app.api.dependencies import (
    get_current_user,
    get_current_user_optional,
    check_content_generation_limit,
    check_transformation_limit
)
from app.core.exceptions import ValidationError, EngineError, AIServiceError
from app.ai.text_intelligence_engine import (
    TextIntelligenceEngine,
    GenerationRequest,
    SummarizationRequest,
    ToneTransformationRequest,
    TranslationRequest,
    PlatformAdaptationRequest
)
from app.ai.creative_assistant_engine import (
    CreativeAssistantEngine,
    CreativeContext,
    SuggestionRequest,
    Feedback,
    DesignAssistanceRequest,
    MarketingAssistanceRequest
)
from app.ai.social_media_planner import (
    SocialMediaPlanner,
    OptimizationRequest,
    HashtagRequest,
    CTARequest,
    PostingTimeRequest,
    EngagementPredictionRequest
)

logger = logging.getLogger(__name__)
router = APIRouter()

from app.ai.discovery_analytics_engine import (
    DiscoveryAnalyticsEngine,
    ContentTaggingRequest,
    TrendAnalysisRequest,
    EngagementAnalysisRequest,
    ImprovementSuggestionsRequest
)
from app.ai.image_generation_engine import (
    ImageGenerationEngine,
    ImageGenerationRequest,
    ImageSpecification
)
from app.ai.audio_generation_engine import (
    AudioGenerationEngine,
    AudioGenerationRequest,
    AudioSpecification
)
from app.ai.video_pipeline_engine import (
    VideoPipelineEngine,
    VideoGenerationRequest,
    VideoSpecification
)

# Initialize engines (singleton instances)
text_engine = TextIntelligenceEngine()
creative_engine = CreativeAssistantEngine()
social_planner = SocialMediaPlanner()
analytics_engine = DiscoveryAnalyticsEngine()
image_engine = ImageGenerationEngine()
audio_engine = AudioGenerationEngine()
video_engine = VideoPipelineEngine()


# ============================================================================
# TEXT INTELLIGENCE ENGINE ENDPOINTS
# ============================================================================

@router.post("/text/generate", status_code=status.HTTP_200_OK)
async def generate_text_content(
    request: GenerationRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Generate text content using the Text Intelligence Engine.
    
    Supports generation of blogs, captions, scripts, articles, and more
    with customizable tone, length, and platform optimization.
    """
    try:
        logger.info(f"Text generation request from user {user.username if user else 'anonymous'}")
        result = await text_engine.generate_content(request)
        
        return {
            "success": True,
            "content": result.content,
            "metadata": result.metadata,
            "word_count": result.word_count,
            "character_count": result.character_count,
            "estimated_reading_time": result.estimated_reading_time,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {str(e)}"
        )


@router.post("/text/summarize", status_code=status.HTTP_200_OK)
async def summarize_text_content(
    request: SummarizationRequest,
    user: User = Depends(get_current_user),
    _: User = Depends(check_transformation_limit)
):
    """
    Summarize text content with length control.
    
    Condenses long-form content while preserving key information
    and maintaining the original meaning.
    """
    try:
        logger.info(f"Text summarization request from user {user.username}")
        result = await text_engine.summarize_content(request)
        
        return {
            "success": True,
            "summary": result.content,
            "metadata": result.metadata,
            "word_count": result.word_count,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/text/transform-tone", status_code=status.HTTP_200_OK)
async def transform_text_tone(
    request: ToneTransformationRequest,
    user: User = Depends(get_current_user),
    _: User = Depends(check_transformation_limit)
):
    """
    Transform content tone while preserving meaning.
    
    Rewrites content to match the specified tone (professional, casual,
    friendly, formal, etc.) while maintaining factual accuracy.
    """
    try:
        logger.info(f"Tone transformation request from user {user.username}")
        result = await text_engine.transform_tone(request)
        
        return {
            "success": True,
            "transformed_content": result.content,
            "metadata": result.metadata,
            "word_count": result.word_count,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tone transformation failed: {str(e)}"
        )


@router.post("/text/translate", status_code=status.HTTP_200_OK)
async def translate_text_content(
    request: TranslationRequest,
    user: User = Depends(get_current_user),
    _: User = Depends(check_transformation_limit)
):
    """
    Translate content to target language.
    
    Converts content to the specified language while maintaining
    meaning, tone, and formatting.
    """
    try:
        logger.info(f"Translation request from user {user.username}")
        result = await text_engine.translate_content(request)
        
        return {
            "success": True,
            "translated_content": result.content,
            "metadata": result.metadata,
            "word_count": result.word_count,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post("/text/adapt-platform", status_code=status.HTTP_200_OK)
async def adapt_text_for_platform(
    request: PlatformAdaptationRequest,
    user: User = Depends(get_current_user),
    _: User = Depends(check_transformation_limit)
):
    """
    Adapt content for specific platform requirements.
    
    Optimizes content for platform-specific constraints including
    length limits, formatting, and engagement patterns.
    """
    try:
        logger.info(f"Platform adaptation request from user {user.username}")
        result = await text_engine.adapt_for_platform(request)
        
        return {
            "success": True,
            "adapted_content": result.content,
            "metadata": result.metadata,
            "word_count": result.word_count,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Platform adaptation failed: {str(e)}"
        )


# ============================================================================
# CREATIVE ASSISTANT ENGINE ENDPOINTS
# ============================================================================

@router.post("/creative/start-session", status_code=status.HTTP_201_CREATED)
async def start_creative_session(
    context: CreativeContext,
    user: User = Depends(get_current_user_optional)
):
    """
    Start a new creative assistance session.
    
    Initializes an interactive creative session with context tracking
    for iterative collaboration and refinement.
    """
    try:
        username = user.username if user else "anonymous"
        logger.info(f"Starting creative session for user {username}")
        session_id = await creative_engine.start_creative_session(context)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Creative session started successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session creation failed: {str(e)}"
        )


@router.post("/creative/{session_id}/suggestions", status_code=status.HTTP_200_OK)
async def get_creative_suggestions(
    session_id: str,
    request: SuggestionRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Get creative suggestions for a session.
    
    Generates contextual suggestions for ideas, rewrites, hooks,
    headlines, and other creative elements.
    """
    try:
        logger.info(f"Generating suggestions for session {session_id}")
        suggestions = await creative_engine.provide_suggestions(session_id, request)
        
        return {
            "success": True,
            "session_id": session_id,
            "suggestions": [
                {
                    "id": s.id,
                    "type": s.type,
                    "content": s.content,
                    "rationale": s.rationale,
                    "confidence_score": s.confidence_score,
                    "metadata": s.metadata
                }
                for s in suggestions
            ]
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Suggestion generation failed: {str(e)}"
        )


@router.post("/creative/{session_id}/refine", status_code=status.HTTP_200_OK)
async def refine_creative_suggestions(
    session_id: str,
    feedback: Feedback,
    user: User = Depends(get_current_user_optional)
):
    """
    Refine suggestions based on user feedback.
    
    Iteratively improves suggestions by incorporating user feedback
    and preferences for better results.
    """
    try:
        logger.info(f"Refining suggestions for session {session_id}")
        refined_suggestions = await creative_engine.refine_suggestions(session_id, feedback)
        
        return {
            "success": True,
            "session_id": session_id,
            "refined_suggestions": [
                {
                    "id": s.id,
                    "type": s.type,
                    "content": s.content,
                    "rationale": s.rationale,
                    "confidence_score": s.confidence_score
                }
                for s in refined_suggestions
            ]
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Refinement failed: {str(e)}"
        )


@router.post("/creative/{session_id}/design-assistance", status_code=status.HTTP_200_OK)
async def get_design_assistance(
    session_id: str,
    request: DesignAssistanceRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Get design assistance and visual suggestions.
    
    Provides design recommendations for layouts, colors, typography,
    and visual hierarchy.
    """
    try:
        logger.info(f"Providing design assistance for session {session_id}")
        suggestions = await creative_engine.provide_design_assistance(session_id, request)
        
        return {
            "success": True,
            "session_id": session_id,
            "design_suggestions": [
                {
                    "id": s.id,
                    "content": s.content,
                    "rationale": s.rationale,
                    "confidence_score": s.confidence_score
                }
                for s in suggestions
            ]
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Design assistance failed: {str(e)}"
        )


@router.post("/creative/{session_id}/marketing-assistance", status_code=status.HTTP_200_OK)
async def get_marketing_assistance(
    session_id: str,
    request: MarketingAssistanceRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Get marketing assistance and campaign ideas.
    
    Provides strategic marketing suggestions including campaigns,
    CTAs, value propositions, and messaging strategies.
    """
    try:
        logger.info(f"Providing marketing assistance for session {session_id}")
        suggestions = await creative_engine.provide_marketing_assistance(session_id, request)
        
        return {
            "success": True,
            "session_id": session_id,
            "marketing_suggestions": [
                {
                    "id": s.id,
                    "content": s.content,
                    "rationale": s.rationale,
                    "confidence_score": s.confidence_score
                }
                for s in suggestions
            ]
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Marketing assistance failed: {str(e)}"
        )


@router.get("/creative/{session_id}", status_code=status.HTTP_200_OK)
async def get_creative_session(
    session_id: str,
    user: User = Depends(get_current_user_optional)
):
    """
    Get creative session details.
    
    Retrieves the current state and history of a creative session.
    """
    try:
        session = creative_engine.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return {
            "success": True,
            "session_id": session.id,
            "context": session.context.dict(),
            "interactions_count": len(session.interactions),
            "tokens_used": session.tokens_used,
            "cost": session.cost,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session retrieval failed: {str(e)}"
        )


@router.delete("/creative/{session_id}", status_code=status.HTTP_200_OK)
async def end_creative_session(
    session_id: str,
    user: User = Depends(get_current_user)
):
    """
    End a creative session.
    
    Terminates the session and returns a summary of the session statistics.
    """
    try:
        summary = creative_engine.end_session(session_id)
        
        return {
            "success": True,
            "message": "Session ended successfully",
            "summary": summary
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session termination failed: {str(e)}"
        )


# ============================================================================
# SOCIAL MEDIA PLANNER ENDPOINTS
# ============================================================================

@router.post("/social/optimize", status_code=status.HTTP_200_OK)
async def optimize_for_social_platform(
    request: OptimizationRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Optimize content for specific social media platform.
    
    Adapts content to platform requirements including length limits,
    hashtags, CTAs, and engagement optimization.
    """
    try:
        logger.info(f"Optimizing content for {request.platform}")
        result = await social_planner.optimize_for_platform(request)
        
        return {
            "success": True,
            "original_content": result.original_content,
            "optimized_content": result.optimized_content,
            "platform": result.platform.value,
            "hashtags": result.hashtags,
            "call_to_action": result.call_to_action,
            "character_count": result.character_count,
            "engagement_prediction": result.engagement_prediction,
            "optimization_notes": result.optimization_notes,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/social/hashtags", status_code=status.HTTP_200_OK)
async def generate_hashtags(
    request: HashtagRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Generate relevant hashtags for content.
    
    Creates platform-appropriate hashtags based on content analysis
    and trending topics.
    """
    try:
        logger.info(f"Generating hashtags for {request.platform}")
        hashtags = await social_planner.generate_hashtags(request)
        
        return {
            "success": True,
            "hashtags": hashtags,
            "count": len(hashtags),
            "platform": request.platform.value
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hashtag generation failed: {str(e)}"
        )


@router.post("/social/cta", status_code=status.HTTP_200_OK)
async def generate_call_to_action(
    request: CTARequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Generate call-to-action text.
    
    Creates effective CTAs optimized for the specified platform
    and campaign goal.
    """
    try:
        logger.info(f"Generating CTA for {request.platform}")
        cta = await social_planner.generate_cta(request)
        
        return {
            "success": True,
            "call_to_action": cta,
            "platform": request.platform.value,
            "goal": request.goal
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CTA generation failed: {str(e)}"
        )


@router.post("/social/posting-times", status_code=status.HTTP_200_OK)
async def suggest_posting_times(
    request: PostingTimeRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Suggest optimal posting times.
    
    Analyzes platform patterns and audience behavior to recommend
    the best times to post content for maximum engagement.
    """
    try:
        logger.info(f"Suggesting posting times for {request.platform}")
        suggestions = await social_planner.suggest_posting_times(request)
        
        return {
            "success": True,
            "platform": request.platform.value,
            "suggestions": [
                {
                    "datetime": s.datetime.isoformat(),
                    "time_slot": s.time_slot.value,
                    "day_of_week": s.day_of_week,
                    "confidence_score": s.confidence_score,
                    "expected_reach": s.expected_reach,
                    "rationale": s.rationale
                }
                for s in suggestions
            ]
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Posting time suggestion failed: {str(e)}"
        )


@router.post("/social/predict-engagement", status_code=status.HTTP_200_OK)
async def predict_engagement(
    request: EngagementPredictionRequest,
    user: User = Depends(get_current_user_optional)
):
    """
    Predict engagement score for content.
    
    Analyzes content and context to predict expected engagement
    metrics and provide optimization recommendations.
    """
    try:
        logger.info(f"Predicting engagement for {request.platform}")
        score = await social_planner.predict_engagement(request)
        
        return {
            "success": True,
            "platform": request.platform.value,
            "overall_score": score.overall_score,
            "predicted_likes": score.predicted_likes,
            "predicted_shares": score.predicted_shares,
            "predicted_comments": score.predicted_comments,
            "predicted_reach": score.predicted_reach,
            "confidence": score.confidence,
            "factors": score.factors,
            "recommendations": score.recommendations
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engagement prediction failed: {str(e)}"
        )


# ============================================================================
# DISCOVERY ANALYTICS ENGINE ENDPOINTS
# ============================================================================

@router.post("/analytics/tag-content", status_code=status.HTTP_200_OK)
async def auto_tag_content(
    request: ContentTaggingRequest,
    user: User = Depends(get_current_user)
):
    """
    Automatically tag content with topics, keywords, and sentiment.
    
    Analyzes content to extract relevant tags, identify topics,
    detect sentiment, and recognize entities.
    """
    try:
        logger.info(f"Auto-tagging content for user {user.username}")
        result = await analytics_engine.auto_tag_content(request)
        
        return {
            "success": True,
            "tags": [
                {
                    "name": t.name,
                    "category": t.category.value,
                    "confidence": t.confidence,
                    "relevance_score": t.relevance_score
                }
                for t in result.tags
            ],
            "topics": result.topics,
            "keywords": result.keywords,
            "entities": result.entities,
            "sentiment": result.sentiment.value,
            "sentiment_score": result.sentiment_score,
            "confidence": result.confidence,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content tagging failed: {str(e)}"
        )


@router.post("/analytics/improvement-suggestions", status_code=status.HTTP_200_OK)
async def generate_improvement_suggestions(
    request: ImprovementSuggestionsRequest,
    user: User = Depends(get_current_user)
):
    """
    Generate AI-powered improvement suggestions for content.
    
    Analyzes content and performance to provide actionable
    recommendations for optimization.
    """
    try:
        logger.info(f"Generating improvement suggestions for user {user.username}")
        result = await analytics_engine.generate_improvement_suggestions(request)
        
        return {
            "success": True,
            "suggestions": [
                {
                    "id": s.suggestion_id,
                    "category": s.category,
                    "priority": s.priority,
                    "title": s.title,
                    "description": s.description,
                    "expected_impact": s.expected_impact,
                    "implementation_effort": s.implementation_effort,
                    "specific_actions": s.specific_actions,
                    "rationale": s.rationale,
                    "confidence": s.confidence
                }
                for s in result.suggestions
            ],
            "overall_assessment": result.overall_assessment,
            "priority_actions": result.priority_actions,
            "estimated_improvement_potential": result.estimated_improvement_potential,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Improvement suggestion generation failed: {str(e)}"
        )


# ============================================================================
# IMAGE GENERATION ENGINE ENDPOINTS
# ============================================================================

@router.post("/media/image/generate", status_code=status.HTTP_201_CREATED)
async def generate_image(
    request: ImageGenerationRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Generate an image based on specifications.
    
    Creates visual content including thumbnails, posters, banners,
    and social media graphics with customizable styles and dimensions.
    """
    try:
        logger.info(f"Generating {request.image_type} image for user {user.username if user else 'anonymous'}")
        result = await image_engine.generate_image(request)
        
        return {
            "success": True,
            "image_id": result.image_id,
            "image_type": result.image_type.value,
            "file_url": result.file_url,
            "specification": {
                "width": result.specification.width,
                "height": result.specification.height,
                "format": result.specification.format.value,
                "quality": result.specification.quality
            },
            "file_size_mb": result.get_file_size_mb(),
            "metadata": result.metadata,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )


@router.get("/media/image/{image_id}", status_code=status.HTTP_200_OK)
async def get_image(
    image_id: str,
    user: User = Depends(get_current_user)
):
    """
    Retrieve image metadata by ID.
    
    Returns information about a previously generated image.
    """
    try:
        image_info = await image_engine.get_image(image_id)
        
        if not image_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found"
            )
        
        return {
            "success": True,
            **image_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image retrieval failed: {str(e)}"
        )


@router.delete("/media/image/{image_id}", status_code=status.HTTP_200_OK)
async def delete_image(
    image_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete an image from storage.
    
    Permanently removes the image file.
    """
    try:
        deleted = await image_engine.delete_image(image_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Image {image_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image deletion failed: {str(e)}"
        )


# ============================================================================
# AUDIO GENERATION ENGINE ENDPOINTS
# ============================================================================

@router.post("/media/audio/generate", status_code=status.HTTP_201_CREATED)
async def generate_audio(
    request: AudioGenerationRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Generate audio content.
    
    Creates voiceovers, narration, background music, and other
    audio content with customizable voice styles and formats.
    """
    try:
        logger.info(f"Generating {request.audio_type} audio for user {user.username if user else 'anonymous'}")
        result = await audio_engine.generate_audio(request)
        
        return {
            "success": True,
            "audio_id": result.audio_id,
            "audio_type": result.audio_type.value,
            "file_url": result.file_url,
            "specification": {
                "format": result.specification.format.value,
                "sample_rate": result.specification.sample_rate,
                "bitrate": result.specification.bitrate,
                "channels": result.specification.channels
            },
            "duration": result.get_duration_formatted(),
            "duration_seconds": result.duration_seconds,
            "file_size_mb": result.get_file_size_mb(),
            "metadata": result.metadata,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio generation failed: {str(e)}"
        )


@router.get("/media/audio/{audio_id}", status_code=status.HTTP_200_OK)
async def get_audio(
    audio_id: str,
    user: User = Depends(get_current_user)
):
    """
    Retrieve audio metadata by ID.
    
    Returns information about a previously generated audio file.
    """
    try:
        audio_info = await audio_engine.get_audio(audio_id)
        
        if not audio_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audio {audio_id} not found"
            )
        
        return {
            "success": True,
            **audio_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio retrieval failed: {str(e)}"
        )


@router.delete("/media/audio/{audio_id}", status_code=status.HTTP_200_OK)
async def delete_audio(
    audio_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete an audio file from storage.
    
    Permanently removes the audio file.
    """
    try:
        deleted = await audio_engine.delete_audio(audio_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audio {audio_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Audio {audio_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio deletion failed: {str(e)}"
        )


# ============================================================================
# VIDEO PIPELINE ENGINE ENDPOINTS
# ============================================================================

@router.post("/media/video/generate", status_code=status.HTTP_201_CREATED)
async def generate_video(
    request: VideoGenerationRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Generate a video.
    
    Creates short-form videos, explainers, tutorials, and promotional
    content with customizable styles, quality, and features.
    """
    try:
        logger.info(f"Generating {request.video_type} video for user {user.username if user else 'anonymous'}")
        result = await video_engine.generate_video(request)
        
        return {
            "success": True,
            "video_id": result.video_id,
            "video_type": result.video_type.value,
            "file_url": result.file_url,
            "thumbnail_url": result.thumbnail_url,
            "specification": {
                "width": result.specification.width,
                "height": result.specification.height,
                "format": result.specification.format.value,
                "quality": result.specification.quality.value,
                "fps": result.specification.fps
            },
            "duration": result.get_duration_formatted(),
            "duration_seconds": result.duration_seconds,
            "file_size_mb": result.get_file_size_mb(),
            "metadata": result.metadata,
            "tokens_used": result.tokens_used,
            "cost": result.cost
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EngineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {str(e)}"
        )


@router.get("/media/video/{video_id}", status_code=status.HTTP_200_OK)
async def get_video(
    video_id: str,
    user: User = Depends(get_current_user)
):
    """
    Retrieve video metadata by ID.
    
    Returns information about a previously generated video.
    """
    try:
        video_info = await video_engine.get_video(video_id)
        
        if not video_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video {video_id} not found"
            )
        
        return {
            "success": True,
            **video_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video retrieval failed: {str(e)}"
        )


@router.delete("/media/video/{video_id}", status_code=status.HTTP_200_OK)
async def delete_video(
    video_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete a video from storage.
    
    Permanently removes the video file and its thumbnail.
    """
    try:
        deleted = await video_engine.delete_video(video_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video {video_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Video {video_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video deletion failed: {str(e)}"
        )


# ============================================================================
# ENGINE STATISTICS ENDPOINTS
# ============================================================================

@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_engine_statistics(
    user: User = Depends(get_current_user)
):
    """
    Get usage statistics for all engines.
    
    Returns aggregated usage metrics including tokens used,
    costs, and generation counts across all engines.
    """
    try:
        return {
            "success": True,
            "text_engine": text_engine.get_usage_stats(),
            "creative_engine": {
                "active_sessions": len(creative_engine.sessions),
                "total_tokens_used": creative_engine.total_tokens_used,
                "total_cost": creative_engine.total_cost
            },
            "social_planner": {
                "total_tokens_used": social_planner.total_tokens_used,
                "total_cost": social_planner.total_cost
            },
            "analytics_engine": {
                "total_tokens_used": analytics_engine.total_tokens_used,
                "total_cost": analytics_engine.total_cost
            },
            "image_engine": image_engine.get_usage_stats(),
            "audio_engine": audio_engine.get_usage_stats(),
            "video_engine": video_engine.get_usage_stats()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve engine statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics retrieval failed: {str(e)}"
        )

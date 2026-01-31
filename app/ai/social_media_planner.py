"""
Social Media Planner for ContentFlow AI.

This module implements the Social Media Planner responsible for platform-specific
content optimization, hashtag generation, posting time suggestions, content calendar
management, and engagement prediction.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import google.generativeai as genai
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.core.exceptions import (
    EngineError, ValidationError, AIServiceError
)
from app.models.base import Platform
from app.models.content import OptimizationData, EngagementMetrics

logger = logging.getLogger(__name__)


class AudienceType(str, Enum):
    """Enumeration of audience types."""
    GENERAL = "general"
    BUSINESS = "business"
    TECH = "tech"
    LIFESTYLE = "lifestyle"
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    HEALTH = "health"
    FINANCE = "finance"


class PostingTimeSlot(str, Enum):
    """Enumeration of posting time slots."""
    EARLY_MORNING = "early_morning"  # 6-9 AM
    MORNING = "morning"  # 9-12 PM
    AFTERNOON = "afternoon"  # 12-3 PM
    LATE_AFTERNOON = "late_afternoon"  # 3-6 PM
    EVENING = "evening"  # 6-9 PM
    NIGHT = "night"  # 9 PM-12 AM


class OptimizationRequest(BaseModel):
    """Model for platform optimization requests."""
    content: str
    platform: Platform
    target_audience: Optional[AudienceType] = AudienceType.GENERAL
    include_hashtags: bool = True
    include_cta: bool = True
    optimize_length: bool = True
    brand_voice: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class HashtagRequest(BaseModel):
    """Model for hashtag generation requests."""
    content: str
    platform: Platform
    count: int = Field(default=5, ge=1, le=30)
    trending_only: bool = False
    niche_specific: bool = True
    target_audience: Optional[AudienceType] = None


class CTARequest(BaseModel):
    """Model for call-to-action generation requests."""
    content: str
    platform: Platform
    goal: str  # engagement, conversion, awareness, traffic
    tone: str = "friendly"
    include_emoji: bool = True


class PostingTimeRequest(BaseModel):
    """Model for optimal posting time requests."""
    platform: Platform
    target_audience: AudienceType
    timezone: str = "UTC"
    content_type: str = "general"  # promotional, educational, entertainment
    days_ahead: int = Field(default=7, ge=1, le=30)


class CalendarEntry(BaseModel):
    """Model for content calendar entries."""
    id: str
    content_id: str
    platform: Platform
    scheduled_time: datetime
    content_preview: str
    status: str = "scheduled"  # scheduled, published, failed
    engagement_prediction: float = 0.0
    actual_engagement: Optional[float] = None


class EngagementPredictionRequest(BaseModel):
    """Model for engagement prediction requests."""
    content: str
    platform: Platform
    posting_time: datetime
    target_audience: AudienceType
    has_media: bool = False
    hashtag_count: int = 0
    historical_performance: Optional[Dict[str, float]] = None


class OptimizedContent(BaseModel):
    """Model for optimized social media content."""
    original_content: str
    optimized_content: str
    platform: Platform
    hashtags: List[str] = Field(default_factory=list)
    call_to_action: Optional[str] = None
    character_count: int = 0
    estimated_reach: int = 0
    engagement_prediction: float = 0.0
    optimization_notes: List[str] = Field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.optimized_content:
            self.character_count = len(self.optimized_content)


class PostingTimeSuggestion(BaseModel):
    """Model for posting time suggestions."""
    datetime: datetime
    time_slot: PostingTimeSlot
    confidence_score: float = Field(ge=0.0, le=1.0)
    expected_reach: int = 0
    rationale: str
    day_of_week: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.datetime and not self.day_of_week:
            self.day_of_week = self.datetime.strftime("%A")


class EngagementScore(BaseModel):
    """Model for engagement prediction scores."""
    overall_score: float = Field(ge=0.0, le=100.0)
    predicted_likes: int = 0
    predicted_shares: int = 0
    predicted_comments: int = 0
    predicted_reach: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    factors: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class SocialMediaPlanner:
    """
    Social Media Planner for platform-specific optimization and scheduling.
    
    This engine handles:
    - Platform-specific content optimization
    - Hashtag and CTA generation
    - Optimal posting time suggestions
    - Content calendar management
    - Engagement prediction and scoring
    """
    
    def __init__(self):
        """Initialize the Social Media Planner."""
        self.gemini_client = None
        self.calendar: Dict[str, List[CalendarEntry]] = {}  # platform -> entries
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cost_per_token = 0.000001  # $0.000001 per token (example rate)
        self._initialize_gemini()
        self._initialize_platform_specs()
    
    def _initialize_gemini(self):
        """Initialize Google Gemini client."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. Social Media Planner will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Social Media Planner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise EngineError("social_media_planner", f"Initialization failed: {e}")
    
    def _initialize_platform_specs(self):
        """Initialize platform-specific specifications."""
        self.platform_specs = {
            Platform.TWITTER: {
                "max_length": 280,
                "optimal_hashtags": 2,
                "best_times": ["9:00", "12:00", "17:00"],
                "character_limit": 280,
                "supports_threads": True
            },
            Platform.INSTAGRAM: {
                "max_length": 2200,
                "optimal_hashtags": 11,
                "best_times": ["11:00", "13:00", "19:00"],
                "character_limit": 2200,
                "supports_carousel": True
            },
            Platform.FACEBOOK: {
                "max_length": 63206,
                "optimal_hashtags": 3,
                "best_times": ["13:00", "15:00", "19:00"],
                "character_limit": 63206,
                "supports_long_form": True
            },
            Platform.LINKEDIN: {
                "max_length": 3000,
                "optimal_hashtags": 5,
                "best_times": ["8:00", "12:00", "17:00"],
                "character_limit": 3000,
                "professional_tone": True
            },
            Platform.TIKTOK: {
                "max_length": 2200,
                "optimal_hashtags": 5,
                "best_times": ["18:00", "19:00", "21:00"],
                "character_limit": 2200,
                "trending_focused": True
            },
            Platform.YOUTUBE: {
                "max_length": 5000,
                "optimal_hashtags": 15,
                "best_times": ["14:00", "17:00", "20:00"],
                "character_limit": 5000,
                "seo_important": True
            }
        }

    
    async def optimize_for_platform(
        self, 
        request: OptimizationRequest
    ) -> OptimizedContent:
        """
        Optimize content for specific platform requirements.
        
        Args:
            request: Optimization request with content and parameters
            
        Returns:
            OptimizedContent with platform-optimized content
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If optimization fails
        """
        try:
            logger.info(f"Optimizing content for {request.platform}")
            
            # Validate request
            self._validate_optimization_request(request)
            
            # Get platform specifications
            specs = self.platform_specs.get(request.platform, {})
            
            # Build optimization prompt
            prompt = self._build_optimization_prompt(request, specs)
            
            # Generate optimized content using Gemini
            optimized_text = await self._generate_with_gemini(prompt)
            
            # Post-process optimization
            processed_content = self._post_process_optimization(
                optimized_text,
                request.platform,
                specs
            )
            
            # Generate hashtags if requested
            hashtags = []
            if request.include_hashtags:
                hashtag_request = HashtagRequest(
                    content=processed_content,
                    platform=request.platform,
                    count=specs.get("optimal_hashtags", 5),
                    target_audience=request.target_audience
                )
                hashtags = await self.generate_hashtags(hashtag_request)
            
            # Generate CTA if requested
            cta = None
            if request.include_cta:
                cta_request = CTARequest(
                    content=processed_content,
                    platform=request.platform,
                    goal="engagement"
                )
                cta = await self.generate_cta(cta_request)
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(request.content + processed_content)
            cost = tokens_used * self.cost_per_token
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = OptimizedContent(
                original_content=request.content,
                optimized_content=processed_content,
                platform=request.platform,
                hashtags=hashtags,
                call_to_action=cta,
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Content optimization completed: {result.character_count} characters")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Content optimization failed: {e}")
            raise EngineError("social_media_planner", f"Optimization failed: {e}")

    
    async def generate_hashtags(
        self, 
        request: HashtagRequest
    ) -> List[str]:
        """
        Generate relevant hashtags for content.
        
        Args:
            request: Hashtag generation request
            
        Returns:
            List of relevant hashtags
            
        Raises:
            ValidationError: If request is invalid
            EngineError: If generation fails
        """
        try:
            logger.info(f"Generating {request.count} hashtags for {request.platform}")
            
            # Validate request
            self._validate_hashtag_request(request)
            
            # Build hashtag generation prompt
            prompt = self._build_hashtag_prompt(request)
            
            # Generate hashtags using Gemini
            hashtags_text = await self._generate_with_gemini(prompt)
            
            # Parse and clean hashtags
            hashtags = self._parse_hashtags(hashtags_text, request.count)
            
            # Update usage tracking
            tokens_used = self._estimate_tokens(request.content + hashtags_text)
            self.total_tokens_used += tokens_used
            self.total_cost += tokens_used * self.cost_per_token
            
            logger.info(f"Generated {len(hashtags)} hashtags")
            return hashtags
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Hashtag generation failed: {e}")
            raise EngineError("social_media_planner", f"Hashtag generation failed: {e}")
    
    async def generate_cta(
        self, 
        request: CTARequest
    ) -> str:
        """
        Generate call-to-action text.
        
        Args:
            request: CTA generation request
            
        Returns:
            Call-to-action text
            
        Raises:
            ValidationError: If request is invalid
            EngineError: If generation fails
        """
        try:
            logger.info(f"Generating CTA for {request.platform}")
            
            # Validate request
            self._validate_cta_request(request)
            
            # Build CTA generation prompt
            prompt = self._build_cta_prompt(request)
            
            # Generate CTA using Gemini
            cta = await self._generate_with_gemini(prompt)
            
            # Clean and format CTA
            cta = self._clean_cta(cta, request.include_emoji)
            
            # Update usage tracking
            tokens_used = self._estimate_tokens(request.content + cta)
            self.total_tokens_used += tokens_used
            self.total_cost += tokens_used * self.cost_per_token
            
            logger.info(f"Generated CTA: {cta}")
            return cta
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"CTA generation failed: {e}")
            raise EngineError("social_media_planner", f"CTA generation failed: {e}")

    
    async def suggest_posting_times(
        self, 
        request: PostingTimeRequest
    ) -> List[PostingTimeSuggestion]:
        """
        Suggest optimal posting times based on platform and audience.
        
        Args:
            request: Posting time request with parameters
            
        Returns:
            List of posting time suggestions
            
        Raises:
            ValidationError: If request is invalid
            EngineError: If suggestion generation fails
        """
        try:
            logger.info(f"Suggesting posting times for {request.platform}")
            
            # Validate request
            self._validate_posting_time_request(request)
            
            # Get platform specifications
            specs = self.platform_specs.get(request.platform, {})
            best_times = specs.get("best_times", ["12:00", "15:00", "18:00"])
            
            # Generate suggestions for the next N days
            suggestions = []
            current_date = datetime.utcnow()
            
            for day_offset in range(request.days_ahead):
                target_date = current_date + timedelta(days=day_offset)
                
                # Skip weekends for LinkedIn (business platform)
                if request.platform == Platform.LINKEDIN and target_date.weekday() >= 5:
                    continue
                
                # Generate suggestions for each best time
                for time_str in best_times[:3]:  # Top 3 times per day
                    hour, minute = map(int, time_str.split(":"))
                    posting_time = target_date.replace(
                        hour=hour, 
                        minute=minute, 
                        second=0, 
                        microsecond=0
                    )
                    
                    # Determine time slot
                    time_slot = self._determine_time_slot(hour)
                    
                    # Calculate confidence score based on various factors
                    confidence = self._calculate_time_confidence(
                        posting_time,
                        request.platform,
                        request.target_audience,
                        request.content_type
                    )
                    
                    # Estimate expected reach
                    expected_reach = self._estimate_reach_for_time(
                        posting_time,
                        request.platform,
                        request.target_audience
                    )
                    
                    # Generate rationale
                    rationale = self._generate_time_rationale(
                        posting_time,
                        request.platform,
                        time_slot,
                        request.target_audience
                    )
                    
                    suggestion = PostingTimeSuggestion(
                        datetime=posting_time,
                        time_slot=time_slot,
                        confidence_score=confidence,
                        expected_reach=expected_reach,
                        rationale=rationale
                    )
                    suggestions.append(suggestion)
            
            # Sort by confidence score
            suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
            
            logger.info(f"Generated {len(suggestions)} posting time suggestions")
            return suggestions[:request.days_ahead * 2]  # Return top suggestions
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Posting time suggestion failed: {e}")
            raise EngineError("social_media_planner", f"Time suggestion failed: {e}")

    
    async def predict_engagement(
        self, 
        request: EngagementPredictionRequest
    ) -> EngagementScore:
        """
        Predict engagement score for content.
        
        Args:
            request: Engagement prediction request
            
        Returns:
            EngagementScore with predictions and recommendations
            
        Raises:
            ValidationError: If request is invalid
            EngineError: If prediction fails
        """
        try:
            logger.info(f"Predicting engagement for {request.platform}")
            
            # Validate request
            self._validate_engagement_request(request)
            
            # Calculate base score factors
            factors = {}
            
            # Content quality factor (0-1)
            content_quality = self._assess_content_quality(request.content)
            factors["content_quality"] = content_quality
            
            # Timing factor (0-1)
            timing_factor = self._assess_timing(request.posting_time, request.platform)
            factors["timing"] = timing_factor
            
            # Media factor (0-1)
            media_factor = 0.8 if request.has_media else 0.5
            factors["media_presence"] = media_factor
            
            # Hashtag factor (0-1)
            optimal_hashtags = self.platform_specs.get(request.platform, {}).get("optimal_hashtags", 5)
            hashtag_factor = min(1.0, request.hashtag_count / optimal_hashtags) if request.hashtag_count > 0 else 0.3
            factors["hashtag_optimization"] = hashtag_factor
            
            # Platform-specific factor
            platform_factor = self._get_platform_engagement_multiplier(request.platform)
            factors["platform_multiplier"] = platform_factor
            
            # Calculate overall score (0-100)
            base_score = (
                content_quality * 0.35 +
                timing_factor * 0.25 +
                media_factor * 0.20 +
                hashtag_factor * 0.15 +
                platform_factor * 0.05
            ) * 100
            
            # Adjust based on historical performance if available
            if request.historical_performance:
                historical_avg = request.historical_performance.get("average_score", 50)
                base_score = (base_score * 0.7) + (historical_avg * 0.3)
            
            # Calculate predicted metrics
            predicted_reach = self._predict_reach(base_score, request.platform)
            predicted_likes = int(predicted_reach * 0.05)  # 5% engagement rate
            predicted_shares = int(predicted_likes * 0.15)  # 15% of likes
            predicted_comments = int(predicted_likes * 0.10)  # 10% of likes
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                request.content,
                request.has_media,
                request.historical_performance is not None
            )
            
            # Generate recommendations
            recommendations = self._generate_engagement_recommendations(
                factors,
                request.platform,
                request.has_media,
                request.hashtag_count
            )
            
            result = EngagementScore(
                overall_score=round(base_score, 2),
                predicted_likes=predicted_likes,
                predicted_shares=predicted_shares,
                predicted_comments=predicted_comments,
                predicted_reach=predicted_reach,
                confidence=confidence,
                factors=factors,
                recommendations=recommendations
            )
            
            logger.info(f"Engagement prediction completed: {result.overall_score}/100")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Engagement prediction failed: {e}")
            raise EngineError("social_media_planner", f"Prediction failed: {e}")

    
    def schedule_content(
        self,
        content_id: str,
        platform: Platform,
        scheduled_time: datetime,
        content_preview: str,
        engagement_prediction: float = 0.0
    ) -> CalendarEntry:
        """
        Schedule content in the calendar.
        
        Args:
            content_id: ID of the content to schedule
            platform: Target platform
            scheduled_time: When to publish
            content_preview: Preview of the content
            engagement_prediction: Predicted engagement score
            
        Returns:
            CalendarEntry for the scheduled content
        """
        try:
            import uuid
            
            entry = CalendarEntry(
                id=str(uuid.uuid4()),
                content_id=content_id,
                platform=platform,
                scheduled_time=scheduled_time,
                content_preview=content_preview[:100],  # Limit preview length
                engagement_prediction=engagement_prediction
            )
            
            # Add to calendar
            platform_key = platform.value
            if platform_key not in self.calendar:
                self.calendar[platform_key] = []
            
            self.calendar[platform_key].append(entry)
            
            logger.info(f"Content scheduled for {platform} at {scheduled_time}")
            return entry
            
        except Exception as e:
            logger.error(f"Content scheduling failed: {e}")
            raise EngineError("social_media_planner", f"Scheduling failed: {e}")
    
    def get_calendar(
        self,
        platform: Optional[Platform] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[CalendarEntry]:
        """
        Get calendar entries with optional filtering.
        
        Args:
            platform: Filter by platform (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of calendar entries
        """
        entries = []
        
        # Get entries for specific platform or all platforms
        if platform:
            entries = self.calendar.get(platform.value, [])
        else:
            for platform_entries in self.calendar.values():
                entries.extend(platform_entries)
        
        # Apply date filters
        if start_date:
            entries = [e for e in entries if e.scheduled_time >= start_date]
        if end_date:
            entries = [e for e in entries if e.scheduled_time <= end_date]
        
        # Sort by scheduled time
        entries.sort(key=lambda x: x.scheduled_time)
        
        return entries
    
    def update_calendar_entry(
        self,
        entry_id: str,
        status: Optional[str] = None,
        actual_engagement: Optional[float] = None
    ) -> bool:
        """
        Update a calendar entry.
        
        Args:
            entry_id: ID of the entry to update
            status: New status (optional)
            actual_engagement: Actual engagement score (optional)
            
        Returns:
            True if updated successfully, False otherwise
        """
        for platform_entries in self.calendar.values():
            for entry in platform_entries:
                if entry.id == entry_id:
                    if status:
                        entry.status = status
                    if actual_engagement is not None:
                        entry.actual_engagement = actual_engagement
                    logger.info(f"Calendar entry {entry_id} updated")
                    return True
        
        logger.warning(f"Calendar entry {entry_id} not found")
        return False
    
    def remove_calendar_entry(self, entry_id: str) -> bool:
        """
        Remove a calendar entry.
        
        Args:
            entry_id: ID of the entry to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        for platform_key, platform_entries in self.calendar.items():
            for i, entry in enumerate(platform_entries):
                if entry.id == entry_id:
                    del self.calendar[platform_key][i]
                    logger.info(f"Calendar entry {entry_id} removed")
                    return True
        
        logger.warning(f"Calendar entry {entry_id} not found")
        return False

    
    # Private helper methods
    
    def _validate_optimization_request(self, request: OptimizationRequest):
        """Validate optimization request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if len(request.content) > 50000:
            raise ValidationError("Content exceeds maximum length of 50000 characters")
    
    def _validate_hashtag_request(self, request: HashtagRequest):
        """Validate hashtag request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if request.count < 1 or request.count > 30:
            raise ValidationError("Hashtag count must be between 1 and 30")
    
    def _validate_cta_request(self, request: CTARequest):
        """Validate CTA request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if not request.goal:
            raise ValidationError("Goal is required for CTA generation")
    
    def _validate_posting_time_request(self, request: PostingTimeRequest):
        """Validate posting time request."""
        if request.days_ahead < 1 or request.days_ahead > 30:
            raise ValidationError("Days ahead must be between 1 and 30")
    
    def _validate_engagement_request(self, request: EngagementPredictionRequest):
        """Validate engagement prediction request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if request.posting_time < datetime.utcnow():
            raise ValidationError("Posting time cannot be in the past")

    
    def _build_optimization_prompt(
        self, 
        request: OptimizationRequest, 
        specs: Dict[str, Any]
    ) -> str:
        """Build prompt for content optimization."""
        prompt_parts = [
            f"Optimize the following content for {request.platform.value}.",
            f"\nPlatform specifications:",
            f"- Maximum length: {specs.get('character_limit', 'unlimited')} characters",
            f"- Optimal hashtag count: {specs.get('optimal_hashtags', 5)}"
        ]
        
        if request.target_audience:
            prompt_parts.append(f"\nTarget audience: {request.target_audience.value}")
        
        if request.brand_voice:
            prompt_parts.append(f"\nBrand voice: {request.brand_voice}")
        
        if request.keywords:
            prompt_parts.append(f"\nKeywords to include: {', '.join(request.keywords)}")
        
        if request.optimize_length:
            prompt_parts.append(f"\nOptimize length for maximum engagement on {request.platform.value}")
        
        prompt_parts.append(f"\n\nOriginal content:\n{request.content}")
        prompt_parts.append("\n\nProvide optimized content that:")
        prompt_parts.append("1. Meets platform requirements")
        prompt_parts.append("2. Maximizes engagement potential")
        prompt_parts.append("3. Maintains the core message")
        prompt_parts.append("4. Uses platform-appropriate language and style")
        
        return "".join(prompt_parts)
    
    def _build_hashtag_prompt(self, request: HashtagRequest) -> str:
        """Build prompt for hashtag generation."""
        prompt_parts = [
            f"Generate {request.count} relevant hashtags for {request.platform.value}.",
            f"\nContent: {request.content[:500]}"  # Limit content length
        ]
        
        if request.target_audience:
            prompt_parts.append(f"\nTarget audience: {request.target_audience.value}")
        
        if request.trending_only:
            prompt_parts.append("\nFocus on trending hashtags")
        
        if request.niche_specific:
            prompt_parts.append("\nInclude niche-specific hashtags")
        
        prompt_parts.append("\n\nProvide hashtags that:")
        prompt_parts.append("1. Are relevant to the content")
        prompt_parts.append("2. Have good reach potential")
        prompt_parts.append("3. Mix popular and niche tags")
        prompt_parts.append("4. Are appropriate for the platform")
        prompt_parts.append("\nFormat: List hashtags one per line with # prefix")
        
        return "".join(prompt_parts)
    
    def _build_cta_prompt(self, request: CTARequest) -> str:
        """Build prompt for CTA generation."""
        prompt_parts = [
            f"Generate a compelling call-to-action for {request.platform.value}.",
            f"\nContent context: {request.content[:300]}",
            f"\nGoal: {request.goal}",
            f"\nTone: {request.tone}"
        ]
        
        if request.include_emoji:
            prompt_parts.append("\nInclude relevant emoji")
        
        prompt_parts.append("\n\nProvide a CTA that:")
        prompt_parts.append("1. Is clear and actionable")
        prompt_parts.append("2. Aligns with the goal")
        prompt_parts.append("3. Matches the platform style")
        prompt_parts.append("4. Encourages user engagement")
        prompt_parts.append("\nProvide only the CTA text, no explanation.")
        
        return "".join(prompt_parts)

    
    def _post_process_optimization(
        self, 
        content: str, 
        platform: Platform,
        specs: Dict[str, Any]
    ) -> str:
        """Post-process optimized content."""
        # Remove markdown code blocks if present
        if content.startswith("```") and content.endswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        # Enforce character limit
        char_limit = specs.get("character_limit")
        if char_limit and len(content) > char_limit:
            content = content[:char_limit-3] + "..."
        
        return content.strip()
    
    def _parse_hashtags(self, text: str, max_count: int) -> List[str]:
        """Parse hashtags from generated text."""
        hashtags = []
        lines = text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract all hashtags from the line
            words = line.split()
            for word in words:
                # Clean the word and check if it starts with #
                word = word.strip('.,!?;:')
                if word.startswith("#"):
                    hashtags.append(word)
            
            if len(hashtags) >= max_count:
                break
        
        # Clean and deduplicate
        hashtags = list(dict.fromkeys(hashtags))  # Remove duplicates while preserving order
        return hashtags[:max_count]
    
    def _clean_cta(self, cta: str, include_emoji: bool) -> str:
        """Clean and format CTA text."""
        # Remove quotes if present
        cta = cta.strip().strip('"').strip("'")
        
        # Remove any explanatory text
        if "\n" in cta:
            cta = cta.split("\n")[0]
        
        # Limit length
        if len(cta) > 100:
            cta = cta[:97] + "..."
        
        return cta.strip()
    
    def _determine_time_slot(self, hour: int) -> PostingTimeSlot:
        """Determine time slot from hour."""
        if 6 <= hour < 9:
            return PostingTimeSlot.EARLY_MORNING
        elif 9 <= hour < 12:
            return PostingTimeSlot.MORNING
        elif 12 <= hour < 15:
            return PostingTimeSlot.AFTERNOON
        elif 15 <= hour < 18:
            return PostingTimeSlot.LATE_AFTERNOON
        elif 18 <= hour < 21:
            return PostingTimeSlot.EVENING
        else:
            return PostingTimeSlot.NIGHT
    
    def _calculate_time_confidence(
        self,
        posting_time: datetime,
        platform: Platform,
        audience: AudienceType,
        content_type: str
    ) -> float:
        """Calculate confidence score for posting time."""
        base_confidence = 0.7
        
        # Adjust for day of week
        day_of_week = posting_time.weekday()
        if platform == Platform.LINKEDIN:
            # Weekdays are better for LinkedIn
            if day_of_week < 5:
                base_confidence += 0.15
        elif platform in [Platform.INSTAGRAM, Platform.TIKTOK]:
            # Weekends are good for entertainment platforms
            if day_of_week >= 5:
                base_confidence += 0.1
        
        # Adjust for time of day
        hour = posting_time.hour
        if platform == Platform.LINKEDIN and 8 <= hour <= 17:
            base_confidence += 0.1
        elif platform in [Platform.INSTAGRAM, Platform.TIKTOK] and 18 <= hour <= 22:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _estimate_reach_for_time(
        self,
        posting_time: datetime,
        platform: Platform,
        audience: AudienceType
    ) -> int:
        """Estimate expected reach for posting time."""
        # Base reach varies by platform
        base_reach = {
            Platform.TWITTER: 500,
            Platform.INSTAGRAM: 800,
            Platform.FACEBOOK: 600,
            Platform.LINKEDIN: 400,
            Platform.TIKTOK: 1000,
            Platform.YOUTUBE: 700
        }.get(platform, 500)
        
        # Adjust for time of day
        hour = posting_time.hour
        if 12 <= hour <= 14 or 18 <= hour <= 21:
            base_reach = int(base_reach * 1.3)  # Peak times
        
        return base_reach
    
    def _generate_time_rationale(
        self,
        posting_time: datetime,
        platform: Platform,
        time_slot: PostingTimeSlot,
        audience: AudienceType
    ) -> str:
        """Generate rationale for posting time suggestion."""
        day_name = posting_time.strftime("%A")
        time_str = posting_time.strftime("%I:%M %p")
        
        rationales = {
            PostingTimeSlot.MORNING: f"Morning engagement on {platform.value} is typically high",
            PostingTimeSlot.AFTERNOON: f"Lunch break browsing peaks on {platform.value}",
            PostingTimeSlot.EVENING: f"Evening is prime time for {platform.value} engagement",
            PostingTimeSlot.LATE_AFTERNOON: f"Post-work browsing increases on {platform.value}"
        }
        
        base_rationale = rationales.get(time_slot, f"Good engagement window for {platform.value}")
        return f"{day_name} at {time_str}: {base_rationale}"

    
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality (0-1)."""
        score = 0.5  # Base score
        
        # Length factor
        word_count = len(content.split())
        if 50 <= word_count <= 300:
            score += 0.2
        elif 20 <= word_count < 50:
            score += 0.1
        elif word_count < 10:
            score -= 0.2
        
        # Readability factor (simple heuristic)
        if word_count > 0:
            avg_word_length = sum(len(word) for word in content.split()) / word_count
            if 4 <= avg_word_length <= 7:
                score += 0.15
        
        # Engagement indicators
        if "?" in content:  # Questions engage
            score += 0.05
        if any(word in content.lower() for word in ["you", "your"]):  # Direct address
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _assess_timing(self, posting_time: datetime, platform: Platform) -> float:
        """Assess timing quality (0-1)."""
        hour = posting_time.hour
        day_of_week = posting_time.weekday()
        
        # Platform-specific optimal times
        optimal_hours = {
            Platform.TWITTER: [9, 12, 17],
            Platform.INSTAGRAM: [11, 13, 19],
            Platform.FACEBOOK: [13, 15, 19],
            Platform.LINKEDIN: [8, 12, 17],
            Platform.TIKTOK: [18, 19, 21],
            Platform.YOUTUBE: [14, 17, 20]
        }.get(platform, [12, 15, 18])
        
        # Calculate distance from optimal times
        min_distance = min(abs(hour - opt_hour) for opt_hour in optimal_hours)
        
        if min_distance == 0:
            timing_score = 1.0
        elif min_distance <= 2:
            timing_score = 0.8
        elif min_distance <= 4:
            timing_score = 0.6
        else:
            timing_score = 0.4
        
        # Adjust for day of week
        if platform == Platform.LINKEDIN and day_of_week >= 5:
            timing_score *= 0.7  # Weekends are worse for LinkedIn
        
        return timing_score
    
    def _get_platform_engagement_multiplier(self, platform: Platform) -> float:
        """Get platform-specific engagement multiplier."""
        multipliers = {
            Platform.TIKTOK: 0.9,  # High engagement platform
            Platform.INSTAGRAM: 0.85,
            Platform.TWITTER: 0.75,
            Platform.FACEBOOK: 0.7,
            Platform.LINKEDIN: 0.65,
            Platform.YOUTUBE: 0.8
        }
        return multipliers.get(platform, 0.7)
    
    def _predict_reach(self, engagement_score: float, platform: Platform) -> int:
        """Predict reach based on engagement score."""
        # Base reach by platform
        base_reach = {
            Platform.TWITTER: 1000,
            Platform.INSTAGRAM: 1500,
            Platform.FACEBOOK: 1200,
            Platform.LINKEDIN: 800,
            Platform.TIKTOK: 2000,
            Platform.YOUTUBE: 1800
        }.get(platform, 1000)
        
        # Scale by engagement score
        return int(base_reach * (engagement_score / 100) * 2)
    
    def _calculate_prediction_confidence(
        self,
        content: str,
        has_media: bool,
        has_historical: bool
    ) -> float:
        """Calculate confidence in prediction."""
        confidence = 0.6  # Base confidence
        
        if has_media:
            confidence += 0.15
        
        if has_historical:
            confidence += 0.2
        
        if len(content.split()) > 30:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _generate_engagement_recommendations(
        self,
        factors: Dict[str, float],
        platform: Platform,
        has_media: bool,
        hashtag_count: int
    ) -> List[str]:
        """Generate recommendations to improve engagement."""
        recommendations = []
        
        if factors.get("content_quality", 0) < 0.6:
            recommendations.append("Improve content quality with more engaging language")
        
        if factors.get("timing", 0) < 0.7:
            recommendations.append(f"Consider posting during peak times for {platform.value}")
        
        if not has_media:
            recommendations.append("Add images or videos to increase engagement")
        
        optimal_hashtags = self.platform_specs.get(platform, {}).get("optimal_hashtags", 5)
        if hashtag_count < optimal_hashtags:
            recommendations.append(f"Add more hashtags (optimal: {optimal_hashtags} for {platform.value})")
        elif hashtag_count > optimal_hashtags * 2:
            recommendations.append(f"Reduce hashtag count (optimal: {optimal_hashtags} for {platform.value})")
        
        if factors.get("content_quality", 0) > 0.8 and factors.get("timing", 0) > 0.8:
            recommendations.append("Content is well-optimized! Consider A/B testing variations")
        
        return recommendations

    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content using Gemini with error handling."""
        if not self.gemini_client:
            raise EngineError(
                "social_media_planner",
                "Gemini client not initialized. Please configure GOOGLE_API_KEY."
            )
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.gemini_client.generate_content,
                    prompt
                )
                
                if not response or not response.text:
                    raise AIServiceError("gemini", "Empty response received")
                
                return response.text.strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Gemini generation attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"All Gemini generation attempts failed: {e}")
                    raise AIServiceError("gemini", str(e))
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """Get engine usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cost_per_token": self.cost_per_token,
            "calendar_entries": sum(len(entries) for entries in self.calendar.values())
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0
    
    def clear_calendar(self, platform: Optional[Platform] = None):
        """
        Clear calendar entries.
        
        Args:
            platform: Clear specific platform only (optional)
        """
        if platform:
            self.calendar[platform.value] = []
            logger.info(f"Calendar cleared for {platform}")
        else:
            self.calendar = {}
            logger.info("All calendar entries cleared")

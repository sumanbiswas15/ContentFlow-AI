"""
Discovery Analytics Engine for ContentFlow AI.

This module implements the Discovery Analytics Engine responsible for content
tagging, trend analysis, engagement analytics, and AI-powered improvement suggestions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from collections import Counter

import google.generativeai as genai
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.core.exceptions import (
    EngineError, ValidationError, AIServiceError
)
from app.models.base import ContentType, Platform
from app.models.content import EngagementMetrics

logger = logging.getLogger(__name__)


class SentimentType(str, Enum):
    """Enumeration of sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class TrendType(str, Enum):
    """Enumeration of trend types."""
    RISING = "rising"
    DECLINING = "declining"
    STABLE = "stable"
    EMERGING = "emerging"
    VIRAL = "viral"


class TagCategory(str, Enum):
    """Enumeration of tag categories."""
    TOPIC = "topic"
    KEYWORD = "keyword"
    ENTITY = "entity"
    THEME = "theme"
    INDUSTRY = "industry"


class Tag(BaseModel):
    """Model for content tags."""
    name: str
    category: TagCategory
    confidence: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def __hash__(self):
        return hash((self.name, self.category))
    
    def __eq__(self, other):
        if not isinstance(other, Tag):
            return False
        return self.name == other.name and self.category == other.category


class ContentTaggingRequest(BaseModel):
    """Model for content tagging requests."""
    content: str
    content_type: ContentType = ContentType.TEXT
    include_sentiment: bool = True
    include_entities: bool = True
    max_tags: int = Field(default=10, ge=1, le=50)
    language: str = "en"


class TaggingResult(BaseModel):
    """Model for tagging results."""
    tags: List[Tag] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = 0
    cost: float = 0.0


class TrendAnalysisRequest(BaseModel):
    """Model for trend analysis requests."""
    time_period: str = "7d"  # 1d, 7d, 30d, 90d
    content_ids: List[str] = Field(default_factory=list)
    platforms: List[Platform] = Field(default_factory=list)
    min_engagement_threshold: int = 0
    include_predictions: bool = True


class TrendPattern(BaseModel):
    """Model for identified trend patterns."""
    pattern_id: str
    pattern_type: TrendType
    topic: str
    frequency: int
    growth_rate: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0)
    related_keywords: List[str] = Field(default_factory=list)
    time_range: str
    platforms: List[Platform] = Field(default_factory=list)


class TrendAnalysis(BaseModel):
    """Model for trend analysis results."""
    patterns: List[TrendPattern] = Field(default_factory=list)
    emerging_topics: List[str] = Field(default_factory=list)
    declining_topics: List[str] = Field(default_factory=list)
    top_keywords: List[tuple] = Field(default_factory=list)  # (keyword, count)
    analysis_period: str
    total_content_analyzed: int = 0
    insights: List[str] = Field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0


class EngagementAnalysisRequest(BaseModel):
    """Model for engagement analysis requests."""
    content_id: str
    platform: Optional[Platform] = None
    include_comparisons: bool = True
    include_breakdown: bool = True


class EngagementAnalysis(BaseModel):
    """Model for detailed engagement analysis."""
    content_id: str
    metrics: EngagementMetrics
    performance_score: float = Field(ge=0.0, le=100.0)
    platform_breakdown: Dict[Platform, EngagementMetrics] = Field(default_factory=dict)
    time_series_data: List[Dict[str, Any]] = Field(default_factory=list)
    peak_engagement_time: Optional[datetime] = None
    audience_demographics: Dict[str, Any] = Field(default_factory=dict)
    comparison_to_average: float = 0.0  # percentage difference
    insights: List[str] = Field(default_factory=list)


class ImprovementSuggestion(BaseModel):
    """Model for content improvement suggestions."""
    suggestion_id: str
    category: str  # content, timing, platform, engagement, seo
    priority: str = "medium"  # low, medium, high, critical
    title: str
    description: str
    expected_impact: str  # low, medium, high
    implementation_effort: str = "medium"  # low, medium, high
    specific_actions: List[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class ImprovementSuggestionsRequest(BaseModel):
    """Model for improvement suggestions requests."""
    content: str
    content_id: Optional[str] = None
    current_metrics: Optional[EngagementMetrics] = None
    target_platform: Optional[Platform] = None
    goals: List[str] = Field(default_factory=list)
    max_suggestions: int = Field(default=5, ge=1, le=20)


class ImprovementSuggestionsResult(BaseModel):
    """Model for improvement suggestions results."""
    suggestions: List[ImprovementSuggestion] = Field(default_factory=list)
    overall_assessment: str
    priority_actions: List[str] = Field(default_factory=list)
    estimated_improvement_potential: float = Field(default=0.0, ge=0.0, le=100.0)
    tokens_used: int = 0
    cost: float = 0.0


class DiscoveryAnalyticsEngine:
    """
    Discovery Analytics Engine for content analysis and insights.
    
    This engine handles:
    - Automatic content tagging with topics, keywords, and sentiment
    - Trend analysis and pattern discovery
    - Engagement metrics calculation and analysis
    - AI-powered improvement suggestion generation
    """
    
    def __init__(self):
        """Initialize the Discovery Analytics Engine."""
        self.gemini_client = None
        self.content_cache: Dict[str, TaggingResult] = {}
        self.trend_cache: Dict[str, TrendAnalysis] = {}
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cost_per_token = 0.000001  # $0.000001 per token (example rate)
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Google Gemini client."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. Discovery Analytics Engine will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Discovery Analytics Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise EngineError("discovery_analytics", f"Initialization failed: {e}")
    
    async def auto_tag_content(
        self, 
        request: ContentTaggingRequest
    ) -> TaggingResult:
        """
        Automatically tag content with topics, keywords, and sentiment.
        
        Args:
            request: Content tagging request with parameters
            
        Returns:
            TaggingResult with tags, topics, keywords, entities, and sentiment
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If tagging fails
        """
        try:
            logger.info(f"Auto-tagging content ({len(request.content)} chars)")
            
            # Validate request
            self._validate_tagging_request(request)
            
            # Build tagging prompt
            prompt = self._build_tagging_prompt(request)
            
            # Generate tags using Gemini
            tagging_text = await self._generate_with_gemini(prompt)
            
            # Parse tagging results
            result = self._parse_tagging_results(
                tagging_text,
                request.max_tags,
                request.include_sentiment,
                request.include_entities
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(request.content + tagging_text)
            cost = tokens_used * self.cost_per_token
            
            result.tokens_used = tokens_used
            result.cost = cost
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            logger.info(f"Auto-tagging completed: {len(result.tags)} tags, sentiment: {result.sentiment}")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Auto-tagging failed: {e}")
            raise EngineError("discovery_analytics", f"Auto-tagging failed: {e}")

    
    async def analyze_trends(
        self, 
        request: TrendAnalysisRequest,
        content_data: List[Dict[str, Any]]
    ) -> TrendAnalysis:
        """
        Analyze trends and discover patterns in content.
        
        Args:
            request: Trend analysis request with parameters
            content_data: List of content items with metadata for analysis
            
        Returns:
            TrendAnalysis with identified patterns and insights
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If analysis fails
        """
        try:
            logger.info(f"Analyzing trends for {len(content_data)} content items")
            
            # Validate request
            self._validate_trend_request(request)
            
            # Extract and aggregate data
            aggregated_data = self._aggregate_content_data(content_data, request)
            
            # Identify patterns
            patterns = self._identify_patterns(aggregated_data, request.time_period)
            
            # Build trend analysis prompt for AI insights
            prompt = self._build_trend_analysis_prompt(aggregated_data, patterns, request)
            
            # Generate AI insights using Gemini
            insights_text = await self._generate_with_gemini(prompt)
            
            # Parse insights
            insights = self._parse_insights(insights_text)
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(str(aggregated_data) + insights_text)
            cost = tokens_used * self.cost_per_token
            
            # Create result
            result = TrendAnalysis(
                patterns=patterns,
                emerging_topics=aggregated_data.get("emerging_topics", []),
                declining_topics=aggregated_data.get("declining_topics", []),
                top_keywords=aggregated_data.get("top_keywords", []),
                analysis_period=request.time_period,
                total_content_analyzed=len(content_data),
                insights=insights,
                tokens_used=tokens_used,
                cost=cost
            )
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            logger.info(f"Trend analysis completed: {len(patterns)} patterns identified")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise EngineError("discovery_analytics", f"Trend analysis failed: {e}")

    
    async def calculate_engagement_metrics(
        self, 
        request: EngagementAnalysisRequest,
        raw_metrics: EngagementMetrics,
        historical_data: Optional[List[EngagementMetrics]] = None
    ) -> EngagementAnalysis:
        """
        Calculate and analyze engagement metrics for content.
        
        Args:
            request: Engagement analysis request
            raw_metrics: Raw engagement metrics for the content
            historical_data: Historical metrics for comparison (optional)
            
        Returns:
            EngagementAnalysis with detailed metrics and insights
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If calculation fails
        """
        try:
            logger.info(f"Calculating engagement metrics for content {request.content_id}")
            
            # Validate request
            self._validate_engagement_request(request)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(raw_metrics)
            
            # Calculate comparison to average if historical data available
            comparison = 0.0
            if historical_data:
                comparison = self._calculate_comparison_to_average(raw_metrics, historical_data)
            
            # Generate insights
            insights = self._generate_engagement_insights(
                raw_metrics,
                performance_score,
                comparison,
                request.platform
            )
            
            # Create result
            result = EngagementAnalysis(
                content_id=request.content_id,
                metrics=raw_metrics,
                performance_score=performance_score,
                comparison_to_average=comparison,
                insights=insights
            )
            
            logger.info(f"Engagement analysis completed: score {performance_score:.2f}/100")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Engagement metrics calculation failed: {e}")
            raise EngineError("discovery_analytics", f"Metrics calculation failed: {e}")

    
    async def generate_improvement_suggestions(
        self, 
        request: ImprovementSuggestionsRequest
    ) -> ImprovementSuggestionsResult:
        """
        Generate AI-powered improvement suggestions for content.
        
        Args:
            request: Improvement suggestions request with content and context
            
        Returns:
            ImprovementSuggestionsResult with actionable suggestions
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If generation fails
        """
        try:
            logger.info(f"Generating improvement suggestions for content")
            
            # Validate request
            self._validate_improvement_request(request)
            
            # Build improvement suggestions prompt
            prompt = self._build_improvement_prompt(request)
            
            # Generate suggestions using Gemini
            suggestions_text = await self._generate_with_gemini(prompt)
            
            # Parse suggestions
            suggestions = self._parse_improvement_suggestions(
                suggestions_text,
                request.max_suggestions
            )
            
            # Generate overall assessment
            assessment = self._generate_overall_assessment(
                request.content,
                request.current_metrics,
                suggestions
            )
            
            # Extract priority actions
            priority_actions = [
                s.title for s in suggestions 
                if s.priority in ["high", "critical"]
            ][:3]
            
            # Estimate improvement potential
            improvement_potential = self._estimate_improvement_potential(
                suggestions,
                request.current_metrics
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(request.content + suggestions_text)
            cost = tokens_used * self.cost_per_token
            
            # Create result
            result = ImprovementSuggestionsResult(
                suggestions=suggestions,
                overall_assessment=assessment,
                priority_actions=priority_actions,
                estimated_improvement_potential=improvement_potential,
                tokens_used=tokens_used,
                cost=cost
            )
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            logger.info(f"Generated {len(suggestions)} improvement suggestions")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Improvement suggestions generation failed: {e}")
            raise EngineError("discovery_analytics", f"Suggestion generation failed: {e}")

    
    # Private helper methods
    
    def _validate_tagging_request(self, request: ContentTaggingRequest):
        """Validate content tagging request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if len(request.content) > 100000:
            raise ValidationError("Content exceeds maximum length of 100000 characters")
        
        if request.max_tags < 1 or request.max_tags > 50:
            raise ValidationError("Max tags must be between 1 and 50")
    
    def _validate_trend_request(self, request: TrendAnalysisRequest):
        """Validate trend analysis request."""
        valid_periods = ["1d", "7d", "30d", "90d"]
        if request.time_period not in valid_periods:
            raise ValidationError(f"Time period must be one of: {', '.join(valid_periods)}")
    
    def _validate_engagement_request(self, request: EngagementAnalysisRequest):
        """Validate engagement analysis request."""
        if not request.content_id:
            raise ValidationError("Content ID is required")
    
    def _validate_improvement_request(self, request: ImprovementSuggestionsRequest):
        """Validate improvement suggestions request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if request.max_suggestions < 1 or request.max_suggestions > 20:
            raise ValidationError("Max suggestions must be between 1 and 20")

    
    def _build_tagging_prompt(self, request: ContentTaggingRequest) -> str:
        """Build prompt for content tagging."""
        prompt_parts = [
            "Analyze the following content and provide comprehensive tagging information.",
            f"\n\nContent:\n{request.content[:5000]}"  # Limit to first 5000 chars
        ]
        
        prompt_parts.append("\n\nProvide the following information:")
        prompt_parts.append(f"\n1. Up to {request.max_tags} relevant tags (categorized as topic, keyword, entity, theme, or industry)")
        prompt_parts.append("\n2. Main topics (3-5 key topics)")
        prompt_parts.append("\n3. Important keywords (5-10 keywords)")
        
        if request.include_entities:
            prompt_parts.append("\n4. Named entities (people, organizations, locations)")
        
        if request.include_sentiment:
            prompt_parts.append("\n5. Overall sentiment (positive, negative, neutral, or mixed) with a score from -1 to 1")
        
        prompt_parts.append("\n\nFormat your response as:")
        prompt_parts.append("\nTAGS: tag1:category:confidence, tag2:category:confidence, ...")
        prompt_parts.append("\nTOPICS: topic1, topic2, topic3")
        prompt_parts.append("\nKEYWORDS: keyword1, keyword2, keyword3")
        
        if request.include_entities:
            prompt_parts.append("\nENTITIES: entity1, entity2, entity3")
        
        if request.include_sentiment:
            prompt_parts.append("\nSENTIMENT: type:score")
        
        return "".join(prompt_parts)
    
    def _build_trend_analysis_prompt(
        self, 
        aggregated_data: Dict[str, Any],
        patterns: List[TrendPattern],
        request: TrendAnalysisRequest
    ) -> str:
        """Build prompt for trend analysis insights."""
        prompt_parts = [
            f"Analyze the following content trends over the past {request.time_period}.",
            f"\n\nTotal content analyzed: {aggregated_data.get('total_items', 0)}",
            f"\nTop keywords: {', '.join([k for k, _ in aggregated_data.get('top_keywords', [])[:10]])}",
            f"\nEmerging topics: {', '.join(aggregated_data.get('emerging_topics', [])[:5])}",
            f"\nDeclining topics: {', '.join(aggregated_data.get('declining_topics', [])[:5])}"
        ]
        
        if patterns:
            prompt_parts.append(f"\n\nIdentified {len(patterns)} trend patterns:")
            for i, pattern in enumerate(patterns[:5], 1):
                prompt_parts.append(
                    f"\n{i}. {pattern.topic} - {pattern.pattern_type.value} "
                    f"(frequency: {pattern.frequency}, growth: {pattern.growth_rate:.1%})"
                )
        
        prompt_parts.append("\n\nProvide 5-7 key insights about these trends.")
        prompt_parts.append("Focus on actionable observations and strategic recommendations.")
        prompt_parts.append("Format each insight on a new line starting with a dash (-).")
        
        return "".join(prompt_parts)

    
    def _build_improvement_prompt(self, request: ImprovementSuggestionsRequest) -> str:
        """Build prompt for improvement suggestions."""
        prompt_parts = [
            "Analyze the following content and provide specific improvement suggestions.",
            f"\n\nContent:\n{request.content[:5000]}"  # Limit to first 5000 chars
        ]
        
        if request.current_metrics:
            prompt_parts.append(f"\n\nCurrent Performance:")
            prompt_parts.append(f"\n- Views: {request.current_metrics.views}")
            prompt_parts.append(f"\n- Likes: {request.current_metrics.likes}")
            prompt_parts.append(f"\n- Shares: {request.current_metrics.shares}")
            prompt_parts.append(f"\n- Comments: {request.current_metrics.comments}")
            prompt_parts.append(f"\n- Engagement Rate: {request.current_metrics.engagement_rate:.2f}%")
        
        if request.target_platform:
            prompt_parts.append(f"\n\nTarget Platform: {request.target_platform.value}")
        
        if request.goals:
            prompt_parts.append(f"\n\nGoals: {', '.join(request.goals)}")
        
        prompt_parts.append(f"\n\nProvide up to {request.max_suggestions} specific, actionable improvement suggestions.")
        prompt_parts.append("\nFor each suggestion, include:")
        prompt_parts.append("\n- Category (content, timing, platform, engagement, or seo)")
        prompt_parts.append("\n- Priority (low, medium, high, or critical)")
        prompt_parts.append("\n- Title (brief, clear title)")
        prompt_parts.append("\n- Description (detailed explanation)")
        prompt_parts.append("\n- Expected Impact (low, medium, or high)")
        prompt_parts.append("\n- Implementation Effort (low, medium, or high)")
        prompt_parts.append("\n- Specific Actions (2-4 concrete steps)")
        prompt_parts.append("\n- Rationale (why this will help)")
        
        prompt_parts.append("\n\nFormat each suggestion as:")
        prompt_parts.append("\nSUGGESTION N:")
        prompt_parts.append("\nCategory: [category]")
        prompt_parts.append("\nPriority: [priority]")
        prompt_parts.append("\nTitle: [title]")
        prompt_parts.append("\nDescription: [description]")
        prompt_parts.append("\nExpected Impact: [impact]")
        prompt_parts.append("\nImplementation Effort: [effort]")
        prompt_parts.append("\nActions: [action1], [action2], [action3]")
        prompt_parts.append("\nRationale: [rationale]")
        
        return "".join(prompt_parts)

    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content using Gemini with error handling."""
        if not self.gemini_client:
            raise EngineError(
                "discovery_analytics",
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
    
    def _parse_tagging_results(
        self, 
        text: str, 
        max_tags: int,
        include_sentiment: bool,
        include_entities: bool
    ) -> TaggingResult:
        """Parse tagging results from generated text."""
        result = TaggingResult()
        
        lines = text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse tags
            if line.startswith("TAGS:"):
                tags_str = line.replace("TAGS:", "").strip()
                tag_items = [t.strip() for t in tags_str.split(",") if t.strip()]
                
                for tag_item in tag_items[:max_tags]:
                    parts = tag_item.split(":")
                    if len(parts) >= 2:
                        tag_name = parts[0].strip()
                        tag_category = parts[1].strip().lower()
                        confidence = float(parts[2]) if len(parts) > 2 else 0.8
                        
                        # Map category string to enum
                        category_map = {
                            "topic": TagCategory.TOPIC,
                            "keyword": TagCategory.KEYWORD,
                            "entity": TagCategory.ENTITY,
                            "theme": TagCategory.THEME,
                            "industry": TagCategory.INDUSTRY
                        }
                        category = category_map.get(tag_category, TagCategory.KEYWORD)
                        
                        tag = Tag(
                            name=tag_name,
                            category=category,
                            confidence=confidence,
                            relevance_score=confidence
                        )
                        result.tags.append(tag)
            
            # Parse topics
            elif line.startswith("TOPICS:"):
                topics_str = line.replace("TOPICS:", "").strip()
                result.topics = [t.strip() for t in topics_str.split(",") if t.strip()]
            
            # Parse keywords
            elif line.startswith("KEYWORDS:"):
                keywords_str = line.replace("KEYWORDS:", "").strip()
                result.keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            
            # Parse entities
            elif line.startswith("ENTITIES:") and include_entities:
                entities_str = line.replace("ENTITIES:", "").strip()
                result.entities = [e.strip() for e in entities_str.split(",") if e.strip()]
            
            # Parse sentiment
            elif line.startswith("SENTIMENT:") and include_sentiment:
                sentiment_str = line.replace("SENTIMENT:", "").strip()
                parts = sentiment_str.split(":")
                if len(parts) >= 2:
                    sentiment_type = parts[0].strip().lower()
                    sentiment_score = float(parts[1].strip())
                    
                    # Map sentiment string to enum
                    sentiment_map = {
                        "positive": SentimentType.POSITIVE,
                        "negative": SentimentType.NEGATIVE,
                        "neutral": SentimentType.NEUTRAL,
                        "mixed": SentimentType.MIXED
                    }
                    result.sentiment = sentiment_map.get(sentiment_type, SentimentType.NEUTRAL)
                    result.sentiment_score = sentiment_score
        
        # Calculate overall confidence
        if result.tags:
            result.confidence = sum(t.confidence for t in result.tags) / len(result.tags)
        else:
            result.confidence = 0.5
        
        return result

    
    def _aggregate_content_data(
        self, 
        content_data: List[Dict[str, Any]],
        request: TrendAnalysisRequest
    ) -> Dict[str, Any]:
        """Aggregate content data for trend analysis."""
        # Extract all tags and keywords
        all_keywords = []
        all_topics = []
        topic_frequency = Counter()
        keyword_frequency = Counter()
        
        for item in content_data:
            # Extract keywords
            keywords = item.get("keywords", [])
            all_keywords.extend(keywords)
            keyword_frequency.update(keywords)
            
            # Extract topics
            topics = item.get("topics", [])
            all_topics.extend(topics)
            topic_frequency.update(topics)
        
        # Identify emerging and declining topics
        # For simplicity, use frequency thresholds
        total_items = len(content_data)
        emerging_threshold = max(2, total_items * 0.1)  # 10% of content
        
        emerging_topics = [
            topic for topic, count in topic_frequency.most_common()
            if count >= emerging_threshold
        ][:10]
        
        # Declining topics would require historical comparison
        # For now, use less frequent topics
        declining_topics = [
            topic for topic, count in topic_frequency.most_common()
            if count < emerging_threshold and count > 0
        ][-10:]
        
        return {
            "total_items": total_items,
            "top_keywords": keyword_frequency.most_common(20),
            "top_topics": topic_frequency.most_common(20),
            "emerging_topics": emerging_topics,
            "declining_topics": declining_topics,
            "unique_keywords": len(keyword_frequency),
            "unique_topics": len(topic_frequency)
        }
    
    def _identify_patterns(
        self, 
        aggregated_data: Dict[str, Any],
        time_period: str
    ) -> List[TrendPattern]:
        """Identify trend patterns from aggregated data."""
        patterns = []
        
        # Analyze top topics for patterns
        top_topics = aggregated_data.get("top_topics", [])
        emerging_topics = aggregated_data.get("emerging_topics", [])
        
        for topic, frequency in top_topics[:10]:
            # Determine pattern type
            if topic in emerging_topics:
                pattern_type = TrendType.EMERGING
                growth_rate = 0.5  # Placeholder
            elif frequency > aggregated_data["total_items"] * 0.3:
                pattern_type = TrendType.VIRAL
                growth_rate = 0.8
            else:
                pattern_type = TrendType.STABLE
                growth_rate = 0.0
            
            # Get related keywords
            related_keywords = [
                kw for kw, _ in aggregated_data.get("top_keywords", [])
                if topic.lower() in kw.lower() or kw.lower() in topic.lower()
            ][:5]
            
            pattern = TrendPattern(
                pattern_id=f"pattern_{len(patterns)}",
                pattern_type=pattern_type,
                topic=topic,
                frequency=frequency,
                growth_rate=growth_rate,
                confidence=min(1.0, frequency / aggregated_data["total_items"]),
                related_keywords=related_keywords,
                time_range=time_period,
                platforms=[]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _parse_insights(self, text: str) -> List[str]:
        """Parse insights from generated text."""
        insights = []
        
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                insight = line.lstrip("-•").strip()
                if insight:
                    insights.append(insight)
        
        return insights

    
    def _calculate_performance_score(self, metrics: EngagementMetrics) -> float:
        """Calculate overall performance score from engagement metrics."""
        # Weighted scoring system
        weights = {
            "engagement_rate": 0.35,
            "reach": 0.25,
            "shares": 0.20,
            "comments": 0.15,
            "likes": 0.05
        }
        
        # Normalize metrics to 0-100 scale
        # These thresholds are examples and should be adjusted based on platform
        normalized_scores = {
            "engagement_rate": min(100, metrics.engagement_rate * 10),  # 10% = 100 points
            "reach": min(100, metrics.reach / 100),  # 10k reach = 100 points
            "shares": min(100, metrics.shares / 10),  # 1k shares = 100 points
            "comments": min(100, metrics.comments / 10),  # 1k comments = 100 points
            "likes": min(100, metrics.likes / 100)  # 10k likes = 100 points
        }
        
        # Calculate weighted score
        score = sum(
            normalized_scores[metric] * weight
            for metric, weight in weights.items()
        )
        
        return round(score, 2)
    
    def _calculate_comparison_to_average(
        self, 
        current: EngagementMetrics,
        historical: List[EngagementMetrics]
    ) -> float:
        """Calculate percentage difference from historical average."""
        if not historical:
            return 0.0
        
        # Calculate average engagement rate from historical data
        avg_engagement = sum(m.engagement_rate for m in historical) / len(historical)
        
        if avg_engagement == 0:
            return 0.0
        
        # Calculate percentage difference
        difference = ((current.engagement_rate - avg_engagement) / avg_engagement) * 100
        
        return round(difference, 2)
    
    def _generate_engagement_insights(
        self,
        metrics: EngagementMetrics,
        performance_score: float,
        comparison: float,
        platform: Optional[Platform]
    ) -> List[str]:
        """Generate insights from engagement metrics."""
        insights = []
        
        # Performance assessment
        if performance_score >= 80:
            insights.append("Excellent performance - content is resonating strongly with the audience")
        elif performance_score >= 60:
            insights.append("Good performance - content is performing above average")
        elif performance_score >= 40:
            insights.append("Moderate performance - there's room for improvement")
        else:
            insights.append("Low performance - consider revising content strategy")
        
        # Comparison insight
        if comparison > 20:
            insights.append(f"Performance is {comparison:.1f}% above your average - keep up the good work!")
        elif comparison < -20:
            insights.append(f"Performance is {comparison:.1f}% below your average - analyze what's different")
        
        # Engagement rate insight
        if metrics.engagement_rate > 5:
            insights.append("High engagement rate indicates strong audience connection")
        elif metrics.engagement_rate < 1:
            insights.append("Low engagement rate - consider more interactive content")
        
        # Shares insight
        if metrics.shares > metrics.likes * 0.2:
            insights.append("High share rate suggests valuable, shareable content")
        elif metrics.shares < metrics.likes * 0.05:
            insights.append("Low share rate - add more shareable elements or calls-to-action")
        
        # Comments insight
        if metrics.comments > metrics.likes * 0.1:
            insights.append("Strong comment activity indicates engaging, discussion-worthy content")
        
        return insights

    
    def _parse_improvement_suggestions(
        self, 
        text: str,
        max_suggestions: int
    ) -> List[ImprovementSuggestion]:
        """Parse improvement suggestions from generated text."""
        suggestions = []
        
        # Split by suggestion markers
        suggestion_blocks = []
        current_block = []
        
        lines = text.strip().split("\n")
        for line in lines:
            if line.strip().startswith("SUGGESTION"):
                if current_block:
                    suggestion_blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        
        if current_block:
            suggestion_blocks.append("\n".join(current_block))
        
        # Parse each suggestion block
        for i, block in enumerate(suggestion_blocks[:max_suggestions]):
            suggestion_data = {
                "category": "content",
                "priority": "medium",
                "title": "",
                "description": "",
                "expected_impact": "medium",
                "implementation_effort": "medium",
                "specific_actions": [],
                "rationale": ""
            }
            
            for line in block.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("Category:"):
                    suggestion_data["category"] = line.replace("Category:", "").strip().lower()
                elif line.startswith("Priority:"):
                    suggestion_data["priority"] = line.replace("Priority:", "").strip().lower()
                elif line.startswith("Title:"):
                    suggestion_data["title"] = line.replace("Title:", "").strip()
                elif line.startswith("Description:"):
                    suggestion_data["description"] = line.replace("Description:", "").strip()
                elif line.startswith("Expected Impact:"):
                    suggestion_data["expected_impact"] = line.replace("Expected Impact:", "").strip().lower()
                elif line.startswith("Implementation Effort:"):
                    suggestion_data["implementation_effort"] = line.replace("Implementation Effort:", "").strip().lower()
                elif line.startswith("Actions:"):
                    actions_str = line.replace("Actions:", "").strip()
                    suggestion_data["specific_actions"] = [
                        a.strip() for a in actions_str.split(",") if a.strip()
                    ]
                elif line.startswith("Rationale:"):
                    suggestion_data["rationale"] = line.replace("Rationale:", "").strip()
            
            # Create suggestion if we have minimum required fields
            if suggestion_data["title"] and suggestion_data["description"]:
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"suggestion_{i+1}",
                    category=suggestion_data["category"],
                    priority=suggestion_data["priority"],
                    title=suggestion_data["title"],
                    description=suggestion_data["description"],
                    expected_impact=suggestion_data["expected_impact"],
                    implementation_effort=suggestion_data["implementation_effort"],
                    specific_actions=suggestion_data["specific_actions"],
                    rationale=suggestion_data["rationale"],
                    confidence=0.8
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_overall_assessment(
        self,
        content: str,
        current_metrics: Optional[EngagementMetrics],
        suggestions: List[ImprovementSuggestion]
    ) -> str:
        """Generate overall assessment of content."""
        assessment_parts = []
        
        # Content length assessment
        word_count = len(content.split())
        if word_count < 50:
            assessment_parts.append("Content is quite brief")
        elif word_count > 1000:
            assessment_parts.append("Content is comprehensive and detailed")
        else:
            assessment_parts.append("Content length is appropriate")
        
        # Performance assessment
        if current_metrics:
            if current_metrics.engagement_rate > 5:
                assessment_parts.append("with strong engagement")
            elif current_metrics.engagement_rate < 1:
                assessment_parts.append("but engagement could be improved")
        
        # Suggestions assessment
        high_priority_count = sum(1 for s in suggestions if s.priority in ["high", "critical"])
        if high_priority_count > 0:
            assessment_parts.append(f"with {high_priority_count} high-priority improvements identified")
        else:
            assessment_parts.append("with opportunities for optimization")
        
        return ". ".join(assessment_parts) + "."
    
    def _estimate_improvement_potential(
        self,
        suggestions: List[ImprovementSuggestion],
        current_metrics: Optional[EngagementMetrics]
    ) -> float:
        """Estimate potential improvement percentage."""
        if not suggestions:
            return 0.0
        
        # Base potential on suggestion priorities and expected impacts
        impact_scores = {
            "low": 5,
            "medium": 15,
            "high": 30
        }
        
        priority_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
            "critical": 2.0
        }
        
        total_potential = 0.0
        for suggestion in suggestions:
            base_impact = impact_scores.get(suggestion.expected_impact, 15)
            multiplier = priority_multipliers.get(suggestion.priority, 1.0)
            total_potential += base_impact * multiplier
        
        # Cap at 100%
        return min(100.0, round(total_potential, 2))
    
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
            "cached_tagging_results": len(self.content_cache),
            "cached_trend_analyses": len(self.trend_cache)
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0
    
    def clear_caches(self):
        """Clear all caches."""
        self.content_cache.clear()
        self.trend_cache.clear()
        logger.info("Caches cleared")

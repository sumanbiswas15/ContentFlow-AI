"""
Unit tests for Discovery Analytics Engine.

Tests cover:
- Automatic content tagging with topics, keywords, and sentiment
- Trend analysis and pattern discovery
- Engagement metrics calculation
- AI-powered improvement suggestions
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.ai.discovery_analytics_engine import (
    DiscoveryAnalyticsEngine,
    ContentTaggingRequest,
    TaggingResult,
    Tag,
    TagCategory,
    SentimentType,
    TrendAnalysisRequest,
    TrendAnalysis,
    TrendPattern,
    TrendType,
    EngagementAnalysisRequest,
    EngagementAnalysis,
    ImprovementSuggestionsRequest,
    ImprovementSuggestionsResult,
    ImprovementSuggestion
)
from app.models.base import ContentType, Platform
from app.models.content import EngagementMetrics
from app.core.exceptions import ValidationError, EngineError, AIServiceError


@pytest.fixture
def engine():
    """Create a Discovery Analytics Engine instance for testing."""
    with patch('app.ai.discovery_analytics_engine.genai'):
        engine = DiscoveryAnalyticsEngine()
        # Mock the Gemini client
        engine.gemini_client = Mock()
        return engine


@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return """
    Artificial Intelligence is transforming the technology industry.
    Machine learning and deep learning are key components of modern AI systems.
    Companies are investing heavily in AI research and development.
    """


@pytest.fixture
def sample_engagement_metrics():
    """Sample engagement metrics for testing."""
    return EngagementMetrics(
        views=10000,
        likes=500,
        shares=100,
        comments=50,
        reach=8000,
        impressions=12000,
        engagement_rate=6.25
    )



class TestContentTagging:
    """Tests for automatic content tagging functionality."""
    
    @pytest.mark.asyncio
    async def test_auto_tag_content_success(self, engine, sample_content):
        """Test successful content tagging."""
        # Mock Gemini response
        mock_response = """
        TAGS: AI:topic:0.9, machine learning:keyword:0.85, technology:industry:0.8
        TOPICS: Artificial Intelligence, Machine Learning, Technology Industry
        KEYWORDS: AI, machine learning, deep learning, research, development
        ENTITIES: None
        SENTIMENT: positive:0.7
        """
        
        engine.gemini_client.generate_content = Mock(
            return_value=Mock(text=mock_response)
        )
        
        request = ContentTaggingRequest(
            content=sample_content,
            max_tags=10,
            include_sentiment=True,
            include_entities=True
        )
        
        result = await engine.auto_tag_content(request)
        
        assert isinstance(result, TaggingResult)
        assert len(result.tags) > 0
        assert len(result.topics) > 0
        assert len(result.keywords) > 0
        assert result.sentiment == SentimentType.POSITIVE
        assert result.sentiment_score == 0.7
        assert result.tokens_used > 0
        assert result.cost > 0
    
    @pytest.mark.asyncio
    async def test_auto_tag_content_empty_content(self, engine):
        """Test tagging with empty content raises ValidationError."""
        request = ContentTaggingRequest(
            content="",
            max_tags=10
        )
        
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            await engine.auto_tag_content(request)
    
    @pytest.mark.asyncio
    async def test_auto_tag_content_too_long(self, engine):
        """Test tagging with content exceeding max length."""
        request = ContentTaggingRequest(
            content="x" * 100001,  # Exceeds 100000 char limit
            max_tags=10
        )
        
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            await engine.auto_tag_content(request)
    
    @pytest.mark.asyncio
    async def test_auto_tag_content_invalid_max_tags(self, engine, sample_content):
        """Test tagging with invalid max_tags parameter."""
        # Pydantic validates this at model creation, so we expect a Pydantic ValidationError
        with pytest.raises(Exception):  # Pydantic validation error
            request = ContentTaggingRequest(
                content=sample_content,
                max_tags=100  # Exceeds limit of 50
            )
    
    @pytest.mark.asyncio
    async def test_parse_tagging_results(self, engine):
        """Test parsing of tagging results."""
        mock_response = """
        TAGS: AI:topic:0.9, machine learning:keyword:0.85
        TOPICS: Artificial Intelligence, Machine Learning
        KEYWORDS: AI, ML, deep learning
        ENTITIES: Google, OpenAI
        SENTIMENT: positive:0.8
        """
        
        result = engine._parse_tagging_results(
            mock_response,
            max_tags=10,
            include_sentiment=True,
            include_entities=True
        )
        
        assert len(result.tags) == 2
        assert result.tags[0].name == "AI"
        assert result.tags[0].category == TagCategory.TOPIC
        assert result.tags[0].confidence == 0.9
        
        assert "Artificial Intelligence" in result.topics
        assert "AI" in result.keywords
        assert "Google" in result.entities
        assert result.sentiment == SentimentType.POSITIVE
        assert result.sentiment_score == 0.8



class TestTrendAnalysis:
    """Tests for trend analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_trends_success(self, engine):
        """Test successful trend analysis."""
        # Mock content data
        content_data = [
            {
                "id": "1",
                "keywords": ["AI", "machine learning", "technology"],
                "topics": ["Artificial Intelligence", "Technology"]
            },
            {
                "id": "2",
                "keywords": ["AI", "deep learning", "neural networks"],
                "topics": ["Artificial Intelligence", "Deep Learning"]
            },
            {
                "id": "3",
                "keywords": ["blockchain", "cryptocurrency", "technology"],
                "topics": ["Blockchain", "Technology"]
            }
        ]
        
        # Mock Gemini response
        mock_insights = """
        - AI and machine learning are dominant trends
        - Technology topics show consistent engagement
        - Emerging interest in blockchain and cryptocurrency
        - Deep learning gaining traction
        - Cross-platform content performing well
        """
        
        engine.gemini_client.generate_content = Mock(
            return_value=Mock(text=mock_insights)
        )
        
        request = TrendAnalysisRequest(
            time_period="7d",
            include_predictions=True
        )
        
        result = await engine.analyze_trends(request, content_data)
        
        assert isinstance(result, TrendAnalysis)
        assert len(result.patterns) > 0
        assert len(result.insights) > 0
        assert result.total_content_analyzed == 3
        assert result.analysis_period == "7d"
        assert result.tokens_used > 0
    
    @pytest.mark.asyncio
    async def test_analyze_trends_invalid_period(self, engine):
        """Test trend analysis with invalid time period."""
        request = TrendAnalysisRequest(
            time_period="invalid"
        )
        
        with pytest.raises(ValidationError, match="Time period must be one of"):
            await engine.analyze_trends(request, [])
    
    def test_aggregate_content_data(self, engine):
        """Test content data aggregation."""
        content_data = [
            {
                "keywords": ["AI", "ML"],
                "topics": ["Technology", "AI"]
            },
            {
                "keywords": ["AI", "deep learning"],
                "topics": ["Technology", "Deep Learning"]
            }
        ]
        
        request = TrendAnalysisRequest(time_period="7d")
        aggregated = engine._aggregate_content_data(content_data, request)
        
        assert aggregated["total_items"] == 2
        assert len(aggregated["top_keywords"]) > 0
        assert len(aggregated["top_topics"]) > 0
        assert "AI" in [kw for kw, _ in aggregated["top_keywords"]]
    
    def test_identify_patterns(self, engine):
        """Test pattern identification."""
        aggregated_data = {
            "total_items": 10,
            "top_topics": [("AI", 8), ("Technology", 6), ("Blockchain", 3)],
            "top_keywords": [("machine learning", 7), ("AI", 8)],
            "emerging_topics": ["AI", "Technology"],
            "declining_topics": []
        }
        
        patterns = engine._identify_patterns(aggregated_data, "7d")
        
        assert len(patterns) > 0
        assert all(isinstance(p, TrendPattern) for p in patterns)
        assert patterns[0].topic == "AI"
        assert patterns[0].frequency == 8



class TestEngagementMetrics:
    """Tests for engagement metrics calculation."""
    
    @pytest.mark.asyncio
    async def test_calculate_engagement_metrics_success(self, engine, sample_engagement_metrics):
        """Test successful engagement metrics calculation."""
        request = EngagementAnalysisRequest(
            content_id="test_content_123",
            platform=Platform.INSTAGRAM,
            include_comparisons=True
        )
        
        result = await engine.calculate_engagement_metrics(
            request,
            sample_engagement_metrics
        )
        
        assert isinstance(result, EngagementAnalysis)
        assert result.content_id == "test_content_123"
        assert result.metrics == sample_engagement_metrics
        assert 0 <= result.performance_score <= 100
        assert len(result.insights) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_engagement_metrics_with_historical(
        self, engine, sample_engagement_metrics
    ):
        """Test engagement metrics calculation with historical data."""
        historical_data = [
            EngagementMetrics(
                views=5000, likes=250, shares=50, comments=25,
                reach=4000, engagement_rate=3.0
            ),
            EngagementMetrics(
                views=6000, likes=300, shares=60, comments=30,
                reach=5000, engagement_rate=4.0
            )
        ]
        
        request = EngagementAnalysisRequest(
            content_id="test_content_123",
            include_comparisons=True
        )
        
        result = await engine.calculate_engagement_metrics(
            request,
            sample_engagement_metrics,
            historical_data
        )
        
        assert result.comparison_to_average != 0.0
        # Current engagement rate (6.25) is higher than historical average (3.5)
        assert result.comparison_to_average > 0
    
    @pytest.mark.asyncio
    async def test_calculate_engagement_metrics_empty_content_id(self, engine):
        """Test engagement metrics with empty content ID."""
        request = EngagementAnalysisRequest(
            content_id="",
            platform=Platform.TWITTER
        )
        
        with pytest.raises(ValidationError, match="Content ID is required"):
            await engine.calculate_engagement_metrics(
                request,
                EngagementMetrics()
            )
    
    def test_calculate_performance_score(self, engine, sample_engagement_metrics):
        """Test performance score calculation."""
        score = engine._calculate_performance_score(sample_engagement_metrics)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
        # With good metrics, score should be reasonably high
        assert score > 30
    
    def test_calculate_performance_score_zero_metrics(self, engine):
        """Test performance score with zero metrics."""
        zero_metrics = EngagementMetrics()
        score = engine._calculate_performance_score(zero_metrics)
        
        assert score == 0.0
    
    def test_generate_engagement_insights(self, engine, sample_engagement_metrics):
        """Test engagement insights generation."""
        insights = engine._generate_engagement_insights(
            sample_engagement_metrics,
            performance_score=75.0,
            comparison=15.0,
            platform=Platform.INSTAGRAM
        )
        
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
        assert any("good" in insight.lower() or "excellent" in insight.lower() 
                  for insight in insights)



class TestImprovementSuggestions:
    """Tests for AI-powered improvement suggestions."""
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_success(self, engine, sample_content):
        """Test successful improvement suggestions generation."""
        # Mock Gemini response
        mock_suggestions = """
        SUGGESTION 1:
        Category: content
        Priority: high
        Title: Enhance Opening Hook
        Description: Add a compelling question or statistic in the first sentence
        Expected Impact: high
        Implementation Effort: low
        Actions: Rewrite opening sentence, Add engaging question, Include relevant statistic
        Rationale: Strong openings increase reader engagement and reduce bounce rates
        
        SUGGESTION 2:
        Category: seo
        Priority: medium
        Title: Optimize Keywords
        Description: Include more relevant keywords naturally throughout the content
        Expected Impact: medium
        Implementation Effort: medium
        Actions: Research target keywords, Integrate keywords naturally, Update meta tags
        Rationale: Better keyword optimization improves search visibility
        """
        
        engine.gemini_client.generate_content = Mock(
            return_value=Mock(text=mock_suggestions)
        )
        
        request = ImprovementSuggestionsRequest(
            content=sample_content,
            content_id="test_123",
            max_suggestions=5
        )
        
        result = await engine.generate_improvement_suggestions(request)
        
        assert isinstance(result, ImprovementSuggestionsResult)
        assert len(result.suggestions) > 0
        assert len(result.suggestions) <= 5
        assert result.overall_assessment
        assert result.estimated_improvement_potential >= 0
        assert result.tokens_used > 0
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_with_metrics(
        self, engine, sample_content, sample_engagement_metrics
    ):
        """Test improvement suggestions with current metrics."""
        mock_suggestions = """
        SUGGESTION 1:
        Category: engagement
        Priority: high
        Title: Add Call-to-Action
        Description: Include clear CTAs to boost engagement
        Expected Impact: high
        Implementation Effort: low
        Actions: Add CTA buttons, Use action verbs, Create urgency
        Rationale: CTAs significantly increase user interaction rates
        """
        
        engine.gemini_client.generate_content = Mock(
            return_value=Mock(text=mock_suggestions)
        )
        
        request = ImprovementSuggestionsRequest(
            content=sample_content,
            current_metrics=sample_engagement_metrics,
            target_platform=Platform.INSTAGRAM,
            goals=["increase engagement", "boost shares"]
        )
        
        result = await engine.generate_improvement_suggestions(request)
        
        assert len(result.suggestions) > 0
        assert len(result.priority_actions) >= 0
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_empty_content(self, engine):
        """Test improvement suggestions with empty content."""
        request = ImprovementSuggestionsRequest(
            content="",
            max_suggestions=5
        )
        
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            await engine.generate_improvement_suggestions(request)
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_invalid_max(self, engine, sample_content):
        """Test improvement suggestions with invalid max_suggestions."""
        # Pydantic validates this at model creation, so we expect a Pydantic ValidationError
        with pytest.raises(Exception):  # Pydantic validation error
            request = ImprovementSuggestionsRequest(
                content=sample_content,
                max_suggestions=100  # Exceeds limit of 20
            )
    
    def test_parse_improvement_suggestions(self, engine):
        """Test parsing of improvement suggestions."""
        mock_text = """
        SUGGESTION 1:
        Category: content
        Priority: high
        Title: Improve Headline
        Description: Make the headline more compelling
        Expected Impact: high
        Implementation Effort: low
        Actions: Use power words, Add numbers, Create curiosity
        Rationale: Headlines are the first thing readers see
        """
        
        suggestions = engine._parse_improvement_suggestions(mock_text, max_suggestions=5)
        
        assert len(suggestions) == 1
        assert suggestions[0].title == "Improve Headline"
        assert suggestions[0].category == "content"
        assert suggestions[0].priority == "high"
        assert len(suggestions[0].specific_actions) > 0
    
    def test_estimate_improvement_potential(self, engine):
        """Test improvement potential estimation."""
        suggestions = [
            ImprovementSuggestion(
                suggestion_id="1",
                category="content",
                priority="high",
                title="Test",
                description="Test description",
                expected_impact="high",
                implementation_effort="low",
                specific_actions=["action1"],
                rationale="test",
                confidence=0.8
            ),
            ImprovementSuggestion(
                suggestion_id="2",
                category="seo",
                priority="medium",
                title="Test 2",
                description="Test description 2",
                expected_impact="medium",
                implementation_effort="medium",
                specific_actions=["action2"],
                rationale="test",
                confidence=0.7
            )
        ]
        
        potential = engine._estimate_improvement_potential(suggestions, None)
        
        assert 0 <= potential <= 100
        assert potential > 0  # Should have some potential with suggestions



class TestEngineUtilities:
    """Tests for engine utility methods."""
    
    def test_estimate_tokens(self, engine):
        """Test token estimation."""
        text = "This is a test sentence with multiple words."
        tokens = engine._estimate_tokens(text)
        
        assert tokens > 0
        assert isinstance(tokens, int)
        # Rough check: should be approximately len(text) / 4
        assert tokens == len(text) // 4
    
    def test_get_usage_stats(self, engine):
        """Test usage statistics retrieval."""
        # Set some usage
        engine.total_tokens_used = 1000
        engine.total_cost = 0.001
        
        stats = engine.get_usage_stats()
        
        assert stats["total_tokens_used"] == 1000
        assert stats["total_cost"] == 0.001
        assert "cost_per_token" in stats
        assert "cached_tagging_results" in stats
        assert "cached_trend_analyses" in stats
    
    def test_reset_usage_stats(self, engine):
        """Test usage statistics reset."""
        engine.total_tokens_used = 1000
        engine.total_cost = 0.001
        
        engine.reset_usage_stats()
        
        assert engine.total_tokens_used == 0
        assert engine.total_cost == 0.0
    
    def test_clear_caches(self, engine):
        """Test cache clearing."""
        # Add some cache entries
        engine.content_cache["test1"] = TaggingResult()
        engine.trend_cache["test2"] = TrendAnalysis(
            patterns=[],
            analysis_period="7d"
        )
        
        engine.clear_caches()
        
        assert len(engine.content_cache) == 0
        assert len(engine.trend_cache) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_gemini_client_not_initialized(self):
        """Test behavior when Gemini client is not initialized."""
        with patch('app.ai.discovery_analytics_engine.genai'):
            engine = DiscoveryAnalyticsEngine()
            engine.gemini_client = None
            
            request = ContentTaggingRequest(
                content="Test content",
                max_tags=5
            )
            
            with pytest.raises(EngineError, match="Gemini client not initialized"):
                await engine.auto_tag_content(request)
    
    @pytest.mark.asyncio
    async def test_gemini_empty_response(self, engine, sample_content):
        """Test handling of empty Gemini response."""
        engine.gemini_client.generate_content = Mock(
            return_value=Mock(text="")
        )
        
        request = ContentTaggingRequest(
            content=sample_content,
            max_tags=5
        )
        
        # Empty response triggers AIServiceError which gets wrapped in EngineError
        with pytest.raises(EngineError, match="Auto-tagging failed"):
            await engine.auto_tag_content(request)
    
    def test_parse_insights_empty_text(self, engine):
        """Test parsing insights from empty text."""
        insights = engine._parse_insights("")
        
        assert insights == []
    
    def test_parse_insights_no_markers(self, engine):
        """Test parsing insights without bullet markers."""
        text = "This is just plain text without markers"
        insights = engine._parse_insights(text)
        
        assert insights == []
    
    def test_calculate_comparison_to_average_empty_historical(self, engine):
        """Test comparison calculation with empty historical data."""
        current = EngagementMetrics(engagement_rate=5.0)
        comparison = engine._calculate_comparison_to_average(current, [])
        
        assert comparison == 0.0
    
    def test_calculate_comparison_to_average_zero_average(self, engine):
        """Test comparison calculation when historical average is zero."""
        current = EngagementMetrics(engagement_rate=5.0)
        historical = [EngagementMetrics(engagement_rate=0.0)]
        
        comparison = engine._calculate_comparison_to_average(current, historical)
        
        assert comparison == 0.0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_content_analysis_workflow(self, engine, sample_content):
        """Test complete workflow: tagging -> metrics -> suggestions."""
        # Mock Gemini responses
        tagging_response = """
        TAGS: AI:topic:0.9, technology:industry:0.8
        TOPICS: Artificial Intelligence, Technology
        KEYWORDS: AI, machine learning, technology
        ENTITIES: None
        SENTIMENT: positive:0.7
        """
        
        suggestions_response = """
        SUGGESTION 1:
        Category: content
        Priority: high
        Title: Add Examples
        Description: Include real-world examples
        Expected Impact: high
        Implementation Effort: medium
        Actions: Research examples, Add case studies, Include statistics
        Rationale: Examples make content more relatable
        """
        
        engine.gemini_client.generate_content = Mock(
            side_effect=[
                Mock(text=tagging_response),
                Mock(text=suggestions_response)
            ]
        )
        
        # Step 1: Tag content
        tagging_request = ContentTaggingRequest(
            content=sample_content,
            max_tags=10
        )
        tagging_result = await engine.auto_tag_content(tagging_request)
        
        assert len(tagging_result.tags) > 0
        assert tagging_result.sentiment == SentimentType.POSITIVE
        
        # Step 2: Generate improvement suggestions
        improvement_request = ImprovementSuggestionsRequest(
            content=sample_content,
            max_suggestions=5
        )
        improvement_result = await engine.generate_improvement_suggestions(improvement_request)
        
        assert len(improvement_result.suggestions) > 0
        assert improvement_result.estimated_improvement_potential > 0
        
        # Verify total usage tracking
        assert engine.total_tokens_used > 0
        assert engine.total_cost > 0

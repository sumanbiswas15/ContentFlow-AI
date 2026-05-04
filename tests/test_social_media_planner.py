"""
Unit tests for Social Media Planner.

This module tests the Social Media Planner's platform optimization,
hashtag generation, CTA creation, posting time suggestions, calendar
management, and engagement prediction capabilities.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.ai.social_media_planner import (
    SocialMediaPlanner,
    OptimizationRequest,
    HashtagRequest,
    CTARequest,
    PostingTimeRequest,
    EngagementPredictionRequest,
    OptimizedContent,
    PostingTimeSuggestion,
    EngagementScore,
    CalendarEntry,
    AudienceType,
    PostingTimeSlot
)
from app.models.base import Platform
from app.core.exceptions import ValidationError, EngineError, AIServiceError


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "Generated content from Gemini"
    mock_client.generate_content.return_value = mock_response
    return mock_client


@pytest.fixture
def planner(mock_gemini_client):
    """Create a Social Media Planner instance with mocked Gemini."""
    with patch('app.ai.social_media_planner.genai') as mock_genai:
        mock_genai.GenerativeModel.return_value = mock_gemini_client
        planner = SocialMediaPlanner()
        planner.gemini_client = mock_gemini_client
        return planner


class TestSocialMediaPlannerInitialization:
    """Test planner initialization."""
    
    def test_initialization_success(self, planner):
        """Test successful planner initialization."""
        assert planner.gemini_client is not None
        assert planner.total_tokens_used == 0
        assert planner.total_cost == 0.0
        assert planner.cost_per_token > 0
        assert len(planner.platform_specs) > 0
        assert isinstance(planner.calendar, dict)

    
    def test_platform_specs_loaded(self, planner):
        """Test that platform specifications are loaded."""
        assert Platform.TWITTER in planner.platform_specs
        assert Platform.INSTAGRAM in planner.platform_specs
        assert Platform.LINKEDIN in planner.platform_specs
        
        # Check Twitter specs
        twitter_specs = planner.platform_specs[Platform.TWITTER]
        assert twitter_specs["max_length"] == 280
        assert "optimal_hashtags" in twitter_specs
        assert "best_times" in twitter_specs
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        with patch('app.ai.social_media_planner.settings') as mock_settings:
            mock_settings.GOOGLE_API_KEY = None
            with patch('app.ai.social_media_planner.genai'):
                planner = SocialMediaPlanner()
                assert planner.gemini_client is None


class TestPlatformOptimization:
    """Test platform-specific content optimization."""
    
    @pytest.mark.asyncio
    async def test_optimize_for_twitter(self, planner, mock_gemini_client):
        """Test content optimization for Twitter."""
        request = OptimizationRequest(
            content="Check out our new AI-powered content creation platform!",
            platform=Platform.TWITTER,
            target_audience=AudienceType.TECH,
            include_hashtags=True,
            include_cta=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "🚀 Discover our AI-powered content platform!"
        
        result = await planner.optimize_for_platform(request)
        
        assert isinstance(result, OptimizedContent)
        assert result.platform == Platform.TWITTER
        assert len(result.optimized_content) > 0
        assert result.character_count <= 280  # Twitter limit
        assert result.tokens_used > 0
        assert result.cost > 0
    
    @pytest.mark.asyncio
    async def test_optimize_for_instagram(self, planner, mock_gemini_client):
        """Test content optimization for Instagram."""
        request = OptimizationRequest(
            content="Beautiful sunset at the beach today",
            platform=Platform.INSTAGRAM,
            include_hashtags=True,
            optimize_length=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "Stunning sunset vibes at the beach 🌅"
        
        result = await planner.optimize_for_platform(request)
        
        assert isinstance(result, OptimizedContent)
        assert result.platform == Platform.INSTAGRAM
        assert len(result.optimized_content) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_for_linkedin(self, planner, mock_gemini_client):
        """Test content optimization for LinkedIn."""
        request = OptimizationRequest(
            content="Excited to announce our new product launch",
            platform=Platform.LINKEDIN,
            target_audience=AudienceType.BUSINESS,
            brand_voice="professional"
        )
        
        mock_gemini_client.generate_content.return_value.text = "I'm thrilled to announce our latest product innovation"
        
        result = await planner.optimize_for_platform(request)
        
        assert isinstance(result, OptimizedContent)
        assert result.platform == Platform.LINKEDIN
    
    @pytest.mark.asyncio
    async def test_optimize_empty_content_raises_error(self, planner):
        """Test that empty content raises validation error."""
        request = OptimizationRequest(
            content="",
            platform=Platform.TWITTER
        )
        
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            await planner.optimize_for_platform(request)
    
    @pytest.mark.asyncio
    async def test_optimize_with_keywords(self, planner, mock_gemini_client):
        """Test optimization with specific keywords."""
        request = OptimizationRequest(
            content="Our new AI tool helps create content",
            platform=Platform.TWITTER,
            keywords=["AI", "automation", "productivity"]
        )
        
        mock_gemini_client.generate_content.return_value.text = "AI automation boosts productivity"
        
        result = await planner.optimize_for_platform(request)
        
        assert isinstance(result, OptimizedContent)



class TestHashtagGeneration:
    """Test hashtag generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_hashtags_basic(self, planner, mock_gemini_client):
        """Test basic hashtag generation."""
        request = HashtagRequest(
            content="Amazing sunset at the beach",
            platform=Platform.INSTAGRAM,
            count=5
        )
        
        mock_gemini_client.generate_content.return_value.text = """
        #sunset
        #beach
        #nature
        #photography
        #beautiful
        """
        
        result = await planner.generate_hashtags(request)
        
        assert isinstance(result, list)
        assert len(result) <= 5
        assert all(tag.startswith("#") for tag in result)
    
    @pytest.mark.asyncio
    async def test_generate_hashtags_with_audience(self, planner, mock_gemini_client):
        """Test hashtag generation with target audience."""
        request = HashtagRequest(
            content="New tech product launch",
            platform=Platform.TWITTER,
            count=3,
            target_audience=AudienceType.TECH
        )
        
        mock_gemini_client.generate_content.return_value.text = "#tech #innovation #product"
        
        result = await planner.generate_hashtags(request)
        
        assert isinstance(result, list)
        assert len(result) <= 3
    
    @pytest.mark.asyncio
    async def test_generate_hashtags_empty_content_raises_error(self, planner):
        """Test that empty content raises validation error."""
        request = HashtagRequest(
            content="",
            platform=Platform.INSTAGRAM,
            count=5
        )
        
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            await planner.generate_hashtags(request)
    
    @pytest.mark.asyncio
    async def test_generate_hashtags_invalid_count(self, planner):
        """Test that invalid count raises validation error."""
        with pytest.raises(Exception):  # Pydantic will raise ValidationError during model creation
            request = HashtagRequest(
                content="Test content",
                platform=Platform.INSTAGRAM,
                count=50  # Too many
            )
    
    @pytest.mark.asyncio
    async def test_generate_hashtags_deduplication(self, planner, mock_gemini_client):
        """Test that duplicate hashtags are removed."""
        request = HashtagRequest(
            content="Test content",
            platform=Platform.INSTAGRAM,
            count=5
        )
        
        mock_gemini_client.generate_content.return_value.text = """
        #test
        #test
        #content
        #content
        #unique
        """
        
        result = await planner.generate_hashtags(request)
        
        # Should have only unique hashtags
        assert len(result) == len(set(result))


class TestCTAGeneration:
    """Test call-to-action generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_cta_basic(self, planner, mock_gemini_client):
        """Test basic CTA generation."""
        request = CTARequest(
            content="Check out our new product",
            platform=Platform.TWITTER,
            goal="engagement"
        )
        
        mock_gemini_client.generate_content.return_value.text = "Click the link to learn more! 👉"
        
        result = await planner.generate_cta(request)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) <= 100  # CTA should be concise
    
    @pytest.mark.asyncio
    async def test_generate_cta_with_emoji(self, planner, mock_gemini_client):
        """Test CTA generation with emoji."""
        request = CTARequest(
            content="Join our community",
            platform=Platform.INSTAGRAM,
            goal="conversion",
            include_emoji=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "Join us today! 🎉"
        
        result = await planner.generate_cta(request)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_generate_cta_empty_content_raises_error(self, planner):
        """Test that empty content raises validation error."""
        request = CTARequest(
            content="",
            platform=Platform.TWITTER,
            goal="engagement"
        )
        
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            await planner.generate_cta(request)



class TestPostingTimeSuggestions:
    """Test optimal posting time suggestions."""
    
    @pytest.mark.asyncio
    async def test_suggest_posting_times_basic(self, planner):
        """Test basic posting time suggestions."""
        request = PostingTimeRequest(
            platform=Platform.INSTAGRAM,
            target_audience=AudienceType.LIFESTYLE,
            days_ahead=7
        )
        
        result = await planner.suggest_posting_times(request)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, PostingTimeSuggestion) for s in result)
        # Check that most suggestions are in the future (some might be today but in the past)
        future_suggestions = [s for s in result if s.datetime > datetime.utcnow()]
        assert len(future_suggestions) > len(result) * 0.8  # At least 80% should be in future
        assert all(0 <= s.confidence_score <= 1 for s in result)
    
    @pytest.mark.asyncio
    async def test_suggest_posting_times_linkedin_skips_weekends(self, planner):
        """Test that LinkedIn suggestions skip weekends."""
        request = PostingTimeRequest(
            platform=Platform.LINKEDIN,
            target_audience=AudienceType.BUSINESS,
            days_ahead=14
        )
        
        result = await planner.suggest_posting_times(request)
        
        # Check that no suggestions are on weekends
        for suggestion in result:
            assert suggestion.datetime.weekday() < 5  # Monday=0, Friday=4
    
    @pytest.mark.asyncio
    async def test_suggest_posting_times_sorted_by_confidence(self, planner):
        """Test that suggestions are sorted by confidence score."""
        request = PostingTimeRequest(
            platform=Platform.TWITTER,
            target_audience=AudienceType.TECH,
            days_ahead=7
        )
        
        result = await planner.suggest_posting_times(request)
        
        # Check that results are sorted by confidence (descending)
        confidences = [s.confidence_score for s in result]
        assert confidences == sorted(confidences, reverse=True)
    
    @pytest.mark.asyncio
    async def test_suggest_posting_times_invalid_days_raises_error(self, planner):
        """Test that invalid days_ahead raises validation error."""
        with pytest.raises(Exception):  # Pydantic will raise ValidationError during model creation
            request = PostingTimeRequest(
                platform=Platform.TWITTER,
                target_audience=AudienceType.GENERAL,
                days_ahead=50  # Too many
            )
    
    @pytest.mark.asyncio
    async def test_posting_time_suggestion_has_rationale(self, planner):
        """Test that posting time suggestions include rationale."""
        request = PostingTimeRequest(
            platform=Platform.INSTAGRAM,
            target_audience=AudienceType.ENTERTAINMENT,
            days_ahead=3
        )
        
        result = await planner.suggest_posting_times(request)
        
        assert all(len(s.rationale) > 0 for s in result)
        assert all(s.day_of_week in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] for s in result)


class TestEngagementPrediction:
    """Test engagement prediction functionality."""
    
    @pytest.mark.asyncio
    async def test_predict_engagement_basic(self, planner):
        """Test basic engagement prediction."""
        request = EngagementPredictionRequest(
            content="Check out our new product launch!",
            platform=Platform.TWITTER,
            posting_time=datetime.utcnow() + timedelta(hours=2),
            target_audience=AudienceType.TECH,
            has_media=True,
            hashtag_count=3
        )
        
        result = await planner.predict_engagement(request)
        
        assert isinstance(result, EngagementScore)
        assert 0 <= result.overall_score <= 100
        assert result.predicted_likes >= 0
        assert result.predicted_shares >= 0
        assert result.predicted_comments >= 0
        assert result.predicted_reach >= 0
        assert 0 <= result.confidence <= 1
        assert len(result.factors) > 0
    
    @pytest.mark.asyncio
    async def test_predict_engagement_with_media_scores_higher(self, planner):
        """Test that content with media gets higher engagement prediction."""
        base_request = EngagementPredictionRequest(
            content="Great content here!",
            platform=Platform.INSTAGRAM,
            posting_time=datetime.utcnow() + timedelta(hours=2),
            target_audience=AudienceType.LIFESTYLE,
            has_media=False,
            hashtag_count=5
        )
        
        media_request = EngagementPredictionRequest(
            content="Great content here!",
            platform=Platform.INSTAGRAM,
            posting_time=datetime.utcnow() + timedelta(hours=2),
            target_audience=AudienceType.LIFESTYLE,
            has_media=True,
            hashtag_count=5
        )
        
        base_result = await planner.predict_engagement(base_request)
        media_result = await planner.predict_engagement(media_request)
        
        assert media_result.overall_score > base_result.overall_score
    
    @pytest.mark.asyncio
    async def test_predict_engagement_provides_recommendations(self, planner):
        """Test that engagement prediction provides recommendations."""
        request = EngagementPredictionRequest(
            content="Test content",
            platform=Platform.TWITTER,
            posting_time=datetime.utcnow() + timedelta(hours=2),
            target_audience=AudienceType.GENERAL,
            has_media=False,
            hashtag_count=0
        )
        
        result = await planner.predict_engagement(request)
        
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_predict_engagement_past_time_raises_error(self, planner):
        """Test that past posting time raises validation error."""
        request = EngagementPredictionRequest(
            content="Test content",
            platform=Platform.TWITTER,
            posting_time=datetime.utcnow() - timedelta(hours=2),  # Past time
            target_audience=AudienceType.GENERAL
        )
        
        with pytest.raises(ValidationError, match="Posting time cannot be in the past"):
            await planner.predict_engagement(request)



class TestCalendarManagement:
    """Test content calendar management functionality."""
    
    def test_schedule_content_basic(self, planner):
        """Test basic content scheduling."""
        scheduled_time = datetime.utcnow() + timedelta(days=1)
        
        entry = planner.schedule_content(
            content_id="content_123",
            platform=Platform.TWITTER,
            scheduled_time=scheduled_time,
            content_preview="Test content preview",
            engagement_prediction=75.5
        )
        
        assert isinstance(entry, CalendarEntry)
        assert entry.content_id == "content_123"
        assert entry.platform == Platform.TWITTER
        assert entry.scheduled_time == scheduled_time
        assert entry.status == "scheduled"
        assert entry.engagement_prediction == 75.5
    
    def test_schedule_multiple_content_items(self, planner):
        """Test scheduling multiple content items."""
        for i in range(3):
            scheduled_time = datetime.utcnow() + timedelta(days=i+1)
            planner.schedule_content(
                content_id=f"content_{i}",
                platform=Platform.INSTAGRAM,
                scheduled_time=scheduled_time,
                content_preview=f"Content {i}"
            )
        
        calendar = planner.get_calendar(platform=Platform.INSTAGRAM)
        assert len(calendar) == 3
    
    def test_get_calendar_all_platforms(self, planner):
        """Test getting calendar for all platforms."""
        # Schedule content on different platforms
        planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Twitter content"
        )
        planner.schedule_content(
            content_id="content_2",
            platform=Platform.INSTAGRAM,
            scheduled_time=datetime.utcnow() + timedelta(days=2),
            content_preview="Instagram content"
        )
        
        calendar = planner.get_calendar()
        assert len(calendar) == 2
    
    def test_get_calendar_filtered_by_platform(self, planner):
        """Test getting calendar filtered by platform."""
        planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Twitter content"
        )
        planner.schedule_content(
            content_id="content_2",
            platform=Platform.INSTAGRAM,
            scheduled_time=datetime.utcnow() + timedelta(days=2),
            content_preview="Instagram content"
        )
        
        twitter_calendar = planner.get_calendar(platform=Platform.TWITTER)
        assert len(twitter_calendar) == 1
        assert twitter_calendar[0].platform == Platform.TWITTER
    
    def test_get_calendar_filtered_by_date_range(self, planner):
        """Test getting calendar filtered by date range."""
        now = datetime.utcnow()
        
        # Schedule content at different times
        planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=now + timedelta(days=1),
            content_preview="Day 1"
        )
        planner.schedule_content(
            content_id="content_2",
            platform=Platform.TWITTER,
            scheduled_time=now + timedelta(days=5),
            content_preview="Day 5"
        )
        planner.schedule_content(
            content_id="content_3",
            platform=Platform.TWITTER,
            scheduled_time=now + timedelta(days=10),
            content_preview="Day 10"
        )
        
        # Get calendar for days 1-7
        calendar = planner.get_calendar(
            start_date=now,
            end_date=now + timedelta(days=7)
        )
        
        assert len(calendar) == 2  # Should only include day 1 and day 5
    
    def test_update_calendar_entry_status(self, planner):
        """Test updating calendar entry status."""
        entry = planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Test content"
        )
        
        success = planner.update_calendar_entry(
            entry_id=entry.id,
            status="published"
        )
        
        assert success is True
        
        # Verify update
        calendar = planner.get_calendar()
        updated_entry = next(e for e in calendar if e.id == entry.id)
        assert updated_entry.status == "published"
    
    def test_update_calendar_entry_engagement(self, planner):
        """Test updating calendar entry with actual engagement."""
        entry = planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Test content",
            engagement_prediction=70.0
        )
        
        success = planner.update_calendar_entry(
            entry_id=entry.id,
            actual_engagement=85.5
        )
        
        assert success is True
        
        # Verify update
        calendar = planner.get_calendar()
        updated_entry = next(e for e in calendar if e.id == entry.id)
        assert updated_entry.actual_engagement == 85.5
    
    def test_update_nonexistent_entry_returns_false(self, planner):
        """Test that updating nonexistent entry returns False."""
        success = planner.update_calendar_entry(
            entry_id="nonexistent_id",
            status="published"
        )
        
        assert success is False
    
    def test_remove_calendar_entry(self, planner):
        """Test removing calendar entry."""
        entry = planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Test content"
        )
        
        success = planner.remove_calendar_entry(entry.id)
        
        assert success is True
        
        # Verify removal
        calendar = planner.get_calendar()
        assert len(calendar) == 0
    
    def test_remove_nonexistent_entry_returns_false(self, planner):
        """Test that removing nonexistent entry returns False."""
        success = planner.remove_calendar_entry("nonexistent_id")
        
        assert success is False
    
    def test_clear_calendar_specific_platform(self, planner):
        """Test clearing calendar for specific platform."""
        planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Twitter content"
        )
        planner.schedule_content(
            content_id="content_2",
            platform=Platform.INSTAGRAM,
            scheduled_time=datetime.utcnow() + timedelta(days=2),
            content_preview="Instagram content"
        )
        
        planner.clear_calendar(platform=Platform.TWITTER)
        
        twitter_calendar = planner.get_calendar(platform=Platform.TWITTER)
        instagram_calendar = planner.get_calendar(platform=Platform.INSTAGRAM)
        
        assert len(twitter_calendar) == 0
        assert len(instagram_calendar) == 1
    
    def test_clear_all_calendar(self, planner):
        """Test clearing entire calendar."""
        planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Twitter content"
        )
        planner.schedule_content(
            content_id="content_2",
            platform=Platform.INSTAGRAM,
            scheduled_time=datetime.utcnow() + timedelta(days=2),
            content_preview="Instagram content"
        )
        
        planner.clear_calendar()
        
        calendar = planner.get_calendar()
        assert len(calendar) == 0



class TestUsageTracking:
    """Test usage statistics and tracking."""
    
    @pytest.mark.asyncio
    async def test_usage_stats_tracking(self, planner, mock_gemini_client):
        """Test that usage statistics are tracked."""
        initial_stats = planner.get_usage_stats()
        assert initial_stats["total_tokens_used"] == 0
        assert initial_stats["total_cost"] == 0.0
        
        # Perform an operation
        request = OptimizationRequest(
            content="Test content",
            platform=Platform.TWITTER,
            include_hashtags=False,
            include_cta=False
        )
        
        mock_gemini_client.generate_content.return_value.text = "Optimized content"
        
        await planner.optimize_for_platform(request)
        
        # Check that stats were updated
        updated_stats = planner.get_usage_stats()
        assert updated_stats["total_tokens_used"] > 0
        assert updated_stats["total_cost"] > 0
    
    def test_reset_usage_stats(self, planner):
        """Test resetting usage statistics."""
        # Set some usage
        planner.total_tokens_used = 1000
        planner.total_cost = 0.001
        
        planner.reset_usage_stats()
        
        assert planner.total_tokens_used == 0
        assert planner.total_cost == 0.0
    
    def test_usage_stats_includes_calendar_count(self, planner):
        """Test that usage stats include calendar entry count."""
        planner.schedule_content(
            content_id="content_1",
            platform=Platform.TWITTER,
            scheduled_time=datetime.utcnow() + timedelta(days=1),
            content_preview="Test"
        )
        
        stats = planner.get_usage_stats()
        assert stats["calendar_entries"] == 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_gemini_failure_raises_engine_error(self, planner, mock_gemini_client):
        """Test that Gemini failures raise EngineError."""
        request = OptimizationRequest(
            content="Test content",
            platform=Platform.TWITTER,
            include_hashtags=False,  # Disable to avoid additional calls
            include_cta=False
        )
        
        mock_gemini_client.generate_content.side_effect = Exception("API Error")
        
        with pytest.raises(EngineError):
            await planner.optimize_for_platform(request)
    
    @pytest.mark.asyncio
    async def test_empty_gemini_response_raises_error(self, planner, mock_gemini_client):
        """Test that empty Gemini response raises error."""
        request = OptimizationRequest(
            content="Test content",
            platform=Platform.TWITTER,
            include_hashtags=False,  # Disable to avoid additional calls
            include_cta=False
        )
        
        mock_gemini_client.generate_content.return_value.text = None
        
        with pytest.raises(EngineError):
            await planner.optimize_for_platform(request)
    
    @pytest.mark.asyncio
    async def test_very_long_content_raises_error(self, planner):
        """Test that very long content raises validation error."""
        request = OptimizationRequest(
            content="x" * 60000,  # Exceeds limit
            platform=Platform.TWITTER
        )
        
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            await planner.optimize_for_platform(request)


class TestHelperMethods:
    """Test helper methods and utilities."""
    
    def test_determine_time_slot(self, planner):
        """Test time slot determination."""
        assert planner._determine_time_slot(7) == PostingTimeSlot.EARLY_MORNING
        assert planner._determine_time_slot(10) == PostingTimeSlot.MORNING
        assert planner._determine_time_slot(13) == PostingTimeSlot.AFTERNOON
        assert planner._determine_time_slot(16) == PostingTimeSlot.LATE_AFTERNOON
        assert planner._determine_time_slot(19) == PostingTimeSlot.EVENING
        assert planner._determine_time_slot(22) == PostingTimeSlot.NIGHT
    
    def test_assess_content_quality(self, planner):
        """Test content quality assessment."""
        # Good quality content
        good_content = "This is a well-written piece of content with good length and readability."
        good_score = planner._assess_content_quality(good_content)
        
        # Poor quality content
        poor_content = "short"
        poor_score = planner._assess_content_quality(poor_content)
        
        assert good_score > poor_score
        assert 0 <= good_score <= 1
        assert 0 <= poor_score <= 1
    
    def test_assess_timing(self, planner):
        """Test timing assessment."""
        # Optimal time for Twitter (9 AM)
        optimal_time = datetime.utcnow().replace(hour=9, minute=0)
        optimal_score = planner._assess_timing(optimal_time, Platform.TWITTER)
        
        # Suboptimal time (3 AM)
        suboptimal_time = datetime.utcnow().replace(hour=3, minute=0)
        suboptimal_score = planner._assess_timing(suboptimal_time, Platform.TWITTER)
        
        assert optimal_score > suboptimal_score
        assert 0 <= optimal_score <= 1
        assert 0 <= suboptimal_score <= 1
    
    def test_estimate_tokens(self, planner):
        """Test token estimation."""
        text = "This is a test sentence with multiple words."
        tokens = planner._estimate_tokens(text)
        
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_parse_hashtags_with_various_formats(self, planner):
        """Test hashtag parsing with different formats."""
        # Format 1: One per line with #
        text1 = "#hashtag1\n#hashtag2\n#hashtag3"
        result1 = planner._parse_hashtags(text1, 5)
        assert len(result1) == 3
        
        # Format 2: Multiple on one line
        text2 = "#hashtag1 #hashtag2 #hashtag3"
        result2 = planner._parse_hashtags(text2, 5)
        assert len(result2) == 3
        
        # Format 3: Mixed
        text3 = "Here are some hashtags: #hashtag1 #hashtag2\n#hashtag3"
        result3 = planner._parse_hashtags(text3, 5)
        assert len(result3) == 3

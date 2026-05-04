"""
Tests for specialized engine API endpoints.

This module tests the REST API endpoints for all specialized AI engines.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from app.main import app
from app.models.users import User, UsageStats
from app.models.base import ContentType, Platform
from app.ai.text_intelligence_engine import TextContent, ToneType, ContentGenerationType


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Create mock user for testing."""
    return User(
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        hashed_password="$2b$12$test_hashed_password",
        usage_stats=UsageStats()
    )


@pytest.fixture
def mock_auth(mock_user):
    """Mock authentication dependency."""
    with patch("app.api.dependencies.get_current_user", return_value=mock_user):
        with patch("app.api.dependencies.check_content_generation_limit", return_value=mock_user):
            with patch("app.api.dependencies.check_transformation_limit", return_value=mock_user):
                yield


class TestTextIntelligenceEndpoints:
    """Tests for Text Intelligence Engine endpoints."""
    
    @patch("app.api.v1.endpoints.engines.text_engine.generate_content")
    async def test_generate_text_content(self, mock_generate, client, mock_auth):
        """Test text content generation endpoint."""
        # Mock the engine response
        mock_result = TextContent(
            content="Generated blog post content",
            metadata={"content_type": "blog", "tone": "professional"},
            word_count=100,
            character_count=500,
            estimated_reading_time=1,
            tokens_used=150,
            cost=0.0001
        )
        mock_generate.return_value = mock_result
        
        # Make request
        response = client.post(
            "/api/v1/engines/text/generate",
            json={
                "content_type": "blog",
                "prompt": "Write about AI technology",
                "tone": "professional",
                "target_length": 100
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "content" in data
        assert data["word_count"] == 100

    
    @patch("app.api.v1.endpoints.engines.text_engine.summarize_content")
    async def test_summarize_text_content(self, mock_summarize, client, mock_auth):
        """Test text summarization endpoint."""
        mock_result = TextContent(
            content="Summarized content",
            metadata={"original_length": 500, "target_length": 100},
            word_count=100,
            tokens_used=200,
            cost=0.0002
        )
        mock_summarize.return_value = mock_result
        
        response = client.post(
            "/api/v1/engines/text/summarize",
            json={
                "content": "Long content to summarize...",
                "target_length": 100
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "summary" in data


class TestCreativeAssistantEndpoints:
    """Tests for Creative Assistant Engine endpoints."""
    
    @patch("app.api.v1.endpoints.engines.creative_engine.start_creative_session")
    async def test_start_creative_session(self, mock_start, client, mock_auth):
        """Test starting a creative session."""
        mock_start.return_value = "session-123"
        
        response = client.post(
            "/api/v1/engines/creative/start-session",
            json={
                "session_type": "ideation",
                "topic": "Marketing campaign ideas",
                "target_audience": "Young professionals"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "session_id" in data
    
    @patch("app.api.v1.endpoints.engines.creative_engine.provide_suggestions")
    async def test_get_creative_suggestions(self, mock_suggestions, client, mock_auth):
        """Test getting creative suggestions."""
        from app.ai.creative_assistant_engine import Suggestion, SuggestionType
        
        mock_suggestions.return_value = [
            Suggestion(
                type=SuggestionType.IDEA,
                content="Creative idea 1",
                rationale="This works because...",
                confidence_score=0.9
            )
        ]
        
        response = client.post(
            "/api/v1/engines/creative/session-123/suggestions",
            json={
                "suggestion_type": "idea",
                "context": "Need marketing ideas",
                "count": 3
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "suggestions" in data


class TestSocialMediaPlannerEndpoints:
    """Tests for Social Media Planner endpoints."""
    
    @patch("app.api.v1.endpoints.engines.social_planner.optimize_for_platform")
    async def test_optimize_for_platform(self, mock_optimize, client, mock_auth):
        """Test social media platform optimization."""
        from app.ai.social_media_planner import OptimizedContent
        from app.models.base import Platform
        
        mock_optimize.return_value = OptimizedContent(
            original_content="Original post",
            optimized_content="Optimized post with hashtags",
            platform=Platform.TWITTER,
            hashtags=["#AI", "#Tech"],
            call_to_action="Learn more!",
            tokens_used=100,
            cost=0.0001
        )
        
        response = client.post(
            "/api/v1/engines/social/optimize",
            json={
                "content": "Original post",
                "platform": "twitter",
                "include_hashtags": True,
                "include_cta": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "optimized_content" in data
        assert "hashtags" in data



class TestDiscoveryAnalyticsEndpoints:
    """Tests for Discovery Analytics Engine endpoints."""
    
    @patch("app.api.v1.endpoints.engines.analytics_engine.auto_tag_content")
    async def test_auto_tag_content(self, mock_tag, client, mock_auth):
        """Test content auto-tagging."""
        from app.ai.discovery_analytics_engine import TaggingResult, Tag, TagCategory, SentimentType
        
        mock_tag.return_value = TaggingResult(
            tags=[
                Tag(name="AI", category=TagCategory.TOPIC, confidence=0.9),
                Tag(name="technology", category=TagCategory.KEYWORD, confidence=0.85)
            ],
            topics=["AI", "Technology"],
            keywords=["artificial intelligence", "machine learning"],
            sentiment=SentimentType.POSITIVE,
            sentiment_score=0.7,
            tokens_used=100,
            cost=0.0001
        )
        
        response = client.post(
            "/api/v1/engines/analytics/tag-content",
            json={
                "content": "Article about AI technology",
                "content_type": "text",
                "include_sentiment": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "tags" in data
        assert "sentiment" in data


class TestMediaGenerationEndpoints:
    """Tests for media generation engine endpoints."""
    
    @patch("app.api.v1.endpoints.engines.image_engine.generate_image")
    async def test_generate_image(self, mock_generate, client, mock_auth):
        """Test image generation."""
        from app.ai.image_generation_engine import GeneratedImage, ImageType, ImageSpecification, ImageFormat
        
        mock_generate.return_value = GeneratedImage(
            image_id="img-123",
            image_type=ImageType.THUMBNAIL,
            file_path="/storage/images/img-123.png",
            file_url="/storage/images/img-123.png",
            specification=ImageSpecification(width=320, height=180, format=ImageFormat.PNG),
            file_size_bytes=50000,
            tokens_used=100,
            cost=0.02
        )
        
        response = client.post(
            "/api/v1/engines/media/image/generate",
            json={
                "image_type": "thumbnail",
                "prompt": "Modern tech thumbnail",
                "style": "professional",
                "specification": {
                    "width": 320,
                    "height": 180,
                    "format": "png",
                    "quality": 85
                }
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "image_id" in data
        assert "file_url" in data
    
    @patch("app.api.v1.endpoints.engines.audio_engine.generate_audio")
    async def test_generate_audio(self, mock_generate, client, mock_auth):
        """Test audio generation."""
        from app.ai.audio_generation_engine import GeneratedAudio, AudioType, AudioSpecification, AudioFormat
        
        mock_generate.return_value = GeneratedAudio(
            audio_id="audio-123",
            audio_type=AudioType.VOICEOVER,
            file_path="/storage/audio/audio-123.mp3",
            file_url="/storage/audio/audio-123.mp3",
            specification=AudioSpecification(format=AudioFormat.MP3),
            duration_seconds=30.0,
            file_size_bytes=500000,
            tokens_used=150,
            cost=0.05
        )
        
        response = client.post(
            "/api/v1/engines/media/audio/generate",
            json={
                "audio_type": "voiceover",
                "text": "Welcome to our platform",
                "voice_style": "professional",
                "specification": {
                    "format": "mp3",
                    "sample_rate": 44100,
                    "bitrate": 128,
                    "channels": 2
                }
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "audio_id" in data
        assert "duration" in data
    
    @patch("app.api.v1.endpoints.engines.video_engine.generate_video")
    async def test_generate_video(self, mock_generate, client, mock_auth):
        """Test video generation."""
        from app.ai.video_pipeline_engine import GeneratedVideo, VideoType, VideoSpecification, VideoFormat, VideoQuality
        
        mock_generate.return_value = GeneratedVideo(
            video_id="video-123",
            video_type=VideoType.SHORT_FORM,
            file_path="/storage/videos/video-123.mp4",
            file_url="/storage/videos/video-123.mp4",
            specification=VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=60
            ),
            duration_seconds=60.0,
            file_size_bytes=5000000,
            tokens_used=500,
            cost=0.50
        )
        
        response = client.post(
            "/api/v1/engines/media/video/generate",
            json={
                "video_type": "short_form",
                "script": "Video script content",
                "style": "professional",
                "specification": {
                    "width": 1920,
                    "height": 1080,
                    "format": "mp4",
                    "quality": "high",
                    "fps": 30,
                    "duration_seconds": 60,
                    "bitrate_kbps": 5000
                },
                "include_audio": True
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "video_id" in data
        assert "file_url" in data


class TestEngineStatistics:
    """Tests for engine statistics endpoints."""
    
    def test_get_engine_statistics(self, client, mock_user):
        """Test retrieving engine statistics."""
        # Mock the authentication dependency
        from app.api.dependencies import get_current_user
        
        def override_get_current_user():
            return mock_user
        
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        try:
            response = client.get("/api/v1/engines/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "text_engine" in data
            assert "creative_engine" in data
            assert "social_planner" in data
            assert "analytics_engine" in data
            assert "image_engine" in data
            assert "audio_engine" in data
            assert "video_engine" in data
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()

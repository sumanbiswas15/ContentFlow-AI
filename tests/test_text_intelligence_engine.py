"""
Unit tests for Text Intelligence Engine.

This module tests the Text Intelligence Engine's content generation,
summarization, transformation, and translation capabilities.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.ai.text_intelligence_engine import (
    TextIntelligenceEngine,
    GenerationRequest,
    SummarizationRequest,
    ToneTransformationRequest,
    TranslationRequest,
    PlatformAdaptationRequest,
    TextContent,
    ToneType,
    ContentGenerationType
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
def engine(mock_gemini_client):
    """Create a Text Intelligence Engine instance with mocked Gemini."""
    with patch('app.ai.text_intelligence_engine.genai') as mock_genai:
        mock_genai.GenerativeModel.return_value = mock_gemini_client
        engine = TextIntelligenceEngine()
        engine.gemini_client = mock_gemini_client
        return engine


class TestTextIntelligenceEngineInitialization:
    """Test engine initialization."""
    
    def test_initialization_success(self, engine):
        """Test successful engine initialization."""
        assert engine.gemini_client is not None
        assert engine.total_tokens_used == 0
        assert engine.total_cost == 0.0
        assert engine.cost_per_token > 0
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        with patch('app.ai.text_intelligence_engine.settings') as mock_settings:
            mock_settings.GOOGLE_API_KEY = None
            with patch('app.ai.text_intelligence_engine.genai'):
                engine = TextIntelligenceEngine()
                assert engine.gemini_client is None


class TestContentGeneration:
    """Test content generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_blog_content(self, engine, mock_gemini_client):
        """Test blog content generation."""
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="Write about AI in content creation",
            tone=ToneType.PROFESSIONAL,
            target_length=500
        )
        
        mock_gemini_client.generate_content.return_value.text = "AI is revolutionizing content creation..."
        
        result = await engine.generate_content(request)
        
        assert isinstance(result, TextContent)
        assert len(result.content) > 0
        assert result.word_count > 0
        assert result.tokens_used > 0
        assert result.cost > 0
        assert result.metadata["content_type"] == ContentGenerationType.BLOG
    
    @pytest.mark.asyncio
    async def test_generate_caption_content(self, engine, mock_gemini_client):
        """Test caption content generation."""
        request = GenerationRequest(
            content_type=ContentGenerationType.CAPTION,
            prompt="Caption for a sunset photo",
            tone=ToneType.CASUAL,
            keywords=["sunset", "beautiful", "nature"]
        )
        
        mock_gemini_client.generate_content.return_value.text = "Beautiful sunset vibes 🌅"
        
        result = await engine.generate_content(request)
        
        assert isinstance(result, TextContent)
        assert len(result.content) > 0
        assert result.metadata["keywords"] == ["sunset", "beautiful", "nature"]
    
    @pytest.mark.asyncio
    async def test_generate_script_content(self, engine, mock_gemini_client):
        """Test script content generation."""
        request = GenerationRequest(
            content_type=ContentGenerationType.SCRIPT,
            prompt="Script for a product demo video",
            tone=ToneType.FRIENDLY,
            target_length=300
        )
        
        mock_gemini_client.generate_content.return_value.text = "Welcome to our product demo..."
        
        result = await engine.generate_content(request)
        
        assert isinstance(result, TextContent)
        assert len(result.content) > 0
        assert result.metadata["content_type"] == ContentGenerationType.SCRIPT
    
    @pytest.mark.asyncio
    async def test_generate_with_platform_optimization(self, engine, mock_gemini_client):
        """Test content generation with platform optimization."""
        request = GenerationRequest(
            content_type=ContentGenerationType.SOCIAL_POST,
            prompt="Announce new product launch",
            platform=Platform.TWITTER,
            tone=ToneType.PERSUASIVE
        )
        
        mock_gemini_client.generate_content.return_value.text = "Exciting news! Our new product is here!"
        
        result = await engine.generate_content(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["platform"] == Platform.TWITTER.value
    
    @pytest.mark.asyncio
    async def test_generate_empty_prompt_fails(self, engine):
        """Test that empty prompt raises validation error."""
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="",
            tone=ToneType.PROFESSIONAL
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.generate_content(request)
        
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_invalid_target_length_fails(self, engine):
        """Test that invalid target length raises validation error."""
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="Test prompt",
            target_length=5  # Too short
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.generate_content(request)
        
        assert "at least 10 words" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_excessive_target_length_fails(self, engine):
        """Test that excessive target length raises validation error."""
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="Test prompt",
            target_length=15000  # Too long
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.generate_content(request)
        
        assert "cannot exceed 10000 words" in str(exc_info.value)


class TestSummarization:
    """Test content summarization functionality."""
    
    @pytest.mark.asyncio
    async def test_summarize_long_content(self, engine, mock_gemini_client):
        """Test summarization of long content."""
        long_content = " ".join(["This is a test sentence."] * 100)
        
        request = SummarizationRequest(
            content=long_content,
            target_length=50,
            preserve_key_points=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "This is a concise summary of the content."
        
        result = await engine.summarize_content(request)
        
        assert isinstance(result, TextContent)
        assert len(result.content) > 0
        assert result.word_count < len(long_content.split())
        assert result.metadata["target_length"] == 50
    
    @pytest.mark.asyncio
    async def test_summarize_with_bullet_points(self, engine, mock_gemini_client):
        """Test summarization with bullet points."""
        content = " ".join(["First point about AI. Second point about automation. Third point about efficiency."] * 5)
        
        request = SummarizationRequest(
            content=content,
            target_length=30,
            bullet_points=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "AI improves automation\nAutomation increases efficiency"
        
        result = await engine.summarize_content(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["bullet_points"] is True
    
    @pytest.mark.asyncio
    async def test_summarize_empty_content_fails(self, engine):
        """Test that empty content raises validation error."""
        request = SummarizationRequest(
            content="",
            target_length=50
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.summarize_content(request)
        
        assert "Content cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_summarize_target_longer_than_content_fails(self, engine):
        """Test that target length longer than content fails."""
        request = SummarizationRequest(
            content="Short content here.",
            target_length=100
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.summarize_content(request)
        
        assert "must be less than content length" in str(exc_info.value)


class TestToneTransformation:
    """Test tone transformation functionality."""
    
    @pytest.mark.asyncio
    async def test_transform_to_professional_tone(self, engine, mock_gemini_client):
        """Test transformation to professional tone."""
        request = ToneTransformationRequest(
            content="Hey! Check out this cool new feature we built!",
            target_tone=ToneType.PROFESSIONAL,
            preserve_facts=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "We are pleased to announce our new feature."
        
        result = await engine.transform_tone(request)
        
        assert isinstance(result, TextContent)
        assert len(result.content) > 0
        assert result.metadata["target_tone"] == ToneType.PROFESSIONAL
    
    @pytest.mark.asyncio
    async def test_transform_to_casual_tone(self, engine, mock_gemini_client):
        """Test transformation to casual tone."""
        request = ToneTransformationRequest(
            content="We are pleased to inform you of our quarterly results.",
            target_tone=ToneType.CASUAL,
            preserve_facts=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "Hey! Here are our quarterly results."
        
        result = await engine.transform_tone(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["target_tone"] == ToneType.CASUAL
    
    @pytest.mark.asyncio
    async def test_transform_with_length_maintenance(self, engine, mock_gemini_client):
        """Test tone transformation with length maintenance."""
        original_content = "This is a test sentence with exactly ten words here."
        
        request = ToneTransformationRequest(
            content=original_content,
            target_tone=ToneType.FRIENDLY,
            maintain_length=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "This is a friendly test sentence with ten words."
        
        result = await engine.transform_tone(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["maintain_length"] is True
    
    @pytest.mark.asyncio
    async def test_transform_empty_content_fails(self, engine):
        """Test that empty content raises validation error."""
        request = ToneTransformationRequest(
            content="",
            target_tone=ToneType.PROFESSIONAL
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.transform_tone(request)
        
        assert "Content cannot be empty" in str(exc_info.value)


class TestTranslation:
    """Test content translation functionality."""
    
    @pytest.mark.asyncio
    async def test_translate_to_spanish(self, engine, mock_gemini_client):
        """Test translation to Spanish."""
        request = TranslationRequest(
            content="Hello, how are you?",
            target_language="es",
            preserve_formatting=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "Hola, ¿cómo estás?"
        
        result = await engine.translate_content(request)
        
        assert isinstance(result, TextContent)
        assert len(result.content) > 0
        assert result.metadata["target_language"] == "es"
    
    @pytest.mark.asyncio
    async def test_translate_to_french(self, engine, mock_gemini_client):
        """Test translation to French."""
        request = TranslationRequest(
            content="Welcome to our platform",
            target_language="fr",
            preserve_formatting=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "Bienvenue sur notre plateforme"
        
        result = await engine.translate_content(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["target_language"] == "fr"
    
    @pytest.mark.asyncio
    async def test_translate_with_context(self, engine, mock_gemini_client):
        """Test translation with context."""
        request = TranslationRequest(
            content="Bank",
            target_language="es",
            context="financial institution"
        )
        
        mock_gemini_client.generate_content.return_value.text = "Banco"
        
        result = await engine.translate_content(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["context"] == "financial institution"
    
    @pytest.mark.asyncio
    async def test_translate_empty_content_fails(self, engine):
        """Test that empty content raises validation error."""
        request = TranslationRequest(
            content="",
            target_language="es"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.translate_content(request)
        
        assert "Content cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_translate_invalid_language_fails(self, engine):
        """Test that invalid language raises validation error."""
        request = TranslationRequest(
            content="Test content",
            target_language="x"  # Too short
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.translate_content(request)
        
        assert "Invalid target language" in str(exc_info.value)


class TestPlatformAdaptation:
    """Test platform-specific content adaptation."""
    
    @pytest.mark.asyncio
    async def test_adapt_for_twitter(self, engine, mock_gemini_client):
        """Test adaptation for Twitter."""
        request = PlatformAdaptationRequest(
            content="This is a long blog post about AI that needs to be adapted for Twitter.",
            target_platform=Platform.TWITTER,
            include_hashtags=True,
            include_cta=True,
            optimize_length=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "AI is transforming content! #AI #ContentCreation"
        
        result = await engine.adapt_for_platform(request)
        
        assert isinstance(result, TextContent)
        assert len(result.content) <= 280  # Twitter limit
        assert result.metadata["target_platform"] == Platform.TWITTER
    
    @pytest.mark.asyncio
    async def test_adapt_for_instagram(self, engine, mock_gemini_client):
        """Test adaptation for Instagram."""
        request = PlatformAdaptationRequest(
            content="Product announcement content",
            target_platform=Platform.INSTAGRAM,
            include_hashtags=True,
            include_cta=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "New product alert! 🎉 #NewProduct #Innovation"
        
        result = await engine.adapt_for_platform(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["target_platform"] == Platform.INSTAGRAM
    
    @pytest.mark.asyncio
    async def test_adapt_for_linkedin(self, engine, mock_gemini_client):
        """Test adaptation for LinkedIn."""
        request = PlatformAdaptationRequest(
            content="Professional article about industry trends",
            target_platform=Platform.LINKEDIN,
            include_hashtags=False,
            include_cta=True
        )
        
        mock_gemini_client.generate_content.return_value.text = "Industry trends analysis..."
        
        result = await engine.adapt_for_platform(request)
        
        assert isinstance(result, TextContent)
        assert result.metadata["target_platform"] == Platform.LINKEDIN
    
    @pytest.mark.asyncio
    async def test_adapt_empty_content_fails(self, engine):
        """Test that empty content raises validation error."""
        request = PlatformAdaptationRequest(
            content="",
            target_platform=Platform.TWITTER
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await engine.adapt_for_platform(request)
        
        assert "Content cannot be empty" in str(exc_info.value)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_gemini_client_not_initialized(self):
        """Test error when Gemini client is not initialized."""
        engine = TextIntelligenceEngine()
        engine.gemini_client = None
        
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="Test prompt"
        )
        
        with pytest.raises(EngineError) as exc_info:
            await engine.generate_content(request)
        
        assert "Gemini client not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_gemini_api_failure_with_retry(self, engine, mock_gemini_client):
        """Test Gemini API failure with retry logic."""
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="Test prompt"
        )
        
        # Simulate failures then success
        mock_gemini_client.generate_content.side_effect = [
            Exception("API Error"),
            Exception("API Error"),
            Mock(text="Success after retries")
        ]
        
        result = await engine.generate_content(request)
        
        assert isinstance(result, TextContent)
        assert "Success after retries" in result.content
        assert mock_gemini_client.generate_content.call_count == 3
    
    @pytest.mark.asyncio
    async def test_gemini_api_complete_failure(self, engine, mock_gemini_client):
        """Test complete Gemini API failure after all retries."""
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="Test prompt"
        )
        
        # Simulate all failures
        mock_gemini_client.generate_content.side_effect = Exception("API Error")
        
        with pytest.raises(EngineError) as exc_info:
            await engine.generate_content(request)
        
        assert "text_intelligence" in str(exc_info.value).lower()


class TestUsageTracking:
    """Test usage statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_usage_stats_tracking(self, engine, mock_gemini_client):
        """Test that usage statistics are tracked correctly."""
        initial_tokens = engine.total_tokens_used
        initial_cost = engine.total_cost
        
        request = GenerationRequest(
            content_type=ContentGenerationType.BLOG,
            prompt="Test prompt for usage tracking"
        )
        
        mock_gemini_client.generate_content.return_value.text = "Generated content"
        
        result = await engine.generate_content(request)
        
        assert engine.total_tokens_used > initial_tokens
        assert engine.total_cost > initial_cost
        assert result.tokens_used > 0
        assert result.cost > 0
    
    def test_get_usage_stats(self, engine):
        """Test getting usage statistics."""
        stats = engine.get_usage_stats()
        
        assert "total_tokens_used" in stats
        assert "total_cost" in stats
        assert "cost_per_token" in stats
        assert isinstance(stats["total_tokens_used"], int)
        assert isinstance(stats["total_cost"], float)
    
    def test_reset_usage_stats(self, engine):
        """Test resetting usage statistics."""
        engine.total_tokens_used = 1000
        engine.total_cost = 10.0
        
        engine.reset_usage_stats()
        
        assert engine.total_tokens_used == 0
        assert engine.total_cost == 0.0


class TestTextContentModel:
    """Test TextContent model functionality."""
    
    def test_text_content_metrics_calculation(self):
        """Test that TextContent calculates metrics correctly."""
        content = "This is a test content with multiple words for testing."
        
        text_content = TextContent(content=content)
        
        assert text_content.word_count == len(content.split())
        assert text_content.character_count == len(content)
        assert text_content.estimated_reading_time > 0
    
    def test_text_content_empty(self):
        """Test TextContent with empty content."""
        text_content = TextContent(content="")
        
        assert text_content.word_count == 0
        assert text_content.character_count == 0

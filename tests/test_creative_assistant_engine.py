"""
Unit tests for Creative Assistant Engine.

Tests cover:
- Creative session management
- Suggestion generation
- Iterative refinement based on feedback
- Design assistance
- Marketing assistance
- Context preservation
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from app.ai.creative_assistant_engine import (
    CreativeAssistantEngine,
    CreativeContext,
    SuggestionRequest,
    Feedback,
    Suggestion,
    Interaction,
    DesignAssistanceRequest,
    MarketingAssistanceRequest,
    SuggestionType,
    FeedbackType
)
from app.models.base import CreativeSessionType
from app.core.exceptions import ValidationError, EngineError


@pytest.fixture
def engine():
    """Create a Creative Assistant Engine instance for testing."""
    with patch('app.ai.creative_assistant_engine.settings') as mock_settings:
        mock_settings.GOOGLE_API_KEY = "test-api-key"
        engine = CreativeAssistantEngine()
        # Mock the Gemini client
        engine.gemini_client = Mock()
        return engine


@pytest.fixture
def creative_context():
    """Create a sample creative context."""
    return CreativeContext(
        session_type=CreativeSessionType.IDEATION,
        topic="Sustainable Fashion Campaign",
        target_audience="Millennials and Gen Z interested in eco-friendly products",
        brand_voice="Authentic, inspiring, and action-oriented",
        goals=["Increase brand awareness", "Drive website traffic"],
        constraints=["Budget-friendly", "Social media focused"]
    )


@pytest.fixture
def suggestion_request():
    """Create a sample suggestion request."""
    return SuggestionRequest(
        suggestion_type=SuggestionType.IDEA,
        context="Need creative campaign ideas for sustainable fashion brand",
        count=3,
        style_preferences=["innovative", "engaging"]
    )


@pytest.fixture
def design_request():
    """Create a sample design assistance request."""
    return DesignAssistanceRequest(
        design_type="layout",
        content_description="Instagram post for sustainable fashion campaign",
        platform="Instagram",
        dimensions="1080x1080",
        style_preferences=["minimalist", "eco-friendly colors"]
    )


@pytest.fixture
def marketing_request():
    """Create a sample marketing assistance request."""
    return MarketingAssistanceRequest(
        marketing_type="campaign",
        product_service="Sustainable fashion clothing line",
        target_audience="Environmentally conscious millennials",
        unique_selling_points=["100% organic materials", "Carbon-neutral shipping"],
        campaign_goals=["Increase brand awareness", "Drive sales"]
    )


class TestCreativeSessionManagement:
    """Tests for creative session management."""
    
    @pytest.mark.asyncio
    async def test_start_creative_session_success(self, engine, creative_context):
        """Test successful creative session creation."""
        session_id = await engine.start_creative_session(creative_context)
        
        assert session_id is not None
        assert session_id in engine.sessions
        
        session = engine.get_session(session_id)
        assert session is not None
        assert session.context.topic == creative_context.topic
        assert session.context.session_type == creative_context.session_type
        assert len(session.interactions) == 0
    
    @pytest.mark.asyncio
    async def test_start_session_with_empty_topic(self, engine):
        """Test session creation fails with empty topic."""
        context = CreativeContext(
            session_type=CreativeSessionType.IDEATION,
            topic=""
        )
        
        with pytest.raises(ValidationError, match="Topic cannot be empty"):
            await engine.start_creative_session(context)
    
    @pytest.mark.asyncio
    async def test_get_session_not_found(self, engine):
        """Test getting non-existent session raises error."""
        with pytest.raises(ValidationError, match="Session .* not found"):
            engine._get_session("non-existent-id")
    
    @pytest.mark.asyncio
    async def test_end_session_success(self, engine, creative_context):
        """Test ending a session returns summary."""
        session_id = await engine.start_creative_session(creative_context)
        
        summary = engine.end_session(session_id)
        
        assert summary["session_id"] == session_id
        assert summary["session_type"] == creative_context.session_type
        assert "duration" in summary
        assert "interactions" in summary
        assert session_id not in engine.sessions
    
    @pytest.mark.asyncio
    async def test_maintain_context(self, engine, creative_context):
        """Test maintaining session context."""
        session_id = await engine.start_creative_session(creative_context)
        
        interaction = Interaction(
            interaction_type="test",
            content={"test": "data"}
        )
        
        await engine.maintain_context(session_id, interaction)
        
        session = engine.get_session(session_id)
        assert len(session.interactions) == 1
        assert session.interactions[0].interaction_type == "test"


class TestSuggestionGeneration:
    """Tests for suggestion generation."""
    
    @pytest.mark.asyncio
    async def test_provide_suggestions_success(self, engine, creative_context, suggestion_request):
        """Test successful suggestion generation."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Mock Gemini response
        mock_response = """1. Launch a social media challenge encouraging users to share their sustainable fashion choices
2. Create a documentary-style video series showcasing the journey from raw materials to finished products
3. Partner with eco-influencers for authentic storytelling campaigns"""
        
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        suggestions = await engine.provide_suggestions(session_id, suggestion_request)
        
        assert len(suggestions) == 3
        assert all(isinstance(s, Suggestion) for s in suggestions)
        assert all(s.type == SuggestionType.IDEA for s in suggestions)
        assert all(s.content for s in suggestions)
        
        # Check session was updated
        session = engine.get_session(session_id)
        assert len(session.interactions) == 1
        assert session.tokens_used > 0
    
    @pytest.mark.asyncio
    async def test_provide_suggestions_with_empty_context(self, engine, creative_context):
        """Test suggestion generation fails with empty context."""
        session_id = await engine.start_creative_session(creative_context)
        
        request = SuggestionRequest(
            suggestion_type=SuggestionType.IDEA,
            context="",
            count=3
        )
        
        with pytest.raises(ValidationError, match="Context cannot be empty"):
            await engine.provide_suggestions(session_id, request)
    
    @pytest.mark.asyncio
    async def test_provide_suggestions_invalid_count(self, engine, creative_context):
        """Test suggestion generation fails with invalid count."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Pydantic validation catches this before our custom validation
        from pydantic import ValidationError as PydanticValidationError
        
        with pytest.raises(PydanticValidationError):
            request = SuggestionRequest(
                suggestion_type=SuggestionType.IDEA,
                context="Test context",
                count=15  # Exceeds maximum
            )
    
    @pytest.mark.asyncio
    async def test_provide_suggestions_different_types(self, engine, creative_context):
        """Test generating different types of suggestions."""
        session_id = await engine.start_creative_session(creative_context)
        
        suggestion_types = [
            SuggestionType.HOOK,
            SuggestionType.HEADLINE,
            SuggestionType.CTA
        ]
        
        for suggestion_type in suggestion_types:
            request = SuggestionRequest(
                suggestion_type=suggestion_type,
                context=f"Generate {suggestion_type.value}",
                count=2
            )
            
            mock_response = "1. First suggestion\n2. Second suggestion"
            engine._generate_with_gemini = AsyncMock(return_value=mock_response)
            
            suggestions = await engine.provide_suggestions(session_id, request)
            
            assert len(suggestions) == 2
            assert all(s.type == suggestion_type for s in suggestions)


class TestIterativeRefinement:
    """Tests for iterative refinement based on feedback."""
    
    @pytest.mark.asyncio
    async def test_refine_suggestions_positive_feedback(self, engine, creative_context, suggestion_request):
        """Test refining suggestions with positive feedback."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Generate initial suggestions
        mock_response = "1. Original suggestion one\n2. Original suggestion two"
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        suggestions = await engine.provide_suggestions(session_id, suggestion_request)
        original_suggestion = suggestions[0]
        
        # Provide positive feedback
        feedback = Feedback(
            suggestion_id=original_suggestion.id,
            feedback_type=FeedbackType.POSITIVE,
            comments="Love the creativity!",
            preferred_elements=["innovative approach", "engaging tone"]
        )
        
        # Mock refined response
        refined_response = "1. Refined suggestion one\n2. Refined suggestion two\n3. Refined suggestion three"
        engine._generate_with_gemini = AsyncMock(return_value=refined_response)
        
        refined_suggestions = await engine.refine_suggestions(session_id, feedback)
        
        assert len(refined_suggestions) == 3
        assert all(isinstance(s, Suggestion) for s in refined_suggestions)
        
        # Check session was updated with feedback interaction
        session = engine.get_session(session_id)
        assert len(session.interactions) == 2
        assert session.interactions[1].interaction_type == "feedback"
    
    @pytest.mark.asyncio
    async def test_refine_suggestions_negative_feedback(self, engine, creative_context, suggestion_request):
        """Test refining suggestions with negative feedback."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Generate initial suggestions
        mock_response = "1. Original suggestion"
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        suggestions = await engine.provide_suggestions(session_id, suggestion_request)
        original_suggestion = suggestions[0]
        
        # Provide negative feedback
        feedback = Feedback(
            suggestion_id=original_suggestion.id,
            feedback_type=FeedbackType.NEGATIVE,
            comments="Too generic",
            disliked_elements=["lack of specificity"]
        )
        
        # Mock refined response
        refined_response = "1. More specific suggestion\n2. Another specific suggestion\n3. Third specific suggestion"
        engine._generate_with_gemini = AsyncMock(return_value=refined_response)
        
        refined_suggestions = await engine.refine_suggestions(session_id, feedback)
        
        assert len(refined_suggestions) == 3
    
    @pytest.mark.asyncio
    async def test_refine_suggestions_with_direction(self, engine, creative_context, suggestion_request):
        """Test refining suggestions with specific direction."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Generate initial suggestions
        mock_response = "1. Original suggestion"
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        suggestions = await engine.provide_suggestions(session_id, suggestion_request)
        original_suggestion = suggestions[0]
        
        # Provide refinement feedback with direction
        feedback = Feedback(
            suggestion_id=original_suggestion.id,
            feedback_type=FeedbackType.REFINEMENT,
            refinement_direction="Make it more focused on social media engagement"
        )
        
        # Mock refined response
        refined_response = "1. Social media focused suggestion\n2. Another social suggestion\n3. Third social suggestion"
        engine._generate_with_gemini = AsyncMock(return_value=refined_response)
        
        refined_suggestions = await engine.refine_suggestions(session_id, feedback)
        
        assert len(refined_suggestions) == 3
    
    @pytest.mark.asyncio
    async def test_refine_nonexistent_suggestion(self, engine, creative_context):
        """Test refining non-existent suggestion raises error."""
        session_id = await engine.start_creative_session(creative_context)
        
        feedback = Feedback(
            suggestion_id="non-existent-id",
            feedback_type=FeedbackType.POSITIVE
        )
        
        with pytest.raises(ValidationError, match="Suggestion .* not found"):
            await engine.refine_suggestions(session_id, feedback)


class TestDesignAssistance:
    """Tests for design assistance capabilities."""
    
    @pytest.mark.asyncio
    async def test_provide_design_assistance_success(self, engine, creative_context, design_request):
        """Test successful design assistance."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Mock Gemini response
        mock_response = """1. Use a minimalist layout with eco-friendly green and earth tones
2. Feature product images with natural backgrounds and soft lighting
3. Incorporate sustainable materials texture in the background"""
        
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        suggestions = await engine.provide_design_assistance(session_id, design_request)
        
        assert len(suggestions) == 3
        assert all(isinstance(s, Suggestion) for s in suggestions)
        assert all("design_type" in s.metadata for s in suggestions)
        assert all(s.metadata["design_type"] == "layout" for s in suggestions)
        
        # Check session was updated
        session = engine.get_session(session_id)
        assert len(session.interactions) == 1
        assert session.interactions[0].interaction_type == "design_request"
    
    @pytest.mark.asyncio
    async def test_design_assistance_empty_description(self, engine, creative_context):
        """Test design assistance fails with empty description."""
        session_id = await engine.start_creative_session(creative_context)
        
        request = DesignAssistanceRequest(
            design_type="layout",
            content_description=""
        )
        
        with pytest.raises(ValidationError, match="Content description cannot be empty"):
            await engine.provide_design_assistance(session_id, request)
    
    @pytest.mark.asyncio
    async def test_design_assistance_different_types(self, engine, creative_context):
        """Test design assistance for different design types."""
        session_id = await engine.start_creative_session(creative_context)
        
        design_types = ["layout", "color_scheme", "typography", "visual_hierarchy"]
        
        for design_type in design_types:
            request = DesignAssistanceRequest(
                design_type=design_type,
                content_description="Test content"
            )
            
            mock_response = "1. Design suggestion one\n2. Design suggestion two"
            engine._generate_with_gemini = AsyncMock(return_value=mock_response)
            
            suggestions = await engine.provide_design_assistance(session_id, request)
            
            assert len(suggestions) >= 1
            assert all(s.metadata["design_type"] == design_type for s in suggestions)


class TestMarketingAssistance:
    """Tests for marketing assistance capabilities."""
    
    @pytest.mark.asyncio
    async def test_provide_marketing_assistance_success(self, engine, creative_context, marketing_request):
        """Test successful marketing assistance."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Mock Gemini response
        mock_response = """1. Launch an influencer partnership campaign highlighting sustainability
2. Create a user-generated content campaign with eco-friendly hashtags
3. Develop an email series educating customers about sustainable fashion"""
        
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        suggestions = await engine.provide_marketing_assistance(session_id, marketing_request)
        
        assert len(suggestions) == 3
        assert all(isinstance(s, Suggestion) for s in suggestions)
        assert all("marketing_type" in s.metadata for s in suggestions)
        assert all(s.metadata["marketing_type"] == "campaign" for s in suggestions)
        
        # Check session was updated
        session = engine.get_session(session_id)
        assert len(session.interactions) == 1
        assert session.interactions[0].interaction_type == "marketing_request"
    
    @pytest.mark.asyncio
    async def test_marketing_assistance_empty_product(self, engine, creative_context):
        """Test marketing assistance fails with empty product description."""
        session_id = await engine.start_creative_session(creative_context)
        
        request = MarketingAssistanceRequest(
            marketing_type="campaign",
            product_service="",
            target_audience="Test audience"
        )
        
        with pytest.raises(ValidationError, match="Product/service description cannot be empty"):
            await engine.provide_marketing_assistance(session_id, request)
    
    @pytest.mark.asyncio
    async def test_marketing_assistance_empty_audience(self, engine, creative_context):
        """Test marketing assistance fails with empty target audience."""
        session_id = await engine.start_creative_session(creative_context)
        
        request = MarketingAssistanceRequest(
            marketing_type="campaign",
            product_service="Test product",
            target_audience=""
        )
        
        with pytest.raises(ValidationError, match="Target audience cannot be empty"):
            await engine.provide_marketing_assistance(session_id, request)
    
    @pytest.mark.asyncio
    async def test_marketing_assistance_different_types(self, engine, creative_context):
        """Test marketing assistance for different marketing types."""
        session_id = await engine.start_creative_session(creative_context)
        
        marketing_types = ["campaign", "cta", "value_proposition", "messaging"]
        
        for marketing_type in marketing_types:
            request = MarketingAssistanceRequest(
                marketing_type=marketing_type,
                product_service="Test product",
                target_audience="Test audience"
            )
            
            mock_response = "1. Marketing suggestion one\n2. Marketing suggestion two"
            engine._generate_with_gemini = AsyncMock(return_value=mock_response)
            
            suggestions = await engine.provide_marketing_assistance(session_id, request)
            
            assert len(suggestions) >= 1
            assert all(s.metadata["marketing_type"] == marketing_type for s in suggestions)


class TestContextPreservation:
    """Tests for context preservation across interactions."""
    
    @pytest.mark.asyncio
    async def test_conversation_history_tracking(self, engine, creative_context, suggestion_request):
        """Test that conversation history is tracked correctly."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Generate multiple suggestions
        mock_response = "1. Suggestion one\n2. Suggestion two"
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        for i in range(3):
            await engine.provide_suggestions(session_id, suggestion_request)
        
        session = engine.get_session(session_id)
        assert len(session.interactions) == 3
        
        # Check conversation history is formatted
        history = session.get_conversation_history()
        assert history is not None
        assert len(history) > 0
    
    @pytest.mark.asyncio
    async def test_context_influences_suggestions(self, engine, creative_context, suggestion_request):
        """Test that session context influences suggestion generation."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Mock to capture the prompt
        prompts_captured = []
        
        async def capture_prompt(prompt):
            prompts_captured.append(prompt)
            return "1. Test suggestion"
        
        engine._generate_with_gemini = AsyncMock(side_effect=capture_prompt)
        
        await engine.provide_suggestions(session_id, suggestion_request)
        
        # Check that context elements are in the prompt
        prompt = prompts_captured[0]
        assert creative_context.topic in prompt
        assert creative_context.target_audience in prompt
        assert creative_context.brand_voice in prompt


class TestUsageTracking:
    """Tests for usage tracking and statistics."""
    
    @pytest.mark.asyncio
    async def test_usage_stats_tracking(self, engine, creative_context, suggestion_request):
        """Test that usage statistics are tracked correctly."""
        session_id = await engine.start_creative_session(creative_context)
        
        initial_stats = engine.get_usage_stats()
        initial_tokens = initial_stats["total_tokens_used"]
        initial_cost = initial_stats["total_cost"]
        
        # Generate suggestions
        mock_response = "1. Test suggestion"
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        await engine.provide_suggestions(session_id, suggestion_request)
        
        # Check stats were updated
        updated_stats = engine.get_usage_stats()
        assert updated_stats["total_tokens_used"] > initial_tokens
        assert updated_stats["total_cost"] > initial_cost
        assert updated_stats["active_sessions"] == 1
    
    @pytest.mark.asyncio
    async def test_session_cost_tracking(self, engine, creative_context, suggestion_request):
        """Test that session-level costs are tracked."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Generate suggestions
        mock_response = "1. Test suggestion"
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        await engine.provide_suggestions(session_id, suggestion_request)
        
        session = engine.get_session(session_id)
        assert session.tokens_used > 0
        assert session.cost > 0
    
    def test_reset_usage_stats(self, engine):
        """Test resetting usage statistics."""
        engine.total_tokens_used = 1000
        engine.total_cost = 10.0
        
        engine.reset_usage_stats()
        
        assert engine.total_tokens_used == 0
        assert engine.total_cost == 0.0


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_gemini_client_not_initialized(self):
        """Test error when Gemini client is not initialized."""
        with patch('app.ai.creative_assistant_engine.settings') as mock_settings:
            mock_settings.GOOGLE_API_KEY = None
            engine = CreativeAssistantEngine()
            
            context = CreativeContext(
                session_type=CreativeSessionType.IDEATION,
                topic="Test"
            )
            session_id = await engine.start_creative_session(context)
            
            request = SuggestionRequest(
                suggestion_type=SuggestionType.IDEA,
                context="Test",
                count=1
            )
            
            with pytest.raises(EngineError, match="Gemini client not initialized"):
                await engine.provide_suggestions(session_id, request)
    
    @pytest.mark.asyncio
    async def test_suggestion_parsing_fallback(self, engine, creative_context, suggestion_request):
        """Test that suggestion parsing falls back gracefully."""
        session_id = await engine.start_creative_session(creative_context)
        
        # Mock response without numbered format
        mock_response = "This is a single suggestion without numbers"
        engine._generate_with_gemini = AsyncMock(return_value=mock_response)
        
        suggestions = await engine.provide_suggestions(session_id, suggestion_request)
        
        # Should still create at least one suggestion
        assert len(suggestions) >= 1
        assert suggestions[0].content == mock_response


class TestHelperMethods:
    """Tests for helper methods."""
    
    def test_estimate_tokens(self, engine):
        """Test token estimation."""
        text = "This is a test string with some words"
        tokens = engine._estimate_tokens(text)
        
        assert tokens > 0
        assert tokens == len(text) // 4
    
    def test_find_suggestion(self, engine):
        """Test finding suggestions in session history."""
        suggestion = Suggestion(
            type=SuggestionType.IDEA,
            content="Test suggestion"
        )
        
        interaction = Interaction(
            interaction_type="test",
            content={},
            suggestions=[suggestion]
        )
        
        context = CreativeContext(
            session_type=CreativeSessionType.IDEATION,
            topic="Test"
        )
        
        from app.ai.creative_assistant_engine import CreativeSession
        session = CreativeSession(context=context)
        session.add_interaction(interaction)
        
        found = engine._find_suggestion(session, suggestion.id)
        assert found is not None
        assert found.id == suggestion.id
        
        not_found = engine._find_suggestion(session, "non-existent-id")
        assert not_found is None

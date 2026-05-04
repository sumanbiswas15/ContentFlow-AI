"""
Creative Assistant Engine for ContentFlow AI.

This module implements the Creative Assistant Engine responsible for interactive,
iterative creative collaboration including ideation, design assistance, marketing
support, and creative refinement based on user feedback.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import google.generativeai as genai
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.core.exceptions import (
    EngineError, ValidationError, AIServiceError
)
from app.models.base import CreativeSessionType

logger = logging.getLogger(__name__)


class SuggestionType(str, Enum):
    """Enumeration of suggestion types."""
    IDEA = "idea"
    REWRITE = "rewrite"
    HOOK = "hook"
    HEADLINE = "headline"
    CTA = "call_to_action"
    VISUAL_CONCEPT = "visual_concept"
    LAYOUT = "layout"
    CAMPAIGN = "campaign"
    TAGLINE = "tagline"
    STORY_ANGLE = "story_angle"


class FeedbackType(str, Enum):
    """Enumeration of feedback types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    REFINEMENT = "refinement"
    DIRECTION_CHANGE = "direction_change"


class CreativeContext(BaseModel):
    """Model for creative session context."""
    session_type: CreativeSessionType
    topic: str
    target_audience: Optional[str] = None
    brand_voice: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    reference_materials: List[str] = Field(default_factory=list)
    additional_context: Dict[str, Any] = Field(default_factory=dict)


class SuggestionRequest(BaseModel):
    """Model for suggestion requests."""
    suggestion_type: SuggestionType
    context: str
    count: int = Field(default=3, ge=1, le=10)
    style_preferences: List[str] = Field(default_factory=list)
    avoid_patterns: List[str] = Field(default_factory=list)
    previous_suggestions: List[str] = Field(default_factory=list)


class Feedback(BaseModel):
    """Model for user feedback on suggestions."""
    suggestion_id: str
    feedback_type: FeedbackType
    comments: Optional[str] = None
    preferred_elements: List[str] = Field(default_factory=list)
    disliked_elements: List[str] = Field(default_factory=list)
    refinement_direction: Optional[str] = None


class Suggestion(BaseModel):
    """Model for creative suggestions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: SuggestionType
    content: str
    rationale: Optional[str] = None
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Interaction(BaseModel):
    """Model for session interactions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    interaction_type: str  # request, feedback, refinement
    content: Dict[str, Any]
    suggestions: List[Suggestion] = Field(default_factory=list)


class CreativeSession(BaseModel):
    """Model for creative assistance sessions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context: CreativeContext
    interactions: List[Interaction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = 0
    cost: float = 0.0
    
    def add_interaction(self, interaction: Interaction):
        """Add an interaction to the session."""
        self.interactions.append(interaction)
        self.updated_at = datetime.utcnow()
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history for context."""
        history_parts = []
        for interaction in self.interactions[-5:]:  # Last 5 interactions
            history_parts.append(f"Type: {interaction.interaction_type}")
            if interaction.suggestions:
                history_parts.append(f"Suggestions: {len(interaction.suggestions)}")
        return "\n".join(history_parts)


class DesignAssistanceRequest(BaseModel):
    """Model for design assistance requests."""
    design_type: str  # layout, color_scheme, typography, visual_hierarchy
    content_description: str
    platform: Optional[str] = None
    dimensions: Optional[str] = None
    style_preferences: List[str] = Field(default_factory=list)


class MarketingAssistanceRequest(BaseModel):
    """Model for marketing assistance requests."""
    marketing_type: str  # campaign, cta, value_proposition, messaging
    product_service: str
    target_audience: str
    unique_selling_points: List[str] = Field(default_factory=list)
    campaign_goals: List[str] = Field(default_factory=list)
    budget_tier: Optional[str] = None


class CreativeAssistantEngine:
    """
    Creative Assistant Engine for interactive creative collaboration.
    
    This engine handles:
    - Creative session management with context tracking
    - Suggestion generation for ideas, rewrites, and hooks
    - Design and marketing assistance
    - Iterative refinement based on user feedback
    """
    
    def __init__(self):
        """Initialize the Creative Assistant Engine."""
        self.gemini_client = None
        self.sessions: Dict[str, CreativeSession] = {}
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cost_per_token = 0.000001  # $0.000001 per token (example rate)
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Google Gemini client."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. Creative Assistant Engine will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Creative Assistant Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise EngineError("creative_assistant", f"Initialization failed: {e}")
    
    async def start_creative_session(
        self, 
        context: CreativeContext
    ) -> str:
        """
        Start a new creative assistance session.
        
        Args:
            context: Creative context for the session
            
        Returns:
            Session ID for tracking the session
            
        Raises:
            ValidationError: If context is invalid
        """
        try:
            logger.info(f"Starting creative session: {context.session_type}")
            
            # Validate context
            self._validate_creative_context(context)
            
            # Create new session
            session = CreativeSession(context=context)
            self.sessions[session.id] = session
            
            logger.info(f"Creative session started: {session.id}")
            return session.id
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to start creative session: {e}")
            raise EngineError("creative_assistant", f"Session creation failed: {e}")
    
    async def provide_suggestions(
        self, 
        session_id: str, 
        request: SuggestionRequest
    ) -> List[Suggestion]:
        """
        Provide creative suggestions based on request.
        
        Args:
            session_id: ID of the creative session
            request: Suggestion request with parameters
            
        Returns:
            List of creative suggestions
            
        Raises:
            ValidationError: If request is invalid
            EngineError: If suggestion generation fails
        """
        try:
            logger.info(f"Generating {request.count} {request.suggestion_type} suggestions")
            
            # Validate session and request
            session = self._get_session(session_id)
            self._validate_suggestion_request(request)
            
            # Build suggestion prompt with context
            prompt = self._build_suggestion_prompt(session, request)
            
            # Generate suggestions using Gemini
            suggestions_text = await self._generate_with_gemini(prompt)
            
            # Parse and structure suggestions
            suggestions = self._parse_suggestions(
                suggestions_text,
                request.suggestion_type,
                request.count
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(prompt + suggestions_text)
            cost = tokens_used * self.cost_per_token
            
            # Update session and usage tracking
            interaction = Interaction(
                interaction_type="request",
                content=request.dict(),
                suggestions=suggestions
            )
            session.add_interaction(interaction)
            session.tokens_used += tokens_used
            session.cost += cost
            
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            logger.info(f"Generated {len(suggestions)} suggestions")
            return suggestions
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            raise EngineError("creative_assistant", f"Suggestion generation failed: {e}")
    
    async def refine_suggestions(
        self, 
        session_id: str, 
        feedback: Feedback
    ) -> List[Suggestion]:
        """
        Refine suggestions based on user feedback.
        
        Args:
            session_id: ID of the creative session
            feedback: User feedback on previous suggestions
            
        Returns:
            List of refined suggestions
            
        Raises:
            ValidationError: If feedback is invalid
            EngineError: If refinement fails
        """
        try:
            logger.info(f"Refining suggestions based on {feedback.feedback_type} feedback")
            
            # Validate session and feedback
            session = self._get_session(session_id)
            self._validate_feedback(feedback)
            
            # Find the original suggestion
            original_suggestion = self._find_suggestion(session, feedback.suggestion_id)
            if not original_suggestion:
                raise ValidationError(f"Suggestion {feedback.suggestion_id} not found")
            
            # Build refinement prompt with feedback
            prompt = self._build_refinement_prompt(session, original_suggestion, feedback)
            
            # Generate refined suggestions using Gemini
            refined_text = await self._generate_with_gemini(prompt)
            
            # Parse refined suggestions
            refined_suggestions = self._parse_suggestions(
                refined_text,
                original_suggestion.type,
                3  # Generate 3 refined versions
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(prompt + refined_text)
            cost = tokens_used * self.cost_per_token
            
            # Update session and usage tracking
            interaction = Interaction(
                interaction_type="feedback",
                content=feedback.dict(),
                suggestions=refined_suggestions
            )
            session.add_interaction(interaction)
            session.tokens_used += tokens_used
            session.cost += cost
            
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            logger.info(f"Generated {len(refined_suggestions)} refined suggestions")
            return refined_suggestions
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Suggestion refinement failed: {e}")
            raise EngineError("creative_assistant", f"Refinement failed: {e}")
    
    async def provide_design_assistance(
        self, 
        session_id: str, 
        request: DesignAssistanceRequest
    ) -> List[Suggestion]:
        """
        Provide design assistance and visual suggestions.
        
        Args:
            session_id: ID of the creative session
            request: Design assistance request
            
        Returns:
            List of design suggestions
            
        Raises:
            ValidationError: If request is invalid
            EngineError: If design assistance fails
        """
        try:
            logger.info(f"Providing design assistance for {request.design_type}")
            
            # Validate session and request
            session = self._get_session(session_id)
            self._validate_design_request(request)
            
            # Build design assistance prompt
            prompt = self._build_design_prompt(session, request)
            
            # Generate design suggestions using Gemini
            design_text = await self._generate_with_gemini(prompt)
            
            # Parse design suggestions
            suggestions = self._parse_design_suggestions(design_text, request.design_type)
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(prompt + design_text)
            cost = tokens_used * self.cost_per_token
            
            # Update session and usage tracking
            interaction = Interaction(
                interaction_type="design_request",
                content=request.dict(),
                suggestions=suggestions
            )
            session.add_interaction(interaction)
            session.tokens_used += tokens_used
            session.cost += cost
            
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            logger.info(f"Generated {len(suggestions)} design suggestions")
            return suggestions
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Design assistance failed: {e}")
            raise EngineError("creative_assistant", f"Design assistance failed: {e}")
    
    async def provide_marketing_assistance(
        self, 
        session_id: str, 
        request: MarketingAssistanceRequest
    ) -> List[Suggestion]:
        """
        Provide marketing assistance and campaign ideas.
        
        Args:
            session_id: ID of the creative session
            request: Marketing assistance request
            
        Returns:
            List of marketing suggestions
            
        Raises:
            ValidationError: If request is invalid
            EngineError: If marketing assistance fails
        """
        try:
            logger.info(f"Providing marketing assistance for {request.marketing_type}")
            
            # Validate session and request
            session = self._get_session(session_id)
            self._validate_marketing_request(request)
            
            # Build marketing assistance prompt
            prompt = self._build_marketing_prompt(session, request)
            
            # Generate marketing suggestions using Gemini
            marketing_text = await self._generate_with_gemini(prompt)
            
            # Parse marketing suggestions
            suggestions = self._parse_marketing_suggestions(
                marketing_text, 
                request.marketing_type
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(prompt + marketing_text)
            cost = tokens_used * self.cost_per_token
            
            # Update session and usage tracking
            interaction = Interaction(
                interaction_type="marketing_request",
                content=request.dict(),
                suggestions=suggestions
            )
            session.add_interaction(interaction)
            session.tokens_used += tokens_used
            session.cost += cost
            
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            logger.info(f"Generated {len(suggestions)} marketing suggestions")
            return suggestions
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Marketing assistance failed: {e}")
            raise EngineError("creative_assistant", f"Marketing assistance failed: {e}")
    
    async def maintain_context(
        self, 
        session_id: str, 
        interaction: Interaction
    ) -> None:
        """
        Maintain session context by adding interaction.
        
        Args:
            session_id: ID of the creative session
            interaction: Interaction to add to session history
            
        Raises:
            ValidationError: If session not found
        """
        try:
            session = self._get_session(session_id)
            session.add_interaction(interaction)
            logger.info(f"Context updated for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to maintain context: {e}")
            raise EngineError("creative_assistant", f"Context maintenance failed: {e}")
    
    def get_session(self, session_id: str) -> Optional[CreativeSession]:
        """
        Get a creative session by ID.
        
        Args:
            session_id: ID of the session
            
        Returns:
            CreativeSession if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a creative session and return summary.
        
        Args:
            session_id: ID of the session to end
            
        Returns:
            Session summary with statistics
            
        Raises:
            ValidationError: If session not found
        """
        try:
            session = self._get_session(session_id)
            
            summary = {
                "session_id": session.id,
                "session_type": session.context.session_type,
                "duration": (datetime.utcnow() - session.created_at).total_seconds(),
                "interactions": len(session.interactions),
                "total_suggestions": sum(
                    len(i.suggestions) for i in session.interactions
                ),
                "tokens_used": session.tokens_used,
                "cost": session.cost
            }
            
            # Remove session from active sessions
            del self.sessions[session_id]
            
            logger.info(f"Session {session_id} ended")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            raise EngineError("creative_assistant", f"Session termination failed: {e}")
    
    # Private helper methods
    
    def _get_session(self, session_id: str) -> CreativeSession:
        """Get session or raise error if not found."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValidationError(f"Session {session_id} not found")
        return session
    
    def _validate_creative_context(self, context: CreativeContext):
        """Validate creative context."""
        if not context.topic or not context.topic.strip():
            raise ValidationError("Topic cannot be empty")
    
    def _validate_suggestion_request(self, request: SuggestionRequest):
        """Validate suggestion request."""
        if not request.context or not request.context.strip():
            raise ValidationError("Context cannot be empty")
        
        if request.count < 1 or request.count > 10:
            raise ValidationError("Count must be between 1 and 10")
    
    def _validate_feedback(self, feedback: Feedback):
        """Validate feedback."""
        if not feedback.suggestion_id:
            raise ValidationError("Suggestion ID is required")
    
    def _validate_design_request(self, request: DesignAssistanceRequest):
        """Validate design assistance request."""
        if not request.content_description or not request.content_description.strip():
            raise ValidationError("Content description cannot be empty")
    
    def _validate_marketing_request(self, request: MarketingAssistanceRequest):
        """Validate marketing assistance request."""
        if not request.product_service or not request.product_service.strip():
            raise ValidationError("Product/service description cannot be empty")
        
        if not request.target_audience or not request.target_audience.strip():
            raise ValidationError("Target audience cannot be empty")
    
    def _build_suggestion_prompt(
        self, 
        session: CreativeSession, 
        request: SuggestionRequest
    ) -> str:
        """Build prompt for suggestion generation."""
        prompt_parts = [
            f"You are a creative assistant helping with {session.context.session_type.value}.",
            f"\nSession Topic: {session.context.topic}"
        ]
        
        if session.context.target_audience:
            prompt_parts.append(f"\nTarget Audience: {session.context.target_audience}")
        
        if session.context.brand_voice:
            prompt_parts.append(f"\nBrand Voice: {session.context.brand_voice}")
        
        if session.context.goals:
            prompt_parts.append(f"\nGoals: {', '.join(session.context.goals)}")
        
        if session.context.constraints:
            prompt_parts.append(f"\nConstraints: {', '.join(session.context.constraints)}")
        
        # Add conversation history for context
        if session.interactions:
            prompt_parts.append(f"\n\nPrevious interactions: {session.get_conversation_history()}")
        
        prompt_parts.append(f"\n\nGenerate {request.count} creative {request.suggestion_type.value} suggestions.")
        prompt_parts.append(f"\nContext: {request.context}")
        
        if request.style_preferences:
            prompt_parts.append(f"\nStyle preferences: {', '.join(request.style_preferences)}")
        
        if request.avoid_patterns:
            prompt_parts.append(f"\nAvoid: {', '.join(request.avoid_patterns)}")
        
        if request.previous_suggestions:
            prompt_parts.append(f"\nPrevious suggestions (avoid repeating): {', '.join(request.previous_suggestions[:3])}")
        
        prompt_parts.append("\n\nProvide creative, unique, and actionable suggestions.")
        prompt_parts.append("Format each suggestion on a new line starting with a number (1., 2., etc.).")
        
        return "".join(prompt_parts)
    
    def _build_refinement_prompt(
        self, 
        session: CreativeSession, 
        original: Suggestion,
        feedback: Feedback
    ) -> str:
        """Build prompt for suggestion refinement."""
        prompt_parts = [
            f"You are refining a creative suggestion based on user feedback.",
            f"\nOriginal suggestion: {original.content}"
        ]
        
        if feedback.feedback_type == FeedbackType.POSITIVE:
            prompt_parts.append("\nThe user liked this suggestion. Generate similar variations.")
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            prompt_parts.append("\nThe user didn't like this suggestion. Generate different alternatives.")
        elif feedback.feedback_type == FeedbackType.REFINEMENT:
            prompt_parts.append("\nThe user wants refinements to this suggestion.")
        
        if feedback.comments:
            prompt_parts.append(f"\nUser comments: {feedback.comments}")
        
        if feedback.preferred_elements:
            prompt_parts.append(f"\nKeep these elements: {', '.join(feedback.preferred_elements)}")
        
        if feedback.disliked_elements:
            prompt_parts.append(f"\nRemove/change these elements: {', '.join(feedback.disliked_elements)}")
        
        if feedback.refinement_direction:
            prompt_parts.append(f"\nRefinement direction: {feedback.refinement_direction}")
        
        prompt_parts.append("\n\nGenerate 3 refined suggestions incorporating this feedback.")
        prompt_parts.append("Format each suggestion on a new line starting with a number (1., 2., etc.).")
        
        return "".join(prompt_parts)
    
    def _build_design_prompt(
        self, 
        session: CreativeSession, 
        request: DesignAssistanceRequest
    ) -> str:
        """Build prompt for design assistance."""
        prompt_parts = [
            f"You are a design assistant providing {request.design_type} suggestions.",
            f"\nContent description: {request.content_description}"
        ]
        
        if request.platform:
            prompt_parts.append(f"\nPlatform: {request.platform}")
        
        if request.dimensions:
            prompt_parts.append(f"\nDimensions: {request.dimensions}")
        
        if request.style_preferences:
            prompt_parts.append(f"\nStyle preferences: {', '.join(request.style_preferences)}")
        
        if session.context.brand_voice:
            prompt_parts.append(f"\nBrand voice: {session.context.brand_voice}")
        
        prompt_parts.append("\n\nProvide 3-5 detailed design suggestions.")
        prompt_parts.append("Include specific recommendations for visual elements, colors, typography, and layout.")
        prompt_parts.append("Format each suggestion on a new line starting with a number (1., 2., etc.).")
        
        return "".join(prompt_parts)
    
    def _build_marketing_prompt(
        self, 
        session: CreativeSession, 
        request: MarketingAssistanceRequest
    ) -> str:
        """Build prompt for marketing assistance."""
        prompt_parts = [
            f"You are a marketing strategist providing {request.marketing_type} suggestions.",
            f"\nProduct/Service: {request.product_service}",
            f"\nTarget Audience: {request.target_audience}"
        ]
        
        if request.unique_selling_points:
            prompt_parts.append(f"\nUnique Selling Points: {', '.join(request.unique_selling_points)}")
        
        if request.campaign_goals:
            prompt_parts.append(f"\nCampaign Goals: {', '.join(request.campaign_goals)}")
        
        if request.budget_tier:
            prompt_parts.append(f"\nBudget Tier: {request.budget_tier}")
        
        prompt_parts.append("\n\nProvide 3-5 strategic marketing suggestions.")
        prompt_parts.append("Include actionable recommendations with clear rationale.")
        prompt_parts.append("Format each suggestion on a new line starting with a number (1., 2., etc.).")
        
        return "".join(prompt_parts)
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content using Gemini with error handling."""
        if not self.gemini_client:
            raise EngineError(
                "creative_assistant",
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
    
    def _parse_suggestions(
        self, 
        text: str, 
        suggestion_type: SuggestionType,
        count: int
    ) -> List[Suggestion]:
        """Parse suggestions from generated text."""
        suggestions = []
        
        # Split by numbered lines
        lines = text.strip().split("\n")
        current_suggestion = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a number
            if line[0].isdigit() and (line[1] == "." or line[1] == ")"):
                # Save previous suggestion if exists
                if current_suggestion:
                    content = " ".join(current_suggestion).strip()
                    if content:
                        suggestions.append(Suggestion(
                            type=suggestion_type,
                            content=content
                        ))
                
                # Start new suggestion (remove number prefix)
                current_suggestion = [line[2:].strip()]
            else:
                # Continue current suggestion
                if current_suggestion:
                    current_suggestion.append(line)
        
        # Add last suggestion
        if current_suggestion:
            content = " ".join(current_suggestion).strip()
            if content:
                suggestions.append(Suggestion(
                    type=suggestion_type,
                    content=content
                ))
        
        # If parsing failed, create single suggestion from entire text
        if not suggestions and text.strip():
            suggestions.append(Suggestion(
                type=suggestion_type,
                content=text.strip()
            ))
        
        # Limit to requested count
        return suggestions[:count]
    
    def _parse_design_suggestions(
        self, 
        text: str, 
        design_type: str
    ) -> List[Suggestion]:
        """Parse design suggestions from generated text."""
        suggestions = self._parse_suggestions(text, SuggestionType.VISUAL_CONCEPT, 5)
        
        # Add design-specific metadata
        for suggestion in suggestions:
            suggestion.metadata["design_type"] = design_type
        
        return suggestions
    
    def _parse_marketing_suggestions(
        self, 
        text: str, 
        marketing_type: str
    ) -> List[Suggestion]:
        """Parse marketing suggestions from generated text."""
        suggestions = self._parse_suggestions(text, SuggestionType.CAMPAIGN, 5)
        
        # Add marketing-specific metadata
        for suggestion in suggestions:
            suggestion.metadata["marketing_type"] = marketing_type
        
        return suggestions
    
    def _find_suggestion(
        self, 
        session: CreativeSession, 
        suggestion_id: str
    ) -> Optional[Suggestion]:
        """Find a suggestion in session history."""
        for interaction in session.interactions:
            for suggestion in interaction.suggestions:
                if suggestion.id == suggestion_id:
                    return suggestion
        return None
    
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
            "active_sessions": len(self.sessions)
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0

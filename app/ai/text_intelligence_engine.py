"""
Text Intelligence Engine for ContentFlow AI.

This module implements the Text Intelligence Engine responsible for text-based
operations including content generation, summarization, transformation, and translation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import google.generativeai as genai
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.core.exceptions import (
    EngineError, ValidationError, AIServiceError, UsageLimitError
)
from app.models.base import (
    ContentType, TransformationType, Platform, ContentFormat
)

logger = logging.getLogger(__name__)


class ToneType(str, Enum):
    """Enumeration of tone types for content transformation."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    HUMOROUS = "humorous"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"
    INSPIRATIONAL = "inspirational"
    EMPATHETIC = "empathetic"
    AUTHORITATIVE = "authoritative"


class ContentGenerationType(str, Enum):
    """Enumeration of content generation types."""
    BLOG = "blog"
    CAPTION = "caption"
    SCRIPT = "script"
    ARTICLE = "article"
    SOCIAL_POST = "social_post"
    EMAIL = "email"
    PRODUCT_DESCRIPTION = "product_description"


class GenerationRequest(BaseModel):
    """Model for content generation requests."""
    content_type: ContentGenerationType
    prompt: str
    tone: Optional[ToneType] = ToneType.PROFESSIONAL
    target_length: Optional[int] = None  # words
    keywords: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    platform: Optional[Platform] = None
    language: str = "en"


class SummarizationRequest(BaseModel):
    """Model for content summarization requests."""
    content: str
    target_length: int = 100  # words
    format: ContentFormat = ContentFormat.PLAIN_TEXT
    preserve_key_points: bool = True
    bullet_points: bool = False


class ToneTransformationRequest(BaseModel):
    """Model for tone transformation requests."""
    content: str
    target_tone: ToneType
    preserve_facts: bool = True
    maintain_length: bool = False


class TranslationRequest(BaseModel):
    """Model for translation requests."""
    content: str
    target_language: str
    preserve_formatting: bool = True
    context: Optional[str] = None


class PlatformAdaptationRequest(BaseModel):
    """Model for platform-specific adaptation requests."""
    content: str
    target_platform: Platform
    include_hashtags: bool = True
    include_cta: bool = True
    optimize_length: bool = True


class TextContent(BaseModel):
    """Model for generated text content."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    word_count: int = 0
    character_count: int = 0
    estimated_reading_time: int = 0  # minutes
    tokens_used: int = 0
    cost: float = 0.0
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.content:
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate content metrics."""
        self.character_count = len(self.content)
        self.word_count = len(self.content.split())
        self.estimated_reading_time = max(1, round(self.word_count / 200))


class TextIntelligenceEngine:
    """
    Text Intelligence Engine for content generation, transformation, and analysis.
    
    This engine handles all text-based operations including:
    - Content generation (blogs, captions, scripts)
    - Summarization with length control
    - Tone transformation
    - Translation
    - Platform-specific adaptation
    """
    
    def __init__(self):
        """Initialize the Text Intelligence Engine."""
        self.gemini_client = None
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cost_per_token = 0.000001  # $0.000001 per token (example rate)
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Google Gemini client."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. Text Intelligence Engine will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            # Use gemini-2.5-flash - the latest stable model
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Text Intelligence Engine initialized with gemini-2.5-flash")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise EngineError("text_intelligence", f"Initialization failed: {e}")
    
    async def generate_content(
        self, 
        request: GenerationRequest
    ) -> TextContent:
        """
        Generate content based on the provided request.
        
        Args:
            request: Content generation request with parameters
            
        Returns:
            TextContent with generated content and metadata
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If content generation fails
        """
        try:
            logger.info(f"Generating {request.content_type} content")
            
            # Validate request
            self._validate_generation_request(request)
            
            # Build generation prompt
            prompt = self._build_generation_prompt(request)
            
            # Generate content using Gemini
            generated_text = await self._generate_with_gemini(prompt)
            
            # Post-process content
            processed_content = self._post_process_content(
                generated_text, 
                request.content_type,
                request.target_length
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(prompt + processed_content)
            cost = tokens_used * self.cost_per_token
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = TextContent(
                content=processed_content,
                metadata={
                    "content_type": request.content_type,
                    "tone": request.tone,
                    "language": request.language,
                    "keywords": request.keywords,
                    "platform": request.platform.value if request.platform else None
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Content generation completed: {result.word_count} words")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise EngineError("text_intelligence", f"Content generation failed: {e}")
    
    async def summarize_content(
        self, 
        request: SummarizationRequest
    ) -> TextContent:
        """
        Summarize content with length control.
        
        Args:
            request: Summarization request with content and parameters
            
        Returns:
            TextContent with summarized content
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If summarization fails
        """
        try:
            logger.info(f"Summarizing content to {request.target_length} words")
            
            # Validate request
            self._validate_summarization_request(request)
            
            # Build summarization prompt
            prompt = self._build_summarization_prompt(request)
            
            # Generate summary using Gemini
            summary = await self._generate_with_gemini(prompt)
            
            # Post-process summary
            processed_summary = self._post_process_summary(
                summary,
                request.target_length,
                request.bullet_points
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(request.content + processed_summary)
            cost = tokens_used * self.cost_per_token
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = TextContent(
                content=processed_summary,
                metadata={
                    "original_length": len(request.content.split()),
                    "target_length": request.target_length,
                    "format": request.format,
                    "bullet_points": request.bullet_points
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Summarization completed: {result.word_count} words")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise EngineError("text_intelligence", f"Summarization failed: {e}")
    
    async def transform_tone(
        self, 
        request: ToneTransformationRequest
    ) -> TextContent:
        """
        Transform content tone while preserving meaning.
        
        Args:
            request: Tone transformation request
            
        Returns:
            TextContent with transformed content
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If transformation fails
        """
        try:
            logger.info(f"Transforming tone to {request.target_tone}")
            
            # Validate request
            self._validate_tone_request(request)
            
            # Build transformation prompt
            prompt = self._build_tone_transformation_prompt(request)
            
            # Transform using Gemini
            transformed = await self._generate_with_gemini(prompt)
            
            # Post-process transformation
            processed_content = self._post_process_transformation(
                transformed,
                request.maintain_length,
                len(request.content.split())
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(request.content + processed_content)
            cost = tokens_used * self.cost_per_token
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = TextContent(
                content=processed_content,
                metadata={
                    "original_tone": "unknown",
                    "target_tone": request.target_tone,
                    "preserve_facts": request.preserve_facts,
                    "maintain_length": request.maintain_length
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Tone transformation completed")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Tone transformation failed: {e}")
            raise EngineError("text_intelligence", f"Tone transformation failed: {e}")
    
    async def translate_content(
        self, 
        request: TranslationRequest
    ) -> TextContent:
        """
        Translate content to target language.
        
        Args:
            request: Translation request
            
        Returns:
            TextContent with translated content
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If translation fails
        """
        try:
            logger.info(f"Translating content to {request.target_language}")
            
            # Validate request
            self._validate_translation_request(request)
            
            # Build translation prompt
            prompt = self._build_translation_prompt(request)
            
            # Translate using Gemini
            translated = await self._generate_with_gemini(prompt)
            
            # Post-process translation
            processed_content = self._post_process_translation(
                translated,
                request.preserve_formatting
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(request.content + processed_content)
            cost = tokens_used * self.cost_per_token
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = TextContent(
                content=processed_content,
                metadata={
                    "target_language": request.target_language,
                    "preserve_formatting": request.preserve_formatting,
                    "context": request.context
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Translation completed")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise EngineError("text_intelligence", f"Translation failed: {e}")
    
    async def adapt_for_platform(
        self, 
        request: PlatformAdaptationRequest
    ) -> TextContent:
        """
        Adapt content for specific platform requirements.
        
        Args:
            request: Platform adaptation request
            
        Returns:
            TextContent with platform-optimized content
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If adaptation fails
        """
        try:
            logger.info(f"Adapting content for {request.target_platform}")
            
            # Validate request
            self._validate_platform_request(request)
            
            # Build adaptation prompt
            prompt = self._build_platform_adaptation_prompt(request)
            
            # Adapt using Gemini
            adapted = await self._generate_with_gemini(prompt)
            
            # Post-process adaptation
            processed_content = self._post_process_platform_adaptation(
                adapted,
                request.target_platform,
                request.optimize_length
            )
            
            # Calculate tokens and cost
            tokens_used = self._estimate_tokens(request.content + processed_content)
            cost = tokens_used * self.cost_per_token
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = TextContent(
                content=processed_content,
                metadata={
                    "target_platform": request.target_platform,
                    "include_hashtags": request.include_hashtags,
                    "include_cta": request.include_cta,
                    "optimize_length": request.optimize_length
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Platform adaptation completed")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Platform adaptation failed: {e}")
            raise EngineError("text_intelligence", f"Platform adaptation failed: {e}")
    
    # Private helper methods
    
    def _validate_generation_request(self, request: GenerationRequest):
        """Validate content generation request."""
        if not request.prompt or not request.prompt.strip():
            raise ValidationError("Prompt cannot be empty")
        
        if request.target_length and request.target_length < 10:
            raise ValidationError("Target length must be at least 10 words")
        
        if request.target_length and request.target_length > 10000:
            raise ValidationError("Target length cannot exceed 10000 words")
    
    def _validate_summarization_request(self, request: SummarizationRequest):
        """Validate summarization request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if request.target_length < 10:
            raise ValidationError("Target length must be at least 10 words")
        
        content_length = len(request.content.split())
        if request.target_length >= content_length:
            raise ValidationError(
                f"Target length ({request.target_length}) must be less than content length ({content_length})"
            )
    
    def _validate_tone_request(self, request: ToneTransformationRequest):
        """Validate tone transformation request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
    
    def _validate_translation_request(self, request: TranslationRequest):
        """Validate translation request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
        
        if not request.target_language or len(request.target_language) < 2:
            raise ValidationError("Invalid target language")
    
    def _validate_platform_request(self, request: PlatformAdaptationRequest):
        """Validate platform adaptation request."""
        if not request.content or not request.content.strip():
            raise ValidationError("Content cannot be empty")
    
    def _build_generation_prompt(self, request: GenerationRequest) -> str:
        """Build prompt for content generation."""
        # Detect user intent from the prompt
        prompt_lower = request.prompt.lower()
        
        # Build base instruction based on content type
        prompt_parts = []
        
        # Check if user wants specific action (summarize, explain, analyze, etc.)
        if any(word in prompt_lower for word in ['summarize', 'summary', 'brief', 'condense']):
            prompt_parts.append("Create a concise summary that captures the main points.")
        elif any(word in prompt_lower for word in ['explain', 'elaborate', 'detail', 'describe']):
            prompt_parts.append("Provide a detailed explanation with clear examples and context.")
        elif any(word in prompt_lower for word in ['analyze', 'analysis', 'examine', 'evaluate']):
            prompt_parts.append("Conduct a thorough analysis examining key aspects and implications.")
        elif any(word in prompt_lower for word in ['research', 'investigate', 'explore', 'deep dive']):
            prompt_parts.append("Perform comprehensive research covering all relevant aspects in depth.")
        elif any(word in prompt_lower for word in ['think', 'reasoning', 'logic', 'consider']):
            prompt_parts.append("Think through this carefully, showing your reasoning and thought process.")
        else:
            # Default generation instruction
            prompt_parts.append(f"Generate a {request.content_type.value} with a {request.tone.value} tone.")
        
        prompt_parts.append(f"\n\nUser Request: {request.prompt}")
        
        if request.target_length:
            prompt_parts.append(f"\nTarget length: approximately {request.target_length} words")
        
        if request.keywords:
            prompt_parts.append(f"\nInclude these keywords: {', '.join(request.keywords)}")
        
        if request.platform:
            prompt_parts.append(f"\nOptimize for {request.platform.value}")
        
        if request.context:
            prompt_parts.append(f"\nContext: {request.context}")
        
        prompt_parts.append("\n\nGenerate high-quality, engaging content that directly addresses the user's request.")
        
        return "".join(prompt_parts)
    
    def _build_summarization_prompt(self, request: SummarizationRequest) -> str:
        """Build prompt for summarization."""
        prompt_parts = [
            f"You are a text summarization expert. Your task is to create a concise summary of the provided content.",
            f"\n\nIMPORTANT: Create a SUMMARY, not an explanation. A summary condenses the main points briefly.",
            f"\nTarget length: approximately {request.target_length} words (be concise and stay within this limit)."
        ]
        
        if request.preserve_key_points:
            prompt_parts.append("\nPreserve all key points and important information from the original content.")
        
        if request.bullet_points:
            prompt_parts.append("\nFormat the summary as clear, concise bullet points.")
        else:
            prompt_parts.append("\nProvide a flowing, paragraph-style summary.")
        
        prompt_parts.append("\n\nRules:")
        prompt_parts.append("\n- Extract and condense the main ideas")
        prompt_parts.append("\n- Remove redundant information")
        prompt_parts.append("\n- Keep it brief and to the point")
        prompt_parts.append("\n- Do NOT add explanations or interpretations")
        prompt_parts.append("\n- Do NOT expand on the content")
        prompt_parts.append(f"\n- Stay within {request.target_length} words")
        
        prompt_parts.append(f"\n\nContent to summarize:\n{request.content}")
        prompt_parts.append(f"\n\nProvide ONLY the summary (approximately {request.target_length} words):")
        
        return "".join(prompt_parts)
    
    def _build_tone_transformation_prompt(self, request: ToneTransformationRequest) -> str:
        """Build prompt for tone transformation."""
        prompt_parts = [
            f"Rewrite the following content with a {request.target_tone.value} tone."
        ]
        
        if request.preserve_facts:
            prompt_parts.append("\nPreserve all factual information and key details.")
        
        if request.maintain_length:
            prompt_parts.append("\nMaintain approximately the same length as the original.")
        
        prompt_parts.append(f"\n\nContent to transform:\n{request.content}")
        
        return "".join(prompt_parts)
    
    def _build_translation_prompt(self, request: TranslationRequest) -> str:
        """Build prompt for translation."""
        prompt_parts = [
            f"Translate the following content to {request.target_language}."
        ]
        
        if request.preserve_formatting:
            prompt_parts.append("\nPreserve the original formatting and structure.")
        
        if request.context:
            prompt_parts.append(f"\nContext: {request.context}")
        
        prompt_parts.append(f"\n\nContent to translate:\n{request.content}")
        
        return "".join(prompt_parts)
    
    def _build_platform_adaptation_prompt(self, request: PlatformAdaptationRequest) -> str:
        """Build prompt for platform adaptation."""
        platform_specs = self._get_platform_specifications(request.target_platform)
        
        prompt_parts = [
            f"Adapt the following content for {request.target_platform.value}.",
            f"\nPlatform specifications: {platform_specs}"
        ]
        
        if request.include_hashtags:
            prompt_parts.append("\nInclude relevant hashtags.")
        
        if request.include_cta:
            prompt_parts.append("\nInclude an effective call-to-action.")
        
        if request.optimize_length:
            prompt_parts.append("\nOptimize the length for maximum engagement on this platform.")
        
        prompt_parts.append(f"\n\nContent to adapt:\n{request.content}")
        
        return "".join(prompt_parts)
    
    def _get_platform_specifications(self, platform: Platform) -> str:
        """Get platform-specific specifications."""
        specs = {
            Platform.TWITTER: "280 character limit, concise and engaging",
            Platform.INSTAGRAM: "2200 character limit, visual-focused, hashtag-friendly",
            Platform.FACEBOOK: "63206 character limit, conversational and engaging",
            Platform.LINKEDIN: "3000 character limit, professional and informative",
            Platform.TIKTOK: "2200 character limit, casual and trendy",
            Platform.YOUTUBE: "5000 character limit, descriptive and keyword-rich",
            Platform.GENERIC: "No specific constraints"
        }
        return specs.get(platform, "No specific constraints")
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content using Gemini with error handling."""
        if not self.gemini_client:
            raise EngineError(
                "text_intelligence",
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
    
    def _post_process_content(
        self, 
        content: str, 
        content_type: ContentGenerationType,
        target_length: Optional[int]
    ) -> str:
        """Post-process generated content."""
        # Remove any markdown code blocks if present
        if content.startswith("```") and content.endswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        # Trim to target length if specified
        if target_length:
            words = content.split()
            if len(words) > target_length * 1.1:  # Allow 10% overflow
                content = " ".join(words[:target_length])
        
        return content.strip()
    
    def _post_process_summary(
        self, 
        summary: str, 
        target_length: int,
        bullet_points: bool
    ) -> str:
        """Post-process summary."""
        # Ensure bullet point formatting if requested
        if bullet_points and not any(summary.startswith(marker) for marker in ["- ", "* ", "• "]):
            lines = summary.split("\n")
            summary = "\n".join([f"- {line.strip()}" for line in lines if line.strip()])
        
        # Trim to target length
        words = summary.split()
        if len(words) > target_length * 1.1:
            summary = " ".join(words[:target_length])
        
        return summary.strip()
    
    def _post_process_transformation(
        self, 
        content: str, 
        maintain_length: bool,
        original_length: int
    ) -> str:
        """Post-process tone transformation."""
        if maintain_length:
            words = content.split()
            target_length = int(original_length * 1.1)  # Allow 10% variation
            if len(words) > target_length:
                content = " ".join(words[:target_length])
        
        return content.strip()
    
    def _post_process_translation(
        self, 
        content: str, 
        preserve_formatting: bool
    ) -> str:
        """Post-process translation."""
        # Basic cleanup
        return content.strip()
    
    def _post_process_platform_adaptation(
        self, 
        content: str, 
        platform: Platform,
        optimize_length: bool
    ) -> str:
        """Post-process platform adaptation."""
        if optimize_length:
            # Enforce platform character limits
            limits = {
                Platform.TWITTER: 280,
                Platform.INSTAGRAM: 2200,
                Platform.LINKEDIN: 3000,
                Platform.TIKTOK: 2200,
                Platform.YOUTUBE: 5000
            }
            
            limit = limits.get(platform)
            if limit and len(content) > limit:
                content = content[:limit-3] + "..."
        
        return content.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """Get engine usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cost_per_token": self.cost_per_token
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0

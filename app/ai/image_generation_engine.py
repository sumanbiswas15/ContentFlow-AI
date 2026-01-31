"""
Image Generation Engine for ContentFlow AI.

This module implements the Image Generation Engine responsible for creating
visual content including thumbnails and posters with secure storage integration.
"""

import asyncio
import logging
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pathlib import Path

import google.generativeai as genai
from pydantic import BaseModel, Field, validator

from app.core.config import settings
from app.core.exceptions import (
    EngineError, ValidationError, AIServiceError, StorageError
)
from app.models.base import ContentType

logger = logging.getLogger(__name__)


class ImageType(str, Enum):
    """Enumeration of image generation types."""
    THUMBNAIL = "thumbnail"
    POSTER = "poster"
    BANNER = "banner"
    SOCIAL_MEDIA_POST = "social_media_post"
    LOGO = "logo"
    ILLUSTRATION = "illustration"


class ImageFormat(str, Enum):
    """Enumeration of supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    SVG = "svg"


class ImageStyle(str, Enum):
    """Enumeration of image styles."""
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    MINIMALIST = "minimalist"
    ABSTRACT = "abstract"
    CARTOON = "cartoon"
    PROFESSIONAL = "professional"
    MODERN = "modern"
    VINTAGE = "vintage"


class ImageSpecification(BaseModel):
    """Model for image generation specifications."""
    width: int = Field(ge=64, le=4096, description="Image width in pixels")
    height: int = Field(ge=64, le=4096, description="Image height in pixels")
    format: ImageFormat = ImageFormat.PNG
    quality: int = Field(default=85, ge=1, le=100, description="Image quality (1-100)")
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        """Validate image dimensions are reasonable."""
        if v % 8 != 0:
            # Round to nearest multiple of 8 for better compression
            v = ((v + 7) // 8) * 8
        return v
    
    def get_aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height
    
    def get_total_pixels(self) -> int:
        """Calculate total pixel count."""
        return self.width * self.height


class ImageGenerationRequest(BaseModel):
    """Model for image generation requests."""
    image_type: ImageType
    prompt: str = Field(min_length=10, max_length=2000)
    style: ImageStyle = ImageStyle.PROFESSIONAL
    specification: ImageSpecification
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt content."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class GeneratedImage(BaseModel):
    """Model for generated image metadata."""
    image_id: str
    image_type: ImageType
    file_path: str
    file_url: str
    specification: ImageSpecification
    file_size_bytes: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    cost: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)


class ImageGenerationEngine:
    """
    Image Generation Engine for creating visual content.
    
    This engine handles:
    - Thumbnail and poster generation
    - Image specification validation and processing
    - Secure storage integration for generated images
    - Cost tracking and usage monitoring
    """
    
    def __init__(self):
        """Initialize the Image Generation Engine."""
        self.gemini_client = None
        self.storage_path = Path(settings.LOCAL_STORAGE_PATH)
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cost_per_image = 0.02  # $0.02 per image (example rate)
        self._initialize_storage()
        self._initialize_gemini()
    
    def _initialize_storage(self):
        """Initialize storage backend."""
        try:
            # Create storage directories if they don't exist
            self.images_path = self.storage_path / "images"
            self.thumbnails_path = self.images_path / "thumbnails"
            self.posters_path = self.images_path / "posters"
            
            for path in [self.images_path, self.thumbnails_path, self.posters_path]:
                path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Image storage initialized at {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise StorageError("initialization", str(e))
    
    def _initialize_gemini(self):
        """Initialize Google Gemini client for image generation."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. Image Generation Engine will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            # Use Gemini's native image generation model
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash-image')
            logger.info("Image Generation Engine initialized with gemini-2.5-flash-image")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise EngineError("image_generation", f"Initialization failed: {e}")
    
    async def generate_image(
        self, 
        request: ImageGenerationRequest
    ) -> GeneratedImage:
        """
        Generate an image based on the provided request.
        
        Args:
            request: Image generation request with specifications
            
        Returns:
            GeneratedImage with metadata and storage information
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If image generation fails
            StorageError: If storage operations fail
        """
        try:
            logger.info(f"Generating {request.image_type} image")
            
            # Validate request
            await self._validate_generation_request(request)
            
            # Enhance prompt using AI if available
            enhanced_prompt = await self._enhance_prompt(request)
            
            # Generate image (simulated for now - would call actual image generation API)
            image_data = await self._generate_image_data(enhanced_prompt, request)
            
            # Store image securely
            stored_image = await self._store_image(image_data, request)
            
            # Calculate cost
            tokens_used = self._estimate_tokens(request.prompt)
            cost = self.cost_per_image
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = GeneratedImage(
                image_id=stored_image["image_id"],
                image_type=request.image_type,
                file_path=stored_image["file_path"],
                file_url=stored_image["file_url"],
                specification=request.specification,
                file_size_bytes=stored_image["file_size"],
                metadata={
                    "prompt": request.prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "style": request.style,
                    "seed": request.seed,
                    "context": request.context
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Image generation completed: {result.image_id}")
            return result
            
        except ValidationError:
            raise
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise EngineError("image_generation", f"Image generation failed: {e}")
    
    async def generate_thumbnail(
        self,
        prompt: str,
        width: int = 320,
        height: int = 180,
        style: ImageStyle = ImageStyle.PROFESSIONAL
    ) -> GeneratedImage:
        """
        Generate a thumbnail image.
        
        Args:
            prompt: Description of the thumbnail content
            width: Thumbnail width (default 320px)
            height: Thumbnail height (default 180px)
            style: Visual style for the thumbnail
            
        Returns:
            GeneratedImage with thumbnail metadata
        """
        request = ImageGenerationRequest(
            image_type=ImageType.THUMBNAIL,
            prompt=prompt,
            style=style,
            specification=ImageSpecification(
                width=width,
                height=height,
                format=ImageFormat.JPEG,
                quality=85
            )
        )
        return await self.generate_image(request)
    
    async def generate_poster(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
        style: ImageStyle = ImageStyle.PROFESSIONAL
    ) -> GeneratedImage:
        """
        Generate a poster image.
        
        Args:
            prompt: Description of the poster content
            width: Poster width (default 1920px)
            height: Poster height (default 1080px)
            style: Visual style for the poster
            
        Returns:
            GeneratedImage with poster metadata
        """
        request = ImageGenerationRequest(
            image_type=ImageType.POSTER,
            prompt=prompt,
            style=style,
            specification=ImageSpecification(
                width=width,
                height=height,
                format=ImageFormat.PNG,
                quality=95
            )
        )
        return await self.generate_image(request)
    
    async def validate_image_specification(
        self,
        specification: ImageSpecification
    ) -> Dict[str, Any]:
        """
        Validate image specification and provide recommendations.
        
        Args:
            specification: Image specification to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check dimensions
        total_pixels = specification.get_total_pixels()
        if total_pixels > 8_000_000:  # 8 megapixels
            validation_result["warnings"].append(
                f"Large image size ({total_pixels} pixels) may result in slow processing"
            )
        
        # Check aspect ratio
        aspect_ratio = specification.get_aspect_ratio()
        common_ratios = {
            16/9: "16:9 (widescreen)",
            4/3: "4:3 (standard)",
            1/1: "1:1 (square)",
            9/16: "9:16 (portrait)"
        }
        
        closest_ratio = min(common_ratios.keys(), key=lambda r: abs(r - aspect_ratio))
        if abs(closest_ratio - aspect_ratio) > 0.1:
            validation_result["recommendations"].append(
                f"Consider using a standard aspect ratio like {common_ratios[closest_ratio]}"
            )
        
        # Check format and quality
        if specification.format == ImageFormat.JPEG and specification.quality < 70:
            validation_result["warnings"].append(
                "JPEG quality below 70 may result in visible artifacts"
            )
        
        if specification.format == ImageFormat.PNG and total_pixels > 4_000_000:
            validation_result["recommendations"].append(
                "Consider using JPEG or WEBP for large images to reduce file size"
            )
        
        return validation_result
    
    async def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image metadata by ID.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            Dictionary with image metadata or None if not found
        """
        try:
            # Search for image in storage directories
            for base_path in [self.thumbnails_path, self.posters_path, self.images_path]:
                for file_path in base_path.glob(f"{image_id}.*"):
                    if file_path.is_file():
                        return {
                            "image_id": image_id,
                            "file_path": str(file_path),
                            "file_url": self._get_file_url(file_path),
                            "file_size": file_path.stat().st_size,
                            "format": file_path.suffix[1:],
                            "exists": True
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve image {image_id}: {e}")
            raise StorageError("retrieval", str(e))
    
    async def delete_image(self, image_id: str) -> bool:
        """
        Delete an image from storage.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            image_info = await self.get_image(image_id)
            if not image_info:
                return False
            
            file_path = Path(image_info["file_path"])
            file_path.unlink()
            logger.info(f"Deleted image {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete image {image_id}: {e}")
            raise StorageError("deletion", str(e))
    
    # Private helper methods
    
    async def _validate_generation_request(self, request: ImageGenerationRequest):
        """Validate image generation request."""
        # Validate specification
        spec_validation = await self.validate_image_specification(request.specification)
        
        if not spec_validation["is_valid"]:
            raise ValidationError(
                "Invalid image specification",
                details={"errors": spec_validation["errors"]}
            )
        
        # Check storage space
        estimated_size = self._estimate_file_size(request.specification)
        if estimated_size > 50 * 1024 * 1024:  # 50 MB limit
            raise ValidationError(
                f"Estimated file size ({estimated_size / 1024 / 1024:.2f} MB) exceeds limit"
            )
    
    async def _enhance_prompt(self, request: ImageGenerationRequest) -> str:
        """Enhance the image generation prompt using AI."""
        # Skip prompt enhancement to save quota - image generation already uses quota
        logger.info("Using original prompt (enhancement disabled to save quota)")
        return request.prompt
    
    async def _generate_image_data(
        self,
        prompt: str,
        request: ImageGenerationRequest
    ) -> bytes:
        """
        Generate image data using Google Gemini's native image generation.
        
        Note: Image generation requires a paid Gemini API plan.
        Free tier users will get enhanced placeholder images.
        """
        if not self.gemini_client:
            logger.warning("Gemini client not initialized, using placeholder")
            return await self._generate_placeholder_image(request)
        
        logger.info(f"Attempting Gemini image generation: {prompt[:100]}...")
        
        try:
            # Build the full prompt with style guidance
            full_prompt = self._build_image_prompt(prompt, request)
            
            # Generate image using Gemini's native image generation
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                full_prompt
            )
            
            # Extract image data from response
            if response and response.parts:
                for part in response.parts:
                    # Check if part contains inline image data
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Get the base64 encoded image data
                        import base64
                        image_bytes = base64.b64decode(part.inline_data.data)
                        logger.info(f"Image generated successfully ({len(image_bytes)} bytes)")
                        return image_bytes
                    # Check if part has mime_type and data attributes
                    elif hasattr(part, 'mime_type') and hasattr(part, 'data'):
                        import base64
                        image_bytes = base64.b64decode(part.data)
                        logger.info(f"Image generated successfully ({len(image_bytes)} bytes)")
                        return image_bytes
            
            # If no image data found in response, raise error
            raise AIServiceError("No image data in Gemini response")
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.warning("Gemini image generation quota exceeded (requires paid plan)")
            else:
                logger.error(f"Gemini image generation failed: {e}")
            
            # Fall back to enhanced placeholder
            logger.info("Using enhanced placeholder image")
            return await self._generate_placeholder_image(request)
    
    def _get_gemini_aspect_ratio(self, specification: ImageSpecification) -> str:
        """
        Convert image dimensions to Gemini's supported aspect ratios.
        
        Supported ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
        """
        ratio = specification.width / specification.height
        
        # Map to closest supported ratio
        ratio_map = {
            1.0: "1:1",
            0.67: "2:3",
            1.5: "3:2",
            0.75: "3:4",
            1.33: "4:3",
            0.8: "4:5",
            1.25: "5:4",
            0.56: "9:16",
            1.78: "16:9",
            2.33: "21:9"
        }
        
        # Find closest ratio
        closest = min(ratio_map.keys(), key=lambda x: abs(x - ratio))
        return ratio_map[closest]
    
    def _build_image_prompt(self, prompt: str, request: ImageGenerationRequest) -> str:
        """Build a detailed prompt for Gemini image generation."""
        style_guidance = {
            ImageStyle.REALISTIC: "photorealistic, high-resolution, detailed",
            ImageStyle.ARTISTIC: "artistic, creative, expressive",
            ImageStyle.MINIMALIST: "minimalist, clean, simple composition",
            ImageStyle.ABSTRACT: "abstract, conceptual, non-representational",
            ImageStyle.CARTOON: "cartoon style, illustrated, colorful",
            ImageStyle.PROFESSIONAL: "professional, polished, high-quality",
            ImageStyle.MODERN: "modern, contemporary, sleek design",
            ImageStyle.VINTAGE: "vintage, retro, nostalgic aesthetic"
        }
        
        style_desc = style_guidance.get(request.style, "professional")
        
        # Build comprehensive prompt
        full_prompt = f"{prompt}. Style: {style_desc}."
        
        # Add image type specific guidance
        if request.image_type == ImageType.THUMBNAIL:
            full_prompt += " Optimized for thumbnail display, clear focal point."
        elif request.image_type == ImageType.POSTER:
            full_prompt += " Poster design, bold composition, eye-catching."
        elif request.image_type == ImageType.LOGO:
            full_prompt += " Logo design, simple, memorable, scalable."
        
        return full_prompt
    
    async def _generate_placeholder_image(self, request: ImageGenerationRequest) -> bytes:
        """Generate a placeholder image as fallback."""
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        try:
            width = request.specification.width
            height = request.specification.height
            
            # Create a gradient background
            img = Image.new('RGB', (width, height), color=(30, 30, 50))
            draw = ImageDraw.Draw(img)
            
            # Add gradient effect
            for y in range(height):
                color_value = int(30 + (y / height) * 100)
                draw.line([(0, y), (width, y)], fill=(color_value, color_value // 2, color_value + 50))
            
            # Add text overlay
            text = f"AI Generated Image\n{request.image_type.value}\n{width}x{height}"
            
            try:
                font = ImageFont.truetype("arial.ttf", size=min(width, height) // 20)
            except:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((width - text_width) // 2, (height - text_height) // 2)
            
            draw.text((position[0] + 2, position[1] + 2), text, fill=(0, 0, 0), font=font, align='center')
            draw.text(position, text, fill=(255, 255, 255), font=font, align='center')
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=request.specification.format.value.upper())
            return img_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate placeholder: {e}")
            # Return minimal PNG as last resort
            return (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
                b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            )
    
    async def _store_image(
        self,
        image_data: bytes,
        request: ImageGenerationRequest
    ) -> Dict[str, Any]:
        """
        Store generated image securely.
        
        Args:
            image_data: Raw image bytes
            request: Original generation request
            
        Returns:
            Dictionary with storage information
        """
        try:
            # Generate unique image ID
            image_id = self._generate_image_id(request)
            
            # Determine storage path based on image type
            if request.image_type == ImageType.THUMBNAIL:
                base_path = self.thumbnails_path
            elif request.image_type == ImageType.POSTER:
                base_path = self.posters_path
            else:
                base_path = self.images_path
            
            # Create filename with format extension
            filename = f"{image_id}.{request.specification.format.value}"
            file_path = base_path / filename
            
            # Write image data to file
            await asyncio.to_thread(file_path.write_bytes, image_data)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Generate file URL
            file_url = self._get_file_url(file_path)
            
            logger.info(f"Image stored: {file_path} ({file_size} bytes)")
            
            return {
                "image_id": image_id,
                "file_path": str(file_path),
                "file_url": file_url,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            raise StorageError("write", str(e))
    
    def _generate_image_id(self, request: ImageGenerationRequest) -> str:
        """Generate unique image ID based on request parameters."""
        # Create hash from prompt, timestamp, and random seed
        content = f"{request.prompt}_{datetime.utcnow().isoformat()}_{request.seed or 0}"
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]
    
    def _get_file_url(self, file_path: Path) -> str:
        """Generate URL for accessing the stored image."""
        # Convert absolute path to relative URL
        relative_path = file_path.relative_to(self.storage_path)
        return f"/storage/{relative_path.as_posix()}"
    
    def _estimate_file_size(self, specification: ImageSpecification) -> int:
        """Estimate file size based on image specification."""
        total_pixels = specification.get_total_pixels()
        
        # Rough estimates based on format
        if specification.format == ImageFormat.PNG:
            # PNG: ~3-4 bytes per pixel (with compression)
            return int(total_pixels * 3.5)
        elif specification.format == ImageFormat.JPEG:
            # JPEG: varies with quality, ~0.5-2 bytes per pixel
            quality_factor = specification.quality / 100
            return int(total_pixels * (0.5 + 1.5 * quality_factor))
        elif specification.format == ImageFormat.WEBP:
            # WebP: similar to JPEG but slightly better compression
            quality_factor = specification.quality / 100
            return int(total_pixels * (0.4 + 1.2 * quality_factor))
        else:
            # Default estimate
            return int(total_pixels * 2)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """Get engine usage statistics."""
        return {
            "total_images_generated": self.total_tokens_used // 100,  # Rough estimate
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cost_per_image": self.cost_per_image
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0

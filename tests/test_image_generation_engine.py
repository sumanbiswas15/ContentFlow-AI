"""
Unit tests for Image Generation Engine.

Tests cover:
- Image generation with various specifications
- Thumbnail and poster generation
- Image specification validation
- Storage integration
- Error handling
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.ai.image_generation_engine import (
    ImageGenerationEngine,
    ImageGenerationRequest,
    ImageSpecification,
    ImageType,
    ImageFormat,
    ImageStyle,
    GeneratedImage
)
from app.core.exceptions import ValidationError, EngineError, StorageError


@pytest.fixture
def engine():
    """Create an Image Generation Engine instance for testing."""
    with patch('app.ai.image_generation_engine.settings') as mock_settings:
        mock_settings.GOOGLE_API_KEY = "test_key"
        mock_settings.LOCAL_STORAGE_PATH = "./test_storage"
        engine = ImageGenerationEngine()
        yield engine
        # Cleanup test storage
        import shutil
        if Path("./test_storage").exists():
            shutil.rmtree("./test_storage")


@pytest.fixture
def valid_thumbnail_request():
    """Create a valid thumbnail generation request."""
    return ImageGenerationRequest(
        image_type=ImageType.THUMBNAIL,
        prompt="A professional thumbnail for a tech tutorial video",
        style=ImageStyle.PROFESSIONAL,
        specification=ImageSpecification(
            width=320,
            height=180,
            format=ImageFormat.JPEG,
            quality=85
        )
    )


@pytest.fixture
def valid_poster_request():
    """Create a valid poster generation request."""
    return ImageGenerationRequest(
        image_type=ImageType.POSTER,
        prompt="An inspiring poster about artificial intelligence",
        style=ImageStyle.MODERN,
        specification=ImageSpecification(
            width=1920,
            height=1080,
            format=ImageFormat.PNG,
            quality=95
        )
    )


class TestImageSpecification:
    """Tests for ImageSpecification model."""
    
    def test_valid_specification(self):
        """Test creating a valid image specification."""
        spec = ImageSpecification(
            width=1920,
            height=1080,
            format=ImageFormat.PNG,
            quality=90
        )
        assert spec.width == 1920
        assert spec.height == 1080
        assert spec.format == ImageFormat.PNG
        assert spec.quality == 90
    
    def test_dimension_rounding(self):
        """Test that dimensions are rounded to multiples of 8."""
        spec = ImageSpecification(
            width=1923,  # Should round to 1920
            height=1077,  # Should round to 1080
            format=ImageFormat.PNG
        )
        assert spec.width % 8 == 0
        assert spec.height % 8 == 0
    
    def test_aspect_ratio_calculation(self):
        """Test aspect ratio calculation."""
        spec = ImageSpecification(width=1920, height=1080, format=ImageFormat.PNG)
        assert abs(spec.get_aspect_ratio() - 16/9) < 0.01
    
    def test_total_pixels_calculation(self):
        """Test total pixel count calculation."""
        spec = ImageSpecification(width=1920, height=1080, format=ImageFormat.PNG)
        assert spec.get_total_pixels() == 1920 * 1080
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError):
            ImageSpecification(width=32, height=1080, format=ImageFormat.PNG)
        
        with pytest.raises(ValueError):
            ImageSpecification(width=1920, height=5000, format=ImageFormat.PNG)
    
    def test_invalid_quality(self):
        """Test validation of invalid quality values."""
        with pytest.raises(ValueError):
            ImageSpecification(width=1920, height=1080, format=ImageFormat.PNG, quality=0)
        
        with pytest.raises(ValueError):
            ImageSpecification(width=1920, height=1080, format=ImageFormat.PNG, quality=101)


class TestImageGenerationRequest:
    """Tests for ImageGenerationRequest model."""
    
    def test_valid_request(self, valid_thumbnail_request):
        """Test creating a valid image generation request."""
        assert valid_thumbnail_request.image_type == ImageType.THUMBNAIL
        assert valid_thumbnail_request.style == ImageStyle.PROFESSIONAL
        assert len(valid_thumbnail_request.prompt) >= 10
    
    def test_empty_prompt(self):
        """Test that empty prompts are rejected."""
        with pytest.raises(ValueError):
            ImageGenerationRequest(
                image_type=ImageType.THUMBNAIL,
                prompt="",
                style=ImageStyle.PROFESSIONAL,
                specification=ImageSpecification(width=320, height=180, format=ImageFormat.PNG)
            )
    
    def test_short_prompt(self):
        """Test that very short prompts are rejected."""
        with pytest.raises(ValueError):
            ImageGenerationRequest(
                image_type=ImageType.THUMBNAIL,
                prompt="short",
                style=ImageStyle.PROFESSIONAL,
                specification=ImageSpecification(width=320, height=180, format=ImageFormat.PNG)
            )
    
    def test_prompt_whitespace_trimming(self):
        """Test that prompt whitespace is trimmed."""
        request = ImageGenerationRequest(
            image_type=ImageType.THUMBNAIL,
            prompt="  A test prompt with whitespace  ",
            style=ImageStyle.PROFESSIONAL,
            specification=ImageSpecification(width=320, height=180, format=ImageFormat.PNG)
        )
        assert request.prompt == "A test prompt with whitespace"


class TestImageGenerationEngine:
    """Tests for ImageGenerationEngine class."""
    
    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.storage_path.exists()
        assert engine.images_path.exists()
        assert engine.thumbnails_path.exists()
        assert engine.posters_path.exists()
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail(self, engine):
        """Test thumbnail generation."""
        result = await engine.generate_thumbnail(
            prompt="A professional tech tutorial thumbnail",
            width=320,
            height=180
        )
        
        assert isinstance(result, GeneratedImage)
        assert result.image_type == ImageType.THUMBNAIL
        assert result.specification.width == 320
        assert result.specification.height == 184  # 180 rounded to multiple of 8
        assert result.file_size_bytes > 0
        assert result.image_id is not None
    
    @pytest.mark.asyncio
    async def test_generate_poster(self, engine):
        """Test poster generation."""
        result = await engine.generate_poster(
            prompt="An inspiring AI poster",
            width=1920,
            height=1080
        )
        
        assert isinstance(result, GeneratedImage)
        assert result.image_type == ImageType.POSTER
        assert result.specification.width == 1920
        assert result.specification.height == 1080
        assert result.file_size_bytes > 0
        assert result.image_id is not None
    
    @pytest.mark.asyncio
    async def test_generate_image_with_custom_request(self, engine, valid_thumbnail_request):
        """Test image generation with custom request."""
        result = await engine.generate_image(valid_thumbnail_request)
        
        assert isinstance(result, GeneratedImage)
        assert result.image_type == valid_thumbnail_request.image_type
        assert result.specification == valid_thumbnail_request.specification
        assert result.metadata["prompt"] == valid_thumbnail_request.prompt
        assert result.metadata["style"] == valid_thumbnail_request.style
    
    @pytest.mark.asyncio
    async def test_validate_image_specification(self, engine):
        """Test image specification validation."""
        spec = ImageSpecification(
            width=1920,
            height=1080,
            format=ImageFormat.PNG,
            quality=90
        )
        
        validation = await engine.validate_image_specification(spec)
        
        assert validation["is_valid"] is True
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)
        assert isinstance(validation["recommendations"], list)
    
    @pytest.mark.asyncio
    async def test_validate_large_image_warning(self, engine):
        """Test that large images generate warnings."""
        spec = ImageSpecification(
            width=4096,
            height=4096,
            format=ImageFormat.PNG,
            quality=90
        )
        
        validation = await engine.validate_image_specification(spec)
        
        assert len(validation["warnings"]) > 0
        assert any("Large image size" in w for w in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_validate_low_quality_jpeg_warning(self, engine):
        """Test that low quality JPEG generates warnings."""
        spec = ImageSpecification(
            width=1920,
            height=1080,
            format=ImageFormat.JPEG,
            quality=50
        )
        
        validation = await engine.validate_image_specification(spec)
        
        assert len(validation["warnings"]) > 0
        assert any("quality below 70" in w for w in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_get_image(self, engine, valid_thumbnail_request):
        """Test retrieving generated image metadata."""
        # Generate an image first
        generated = await engine.generate_image(valid_thumbnail_request)
        
        # Retrieve it
        retrieved = await engine.get_image(generated.image_id)
        
        assert retrieved is not None
        assert retrieved["image_id"] == generated.image_id
        assert retrieved["exists"] is True
        assert retrieved["file_size"] > 0
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_image(self, engine):
        """Test retrieving non-existent image returns None."""
        result = await engine.get_image("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_image(self, engine, valid_thumbnail_request):
        """Test deleting a generated image."""
        # Generate an image first
        generated = await engine.generate_image(valid_thumbnail_request)
        
        # Delete it
        deleted = await engine.delete_image(generated.image_id)
        assert deleted is True
        
        # Verify it's gone
        retrieved = await engine.get_image(generated.image_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_image(self, engine):
        """Test deleting non-existent image returns False."""
        result = await engine.delete_image("nonexistent_id")
        assert result is False
    
    def test_usage_stats_tracking(self, engine):
        """Test that usage statistics are tracked."""
        initial_stats = engine.get_usage_stats()
        assert "total_tokens_used" in initial_stats
        assert "total_cost" in initial_stats
        assert "cost_per_image" in initial_stats
    
    def test_reset_usage_stats(self, engine):
        """Test resetting usage statistics."""
        engine.total_tokens_used = 1000
        engine.total_cost = 10.0
        
        engine.reset_usage_stats()
        
        assert engine.total_tokens_used == 0
        assert engine.total_cost == 0.0


class TestImageStorage:
    """Tests for image storage functionality."""
    
    @pytest.mark.asyncio
    async def test_storage_path_creation(self, engine):
        """Test that storage paths are created correctly."""
        assert engine.storage_path.exists()
        assert engine.images_path.exists()
        assert engine.thumbnails_path.exists()
        assert engine.posters_path.exists()
    
    @pytest.mark.asyncio
    async def test_thumbnail_stored_in_correct_location(self, engine):
        """Test that thumbnails are stored in the thumbnails directory."""
        result = await engine.generate_thumbnail(
            prompt="Test thumbnail",
            width=320,
            height=180
        )
        
        file_path = Path(result.file_path)
        assert file_path.parent.name == "thumbnails"
    
    @pytest.mark.asyncio
    async def test_poster_stored_in_correct_location(self, engine):
        """Test that posters are stored in the posters directory."""
        result = await engine.generate_poster(
            prompt="Test poster",
            width=1920,
            height=1080
        )
        
        file_path = Path(result.file_path)
        assert file_path.parent.name == "posters"
    
    @pytest.mark.asyncio
    async def test_file_url_generation(self, engine, valid_thumbnail_request):
        """Test that file URLs are generated correctly."""
        result = await engine.generate_image(valid_thumbnail_request)
        
        assert result.file_url.startswith("/storage/")
        assert result.image_id in result.file_url


class TestErrorHandling:
    """Tests for error handling in image generation."""
    
    @pytest.mark.asyncio
    async def test_invalid_specification_raises_error(self, engine):
        """Test that invalid specifications raise ValidationError."""
        # Pydantic will catch dimensions that are too large before our validation
        # So we test with a valid spec but file size that's too large
        with pytest.raises(ValidationError) as exc_info:
            # Create a request that will pass Pydantic validation but fail our size check
            request = ImageGenerationRequest(
                image_type=ImageType.POSTER,
                prompt="Test prompt for validation",
                style=ImageStyle.PROFESSIONAL,
                specification=ImageSpecification(
                    width=4096,  # Maximum allowed
                    height=4096,  # Maximum allowed - will exceed our 50MB limit
                    format=ImageFormat.PNG,
                    quality=100
                )
            )
            await engine.generate_image(request)
        
        assert "file size" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self, engine):
        """Test handling of storage errors."""
        # Mock storage failure
        with patch.object(engine, '_store_image', side_effect=StorageError("write", "Disk full")):
            request = ImageGenerationRequest(
                image_type=ImageType.THUMBNAIL,
                prompt="Test prompt for storage error",
                style=ImageStyle.PROFESSIONAL,
                specification=ImageSpecification(
                    width=320,
                    height=180,
                    format=ImageFormat.PNG
                )
            )
            
            with pytest.raises(StorageError):
                await engine.generate_image(request)


class TestCostTracking:
    """Tests for cost tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_cost_tracked_per_generation(self, engine, valid_thumbnail_request):
        """Test that costs are tracked for each generation."""
        initial_cost = engine.total_cost
        
        result = await engine.generate_image(valid_thumbnail_request)
        
        assert result.cost > 0
        assert engine.total_cost > initial_cost
    
    @pytest.mark.asyncio
    async def test_token_usage_tracked(self, engine, valid_thumbnail_request):
        """Test that token usage is tracked."""
        initial_tokens = engine.total_tokens_used
        
        result = await engine.generate_image(valid_thumbnail_request)
        
        assert result.tokens_used > 0
        assert engine.total_tokens_used > initial_tokens


class TestImageMetadata:
    """Tests for image metadata handling."""
    
    @pytest.mark.asyncio
    async def test_metadata_includes_prompt(self, engine, valid_thumbnail_request):
        """Test that metadata includes the original prompt."""
        result = await engine.generate_image(valid_thumbnail_request)
        
        assert "prompt" in result.metadata
        assert result.metadata["prompt"] == valid_thumbnail_request.prompt
    
    @pytest.mark.asyncio
    async def test_metadata_includes_style(self, engine, valid_thumbnail_request):
        """Test that metadata includes the style."""
        result = await engine.generate_image(valid_thumbnail_request)
        
        assert "style" in result.metadata
        assert result.metadata["style"] == valid_thumbnail_request.style
    
    @pytest.mark.asyncio
    async def test_metadata_includes_enhanced_prompt(self, engine, valid_thumbnail_request):
        """Test that metadata includes the enhanced prompt."""
        result = await engine.generate_image(valid_thumbnail_request)
        
        assert "enhanced_prompt" in result.metadata
        assert result.metadata["enhanced_prompt"] is not None
    
    @pytest.mark.asyncio
    async def test_generated_at_timestamp(self, engine, valid_thumbnail_request):
        """Test that generated_at timestamp is set."""
        before = datetime.utcnow()
        result = await engine.generate_image(valid_thumbnail_request)
        after = datetime.utcnow()
        
        assert before <= result.generated_at <= after


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_minimum_dimensions(self, engine):
        """Test generation with minimum allowed dimensions."""
        result = await engine.generate_thumbnail(
            prompt="Minimum size thumbnail test",
            width=64,
            height=64
        )
        
        assert result.specification.width == 64
        assert result.specification.height == 64
    
    @pytest.mark.asyncio
    async def test_square_aspect_ratio(self, engine):
        """Test generation with square aspect ratio."""
        result = await engine.generate_thumbnail(
            prompt="Square thumbnail test",
            width=512,
            height=512
        )
        
        assert result.specification.get_aspect_ratio() == 1.0
    
    @pytest.mark.asyncio
    async def test_portrait_orientation(self, engine):
        """Test generation with portrait orientation."""
        request = ImageGenerationRequest(
            image_type=ImageType.POSTER,
            prompt="Portrait orientation poster",
            style=ImageStyle.MODERN,
            specification=ImageSpecification(
                width=1080,
                height=1920,
                format=ImageFormat.PNG
            )
        )
        
        result = await engine.generate_image(request)
        assert result.specification.height > result.specification.width
    
    @pytest.mark.asyncio
    async def test_different_image_formats(self, engine):
        """Test generation with different image formats."""
        formats = [ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.WEBP]
        
        for fmt in formats:
            request = ImageGenerationRequest(
                image_type=ImageType.THUMBNAIL,
                prompt=f"Test {fmt.value} format",
                style=ImageStyle.PROFESSIONAL,
                specification=ImageSpecification(
                    width=320,
                    height=180,
                    format=fmt
                )
            )
            
            result = await engine.generate_image(request)
            assert result.specification.format == fmt
            assert result.file_path.endswith(f".{fmt.value}")

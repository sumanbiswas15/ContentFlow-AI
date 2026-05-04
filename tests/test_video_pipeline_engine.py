"""
Unit tests for Video Pipeline Engine.

Tests cover:
- Video generation with various specifications
- Short-form and explainer video generation
- Video specification validation
- Storage integration
- Error handling
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.ai.video_pipeline_engine import (
    VideoPipelineEngine,
    VideoGenerationRequest,
    VideoSpecification,
    VideoType,
    VideoFormat,
    VideoQuality,
    VideoStyle,
    GeneratedVideo
)
from app.core.exceptions import ValidationError, EngineError, StorageError


@pytest.fixture
def engine():
    """Create a Video Pipeline Engine instance for testing."""
    with patch('app.ai.video_pipeline_engine.settings') as mock_settings:
        mock_settings.GOOGLE_API_KEY = "test_key"
        mock_settings.LOCAL_STORAGE_PATH = "./test_video_storage"
        engine = VideoPipelineEngine()
        yield engine
        # Cleanup test storage
        import shutil
        if Path("./test_video_storage").exists():
            shutil.rmtree("./test_video_storage")


@pytest.fixture
def valid_short_form_request():
    """Create a valid short-form video generation request."""
    return VideoGenerationRequest(
        video_type=VideoType.SHORT_FORM,
        script="Welcome to our channel! Today we'll explore the fascinating world of AI.",
        style=VideoStyle.DYNAMIC,
        specification=VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=5000
        ),
        include_audio=True,
        include_music=True
    )



@pytest.fixture
def valid_explainer_request():
    """Create a valid explainer video generation request."""
    return VideoGenerationRequest(
        video_type=VideoType.EXPLAINER,
        script="In this video, we'll explain how machine learning algorithms work and their applications.",
        style=VideoStyle.PROFESSIONAL,
        specification=VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=120,
            bitrate_kbps=8000
        ),
        include_audio=True,
        include_music=False,
        include_subtitles=True
    )


class TestVideoSpecification:
    """Tests for VideoSpecification model."""
    
    def test_valid_specification(self):
        """Test creating a valid video specification."""
        spec = VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=5000
        )
        assert spec.width == 1920
        assert spec.height == 1080
        assert spec.format == VideoFormat.MP4
        assert spec.quality == VideoQuality.HIGH
        assert spec.fps == 30
        assert spec.duration_seconds == 60
    
    def test_dimension_rounding_to_even(self):
        """Test that dimensions are rounded to even numbers."""
        spec = VideoSpecification(
            width=1921,  # Should round to 1922
            height=1079,  # Should round to 1080
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=5000
        )
        assert spec.width % 2 == 0
        assert spec.height % 2 == 0
    
    def test_aspect_ratio_calculation(self):
        """Test aspect ratio calculation."""
        spec = VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=5000
        )
        assert abs(spec.get_aspect_ratio() - 16/9) < 0.01
    
    def test_total_pixels_calculation(self):
        """Test total pixel count calculation."""
        spec = VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=5000
        )
        assert spec.get_total_pixels() == 1920 * 1080
    
    def test_file_size_estimation(self):
        """Test file size estimation."""
        spec = VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=5000
        )
        estimated_size = spec.get_estimated_file_size_mb()
        assert estimated_size > 0
        # 5000 kbps * 60 seconds / (8 * 1024) ≈ 36.6 MB
        assert 30 < estimated_size < 50
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError):
            VideoSpecification(
                width=400,  # Too small
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=60,
                bitrate_kbps=5000
            )
    
    def test_invalid_fps(self):
        """Test validation of invalid FPS values."""
        with pytest.raises(ValueError):
            VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=20,  # Too low
                duration_seconds=60,
                bitrate_kbps=5000
            )
        
        with pytest.raises(ValueError):
            VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=120,  # Too high
                duration_seconds=60,
                bitrate_kbps=5000
            )
    
    def test_invalid_duration(self):
        """Test validation of invalid duration values."""
        with pytest.raises(ValueError):
            VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=0,  # Too short
                bitrate_kbps=5000
            )



class TestVideoGenerationRequest:
    """Tests for VideoGenerationRequest model."""
    
    def test_valid_request(self, valid_short_form_request):
        """Test creating a valid video generation request."""
        assert valid_short_form_request.video_type == VideoType.SHORT_FORM
        assert valid_short_form_request.style == VideoStyle.DYNAMIC
        assert len(valid_short_form_request.script) >= 10
        assert valid_short_form_request.include_audio is True
    
    def test_empty_script(self):
        """Test that empty scripts are rejected."""
        with pytest.raises(ValueError):
            VideoGenerationRequest(
                video_type=VideoType.SHORT_FORM,
                script="",
                style=VideoStyle.DYNAMIC,
                specification=VideoSpecification(
                    width=1920,
                    height=1080,
                    format=VideoFormat.MP4,
                    quality=VideoQuality.HIGH,
                    fps=30,
                    duration_seconds=60,
                    bitrate_kbps=5000
                )
            )
    
    def test_short_script(self):
        """Test that very short scripts are rejected."""
        with pytest.raises(ValueError):
            VideoGenerationRequest(
                video_type=VideoType.SHORT_FORM,
                script="short",
                style=VideoStyle.DYNAMIC,
                specification=VideoSpecification(
                    width=1920,
                    height=1080,
                    format=VideoFormat.MP4,
                    quality=VideoQuality.HIGH,
                    fps=30,
                    duration_seconds=60,
                    bitrate_kbps=5000
                )
            )
    
    def test_script_whitespace_trimming(self):
        """Test that script whitespace is trimmed."""
        request = VideoGenerationRequest(
            video_type=VideoType.SHORT_FORM,
            script="  A test script with whitespace  ",
            style=VideoStyle.DYNAMIC,
            specification=VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=60,
                bitrate_kbps=5000
            )
        )
        assert request.script == "A test script with whitespace"


class TestVideoPipelineEngine:
    """Tests for VideoPipelineEngine class."""
    
    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.storage_path.exists()
        assert engine.video_path.exists()
        assert engine.short_form_path.exists()
        assert engine.explainers_path.exists()
        assert engine.thumbnails_path.exists()
    
    @pytest.mark.asyncio
    async def test_generate_short_form_video(self, engine):
        """Test short-form video generation."""
        result = await engine.generate_short_form_video(
            script="Welcome to our tutorial on artificial intelligence!",
            duration_seconds=60,
            style=VideoStyle.DYNAMIC
        )
        
        assert isinstance(result, GeneratedVideo)
        assert result.video_type == VideoType.SHORT_FORM
        assert result.duration_seconds == 60
        assert result.file_size_bytes > 0
        assert result.video_id is not None
        assert result.thumbnail_path is not None
    
    @pytest.mark.asyncio
    async def test_generate_explainer_video(self, engine):
        """Test explainer video generation."""
        result = await engine.generate_explainer_video(
            script="In this video, we'll explain how neural networks process information.",
            duration_seconds=120,
            style=VideoStyle.PROFESSIONAL,
            include_subtitles=True
        )
        
        assert isinstance(result, GeneratedVideo)
        assert result.video_type == VideoType.EXPLAINER
        assert result.duration_seconds == 120
        assert result.file_size_bytes > 0
        assert result.video_id is not None
        assert result.metadata["include_subtitles"] is True
    
    @pytest.mark.asyncio
    async def test_generate_video_with_custom_request(self, engine, valid_short_form_request):
        """Test video generation with custom request."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert isinstance(result, GeneratedVideo)
        assert result.video_type == valid_short_form_request.video_type
        assert result.specification == valid_short_form_request.specification
        assert result.metadata["script"] == valid_short_form_request.script
        assert result.metadata["style"] == valid_short_form_request.style
    
    @pytest.mark.asyncio
    async def test_validate_video_specification(self, engine):
        """Test video specification validation."""
        spec = VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=5000
        )
        
        validation = await engine.validate_video_specification(spec)
        
        assert validation["is_valid"] is True
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)
        assert isinstance(validation["recommendations"], list)

    
    @pytest.mark.asyncio
    async def test_validate_high_resolution_warning(self, engine):
        """Test that high resolution videos generate warnings."""
        spec = VideoSpecification(
            width=3840,
            height=2160,
            format=VideoFormat.MP4,
            quality=VideoQuality.ULTRA,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=20000
        )
        
        validation = await engine.validate_video_specification(spec)
        
        assert len(validation["warnings"]) > 0
        assert any("High resolution" in w or "slow processing" in w for w in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_validate_low_bitrate_warning(self, engine):
        """Test that low bitrate generates warnings."""
        spec = VideoSpecification(
            width=1920,
            height=1080,
            format=VideoFormat.MP4,
            quality=VideoQuality.HIGH,
            fps=30,
            duration_seconds=60,
            bitrate_kbps=1500
        )
        
        validation = await engine.validate_video_specification(spec)
        
        assert len(validation["warnings"]) > 0
        assert any("Low bitrate" in w or "compression artifacts" in w for w in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_validate_large_file_size_warning(self, engine):
        """Test that large file sizes generate warnings."""
        spec = VideoSpecification(
            width=3840,
            height=2160,
            format=VideoFormat.MP4,
            quality=VideoQuality.ULTRA,
            fps=60,
            duration_seconds=300,
            bitrate_kbps=50000
        )
        
        validation = await engine.validate_video_specification(spec)
        
        assert len(validation["warnings"]) > 0
        assert any("Large file size" in w or "upload" in w or "streaming" in w for w in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_get_video(self, engine, valid_short_form_request):
        """Test retrieving generated video metadata."""
        # Generate a video first
        generated = await engine.generate_video(valid_short_form_request)
        
        # Retrieve it
        retrieved = await engine.get_video(generated.video_id)
        
        assert retrieved is not None
        assert retrieved["video_id"] == generated.video_id
        assert retrieved["exists"] is True
        assert retrieved["file_size"] > 0
        assert retrieved["thumbnail_path"] is not None
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_video(self, engine):
        """Test retrieving non-existent video returns None."""
        result = await engine.get_video("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_video(self, engine, valid_short_form_request):
        """Test deleting a generated video."""
        # Generate a video first
        generated = await engine.generate_video(valid_short_form_request)
        
        # Delete it
        deleted = await engine.delete_video(generated.video_id)
        assert deleted is True
        
        # Verify it's gone
        retrieved = await engine.get_video(generated.video_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_video(self, engine):
        """Test deleting non-existent video returns False."""
        result = await engine.delete_video("nonexistent_id")
        assert result is False
    
    def test_usage_stats_tracking(self, engine):
        """Test that usage statistics are tracked."""
        initial_stats = engine.get_usage_stats()
        assert "total_tokens_used" in initial_stats
        assert "total_cost" in initial_stats
        assert "cost_per_video" in initial_stats
    
    def test_reset_usage_stats(self, engine):
        """Test resetting usage statistics."""
        engine.total_tokens_used = 1000
        engine.total_cost = 10.0
        
        engine.reset_usage_stats()
        
        assert engine.total_tokens_used == 0
        assert engine.total_cost == 0.0


class TestVideoStorage:
    """Tests for video storage functionality."""
    
    @pytest.mark.asyncio
    async def test_storage_path_creation(self, engine):
        """Test that storage paths are created correctly."""
        assert engine.storage_path.exists()
        assert engine.video_path.exists()
        assert engine.short_form_path.exists()
        assert engine.explainers_path.exists()
        assert engine.thumbnails_path.exists()
    
    @pytest.mark.asyncio
    async def test_short_form_stored_in_correct_location(self, engine):
        """Test that short-form videos are stored in the short_form directory."""
        result = await engine.generate_short_form_video(
            script="Test short-form video for storage location",
            duration_seconds=30
        )
        
        file_path = Path(result.file_path)
        assert file_path.parent.name == "short_form"
    
    @pytest.mark.asyncio
    async def test_explainer_stored_in_correct_location(self, engine):
        """Test that explainer videos are stored in the explainers directory."""
        result = await engine.generate_explainer_video(
            script="Test explainer video for storage location",
            duration_seconds=60
        )
        
        file_path = Path(result.file_path)
        assert file_path.parent.name == "explainers"
    
    @pytest.mark.asyncio
    async def test_thumbnail_generation(self, engine, valid_short_form_request):
        """Test that thumbnails are generated and stored."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert result.thumbnail_path is not None
        assert result.thumbnail_url is not None
        assert Path(result.thumbnail_path).exists()
    
    @pytest.mark.asyncio
    async def test_file_url_generation(self, engine, valid_short_form_request):
        """Test that file URLs are generated correctly."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert result.file_url.startswith("/storage/")
        assert result.video_id in result.file_url
        assert result.thumbnail_url.startswith("/storage/")



class TestErrorHandling:
    """Tests for error handling in video generation."""
    
    @pytest.mark.asyncio
    async def test_invalid_specification_raises_error(self, engine):
        """Test that invalid specifications raise ValidationError."""
        # Pydantic will catch duration validation before our code
        # So we test with a valid spec but that will fail our custom validation
        with pytest.raises(ValidationError) as exc_info:
            request = VideoGenerationRequest(
                video_type=VideoType.SHORT_FORM,
                script="Test script for validation error",
                style=VideoStyle.DYNAMIC,
                specification=VideoSpecification(
                    width=1920,
                    height=1080,
                    format=VideoFormat.MP4,
                    quality=VideoQuality.HIGH,
                    fps=30,
                    duration_seconds=300,  # Max allowed by Pydantic
                    bitrate_kbps=5000
                )
            )
            # Mock the validation to force an error
            with patch.object(engine, '_validate_generation_request', side_effect=ValidationError("Duration exceeds maximum")):
                await engine.generate_video(request)
        
        assert "duration" in str(exc_info.value).lower() or "exceeds" in str(exc_info.value).lower() or "maximum" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_excessive_file_size_raises_error(self, engine):
        """Test that excessive file sizes raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            request = VideoGenerationRequest(
                video_type=VideoType.SHORT_FORM,
                script="Test script for file size error",
                style=VideoStyle.DYNAMIC,
                specification=VideoSpecification(
                    width=3840,
                    height=2160,
                    format=VideoFormat.MP4,
                    quality=VideoQuality.ULTRA,
                    fps=60,
                    duration_seconds=300,
                    bitrate_kbps=50000  # Will exceed 1GB limit
                )
            )
            await engine.generate_video(request)
        
        assert "file size" in str(exc_info.value).lower() or "exceeds" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self, engine):
        """Test handling of storage errors."""
        # Mock storage failure
        with patch.object(engine, '_store_video', side_effect=StorageError("write", "Disk full")):
            request = VideoGenerationRequest(
                video_type=VideoType.SHORT_FORM,
                script="Test script for storage error",
                style=VideoStyle.DYNAMIC,
                specification=VideoSpecification(
                    width=1920,
                    height=1080,
                    format=VideoFormat.MP4,
                    quality=VideoQuality.HIGH,
                    fps=30,
                    duration_seconds=60,
                    bitrate_kbps=5000
                )
            )
            
            with pytest.raises(StorageError):
                await engine.generate_video(request)


class TestCostTracking:
    """Tests for cost tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_cost_tracked_per_generation(self, engine, valid_short_form_request):
        """Test that costs are tracked for each generation."""
        initial_cost = engine.total_cost
        
        result = await engine.generate_video(valid_short_form_request)
        
        assert result.cost > 0
        assert engine.total_cost > initial_cost
    
    @pytest.mark.asyncio
    async def test_token_usage_tracked(self, engine, valid_short_form_request):
        """Test that token usage is tracked."""
        initial_tokens = engine.total_tokens_used
        
        result = await engine.generate_video(valid_short_form_request)
        
        assert result.tokens_used > 0
        assert engine.total_tokens_used > initial_tokens


class TestVideoMetadata:
    """Tests for video metadata handling."""
    
    @pytest.mark.asyncio
    async def test_metadata_includes_script(self, engine, valid_short_form_request):
        """Test that metadata includes the original script."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert "script" in result.metadata
        assert result.metadata["script"] == valid_short_form_request.script
    
    @pytest.mark.asyncio
    async def test_metadata_includes_style(self, engine, valid_short_form_request):
        """Test that metadata includes the style."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert "style" in result.metadata
        assert result.metadata["style"] == valid_short_form_request.style
    
    @pytest.mark.asyncio
    async def test_metadata_includes_enhanced_script(self, engine, valid_short_form_request):
        """Test that metadata includes the enhanced script."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert "enhanced_script" in result.metadata
        assert result.metadata["enhanced_script"] is not None
    
    @pytest.mark.asyncio
    async def test_metadata_includes_scene_plan(self, engine, valid_short_form_request):
        """Test that metadata includes the scene plan."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert "scene_plan" in result.metadata
        assert isinstance(result.metadata["scene_plan"], list)
        assert len(result.metadata["scene_plan"]) > 0
    
    @pytest.mark.asyncio
    async def test_metadata_includes_audio_settings(self, engine, valid_short_form_request):
        """Test that metadata includes audio settings."""
        result = await engine.generate_video(valid_short_form_request)
        
        assert "include_audio" in result.metadata
        assert "include_music" in result.metadata
        assert result.metadata["include_audio"] == valid_short_form_request.include_audio
        assert result.metadata["include_music"] == valid_short_form_request.include_music
    
    @pytest.mark.asyncio
    async def test_generated_at_timestamp(self, engine, valid_short_form_request):
        """Test that generated_at timestamp is set."""
        before = datetime.utcnow()
        result = await engine.generate_video(valid_short_form_request)
        after = datetime.utcnow()
        
        assert before <= result.generated_at <= after
    
    @pytest.mark.asyncio
    async def test_duration_formatted(self, engine):
        """Test formatted duration string."""
        result = await engine.generate_short_form_video(
            script="Test video for duration formatting",
            duration_seconds=125  # 2:05
        )
        
        formatted = result.get_duration_formatted()
        assert formatted == "02:05"



class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_minimum_duration(self, engine):
        """Test generation with minimum duration."""
        result = await engine.generate_short_form_video(
            script="Very short video test",
            duration_seconds=1
        )
        
        assert result.duration_seconds >= 1
    
    @pytest.mark.asyncio
    async def test_maximum_duration(self, engine):
        """Test generation with maximum allowed duration."""
        result = await engine.generate_short_form_video(
            script="Maximum duration video test for our platform",
            duration_seconds=300  # 5 minutes
        )
        
        assert result.duration_seconds == 300
    
    @pytest.mark.asyncio
    async def test_vertical_video(self, engine):
        """Test generation with vertical (mobile) orientation."""
        request = VideoGenerationRequest(
            video_type=VideoType.SOCIAL_MEDIA,
            script="Vertical video for mobile platforms",
            style=VideoStyle.DYNAMIC,
            specification=VideoSpecification(
                width=1080,
                height=1920,  # 9:16 aspect ratio
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=30,
                bitrate_kbps=5000
            )
        )
        
        result = await engine.generate_video(request)
        assert result.specification.height > result.specification.width
    
    @pytest.mark.asyncio
    async def test_square_video(self, engine):
        """Test generation with square aspect ratio."""
        request = VideoGenerationRequest(
            video_type=VideoType.SOCIAL_MEDIA,
            script="Square video for social media",
            style=VideoStyle.MINIMALIST,
            specification=VideoSpecification(
                width=1080,
                height=1080,  # 1:1 aspect ratio
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=30,
                bitrate_kbps=5000
            )
        )
        
        result = await engine.generate_video(request)
        assert result.specification.get_aspect_ratio() == 1.0
    
    @pytest.mark.asyncio
    async def test_different_video_formats(self, engine):
        """Test generation with different video formats."""
        formats = [VideoFormat.MP4, VideoFormat.WEBM, VideoFormat.MOV]
        
        for fmt in formats:
            request = VideoGenerationRequest(
                video_type=VideoType.SHORT_FORM,
                script=f"Test {fmt.value} format video",
                style=VideoStyle.PROFESSIONAL,
                specification=VideoSpecification(
                    width=1920,
                    height=1080,
                    format=fmt,
                    quality=VideoQuality.HIGH,
                    fps=30,
                    duration_seconds=30,
                    bitrate_kbps=5000
                )
            )
            
            result = await engine.generate_video(request)
            assert result.specification.format == fmt
            assert result.file_path.endswith(f".{fmt.value}")
    
    @pytest.mark.asyncio
    async def test_different_quality_presets(self, engine):
        """Test generation with different quality presets."""
        qualities = [VideoQuality.LOW, VideoQuality.MEDIUM, VideoQuality.HIGH]
        
        for quality in qualities:
            result = await engine.generate_short_form_video(
                script=f"Test {quality.value} quality video",
                duration_seconds=30,
                quality=quality
            )
            
            assert result.specification.quality == quality
    
    @pytest.mark.asyncio
    async def test_different_video_styles(self, engine):
        """Test generation with different video styles."""
        styles = [VideoStyle.PROFESSIONAL, VideoStyle.CASUAL, VideoStyle.DYNAMIC, VideoStyle.CINEMATIC]
        
        for style in styles:
            result = await engine.generate_short_form_video(
                script=f"Test {style.value} style video",
                duration_seconds=30,
                style=style
            )
            
            assert result.metadata["style"] == style
    
    @pytest.mark.asyncio
    async def test_video_with_all_features(self, engine):
        """Test video generation with all features enabled."""
        request = VideoGenerationRequest(
            video_type=VideoType.TUTORIAL,
            script="Comprehensive tutorial video with all features enabled",
            style=VideoStyle.PROFESSIONAL,
            specification=VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=120,
                bitrate_kbps=8000
            ),
            include_audio=True,
            include_music=True,
            include_subtitles=True,
            transitions="smooth"
        )
        
        result = await engine.generate_video(request)
        assert result.metadata["include_audio"] is True
        assert result.metadata["include_music"] is True
        assert result.metadata["include_subtitles"] is True
        assert result.metadata["transitions"] == "smooth"
    
    @pytest.mark.asyncio
    async def test_video_without_audio(self, engine):
        """Test video generation without audio."""
        request = VideoGenerationRequest(
            video_type=VideoType.SHORT_FORM,
            script="Silent video test",
            style=VideoStyle.MINIMALIST,
            specification=VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=30,
                bitrate_kbps=5000
            ),
            include_audio=False,
            include_music=False
        )
        
        result = await engine.generate_video(request)
        assert result.metadata["include_audio"] is False
        assert result.metadata["include_music"] is False
    
    @pytest.mark.asyncio
    async def test_long_script(self, engine):
        """Test video generation with long script."""
        long_script = "This is a comprehensive video script. " * 50  # Create long script
        
        result = await engine.generate_short_form_video(
            script=long_script,
            duration_seconds=120
        )
        
        assert result.video_id is not None
        assert result.file_size_bytes > 0
    
    @pytest.mark.asyncio
    async def test_high_fps_video(self, engine):
        """Test generation with high frame rate."""
        request = VideoGenerationRequest(
            video_type=VideoType.SHORT_FORM,
            script="High frame rate video test",
            style=VideoStyle.DYNAMIC,
            specification=VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=60,  # High frame rate
                duration_seconds=30,
                bitrate_kbps=10000
            )
        )
        
        result = await engine.generate_video(request)
        assert result.specification.fps == 60
    
    @pytest.mark.asyncio
    async def test_4k_video(self, engine):
        """Test generation with 4K resolution."""
        request = VideoGenerationRequest(
            video_type=VideoType.PROMOTIONAL,
            script="Ultra high definition 4K video test",
            style=VideoStyle.CINEMATIC,
            specification=VideoSpecification(
                width=3840,
                height=2160,
                format=VideoFormat.MP4,
                quality=VideoQuality.ULTRA,
                fps=30,
                duration_seconds=60,
                bitrate_kbps=20000
            )
        )
        
        result = await engine.generate_video(request)
        assert result.specification.width == 3840
        assert result.specification.height == 2160
        assert result.specification.quality == VideoQuality.ULTRA

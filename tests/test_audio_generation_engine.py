"""
Unit tests for Audio Generation Engine.

Tests cover:
- Audio generation with various specifications
- Voiceover, narration, and background music generation
- Audio specification validation
- Storage integration
- Error handling
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.ai.audio_generation_engine import (
    AudioGenerationEngine,
    AudioGenerationRequest,
    AudioSpecification,
    AudioType,
    AudioFormat,
    VoiceStyle,
    MusicGenre,
    GeneratedAudio
)
from app.core.exceptions import ValidationError, EngineError, StorageError


@pytest.fixture
def engine():
    """Create an Audio Generation Engine instance for testing."""
    with patch('app.ai.audio_generation_engine.settings') as mock_settings:
        mock_settings.GOOGLE_API_KEY = "test_key"
        mock_settings.LOCAL_STORAGE_PATH = "./test_audio_storage"
        engine = AudioGenerationEngine()
        yield engine
        # Cleanup test storage
        import shutil
        if Path("./test_audio_storage").exists():
            shutil.rmtree("./test_audio_storage")


@pytest.fixture
def valid_voiceover_request():
    """Create a valid voiceover generation request."""
    return AudioGenerationRequest(
        audio_type=AudioType.VOICEOVER,
        text="Welcome to our tutorial on artificial intelligence and machine learning.",
        voice_style=VoiceStyle.PROFESSIONAL,
        specification=AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=44100,
            bitrate=128,
            channels=2
        )
    )


@pytest.fixture
def valid_music_request():
    """Create a valid background music generation request."""
    return AudioGenerationRequest(
        audio_type=AudioType.BACKGROUND_MUSIC,
        music_genre=MusicGenre.AMBIENT,
        mood="calm and focused",
        tempo="medium",
        specification=AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=44100,
            bitrate=192,
            channels=2,
            duration_seconds=60
        )
    )


class TestAudioSpecification:
    """Tests for AudioSpecification model."""
    
    def test_valid_specification(self):
        """Test creating a valid audio specification."""
        spec = AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=44100,
            bitrate=128,
            channels=2
        )
        assert spec.format == AudioFormat.MP3
        assert spec.sample_rate == 44100
        assert spec.bitrate == 128
        assert spec.channels == 2
    
    def test_sample_rate_normalization(self):
        """Test that sample rates are normalized to common values."""
        spec = AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=45000,  # Should normalize to 44100
            bitrate=128,
            channels=2
        )
        assert spec.sample_rate in [8000, 16000, 22050, 44100, 48000, 96000, 192000]
    
    def test_file_size_estimation(self):
        """Test file size estimation."""
        spec = AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=44100,
            bitrate=128,
            channels=2
        )
        estimated_size = spec.get_estimated_file_size_mb(60)  # 60 seconds
        assert estimated_size > 0
        assert estimated_size < 10  # Should be reasonable for 60s at 128kbps
    
    def test_invalid_sample_rate(self):
        """Test validation of invalid sample rates."""
        with pytest.raises(ValueError):
            AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=5000,  # Too low
                bitrate=128,
                channels=2
            )
    
    def test_invalid_bitrate(self):
        """Test validation of invalid bitrates."""
        with pytest.raises(ValueError):
            AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=44100,
                bitrate=32,  # Too low
                channels=2
            )
        
        with pytest.raises(ValueError):
            AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=44100,
                bitrate=400,  # Too high
                channels=2
            )
    
    def test_invalid_channels(self):
        """Test validation of invalid channel counts."""
        with pytest.raises(ValueError):
            AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=44100,
                bitrate=128,
                channels=0
            )
        
        with pytest.raises(ValueError):
            AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=44100,
                bitrate=128,
                channels=3
            )


class TestAudioGenerationRequest:
    """Tests for AudioGenerationRequest model."""
    
    def test_valid_voiceover_request(self, valid_voiceover_request):
        """Test creating a valid voiceover request."""
        assert valid_voiceover_request.audio_type == AudioType.VOICEOVER
        assert valid_voiceover_request.text is not None
        assert valid_voiceover_request.voice_style == VoiceStyle.PROFESSIONAL
    
    def test_valid_music_request(self, valid_music_request):
        """Test creating a valid music request."""
        assert valid_music_request.audio_type == AudioType.BACKGROUND_MUSIC
        assert valid_music_request.music_genre == MusicGenre.AMBIENT
        assert valid_music_request.specification.duration_seconds == 60
    
    def test_voiceover_requires_text(self):
        """Test that voiceover requests require text."""
        with pytest.raises(ValueError) as exc_info:
            AudioGenerationRequest(
                audio_type=AudioType.VOICEOVER,
                text="",
                voice_style=VoiceStyle.PROFESSIONAL,
                specification=AudioSpecification(
                    format=AudioFormat.MP3,
                    sample_rate=44100,
                    bitrate=128,
                    channels=2
                )
            )
        assert "required" in str(exc_info.value).lower()
    
    def test_narration_requires_text(self):
        """Test that narration requests require text."""
        with pytest.raises(ValueError) as exc_info:
            AudioGenerationRequest(
                audio_type=AudioType.NARRATION,
                text=None,
                voice_style=VoiceStyle.CALM,
                specification=AudioSpecification(
                    format=AudioFormat.MP3,
                    sample_rate=44100,
                    bitrate=128,
                    channels=2
                )
            )
        assert "required" in str(exc_info.value).lower()
    
    def test_music_default_genre(self):
        """Test that background music gets default genre if not specified."""
        request = AudioGenerationRequest(
            audio_type=AudioType.BACKGROUND_MUSIC,
            specification=AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=44100,
                bitrate=192,
                channels=2,
                duration_seconds=30
            )
        )
        assert request.music_genre == MusicGenre.AMBIENT
    
    def test_text_whitespace_trimming(self):
        """Test that text whitespace is trimmed."""
        request = AudioGenerationRequest(
            audio_type=AudioType.VOICEOVER,
            text="  Test voiceover text  ",
            voice_style=VoiceStyle.PROFESSIONAL,
            specification=AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=44100,
                bitrate=128,
                channels=2
            )
        )
        assert request.text == "Test voiceover text"


class TestAudioGenerationEngine:
    """Tests for AudioGenerationEngine class."""
    
    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.storage_path.exists()
        assert engine.audio_path.exists()
        assert engine.voiceovers_path.exists()
        assert engine.narrations_path.exists()
        assert engine.music_path.exists()
    
    @pytest.mark.asyncio
    async def test_generate_voiceover(self, engine):
        """Test voiceover generation."""
        result = await engine.generate_voiceover(
            text="This is a test voiceover for our application.",
            voice_style=VoiceStyle.PROFESSIONAL
        )
        
        assert isinstance(result, GeneratedAudio)
        assert result.audio_type == AudioType.VOICEOVER
        assert result.duration_seconds > 0
        assert result.file_size_bytes > 0
        assert result.audio_id is not None
    
    @pytest.mark.asyncio
    async def test_generate_narration(self, engine):
        """Test narration generation."""
        result = await engine.generate_narration(
            text="Once upon a time, in a land far away, there lived a wise old wizard.",
            voice_style=VoiceStyle.CALM
        )
        
        assert isinstance(result, GeneratedAudio)
        assert result.audio_type == AudioType.NARRATION
        assert result.duration_seconds > 0
        assert result.file_size_bytes > 0
        assert result.audio_id is not None
    
    @pytest.mark.asyncio
    async def test_generate_background_music(self, engine):
        """Test background music generation."""
        result = await engine.generate_background_music(
            genre=MusicGenre.AMBIENT,
            duration_seconds=30,
            mood="peaceful",
            tempo="slow"
        )
        
        assert isinstance(result, GeneratedAudio)
        assert result.audio_type == AudioType.BACKGROUND_MUSIC
        assert result.duration_seconds == 30
        assert result.file_size_bytes > 0
        assert result.audio_id is not None
    
    @pytest.mark.asyncio
    async def test_generate_audio_with_custom_request(self, engine, valid_voiceover_request):
        """Test audio generation with custom request."""
        result = await engine.generate_audio(valid_voiceover_request)
        
        assert isinstance(result, GeneratedAudio)
        assert result.audio_type == valid_voiceover_request.audio_type
        assert result.specification == valid_voiceover_request.specification
        assert result.metadata["text"] == valid_voiceover_request.text
        assert result.metadata["voice_style"] == valid_voiceover_request.voice_style
    
    @pytest.mark.asyncio
    async def test_validate_audio_specification(self, engine):
        """Test audio specification validation."""
        spec = AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=44100,
            bitrate=128,
            channels=2
        )
        
        validation = await engine.validate_audio_specification(spec)
        
        assert validation["is_valid"] is True
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)
        assert isinstance(validation["recommendations"], list)
    
    @pytest.mark.asyncio
    async def test_validate_low_sample_rate_warning(self, engine):
        """Test that low sample rates generate warnings."""
        spec = AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=16000,
            bitrate=128,
            channels=2
        )
        
        validation = await engine.validate_audio_specification(spec)
        
        assert len(validation["warnings"]) > 0
        assert any("Low sample rate" in w for w in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_validate_low_bitrate_warning(self, engine):
        """Test that low bitrate MP3 generates warnings."""
        spec = AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=44100,
            bitrate=96,
            channels=2
        )
        
        validation = await engine.validate_audio_specification(spec)
        
        assert len(validation["warnings"]) > 0
        assert any("bitrate below 128" in w for w in validation["warnings"])
    
    @pytest.mark.asyncio
    async def test_validate_mono_recommendation(self, engine):
        """Test that mono audio generates recommendations."""
        spec = AudioSpecification(
            format=AudioFormat.MP3,
            sample_rate=44100,
            bitrate=128,
            channels=1
        )
        
        validation = await engine.validate_audio_specification(spec)
        
        assert len(validation["recommendations"]) > 0
        assert any("stereo" in r.lower() for r in validation["recommendations"])
    
    @pytest.mark.asyncio
    async def test_get_audio(self, engine, valid_voiceover_request):
        """Test retrieving generated audio metadata."""
        # Generate audio first
        generated = await engine.generate_audio(valid_voiceover_request)
        
        # Retrieve it
        retrieved = await engine.get_audio(generated.audio_id)
        
        assert retrieved is not None
        assert retrieved["audio_id"] == generated.audio_id
        assert retrieved["exists"] is True
        assert retrieved["file_size"] > 0
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_audio(self, engine):
        """Test retrieving non-existent audio returns None."""
        result = await engine.get_audio("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_audio(self, engine, valid_voiceover_request):
        """Test deleting a generated audio file."""
        # Generate audio first
        generated = await engine.generate_audio(valid_voiceover_request)
        
        # Delete it
        deleted = await engine.delete_audio(generated.audio_id)
        assert deleted is True
        
        # Verify it's gone
        retrieved = await engine.get_audio(generated.audio_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_audio(self, engine):
        """Test deleting non-existent audio returns False."""
        result = await engine.delete_audio("nonexistent_id")
        assert result is False
    
    def test_usage_stats_tracking(self, engine):
        """Test that usage statistics are tracked."""
        initial_stats = engine.get_usage_stats()
        assert "total_tokens_used" in initial_stats
        assert "total_cost" in initial_stats
        assert "cost_per_audio" in initial_stats
    
    def test_reset_usage_stats(self, engine):
        """Test resetting usage statistics."""
        engine.total_tokens_used = 1000
        engine.total_cost = 10.0
        
        engine.reset_usage_stats()
        
        assert engine.total_tokens_used == 0
        assert engine.total_cost == 0.0


class TestAudioStorage:
    """Tests for audio storage functionality."""
    
    @pytest.mark.asyncio
    async def test_storage_path_creation(self, engine):
        """Test that storage paths are created correctly."""
        assert engine.storage_path.exists()
        assert engine.audio_path.exists()
        assert engine.voiceovers_path.exists()
        assert engine.narrations_path.exists()
        assert engine.music_path.exists()
    
    @pytest.mark.asyncio
    async def test_voiceover_stored_in_correct_location(self, engine):
        """Test that voiceovers are stored in the voiceovers directory."""
        result = await engine.generate_voiceover(
            text="Test voiceover for storage location",
            voice_style=VoiceStyle.PROFESSIONAL
        )
        
        file_path = Path(result.file_path)
        assert file_path.parent.name == "voiceovers"
    
    @pytest.mark.asyncio
    async def test_narration_stored_in_correct_location(self, engine):
        """Test that narrations are stored in the narrations directory."""
        result = await engine.generate_narration(
            text="Test narration for storage location",
            voice_style=VoiceStyle.CALM
        )
        
        file_path = Path(result.file_path)
        assert file_path.parent.name == "narrations"
    
    @pytest.mark.asyncio
    async def test_music_stored_in_correct_location(self, engine):
        """Test that music is stored in the music directory."""
        result = await engine.generate_background_music(
            genre=MusicGenre.AMBIENT,
            duration_seconds=10
        )
        
        file_path = Path(result.file_path)
        assert file_path.parent.name == "music"
    
    @pytest.mark.asyncio
    async def test_file_url_generation(self, engine, valid_voiceover_request):
        """Test that file URLs are generated correctly."""
        result = await engine.generate_audio(valid_voiceover_request)
        
        assert result.file_url.startswith("/storage/")
        assert result.audio_id in result.file_url


class TestErrorHandling:
    """Tests for error handling in audio generation."""
    
    @pytest.mark.asyncio
    async def test_invalid_specification_raises_error(self, engine):
        """Test that invalid specifications raise ValidationError."""
        # Test with a mock that will make the estimated size exceed the limit
        with patch.object(engine, '_validate_generation_request', side_effect=ValidationError("Estimated file size exceeds limit")):
            request = AudioGenerationRequest(
                audio_type=AudioType.BACKGROUND_MUSIC,
                music_genre=MusicGenre.CINEMATIC,
                specification=AudioSpecification(
                    format=AudioFormat.MP3,
                    sample_rate=44100,
                    bitrate=320,
                    channels=2,
                    duration_seconds=600
                )
            )
            
            with pytest.raises(ValidationError) as exc_info:
                await engine.generate_audio(request)
            
            assert "file size" in str(exc_info.value).lower() or "limit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self, engine):
        """Test handling of storage errors."""
        # Mock storage failure
        with patch.object(engine, '_store_audio', side_effect=StorageError("write", "Disk full")):
            request = AudioGenerationRequest(
                audio_type=AudioType.VOICEOVER,
                text="Test voiceover for storage error",
                voice_style=VoiceStyle.PROFESSIONAL,
                specification=AudioSpecification(
                    format=AudioFormat.MP3,
                    sample_rate=44100,
                    bitrate=128,
                    channels=2
                )
            )
            
            with pytest.raises(StorageError):
                await engine.generate_audio(request)


class TestCostTracking:
    """Tests for cost tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_cost_tracked_per_generation(self, engine, valid_voiceover_request):
        """Test that costs are tracked for each generation."""
        initial_cost = engine.total_cost
        
        result = await engine.generate_audio(valid_voiceover_request)
        
        assert result.cost > 0
        assert engine.total_cost > initial_cost
    
    @pytest.mark.asyncio
    async def test_token_usage_tracked(self, engine, valid_voiceover_request):
        """Test that token usage is tracked."""
        initial_tokens = engine.total_tokens_used
        
        result = await engine.generate_audio(valid_voiceover_request)
        
        assert result.tokens_used > 0
        assert engine.total_tokens_used > initial_tokens


class TestAudioMetadata:
    """Tests for audio metadata handling."""
    
    @pytest.mark.asyncio
    async def test_metadata_includes_text(self, engine, valid_voiceover_request):
        """Test that metadata includes the original text."""
        result = await engine.generate_audio(valid_voiceover_request)
        
        assert "text" in result.metadata
        assert result.metadata["text"] == valid_voiceover_request.text
    
    @pytest.mark.asyncio
    async def test_metadata_includes_voice_style(self, engine, valid_voiceover_request):
        """Test that metadata includes the voice style."""
        result = await engine.generate_audio(valid_voiceover_request)
        
        assert "voice_style" in result.metadata
        assert result.metadata["voice_style"] == valid_voiceover_request.voice_style
    
    @pytest.mark.asyncio
    async def test_metadata_includes_music_genre(self, engine, valid_music_request):
        """Test that metadata includes the music genre."""
        result = await engine.generate_audio(valid_music_request)
        
        assert "music_genre" in result.metadata
        assert result.metadata["music_genre"] == valid_music_request.music_genre
    
    @pytest.mark.asyncio
    async def test_metadata_includes_enhanced_text(self, engine, valid_voiceover_request):
        """Test that metadata includes the enhanced text."""
        result = await engine.generate_audio(valid_voiceover_request)
        
        assert "enhanced_text" in result.metadata
        assert result.metadata["enhanced_text"] is not None
    
    @pytest.mark.asyncio
    async def test_generated_at_timestamp(self, engine, valid_voiceover_request):
        """Test that generated_at timestamp is set."""
        before = datetime.utcnow()
        result = await engine.generate_audio(valid_voiceover_request)
        after = datetime.utcnow()
        
        assert before <= result.generated_at <= after
    
    @pytest.mark.asyncio
    async def test_duration_formatted(self, engine):
        """Test formatted duration string."""
        result = await engine.generate_background_music(
            genre=MusicGenre.AMBIENT,
            duration_seconds=125  # 2:05
        )
        
        formatted = result.get_duration_formatted()
        assert formatted == "02:05"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_minimum_duration(self, engine):
        """Test generation with minimum duration."""
        result = await engine.generate_background_music(
            genre=MusicGenre.AMBIENT,
            duration_seconds=1
        )
        
        assert result.duration_seconds >= 1
    
    @pytest.mark.asyncio
    async def test_maximum_duration(self, engine):
        """Test generation with maximum allowed duration."""
        result = await engine.generate_background_music(
            genre=MusicGenre.AMBIENT,
            duration_seconds=60,  # Keep reasonable for testing
            format=AudioFormat.MP3
        )
        
        assert result.duration_seconds == 60
    
    @pytest.mark.asyncio
    async def test_mono_audio(self, engine):
        """Test generation with mono audio."""
        request = AudioGenerationRequest(
            audio_type=AudioType.VOICEOVER,
            text="Test mono audio generation",
            voice_style=VoiceStyle.PROFESSIONAL,
            specification=AudioSpecification(
                format=AudioFormat.MP3,
                sample_rate=44100,
                bitrate=128,
                channels=1  # Mono
            )
        )
        
        result = await engine.generate_audio(request)
        assert result.specification.channels == 1
    
    @pytest.mark.asyncio
    async def test_different_audio_formats(self, engine):
        """Test generation with different audio formats."""
        formats = [AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OGG]
        
        for fmt in formats:
            request = AudioGenerationRequest(
                audio_type=AudioType.VOICEOVER,
                text=f"Test {fmt.value} format",
                voice_style=VoiceStyle.PROFESSIONAL,
                specification=AudioSpecification(
                    format=fmt,
                    sample_rate=44100,
                    bitrate=128,
                    channels=2
                )
            )
            
            result = await engine.generate_audio(request)
            assert result.specification.format == fmt
            assert result.file_path.endswith(f".{fmt.value}")
    
    @pytest.mark.asyncio
    async def test_different_voice_styles(self, engine):
        """Test generation with different voice styles."""
        styles = [VoiceStyle.PROFESSIONAL, VoiceStyle.CASUAL, VoiceStyle.ENERGETIC]
        
        for style in styles:
            result = await engine.generate_voiceover(
                text=f"Test {style.value} voice style",
                voice_style=style
            )
            
            assert result.metadata["voice_style"] == style
    
    @pytest.mark.asyncio
    async def test_different_music_genres(self, engine):
        """Test generation with different music genres."""
        genres = [MusicGenre.AMBIENT, MusicGenre.CORPORATE, MusicGenre.UPBEAT]
        
        for genre in genres:
            result = await engine.generate_background_music(
                genre=genre,
                duration_seconds=10
            )
            
            assert result.metadata["music_genre"] == genre
    
    @pytest.mark.asyncio
    async def test_long_text_voiceover(self, engine):
        """Test voiceover generation with long text."""
        long_text = "This is a test. " * 100  # Repeat to create long text
        
        result = await engine.generate_voiceover(
            text=long_text,
            voice_style=VoiceStyle.PROFESSIONAL
        )
        
        assert result.audio_id is not None
        assert result.file_size_bytes > 0
    
    @pytest.mark.asyncio
    async def test_high_quality_audio(self, engine):
        """Test generation with high quality settings."""
        request = AudioGenerationRequest(
            audio_type=AudioType.BACKGROUND_MUSIC,
            music_genre=MusicGenre.CLASSICAL,
            specification=AudioSpecification(
                format=AudioFormat.FLAC,
                sample_rate=96000,
                bitrate=320,
                channels=2,
                duration_seconds=30
            )
        )
        
        result = await engine.generate_audio(request)
        assert result.specification.sample_rate == 96000
        assert result.specification.bitrate == 320

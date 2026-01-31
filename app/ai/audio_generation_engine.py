"""
Audio Generation Engine for ContentFlow AI.

This module implements the Audio Generation Engine responsible for creating
audio content including voiceovers, narration, and background music with
secure storage integration.
"""

import asyncio
import logging
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


class AudioType(str, Enum):
    """Enumeration of audio generation types."""
    VOICEOVER = "voiceover"
    NARRATION = "narration"
    BACKGROUND_MUSIC = "background_music"
    SOUND_EFFECT = "sound_effect"
    PODCAST_INTRO = "podcast_intro"
    JINGLE = "jingle"


class AudioFormat(str, Enum):
    """Enumeration of supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    AAC = "aac"
    FLAC = "flac"


class VoiceStyle(str, Enum):
    """Enumeration of voice styles for voiceovers."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENERGETIC = "energetic"
    CALM = "calm"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"
    DRAMATIC = "dramatic"
    NEUTRAL = "neutral"


class MusicGenre(str, Enum):
    """Enumeration of music genres for background music."""
    AMBIENT = "ambient"
    CORPORATE = "corporate"
    UPBEAT = "upbeat"
    CINEMATIC = "cinematic"
    ELECTRONIC = "electronic"
    ACOUSTIC = "acoustic"
    CLASSICAL = "classical"
    JAZZ = "jazz"


class AudioSpecification(BaseModel):
    """Model for audio generation specifications."""
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int = Field(default=44100, ge=8000, le=192000, description="Sample rate in Hz")
    bitrate: int = Field(default=128, ge=64, le=320, description="Bitrate in kbps")
    channels: int = Field(default=2, ge=1, le=2, description="Number of audio channels (1=mono, 2=stereo)")
    duration_seconds: Optional[int] = Field(default=None, ge=1, le=600, description="Target duration in seconds")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        """Validate sample rate is a common value."""
        common_rates = [8000, 16000, 22050, 44100, 48000, 96000, 192000]
        if v not in common_rates:
            # Find closest common rate
            v = min(common_rates, key=lambda x: abs(x - v))
        return v
    
    def get_estimated_file_size_mb(self, duration_seconds: int) -> float:
        """Estimate file size in MB based on duration."""
        # Rough estimation: bitrate (kbps) * duration (s) / 8 / 1024
        return (self.bitrate * duration_seconds) / (8 * 1024)


class AudioGenerationRequest(BaseModel):
    """Model for audio generation requests."""
    audio_type: AudioType
    text: Optional[str] = Field(default=None, max_length=5000, description="Text for voiceover/narration")
    voice_style: Optional[VoiceStyle] = Field(default=VoiceStyle.PROFESSIONAL, description="Voice style for speech")
    music_genre: Optional[MusicGenre] = Field(default=None, description="Genre for background music")
    specification: AudioSpecification
    mood: Optional[str] = Field(default=None, max_length=100, description="Mood or emotion for the audio")
    tempo: Optional[str] = Field(default="medium", description="Tempo for music (slow, medium, fast)")
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('text')
    def validate_text(cls, v, values):
        """Validate text is provided for speech-based audio types."""
        audio_type = values.get('audio_type')
        if audio_type in [AudioType.VOICEOVER, AudioType.NARRATION]:
            if not v or not v.strip():
                raise ValueError(f"Text is required for {audio_type.value}")
        return v.strip() if v else None
    
    @validator('music_genre', always=True)
    def validate_music_genre(cls, v, values):
        """Validate music genre is provided for music-based audio types."""
        audio_type = values.get('audio_type')
        if audio_type == AudioType.BACKGROUND_MUSIC and v is None:
            return MusicGenre.AMBIENT  # Default genre
        return v


class GeneratedAudio(BaseModel):
    """Model for generated audio metadata."""
    audio_id: str
    audio_type: AudioType
    file_path: str
    file_url: str
    specification: AudioSpecification
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    cost: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)
    
    def get_duration_formatted(self) -> str:
        """Get formatted duration string (MM:SS)."""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


class AudioGenerationEngine:
    """
    Audio Generation Engine for creating audio content.
    
    This engine handles:
    - Voiceover and narration generation
    - Background music creation
    - Audio format validation and processing
    - Secure storage integration for generated audio
    - Cost tracking and usage monitoring
    """
    
    def __init__(self):
        """Initialize the Audio Generation Engine."""
        self.gemini_client = None
        self.storage_path = Path(settings.LOCAL_STORAGE_PATH)
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cost_per_audio = 0.05  # $0.05 per audio generation (example rate)
        self._initialize_storage()
        self._initialize_gemini()
    
    def _initialize_storage(self):
        """Initialize storage backend."""
        try:
            # Create storage directories if they don't exist
            self.audio_path = self.storage_path / "audio"
            self.voiceovers_path = self.audio_path / "voiceovers"
            self.narrations_path = self.audio_path / "narrations"
            self.music_path = self.audio_path / "music"
            
            for path in [self.audio_path, self.voiceovers_path, self.narrations_path, self.music_path]:
                path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Audio storage initialized at {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise StorageError("initialization", str(e))
    
    def _initialize_gemini(self):
        """Initialize Google Gemini client for audio generation assistance."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. Audio Generation Engine will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            # Note: Gemini can help with script enhancement and metadata
            # In production, you'd integrate with ElevenLabs, Google TTS, or similar
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Audio Generation Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise EngineError("audio_generation", f"Initialization failed: {e}")
    
    async def generate_audio(
        self, 
        request: AudioGenerationRequest
    ) -> GeneratedAudio:
        """
        Generate audio based on the provided request.
        
        Args:
            request: Audio generation request with specifications
            
        Returns:
            GeneratedAudio with metadata and storage information
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If audio generation fails
            StorageError: If storage operations fail
        """
        try:
            logger.info(f"Generating {request.audio_type} audio")
            
            # Validate request
            await self._validate_generation_request(request)
            
            # Enhance text/script using AI if available
            enhanced_text = await self._enhance_script(request)
            
            # Generate audio (simulated for now - would call actual audio generation API)
            audio_data, duration = await self._generate_audio_data(enhanced_text, request)
            
            # Store audio securely
            stored_audio = await self._store_audio(audio_data, request)
            
            # Calculate cost
            tokens_used = self._estimate_tokens(request.text or "")
            cost = self.cost_per_audio
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = GeneratedAudio(
                audio_id=stored_audio["audio_id"],
                audio_type=request.audio_type,
                file_path=stored_audio["file_path"],
                file_url=stored_audio["file_url"],
                specification=request.specification,
                duration_seconds=duration,
                file_size_bytes=stored_audio["file_size"],
                metadata={
                    "text": request.text,
                    "enhanced_text": enhanced_text,
                    "voice_style": request.voice_style,
                    "music_genre": request.music_genre,
                    "mood": request.mood,
                    "tempo": request.tempo,
                    "context": request.context
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Audio generation completed: {result.audio_id}")
            return result
            
        except ValidationError:
            raise
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise EngineError("audio_generation", f"Audio generation failed: {e}")
    
    async def generate_voiceover(
        self,
        text: str,
        voice_style: VoiceStyle = VoiceStyle.PROFESSIONAL,
        format: AudioFormat = AudioFormat.MP3
    ) -> GeneratedAudio:
        """
        Generate a voiceover from text.
        
        Args:
            text: Text to convert to speech
            voice_style: Voice style for the voiceover
            format: Audio format for output
            
        Returns:
            GeneratedAudio with voiceover metadata
        """
        request = AudioGenerationRequest(
            audio_type=AudioType.VOICEOVER,
            text=text,
            voice_style=voice_style,
            specification=AudioSpecification(
                format=format,
                sample_rate=44100,
                bitrate=128,
                channels=2
            )
        )
        return await self.generate_audio(request)
    
    async def generate_narration(
        self,
        text: str,
        voice_style: VoiceStyle = VoiceStyle.CALM,
        format: AudioFormat = AudioFormat.MP3
    ) -> GeneratedAudio:
        """
        Generate narration from text.
        
        Args:
            text: Text to narrate
            voice_style: Voice style for the narration
            format: Audio format for output
            
        Returns:
            GeneratedAudio with narration metadata
        """
        request = AudioGenerationRequest(
            audio_type=AudioType.NARRATION,
            text=text,
            voice_style=voice_style,
            specification=AudioSpecification(
                format=format,
                sample_rate=44100,
                bitrate=128,
                channels=2
            )
        )
        return await self.generate_audio(request)
    
    async def generate_background_music(
        self,
        genre: MusicGenre,
        duration_seconds: int = 60,
        mood: str = "uplifting",
        tempo: str = "medium",
        format: AudioFormat = AudioFormat.MP3
    ) -> GeneratedAudio:
        """
        Generate background music.
        
        Args:
            genre: Music genre
            duration_seconds: Target duration in seconds
            mood: Mood or emotion for the music
            tempo: Tempo (slow, medium, fast)
            format: Audio format for output
            
        Returns:
            GeneratedAudio with music metadata
        """
        request = AudioGenerationRequest(
            audio_type=AudioType.BACKGROUND_MUSIC,
            music_genre=genre,
            mood=mood,
            tempo=tempo,
            specification=AudioSpecification(
                format=format,
                sample_rate=44100,
                bitrate=192,  # Higher bitrate for music
                channels=2,
                duration_seconds=duration_seconds
            )
        )
        return await self.generate_audio(request)
    
    async def validate_audio_specification(
        self,
        specification: AudioSpecification
    ) -> Dict[str, Any]:
        """
        Validate audio specification and provide recommendations.
        
        Args:
            specification: Audio specification to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check sample rate
        if specification.sample_rate < 22050:
            validation_result["warnings"].append(
                f"Low sample rate ({specification.sample_rate} Hz) may result in reduced audio quality"
            )
        
        # Check bitrate
        if specification.format == AudioFormat.MP3 and specification.bitrate < 128:
            validation_result["warnings"].append(
                "MP3 bitrate below 128 kbps may result in noticeable quality loss"
            )
        
        # Check format and bitrate combination
        if specification.format == AudioFormat.FLAC and specification.bitrate < 256:
            validation_result["recommendations"].append(
                "FLAC is a lossless format; consider using higher bitrate or switch to lossy format"
            )
        
        # Check duration
        if specification.duration_seconds and specification.duration_seconds > 300:
            estimated_size = specification.get_estimated_file_size_mb(specification.duration_seconds)
            if estimated_size > 50:
                validation_result["warnings"].append(
                    f"Long duration ({specification.duration_seconds}s) will result in large file size (~{estimated_size:.1f} MB)"
                )
        
        # Check channels
        if specification.channels == 1:
            validation_result["recommendations"].append(
                "Consider using stereo (2 channels) for better audio experience"
            )
        
        return validation_result
    
    async def get_audio(self, audio_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve audio metadata by ID.
        
        Args:
            audio_id: Unique audio identifier
            
        Returns:
            Dictionary with audio metadata or None if not found
        """
        try:
            # Search for audio in storage directories
            for base_path in [self.voiceovers_path, self.narrations_path, self.music_path, self.audio_path]:
                for file_path in base_path.glob(f"{audio_id}.*"):
                    if file_path.is_file():
                        return {
                            "audio_id": audio_id,
                            "file_path": str(file_path),
                            "file_url": self._get_file_url(file_path),
                            "file_size": file_path.stat().st_size,
                            "format": file_path.suffix[1:],
                            "exists": True
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve audio {audio_id}: {e}")
            raise StorageError("retrieval", str(e))
    
    async def delete_audio(self, audio_id: str) -> bool:
        """
        Delete an audio file from storage.
        
        Args:
            audio_id: Unique audio identifier
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            audio_info = await self.get_audio(audio_id)
            if not audio_info:
                return False
            
            file_path = Path(audio_info["file_path"])
            file_path.unlink()
            logger.info(f"Deleted audio {audio_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete audio {audio_id}: {e}")
            raise StorageError("deletion", str(e))
    
    # Private helper methods
    
    async def _validate_generation_request(self, request: AudioGenerationRequest):
        """Validate audio generation request."""
        # Validate specification
        spec_validation = await self.validate_audio_specification(request.specification)
        
        if not spec_validation["is_valid"]:
            raise ValidationError(
                "Invalid audio specification",
                details={"errors": spec_validation["errors"]}
            )
        
        # Check estimated file size
        if request.specification.duration_seconds:
            estimated_size = request.specification.get_estimated_file_size_mb(
                request.specification.duration_seconds
            )
            if estimated_size > 100:  # 100 MB limit
                raise ValidationError(
                    f"Estimated file size ({estimated_size:.2f} MB) exceeds limit"
                )
    
    async def _enhance_script(self, request: AudioGenerationRequest) -> Optional[str]:
        """Enhance the audio script using AI."""
        if not self.gemini_client or not request.text:
            return request.text
        
        try:
            enhancement_prompt = f"""
            Enhance this script for {request.audio_type.value} generation:
            
            Original text: {request.text}
            Voice style: {request.voice_style.value if request.voice_style else 'N/A'}
            Mood: {request.mood or 'neutral'}
            
            Provide an enhanced version that:
            - Improves clarity and flow
            - Adds appropriate pauses and emphasis markers
            - Maintains the original meaning
            - Optimizes for speech synthesis
            
            Return only the enhanced text, no explanations.
            """
            
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                enhancement_prompt
            )
            
            if response and response.text:
                enhanced = response.text.strip()
                logger.info("Script enhanced successfully")
                return enhanced
            
            return request.text
            
        except Exception as e:
            logger.warning(f"Script enhancement failed, using original: {e}")
            return request.text
    
    async def _generate_audio_data(
        self,
        text: Optional[str],
        request: AudioGenerationRequest
    ) -> tuple[bytes, float]:
        """
        Generate audio data using AI service.
        
        Note: This is a placeholder implementation. In production, you would
        integrate with actual audio generation APIs like:
        - ElevenLabs (voice synthesis)
        - Google Cloud Text-to-Speech
        - Amazon Polly
        - Mubert or AIVA (music generation)
        
        Returns:
            Tuple of (audio_data, duration_seconds)
        """
        logger.info(f"Generating {request.audio_type.value} audio...")
        
        # Simulate API call delay
        await asyncio.sleep(0.5)
        
        # Create a simple placeholder audio (minimal WAV header + silence)
        # In production, this would be the actual generated audio data
        sample_rate = request.specification.sample_rate
        duration = request.specification.duration_seconds or 5  # Default 5 seconds
        
        # Minimal WAV file structure (44-byte header + data)
        num_samples = sample_rate * duration
        num_channels = request.specification.channels
        bytes_per_sample = 2  # 16-bit audio
        
        # WAV header
        placeholder_wav = bytearray()
        placeholder_wav.extend(b'RIFF')
        placeholder_wav.extend((36 + num_samples * num_channels * bytes_per_sample).to_bytes(4, 'little'))
        placeholder_wav.extend(b'WAVE')
        placeholder_wav.extend(b'fmt ')
        placeholder_wav.extend((16).to_bytes(4, 'little'))
        placeholder_wav.extend((1).to_bytes(2, 'little'))  # PCM
        placeholder_wav.extend(num_channels.to_bytes(2, 'little'))
        placeholder_wav.extend(sample_rate.to_bytes(4, 'little'))
        placeholder_wav.extend((sample_rate * num_channels * bytes_per_sample).to_bytes(4, 'little'))
        placeholder_wav.extend((num_channels * bytes_per_sample).to_bytes(2, 'little'))
        placeholder_wav.extend((bytes_per_sample * 8).to_bytes(2, 'little'))
        placeholder_wav.extend(b'data')
        placeholder_wav.extend((num_samples * num_channels * bytes_per_sample).to_bytes(4, 'little'))
        
        # Add silence (zeros) for the audio data
        placeholder_wav.extend(b'\x00' * (num_samples * num_channels * bytes_per_sample))
        
        return bytes(placeholder_wav), float(duration)
    
    async def _store_audio(
        self,
        audio_data: bytes,
        request: AudioGenerationRequest
    ) -> Dict[str, Any]:
        """
        Store generated audio securely.
        
        Args:
            audio_data: Raw audio bytes
            request: Original generation request
            
        Returns:
            Dictionary with storage information
        """
        try:
            # Generate unique audio ID
            audio_id = self._generate_audio_id(request)
            
            # Determine storage path based on audio type
            if request.audio_type == AudioType.VOICEOVER:
                base_path = self.voiceovers_path
            elif request.audio_type == AudioType.NARRATION:
                base_path = self.narrations_path
            elif request.audio_type == AudioType.BACKGROUND_MUSIC:
                base_path = self.music_path
            else:
                base_path = self.audio_path
            
            # Create filename with format extension
            filename = f"{audio_id}.{request.specification.format.value}"
            file_path = base_path / filename
            
            # Write audio data to file
            await asyncio.to_thread(file_path.write_bytes, audio_data)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Generate file URL
            file_url = self._get_file_url(file_path)
            
            logger.info(f"Audio stored: {file_path} ({file_size} bytes)")
            
            return {
                "audio_id": audio_id,
                "file_path": str(file_path),
                "file_url": file_url,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"Failed to store audio: {e}")
            raise StorageError("write", str(e))
    
    def _generate_audio_id(self, request: AudioGenerationRequest) -> str:
        """Generate unique audio ID based on request parameters."""
        # Create hash from text/genre, timestamp, and audio type
        content = f"{request.text or request.music_genre}_{datetime.utcnow().isoformat()}_{request.audio_type}"
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]
    
    def _get_file_url(self, file_path: Path) -> str:
        """Generate URL for accessing the stored audio."""
        # Convert absolute path to relative URL
        relative_path = file_path.relative_to(self.storage_path)
        return f"/storage/{relative_path.as_posix()}"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """Get engine usage statistics."""
        return {
            "total_audio_generated": self.total_tokens_used // 100,  # Rough estimate
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cost_per_audio": self.cost_per_audio
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0

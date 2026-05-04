"""
Video Pipeline Engine for ContentFlow AI.

This module implements the Video Pipeline Engine responsible for orchestrating
short-form video creation with secure storage integration.
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


class VideoType(str, Enum):
    """Enumeration of video generation types."""
    SHORT_FORM = "short_form"
    EXPLAINER = "explainer"
    TUTORIAL = "tutorial"
    PROMOTIONAL = "promotional"
    SOCIAL_MEDIA = "social_media"
    DEMO = "demo"


class VideoFormat(str, Enum):
    """Enumeration of supported video formats."""
    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    AVI = "avi"


class VideoQuality(str, Enum):
    """Enumeration of video quality presets."""
    LOW = "low"          # 480p
    MEDIUM = "medium"    # 720p
    HIGH = "high"        # 1080p
    ULTRA = "ultra"      # 4K


class VideoStyle(str, Enum):
    """Enumeration of video styles."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    DYNAMIC = "dynamic"
    MINIMALIST = "minimalist"
    CINEMATIC = "cinematic"
    ANIMATED = "animated"
    DOCUMENTARY = "documentary"


class VideoSpecification(BaseModel):
    """Model for video generation specifications."""
    width: int = Field(ge=480, le=3840, description="Video width in pixels")
    height: int = Field(ge=360, le=2160, description="Video height in pixels")
    format: VideoFormat = VideoFormat.MP4
    quality: VideoQuality = VideoQuality.HIGH
    fps: int = Field(default=30, ge=24, le=60, description="Frames per second")
    duration_seconds: int = Field(ge=1, le=300, description="Target duration in seconds")
    bitrate_kbps: int = Field(default=5000, ge=1000, le=50000, description="Video bitrate in kbps")
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        """Validate video dimensions are even numbers."""
        if v % 2 != 0:
            v = v + 1  # Round up to even number
        return v
    
    def get_aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height
    
    def get_total_pixels(self) -> int:
        """Calculate total pixel count."""
        return self.width * self.height
    
    def get_estimated_file_size_mb(self) -> float:
        """Estimate file size in MB based on duration and bitrate."""
        # File size = (bitrate in kbps * duration in seconds) / (8 * 1024)
        return (self.bitrate_kbps * self.duration_seconds) / (8 * 1024)


class VideoGenerationRequest(BaseModel):
    """Model for video generation requests."""
    video_type: VideoType
    script: str = Field(min_length=10, max_length=10000, description="Video script or description")
    style: VideoStyle = VideoStyle.PROFESSIONAL
    specification: VideoSpecification
    include_audio: bool = Field(default=True, description="Include audio narration")
    include_music: bool = Field(default=False, description="Include background music")
    include_subtitles: bool = Field(default=False, description="Include subtitles")
    transitions: Optional[str] = Field(default="smooth", description="Transition style between scenes")
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('script')
    def validate_script(cls, v):
        """Validate script content."""
        if not v or not v.strip():
            raise ValueError("Script cannot be empty")
        return v.strip()



class GeneratedVideo(BaseModel):
    """Model for generated video metadata."""
    video_id: str
    video_type: VideoType
    file_path: str
    file_url: str
    specification: VideoSpecification
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    thumbnail_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
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


class VideoPipelineEngine:
    """
    Video Pipeline Engine for orchestrating short-form video creation.
    
    This engine handles:
    - Short-form video orchestration and coordination
    - Video generation with audio and visual elements
    - Video specification validation and processing
    - Secure storage integration for generated videos
    - Cost tracking and usage monitoring
    """
    
    def __init__(self):
        """Initialize the Video Pipeline Engine."""
        self.gemini_client = None
        self.storage_path = Path(settings.LOCAL_STORAGE_PATH)
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cost_per_video = 0.50  # $0.50 per video generation (example rate)
        self._initialize_storage()
        self._initialize_gemini()
    
    def _initialize_storage(self):
        """Initialize storage backend."""
        try:
            # Create storage directories if they don't exist
            self.video_path = self.storage_path / "videos"
            self.short_form_path = self.video_path / "short_form"
            self.explainers_path = self.video_path / "explainers"
            self.thumbnails_path = self.video_path / "thumbnails"
            
            for path in [self.video_path, self.short_form_path, self.explainers_path, self.thumbnails_path]:
                path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Video storage initialized at {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise StorageError("initialization", str(e))

    
    def _initialize_gemini(self):
        """Initialize Google Gemini client for video generation assistance."""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured. Video Pipeline Engine will be limited.")
                return
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            # Note: Gemini can help with script enhancement and scene planning
            # In production, you'd integrate with video generation APIs
            self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Video Pipeline Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise EngineError("video_pipeline", f"Initialization failed: {e}")
    
    async def generate_video(
        self, 
        request: VideoGenerationRequest
    ) -> GeneratedVideo:
        """
        Generate a video based on the provided request.
        
        Args:
            request: Video generation request with specifications
            
        Returns:
            GeneratedVideo with metadata and storage information
            
        Raises:
            ValidationError: If request parameters are invalid
            EngineError: If video generation fails
            StorageError: If storage operations fail
        """
        try:
            logger.info(f"Generating {request.video_type} video")
            
            # Validate request
            await self._validate_generation_request(request)
            
            # Enhance script using AI if available
            enhanced_script = await self._enhance_script(request)
            
            # Plan video scenes
            scene_plan = await self._plan_scenes(enhanced_script, request)
            
            # Generate video (simulated for now - would call actual video generation API)
            video_data, duration = await self._generate_video_data(scene_plan, request)
            
            # Generate thumbnail
            thumbnail_data = await self._generate_thumbnail(request)
            
            # Store video and thumbnail securely
            stored_video = await self._store_video(video_data, thumbnail_data, request)
            
            # Calculate cost
            tokens_used = self._estimate_tokens(request.script)
            cost = self.cost_per_video
            
            # Update usage tracking
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            
            # Create result
            result = GeneratedVideo(
                video_id=stored_video["video_id"],
                video_type=request.video_type,
                file_path=stored_video["file_path"],
                file_url=stored_video["file_url"],
                specification=request.specification,
                duration_seconds=duration,
                file_size_bytes=stored_video["file_size"],
                thumbnail_path=stored_video.get("thumbnail_path"),
                thumbnail_url=stored_video.get("thumbnail_url"),
                metadata={
                    "script": request.script,
                    "enhanced_script": enhanced_script,
                    "scene_plan": scene_plan,
                    "style": request.style,
                    "include_audio": request.include_audio,
                    "include_music": request.include_music,
                    "include_subtitles": request.include_subtitles,
                    "transitions": request.transitions,
                    "context": request.context
                },
                tokens_used=tokens_used,
                cost=cost
            )
            
            logger.info(f"Video generation completed: {result.video_id}")
            return result
            
        except ValidationError:
            raise
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise EngineError("video_pipeline", f"Video generation failed: {e}")

    
    async def generate_short_form_video(
        self,
        script: str,
        duration_seconds: int = 60,
        style: VideoStyle = VideoStyle.DYNAMIC,
        quality: VideoQuality = VideoQuality.HIGH
    ) -> GeneratedVideo:
        """
        Generate a short-form video.
        
        Args:
            script: Video script or description
            duration_seconds: Target duration in seconds
            style: Visual style for the video
            quality: Video quality preset
            
        Returns:
            GeneratedVideo with short-form video metadata
        """
        # Determine dimensions based on quality
        dimensions = self._get_quality_dimensions(quality)
        
        request = VideoGenerationRequest(
            video_type=VideoType.SHORT_FORM,
            script=script,
            style=style,
            specification=VideoSpecification(
                width=dimensions["width"],
                height=dimensions["height"],
                format=VideoFormat.MP4,
                quality=quality,
                fps=30,
                duration_seconds=duration_seconds,
                bitrate_kbps=5000
            ),
            include_audio=True,
            include_music=True
        )
        return await self.generate_video(request)
    
    async def generate_explainer_video(
        self,
        script: str,
        duration_seconds: int = 120,
        style: VideoStyle = VideoStyle.PROFESSIONAL,
        include_subtitles: bool = True
    ) -> GeneratedVideo:
        """
        Generate an explainer video.
        
        Args:
            script: Video script explaining a concept
            duration_seconds: Target duration in seconds
            style: Visual style for the video
            include_subtitles: Whether to include subtitles
            
        Returns:
            GeneratedVideo with explainer video metadata
        """
        request = VideoGenerationRequest(
            video_type=VideoType.EXPLAINER,
            script=script,
            style=style,
            specification=VideoSpecification(
                width=1920,
                height=1080,
                format=VideoFormat.MP4,
                quality=VideoQuality.HIGH,
                fps=30,
                duration_seconds=duration_seconds,
                bitrate_kbps=8000
            ),
            include_audio=True,
            include_music=False,
            include_subtitles=include_subtitles
        )
        return await self.generate_video(request)

    
    async def validate_video_specification(
        self,
        specification: VideoSpecification
    ) -> Dict[str, Any]:
        """
        Validate video specification and provide recommendations.
        
        Args:
            specification: Video specification to validate
            
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
        if total_pixels > 8_000_000:  # 4K threshold
            validation_result["warnings"].append(
                f"High resolution ({specification.width}x{specification.height}) may result in slow processing"
            )
        
        # Check aspect ratio
        aspect_ratio = specification.get_aspect_ratio()
        common_ratios = {
            16/9: "16:9 (widescreen)",
            9/16: "9:16 (vertical/mobile)",
            4/3: "4:3 (standard)",
            1/1: "1:1 (square)"
        }
        
        closest_ratio = min(common_ratios.keys(), key=lambda r: abs(r - aspect_ratio))
        if abs(closest_ratio - aspect_ratio) > 0.1:
            validation_result["recommendations"].append(
                f"Consider using a standard aspect ratio like {common_ratios[closest_ratio]}"
            )
        
        # Check duration and file size
        estimated_size = specification.get_estimated_file_size_mb()
        if estimated_size > 500:  # 500 MB
            validation_result["warnings"].append(
                f"Large file size (~{estimated_size:.1f} MB) may cause upload/streaming issues"
            )
        
        # Check bitrate
        if specification.bitrate_kbps < 2000:
            validation_result["warnings"].append(
                "Low bitrate may result in visible compression artifacts"
            )
        
        if specification.bitrate_kbps > 20000:
            validation_result["recommendations"].append(
                "High bitrate may not provide noticeable quality improvement"
            )
        
        # Check FPS
        if specification.fps < 24:
            validation_result["warnings"].append(
                "Frame rate below 24 fps may appear choppy"
            )
        
        if specification.fps > 30 and specification.quality in [VideoQuality.LOW, VideoQuality.MEDIUM]:
            validation_result["recommendations"].append(
                "High frame rate with lower quality may not provide optimal results"
            )
        
        return validation_result
    
    async def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve video metadata by ID.
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            Dictionary with video metadata or None if not found
        """
        try:
            # Search for video in storage directories
            for base_path in [self.short_form_path, self.explainers_path, self.video_path]:
                for file_path in base_path.glob(f"{video_id}.*"):
                    if file_path.is_file():
                        # Look for thumbnail
                        thumbnail_path = self.thumbnails_path / f"{video_id}.jpg"
                        
                        return {
                            "video_id": video_id,
                            "file_path": str(file_path),
                            "file_url": self._get_file_url(file_path),
                            "file_size": file_path.stat().st_size,
                            "format": file_path.suffix[1:],
                            "thumbnail_path": str(thumbnail_path) if thumbnail_path.exists() else None,
                            "thumbnail_url": self._get_file_url(thumbnail_path) if thumbnail_path.exists() else None,
                            "exists": True
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve video {video_id}: {e}")
            raise StorageError("retrieval", str(e))

    
    async def delete_video(self, video_id: str) -> bool:
        """
        Delete a video from storage.
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            video_info = await self.get_video(video_id)
            if not video_info:
                return False
            
            # Delete video file
            file_path = Path(video_info["file_path"])
            file_path.unlink()
            
            # Delete thumbnail if exists
            if video_info.get("thumbnail_path"):
                thumbnail_path = Path(video_info["thumbnail_path"])
                if thumbnail_path.exists():
                    thumbnail_path.unlink()
            
            logger.info(f"Deleted video {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video {video_id}: {e}")
            raise StorageError("deletion", str(e))
    
    # Private helper methods
    
    async def _validate_generation_request(self, request: VideoGenerationRequest):
        """Validate video generation request."""
        # Validate specification
        spec_validation = await self.validate_video_specification(request.specification)
        
        if not spec_validation["is_valid"]:
            raise ValidationError(
                "Invalid video specification",
                details={"errors": spec_validation["errors"]}
            )
        
        # Check estimated file size
        estimated_size = request.specification.get_estimated_file_size_mb()
        if estimated_size > 1000:  # 1 GB limit
            raise ValidationError(
                f"Estimated file size ({estimated_size:.2f} MB) exceeds limit"
            )
        
        # Check duration
        if request.specification.duration_seconds > 300:  # 5 minutes
            raise ValidationError(
                f"Duration ({request.specification.duration_seconds}s) exceeds maximum of 300 seconds"
            )
    
    async def _enhance_script(self, request: VideoGenerationRequest) -> str:
        """Enhance the video script using AI."""
        if not self.gemini_client:
            return request.script
        
        try:
            enhancement_prompt = f"""
            Enhance this video script for {request.video_type.value} video generation:
            
            Original script: {request.script}
            Video style: {request.style.value}
            Duration: {request.specification.duration_seconds} seconds
            
            Provide an enhanced version that:
            - Improves clarity and engagement
            - Adds scene descriptions and visual cues
            - Optimizes pacing for the target duration
            - Maintains the original message
            
            Return only the enhanced script, no explanations.
            """
            
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                enhancement_prompt
            )
            
            if response and response.text:
                enhanced = response.text.strip()
                logger.info("Script enhanced successfully")
                return enhanced
            
            return request.script
            
        except Exception as e:
            logger.warning(f"Script enhancement failed, using original: {e}")
            return request.script

    
    async def _plan_scenes(self, script: str, request: VideoGenerationRequest) -> List[Dict[str, Any]]:
        """Plan video scenes based on script."""
        if not self.gemini_client:
            # Return basic scene plan
            return [{
                "scene_number": 1,
                "duration": request.specification.duration_seconds,
                "description": script[:200],
                "visual_elements": ["text", "graphics"],
                "audio": "narration" if request.include_audio else "none"
            }]
        
        try:
            planning_prompt = f"""
            Create a scene-by-scene plan for this video:
            
            Script: {script}
            Total duration: {request.specification.duration_seconds} seconds
            Style: {request.style.value}
            
            Provide a JSON array of scenes with:
            - scene_number
            - duration (in seconds)
            - description
            - visual_elements (array)
            - audio (narration/music/none)
            
            Return only the JSON array, no explanations.
            """
            
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                planning_prompt
            )
            
            if response and response.text:
                # Parse JSON response (simplified - in production, use proper JSON parsing)
                logger.info("Scene plan created successfully")
                # For now, return a basic plan
                return [{
                    "scene_number": 1,
                    "duration": request.specification.duration_seconds,
                    "description": script[:200],
                    "visual_elements": ["text", "graphics", "animations"],
                    "audio": "narration" if request.include_audio else "none"
                }]
            
            return [{
                "scene_number": 1,
                "duration": request.specification.duration_seconds,
                "description": script[:200],
                "visual_elements": ["text", "graphics"],
                "audio": "narration" if request.include_audio else "none"
            }]
            
        except Exception as e:
            logger.warning(f"Scene planning failed, using basic plan: {e}")
            return [{
                "scene_number": 1,
                "duration": request.specification.duration_seconds,
                "description": script[:200],
                "visual_elements": ["text", "graphics"],
                "audio": "narration" if request.include_audio else "none"
            }]
    
    async def _generate_video_data(
        self,
        scene_plan: List[Dict[str, Any]],
        request: VideoGenerationRequest
    ) -> tuple[bytes, float]:
        """
        Generate video data using AI service.
        
        Note: This is a placeholder implementation. In production, you would
        integrate with actual video generation APIs like:
        - Runway ML
        - Synthesia
        - D-ID
        - Pictory
        - Custom video rendering pipeline
        
        Returns:
            Tuple of (video_data, duration_seconds)
        """
        logger.info(f"Generating {request.video_type.value} video with {len(scene_plan)} scenes...")
        
        # Simulate API call delay
        await asyncio.sleep(1.0)
        
        # Create a minimal MP4 file structure (placeholder)
        # In production, this would be the actual generated video data
        duration = float(request.specification.duration_seconds)
        
        # Minimal MP4 header (simplified placeholder)
        placeholder_mp4 = bytearray()
        placeholder_mp4.extend(b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00')
        placeholder_mp4.extend(b'isomiso2mp41\x00\x00\x00\x08free')
        placeholder_mp4.extend(b'\x00\x00\x00\x00mdat')
        # Add some placeholder data
        placeholder_mp4.extend(b'\x00' * 1024)
        
        return bytes(placeholder_mp4), duration

    
    async def _generate_thumbnail(self, request: VideoGenerationRequest) -> bytes:
        """Generate thumbnail for the video."""
        # Create a simple placeholder thumbnail (1x1 pixel PNG)
        # In production, this would extract a frame from the video or generate a custom thumbnail
        placeholder_png = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
            b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        return placeholder_png
    
    async def _store_video(
        self,
        video_data: bytes,
        thumbnail_data: bytes,
        request: VideoGenerationRequest
    ) -> Dict[str, Any]:
        """
        Store generated video and thumbnail securely.
        
        Args:
            video_data: Raw video bytes
            thumbnail_data: Raw thumbnail image bytes
            request: Original generation request
            
        Returns:
            Dictionary with storage information
        """
        try:
            # Generate unique video ID
            video_id = self._generate_video_id(request)
            
            # Determine storage path based on video type
            if request.video_type == VideoType.SHORT_FORM:
                base_path = self.short_form_path
            elif request.video_type == VideoType.EXPLAINER:
                base_path = self.explainers_path
            else:
                base_path = self.video_path
            
            # Create filename with format extension
            video_filename = f"{video_id}.{request.specification.format.value}"
            video_file_path = base_path / video_filename
            
            # Write video data to file
            await asyncio.to_thread(video_file_path.write_bytes, video_data)
            
            # Get file size
            file_size = video_file_path.stat().st_size
            
            # Generate file URL
            file_url = self._get_file_url(video_file_path)
            
            # Store thumbnail
            thumbnail_filename = f"{video_id}.jpg"
            thumbnail_file_path = self.thumbnails_path / thumbnail_filename
            await asyncio.to_thread(thumbnail_file_path.write_bytes, thumbnail_data)
            
            # Generate thumbnail URL
            thumbnail_url = self._get_file_url(thumbnail_file_path)
            
            logger.info(f"Video stored: {video_file_path} ({file_size} bytes)")
            
            return {
                "video_id": video_id,
                "file_path": str(video_file_path),
                "file_url": file_url,
                "file_size": file_size,
                "thumbnail_path": str(thumbnail_file_path),
                "thumbnail_url": thumbnail_url
            }
            
        except Exception as e:
            logger.error(f"Failed to store video: {e}")
            raise StorageError("write", str(e))
    
    def _generate_video_id(self, request: VideoGenerationRequest) -> str:
        """Generate unique video ID based on request parameters."""
        # Create hash from script, timestamp, and video type
        content = f"{request.script}_{datetime.utcnow().isoformat()}_{request.video_type}"
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]
    
    def _get_file_url(self, file_path: Path) -> str:
        """Generate URL for accessing the stored video."""
        # Convert absolute path to relative URL
        relative_path = file_path.relative_to(self.storage_path)
        return f"/storage/{relative_path.as_posix()}"
    
    def _get_quality_dimensions(self, quality: VideoQuality) -> Dict[str, int]:
        """Get video dimensions based on quality preset."""
        quality_map = {
            VideoQuality.LOW: {"width": 854, "height": 480},
            VideoQuality.MEDIUM: {"width": 1280, "height": 720},
            VideoQuality.HIGH: {"width": 1920, "height": 1080},
            VideoQuality.ULTRA: {"width": 3840, "height": 2160}
        }
        return quality_map.get(quality, quality_map[VideoQuality.HIGH])
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """Get engine usage statistics."""
        return {
            "total_videos_generated": self.total_tokens_used // 500,  # Rough estimate
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cost_per_video": self.cost_per_video
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0

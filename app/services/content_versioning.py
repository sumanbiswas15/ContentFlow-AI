"""
Content versioning and storage service for ContentFlow AI.

This module implements version tracking, history management, secure storage,
and data integrity validation for content items.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.core.database import get_database
from app.core.exceptions import (
    StorageError, ValidationError, NotFoundError
)
from app.models.content import ContentItem, ContentVersion
from app.models.base import PyObjectId

logger = logging.getLogger(__name__)


class ContentVersioningService:
    """
    Service for managing content versions, history, and secure storage.
    
    This service provides:
    - Version tracking with timestamps
    - Version history management and retrieval
    - Secure object storage integration
    - Data integrity validation and backup systems
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        self.database = database if database is not None else get_database()
        self.storage_backend = settings.STORAGE_BACKEND
        self.local_storage_path = Path(settings.LOCAL_STORAGE_PATH)
        
        # Ensure local storage directory exists
        if self.storage_backend == "local":
            self.local_storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local storage initialized at: {self.local_storage_path}")
    
    async def create_version(
        self,
        content_id: str,
        changes_summary: str,
        user_id: str,
        diff_data: Optional[Dict[str, Any]] = None
    ) -> ContentVersion:
        """
        Create a new version record for content.
        
        Args:
            content_id: ID of the content being versioned
            changes_summary: Description of changes in this version
            user_id: ID of user creating the version
            diff_data: Optional detailed diff information
        
        Returns:
            ContentVersion object
        
        Raises:
            NotFoundError: If content doesn't exist
            StorageError: If version creation fails
        """
        try:
            # Get current content
            content_doc = await self.database.content_items.find_one(
                {"_id": PyObjectId(content_id)}
            )
            
            if not content_doc:
                raise NotFoundError("Content", content_id)
            
            # Extract version directly from document (avoid Pydantic validation issues)
            current_version = content_doc.get("version", 1)
            
            # Create version record
            version = ContentVersion(
                version_number=current_version,
                content_id=content_id,
                changes_summary=changes_summary,
                created_by=user_id,
                created_at=datetime.utcnow(),
                diff_data=diff_data
            )
            
            # Store version in database
            version_doc = version.model_dump()
            await self.database.content_versions.insert_one(version_doc)
            
            logger.info(
                f"Created version {version.version_number} for content {content_id}"
            )
            
            return version
            
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to create version for content {content_id}: {e}")
            raise StorageError("version creation", str(e))
    
    async def get_version_history(
        self,
        content_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[ContentVersion]:
        """
        Get version history for content in chronological order.
        
        Args:
            content_id: ID of the content
            limit: Maximum number of versions to return
            skip: Number of versions to skip (for pagination)
        
        Returns:
            List of ContentVersion objects, newest first
        
        Raises:
            NotFoundError: If content doesn't exist
        """
        try:
            # Verify content exists
            content_doc = await self.database.content_items.find_one(
                {"_id": PyObjectId(content_id)}
            )
            
            if not content_doc:
                raise NotFoundError("Content", content_id)
            
            # Get version history
            cursor = self.database.content_versions.find(
                {"content_id": content_id}
            ).sort("created_at", -1).skip(skip).limit(limit)
            
            versions = []
            async for version_doc in cursor:
                versions.append(ContentVersion(**version_doc))
            
            logger.info(
                f"Retrieved {len(versions)} versions for content {content_id}"
            )
            
            return versions
            
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get version history for {content_id}: {e}")
            raise StorageError("version history retrieval", str(e))
    
    async def get_version(
        self,
        content_id: str,
        version_number: int
    ) -> Optional[ContentVersion]:
        """
        Get a specific version of content.
        
        Args:
            content_id: ID of the content
            version_number: Version number to retrieve
        
        Returns:
            ContentVersion object or None if not found
        """
        try:
            version_doc = await self.database.content_versions.find_one({
                "content_id": content_id,
                "version_number": version_number
            })
            
            if version_doc:
                return ContentVersion(**version_doc)
            
            return None
            
        except Exception as e:
            logger.error(
                f"Failed to get version {version_number} for {content_id}: {e}"
            )
            raise StorageError("version retrieval", str(e))
    
    async def increment_version(
        self,
        content_id: str,
        changes_summary: str,
        user_id: str
    ) -> int:
        """
        Increment content version and create version record.
        
        Args:
            content_id: ID of the content
            changes_summary: Description of changes
            user_id: ID of user making changes
        
        Returns:
            New version number
        
        Raises:
            NotFoundError: If content doesn't exist
            StorageError: If version increment fails
        """
        try:
            # Get current content
            content_doc = await self.database.content_items.find_one(
                {"_id": PyObjectId(content_id)}
            )
            
            if not content_doc:
                raise NotFoundError("Content", content_id)
            
            old_version = content_doc.get("version", 1)
            new_version = old_version + 1
            
            # Create version record for old version
            await self.create_version(
                content_id=content_id,
                changes_summary=changes_summary,
                user_id=user_id
            )
            
            # Update content version
            await self.database.content_items.update_one(
                {"_id": PyObjectId(content_id)},
                {
                    "$set": {
                        "version": new_version,
                        "updated_at": datetime.utcnow(),
                        "updated_by": user_id
                    }
                }
            )
            
            logger.info(
                f"Incremented content {content_id} version from {old_version} to {new_version}"
            )
            
            return new_version
            
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to increment version for {content_id}: {e}")
            raise StorageError("version increment", str(e))
    
    async def store_media_asset(
        self,
        content_id: str,
        asset_data: bytes,
        asset_type: str,
        filename: str
    ) -> str:
        """
        Store media asset securely with integrity validation.
        
        Args:
            content_id: ID of the content this asset belongs to
            asset_data: Binary data of the asset
            asset_type: Type of asset (image, audio, video)
            filename: Original filename
        
        Returns:
            Storage path/URL of the stored asset
        
        Raises:
            ValidationError: If asset data is invalid
            StorageError: If storage operation fails
        """
        try:
            # Validate asset data
            if not asset_data:
                raise ValidationError("Asset data cannot be empty")
            
            # Calculate checksum for integrity
            checksum = self._calculate_checksum(asset_data)
            
            # Generate storage path
            storage_path = self._generate_storage_path(
                content_id, asset_type, filename, checksum
            )
            
            # Store based on backend
            if self.storage_backend == "local":
                stored_path = await self._store_local(storage_path, asset_data)
            elif self.storage_backend == "s3":
                stored_path = await self._store_s3(storage_path, asset_data)
            elif self.storage_backend == "gcs":
                stored_path = await self._store_gcs(storage_path, asset_data)
            else:
                raise StorageError(
                    "asset storage",
                    f"Unsupported storage backend: {self.storage_backend}"
                )
            
            # Store metadata in database
            await self._store_asset_metadata(
                content_id=content_id,
                storage_path=stored_path,
                asset_type=asset_type,
                filename=filename,
                checksum=checksum,
                size_bytes=len(asset_data)
            )
            
            logger.info(
                f"Stored {asset_type} asset for content {content_id} at {stored_path}"
            )
            
            return stored_path
            
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Failed to store media asset: {e}")
            raise StorageError("asset storage", str(e))
    
    async def retrieve_media_asset(
        self,
        storage_path: str,
        validate_integrity: bool = True
    ) -> bytes:
        """
        Retrieve media asset with optional integrity validation.
        
        Args:
            storage_path: Path/URL of the stored asset
            validate_integrity: Whether to validate checksum
        
        Returns:
            Binary data of the asset
        
        Raises:
            NotFoundError: If asset doesn't exist
            StorageError: If retrieval fails or integrity check fails
        """
        try:
            # Retrieve based on backend
            if self.storage_backend == "local":
                asset_data = await self._retrieve_local(storage_path)
            elif self.storage_backend == "s3":
                asset_data = await self._retrieve_s3(storage_path)
            elif self.storage_backend == "gcs":
                asset_data = await self._retrieve_gcs(storage_path)
            else:
                raise StorageError(
                    "asset retrieval",
                    f"Unsupported storage backend: {self.storage_backend}"
                )
            
            # Validate integrity if requested
            if validate_integrity:
                metadata = await self._get_asset_metadata(storage_path)
                if metadata:
                    expected_checksum = metadata.get("checksum")
                    actual_checksum = self._calculate_checksum(asset_data)
                    
                    if expected_checksum != actual_checksum:
                        raise StorageError(
                            "asset retrieval",
                            f"Integrity check failed: checksum mismatch"
                        )
            
            logger.info(f"Retrieved asset from {storage_path}")
            
            return asset_data
            
        except NotFoundError:
            raise
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve media asset: {e}")
            raise StorageError("asset retrieval", str(e))
    
    async def validate_data_integrity(
        self,
        content_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate data integrity for content and its assets.
        
        Args:
            content_id: ID of the content to validate
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        try:
            # Get content
            content_doc = await self.database.content_items.find_one(
                {"_id": PyObjectId(content_id)}
            )
            
            if not content_doc:
                errors.append(f"Content {content_id} not found")
                return False, errors
            
            # Validate content structure (access fields directly)
            if not content_doc.get("title") or not str(content_doc.get("title")).strip():
                errors.append("Content title is empty")
            
            version = content_doc.get("version", 1)
            if version < 1:
                errors.append(f"Invalid version number: {version}")
            
            # Validate version history consistency
            versions = await self.get_version_history(content_id, limit=1000)
            version_numbers = [v.version_number for v in versions]
            
            # Check for gaps in version history
            if version_numbers:
                expected_versions = set(range(1, version))
                actual_versions = set(version_numbers)
                missing_versions = expected_versions - actual_versions
                
                if missing_versions:
                    errors.append(
                        f"Missing version records: {sorted(missing_versions)}"
                    )
            
            # Validate associated media assets
            asset_docs = await self.database.media_assets.find(
                {"content_id": content_id}
            ).to_list(length=None)
            
            for asset_doc in asset_docs:
                storage_path = asset_doc.get("storage_path")
                checksum = asset_doc.get("checksum")
                
                # Check if asset exists
                try:
                    asset_data = await self.retrieve_media_asset(
                        storage_path, validate_integrity=False
                    )
                    
                    # Validate checksum
                    actual_checksum = self._calculate_checksum(asset_data)
                    if checksum != actual_checksum:
                        errors.append(
                            f"Checksum mismatch for asset {storage_path}"
                        )
                        
                except Exception as e:
                    errors.append(f"Failed to retrieve asset {storage_path}: {e}")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info(f"Data integrity validated for content {content_id}")
            else:
                logger.warning(
                    f"Data integrity issues found for content {content_id}: {errors}"
                )
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Failed to validate data integrity: {e}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    async def create_backup(
        self,
        content_id: str,
        backup_location: Optional[str] = None
    ) -> str:
        """
        Create a backup of content and all its versions.
        
        Args:
            content_id: ID of the content to backup
            backup_location: Optional custom backup location
        
        Returns:
            Path to the backup
        
        Raises:
            NotFoundError: If content doesn't exist
            StorageError: If backup creation fails
        """
        try:
            # Get content
            content_doc = await self.database.content_items.find_one(
                {"_id": PyObjectId(content_id)}
            )
            
            if not content_doc:
                raise NotFoundError("Content", content_id)
            
            # Get version history
            versions = await self.get_version_history(content_id, limit=1000)
            
            # Get associated assets
            asset_docs = await self.database.media_assets.find(
                {"content_id": content_id}
            ).to_list(length=None)
            
            # Create backup data structure
            backup_data = {
                "content": content_doc,
                "versions": [v.dict() for v in versions],
                "assets": asset_docs,
                "backup_timestamp": datetime.utcnow().isoformat(),
                "backup_version": "1.0"
            }
            
            # Serialize backup data
            backup_json = json.dumps(backup_data, default=str, indent=2)
            
            # Generate backup path
            if not backup_location:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_location = f"backups/{content_id}_{timestamp}.json"
            
            # Store backup
            if self.storage_backend == "local":
                backup_path = self.local_storage_path / backup_location
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(backup_json)
                stored_path = str(backup_path)
            else:
                # For cloud storage, store as regular file
                stored_path = await self.store_media_asset(
                    content_id=content_id,
                    asset_data=backup_json.encode('utf-8'),
                    asset_type="backup",
                    filename=backup_location
                )
            
            # Record backup in database
            await self.database.content_backups.insert_one({
                "content_id": content_id,
                "backup_path": stored_path,
                "created_at": datetime.utcnow(),
                "size_bytes": len(backup_json),
                "version_count": len(versions),
                "asset_count": len(asset_docs)
            })
            
            logger.info(f"Created backup for content {content_id} at {stored_path}")
            
            return stored_path
            
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to create backup for {content_id}: {e}")
            raise StorageError("backup creation", str(e))
    
    async def restore_from_backup(
        self,
        backup_path: str,
        restore_versions: bool = True,
        restore_assets: bool = True
    ) -> str:
        """
        Restore content from a backup.
        
        Args:
            backup_path: Path to the backup file
            restore_versions: Whether to restore version history
            restore_assets: Whether to restore media assets
        
        Returns:
            ID of the restored content
        
        Raises:
            NotFoundError: If backup doesn't exist
            StorageError: If restore fails
        """
        try:
            # Retrieve backup data
            if self.storage_backend == "local":
                backup_file = Path(backup_path)
                if not backup_file.exists():
                    raise NotFoundError("Backup", backup_path)
                backup_json = backup_file.read_text()
            else:
                backup_data = await self.retrieve_media_asset(backup_path)
                backup_json = backup_data.decode('utf-8')
            
            # Parse backup data
            backup = json.loads(backup_json)
            
            # Restore content (remove _id to let MongoDB generate a new one)
            content_doc = backup["content"]
            if "_id" in content_doc:
                del content_doc["_id"]
            
            result = await self.database.content_items.insert_one(content_doc)
            content_id = str(result.inserted_id)
            
            # Restore versions if requested
            if restore_versions and "versions" in backup:
                for version_data in backup["versions"]:
                    await self.database.content_versions.insert_one(version_data)
            
            # Restore assets if requested
            if restore_assets and "assets" in backup:
                for asset_data in backup["assets"]:
                    await self.database.media_assets.insert_one(asset_data)
            
            logger.info(f"Restored content from backup {backup_path}")
            
            return content_id
            
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_path}: {e}")
            raise StorageError("backup restore", str(e))
    
    # Private helper methods
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()
    
    def _generate_storage_path(
        self,
        content_id: str,
        asset_type: str,
        filename: str,
        checksum: str
    ) -> str:
        """Generate storage path for an asset."""
        # Use first 2 chars of checksum for sharding
        shard = checksum[:2]
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        
        # Sanitize filename
        safe_filename = "".join(
            c for c in filename if c.isalnum() or c in "._-"
        )
        
        return f"{asset_type}/{shard}/{timestamp}/{content_id}_{checksum[:8]}_{safe_filename}"
    
    async def _store_local(self, storage_path: str, data: bytes) -> str:
        """Store asset in local filesystem."""
        full_path = self.local_storage_path / storage_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        return str(storage_path)
    
    async def _retrieve_local(self, storage_path: str) -> bytes:
        """Retrieve asset from local filesystem."""
        full_path = self.local_storage_path / storage_path
        if not full_path.exists():
            raise NotFoundError("Asset", storage_path)
        return full_path.read_bytes()
    
    async def _store_s3(self, storage_path: str, data: bytes) -> str:
        """Store asset in AWS S3."""
        # TODO: Implement S3 storage
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def _retrieve_s3(self, storage_path: str) -> bytes:
        """Retrieve asset from AWS S3."""
        # TODO: Implement S3 retrieval
        raise NotImplementedError("S3 retrieval not yet implemented")
    
    async def _store_gcs(self, storage_path: str, data: bytes) -> str:
        """Store asset in Google Cloud Storage."""
        # TODO: Implement GCS storage
        raise NotImplementedError("GCS storage not yet implemented")
    
    async def _retrieve_gcs(self, storage_path: str) -> bytes:
        """Retrieve asset from Google Cloud Storage."""
        # TODO: Implement GCS retrieval
        raise NotImplementedError("GCS retrieval not yet implemented")
    
    async def _store_asset_metadata(
        self,
        content_id: str,
        storage_path: str,
        asset_type: str,
        filename: str,
        checksum: str,
        size_bytes: int
    ):
        """Store asset metadata in database."""
        await self.database.media_assets.insert_one({
            "content_id": content_id,
            "storage_path": storage_path,
            "asset_type": asset_type,
            "filename": filename,
            "checksum": checksum,
            "size_bytes": size_bytes,
            "created_at": datetime.utcnow()
        })
    
    async def _get_asset_metadata(self, storage_path: str) -> Optional[Dict[str, Any]]:
        """Get asset metadata from database."""
        return await self.database.media_assets.find_one({
            "storage_path": storage_path
        })


# Global service instance (lazy-loaded)
_versioning_service = None


def get_versioning_service() -> ContentVersioningService:
    """Get or create the global versioning service instance."""
    global _versioning_service
    if _versioning_service is None:
        _versioning_service = ContentVersioningService()
    return _versioning_service

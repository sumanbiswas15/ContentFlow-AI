"""
Unit tests for content versioning service.

Tests cover version tracking, history management, secure storage,
and data integrity validation.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from app.services.content_versioning import ContentVersioningService, get_versioning_service
from app.models.content import ContentItem, ContentVersion, ContentMetadata
from app.models.base import ContentType, WorkflowState, PyObjectId
from app.core.exceptions import StorageError, ValidationError, NotFoundError
from app.core.config import settings


@pytest_asyncio.fixture
async def versioning_service(test_db):
    """Create a versioning service with test database."""
    service = ContentVersioningService(database=test_db)
    
    # Use temporary directory for local storage
    temp_dir = tempfile.mkdtemp()
    service.local_storage_path = Path(temp_dir)
    service.storage_backend = "local"
    
    yield service
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def sample_content(test_db):
    """Create sample content for testing."""
    content = ContentItem(
        type=ContentType.TEXT,
        title="Test Content",
        content="This is test content for versioning.",
        content_metadata=ContentMetadata(author="test_user"),
        workflow_state=WorkflowState.CREATE,
        version=1,
        user_id="test_user_123",
        tags=["test", "versioning"]
    )
    
    # Insert into database
    content_dict = content.dict()
    result = await test_db.content_items.insert_one(content_dict)
    content.id = result.inserted_id
    
    return content


class TestVersionCreation:
    """Tests for version creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_version_success(self, versioning_service, sample_content):
        """Test successful version creation."""
        content_id = str(sample_content.id)
        
        version = await versioning_service.create_version(
            content_id=content_id,
            changes_summary="Initial version",
            user_id="test_user_123"
        )
        
        assert version is not None
        assert version.version_number == 1
        assert version.content_id == content_id
        assert version.changes_summary == "Initial version"
        assert version.created_by == "test_user_123"
        assert isinstance(version.created_at, datetime)
    
    @pytest.mark.asyncio
    async def test_create_version_with_diff_data(self, versioning_service, sample_content):
        """Test version creation with diff data."""
        content_id = str(sample_content.id)
        diff_data = {
            "added_lines": ["New line 1", "New line 2"],
            "removed_lines": ["Old line 1"],
            "modified_lines": [{"old": "Old text", "new": "New text"}]
        }
        
        version = await versioning_service.create_version(
            content_id=content_id,
            changes_summary="Added new content",
            user_id="test_user_123",
            diff_data=diff_data
        )
        
        assert version.diff_data == diff_data
    
    @pytest.mark.asyncio
    async def test_create_version_nonexistent_content(self, versioning_service):
        """Test version creation for nonexistent content."""
        with pytest.raises(NotFoundError) as exc_info:
            await versioning_service.create_version(
                content_id="507f1f77bcf86cd799439011",  # Valid ObjectId format
                changes_summary="Test",
                user_id="test_user"
            )
        
        assert "not found" in str(exc_info.value).lower()


class TestVersionHistory:
    """Tests for version history management."""
    
    @pytest.mark.asyncio
    async def test_get_version_history_empty(self, versioning_service, sample_content):
        """Test getting version history when no versions exist."""
        content_id = str(sample_content.id)
        
        versions = await versioning_service.get_version_history(content_id)
        
        assert versions == []
    
    @pytest.mark.asyncio
    async def test_get_version_history_multiple_versions(
        self, versioning_service, sample_content
    ):
        """Test getting version history with multiple versions."""
        content_id = str(sample_content.id)
        
        # Create multiple versions
        for i in range(5):
            await versioning_service.create_version(
                content_id=content_id,
                changes_summary=f"Version {i+1}",
                user_id="test_user_123"
            )
        
        versions = await versioning_service.get_version_history(content_id)
        
        assert len(versions) == 5
        # Should be in reverse chronological order (newest first)
        assert versions[0].changes_summary == "Version 5"
        assert versions[4].changes_summary == "Version 1"
    
    @pytest.mark.asyncio
    async def test_get_version_history_with_pagination(
        self, versioning_service, sample_content
    ):
        """Test version history pagination."""
        content_id = str(sample_content.id)
        
        # Create 10 versions
        for i in range(10):
            await versioning_service.create_version(
                content_id=content_id,
                changes_summary=f"Version {i+1}",
                user_id="test_user_123"
            )
        
        # Get first page
        page1 = await versioning_service.get_version_history(
            content_id, limit=5, skip=0
        )
        assert len(page1) == 5
        
        # Get second page
        page2 = await versioning_service.get_version_history(
            content_id, limit=5, skip=5
        )
        assert len(page2) == 5
        
        # Ensure no overlap
        page1_summaries = {v.changes_summary for v in page1}
        page2_summaries = {v.changes_summary for v in page2}
        assert len(page1_summaries & page2_summaries) == 0
    
    @pytest.mark.asyncio
    async def test_get_specific_version(self, versioning_service, sample_content):
        """Test retrieving a specific version."""
        content_id = str(sample_content.id)
        
        # Create versions
        await versioning_service.create_version(
            content_id=content_id,
            changes_summary="Version 1",
            user_id="test_user_123"
        )
        
        # Get specific version
        version = await versioning_service.get_version(content_id, version_number=1)
        
        assert version is not None
        assert version.version_number == 1
        assert version.changes_summary == "Version 1"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_version(self, versioning_service, sample_content):
        """Test retrieving a version that doesn't exist."""
        content_id = str(sample_content.id)
        
        version = await versioning_service.get_version(content_id, version_number=999)
        
        assert version is None


class TestVersionIncrement:
    """Tests for version increment functionality."""
    
    @pytest.mark.asyncio
    async def test_increment_version_success(self, versioning_service, sample_content):
        """Test successful version increment."""
        content_id = str(sample_content.id)
        initial_version = sample_content.version
        
        new_version = await versioning_service.increment_version(
            content_id=content_id,
            changes_summary="Updated content",
            user_id="test_user_123"
        )
        
        assert new_version == initial_version + 1
        
        # Verify version was updated in database
        updated_content = await versioning_service.database.content_items.find_one(
            {"_id": PyObjectId(content_id)}
        )
        assert updated_content["version"] == new_version
    
    @pytest.mark.asyncio
    async def test_increment_version_creates_history(
        self, versioning_service, sample_content
    ):
        """Test that incrementing version creates history record."""
        content_id = str(sample_content.id)
        
        await versioning_service.increment_version(
            content_id=content_id,
            changes_summary="First update",
            user_id="test_user_123"
        )
        
        # Check version history was created
        versions = await versioning_service.get_version_history(content_id)
        assert len(versions) == 1
        assert versions[0].changes_summary == "First update"
    
    @pytest.mark.asyncio
    async def test_increment_version_multiple_times(
        self, versioning_service, sample_content
    ):
        """Test incrementing version multiple times."""
        content_id = str(sample_content.id)
        
        # Increment 3 times
        for i in range(3):
            new_version = await versioning_service.increment_version(
                content_id=content_id,
                changes_summary=f"Update {i+1}",
                user_id="test_user_123"
            )
            assert new_version == i + 2  # Starting from version 1
        
        # Verify final version
        updated_content = await versioning_service.database.content_items.find_one(
            {"_id": PyObjectId(content_id)}
        )
        assert updated_content["version"] == 4


class TestMediaAssetStorage:
    """Tests for media asset storage functionality."""
    
    @pytest.mark.asyncio
    async def test_store_media_asset_success(self, versioning_service, sample_content):
        """Test successful media asset storage."""
        content_id = str(sample_content.id)
        asset_data = b"This is test image data"
        
        storage_path = await versioning_service.store_media_asset(
            content_id=content_id,
            asset_data=asset_data,
            asset_type="image",
            filename="test_image.jpg"
        )
        
        assert storage_path is not None
        assert "image" in storage_path
        assert "test_image.jpg" in storage_path
    
    @pytest.mark.asyncio
    async def test_store_empty_asset_fails(self, versioning_service, sample_content):
        """Test that storing empty asset fails."""
        content_id = str(sample_content.id)
        
        with pytest.raises(ValidationError) as exc_info:
            await versioning_service.store_media_asset(
                content_id=content_id,
                asset_data=b"",
                asset_type="image",
                filename="empty.jpg"
            )
        
        assert "empty" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_retrieve_media_asset_success(
        self, versioning_service, sample_content
    ):
        """Test successful media asset retrieval."""
        content_id = str(sample_content.id)
        original_data = b"Test video data content"
        
        # Store asset
        storage_path = await versioning_service.store_media_asset(
            content_id=content_id,
            asset_data=original_data,
            asset_type="video",
            filename="test_video.mp4"
        )
        
        # Retrieve asset
        retrieved_data = await versioning_service.retrieve_media_asset(storage_path)
        
        assert retrieved_data == original_data
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_asset_fails(self, versioning_service):
        """Test that retrieving nonexistent asset fails."""
        with pytest.raises(NotFoundError):
            await versioning_service.retrieve_media_asset("nonexistent/path.jpg")
    
    @pytest.mark.asyncio
    async def test_asset_integrity_validation(self, versioning_service, sample_content):
        """Test asset integrity validation with checksum."""
        content_id = str(sample_content.id)
        original_data = b"Important audio data"
        
        # Store asset
        storage_path = await versioning_service.store_media_asset(
            content_id=content_id,
            asset_data=original_data,
            asset_type="audio",
            filename="test_audio.mp3"
        )
        
        # Retrieve with integrity check
        retrieved_data = await versioning_service.retrieve_media_asset(
            storage_path, validate_integrity=True
        )
        
        assert retrieved_data == original_data
    
    @pytest.mark.asyncio
    async def test_checksum_calculation(self, versioning_service):
        """Test checksum calculation for data integrity."""
        data1 = b"Test data"
        data2 = b"Test data"
        data3 = b"Different data"
        
        checksum1 = versioning_service._calculate_checksum(data1)
        checksum2 = versioning_service._calculate_checksum(data2)
        checksum3 = versioning_service._calculate_checksum(data3)
        
        # Same data should produce same checksum
        assert checksum1 == checksum2
        # Different data should produce different checksum
        assert checksum1 != checksum3
        # Checksum should be hex string
        assert len(checksum1) == 64  # SHA-256 produces 64 hex chars


class TestDataIntegrity:
    """Tests for data integrity validation."""
    
    @pytest.mark.asyncio
    async def test_validate_integrity_valid_content(
        self, versioning_service, sample_content
    ):
        """Test integrity validation for valid content."""
        content_id = str(sample_content.id)
        
        is_valid, errors = await versioning_service.validate_data_integrity(content_id)
        
        assert is_valid is True
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_integrity_with_versions(
        self, versioning_service, sample_content
    ):
        """Test integrity validation with version history."""
        content_id = str(sample_content.id)
        
        # Create some versions
        for i in range(3):
            await versioning_service.increment_version(
                content_id=content_id,
                changes_summary=f"Update {i+1}",
                user_id="test_user_123"
            )
        
        is_valid, errors = await versioning_service.validate_data_integrity(content_id)
        
        assert is_valid is True
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_integrity_with_assets(
        self, versioning_service, sample_content
    ):
        """Test integrity validation with media assets."""
        content_id = str(sample_content.id)
        
        # Store an asset
        await versioning_service.store_media_asset(
            content_id=content_id,
            asset_data=b"Test asset data",
            asset_type="image",
            filename="test.jpg"
        )
        
        is_valid, errors = await versioning_service.validate_data_integrity(content_id)
        
        assert is_valid is True
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_integrity_nonexistent_content(self, versioning_service):
        """Test integrity validation for nonexistent content."""
        is_valid, errors = await versioning_service.validate_data_integrity(
            "507f1f77bcf86cd799439011"
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert "not found" in errors[0].lower()


class TestBackupAndRestore:
    """Tests for backup and restore functionality."""
    
    @pytest.mark.asyncio
    async def test_create_backup_success(self, versioning_service, sample_content):
        """Test successful backup creation."""
        content_id = str(sample_content.id)
        
        # Create some versions
        await versioning_service.increment_version(
            content_id=content_id,
            changes_summary="Update 1",
            user_id="test_user_123"
        )
        
        backup_path = await versioning_service.create_backup(content_id)
        
        assert backup_path is not None
        assert "backups" in backup_path
        assert content_id in backup_path
    
    @pytest.mark.asyncio
    async def test_create_backup_with_custom_location(
        self, versioning_service, sample_content
    ):
        """Test backup creation with custom location."""
        content_id = str(sample_content.id)
        custom_location = "custom/backup/location.json"
        
        backup_path = await versioning_service.create_backup(
            content_id, backup_location=custom_location
        )
        
        # Check that custom location is part of the path
        assert "custom" in backup_path
        assert "backup" in backup_path
        assert "location.json" in backup_path
    
    @pytest.mark.asyncio
    async def test_backup_includes_versions(self, versioning_service, sample_content):
        """Test that backup includes version history."""
        content_id = str(sample_content.id)
        
        # Create versions
        for i in range(3):
            await versioning_service.increment_version(
                content_id=content_id,
                changes_summary=f"Update {i+1}",
                user_id="test_user_123"
            )
        
        backup_path = await versioning_service.create_backup(content_id)
        
        # Verify backup was recorded
        backup_record = await versioning_service.database.content_backups.find_one({
            "content_id": content_id
        })
        
        assert backup_record is not None
        assert backup_record["version_count"] == 3
    
    @pytest.mark.asyncio
    async def test_restore_from_backup_success(
        self, versioning_service, sample_content
    ):
        """Test successful restore from backup."""
        content_id = str(sample_content.id)
        original_title = sample_content.title
        
        # Create backup
        backup_path = await versioning_service.create_backup(content_id)
        
        # Delete original content
        await versioning_service.database.content_items.delete_one(
            {"_id": PyObjectId(content_id)}
        )
        
        # Restore from backup
        restored_id = await versioning_service.restore_from_backup(backup_path)
        
        assert restored_id is not None
        
        # Verify content was restored (use the new ID)
        restored_content = await versioning_service.database.content_items.find_one(
            {"_id": PyObjectId(restored_id)}
        )
        assert restored_content is not None
        assert restored_content["title"] == original_title
    
    @pytest.mark.asyncio
    async def test_restore_nonexistent_backup_fails(self, versioning_service):
        """Test that restoring from nonexistent backup fails."""
        with pytest.raises(NotFoundError):
            await versioning_service.restore_from_backup("nonexistent/backup.json")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_version_increments(
        self, versioning_service, sample_content
    ):
        """Test handling of concurrent version increments."""
        content_id = str(sample_content.id)
        
        # Simulate concurrent increments
        tasks = [
            versioning_service.increment_version(
                content_id=content_id,
                changes_summary=f"Concurrent update {i}",
                user_id="test_user_123"
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed (though order may vary)
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 5
    
    @pytest.mark.asyncio
    async def test_large_asset_storage(self, versioning_service, sample_content):
        """Test storing large media assets."""
        content_id = str(sample_content.id)
        
        # Create 1MB of data
        large_data = b"x" * (1024 * 1024)
        
        storage_path = await versioning_service.store_media_asset(
            content_id=content_id,
            asset_data=large_data,
            asset_type="video",
            filename="large_video.mp4"
        )
        
        # Verify storage
        retrieved_data = await versioning_service.retrieve_media_asset(storage_path)
        assert len(retrieved_data) == len(large_data)
    
    @pytest.mark.asyncio
    async def test_special_characters_in_filename(
        self, versioning_service, sample_content
    ):
        """Test handling of special characters in filenames."""
        content_id = str(sample_content.id)
        
        # Filename with special characters
        filename = "test file!@#$%^&*().jpg"
        
        storage_path = await versioning_service.store_media_asset(
            content_id=content_id,
            asset_data=b"test data",
            asset_type="image",
            filename=filename
        )
        
        # Should sanitize filename
        assert storage_path is not None
        # Special chars should be removed
        assert "!@#$%^&*()" not in storage_path


class TestServiceSingleton:
    """Tests for service singleton pattern."""
    
    def test_get_versioning_service_singleton(self):
        """Test that get_versioning_service returns singleton."""
        service1 = get_versioning_service()
        service2 = get_versioning_service()
        
        assert service1 is service2


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

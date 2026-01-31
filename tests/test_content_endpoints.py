"""
Tests for content management API endpoints.

This module tests the REST API endpoints for content creation,
retrieval, update, and deletion.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.models.content import ContentItem, ContentMetadata
from app.models.base import ContentType, WorkflowState
from app.models.users import User, APIKey, UsageStats, UsageLimits


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed_password",
        is_active=True,
        is_verified=True,
        usage_stats=UsageStats(),
        usage_limits=UsageLimits()
    )


@pytest.fixture
def sample_content_item():
    """Create a sample content item for testing."""
    metadata = ContentMetadata(
        author="testuser",
        title="Test Content",
        description="Test description"
    )
    
    return ContentItem(
        type=ContentType.TEXT,
        title="Test Content",
        content="This is test content",
        content_metadata=metadata,
        workflow_state=WorkflowState.CREATE,
        user_id="testuser",
        tags=["test", "sample"]
    )


class TestContentEndpoints:
    """Test content management endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_content_success(self, mock_user, sample_content_item):
        """Test successful content creation."""
        from app.api.v1.endpoints.content import create_content, CreateContentRequest
        
        request = CreateContentRequest(
            type=ContentType.TEXT,
            title="Test Content",
            content="This is test content",
            author="testuser",
            tags=["test", "sample"]
        )
        
        # Mock database
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test_id"))
            mock_db.return_value.content_items = mock_collection
            mock_db.return_value.users = AsyncMock()
            mock_db.return_value.users.update_one = AsyncMock()
            
            response = await create_content(request, mock_user, mock_user)
            
            assert response.title == "Test Content"
            assert response.type == ContentType.TEXT
            assert response.user_id == "testuser"
            assert "test" in response.tags
    
    @pytest.mark.asyncio
    async def test_get_content_success(self, mock_user, sample_content_item):
        """Test successful content retrieval."""
        from app.api.v1.endpoints.content import get_content
        from app.models.base import PyObjectId
        
        content_doc = sample_content_item.dict(by_alias=True)
        # Use a valid ObjectId
        test_id = PyObjectId()
        content_doc["_id"] = test_id
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.find_one = AsyncMock(return_value=content_doc)
            mock_db.return_value.content_items = mock_collection
            
            response = await get_content(str(test_id), mock_user)
            
            assert response.title == "Test Content"
            assert response.user_id == "testuser"
    
    @pytest.mark.asyncio
    async def test_get_content_not_found(self, mock_user):
        """Test content retrieval when content doesn't exist."""
        from app.api.v1.endpoints.content import get_content
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.find_one = AsyncMock(return_value=None)
            mock_db.return_value.content_items = mock_collection
            
            with pytest.raises(HTTPException) as exc_info:
                await get_content("nonexistent_id", mock_user)
            
            assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_content_access_denied(self, mock_user, sample_content_item):
        """Test content retrieval with access denied."""
        from app.api.v1.endpoints.content import get_content
        
        content_doc = sample_content_item.dict(by_alias=True)
        content_doc["_id"] = "test_id"
        content_doc["user_id"] = "otheruser"  # Different user
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.find_one = AsyncMock(return_value=content_doc)
            mock_db.return_value.content_items = mock_collection
            
            with pytest.raises(HTTPException) as exc_info:
                await get_content("test_id", mock_user)
            
            assert exc_info.value.status_code == 403
    
    @pytest.mark.asyncio
    async def test_list_content(self, mock_user, sample_content_item):
        """Test listing content items."""
        from app.api.v1.endpoints.content import list_content
        from bson import ObjectId
        
        content_doc = sample_content_item.dict(by_alias=True)
        content_doc["_id"] = ObjectId()  # Use proper ObjectId instead of string
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.count_documents = AsyncMock(return_value=1)
            
            # Mock cursor
            mock_cursor = AsyncMock()
            mock_cursor.sort = MagicMock(return_value=mock_cursor)
            mock_cursor.skip = MagicMock(return_value=mock_cursor)
            mock_cursor.limit = MagicMock(return_value=mock_cursor)
            mock_cursor.to_list = AsyncMock(return_value=[content_doc])
            
            mock_collection.find = MagicMock(return_value=mock_cursor)
            mock_db.return_value.content_items = mock_collection
            
            # Pass all required parameters with default values
            response = await list_content(
                user=mock_user,
                content_type=None,
                workflow_state=None,
                is_published=None,
                tags=None,
                skip=0,
                limit=20
            )
            
            assert response.total == 1
            assert len(response.items) == 1
            assert response.items[0].title == "Test Content"
    
    @pytest.mark.asyncio
    async def test_update_content(self, mock_user, sample_content_item):
        """Test updating content."""
        from app.api.v1.endpoints.content import update_content, UpdateContentRequest
        from bson import ObjectId
        
        content_doc = sample_content_item.dict(by_alias=True)
        test_id = ObjectId()  # Use proper ObjectId
        content_doc["_id"] = test_id
        
        request = UpdateContentRequest(
            title="Updated Title",
            content="Updated content"
        )
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.find_one = AsyncMock(return_value=content_doc)
            mock_collection.update_one = AsyncMock()
            mock_db.return_value.content_items = mock_collection
            
            # Mock versioning service
            with patch('app.api.v1.endpoints.content.get_versioning_service') as mock_versioning:
                mock_service = AsyncMock()
                mock_service.create_version = AsyncMock()
                mock_versioning.return_value = mock_service
                
                response = await update_content(str(test_id), request, mock_user)
                
                assert response.title == "Updated Title"
                # Version should be incremented
                assert response.version == 2
    
    @pytest.mark.asyncio
    async def test_delete_content(self, mock_user, sample_content_item):
        """Test deleting content."""
        from app.api.v1.endpoints.content import delete_content
        
        content_doc = sample_content_item.dict(by_alias=True)
        content_doc["_id"] = "test_id"
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.find_one = AsyncMock(return_value=content_doc)
            mock_collection.delete_one = AsyncMock()
            mock_db.return_value.content_items = mock_collection
            mock_db.return_value.content_versions = AsyncMock()
            mock_db.return_value.content_versions.delete_many = AsyncMock()
            
            result = await delete_content("test_id", mock_user)
            
            assert result is None
            mock_collection.delete_one.assert_called_once()


class TestCollectionEndpoints:
    """Test collection management endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_collection(self, mock_user):
        """Test creating a content collection."""
        from app.api.v1.endpoints.content import create_collection, CreateCollectionRequest
        
        request = CreateCollectionRequest(
            name="Test Collection",
            description="Test description",
            collection_type="campaign"
        )
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="collection_id"))
            mock_db.return_value.content_collections = mock_collection
            
            response = await create_collection(request, mock_user)
            
            assert response["name"] == "Test Collection"
            assert response["collection_type"] == "campaign"
    
    @pytest.mark.asyncio
    async def test_list_collections(self, mock_user):
        """Test listing collections."""
        from app.api.v1.endpoints.content import list_collections
        from bson import ObjectId
        
        collection_doc = {
            "_id": ObjectId(),  # Use proper ObjectId
            "name": "Test Collection",
            "description": "Test description",
            "collection_type": "campaign",
            "is_public": False,
            "content_ids": ["content1", "content2"],
            "user_id": "testuser",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        with patch('app.api.v1.endpoints.content.get_database') as mock_db:
            mock_collection = AsyncMock()
            mock_collection.count_documents = AsyncMock(return_value=1)
            
            # Mock cursor
            mock_cursor = AsyncMock()
            mock_cursor.sort = MagicMock(return_value=mock_cursor)
            mock_cursor.skip = MagicMock(return_value=mock_cursor)
            mock_cursor.limit = MagicMock(return_value=mock_cursor)
            mock_cursor.to_list = AsyncMock(return_value=[collection_doc])
            
            mock_collection.find = MagicMock(return_value=mock_cursor)
            mock_db.return_value.content_collections = mock_collection
            
            # Pass all required parameters with default values
            response = await list_collections(
                user=mock_user,
                skip=0,
                limit=20
            )
            
            assert response["total"] == 1
            assert len(response["collections"]) == 1
            assert response["collections"][0]["name"] == "Test Collection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

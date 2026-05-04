"""
Content management API endpoints for ContentFlow AI.

This module provides REST API endpoints for creating, reading, updating,
and deleting content items, as well as managing content versions and collections.

Requirements: 6.1, 7.2, 9.1
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.models.content import (
    ContentItem, ContentMetadata, ContentVersion, ContentCollection,
    EngagementMetrics, OptimizationData, ProcessingStep, CostData
)
from app.models.base import ContentType, WorkflowState, Platform, PaginatedResponse
from app.models.users import User
from app.api.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_content_create,
    require_content_read,
    require_content_update,
    require_content_delete,
    check_content_generation_limit
)
from app.core.database import get_database
from app.core.exceptions import ValidationError, NotFoundError
from app.services.content_versioning import get_versioning_service

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models

class CreateContentRequest(BaseModel):
    """Request model for creating content."""
    type: ContentType
    title: str = Field(..., min_length=1, max_length=200)
    content: Any
    author: str
    description: Optional[str] = None
    language: str = "en"
    tags: List[str] = Field(default_factory=list)
    workflow_state: WorkflowState = WorkflowState.CREATE


class UpdateContentRequest(BaseModel):
    """Request model for updating content."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[Any] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    workflow_state: Optional[WorkflowState] = None
    is_published: Optional[bool] = None


class ContentResponse(BaseModel):
    """Response model for content items."""
    id: str
    type: ContentType
    title: str
    content: Any
    content_metadata: ContentMetadata
    workflow_state: WorkflowState
    version: int
    parent_id: Optional[str]
    is_published: bool
    published_at: Optional[datetime]
    user_id: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]


class ContentListResponse(BaseModel):
    """Response model for content list."""
    items: List[ContentResponse]
    total: int
    skip: int
    limit: int
    has_more: bool


class CreateCollectionRequest(BaseModel):
    """Request model for creating a content collection."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    collection_type: str = "general"
    is_public: bool = False


class AddToCollectionRequest(BaseModel):
    """Request model for adding content to a collection."""
    content_ids: List[str]


# Content CRUD Endpoints

@router.post("/", response_model=ContentResponse, status_code=status.HTTP_201_CREATED)
async def create_content(
    request: CreateContentRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Create a new content item.
    
    Creates a new content item with the specified type, title, and content.
    The content is initialized with default metadata and workflow state.
    """
    try:
        # Create content metadata
        metadata = ContentMetadata(
            author=request.author,
            title=request.title,
            description=request.description,
            language=request.language
        )
        
        # Create content item
        username = user.username if user else "anonymous"
        content_item = ContentItem(
            type=request.type,
            title=request.title,
            content=request.content,
            content_metadata=metadata,
            workflow_state=request.workflow_state,
            user_id=username,
            tags=request.tags,
            is_published=True,  # Auto-publish content so it appears in Discovery
            published_at=datetime.utcnow()
        )
        
        # Update metadata based on content
        content_item.update_metadata()
        
        # Save to database
        db = get_database()
        result = await db.content_items.insert_one(content_item.dict(by_alias=True))
        content_item.id = result.inserted_id
        
        # Update user stats if user is authenticated
        if user:
            user.usage_stats.content_items_created += 1
            await db.users.update_one(
                {"_id": user.id},
                {"$set": {"usage_stats": user.usage_stats.dict()}}
            )
        
        logger.info(f"Content created: {content_item.id} by user {username}")
        
        return ContentResponse(
            id=str(content_item.id),
            type=content_item.type,
            title=content_item.title,
            content=content_item.content,
            content_metadata=content_item.content_metadata,
            workflow_state=content_item.workflow_state,
            version=content_item.version,
            parent_id=content_item.parent_id,
            is_published=content_item.is_published,
            published_at=content_item.published_at,
            user_id=content_item.user_id,
            created_at=content_item.created_at,
            updated_at=content_item.updated_at,
            tags=content_item.tags
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create content: {str(e)}"
        )


@router.get("/{content_id}", response_model=ContentResponse)
async def get_content(
    content_id: str,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get a specific content item by ID.
    
    Returns the full content item including metadata, workflow state,
    and processing history.
    """
    try:
        db = get_database()
        content_doc = await db.content_items.find_one({"_id": content_id})
        
        if not content_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content {content_id} not found"
            )
        
        # Check if user has access
        if content_doc.get("user_id") != user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        content_item = ContentItem(**content_doc)
        
        return ContentResponse(
            id=str(content_item.id),
            type=content_item.type,
            title=content_item.title,
            content=content_item.content,
            content_metadata=content_item.content_metadata,
            workflow_state=content_item.workflow_state,
            version=content_item.version,
            parent_id=content_item.parent_id,
            is_published=content_item.is_published,
            published_at=content_item.published_at,
            user_id=content_item.user_id,
            created_at=content_item.created_at,
            updated_at=content_item.updated_at,
            tags=content_item.tags
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content {content_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve content: {str(e)}"
        )


@router.get("/", response_model=ContentListResponse)
async def list_content(
    user: Optional[User] = Depends(get_current_user_optional),
    content_type: Optional[ContentType] = Query(None, description="Filter by content type"),
    workflow_state: Optional[WorkflowState] = Query(None, description="Filter by workflow state"),
    is_published: Optional[bool] = Query(None, description="Filter by published status"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    all_users: bool = Query(False, description="Show content from all users (for discovery)")
):
    """
    List content items.
    
    By default, shows content for the current user only.
    Set all_users=true to show published content from all users (for discovery page).
    
    Supports filtering by content type, workflow state, published status, and tags.
    Results are paginated.
    """
    try:
        db = get_database()
        
        # Build query
        query = {}
        
        # If all_users is False, filter by current user
        if not all_users:
            username = user.username if user else "anonymous"
            query["user_id"] = username
        else:
            # For discovery, only show published content
            query["is_published"] = True
        
        if content_type:
            query["type"] = content_type.value
        
        if workflow_state:
            query["workflow_state"] = workflow_state.value
        
        if is_published is not None and not all_users:
            query["is_published"] = is_published
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            query["tags"] = {"$in": tag_list}
        
        # Get total count
        total = await db.content_items.count_documents(query)
        
        # Get paginated results
        cursor = db.content_items.find(query).sort("created_at", -1).skip(skip).limit(limit)
        content_docs = await cursor.to_list(length=limit)
        
        # Convert to response models
        items = []
        for doc in content_docs:
            content_item = ContentItem(**doc)
            items.append(ContentResponse(
                id=str(content_item.id),
                type=content_item.type,
                title=content_item.title,
                content=content_item.content,
                content_metadata=content_item.content_metadata,
                workflow_state=content_item.workflow_state,
                version=content_item.version,
                parent_id=content_item.parent_id,
                is_published=content_item.is_published,
                published_at=content_item.published_at,
                user_id=content_item.user_id,
                created_at=content_item.created_at,
                updated_at=content_item.updated_at,
                tags=content_item.tags
            ))
        
        return ContentListResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit,
            has_more=skip + len(items) < total
        )
        
    except Exception as e:
        logger.error(f"Error listing content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list content: {str(e)}"
        )


@router.put("/{content_id}", response_model=ContentResponse)
async def update_content(
    content_id: str,
    request: UpdateContentRequest,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Update an existing content item.
    
    Allows updating title, content, description, tags, workflow state,
    and published status. Creates a new version if content is modified.
    """
    try:
        db = get_database()
        content_doc = await db.content_items.find_one({"_id": content_id})
        
        if not content_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content {content_id} not found"
            )
        
        # Check if user has access (skip if no user for debugging)
        if user and content_doc.get("user_id") != user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        content_item = ContentItem(**content_doc)
        
        # Track if content changed (for versioning)
        content_changed = False
        
        # Update fields
        if request.title is not None:
            content_item.title = request.title
            content_item.content_metadata.title = request.title
        
        if request.content is not None:
            content_item.content = request.content
            content_changed = True
        
        if request.description is not None:
            content_item.content_metadata.description = request.description
        
        if request.tags is not None:
            content_item.tags = request.tags
        
        if request.workflow_state is not None:
            content_item.workflow_state = request.workflow_state
        
        if request.is_published is not None:
            content_item.is_published = request.is_published
            if request.is_published and not content_item.published_at:
                content_item.published_at = datetime.utcnow()
        
        # Update metadata
        content_item.update_metadata()
        
        # Create version if content changed (skip if no user)
        if content_changed and user:
            versioning_service = get_versioning_service()
            await versioning_service.create_version(
                content_id=content_id,
                changes_summary="Content updated",
                user_id=user.username
            )
            content_item.version += 1
        
        # Save to database
        await db.content_items.update_one(
            {"_id": content_id},
            {"$set": content_item.dict(by_alias=True, exclude={"id"})}
        )
        
        username = user.username if user else "anonymous"
        logger.info(f"Content updated: {content_id} by user {username}")
        
        return ContentResponse(
            id=str(content_item.id),
            type=content_item.type,
            title=content_item.title,
            content=content_item.content,
            content_metadata=content_item.content_metadata,
            workflow_state=content_item.workflow_state,
            version=content_item.version,
            parent_id=content_item.parent_id,
            is_published=content_item.is_published,
            published_at=content_item.published_at,
            user_id=content_item.user_id,
            created_at=content_item.created_at,
            updated_at=content_item.updated_at,
            tags=content_item.tags
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating content {content_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update content: {str(e)}"
        )


@router.delete("/{content_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_content(
    content_id: str,
    user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Delete a content item.
    
    Permanently removes the content item and all its versions.
    This action cannot be undone.
    """
    try:
        db = get_database()
        content_doc = await db.content_items.find_one({"_id": content_id})
        
        if not content_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content {content_id} not found"
            )
        
        # Check if user has access (skip if no user for debugging)
        if user and content_doc.get("user_id") != user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Delete content
        await db.content_items.delete_one({"_id": content_id})
        
        # Delete versions
        await db.content_versions.delete_many({"content_id": content_id})
        
        username = user.username if user else "anonymous"
        logger.info(f"Content deleted: {content_id} by user {username}")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting content {content_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete content: {str(e)}"
        )


# Version Management Endpoints

@router.get("/{content_id}/versions")
async def get_content_versions(
    content_id: str,
    user: User = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=50, description="Number of versions to return")
):
    """
    Get version history for a content item.
    
    Returns a list of all versions with metadata about changes.
    """
    try:
        db = get_database()
        
        # Check if content exists and user has access
        content_doc = await db.content_items.find_one({"_id": content_id})
        if not content_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content {content_id} not found"
            )
        
        if content_doc.get("user_id") != user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get versions
        versioning_service = get_versioning_service()
        versions = await versioning_service.get_version_history(
            content_id=content_id,
            limit=limit
        )
        
        return {
            "content_id": content_id,
            "versions": versions,
            "total_versions": len(versions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving versions for content {content_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve versions: {str(e)}"
        )


@router.get("/{content_id}/versions/{version_number}")
async def get_content_version(
    content_id: str,
    version_number: int,
    user: User = Depends(get_current_user)
):
    """
    Get a specific version of a content item.
    
    Returns the content as it existed at the specified version.
    """
    try:
        db = get_database()
        
        # Check if content exists and user has access
        content_doc = await db.content_items.find_one({"_id": content_id})
        if not content_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content {content_id} not found"
            )
        
        if content_doc.get("user_id") != user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get specific version
        versioning_service = get_versioning_service()
        version = await versioning_service.get_version(
            content_id=content_id,
            version_number=version_number
        )
        
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version_number} not found"
            )
        
        return version
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving version {version_number} for content {content_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve version: {str(e)}"
        )


# Collection Management Endpoints

@router.post("/collections", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_collection(
    request: CreateCollectionRequest,
    user: User = Depends(get_current_user)
):
    """
    Create a new content collection.
    
    Collections allow grouping related content items for organization
    and batch operations.
    """
    try:
        collection = ContentCollection(
            name=request.name,
            description=request.description,
            collection_type=request.collection_type,
            is_public=request.is_public,
            user_id=user.username
        )
        
        db = get_database()
        result = await db.content_collections.insert_one(collection.dict(by_alias=True))
        collection.id = result.inserted_id
        
        logger.info(f"Collection created: {collection.id} by user {user.username}")
        
        return {
            "id": str(collection.id),
            "name": collection.name,
            "description": collection.description,
            "collection_type": collection.collection_type,
            "is_public": collection.is_public,
            "content_ids": collection.content_ids,
            "created_at": collection.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}"
        )


@router.post("/collections/{collection_id}/content")
async def add_content_to_collection(
    collection_id: str,
    request: AddToCollectionRequest,
    user: User = Depends(get_current_user)
):
    """
    Add content items to a collection.
    
    Adds one or more content items to the specified collection.
    """
    try:
        db = get_database()
        
        # Check if collection exists and user has access
        collection_doc = await db.content_collections.find_one({"_id": collection_id})
        if not collection_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection {collection_id} not found"
            )
        
        if collection_doc.get("user_id") != user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        collection = ContentCollection(**collection_doc)
        
        # Add content items
        for content_id in request.content_ids:
            # Verify content exists and user owns it
            content_doc = await db.content_items.find_one({"_id": content_id})
            if content_doc and content_doc.get("user_id") == user.username:
                collection.add_content(content_id)
        
        # Save collection
        await db.content_collections.update_one(
            {"_id": collection_id},
            {"$set": {
                "content_ids": collection.content_ids,
                "updated_at": collection.updated_at
            }}
        )
        
        logger.info(f"Added {len(request.content_ids)} items to collection {collection_id}")
        
        return {
            "message": f"Added {len(request.content_ids)} items to collection",
            "collection_id": collection_id,
            "total_items": len(collection.content_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding content to collection {collection_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add content to collection: {str(e)}"
        )


@router.get("/collections/{collection_id}")
async def get_collection(
    collection_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get a content collection with its items.
    
    Returns the collection metadata and all content items in the collection.
    """
    try:
        db = get_database()
        
        collection_doc = await db.content_collections.find_one({"_id": collection_id})
        if not collection_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection {collection_id} not found"
            )
        
        # Check access
        if not collection_doc.get("is_public") and collection_doc.get("user_id") != user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        collection = ContentCollection(**collection_doc)
        
        # Get content items
        content_items = []
        for content_id in collection.content_ids:
            content_doc = await db.content_items.find_one({"_id": content_id})
            if content_doc:
                content_item = ContentItem(**content_doc)
                content_items.append({
                    "id": str(content_item.id),
                    "type": content_item.type.value,
                    "title": content_item.title,
                    "workflow_state": content_item.workflow_state.value,
                    "created_at": content_item.created_at.isoformat()
                })
        
        return {
            "id": str(collection.id),
            "name": collection.name,
            "description": collection.description,
            "collection_type": collection.collection_type,
            "is_public": collection.is_public,
            "content_items": content_items,
            "total_items": len(content_items),
            "created_at": collection.created_at.isoformat(),
            "updated_at": collection.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving collection {collection_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve collection: {str(e)}"
        )


@router.get("/collections")
async def list_collections(
    user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """
    List collections for the current user.
    
    Returns all collections owned by the user, paginated.
    """
    try:
        db = get_database()
        
        query = {"user_id": user.username}
        
        total = await db.content_collections.count_documents(query)
        
        cursor = db.content_collections.find(query).sort("created_at", -1).skip(skip).limit(limit)
        collection_docs = await cursor.to_list(length=limit)
        
        collections = []
        for doc in collection_docs:
            collection = ContentCollection(**doc)
            collections.append({
                "id": str(collection.id),
                "name": collection.name,
                "description": collection.description,
                "collection_type": collection.collection_type,
                "is_public": collection.is_public,
                "item_count": len(collection.content_ids),
                "created_at": collection.created_at.isoformat()
            })
        
        return {
            "collections": collections,
            "total": total,
            "skip": skip,
            "limit": limit,
            "has_more": skip + len(collections) < total
        }
        
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )


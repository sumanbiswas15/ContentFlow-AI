"""
Notification system for ContentFlow AI.

This module handles job completion and failure notifications,
including real-time updates and user notifications.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.models.jobs import AsyncJob
from app.models.base import JobStatus

logger = logging.getLogger(__name__)


class NotificationEvent:
    """Represents a notification event."""
    
    def __init__(
        self,
        event_type: str,
        job: AsyncJob,
        timestamp: datetime = None,
        metadata: Dict[str, Any] = None
    ):
        self.event_type = event_type
        self.job = job
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}


class NotificationHandler:
    """Base class for notification handlers."""
    
    async def handle(self, event: NotificationEvent):
        """Handle a notification event."""
        raise NotImplementedError


class DatabaseNotificationHandler(NotificationHandler):
    """Stores notifications in the database."""
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        self.database = database or get_database()
    
    async def handle(self, event: NotificationEvent):
        """Store notification in database."""
        notification_doc = {
            "event_type": event.event_type,
            "job_id": str(event.job.id),
            "user_id": event.job.user_id,
            "job_type": event.job.job_type,
            "engine": event.job.engine,
            "operation": event.job.operation,
            "status": event.job.status,
            "timestamp": event.timestamp,
            "metadata": event.metadata,
            "read": False
        }
        
        await self.database.notifications.insert_one(notification_doc)
        logger.debug(f"Stored notification for job {event.job.id}")


class WebSocketNotificationHandler(NotificationHandler):
    """Sends real-time notifications via WebSocket."""
    
    def __init__(self):
        self.connections: Dict[str, List[Any]] = {}  # user_id -> list of websocket connections
    
    def add_connection(self, user_id: str, websocket):
        """Add a WebSocket connection for a user."""
        if user_id not in self.connections:
            self.connections[user_id] = []
        self.connections[user_id].append(websocket)
        logger.debug(f"Added WebSocket connection for user {user_id}")
    
    def remove_connection(self, user_id: str, websocket):
        """Remove a WebSocket connection for a user."""
        if user_id in self.connections:
            if websocket in self.connections[user_id]:
                self.connections[user_id].remove(websocket)
            if not self.connections[user_id]:
                del self.connections[user_id]
        logger.debug(f"Removed WebSocket connection for user {user_id}")
    
    async def handle(self, event: NotificationEvent):
        """Send notification via WebSocket."""
        user_id = event.job.user_id
        
        if user_id not in self.connections:
            return
        
        message = {
            "type": "job_notification",
            "event_type": event.event_type,
            "job_id": str(event.job.id),
            "job_type": event.job.job_type,
            "engine": event.job.engine,
            "operation": event.job.operation,
            "status": event.job.status,
            "timestamp": event.timestamp.isoformat(),
            "metadata": event.metadata
        }
        
        # Send to all connections for this user
        disconnected = []
        for websocket in self.connections[user_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket notification: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.remove_connection(user_id, websocket)


class EmailNotificationHandler(NotificationHandler):
    """Sends email notifications for important events."""
    
    def __init__(self, email_service=None):
        self.email_service = email_service
        self.enabled = email_service is not None
    
    async def handle(self, event: NotificationEvent):
        """Send email notification."""
        if not self.enabled:
            return
        
        # Only send emails for failures or important completions
        if event.event_type not in ["job_failed", "workflow_completed", "workflow_failed"]:
            return
        
        try:
            subject = self._get_email_subject(event)
            body = self._get_email_body(event)
            
            # In a real implementation, you would send the email here
            # await self.email_service.send_email(
            #     to=event.job.user_id,  # Assuming user_id is email
            #     subject=subject,
            #     body=body
            # )
            
            logger.info(f"Email notification sent for job {event.job.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _get_email_subject(self, event: NotificationEvent) -> str:
        """Generate email subject based on event."""
        if event.event_type == "job_failed":
            return f"ContentFlow AI: Job Failed - {event.job.operation}"
        elif event.event_type == "workflow_completed":
            return f"ContentFlow AI: Workflow Completed"
        elif event.event_type == "workflow_failed":
            return f"ContentFlow AI: Workflow Failed"
        else:
            return f"ContentFlow AI: Job Update - {event.job.operation}"
    
    def _get_email_body(self, event: NotificationEvent) -> str:
        """Generate email body based on event."""
        job = event.job
        
        if event.event_type == "job_failed":
            return f"""
Your ContentFlow AI job has failed:

Job ID: {job.id}
Engine: {job.engine}
Operation: {job.operation}
Error: {job.last_error}
Retry Count: {job.retry_count}

Please check your job status in the ContentFlow AI dashboard.
"""
        else:
            return f"""
Your ContentFlow AI job has been updated:

Job ID: {job.id}
Engine: {job.engine}
Operation: {job.operation}
Status: {job.status}

Check your dashboard for more details.
"""


class NotificationService:
    """
    Central notification service that coordinates multiple handlers.
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        self.database = database or get_database()
        self.handlers: List[NotificationHandler] = []
        
        # Initialize default handlers
        self.db_handler = DatabaseNotificationHandler(database)
        self.websocket_handler = WebSocketNotificationHandler()
        self.email_handler = EmailNotificationHandler()
        
        self.handlers.extend([
            self.db_handler,
            self.websocket_handler,
            self.email_handler
        ])
    
    def add_handler(self, handler: NotificationHandler):
        """Add a custom notification handler."""
        self.handlers.append(handler)
        logger.info(f"Added notification handler: {handler.__class__.__name__}")
    
    def remove_handler(self, handler: NotificationHandler):
        """Remove a notification handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
            logger.info(f"Removed notification handler: {handler.__class__.__name__}")
    
    async def notify_job_started(self, job: AsyncJob):
        """Send notification when job starts."""
        event = NotificationEvent("job_started", job)
        await self._send_notifications(event)
    
    async def notify_job_completed(self, job: AsyncJob):
        """Send notification when job completes successfully."""
        event = NotificationEvent(
            "job_completed",
            job,
            metadata={
                "execution_time_ms": job.execution_time_ms,
                "tokens_used": job.result.tokens_used if job.result else 0,
                "cost": job.result.cost if job.result else 0.0
            }
        )
        await self._send_notifications(event)
    
    async def notify_job_failed(self, job: AsyncJob):
        """Send notification when job fails."""
        event = NotificationEvent(
            "job_failed",
            job,
            metadata={
                "error_message": job.last_error,
                "retry_count": job.retry_count,
                "will_retry": job.should_retry()
            }
        )
        await self._send_notifications(event)
    
    async def notify_job_cancelled(self, job: AsyncJob):
        """Send notification when job is cancelled."""
        event = NotificationEvent("job_cancelled", job)
        await self._send_notifications(event)
    
    async def notify_workflow_completed(self, workflow_id: str, user_id: str):
        """Send notification when workflow completes."""
        # Create a dummy job for workflow notifications
        dummy_job = AsyncJob(
            job_type="workflow",
            engine="workflow_manager",
            operation="workflow_execution",
            parameters={"workflow_id": workflow_id},
            user_id=user_id
        )
        
        event = NotificationEvent(
            "workflow_completed",
            dummy_job,
            metadata={"workflow_id": workflow_id}
        )
        await self._send_notifications(event)
    
    async def notify_workflow_failed(self, workflow_id: str, user_id: str, error_message: str):
        """Send notification when workflow fails."""
        dummy_job = AsyncJob(
            job_type="workflow",
            engine="workflow_manager",
            operation="workflow_execution",
            parameters={"workflow_id": workflow_id},
            user_id=user_id
        )
        
        event = NotificationEvent(
            "workflow_failed",
            dummy_job,
            metadata={
                "workflow_id": workflow_id,
                "error_message": error_message
            }
        )
        await self._send_notifications(event)
    
    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        query = {"user_id": user_id}
        if unread_only:
            query["read"] = False
        
        cursor = self.database.notifications.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        notifications = []
        
        async for doc in cursor:
            notifications.append({
                "id": str(doc["_id"]),
                "event_type": doc["event_type"],
                "job_id": doc["job_id"],
                "job_type": doc["job_type"],
                "engine": doc["engine"],
                "operation": doc["operation"],
                "status": doc["status"],
                "timestamp": doc["timestamp"],
                "metadata": doc["metadata"],
                "read": doc["read"]
            })
        
        return notifications
    
    async def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """Mark a notification as read."""
        result = await self.database.notifications.update_one(
            {"_id": notification_id, "user_id": user_id},
            {"$set": {"read": True}}
        )
        return result.modified_count > 0
    
    async def mark_all_notifications_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user."""
        result = await self.database.notifications.update_many(
            {"user_id": user_id, "read": False},
            {"$set": {"read": True}}
        )
        return result.modified_count
    
    async def _send_notifications(self, event: NotificationEvent):
        """Send notification to all registered handlers."""
        for handler in self.handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(f"Notification handler {handler.__class__.__name__} failed: {e}", exc_info=True)


# Global notification service instance (lazy-loaded)
notification_service = None


def get_notification_service():
    """Get or create the global notification service instance."""
    global notification_service
    if notification_service is None:
        notification_service = NotificationService()
    return notification_service


# Job processor notification handler
async def job_completion_handler(job: AsyncJob, success: bool):
    """Handler function for job processor notifications."""
    notification_service = get_notification_service()
    if success:
        await notification_service.notify_job_completed(job)
    else:
        await notification_service.notify_job_failed(job)
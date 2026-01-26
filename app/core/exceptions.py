"""
Custom exceptions for ContentFlow AI.

This module defines application-specific exceptions with proper error handling
and HTTP status code mapping for API responses.
"""

from typing import Any, Dict, Optional


class ContentFlowException(Exception):
    """Base exception for ContentFlow AI application."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(ContentFlowException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            status_code=400
        )


class AuthenticationError(ContentFlowException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(ContentFlowException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )


class NotFoundError(ContentFlowException):
    """Raised when a resource is not found."""
    
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} with identifier '{identifier}' not found"
        super().__init__(
            message=message,
            error_code="NOT_FOUND_ERROR",
            details={"resource": resource, "identifier": identifier},
            status_code=404
        )


class RateLimitError(ContentFlowException):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429
        )


class AIServiceError(ContentFlowException):
    """Raised when AI service operations fail."""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"AI service '{service}' error: {message}",
            error_code="AI_SERVICE_ERROR",
            details=details or {"service": service},
            status_code=502
        )


class ContentParsingError(ContentFlowException):
    """Raised when content parsing fails."""
    
    def __init__(self, content_type: str, message: str):
        super().__init__(
            message=f"Failed to parse {content_type} content: {message}",
            error_code="CONTENT_PARSING_ERROR",
            details={"content_type": content_type},
            status_code=400
        )


class StorageError(ContentFlowException):
    """Raised when storage operations fail."""
    
    def __init__(self, operation: str, message: str):
        super().__init__(
            message=f"Storage {operation} failed: {message}",
            error_code="STORAGE_ERROR",
            details={"operation": operation},
            status_code=500
        )


class WorkflowError(ContentFlowException):
    """Raised when workflow operations fail."""
    
    def __init__(self, workflow_id: str, message: str):
        super().__init__(
            message=f"Workflow '{workflow_id}' error: {message}",
            error_code="WORKFLOW_ERROR",
            details={"workflow_id": workflow_id},
            status_code=500
        )


class JobProcessingError(ContentFlowException):
    """Raised when job processing fails."""
    
    def __init__(self, job_id: str, message: str):
        super().__init__(
            message=f"Job '{job_id}' processing error: {message}",
            error_code="JOB_PROCESSING_ERROR",
            details={"job_id": job_id},
            status_code=500
        )


class CostLimitError(ContentFlowException):
    """Raised when cost limits are exceeded."""
    
    def __init__(self, limit_type: str, current: float, limit: float):
        message = f"{limit_type} limit exceeded: {current} > {limit}"
        super().__init__(
            message=message,
            error_code="COST_LIMIT_ERROR",
            details={
                "limit_type": limit_type,
                "current": current,
                "limit": limit
            },
            status_code=402
        )


class OrchestrationError(ContentFlowException):
    """Raised when AI orchestration operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ORCHESTRATION_ERROR",
            details=details,
            status_code=500
        )


class EngineError(ContentFlowException):
    """Raised when engine operations fail."""
    
    def __init__(self, engine: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Engine '{engine}' error: {message}",
            error_code="ENGINE_ERROR",
            details=details or {"engine": engine},
            status_code=500
        )


class UsageLimitError(ContentFlowException):
    """Raised when usage limits are exceeded."""
    
    def __init__(self, limit_type: str, message: str = None):
        message = message or f"{limit_type} usage limit exceeded"
        super().__init__(
            message=message,
            error_code="USAGE_LIMIT_ERROR",
            details={"limit_type": limit_type},
            status_code=429
        )
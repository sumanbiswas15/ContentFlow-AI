"""
Error handling middleware for ContentFlow AI.

This module provides centralized error handling and response formatting
for all API endpoints.
"""

import traceback
from typing import Union
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.exceptions import ContentFlowException
from app.core.logging import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for handling exceptions and formatting error responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and handle any exceptions."""
        try:
            response = await call_next(request)
            return response
            
        except ContentFlowException as e:
            logger.error(
                "ContentFlow exception occurred",
                error_code=e.error_code,
                message=e.message,
                details=e.details,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.error_code,
                        "message": e.message,
                        "details": e.details
                    },
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": str(request.state.timestamp) if hasattr(request.state, "timestamp") else None
                }
            )
            
        except HTTPException as e:
            logger.warning(
                "HTTP exception occurred",
                status_code=e.status_code,
                detail=e.detail,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": "HTTP_ERROR",
                        "message": e.detail,
                        "details": {}
                    },
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": str(request.state.timestamp) if hasattr(request.state, "timestamp") else None
                }
            )
            
        except Exception as e:
            logger.error(
                "Unexpected exception occurred",
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred",
                        "details": {}
                    },
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": str(request.state.timestamp) if hasattr(request.state, "timestamp") else None
                }
            )
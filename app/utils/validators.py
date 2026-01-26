"""
Validation utilities for ContentFlow AI.

This module provides common validation functions for data integrity
and business rule enforcement.
"""

import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from app.models.base import ContentType, WorkflowState, Platform


def validate_email_address(email: str) -> bool:
    """Validate email address format using regex."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))


def validate_username(username: str) -> tuple[bool, Optional[str]]:
    """Validate username format and return (is_valid, error_message)."""
    if not username:
        return False, "Username cannot be empty"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(username) > 50:
        return False, "Username cannot exceed 50 characters"
    
    if not re.match(r'^[a-zA-Z0-9_\-]+$', username):
        return False, "Username can only contain letters, numbers, hyphens, and underscores"
    
    return True, None


def validate_password_strength(password: str) -> tuple[bool, List[str]]:
    """Validate password strength and return (is_valid, list_of_issues)."""
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        issues.append("Password must contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password must contain at least one special character")
    
    return len(issues) == 0, issues


def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_content_type(content_type: str) -> bool:
    """Validate content type is supported."""
    try:
        ContentType(content_type)
        return True
    except ValueError:
        return False


def validate_workflow_state(state: str) -> bool:
    """Validate workflow state is valid."""
    try:
        WorkflowState(state)
        return True
    except ValueError:
        return False


def validate_platform(platform: str) -> bool:
    """Validate platform is supported."""
    try:
        Platform(platform)
        return True
    except ValueError:
        return False


def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> tuple[bool, List[str]]:
    """Validate JSON structure has required fields."""
    missing_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """Validate file size is within limits."""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension is allowed."""
    if not filename or '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in [ext.lower() for ext in allowed_extensions]


def validate_content_length(content: str, min_length: int = 1, max_length: int = 100000) -> tuple[bool, Optional[str]]:
    """Validate content length is within acceptable range."""
    if len(content) < min_length:
        return False, f"Content must be at least {min_length} characters long"
    
    if len(content) > max_length:
        return False, f"Content cannot exceed {max_length} characters"
    
    return True, None


def validate_tags(tags: List[str], max_tags: int = 20, max_tag_length: int = 50) -> tuple[bool, List[str]]:
    """Validate tags list."""
    issues = []
    
    if len(tags) > max_tags:
        issues.append(f"Cannot have more than {max_tags} tags")
    
    for tag in tags:
        if not tag or not tag.strip():
            issues.append("Tags cannot be empty")
            continue
        
        if len(tag) > max_tag_length:
            issues.append(f"Tag '{tag}' exceeds maximum length of {max_tag_length} characters")
        
        if not re.match(r'^[a-zA-Z0-9_\-\s]+$', tag):
            issues.append(f"Tag '{tag}' contains invalid characters")
    
    return len(issues) == 0, issues


def validate_priority(priority: int) -> bool:
    """Validate priority is within acceptable range (1-10)."""
    return 1 <= priority <= 10


def validate_pagination_params(skip: int, limit: int, max_limit: int = 100) -> tuple[bool, Optional[str]]:
    """Validate pagination parameters."""
    if skip < 0:
        return False, "Skip parameter cannot be negative"
    
    if limit < 1:
        return False, "Limit parameter must be at least 1"
    
    if limit > max_limit:
        return False, f"Limit parameter cannot exceed {max_limit}"
    
    return True, None


def sanitize_search_query(query: str) -> str:
    """Sanitize search query to prevent injection attacks."""
    if not query:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\';\\]', '', query)
    
    # Limit length
    sanitized = sanitized[:200]
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    return sanitized


def validate_cost_limit(limit: float) -> bool:
    """Validate cost limit is reasonable."""
    return 0 < limit <= 10000  # Between $0 and $10,000


def validate_token_limit(limit: int) -> bool:
    """Validate token limit is reasonable."""
    return 0 < limit <= 10000000  # Between 0 and 10 million tokens


def validate_workflow_transition(current_state: WorkflowState, target_state: WorkflowState) -> tuple[bool, Optional[str]]:
    """Validate workflow state transition is allowed."""
    # Define valid transitions
    valid_transitions = {
        WorkflowState.DISCOVER: [WorkflowState.CREATE, WorkflowState.ANALYZE],
        WorkflowState.CREATE: [WorkflowState.TRANSFORM, WorkflowState.PLAN, WorkflowState.PUBLISH],
        WorkflowState.TRANSFORM: [WorkflowState.PLAN, WorkflowState.PUBLISH, WorkflowState.CREATE],
        WorkflowState.PLAN: [WorkflowState.PUBLISH, WorkflowState.TRANSFORM],
        WorkflowState.PUBLISH: [WorkflowState.ANALYZE, WorkflowState.IMPROVE],
        WorkflowState.ANALYZE: [WorkflowState.IMPROVE, WorkflowState.DISCOVER],
        WorkflowState.IMPROVE: [WorkflowState.CREATE, WorkflowState.TRANSFORM, WorkflowState.DISCOVER]
    }
    
    if current_state not in valid_transitions:
        return False, f"Invalid current state: {current_state}"
    
    if target_state not in valid_transitions[current_state]:
        return False, f"Invalid transition from {current_state} to {target_state}"
    
    return True, None


def validate_content_metadata(metadata: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate content metadata structure and values."""
    issues = []
    
    # Check required fields
    required_fields = ["author"]
    for field in required_fields:
        if field not in metadata:
            issues.append(f"Missing required field: {field}")
    
    # Validate author
    if "author" in metadata and not isinstance(metadata["author"], str):
        issues.append("Author must be a string")
    
    # Validate language code
    if "language" in metadata:
        if not isinstance(metadata["language"], str) or len(metadata["language"]) != 2:
            issues.append("Language must be a 2-character language code")
    
    # Validate content length
    if "content_length" in metadata:
        if not isinstance(metadata["content_length"], int) or metadata["content_length"] < 0:
            issues.append("Content length must be a non-negative integer")
    
    # Validate word count
    if "word_count" in metadata:
        if not isinstance(metadata["word_count"], int) or metadata["word_count"] < 0:
            issues.append("Word count must be a non-negative integer")
    
    return len(issues) == 0, issues


def validate_job_parameters(job_type: str, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate job parameters based on job type."""
    issues = []
    
    if job_type == "content_generation":
        required_params = ["prompt", "content_type"]
        for param in required_params:
            if param not in parameters:
                issues.append(f"Missing required parameter for content generation: {param}")
        
        if "content_type" in parameters and not validate_content_type(parameters["content_type"]):
            issues.append("Invalid content type")
    
    elif job_type == "content_transformation":
        required_params = ["content", "transformation_type"]
        for param in required_params:
            if param not in parameters:
                issues.append(f"Missing required parameter for content transformation: {param}")
    
    elif job_type == "social_media_optimization":
        required_params = ["content", "platform"]
        for param in required_params:
            if param not in parameters:
                issues.append(f"Missing required parameter for social media optimization: {param}")
        
        if "platform" in parameters and not validate_platform(parameters["platform"]):
            issues.append("Invalid platform")
    
    return len(issues) == 0, issues


def validate_cost_data(cost_data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate cost tracking data structure."""
    issues = []
    
    # Validate total_tokens_used
    if "total_tokens_used" in cost_data:
        if not isinstance(cost_data["total_tokens_used"], int) or cost_data["total_tokens_used"] < 0:
            issues.append("Total tokens used must be a non-negative integer")
    
    # Validate total_cost
    if "total_cost" in cost_data:
        if not isinstance(cost_data["total_cost"], (int, float)) or cost_data["total_cost"] < 0:
            issues.append("Total cost must be a non-negative number")
    
    # Validate cost_per_operation
    if "cost_per_operation" in cost_data:
        if not isinstance(cost_data["cost_per_operation"], dict):
            issues.append("Cost per operation must be a dictionary")
        else:
            for operation, cost in cost_data["cost_per_operation"].items():
                if not isinstance(cost, (int, float)) or cost < 0:
                    issues.append(f"Cost for operation '{operation}' must be a non-negative number")
    
    # Validate currency
    if "currency" in cost_data:
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
        if cost_data["currency"] not in valid_currencies:
            issues.append(f"Currency must be one of: {', '.join(valid_currencies)}")
    
    return len(issues) == 0, issues


def validate_engagement_metrics(metrics: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate engagement metrics data structure."""
    issues = []
    
    # Define expected numeric fields
    numeric_fields = ["views", "likes", "shares", "comments", "reach", "impressions"]
    
    for field in numeric_fields:
        if field in metrics:
            if not isinstance(metrics[field], int) or metrics[field] < 0:
                issues.append(f"{field.capitalize()} must be a non-negative integer")
    
    # Validate rates (should be between 0 and 100)
    rate_fields = ["click_through_rate", "engagement_rate"]
    for field in rate_fields:
        if field in metrics:
            if not isinstance(metrics[field], (int, float)) or not (0 <= metrics[field] <= 100):
                issues.append(f"{field.replace('_', ' ').title()} must be between 0 and 100")
    
    return len(issues) == 0, issues
"""
Configuration management for ContentFlow AI.

This module handles all application settings using Pydantic Settings
for type validation and environment variable management.
"""

from typing import List, Optional, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Project Information
    PROJECT_NAME: str = "ContentFlow AI"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Security
    SECRET_KEY: Optional[str] = None
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    @validator("SECRET_KEY", pre=True, always=True)
    def generate_secret_key_if_missing(cls, v):
        """Generate a random secret key if not provided."""
        if v is None or v == "":
            return secrets.token_urlsafe(32)
        return v
    
    # CORS and Security
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "contentflow_ai"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # AI Service Configuration
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # Cost Control
    DEFAULT_DAILY_TOKEN_LIMIT: int = 100000
    DEFAULT_MONTHLY_COST_LIMIT: float = 100.0
    
    # Object Storage Configuration
    STORAGE_BACKEND: str = "local"  # local, s3, gcs
    LOCAL_STORAGE_PATH: str = "./storage"
    
    # AWS S3 Configuration (if using S3)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_BUCKET_NAME: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # Job Processing
    CELERY_BROKER_URL: str = "redis://localhost:6379"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
# ContentFlow AI

ContentFlow AI is a unified AI-driven platform that orchestrates the complete content lifecycle from discovery through analysis and improvement. The system provides modular, AI-coordinated workflows that augment human creativity across text, image, audio, and video content creation while maintaining explainability and cost control.

## Features

- **AI Orchestration**: Central LLM-powered coordination of specialized engines
- **Content Generation**: Text, image, audio, and video creation capabilities
- **Content Transformation**: Summarization, tone adjustment, translation, and platform adaptation
- **Creative Assistance**: Interactive AI collaboration for iterative content improvement
- **Social Media Optimization**: Platform-specific optimization and scheduling
- **Analytics & Discovery**: Content performance analysis and trend discovery
- **Workflow Management**: Complete content lifecycle tracking
- **Cost Control**: Usage monitoring and budget management
- **Security**: API authentication, rate limiting, and access control

## Architecture

The system follows the Orchestrator-Worker pattern with:

- **AI Orchestrator**: Central coordination using Google Gemini LLM
- **Specialized Engines**: Domain-specific workers for different content types
- **Async Processing**: Background job processing with real-time status updates
- **MongoDB Storage**: Document-based data persistence
- **FastAPI Backend**: High-performance async API framework

## Quick Start

### Prerequisites

- Python 3.9+
- MongoDB
- Redis (for job queue)
- Google AI API key
- OpenAI API key (optional)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd contentflow-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the services:
```bash
# Start MongoDB and Redis (using Docker)
docker-compose up -d mongodb redis

# Start the FastAPI application
python -m app.main
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the application is running, you can access:

- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/api/v1/openapi.json`

## Development

### Project Structure

```
app/
├── api/                 # API endpoints and routing
│   ├── v1/             # API version 1
│   │   └── endpoints/  # Individual endpoint modules
│   └── middleware/     # Custom middleware
├── core/               # Core application components
│   ├── config.py       # Configuration management
│   ├── database.py     # Database connection
│   ├── logging.py      # Logging setup
│   └── exceptions.py   # Custom exceptions
├── models/             # Data models and schemas
│   ├── base.py         # Base models and enums
│   ├── content.py      # Content-related models
│   ├── jobs.py         # Job processing models
│   └── users.py        # User and auth models
├── utils/              # Utility functions
│   ├── security.py     # Security utilities
│   └── validators.py   # Validation functions
└── main.py             # FastAPI application entry point

tests/                  # Test suite
├── conftest.py         # Test configuration
└── test_*.py           # Test modules
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m property      # Property-based tests only
```

### Code Quality

```bash
# Format code
black app tests

# Sort imports
isort app tests

# Type checking
mypy app

# Linting
flake8 app tests
```

## Configuration

Key configuration options in `.env`:

### Database
- `MONGODB_URL`: MongoDB connection string
- `MONGODB_DATABASE`: Database name
- `REDIS_URL`: Redis connection string

### AI Services
- `GOOGLE_API_KEY`: Google AI API key for Gemini
- `OPENAI_API_KEY`: OpenAI API key (optional)

### Security
- `SECRET_KEY`: JWT signing key
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time

### Rate Limiting
- `RATE_LIMIT_PER_MINUTE`: Requests per minute per user
- `RATE_LIMIT_BURST`: Burst capacity

### Cost Control
- `DEFAULT_DAILY_TOKEN_LIMIT`: Default daily token limit per user
- `DEFAULT_MONTHLY_COST_LIMIT`: Default monthly cost limit per user

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Content Generation (Coming Soon)
```bash
curl -X POST http://localhost:8000/api/v1/content/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "type": "text",
    "operation": "blog_post",
    "parameters": {
      "topic": "AI in content creation",
      "length": "medium",
      "tone": "professional"
    }
  }'
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format code (`black app tests`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the test suite for usage examples
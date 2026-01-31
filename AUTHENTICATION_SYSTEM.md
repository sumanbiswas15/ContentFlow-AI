# Authentication and Authorization System

## Overview

The ContentFlow AI authentication and authorization system provides comprehensive security features including:

- **API Key Authentication**: Secure API key validation with hashing
- **Rate Limiting**: Configurable rate limits with sliding window algorithm
- **Security Monitoring**: Real-time monitoring of authentication attempts
- **Suspicious Activity Detection**: Automatic detection and blocking of malicious activity
- **Permission-Based Authorization**: Fine-grained access control with role-based permissions

## Requirements Implemented

This system implements the following requirements from the ContentFlow AI specification:

- **Requirement 9.1**: API request authentication and authorization
- **Requirement 9.2**: Rate limiting with appropriate error messages
- **Requirement 9.3**: Suspicious activity detection and security measures
- **Requirement 9.4**: API key validation and permission tracking
- **Requirement 9.5**: Security event logging for audit and monitoring

## Architecture

### Components

1. **AuthService** (`app/services/auth_service.py`)
   - Main authentication and authorization service
   - Coordinates rate limiting and security monitoring
   - Provides unified API for authentication operations

2. **RateLimiter**
   - Implements sliding window rate limiting algorithm
   - Tracks requests per identifier (user, API key, IP)
   - Automatic cleanup of old entries
   - Configurable limits and time windows

3. **SecurityMonitor**
   - Records failed authentication attempts
   - Detects suspicious activity patterns
   - Automatically blocks IPs after excessive failures
   - Provides security event logging and reporting

4. **AuthenticationMiddleware** (`app/middleware/auth_middleware.py`)
   - FastAPI middleware for request authentication
   - Validates API keys on every request
   - Adds rate limit headers to responses
   - Handles authentication errors gracefully

5. **API Dependencies** (`app/api/dependencies.py`)
   - Dependency injection for authentication
   - Permission checking utilities
   - Usage limit validation

## Usage

### API Key Authentication

All API requests (except public endpoints) require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: cf_your_api_key_here" \
     https://api.contentflow.ai/api/v1/content
```

### Creating API Keys

```python
POST /api/v1/auth/api-keys
{
  "name": "Production API Key",
  "permissions": ["content:read", "content:create"],
  "expires_in_days": 90
}
```

Response includes the full API key (only shown once):
```json
{
  "key_id": "key_1234567890",
  "name": "Production API Key",
  "permissions": ["content:read", "content:create"],
  "api_key": "cf_abc123...",
  "created_at": "2024-01-01T00:00:00Z",
  "expires_at": "2024-04-01T00:00:00Z"
}
```

### Rate Limiting

Rate limits are enforced per API key and per IP address:

- **API Key Limit**: 60 requests per minute (configurable)
- **IP Address Limit**: 120 requests per minute (configurable)

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 30
```

When rate limit is exceeded, the API returns:
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Try again in 30 seconds."
}
```

### Permissions

Available permissions:
- `content:read` - Read content
- `content:create` - Create new content
- `content:update` - Update existing content
- `content:delete` - Delete content
- `analytics:read` - View analytics
- `admin` - Full administrative access

### Security Monitoring

The system automatically monitors for suspicious activity:

1. **Failed Authentication Tracking**
   - Records all failed authentication attempts
   - Tracks by user identifier and IP address

2. **Brute Force Detection**
   - Marks activity as suspicious after 3 failures in 15 minutes
   - Logs security events for investigation

3. **Automatic IP Blocking**
   - Blocks IP addresses after 10 failed attempts in 15 minutes
   - Prevents further requests from blocked IPs
   - Admin can manually unblock IPs

4. **Security Event Logging**
   - All security events are logged with severity levels
   - Includes authentication, authorization, and rate limit events
   - Provides audit trail for compliance

## API Endpoints

### Authentication Endpoints

#### Get Current User Info
```
GET /api/v1/auth/me
```
Returns authenticated user information, API key details, and rate limit status.

#### Create API Key
```
POST /api/v1/auth/api-keys
```
Creates a new API key for the authenticated user.

#### List API Keys
```
GET /api/v1/auth/api-keys
```
Lists all API keys for the authenticated user.

#### Revoke API Key
```
DELETE /api/v1/auth/api-keys/{key_id}
```
Revokes (deactivates) an API key.

#### Get Rate Limit Status
```
GET /api/v1/auth/rate-limit
```
Returns current rate limit status for the authenticated API key.

#### Get Usage Statistics
```
GET /api/v1/auth/usage
```
Returns usage statistics and limits for the authenticated user.

### Admin Endpoints (Require Admin Permission)

#### Get Security Report
```
GET /api/v1/auth/security/report
```
Returns comprehensive security report including suspicious activity.

#### Unblock IP Address
```
POST /api/v1/auth/security/unblock-ip
{
  "ip_address": "192.168.1.100"
}
```
Manually unblocks an IP address.

#### Get Suspicious Activity
```
GET /api/v1/auth/security/suspicious-activity
```
Returns detailed suspicious activity report.

#### Reset Rate Limit
```
POST /api/v1/auth/security/reset-rate-limit/{identifier}
```
Manually resets rate limit for a specific identifier.

## Configuration

Configuration is managed through environment variables in `.env`:

```env
# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10

# Cost Control
DEFAULT_DAILY_TOKEN_LIMIT=100000
DEFAULT_MONTHLY_COST_LIMIT=100.0
```

## Testing

The authentication system includes comprehensive unit tests:

```bash
# Run all authentication tests
pytest tests/test_auth_service.py -v

# Run specific test class
pytest tests/test_auth_service.py::TestRateLimiter -v

# Run with coverage
pytest tests/test_auth_service.py --cov=app/services/auth_service
```

Test coverage includes:
- Rate limiting with sliding window
- API key validation and expiration
- Permission checking
- Security monitoring and IP blocking
- Concurrent request handling
- Edge cases and error conditions

## Security Best Practices

1. **API Key Storage**
   - API keys are hashed using SHA-256 before storage
   - Full keys are only shown once during creation
   - Store keys securely (environment variables, secrets manager)

2. **Rate Limiting**
   - Implement both API key and IP-based rate limiting
   - Use appropriate limits based on user tier
   - Monitor rate limit violations

3. **Security Monitoring**
   - Review security reports regularly
   - Investigate suspicious activity promptly
   - Keep audit logs for compliance

4. **Permission Management**
   - Follow principle of least privilege
   - Use specific permissions instead of admin
   - Regularly audit API key permissions

5. **IP Blocking**
   - Monitor blocked IPs for false positives
   - Implement manual unblock process
   - Consider geographic restrictions if needed

## Error Handling

The system provides clear error messages for different scenarios:

### Authentication Errors (401)
```json
{
  "error": "AUTHENTICATION_ERROR",
  "message": "Invalid API key"
}
```

### Authorization Errors (403)
```json
{
  "error": "AUTHORIZATION_ERROR",
  "message": "API key does not have required permission: content:create"
}
```

### Rate Limit Errors (429)
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Try again in 30 seconds."
}
```

### Blocked IP (403)
```json
{
  "error": "BLOCKED",
  "message": "Access denied from this IP address due to suspicious activity."
}
```

## Performance Considerations

1. **In-Memory Rate Limiting**
   - Fast lookups with O(1) complexity
   - Automatic cleanup of old entries
   - Suitable for single-instance deployments

2. **Distributed Deployments**
   - For multi-instance deployments, consider Redis-based rate limiting
   - Shared state across instances
   - Persistent rate limit data

3. **Database Queries**
   - API key validation queries are optimized with indexes
   - Consider caching user/API key data
   - Use connection pooling for database access

## Future Enhancements

Potential improvements for the authentication system:

1. **OAuth2 Support**: Add OAuth2 authentication flow
2. **JWT Tokens**: Implement JWT-based authentication
3. **Redis Integration**: Use Redis for distributed rate limiting
4. **API Key Rotation**: Automatic key rotation policies
5. **Multi-Factor Authentication**: Add MFA for sensitive operations
6. **Webhook Notifications**: Alert on security events
7. **Geographic Restrictions**: IP-based geographic access control
8. **Advanced Analytics**: Detailed usage analytics and reporting

## Troubleshooting

### Common Issues

**Issue**: API key not working
- Verify key is active and not expired
- Check permissions match required operation
- Ensure key is properly formatted in header

**Issue**: Rate limit exceeded
- Check current usage with `/auth/rate-limit`
- Wait for rate limit window to reset
- Contact admin to increase limits if needed

**Issue**: IP blocked
- Contact admin to unblock IP
- Review security logs for cause
- Ensure legitimate traffic patterns

**Issue**: Permission denied
- Verify API key has required permission
- Check permission spelling and format
- Request permission update from admin

## Support

For issues or questions about the authentication system:
- Review this documentation
- Check security logs for detailed error information
- Contact system administrators for access issues
- Submit bug reports with detailed reproduction steps

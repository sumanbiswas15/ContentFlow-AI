# ContentFlow AI - Deployment Guide

This guide covers deploying ContentFlow AI to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [Production Checklist](#production-checklist)
- [Monitoring](#monitoring)
- [Backup & Recovery](#backup--recovery)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or Docker-compatible system
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ SSD
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

### Required Services

- MongoDB 7.0+
- Redis 7.2+
- Nginx (for reverse proxy)

### API Keys

- Google Gemini API key (required for AI features)

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/contentflow-ai.git
cd contentflow-ai
```

### 2. Configure Environment Variables

Create a `.env.production` file:

```bash
cp .env.example .env.production
```

Edit `.env.production` with production values:

```env
# Database
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=<strong-password>
DATABASE_NAME=contentflow_ai

# Redis
REDIS_PASSWORD=<strong-password>

# Security
SECRET_KEY=<generate-strong-secret-key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Google AI
GOOGLE_API_KEY=<your-gemini-api-key>

# Server
WORKERS=4
RATE_LIMIT_PER_MINUTE=100

# Environment
ENVIRONMENT=production
DEBUG=false
```

### 3. Generate Strong Secrets

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate database passwords
openssl rand -base64 32
```

## Docker Deployment

### Development Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

#### 1. Build Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

#### 2. Deploy with Docker Compose

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f backend
```

#### 3. Verify Deployment

```bash
# Check backend health
curl http://localhost/health

# Check API
curl http://localhost/api/v1/health

# Check MongoDB
docker exec contentflow-mongodb-prod mongosh --eval "db.adminCommand('ping')"

# Check Redis
docker exec contentflow-redis-prod redis-cli ping
```

### SSL/TLS Configuration

#### Using Let's Encrypt (Recommended)

1. Install Certbot:
```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx
```

2. Obtain certificate:
```bash
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

3. Update nginx configuration to use SSL (uncomment HTTPS server block in `nginx/nginx.conf`)

4. Restart nginx:
```bash
docker-compose -f docker-compose.prod.yml restart nginx
```

## Production Checklist

### Security

- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable security headers
- [ ] Disable debug mode
- [ ] Review CORS settings
- [ ] Set up API key rotation
- [ ] Configure MongoDB authentication
- [ ] Enable Redis password protection

### Performance

- [ ] Configure worker processes
- [ ] Set up caching (Redis)
- [ ] Enable gzip compression
- [ ] Optimize database indexes
- [ ] Configure connection pooling
- [ ] Set up CDN for static assets
- [ ] Enable HTTP/2
- [ ] Configure load balancing

### Monitoring

- [ ] Set up logging aggregation
- [ ] Configure health checks
- [ ] Set up uptime monitoring
- [ ] Configure error tracking
- [ ] Set up performance monitoring
- [ ] Configure alerts
- [ ] Set up backup monitoring

### Backup

- [ ] Configure automated MongoDB backups
- [ ] Set up storage volume backups
- [ ] Test restore procedures
- [ ] Document backup locations
- [ ] Set up off-site backups

## Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost/health

# MongoDB health
docker exec contentflow-mongodb-prod mongosh --eval "db.serverStatus()"

# Redis health
docker exec contentflow-redis-prod redis-cli info
```

### Logs

```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs

# View specific service logs
docker-compose -f docker-compose.prod.yml logs backend
docker-compose -f docker-compose.prod.yml logs mongodb
docker-compose -f docker-compose.prod.yml logs redis

# Follow logs in real-time
docker-compose -f docker-compose.prod.yml logs -f --tail=100
```

### Metrics

Monitor these key metrics:

- **API Response Time**: < 200ms average
- **Error Rate**: < 1%
- **CPU Usage**: < 70%
- **Memory Usage**: < 80%
- **Disk Usage**: < 80%
- **Database Connections**: Monitor pool usage
- **Cache Hit Rate**: > 80%

### Recommended Monitoring Tools

- **Prometheus** + **Grafana**: Metrics and dashboards
- **ELK Stack**: Log aggregation and analysis
- **Sentry**: Error tracking
- **UptimeRobot**: Uptime monitoring
- **New Relic** / **Datadog**: APM

## Backup & Recovery

### MongoDB Backup

#### Automated Backup Script

Create `scripts/backup-mongodb.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/backups/mongodb"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="contentflow_backup_$TIMESTAMP"

# Create backup
docker exec contentflow-mongodb-prod mongodump \
  --out=/backups/$BACKUP_NAME \
  --username=$MONGO_ROOT_USERNAME \
  --password=$MONGO_ROOT_PASSWORD \
  --authenticationDatabase=admin

# Compress backup
tar -czf $BACKUP_DIR/$BACKUP_NAME.tar.gz -C /backups $BACKUP_NAME

# Remove old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_NAME.tar.gz"
```

#### Schedule with Cron

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /path/to/scripts/backup-mongodb.sh
```

### MongoDB Restore

```bash
# Extract backup
tar -xzf contentflow_backup_TIMESTAMP.tar.gz

# Restore
docker exec -i contentflow-mongodb-prod mongorestore \
  --username=$MONGO_ROOT_USERNAME \
  --password=$MONGO_ROOT_PASSWORD \
  --authenticationDatabase=admin \
  /backups/contentflow_backup_TIMESTAMP
```

### Storage Backup

```bash
# Backup storage directory
tar -czf storage_backup_$(date +%Y%m%d).tar.gz storage/

# Restore storage
tar -xzf storage_backup_YYYYMMDD.tar.gz
```

## Scaling

### Horizontal Scaling

#### 1. Load Balancer Setup

Use Nginx or cloud load balancer to distribute traffic across multiple backend instances.

#### 2. Multiple Backend Instances

Update `docker-compose.prod.yml`:

```yaml
backend:
  deploy:
    replicas: 3
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

#### 3. Database Replication

Set up MongoDB replica set for high availability:

```yaml
mongodb:
  command: mongod --replSet rs0
```

### Vertical Scaling

Increase resources for existing containers:

```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
```

### Caching Strategy

- Use Redis for session storage
- Cache API responses
- Implement CDN for static assets
- Use browser caching headers

## Troubleshooting

### Common Issues

#### 1. Backend Won't Start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs backend

# Common causes:
# - Missing environment variables
# - Database connection issues
# - Port conflicts
```

#### 2. Database Connection Errors

```bash
# Test MongoDB connection
docker exec contentflow-mongodb-prod mongosh \
  --username=$MONGO_ROOT_USERNAME \
  --password=$MONGO_ROOT_PASSWORD

# Check network
docker network inspect contentflow-network
```

#### 3. High Memory Usage

```bash
# Check container stats
docker stats

# Restart services
docker-compose -f docker-compose.prod.yml restart

# Adjust worker count in .env
WORKERS=2
```

#### 4. Slow API Responses

- Check database indexes
- Monitor Redis cache hit rate
- Review slow query logs
- Optimize AI engine calls
- Increase worker processes

### Debug Mode

Enable debug logging temporarily:

```bash
# Set in .env
DEBUG=true
LOG_LEVEL=DEBUG

# Restart backend
docker-compose -f docker-compose.prod.yml restart backend
```

### Performance Profiling

```bash
# Install profiling tools
pip install py-spy

# Profile running application
py-spy top --pid <backend-pid>
```

## Maintenance

### Regular Tasks

- **Daily**: Check logs for errors
- **Weekly**: Review performance metrics
- **Monthly**: Update dependencies
- **Quarterly**: Security audit
- **Annually**: Disaster recovery drill

### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose -f docker-compose.prod.yml build

# Deploy with zero downtime
docker-compose -f docker-compose.prod.yml up -d --no-deps --build backend
```

### Database Maintenance

```bash
# Compact database
docker exec contentflow-mongodb-prod mongosh --eval "db.runCommand({compact: 'content_items'})"

# Rebuild indexes
docker exec contentflow-mongodb-prod mongosh --eval "db.content_items.reIndex()"
```

## Support

For deployment issues:

- Check logs first
- Review this guide
- Search GitHub issues
- Contact support team

---

**Last Updated**: 2024
**Version**: 1.0.0

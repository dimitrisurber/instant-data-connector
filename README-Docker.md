# Docker Deployment Guide

This guide covers the Docker-based deployment options for the PostgreSQL FDW-based Data Connector.

## Quick Start

### Development Environment

```bash
# Start development environment
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f connector-app

# Stop services
docker-compose down
```

### Production Environment

```bash
# Build and deploy to production
./scripts/deploy.sh --environment production --tag v1.0.0

# Or using docker-compose for production
docker-compose -f docker-compose.prod.yml up -d
```

## Architecture

### Components

1. **Application Container** (`connector-app`)
   - Multi-stage build (development/production)
   - Health checks and monitoring
   - Non-root user for security

2. **PostgreSQL with FDW** (`postgres-fdw`)
   - Pre-installed FDW extensions
   - Custom initialization scripts
   - SSL/TLS configuration

3. **Redis** (`redis`)
   - Caching and task queue
   - Persistence configuration
   - Memory optimization

4. **NGINX** (`nginx`)
   - Reverse proxy and load balancer
   - SSL termination
   - Rate limiting

5. **Monitoring Stack**
   - Prometheus for metrics
   - Grafana for visualization
   - AlertManager for notifications

## Configuration

### Environment Variables

#### Application
- `ENVIRONMENT`: deployment environment (development/staging/production)
- `DEBUG`: enable debug mode (true/false)
- `LOG_LEVEL`: logging level (DEBUG/INFO/WARN/ERROR)
- `SECRET_KEY`: application secret key
- `ENCRYPTION_KEY`: data encryption key
- `JWT_SECRET`: JWT signing secret

#### Database
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port (default: 5432)
- `POSTGRES_DB`: database name
- `POSTGRES_USER`: database user
- `POSTGRES_PASSWORD`: database password

#### Redis
- `REDIS_HOST`: Redis host
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password

#### Monitoring
- `PROMETHEUS_ENDPOINT`: Prometheus metrics endpoint
- `GRAFANA_ADMIN_PASSWORD`: Grafana admin password
- `ALERT_WEBHOOK`: Slack/Teams webhook for alerts
- `ALERT_EMAIL`: email for notifications

### Volume Mounts

#### Development
- Source code mounted for hot reload
- Database data persisted
- Logs accessible from host

#### Production
- Data volumes for persistence
- Backup volumes for database dumps
- Log volumes for centralized logging

## Services

### Application Service
- **Port**: 8000
- **Health Check**: `/health`
- **Metrics**: `/metrics`
- **API Documentation**: `/docs`

### Database Service
- **Port**: 5432
- **Health Check**: `pg_isready`
- **Admin Interface**: Adminer on port 8080 (dev only)

### Redis Service
- **Port**: 6379
- **Health Check**: `redis-cli ping`
- **Admin Interface**: Redis Commander on port 8081 (dev only)

### Monitoring Services
- **Prometheus**: Port 9090
- **Grafana**: Port 3000
- **AlertManager**: Port 9093

## Health Checks

### Application Health Check
```bash
curl http://localhost:8000/health
```

### Database Health Check
```bash
docker exec postgres-fdw-container pg_isready -U connector_user -d instant_connector
```

### Redis Health Check
```bash
docker exec redis-container redis-cli ping
```

### System Health Check
```bash
./scripts/health_check.sh --once
```

## Backup and Recovery

### Database Backup
```bash
# Manual backup
./scripts/backup.sh --password your_password

# Automated backup (production)
./scripts/backup.sh --auto-backup --s3-bucket your-backup-bucket
```

### Database Restore
```bash
# Restore from backup
./scripts/restore.sh --backup-file /path/to/backup.sql.gz --password your_password

# Restore from S3
./scripts/restore.sh --s3-bucket your-backup-bucket --s3-key backup-file.sql.gz
```

## Scaling

### Horizontal Scaling
```bash
# Scale application containers
docker-compose up -d --scale connector-app=3

# Scale Celery workers
docker-compose up -d --scale celery-worker=5
```

### Resource Limits
Configured in docker-compose files:
- CPU limits and reservations
- Memory limits and reservations
- Disk I/O limits

## Security

### Container Security
- Non-root users
- Read-only root filesystem where possible
- Security options (no-new-privileges)
- Capability dropping

### Network Security
- Custom bridge networks
- Service isolation
- Port exposure minimization

### Data Security
- Encrypted environment variables
- SSL/TLS for all connections
- Secret management

## Monitoring and Alerting

### Metrics Collection
- Application metrics via Prometheus
- System metrics via Node Exporter
- Database metrics via PostgreSQL Exporter
- Redis metrics via Redis Exporter

### Dashboards
- Overview dashboard for system health
- Database performance dashboard
- Application metrics dashboard
- FDW-specific metrics dashboard

### Alerts
- Service availability alerts
- Performance threshold alerts
- Resource utilization alerts
- Error rate alerts

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
docker-compose logs connector-app

# Check health
curl http://localhost:8000/health

# Verify environment variables
docker-compose exec connector-app env | grep -E "(DATABASE|REDIS|SECRET)"
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose exec postgres-fdw pg_isready

# Check FDW extensions
docker-compose exec postgres-fdw psql -U connector_user -d instant_connector -c "\\dx"

# Test connection from application
docker-compose exec connector-app python -c "import asyncpg; print('Connection test')"
```

#### Redis Connection Issues
```bash
# Check Redis status
docker-compose exec redis redis-cli ping

# Check memory usage
docker-compose exec redis redis-cli info memory

# Check connections
docker-compose exec redis redis-cli info clients
```

### Debugging Commands

```bash
# Enter application container
docker-compose exec connector-app bash

# Check application processes
docker-compose exec connector-app ps aux

# Monitor resource usage
docker stats

# Check container health
docker inspect --format='{{.State.Health.Status}}' container_name
```

### Log Analysis

```bash
# Application logs
docker-compose logs -f connector-app

# Database logs
docker-compose logs -f postgres-fdw

# All service logs
docker-compose logs -f

# Filter logs by level
docker-compose logs connector-app | grep ERROR
```

## Performance Tuning

### Application Tuning
- Adjust worker processes
- Configure connection pooling
- Optimize query caching
- Enable compression

### Database Tuning
- PostgreSQL configuration optimization
- Connection pool sizing
- Query optimization
- Index management

### Redis Tuning
- Memory allocation
- Persistence settings
- Eviction policies
- Connection limits

### Container Tuning
- Resource limits
- Restart policies
- Health check intervals
- Log rotation

## Development Workflow

### Local Development
```bash
# Start development environment
docker-compose up -d

# Make code changes (hot reload enabled)
# Run tests
docker-compose exec connector-app python -m pytest

# Check code quality
docker-compose exec connector-app black --check src/
docker-compose exec connector-app flake8 src/
```

### Testing
```bash
# Run all tests
docker-compose exec connector-app python -m pytest tests/

# Run specific test
docker-compose exec connector-app python -m pytest tests/test_fdw.py

# Run with coverage
docker-compose exec connector-app python -m pytest --cov=instant_connector tests/
```

### Database Migrations
```bash
# Run migrations
docker-compose exec connector-app python -m alembic upgrade head

# Create new migration
docker-compose exec connector-app python -m alembic revision --autogenerate -m "description"
```

## Production Deployment

### Pre-deployment Checklist
- [ ] Update version tags
- [ ] Run full test suite
- [ ] Update configuration
- [ ] Backup production database
- [ ] Verify SSL certificates
- [ ] Check resource limits
- [ ] Update monitoring dashboards

### Deployment Process
```bash
# Deploy using automation script
./scripts/deploy.sh --environment production --tag v1.2.3

# Or manual deployment
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --no-deps connector-app
```

### Post-deployment Verification
```bash
# Check service health
./scripts/health_check.sh --environment production

# Verify application functionality
curl https://api.yourapp.com/health

# Check monitoring dashboards
# - Grafana: https://grafana.yourapp.com
# - Prometheus: https://prometheus.yourapp.com
```

## Support

For issues and questions:
1. Check logs using the debugging commands above
2. Review monitoring dashboards
3. Run health checks
4. Check the troubleshooting section
5. Contact the development team

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Documentation](https://hub.docker.com/_/postgres)
- [Redis Docker Documentation](https://hub.docker.com/_/redis)
- [Prometheus Docker Documentation](https://hub.docker.com/r/prom/prometheus)
- [Grafana Docker Documentation](https://hub.docker.com/r/grafana/grafana)
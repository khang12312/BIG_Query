# 🚀 Advanced AI-Powered Resume Matcher - Deployment Guide

## 📋 Overview

This guide covers the complete deployment of the advanced AI-powered resume matching system with enterprise-grade features including:

- **Advanced ML Features**: Ensemble matching, skill extraction, experience scoring
- **Modern Web Interface**: Real-time dashboard with WebSocket support
- **REST API**: Comprehensive API with authentication and rate limiting
- **Security Features**: Data encryption, GDPR compliance, audit logging
- **Production Deployment**: Docker containers, CI/CD pipeline, monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface  │    │   REST API      │    │  Background     │
│   (Flask + WS)   │    │   (Flask)       │    │  Worker         │
│   Port: 5000     │    │   Port: 5001    │    │  (Celery)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   (Cache/Queue) │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Google         │
                    │   BigQuery AI    │
                    └─────────────────┘
```

## 🛠️ Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **RAM**: Minimum 8GB, Recommended 16GB+
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Storage**: Minimum 50GB free space
- **Network**: Stable internet connection

### Software Requirements
- **Python**: 3.9 or higher
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: Latest version
- **Google Cloud SDK**: Latest version

### Google Cloud Setup
1. **Create Google Cloud Project**
   ```bash
   gcloud projects create your-project-id
   gcloud config set project your-project-id
   ```

2. **Enable Required APIs**
   ```bash
   gcloud services enable bigquery.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Create Service Account**
   ```bash
   gcloud iam service-accounts create resume-matcher-sa \
     --display-name="Resume Matcher Service Account"
   
   gcloud projects add-iam-policy-binding your-project-id \
     --member="serviceAccount:resume-matcher-sa@your-project-id.iam.gserviceaccount.com" \
     --role="roles/bigquery.admin"
   
   gcloud iam service-accounts keys create credentials.json \
     --iam-account=resume-matcher-sa@your-project-id.iam.gserviceaccount.com
   ```

## 📦 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-powered-resume-matcher
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Environment Variables:**
```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
GOOGLE_CLOUD_LOCATION=US

# Application Configuration
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
API_SECRET_KEY=your-api-secret-key

# Database Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Email Configuration (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Monitoring (Optional)
PROMETHEUS_ENDPOINT=http://localhost:9090
GRAFANA_ENDPOINT=http://localhost:3000
```

## 🐳 Docker Deployment

### 1. Build Images
```bash
# Build all services
docker-compose build

# Or build specific service
docker-compose build web
```

### 2. Start Services
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 3. Initialize Database
```bash
# Run initialization script
docker-compose exec web python main.py
```

## ☁️ Cloud Deployment

### Google Cloud Run (Recommended)

1. **Build and Push Images**
   ```bash
   # Configure Docker for GCR
   gcloud auth configure-docker
   
   # Build and push
   docker build -t gcr.io/your-project-id/resume-matcher .
   docker push gcr.io/your-project-id/resume-matcher
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy resume-matcher \
     --image gcr.io/your-project-id/resume-matcher \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --max-instances 10
   ```

### Google Kubernetes Engine (GKE)

1. **Create Cluster**
   ```bash
   gcloud container clusters create resume-matcher-cluster \
     --num-nodes 3 \
     --machine-type e2-standard-4 \
     --zone us-central1-a
   ```

2. **Deploy Application**
   ```bash
   # Apply Kubernetes manifests
   kubectl apply -f k8s/
   
   # Check deployment status
   kubectl get pods
   kubectl get services
   ```

## 🔧 Configuration

### BigQuery Setup
```python
# Initialize BigQuery dataset and tables
from src.bigquery_client import BigQueryAIClient

client = BigQueryAIClient()
client.create_dataset_if_not_exists()
client.create_tables()
```

### Redis Configuration
```bash
# Redis configuration for production
redis-server --port 6379 --maxmemory 2gb --maxmemory-policy allkeys-lru
```

### Nginx Configuration
```nginx
# nginx.conf
upstream web_backend {
    server web:5000;
}

upstream api_backend {
    server api:5001;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://web_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/ {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update docker-compose.yml
volumes:
  - ./ssl:/etc/nginx/ssl:ro
```

### Environment Security
```bash
# Set secure environment variables
export SECRET_KEY=$(openssl rand -hex 32)
export API_SECRET_KEY=$(openssl rand -hex 32)
export JWT_SECRET=$(openssl rand -hex 32)
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'resume-matcher'
    static_configs:
      - targets: ['web:5000', 'api:5001']
```

### Grafana Dashboards
```bash
# Import dashboards
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @grafana/dashboard.json
```

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/ -v

# Test API endpoints
pytest tests/api/ -v
```

### Load Testing
```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.40.0/k6-v0.40.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1

# Run load tests
k6 run performance-tests/load-test.js
```

## 🚀 CI/CD Pipeline

### GitHub Actions Setup
1. **Set Repository Secrets**
   - `GOOGLE_CLOUD_PROJECT_ID`
   - `GOOGLE_APPLICATION_CREDENTIALS`
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`

2. **Workflow Triggers**
   - Push to `main` branch → Production deployment
   - Push to `develop` branch → Staging deployment
   - Pull requests → Testing and validation

### Deployment Commands
```bash
# Manual deployment
./scripts/deploy.sh production

# Rollback deployment
./scripts/rollback.sh production
```

## 🔍 Troubleshooting

### Common Issues

1. **BigQuery Connection Error**
   ```bash
   # Check credentials
   gcloud auth application-default print-access-token
   
   # Verify project access
   gcloud projects describe your-project-id
   ```

2. **Redis Connection Error**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   docker-compose up -d --scale web=2
   
   # Monitor memory usage
   docker stats
   ```

### Log Analysis
```bash
# View application logs
docker-compose logs -f web

# View API logs
docker-compose logs -f api

# View worker logs
docker-compose logs -f worker
```

## 📈 Performance Optimization

### Scaling Configuration
```yaml
# docker-compose.yml
services:
  web:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Caching Strategy
```python
# Redis caching configuration
CACHE_CONFIG = {
    'default_timeout': 300,
    'key_prefix': 'resume_matcher:',
    'serializer': 'json'
}
```

### Database Optimization
```sql
-- BigQuery optimization
CREATE OR REPLACE TABLE `project.dataset.resumes_optimized`
PARTITION BY DATE(created_at)
CLUSTER BY skills, location
AS SELECT * FROM `project.dataset.resumes`;
```

## 🔄 Backup and Recovery

### Data Backup
```bash
# Backup BigQuery data
bq extract project:dataset.resumes gs://backup-bucket/resumes-backup.json

# Backup Redis data
redis-cli --rdb /backup/redis-backup.rdb
```

### Disaster Recovery
```bash
# Restore from backup
bq load project:dataset.resumes gs://backup-bucket/resumes-backup.json

# Restore Redis
redis-cli --pipe < /backup/redis-backup.rdb
```

## 📚 Additional Resources

- [Google BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Docker Documentation](https://docs.docker.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Redis Documentation](https://redis.io/documentation)

## 🆘 Support

For technical support:
- **Documentation**: Check this guide and inline code comments
- **Issues**: Create GitHub issues for bugs and feature requests
- **Community**: Join our Discord server for community support

---

**🎉 Congratulations!** Your advanced AI-powered resume matching system is now ready for production deployment!

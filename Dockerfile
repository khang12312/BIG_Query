# Multi-stage Dockerfile for AI-Powered Resume Matcher
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with error handling
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt || \
    (echo "Some packages failed to install, continuing..." && \
     pip install --no-cache-dir --no-deps -r requirements.txt)

# Install additional production dependencies (already in requirements.txt)
# RUN pip install --no-cache-dir gunicorn flask-limiter PyJWT cryptography

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose ports
EXPOSE 5000 5001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || curl -f http://localhost:5000/ || exit 1

# Create startup script
RUN echo '#!/bin/bash\nset -e\necho "Starting AI-Powered Resume Matcher..."\nif [ -f "web_app/app.py" ]; then\n    exec gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 web_app.app:app\nelse\n    exec python main.py\nfi' > /app/start.sh && \
    chmod +x /app/start.sh

# Default command with fallback
CMD ["/app/start.sh"]

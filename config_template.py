"""
Configuration template for AI-Powered Resume Matcher
Copy this file to config.py and update with your specific values
"""

import os
from typing import Dict, Any

def get_config(section: str) -> Dict[str, Any]:
    """Get configuration for a specific section"""
    
    configs = {
        'bigquery': {
            'project_id': os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id'),
            'dataset_id': 'resume_matcher_ai',
            'location': 'us-central1',  # or your preferred region
            'credentials_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        },
        
        'tables': {
            'resumes': 'resumes',
            'jobs': 'jobs',
            'resume_embeddings': 'resume_embeddings',
            'job_embeddings': 'job_embeddings',
            'matches': 'matches',
            'feedback': 'feedback'
        },
        
        'model': {
            'embedding_model': 'text-embedding-004',  # Latest Google embedding model
            'generation_model': 'gemini-1.5-pro-002',  # Latest Gemini model
            'max_tokens': 1024,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40
        },
        
        'matching': {
            'similarity_threshold': 0.7,
            'max_matches_per_job': 10,
            'max_jobs_per_candidate': 5,
            'batch_size': 100,
            'use_ensemble': True
        },
        
        'security': {
            'secret_key': os.getenv('SECRET_KEY', 'your-secret-key-here'),
            'jwt_secret': os.getenv('JWT_SECRET', 'your-jwt-secret-here'),
            'session_timeout': 3600,  # 1 hour
            'max_login_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'password_min_length': 8
        },
        
        'api': {
            'rate_limit_per_minute': 60,
            'rate_limit_per_hour': 1000,
            'rate_limit_per_day': 10000,
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'allowed_extensions': ['.pdf', '.docx', '.txt'],
            'cors_origins': ['http://localhost:3000', 'https://yourdomain.com']
        },
        
        'redis': {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'db': int(os.getenv('REDIS_DB', 0)),
            'password': os.getenv('REDIS_PASSWORD'),
            'socket_timeout': 5,
            'connection_pool_max_connections': 20
        },
        
        'logging': {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            'file': 'resume_matcher.log',
            'max_bytes': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5
        },
        
        'datasets': {
            'jobs_path': 'src/DataSets/JobsSample',
            'resumes_path': 'src/DataSets/ResumeSample',
            'sample_size_jobs': 200,
            'sample_size_resumes': 100,
            'encoding': 'utf-8'
        },
        
        'web': {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': os.getenv('FLASK_ENV') == 'development',
            'threaded': True,
            'session_cookie_secure': True,
            'session_cookie_httponly': True,
            'session_cookie_samesite': 'Lax'
        },
        
        'monitoring': {
            'prometheus_port': 9090,
            'grafana_port': 3000,
            'health_check_interval': 30,
            'metrics_retention': '7d'
        }
    }
    
    return configs.get(section, {})

# Environment-specific overrides
def get_environment_config():
    """Get environment-specific configuration overrides"""
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return {
            'web': {
                'debug': False,
                'host': '0.0.0.0',
                'port': int(os.getenv('PORT', 5000))
            },
            'logging': {
                'level': 'WARNING'
            },
            'security': {
                'session_cookie_secure': True
            }
        }
    elif env == 'testing':
        return {
            'bigquery': {
                'dataset_id': 'resume_matcher_test'
            },
            'datasets': {
                'sample_size_jobs': 10,
                'sample_size_resumes': 5
            }
        }
    
    return {}

# Validation functions
def validate_config():
    """Validate configuration settings"""
    required_env_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_PROJECT'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
    
    # Validate credentials file exists
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_path and not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
    
    return True

# Usage example:
# from config import get_config, validate_config
# 
# # Validate configuration
# validate_config()
# 
# # Get specific configuration sections
# bigquery_config = get_config('bigquery')
# model_config = get_config('model')
# security_config = get_config('security')

"""
Configuration settings for AI-Powered Resume & Job Matcher
"""

import os
from typing import Dict, Any

# BigQuery Configuration
BIGQUERY_CONFIG = {
    'project_id': os.getenv('GOOGLE_CLOUD_PROJECT', 'divine-catalyst-459423-j5'),
    'dataset_id': 'resume_matcher',
    'location': 'us-central1',  # Changed to specific region for AI Platform
    'credentials_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
}

# Table Names
TABLES = {
    'resumes': 'resumes',
    'jobs': 'job_descriptions', 
    'resume_embeddings': 'resume_embeddings',
    'job_embeddings': 'job_embeddings',
    'matches': 'candidate_matches',
    'feedback': 'candidate_feedback'
}

# Model Configuration
MODEL_CONFIG = {
    'max_tokens': 1024,
    'temperature': 0.7
}

# Matching Configuration
MATCHING_CONFIG = {
    'similarity_threshold': 0.7,
    'max_matches_per_job': 10,
    'max_jobs_per_candidate': 5,
    'vector_search_neighbors': 20
}

# Data Processing Configuration
PROCESSING_CONFIG = {
    'max_text_length': 10000,
    'min_text_length': 100,
    'supported_formats': ['.pdf', '.docx', '.txt'],
    'chunk_size': 1000,
    'chunk_overlap': 200
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'color_palette': 'viridis',
    'dpi': 300,
    'style': 'seaborn-v0_8'
}

def get_config(section: str) -> Dict[str, Any]:
    """Get configuration for a specific section"""
    configs = {
        'bigquery': BIGQUERY_CONFIG,
        'tables': TABLES,
        'model': MODEL_CONFIG,
        'matching': MATCHING_CONFIG,
        'processing': PROCESSING_CONFIG,
        'visualization': VIZ_CONFIG
    }
    return configs.get(section, {})

"""
Pytest configuration and fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def src_path():
    """Return the src directory path"""
    return Path(__file__).parent.parent / 'src'

@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return """
    John Doe
    Software Engineer
    
    Experience:
    - 5 years of Python development
    - Machine Learning expertise
    - BigQuery and cloud technologies
    
    Skills: Python, SQL, Machine Learning, BigQuery, Docker
    
    Education: BS Computer Science
    """

@pytest.fixture
def sample_job_text():
    """Sample job description for testing"""
    return """
    Senior Python Developer
    
    We are looking for an experienced Python developer with:
    - 3+ years Python experience
    - Machine Learning knowledge
    - Cloud platform experience (GCP preferred)
    - SQL and database skills
    
    Requirements:
    - Bachelor's degree in Computer Science
    - Experience with BigQuery
    - Docker containerization knowledge
    """

# Configure pytest to handle missing dependencies gracefully
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )

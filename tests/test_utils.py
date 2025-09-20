"""
Utility tests for the AI-Powered Resume Matcher
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestUtilities:
    """Test utility functions and helpers"""
    
    def test_text_processing_utilities(self):
        """Test basic text processing utilities"""
        # Test basic string operations that might be used in the system
        sample_text = "  Software Engineer with Python experience  \n\n"
        
        # Basic text cleaning simulation
        cleaned = sample_text.strip().replace('\n', ' ')
        assert cleaned == "Software Engineer with Python experience"
        
        # Test case normalization
        normalized = sample_text.lower().strip()
        assert "software engineer" in normalized
        assert "python" in normalized
    
    def test_data_validation_utilities(self):
        """Test data validation helpers"""
        # Test email validation pattern
        valid_emails = ["test@example.com", "user.name@domain.co.uk"]
        invalid_emails = ["invalid-email", "@domain.com", "user@"]
        
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        for email in valid_emails:
            assert re.match(email_pattern, email) is not None
        
        for email in invalid_emails:
            assert re.match(email_pattern, email) is None
    
    def test_file_path_utilities(self):
        """Test file path handling utilities"""
        # Test path operations
        test_path = Path(__file__).parent.parent
        
        assert test_path.exists()
        assert test_path.is_dir()
        
        # Test relative path construction
        src_path = test_path / 'src'
        tests_path = test_path / 'tests'
        
        assert str(src_path).endswith('src')
        assert str(tests_path).endswith('tests')
    
    def test_configuration_utilities(self):
        """Test configuration handling"""
        # Test environment variable handling
        test_env_var = "TEST_VARIABLE"
        test_value = "test_value"
        
        # Set and get environment variable
        os.environ[test_env_var] = test_value
        assert os.getenv(test_env_var) == test_value
        
        # Test default value handling
        non_existent_var = os.getenv("NON_EXISTENT_VAR", "default_value")
        assert non_existent_var == "default_value"
        
        # Clean up
        if test_env_var in os.environ:
            del os.environ[test_env_var]

def test_string_similarity():
    """Test string similarity calculations"""
    # Simple similarity test using basic string operations
    text1 = "Software Engineer"
    text2 = "Software Developer"
    text3 = "Data Scientist"
    
    # Basic word overlap calculation
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    words3 = set(text3.lower().split())
    
    # Calculate Jaccard similarity
    similarity_1_2 = len(words1 & words2) / len(words1 | words2)
    similarity_1_3 = len(words1 & words3) / len(words1 | words3)
    
    # Software Engineer should be more similar to Software Developer
    # than to Data Scientist
    assert similarity_1_2 > similarity_1_3

def test_data_structures():
    """Test data structure operations used in the system"""
    # Test dictionary operations
    resume_data = {
        'name': 'John Doe',
        'skills': ['Python', 'SQL', 'Machine Learning'],
        'experience': '5 years'
    }
    
    job_data = {
        'title': 'Senior Python Developer',
        'requirements': ['Python', 'SQL', 'Django'],
        'experience_required': '3+ years'
    }
    
    # Test skill matching
    resume_skills = set(resume_data['skills'])
    job_requirements = set(job_data['requirements'])
    
    matching_skills = resume_skills & job_requirements
    assert 'Python' in matching_skills
    assert 'SQL' in matching_skills
    assert len(matching_skills) >= 2

@pytest.mark.unit
def test_error_handling_utilities():
    """Test error handling utilities"""
    def safe_divide(a, b):
        """Safe division with error handling"""
        try:
            return a / b
        except ZeroDivisionError:
            return 0
        except TypeError:
            return None
    
    # Test normal operation
    assert safe_divide(10, 2) == 5
    
    # Test division by zero
    assert safe_divide(10, 0) == 0
    
    # Test type error
    assert safe_divide("10", 2) is None

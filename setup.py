"""
Setup script for AI-Powered Resume & Job Matcher
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Fallback requirements if file doesn't exist
    requirements = [
        'google-cloud-bigquery>=3.25.0',
        'bigframes>=1.0.0',
        'pandas>=2.2.2',
        'numpy>=1.26.4',
        'flask>=3.0.3',
        'scikit-learn>=1.4.2',
        'matplotlib>=3.8.3',
        'seaborn>=0.13.2'
    ]

setup(
    name="ai-powered-resume-matcher",
    version="1.0.0",
    author="AI Resume Matcher Team",
    author_email="team@example.com",
    description="AI-Powered Resume & Job Matcher using BigQuery AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ai-powered-resume-matcher",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=8.0.2',
            'pytest-cov>=4.1.0',
            'flake8>=7.0.0',
            'black>=24.2.0',
            'isort>=5.13.2',
            'mypy>=1.8.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'resume-matcher=main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

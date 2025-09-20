Module Reference
================

This section provides an overview of all modules in the AI-Powered Resume & Job Matcher.

Core Modules
------------

The system consists of several core modules that work together to provide AI-powered resume matching:

BigQuery Client (``src.bigquery_client``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles all BigQuery AI operations including:

* ML.GENERATE_EMBEDDING for text vectorization
* VECTOR_SEARCH for semantic similarity matching  
* AI.GENERATE for personalized feedback generation
* Dataset and table management

Data Processor (``src.data_processor``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Responsible for text processing and data cleaning:

* Resume and job description text extraction
* Text normalization and cleaning
* Skill extraction and categorization
* Contact information parsing

Embedding Generator (``src.embedding_generator``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates vector embeddings using BigQuery AI:

* Text-to-vector conversion using ML.GENERATE_EMBEDDING
* Batch processing for large datasets
* Embedding storage and retrieval
* Vector optimization

Semantic Matcher (``src.semantic_matcher``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performs intelligent matching using vector search:

* VECTOR_SEARCH implementation
* Similarity scoring and ranking
* Batch matching capabilities
* Match quality assessment

Feedback Generator (``src.feedback_generator``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates personalized AI feedback:

* AI.GENERATE integration for feedback creation
* Candidate strength analysis
* Improvement recommendations
* Match explanation generation

Utility Modules
---------------

Supporting modules that provide additional functionality:

Visualizer (``src.visualizer``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates analytics dashboards and visualizations:

* Interactive charts and graphs
* Match quality distributions
* Performance metrics
* Export capabilities

Dataset Loader (``src.dataset_loader``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles real dataset processing:

* Naukri.com job dataset processing (22K+ jobs)
* Resume dataset management (962+ resumes)
* Data validation and cleaning
* Category classification

Advanced ML (``src.advanced_ml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides advanced machine learning features:

* Ensemble matching algorithms
* Bias detection and mitigation
* Experience scoring
* Skill gap analysis

Web and API Modules
-------------------

User-facing interfaces and services:

API Module (``src.api``)
~~~~~~~~~~~~~~~~~~~~~~~~

RESTful API for programmatic access:

* Authentication and rate limiting
* Resume and job management endpoints
* Matching and feedback APIs
* Analytics and reporting

Security Module (``src.security``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Security and compliance features:

* GDPR compliance tools
* Data encryption and protection
* Audit logging
* Access control

Worker Module (``src.worker``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Background processing capabilities:

* Celery-based async processing
* Batch job handling
* Notification systems
* Queue management

For detailed API documentation, see the :doc:`api` section.

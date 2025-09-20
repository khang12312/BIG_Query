API Reference
=============

The AI-Powered Resume & Job Matcher provides a comprehensive REST API for programmatic access to all functionality.

Base URL
--------

* Development: ``http://localhost:5001/api``
* Production: ``https://your-domain.com/api``

Authentication
--------------

All API requests require authentication using an API key::

    curl -H "X-API-Key: your-api-key" http://localhost:5001/api/endpoint

Core Endpoints
--------------

Health Check
~~~~~~~~~~~~

Check API health and status.

.. code-block:: http

    GET /api/health

**Response:**

.. code-block:: json

    {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }

Resume Management
~~~~~~~~~~~~~~~~~

Upload and manage resumes.

.. code-block:: http

    POST /api/resumes

**Request Body:**

.. code-block:: json

    {
        "name": "John Doe",
        "email": "john@example.com",
        "resume_text": "Software Engineer with 5 years...",
        "skills": ["Python", "SQL", "Machine Learning"]
    }

**Response:**

.. code-block:: json

    {
        "resume_id": "resume_001",
        "status": "processed",
        "message": "Resume uploaded successfully"
    }

Job Management
~~~~~~~~~~~~~~

Create and manage job postings.

.. code-block:: http

    POST /api/jobs

**Request Body:**

.. code-block:: json

    {
        "title": "Senior Python Developer",
        "company": "Tech Corp",
        "description": "We are looking for...",
        "requirements": ["Python", "Django", "PostgreSQL"]
    }

Matching
~~~~~~~~

Find matches between resumes and jobs.

.. code-block:: http

    POST /api/matches

**Request Body:**

.. code-block:: json

    {
        "job_id": "job_001",
        "top_k": 10,
        "threshold": 0.7
    }

**Response:**

.. code-block:: json

    {
        "matches": [
            {
                "resume_id": "resume_001",
                "similarity_score": 0.85,
                "candidate_name": "John Doe",
                "key_skills": ["Python", "SQL"]
            }
        ],
        "total_matches": 5
    }

Feedback Generation
~~~~~~~~~~~~~~~~~~~

Generate AI-powered feedback for candidates.

.. code-block:: http

    POST /api/feedback

**Request Body:**

.. code-block:: json

    {
        "resume_id": "resume_001",
        "job_id": "job_001"
    }

**Response:**

.. code-block:: json

    {
        "feedback": "Strong technical background...",
        "strengths": ["Python expertise", "ML experience"],
        "improvements": ["Add more project details"],
        "match_score": 85
    }

Analytics
~~~~~~~~~

Get system analytics and insights.

.. code-block:: http

    GET /api/analytics

**Response:**

.. code-block:: json

    {
        "total_resumes": 1000,
        "total_jobs": 200,
        "total_matches": 5000,
        "avg_match_score": 0.72,
        "top_skills": ["Python", "SQL", "JavaScript"]
    }

Error Handling
--------------

The API uses standard HTTP status codes:

* ``200`` - Success
* ``400`` - Bad Request
* ``401`` - Unauthorized
* ``404`` - Not Found
* ``500`` - Internal Server Error

Error responses include details:

.. code-block:: json

    {
        "error": "Invalid request",
        "message": "Missing required field: resume_text",
        "code": 400
    }

Rate Limiting
-------------

API requests are rate limited:

* **Free tier**: 100 requests/hour
* **Premium**: 1000 requests/hour
* **Enterprise**: Unlimited

Rate limit headers are included in responses:

.. code-block:: http

    X-RateLimit-Limit: 100
    X-RateLimit-Remaining: 95
    X-RateLimit-Reset: 1640995200

SDK and Examples
----------------

Python SDK example::

    from resume_matcher_client import ResumeMatcherAPI
    
    client = ResumeMatcherAPI(api_key="your-key")
    
    # Upload resume
    resume = client.upload_resume(
        name="John Doe",
        resume_text="Software Engineer...",
        skills=["Python", "SQL"]
    )
    
    # Find matches
    matches = client.find_matches(
        job_id="job_001",
        top_k=10
    )

JavaScript example::

    const client = new ResumeMatcherAPI('your-api-key');
    
    // Upload resume
    const resume = await client.uploadResume({
        name: 'John Doe',
        resumeText: 'Software Engineer...',
        skills: ['Python', 'SQL']
    });
    
    // Find matches
    const matches = await client.findMatches({
        jobId: 'job_001',
        topK: 10
    });

For more examples and SDKs, visit our GitHub repository.

Installation
============

Requirements
------------

* Python 3.9+
* Google Cloud Project with BigQuery API enabled
* Service account JSON key file

Quick Installation
------------------

1. Clone the repository::

    git clone <repository-url>
    cd ai-powered-resume-matcher

2. Create a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies::

    pip install -r requirements.txt

4. Set up Google Cloud credentials::

    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    export GOOGLE_CLOUD_PROJECT="your-project-id"

5. Run the application::

    python main.py

Docker Installation
-------------------

1. Build and run with Docker Compose::

    docker-compose up -d

2. Access the application:
   
   * Web Interface: http://localhost:5000
   * API: http://localhost:5001
   * Monitoring: http://localhost:3000

Configuration
-------------

The system can be configured through environment variables:

* ``GOOGLE_APPLICATION_CREDENTIALS``: Path to service account JSON
* ``GOOGLE_CLOUD_PROJECT``: Google Cloud project ID
* ``FLASK_ENV``: Flask environment (development/production)
* ``REDIS_URL``: Redis connection URL
* ``LOG_LEVEL``: Logging level (INFO, DEBUG, WARNING, ERROR)

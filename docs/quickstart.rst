Quick Start Guide
=================

This guide will help you get the AI-Powered Resume & Job Matcher up and running quickly.

Prerequisites
-------------

Before you begin, ensure you have:

* Python 3.9 or higher
* Google Cloud Project with BigQuery API enabled
* Service account JSON key file
* Git installed on your system

Step 1: Clone and Setup
-----------------------

1. Clone the repository::

    git clone <your-repository-url>
    cd ai-powered-resume-matcher

2. Create a virtual environment::

    python -m venv venv
    
    # On Windows
    venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate

3. Install dependencies::

    pip install -r requirements.txt

Step 2: Configure Google Cloud
-------------------------------

1. Set environment variables::

    # Windows
    set GOOGLE_APPLICATION_CREDENTIALS=path\to\your\service-account-key.json
    set GOOGLE_CLOUD_PROJECT=your-project-id
    
    # macOS/Linux
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    export GOOGLE_CLOUD_PROJECT="your-project-id"

2. Verify your credentials::

    python test_connection.py

Step 3: Run the Application
---------------------------

1. Start the main application::

    python main.py

2. Or use the web interface::

    cd web_app
    python app.py

3. Access the application:
   
   * Web Interface: http://localhost:5000
   * API Documentation: http://localhost:5001/api/docs

Step 4: Docker Deployment (Optional)
-------------------------------------

For production deployment::

    docker-compose up -d

This will start all services including:

* Web application (port 5000)
* API service (port 5001)
* Redis cache (port 6379)
* Monitoring (port 3000)

Next Steps
----------

* Check out the :doc:`api` documentation for API usage
* Read the :doc:`installation` guide for detailed setup
* Explore the sample data and matching results
* Configure advanced features like real-time processing

Troubleshooting
---------------

**Common Issues:**

1. **Import Errors**: Ensure all dependencies are installed with ``pip install -r requirements.txt``

2. **Google Cloud Authentication**: Verify your service account has BigQuery permissions

3. **Port Conflicts**: Change ports in configuration if 5000/5001 are in use

4. **Memory Issues**: Reduce sample sizes in dataset loading for limited memory systems

For more help, check the GitHub Issues page or contact the development team.

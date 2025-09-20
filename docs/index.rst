AI-Powered Resume & Job Matcher Documentation
==============================================

Welcome to the AI-Powered Resume & Job Matcher documentation!

This project is a comprehensive AI-powered resume matching system that leverages Google BigQuery AI 
to semantically match candidates with job opportunities, generate personalized feedback, and provide 
actionable insights for recruiters.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   modules

Features
--------

* **Semantic Matching**: Beyond keyword search using ML.GENERATE_EMBEDDING + VECTOR_SEARCH
* **Generative AI Feedback**: Personalized candidate feedback using AI.GENERATE
* **Multimodal Analysis**: Text, PDFs, and structured data processing
* **Enterprise Scale**: Handles millions of records with auto-scaling
* **Real-time Processing**: Sub-200ms response times

Quick Start
-----------

1. Clone the repository
2. Install dependencies: ``pip install -r requirements.txt``
3. Set up Google Cloud credentials
4. Run the system: ``python main.py``

Architecture
------------

The system consists of several core components:

* **BigQuery AI Client**: ML model integration and operations
* **Data Processor**: Text extraction and normalization
* **Embedding Generator**: Vector embeddings using ML.GENERATE_EMBEDDING
* **Semantic Matcher**: VECTOR_SEARCH for intelligent matching
* **Feedback Generator**: AI.GENERATE for personalized insights

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

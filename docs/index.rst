AI-Powered Resume & Job Matcher
=================================

Welcome to the AI-Powered Resume & Job Matcher documentation!

This project is a comprehensive AI-powered resume matching system that leverages Google BigQuery AI 
to semantically match candidates with job opportunities.

Features
--------

* Semantic Matching using ML.GENERATE_EMBEDDING + VECTOR_SEARCH
* Generative AI Feedback using AI.GENERATE  
* Multimodal Analysis for text, PDFs, and structured data
* Enterprise Scale handling millions of records
* Real-time Processing with sub-200ms response times

Quick Start
-----------

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Set up Google Cloud credentials
4. Run the system: python main.py

Architecture
------------

The system consists of several core components:

* BigQuery AI Client - ML model integration and operations
* Data Processor - Text extraction and normalization  
* Embedding Generator - Vector embeddings using ML.GENERATE_EMBEDDING
* Semantic Matcher - VECTOR_SEARCH for intelligent matching
* Feedback Generator - AI.GENERATE for personalized insights

Documentation Sections
-----------------------

* Installation Guide
* Quick Start Tutorial
* API Reference
* Module Documentation

For more information, visit our GitHub repository.

"""
BigQuery AI client for resume and job matching operations
"""

import os
from google.cloud import bigquery
import bigframes.pandas as bpd
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from .config import get_config

class BigQueryAIClient:
    """Client for BigQuery AI operations including embeddings, vector search, and text generation"""
    
    def __init__(self):
        self.config = get_config('bigquery')
        self.tables = get_config('tables')
        self.model_config = get_config('model')
        
        # Initialize BigQuery client
        self.client = bigquery.Client(
            project=self.config['project_id'],
            location=self.config['location']
        )
        
        # Initialize BigFrames
        bpd.options.bigquery.project = self.config['project_id']
        bpd.options.bigquery.location = self.config['location']
        
        self.dataset_id = self.config['dataset_id']
        self.project_id = self.config['project_id']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_dataset_if_not_exists(self) -> None:
        """Create BigQuery dataset if it doesn't exist"""
        dataset_ref = self.client.dataset(self.dataset_id)
        
        try:
            # Try to get the dataset in the current location
            dataset = self.client.get_dataset(dataset_ref)
            if dataset.location.lower() != self.config['location'].lower():
                # Delete and recreate in correct location
                self.client.delete_dataset(dataset_ref, delete_contents=True)
                raise Exception("Dataset exists in wrong location")
            self.logger.info(f"Dataset {self.dataset_id} already exists")
        except Exception:
            # Create new dataset in correct location
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.config['location']
            dataset = self.client.create_dataset(dataset)
            self.logger.info(f"Created dataset {self.dataset_id} in {self.config['location']}")
    
    def create_tables(self) -> None:
        """Create necessary tables for the resume matching system"""
        
        # Resume table schema
        resume_schema = [
            bigquery.SchemaField("resume_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("candidate_name", "STRING"),
            bigquery.SchemaField("resume_text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("skills", "STRING"),
            bigquery.SchemaField("experience_years", "INTEGER"),
            bigquery.SchemaField("education", "STRING"),
            bigquery.SchemaField("location", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]
        
        # Job descriptions table schema
        job_schema = [
            bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("company", "STRING"),
            bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("required_skills", "STRING"),
            bigquery.SchemaField("experience_required", "INTEGER"),
            bigquery.SchemaField("location", "STRING"),
            bigquery.SchemaField("salary_range", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]
        
        # Resume embeddings table schema
        resume_embeddings_schema = [
            bigquery.SchemaField("resume_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]
        
        # Job embeddings table schema
        job_embeddings_schema = [
            bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]
        
        tables_to_create = [
            (self.tables['resumes'], resume_schema),
            (self.tables['jobs'], job_schema),
            (self.tables['resume_embeddings'], resume_embeddings_schema),
            (self.tables['job_embeddings'], job_embeddings_schema),
        ]
        
        for table_name, schema in tables_to_create:
            self._create_table_if_not_exists(table_name, schema)
    
    def _create_table_if_not_exists(self, table_name: str, schema: List[bigquery.SchemaField]) -> None:
        """Create a table if it doesn't exist"""
        table_ref = self.client.dataset(self.dataset_id).table(table_name)
        
        try:
            self.client.get_table(table_ref)
            self.logger.info(f"Table {table_name} already exists")
        except Exception:
            table = bigquery.Table(table_ref, schema=schema)
            table = self.client.create_table(table)
            self.logger.info(f"Created table {table_name}")
    
    def generate_embeddings(self, texts: List[str], text_type: str = "resume") -> pd.DataFrame:
        """Generate embeddings using ML.GENERATE_TEXT_EMBEDDING"""
        
        # Create temporary table with texts
        temp_table = f"temp_texts_{text_type}"
        df = pd.DataFrame({
            'id': range(len(texts)),
            'text': texts
        })
        
        # Upload to BigQuery
        table_ref = f"{self.project_id}.{self.dataset_id}.{temp_table}"
        df.to_gbq(table_ref, if_exists='replace', progress_bar=False)
        
        # Generate embeddings using ML.GENERATE_TEXT_EMBEDDING
        query = f"""
        SELECT 
            id,
            text,
            ML.GENERATE_TEXT_EMBEDDING(text,
                MODEL 'textembedding-gecko@003'
            ) AS embedding
        FROM `{table_ref}`
        """
        
        try:
            # First create the embedding model if it doesn't exist
            self._create_embedding_model()
            
            # Generate embeddings
            result = self.client.query(query).to_dataframe()
            
            # Clean up temp table
            self.client.delete_table(table_ref)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            # Clean up temp table
            try:
                self.client.delete_table(table_ref)
            except:
                pass
            raise
    
    def _create_embedding_model(self) -> None:
        """Verify BigQuery setup for embeddings"""
        try:
            # Test query to verify BigQuery ML API access
            test_query = """
            SELECT ML.GENERATE_TEXT_EMBEDDING('test',
                MODEL 'textembedding-gecko@003'
            ) AS embedding
            """
            self.client.query(test_query).result()
            self.logger.info("BigQuery ML API access verified")
        except Exception as e:
            self.logger.warning(f"Could not verify BigQuery ML API access: {e}")
    
    def vector_search(self, query_embedding: List[float], table_name: str, 
                     top_k: int = 10) -> pd.DataFrame:
        """Perform vector search using VECTOR_SEARCH"""
        
        # Convert embedding to string format for BigQuery
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        search_query = f"""
        SELECT 
            base.*,
            distance
        FROM VECTOR_SEARCH(
            TABLE `{self.project_id}.{self.dataset_id}.{table_name}`,
            'embedding',
            (SELECT {embedding_str} as embedding),
            top_k => {top_k}
        )
        """
        
        try:
            result = self.client.query(search_query).to_dataframe()
            return result
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            raise
    
    def generate_feedback(self, resume_text: str, job_description: str, 
                         match_score: float) -> str:
        """Generate personalized feedback using AI.GENERATE"""
        
        prompt = f"""
        As an expert HR consultant, provide personalized feedback for a job candidate.
        
        Job Description:
        {job_description}
        
        Candidate Resume:
        {resume_text}
        
        Match Score: {match_score:.2f}
        
        Please provide:
        1. Strengths that align with the job requirements
        2. Areas for improvement or skill gaps
        3. Specific recommendations for the candidate
        4. Overall assessment and next steps
        
        Keep the feedback constructive, professional, and actionable.
        """
        
        feedback_query = f"""
        SELECT ML.GENERATE_TEXT(
            MODEL `{self.project_id}.{self.dataset_id}.generation_model`,
            STRUCT(
                '{prompt}' AS prompt,
                {self.model_config['max_tokens']} AS max_output_tokens,
                {self.model_config['temperature']} AS temperature
            )
        ) AS feedback
        """
        
        try:
            # Create generation model if needed
            self._create_generation_model()
            
            result = self.client.query(feedback_query).to_dataframe()
            return result.iloc[0]['feedback']
            
        except Exception as e:
            self.logger.error(f"Error generating feedback: {e}")
            return f"Unable to generate personalized feedback. Match score: {match_score:.2f}"
    
    def _create_generation_model(self) -> None:
        """Create text generation model if it doesn't exist"""
        model_query = f"""
        CREATE OR REPLACE MODEL `{self.project_id}.{self.dataset_id}.generation_model`
        REMOTE WITH CONNECTION `projects/{self.project_id}/locations/{self.config['location']}/connections/bq_connection`
        OPTIONS (
            endpoint='{self.model_config["generation_model"]}'
        )
        """
        
        try:
            self.client.query(model_query).result()
            self.logger.info("Generation model created/updated")
        except Exception as e:
            self.logger.warning(f"Could not create generation model: {e}")
    
    def store_embeddings(self, embeddings_df: pd.DataFrame, table_name: str) -> None:
        """Store embeddings in BigQuery table"""
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            embeddings_df.to_gbq(table_ref, if_exists='append', progress_bar=False)
            self.logger.info(f"Stored {len(embeddings_df)} embeddings in {table_name}")
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {e}")
            raise
    
    def get_matches(self, resume_id: str = None, job_id: str = None, 
                   limit: int = 10) -> pd.DataFrame:
        """Get matching results from the database"""
        
        base_query = f"""
        SELECT 
            r.resume_id,
            r.candidate_name,
            j.job_id,
            j.title as job_title,
            j.company,
            similarity_score,
            created_at
        FROM `{self.project_id}.{self.dataset_id}.{self.tables['matches']}` m
        JOIN `{self.project_id}.{self.dataset_id}.{self.tables['resumes']}` r 
            ON m.resume_id = r.resume_id
        JOIN `{self.project_id}.{self.dataset_id}.{self.tables['jobs']}` j 
            ON m.job_id = j.job_id
        """
        
        conditions = []
        if resume_id:
            conditions.append(f"r.resume_id = '{resume_id}'")
        if job_id:
            conditions.append(f"j.job_id = '{job_id}'")
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += f" ORDER BY similarity_score DESC LIMIT {limit}"
        
        try:
            return self.client.query(base_query).to_dataframe()
        except Exception as e:
            self.logger.error(f"Error getting matches: {e}")
            return pd.DataFrame()

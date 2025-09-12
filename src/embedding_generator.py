"""
Embedding generator using BigQuery ML.GENERATE_EMBEDDING
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from .bigquery_client import BigQueryAIClient
from .config import get_config

class EmbeddingGenerator:
    """Generate embeddings for resumes and job descriptions using BigQuery AI"""
    
    def __init__(self):
        self.client = BigQueryAIClient()
        self.config = get_config('matching')
        self.logger = logging.getLogger(__name__)
    
    def generate_resume_embeddings(self, resumes_df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for resume texts"""
        
        if 'resume_text' not in resumes_df.columns:
            raise ValueError("DataFrame must contain 'resume_text' column")
        
        self.logger.info(f"Generating embeddings for {len(resumes_df)} resumes")
        
        try:
            # Prepare texts for embedding
            texts = resumes_df['resume_text'].tolist()
            resume_ids = resumes_df['resume_id'].tolist()
            
            # Generate embeddings using BigQuery AI
            embeddings_result = self.client.generate_embeddings(texts, "resume")
            
            # Create embeddings DataFrame
            embeddings_df = pd.DataFrame({
                'resume_id': [resume_ids[i] for i in embeddings_result['id']],
                'embedding': embeddings_result['embedding'].tolist(),
                'created_at': pd.Timestamp.now()
            })
            
            # Store embeddings in BigQuery
            self.client.store_embeddings(embeddings_df, self.client.tables['resume_embeddings'])
            
            self.logger.info(f"Successfully generated and stored {len(embeddings_df)} resume embeddings")
            return embeddings_df
            
        except Exception as e:
            self.logger.error(f"Error generating resume embeddings: {e}")
            raise
    
    def generate_job_embeddings(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for job description texts"""
        
        if 'description' not in jobs_df.columns:
            raise ValueError("DataFrame must contain 'description' column")
        
        self.logger.info(f"Generating embeddings for {len(jobs_df)} job descriptions")
        
        try:
            # Prepare texts for embedding
            texts = jobs_df['description'].tolist()
            job_ids = jobs_df['job_id'].tolist()
            
            # Generate embeddings using BigQuery AI
            embeddings_result = self.client.generate_embeddings(texts, "job")
            
            # Create embeddings DataFrame
            embeddings_df = pd.DataFrame({
                'job_id': [job_ids[i] for i in embeddings_result['id']],
                'embedding': embeddings_result['embedding'].tolist(),
                'created_at': pd.Timestamp.now()
            })
            
            # Store embeddings in BigQuery
            self.client.store_embeddings(embeddings_df, self.client.tables['job_embeddings'])
            
            self.logger.info(f"Successfully generated and stored {len(embeddings_df)} job embeddings")
            return embeddings_df
            
        except Exception as e:
            self.logger.error(f"Error generating job embeddings: {e}")
            raise
    
    def get_embedding_by_id(self, item_id: str, item_type: str = "resume") -> Optional[List[float]]:
        """Retrieve embedding for a specific resume or job"""
        
        table_name = (self.client.tables['resume_embeddings'] 
                     if item_type == "resume" 
                     else self.client.tables['job_embeddings'])
        
        id_column = 'resume_id' if item_type == "resume" else 'job_id'
        
        query = f"""
        SELECT embedding
        FROM `{self.client.project_id}.{self.client.dataset_id}.{table_name}`
        WHERE {id_column} = '{item_id}'
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        try:
            result = self.client.client.query(query).to_dataframe()
            if not result.empty:
                return result.iloc[0]['embedding']
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving embedding for {item_id}: {e}")
            return None
    
    def batch_generate_embeddings(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate embeddings for multiple datasets in batch"""
        
        results = {}
        
        if 'resumes' in data_dict:
            self.logger.info("Processing resume embeddings...")
            results['resume_embeddings'] = self.generate_resume_embeddings(data_dict['resumes'])
        
        if 'jobs' in data_dict:
            self.logger.info("Processing job embeddings...")
            results['job_embeddings'] = self.generate_job_embeddings(data_dict['jobs'])
        
        return results
    
    def update_embeddings(self, item_ids: List[str], item_type: str = "resume") -> bool:
        """Update embeddings for specific items"""
        
        try:
            if item_type == "resume":
                # Get resume texts
                query = f"""
                SELECT resume_id, resume_text
                FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}`
                WHERE resume_id IN ({','.join([f"'{id}'" for id in item_ids])})
                """
                data_df = self.client.client.query(query).to_dataframe()
                
                if not data_df.empty:
                    self.generate_resume_embeddings(data_df)
            
            else:  # job
                # Get job descriptions
                query = f"""
                SELECT job_id, description
                FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}`
                WHERE job_id IN ({','.join([f"'{id}'" for id in item_ids])})
                """
                data_df = self.client.client.query(query).to_dataframe()
                
                if not data_df.empty:
                    self.generate_job_embeddings(data_df)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating embeddings: {e}")
            return False
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        
        stats = {
            'resume_embeddings': 0,
            'job_embeddings': 0,
            'embedding_dimension': None,
            'last_updated': None
        }
        
        try:
            # Count resume embeddings
            resume_query = f"""
            SELECT COUNT(*) as count, MAX(created_at) as last_updated
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resume_embeddings']}`
            """
            resume_result = self.client.client.query(resume_query).to_dataframe()
            if not resume_result.empty:
                stats['resume_embeddings'] = resume_result.iloc[0]['count']
                stats['last_updated'] = resume_result.iloc[0]['last_updated']
            
            # Count job embeddings
            job_query = f"""
            SELECT COUNT(*) as count, MAX(created_at) as last_updated
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['job_embeddings']}`
            """
            job_result = self.client.client.query(job_query).to_dataframe()
            if not job_result.empty:
                stats['job_embeddings'] = job_result.iloc[0]['count']
                if stats['last_updated'] is None or job_result.iloc[0]['last_updated'] > stats['last_updated']:
                    stats['last_updated'] = job_result.iloc[0]['last_updated']
            
            # Get embedding dimension (sample one embedding)
            if stats['resume_embeddings'] > 0:
                sample_query = f"""
                SELECT embedding
                FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resume_embeddings']}`
                LIMIT 1
                """
                sample_result = self.client.client.query(sample_query).to_dataframe()
                if not sample_result.empty:
                    stats['embedding_dimension'] = len(sample_result.iloc[0]['embedding'])
            
        except Exception as e:
            self.logger.error(f"Error getting embedding statistics: {e}")
        
        return stats

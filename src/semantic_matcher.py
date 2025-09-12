"""
Semantic matcher using BigQuery VECTOR_SEARCH for resume-job matching
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from .bigquery_client import BigQueryAIClient
from .embedding_generator import EmbeddingGenerator
from .config import get_config

class SemanticMatcher:
    """Perform semantic matching between resumes and job descriptions"""
    
    def __init__(self):
        self.client = BigQueryAIClient()
        self.embedding_generator = EmbeddingGenerator()
        self.config = get_config('matching')
        self.logger = logging.getLogger(__name__)
    
    def find_best_candidates(self, job_id: str, top_k: int = None) -> pd.DataFrame:
        """Find best candidate matches for a specific job"""
        
        if top_k is None:
            top_k = self.config['max_matches_per_job']
        
        try:
            # Get job embedding
            job_embedding = self.embedding_generator.get_embedding_by_id(job_id, "job")
            if job_embedding is None:
                raise ValueError(f"No embedding found for job {job_id}")
            
            # Perform vector search on resume embeddings
            matches_df = self.client.vector_search(
                query_embedding=job_embedding,
                table_name=self.client.tables['resume_embeddings'],
                top_k=top_k
            )
            
            if matches_df.empty:
                return pd.DataFrame()
            
            # Join with resume details
            resume_details_query = f"""
            SELECT 
                r.resume_id,
                r.candidate_name,
                r.skills,
                r.experience_years,
                r.education,
                r.location
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}` r
            WHERE r.resume_id IN ({','.join([f"'{id}'" for id in matches_df['resume_id']])})
            """
            
            resume_details = self.client.client.query(resume_details_query).to_dataframe()
            
            # Merge results
            result = matches_df.merge(resume_details, on='resume_id', how='left')
            result['job_id'] = job_id
            result['similarity_score'] = 1 - result['distance']  # Convert distance to similarity
            
            # Filter by similarity threshold
            result = result[result['similarity_score'] >= self.config['similarity_threshold']]
            
            # Sort by similarity score
            result = result.sort_values('similarity_score', ascending=False)
            
            self.logger.info(f"Found {len(result)} candidates for job {job_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error finding candidates for job {job_id}: {e}")
            return pd.DataFrame()
    
    def find_best_jobs(self, resume_id: str, top_k: int = None) -> pd.DataFrame:
        """Find best job matches for a specific resume"""
        
        if top_k is None:
            top_k = self.config['max_jobs_per_candidate']
        
        try:
            # Get resume embedding
            resume_embedding = self.embedding_generator.get_embedding_by_id(resume_id, "resume")
            if resume_embedding is None:
                raise ValueError(f"No embedding found for resume {resume_id}")
            
            # Perform vector search on job embeddings
            matches_df = self.client.vector_search(
                query_embedding=resume_embedding,
                table_name=self.client.tables['job_embeddings'],
                top_k=top_k
            )
            
            if matches_df.empty:
                return pd.DataFrame()
            
            # Join with job details
            job_details_query = f"""
            SELECT 
                j.job_id,
                j.title,
                j.company,
                j.required_skills,
                j.experience_required,
                j.location,
                j.salary_range
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}` j
            WHERE j.job_id IN ({','.join([f"'{id}'" for id in matches_df['job_id']])})
            """
            
            job_details = self.client.client.query(job_details_query).to_dataframe()
            
            # Merge results
            result = matches_df.merge(job_details, on='job_id', how='left')
            result['resume_id'] = resume_id
            result['similarity_score'] = 1 - result['distance']  # Convert distance to similarity
            
            # Filter by similarity threshold
            result = result[result['similarity_score'] >= self.config['similarity_threshold']]
            
            # Sort by similarity score
            result = result.sort_values('similarity_score', ascending=False)
            
            self.logger.info(f"Found {len(result)} jobs for resume {resume_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error finding jobs for resume {resume_id}: {e}")
            return pd.DataFrame()
    
    def batch_match_all(self) -> pd.DataFrame:
        """Perform batch matching for all resumes and jobs"""
        
        try:
            # Get all job IDs
            jobs_query = f"""
            SELECT job_id 
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}`
            """
            jobs_df = self.client.client.query(jobs_query).to_dataframe()
            
            all_matches = []
            
            for _, job_row in jobs_df.iterrows():
                job_id = job_row['job_id']
                matches = self.find_best_candidates(job_id)
                
                if not matches.empty:
                    # Select relevant columns for storage
                    match_records = matches[['resume_id', 'job_id', 'similarity_score']].copy()
                    match_records['created_at'] = pd.Timestamp.now()
                    all_matches.append(match_records)
            
            if all_matches:
                # Combine all matches
                final_matches = pd.concat(all_matches, ignore_index=True)
                
                # Store matches in BigQuery
                table_ref = f"{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}"
                final_matches.to_gbq(table_ref, if_exists='replace', progress_bar=False)
                
                self.logger.info(f"Stored {len(final_matches)} total matches")
                return final_matches
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error in batch matching: {e}")
            return pd.DataFrame()
    
    def get_match_analytics(self) -> Dict[str, Any]:
        """Get analytics about matching results"""
        
        analytics = {
            'total_matches': 0,
            'avg_similarity_score': 0,
            'matches_per_job': 0,
            'matches_per_candidate': 0,
            'top_skills_matched': [],
            'score_distribution': {}
        }
        
        try:
            # Get match statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as total_matches,
                AVG(similarity_score) as avg_similarity,
                COUNT(DISTINCT job_id) as unique_jobs,
                COUNT(DISTINCT resume_id) as unique_resumes
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}`
            """
            
            stats_result = self.client.client.query(stats_query).to_dataframe()
            
            if not stats_result.empty:
                row = stats_result.iloc[0]
                analytics['total_matches'] = int(row['total_matches'])
                analytics['avg_similarity_score'] = float(row['avg_similarity'])
                
                if row['unique_jobs'] > 0:
                    analytics['matches_per_job'] = analytics['total_matches'] / row['unique_jobs']
                if row['unique_resumes'] > 0:
                    analytics['matches_per_candidate'] = analytics['total_matches'] / row['unique_resumes']
            
            # Get score distribution
            distribution_query = f"""
            SELECT 
                CASE 
                    WHEN similarity_score >= 0.9 THEN '0.9-1.0'
                    WHEN similarity_score >= 0.8 THEN '0.8-0.9'
                    WHEN similarity_score >= 0.7 THEN '0.7-0.8'
                    WHEN similarity_score >= 0.6 THEN '0.6-0.7'
                    ELSE '0.5-0.6'
                END as score_range,
                COUNT(*) as count
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}`
            GROUP BY score_range
            ORDER BY score_range DESC
            """
            
            distribution_result = self.client.client.query(distribution_query).to_dataframe()
            
            if not distribution_result.empty:
                analytics['score_distribution'] = dict(
                    zip(distribution_result['score_range'], distribution_result['count'])
                )
            
        except Exception as e:
            self.logger.error(f"Error getting match analytics: {e}")
        
        return analytics
    
    def explain_match(self, resume_id: str, job_id: str) -> Dict[str, Any]:
        """Provide detailed explanation for a specific match"""
        
        try:
            # Get resume and job details
            resume_query = f"""
            SELECT resume_text, skills, experience_years, education
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}`
            WHERE resume_id = '{resume_id}'
            """
            
            job_query = f"""
            SELECT description, required_skills, experience_required, title
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}`
            WHERE job_id = '{job_id}'
            """
            
            resume_data = self.client.client.query(resume_query).to_dataframe()
            job_data = self.client.client.query(job_query).to_dataframe()
            
            if resume_data.empty or job_data.empty:
                return {"error": "Resume or job not found"}
            
            resume_row = resume_data.iloc[0]
            job_row = job_data.iloc[0]
            
            # Get similarity score
            match_query = f"""
            SELECT similarity_score
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}`
            WHERE resume_id = '{resume_id}' AND job_id = '{job_id}'
            """
            
            match_result = self.client.client.query(match_query).to_dataframe()
            similarity_score = match_result.iloc[0]['similarity_score'] if not match_result.empty else 0
            
            # Analyze skill overlap
            resume_skills = set(resume_row['skills'].lower().split(', ')) if resume_row['skills'] else set()
            job_skills = set(job_row['required_skills'].lower().split(', ')) if job_row['required_skills'] else set()
            
            skill_overlap = resume_skills.intersection(job_skills)
            missing_skills = job_skills - resume_skills
            
            explanation = {
                'similarity_score': float(similarity_score),
                'job_title': job_row['title'],
                'skill_match': {
                    'matching_skills': list(skill_overlap),
                    'missing_skills': list(missing_skills),
                    'skill_match_ratio': len(skill_overlap) / len(job_skills) if job_skills else 0
                },
                'experience_match': {
                    'candidate_years': int(resume_row['experience_years']) if resume_row['experience_years'] else 0,
                    'required_years': int(job_row['experience_required']) if job_row['experience_required'] else 0,
                    'meets_requirement': (resume_row['experience_years'] or 0) >= (job_row['experience_required'] or 0)
                },
                'match_quality': self._categorize_match_quality(similarity_score)
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining match {resume_id}-{job_id}: {e}")
            return {"error": str(e)}
    
    def _categorize_match_quality(self, score: float) -> str:
        """Categorize match quality based on similarity score"""
        if score >= 0.9:
            return "Excellent Match"
        elif score >= 0.8:
            return "Very Good Match"
        elif score >= 0.7:
            return "Good Match"
        elif score >= 0.6:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def get_top_matches_summary(self, limit: int = 20) -> pd.DataFrame:
        """Get summary of top matches across all resumes and jobs"""
        
        try:
            summary_query = f"""
            SELECT 
                m.resume_id,
                r.candidate_name,
                m.job_id,
                j.title as job_title,
                j.company,
                m.similarity_score,
                RANK() OVER (ORDER BY m.similarity_score DESC) as match_rank
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}` m
            JOIN `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}` r 
                ON m.resume_id = r.resume_id
            JOIN `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}` j 
                ON m.job_id = j.job_id
            ORDER BY m.similarity_score DESC
            LIMIT {limit}
            """
            
            return self.client.client.query(summary_query).to_dataframe()
            
        except Exception as e:
            self.logger.error(f"Error getting top matches summary: {e}")
            return pd.DataFrame()

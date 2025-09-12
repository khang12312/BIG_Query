"""
Feedback generator using BigQuery AI.GENERATE for personalized candidate feedback
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from .bigquery_client import BigQueryAIClient
from .semantic_matcher import SemanticMatcher
from .config import get_config

class FeedbackGenerator:
    """Generate personalized feedback for candidates using BigQuery AI"""
    
    def __init__(self):
        self.client = BigQueryAIClient()
        self.matcher = SemanticMatcher()
        self.config = get_config('model')
        self.logger = logging.getLogger(__name__)
    
    def generate_candidate_feedback(self, resume_id: str, job_id: str) -> Dict[str, Any]:
        """Generate personalized feedback for a candidate-job match"""
        
        try:
            # Get resume and job details
            resume_query = f"""
            SELECT resume_text, skills, experience_years, education, candidate_name
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}`
            WHERE resume_id = '{resume_id}'
            """
            
            job_query = f"""
            SELECT description, required_skills, experience_required, title, company
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}`
            WHERE job_id = '{job_id}'
            """
            
            resume_data = self.client.client.query(resume_query).to_dataframe()
            job_data = self.client.client.query(job_query).to_dataframe()
            
            if resume_data.empty or job_data.empty:
                return {"error": "Resume or job not found"}
            
            resume_row = resume_data.iloc[0]
            job_row = job_data.iloc[0]
            
            # Get match explanation
            match_explanation = self.matcher.explain_match(resume_id, job_id)
            similarity_score = match_explanation.get('similarity_score', 0)
            
            # Generate AI feedback
            feedback_text = self.client.generate_feedback(
                resume_text=resume_row['resume_text'],
                job_description=job_row['description'],
                match_score=similarity_score
            )
            
            # Structure the feedback
            feedback_data = {
                'resume_id': resume_id,
                'job_id': job_id,
                'candidate_name': resume_row['candidate_name'],
                'job_title': job_row['title'],
                'company': job_row['company'],
                'similarity_score': similarity_score,
                'match_quality': match_explanation.get('match_quality', 'Unknown'),
                'ai_feedback': feedback_text,
                'skill_analysis': match_explanation.get('skill_match', {}),
                'experience_analysis': match_explanation.get('experience_match', {}),
                'created_at': pd.Timestamp.now()
            }
            
            # Store feedback in database
            self._store_feedback(feedback_data)
            
            return feedback_data
            
        except Exception as e:
            self.logger.error(f"Error generating feedback for {resume_id}-{job_id}: {e}")
            return {"error": str(e)}
    
    def generate_batch_feedback(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Generate feedback for multiple matches"""
        
        feedback_results = []
        
        for _, match_row in matches_df.iterrows():
            resume_id = match_row['resume_id']
            job_id = match_row['job_id']
            
            feedback = self.generate_candidate_feedback(resume_id, job_id)
            
            if 'error' not in feedback:
                feedback_results.append(feedback)
            else:
                self.logger.warning(f"Failed to generate feedback for {resume_id}-{job_id}: {feedback['error']}")
        
        return pd.DataFrame(feedback_results)
    
    def generate_improvement_suggestions(self, resume_id: str, target_skills: List[str] = None) -> Dict[str, Any]:
        """Generate improvement suggestions for a candidate"""
        
        try:
            # Get candidate's current profile
            resume_query = f"""
            SELECT resume_text, skills, experience_years, education, candidate_name
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['resumes']}`
            WHERE resume_id = '{resume_id}'
            """
            
            resume_data = self.client.client.query(resume_query).to_dataframe()
            
            if resume_data.empty:
                return {"error": "Resume not found"}
            
            resume_row = resume_data.iloc[0]
            
            # Get candidate's match history
            matches_query = f"""
            SELECT j.title, j.required_skills, m.similarity_score
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['matches']}` m
            JOIN `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}` j 
                ON m.job_id = j.job_id
            WHERE m.resume_id = '{resume_id}'
            ORDER BY m.similarity_score DESC
            LIMIT 10
            """
            
            matches_data = self.client.client.query(matches_query).to_dataframe()
            
            # Analyze skill gaps
            current_skills = set(resume_row['skills'].lower().split(', ')) if resume_row['skills'] else set()
            
            # Collect required skills from job matches
            all_required_skills = set()
            if not matches_data.empty:
                for _, match in matches_data.iterrows():
                    if match['required_skills']:
                        job_skills = set(match['required_skills'].lower().split(', '))
                        all_required_skills.update(job_skills)
            
            # Add target skills if provided
            if target_skills:
                all_required_skills.update([skill.lower() for skill in target_skills])
            
            # Find skill gaps
            skill_gaps = all_required_skills - current_skills
            
            # Generate improvement prompt
            improvement_prompt = f"""
            As a career counselor, provide personalized improvement suggestions for this candidate:
            
            Candidate Profile:
            - Name: {resume_row['candidate_name']}
            - Experience: {resume_row['experience_years']} years
            - Current Skills: {resume_row['skills']}
            - Education: {resume_row['education']}
            
            Skill Gaps Identified: {', '.join(skill_gaps)}
            Average Match Score: {matches_data['similarity_score'].mean():.2f if not matches_data.empty else 'N/A'}
            
            Please provide:
            1. Top 3 skills to develop immediately
            2. Learning resources and certification recommendations
            3. Career progression pathway
            4. Timeline for skill development
            5. How to highlight existing strengths better
            
            Keep suggestions practical and actionable.
            """
            
            # Generate AI suggestions
            suggestions_text = self.client.generate_feedback(
                resume_text=improvement_prompt,
                job_description="Career Development Consultation",
                match_score=0.5  # Neutral score for improvement context
            )
            
            suggestions_data = {
                'resume_id': resume_id,
                'candidate_name': resume_row['candidate_name'],
                'current_skills': list(current_skills),
                'skill_gaps': list(skill_gaps),
                'avg_match_score': float(matches_data['similarity_score'].mean()) if not matches_data.empty else 0,
                'ai_suggestions': suggestions_text,
                'created_at': pd.Timestamp.now()
            }
            
            return suggestions_data
            
        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions for {resume_id}: {e}")
            return {"error": str(e)}
    
    def generate_job_market_insights(self, skills: List[str], experience_years: int) -> Dict[str, Any]:
        """Generate job market insights based on candidate profile"""
        
        try:
            # Analyze job market for similar profiles
            market_query = f"""
            SELECT 
                j.title,
                j.company,
                j.required_skills,
                j.experience_required,
                j.salary_range,
                COUNT(*) as job_count
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['jobs']}` j
            WHERE j.experience_required <= {experience_years + 2}
            AND j.experience_required >= {max(0, experience_years - 2)}
            GROUP BY j.title, j.company, j.required_skills, j.experience_required, j.salary_range
            ORDER BY job_count DESC
            LIMIT 20
            """
            
            market_data = self.client.client.query(market_query).to_dataframe()
            
            # Generate market insights prompt
            skills_str = ', '.join(skills)
            market_prompt = f"""
            As a job market analyst, provide insights for a candidate with:
            - Skills: {skills_str}
            - Experience: {experience_years} years
            
            Based on current job market data, provide:
            1. Most in-demand roles for this profile
            2. Salary expectations and ranges
            3. Geographic hotspots for these skills
            4. Industry trends and growth areas
            5. Skills that are becoming obsolete vs. emerging skills
            6. Recommended career pivots or specializations
            
            Make insights data-driven and actionable.
            """
            
            # Generate AI insights
            insights_text = self.client.generate_feedback(
                resume_text=market_prompt,
                job_description="Job Market Analysis",
                match_score=0.5
            )
            
            insights_data = {
                'skills_analyzed': skills,
                'experience_years': experience_years,
                'market_opportunities': len(market_data),
                'ai_insights': insights_text,
                'top_job_titles': market_data['title'].value_counts().head(5).to_dict() if not market_data.empty else {},
                'created_at': pd.Timestamp.now()
            }
            
            return insights_data
            
        except Exception as e:
            self.logger.error(f"Error generating market insights: {e}")
            return {"error": str(e)}
    
    def _store_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Store feedback in BigQuery"""
        
        try:
            # Create feedback table if it doesn't exist
            feedback_schema = [
                {'name': 'resume_id', 'type': 'STRING'},
                {'name': 'job_id', 'type': 'STRING'},
                {'name': 'candidate_name', 'type': 'STRING'},
                {'name': 'job_title', 'type': 'STRING'},
                {'name': 'company', 'type': 'STRING'},
                {'name': 'similarity_score', 'type': 'FLOAT'},
                {'name': 'match_quality', 'type': 'STRING'},
                {'name': 'ai_feedback', 'type': 'STRING'},
                {'name': 'created_at', 'type': 'TIMESTAMP'}
            ]
            
            # Convert to DataFrame for storage
            feedback_df = pd.DataFrame([{
                'resume_id': feedback_data['resume_id'],
                'job_id': feedback_data['job_id'],
                'candidate_name': feedback_data['candidate_name'],
                'job_title': feedback_data['job_title'],
                'company': feedback_data['company'],
                'similarity_score': feedback_data['similarity_score'],
                'match_quality': feedback_data['match_quality'],
                'ai_feedback': feedback_data['ai_feedback'],
                'created_at': feedback_data['created_at']
            }])
            
            # Store in BigQuery
            table_ref = f"{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['feedback']}"
            feedback_df.to_gbq(table_ref, if_exists='append', progress_bar=False)
            
        except Exception as e:
            self.logger.error(f"Error storing feedback: {e}")
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get analytics about generated feedback"""
        
        analytics = {
            'total_feedback_generated': 0,
            'avg_match_quality': {},
            'feedback_by_company': {},
            'common_improvement_areas': []
        }
        
        try:
            # Get feedback statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as total_feedback,
                match_quality,
                COUNT(*) as quality_count
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['feedback']}`
            GROUP BY match_quality
            """
            
            stats_result = self.client.client.query(stats_query).to_dataframe()
            
            if not stats_result.empty:
                analytics['total_feedback_generated'] = int(stats_result['quality_count'].sum())
                analytics['avg_match_quality'] = dict(
                    zip(stats_result['match_quality'], stats_result['quality_count'])
                )
            
            # Get feedback by company
            company_query = f"""
            SELECT company, COUNT(*) as feedback_count
            FROM `{self.client.project_id}.{self.client.dataset_id}.{self.client.tables['feedback']}`
            GROUP BY company
            ORDER BY feedback_count DESC
            LIMIT 10
            """
            
            company_result = self.client.client.query(company_query).to_dataframe()
            
            if not company_result.empty:
                analytics['feedback_by_company'] = dict(
                    zip(company_result['company'], company_result['feedback_count'])
                )
            
        except Exception as e:
            self.logger.error(f"Error getting feedback analytics: {e}")
        
        return analytics

"""
Background Worker for AI-Powered Resume Matcher
Handles real-time processing, notifications, and batch operations
"""

import redis
import json
import logging
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from celery import Celery
from celery.schedules import crontab
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from bigquery_client import BigQueryAIClient
from advanced_ml import AdvancedMLProcessor
from semantic_matcher import SemanticMatcher
from feedback_generator import FeedbackGenerator
from security import SecurityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

# Initialize Celery
celery_app = Celery(
    'resume_matcher_worker',
    broker=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}/1",
    backend=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}/2"
)

# Initialize components
bq_client = BigQueryAIClient()
ml_processor = AdvancedMLProcessor()
matcher = SemanticMatcher()
feedback_gen = FeedbackGenerator()
security_manager = SecurityManager()

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Periodic tasks
celery_app.conf.beat_schedule = {
    'process-pending-matches': {
        'task': 'worker.process_pending_matches',
        'schedule': crontab(minute='*/5'),  # Every 5 minutes
    },
    'generate-daily-reports': {
        'task': 'worker.generate_daily_reports',
        'schedule': crontab(hour=8, minute=0),  # Daily at 8 AM
    },
    'cleanup-old-data': {
        'task': 'worker.cleanup_old_data',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'update-market-trends': {
        'task': 'worker.update_market_trends',
        'schedule': crontab(hour=6, minute=0),  # Daily at 6 AM
    },
    'bias-analysis': {
        'task': 'worker.run_bias_analysis',
        'schedule': crontab(hour=10, minute=0),  # Daily at 10 AM
    },
}

@celery_app.task(bind=True)
def process_pending_matches(self):
    """Process pending match requests"""
    try:
        logger.info("Starting pending matches processing")
        
        # Get pending matches from Redis queue
        pending_matches = redis_client.lrange('pending_matches', 0, -1)
        
        if not pending_matches:
            logger.info("No pending matches to process")
            return {'status': 'success', 'processed': 0}
        
        processed_count = 0
        
        for match_request in pending_matches:
            try:
                request_data = json.loads(match_request)
                job_id = request_data.get('job_id')
                user_id = request_data.get('user_id')
                
                if not job_id:
                    continue
                
                # Process the match
                matches_df = matcher.find_best_candidates(job_id, top_k=10)
                
                if not matches_df.empty:
                    # Store results
                    results = {
                        'job_id': job_id,
                        'matches': matches_df.to_dict('records'),
                        'processed_at': datetime.utcnow().isoformat(),
                        'user_id': user_id
                    }
                    
                    # Store in Redis for real-time delivery
                    redis_client.setex(
                        f"match_results:{job_id}:{user_id}",
                        3600,  # 1 hour TTL
                        json.dumps(results)
                    )
                    
                    # Send notification
                    send_match_notification.delay(user_id, job_id, len(matches_df))
                
                processed_count += 1
                
                # Remove processed request
                redis_client.lrem('pending_matches', 1, match_request)
                
            except Exception as e:
                logger.error(f"Error processing match request: {e}")
                continue
        
        logger.info(f"Processed {processed_count} pending matches")
        return {'status': 'success', 'processed': processed_count}
        
    except Exception as e:
        logger.error(f"Error in process_pending_matches: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True)
def generate_daily_reports(self):
    """Generate daily analytics reports"""
    try:
        logger.info("Generating daily reports")
        
        # Get yesterday's data
        yesterday = datetime.now() - timedelta(days=1)
        
        # Generate various reports
        reports = {}
        
        # Match statistics
        matches_query = f"""
        SELECT COUNT(*) as total_matches, AVG(similarity_score) as avg_score
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['matches']}`
        WHERE DATE(created_at) = '{yesterday.strftime('%Y-%m-%d')}'
        """
        
        matches_df = bq_client.client.query(matches_query).to_dataframe()
        reports['match_stats'] = matches_df.iloc[0].to_dict() if not matches_df.empty else {}
        
        # Top skills
        skills_query = f"""
        SELECT skills, COUNT(*) as frequency
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE DATE(created_at) = '{yesterday.strftime('%Y-%m-%d')}'
        AND skills IS NOT NULL
        GROUP BY skills
        ORDER BY frequency DESC
        LIMIT 10
        """
        
        skills_df = bq_client.client.query(skills_query).to_dataframe()
        reports['top_skills'] = skills_df.to_dict('records') if not skills_df.empty else []
        
        # Store report
        report_data = {
            'date': yesterday.strftime('%Y-%m-%d'),
            'generated_at': datetime.utcnow().isoformat(),
            'reports': reports
        }
        
        redis_client.setex(
            f"daily_report:{yesterday.strftime('%Y-%m-%d')}",
            86400 * 7,  # 7 days TTL
            json.dumps(report_data)
        )
        
        logger.info("Daily reports generated successfully")
        return {'status': 'success', 'report_date': yesterday.strftime('%Y-%m-%d')}
        
    except Exception as e:
        logger.error(f"Error generating daily reports: {e}")
        raise self.retry(exc=e, countdown=300, max_retries=3)

@celery_app.task(bind=True)
def cleanup_old_data(self):
    """Clean up old data based on retention policies"""
    try:
        logger.info("Starting data cleanup")
        
        cleanup_stats = {
            'resumes_deleted': 0,
            'jobs_deleted': 0,
            'matches_deleted': 0,
            'audit_logs_deleted': 0
        }
        
        # Clean up old resumes
        cutoff_date = datetime.now() - timedelta(days=365)  # 1 year retention
        
        cleanup_query = f"""
        DELETE FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE created_at < '{cutoff_date.isoformat()}'
        """
        
        result = bq_client.client.query(cleanup_query).result()
        cleanup_stats['resumes_deleted'] = result.total_rows if hasattr(result, 'total_rows') else 0
        
        # Clean up old jobs
        cleanup_query = f"""
        DELETE FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['jobs']}`
        WHERE created_at < '{cutoff_date.isoformat()}'
        """
        
        result = bq_client.client.query(cleanup_query).result()
        cleanup_stats['jobs_deleted'] = result.total_rows if hasattr(result, 'total_rows') else 0
        
        # Clean up old matches
        cleanup_query = f"""
        DELETE FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['matches']}`
        WHERE created_at < '{cutoff_date.isoformat()}'
        """
        
        result = bq_client.client.query(cleanup_query).result()
        cleanup_stats['matches_deleted'] = result.total_rows if hasattr(result, 'total_rows') else 0
        
        logger.info(f"Data cleanup completed: {cleanup_stats}")
        return {'status': 'success', 'cleanup_stats': cleanup_stats}
        
    except Exception as e:
        logger.error(f"Error in data cleanup: {e}")
        raise self.retry(exc=e, countdown=600, max_retries=3)

@celery_app.task(bind=True)
def update_market_trends(self):
    """Update market trends and insights"""
    try:
        logger.info("Updating market trends")
        
        # Analyze current market trends
        trends_query = f"""
        SELECT 
            skills,
            COUNT(*) as frequency,
            AVG(experience_years) as avg_experience,
            COUNT(DISTINCT location) as locations_count
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE skills IS NOT NULL
        GROUP BY skills
        ORDER BY frequency DESC
        LIMIT 50
        """
        
        trends_df = bq_client.client.query(trends_query).to_dataframe()
        
        # Calculate trend changes
        trends_data = {
            'updated_at': datetime.utcnow().isoformat(),
            'top_skills': trends_df.to_dict('records') if not trends_df.empty else [],
            'market_insights': {
                'total_skills_analyzed': len(trends_df),
                'most_demanded_skill': trends_df.iloc[0]['skills'] if not trends_df.empty else None,
                'avg_experience_level': trends_df['avg_experience'].mean() if not trends_df.empty else 0
            }
        }
        
        # Store trends
        redis_client.setex(
            'market_trends',
            86400,  # 24 hours TTL
            json.dumps(trends_data)
        )
        
        logger.info("Market trends updated successfully")
        return {'status': 'success', 'trends_updated': len(trends_df)}
        
    except Exception as e:
        logger.error(f"Error updating market trends: {e}")
        raise self.retry(exc=e, countdown=300, max_retries=3)

@celery_app.task(bind=True)
def run_bias_analysis(self):
    """Run bias analysis on recent matches"""
    try:
        logger.info("Running bias analysis")
        
        # Get recent matches for analysis
        matches_query = f"""
        SELECT m.*, r.candidate_name, r.location, r.experience_years
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['matches']}` m
        JOIN `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}` r
        ON m.resume_id = r.resume_id
        WHERE m.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        """
        
        matches_df = bq_client.client.query(matches_query).to_dataframe()
        
        if matches_df.empty:
            logger.info("No recent matches found for bias analysis")
            return {'status': 'success', 'message': 'No data to analyze'}
        
        # Perform bias analysis
        bias_report = ml_processor.detect_bias_patterns(matches_df)
        
        # Store bias report
        bias_data = {
            'analysis_date': datetime.utcnow().isoformat(),
            'sample_size': len(matches_df),
            'bias_report': bias_report,
            'recommendations': generate_bias_recommendations(bias_report)
        }
        
        redis_client.setex(
            'bias_analysis_report',
            86400 * 7,  # 7 days TTL
            json.dumps(bias_data)
        )
        
        logger.info("Bias analysis completed successfully")
        return {'status': 'success', 'bias_report_generated': True}
        
    except Exception as e:
        logger.error(f"Error in bias analysis: {e}")
        raise self.retry(exc=e, countdown=600, max_retries=3)

@celery_app.task(bind=True)
def send_match_notification(self, user_id: str, job_id: str, match_count: int):
    """Send notification about new matches"""
    try:
        logger.info(f"Sending match notification to user {user_id}")
        
        # Get user preferences
        user_preferences = redis_client.get(f"user_preferences:{user_id}")
        if user_preferences:
            preferences = json.loads(user_preferences)
            if not preferences.get('notifications_enabled', True):
                return {'status': 'skipped', 'reason': 'notifications_disabled'}
        
        # Send email notification
        send_email_notification(user_id, job_id, match_count)
        
        # Send in-app notification
        notification_data = {
            'type': 'match_notification',
            'user_id': user_id,
            'job_id': job_id,
            'match_count': match_count,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        redis_client.lpush(f"notifications:{user_id}", json.dumps(notification_data))
        
        logger.info(f"Match notification sent to user {user_id}")
        return {'status': 'success', 'notification_sent': True}
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True)
def process_batch_upload(self, upload_id: str, file_data: Dict[str, Any]):
    """Process batch file upload"""
    try:
        logger.info(f"Processing batch upload {upload_id}")
        
        file_type = file_data.get('type')
        file_content = file_data.get('content')
        
        if not file_type or not file_content:
            raise ValueError("Invalid file data")
        
        processed_count = 0
        
        if file_type == 'resumes':
            # Process resume data
            for resume_data in file_content:
                try:
                    # Validate and process resume
                    processed_resume = process_resume_data(resume_data)
                    
                    # Store in BigQuery
                    store_resume_data(processed_resume)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing resume: {e}")
                    continue
        
        elif file_type == 'jobs':
            # Process job data
            for job_data in file_content:
                try:
                    # Validate and process job
                    processed_job = process_job_data(job_data)
                    
                    # Store in BigQuery
                    store_job_data(processed_job)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing job: {e}")
                    continue
        
        # Update upload status
        upload_status = {
            'upload_id': upload_id,
            'status': 'completed',
            'processed_count': processed_count,
            'completed_at': datetime.utcnow().isoformat()
        }
        
        redis_client.setex(
            f"upload_status:{upload_id}",
            86400,  # 24 hours TTL
            json.dumps(upload_status)
        )
        
        logger.info(f"Batch upload {upload_id} completed: {processed_count} items processed")
        return {'status': 'success', 'processed_count': processed_count}
        
    except Exception as e:
        logger.error(f"Error processing batch upload: {e}")
        
        # Update upload status with error
        upload_status = {
            'upload_id': upload_id,
            'status': 'failed',
            'error': str(e),
            'failed_at': datetime.utcnow().isoformat()
        }
        
        redis_client.setex(
            f"upload_status:{upload_id}",
            86400,
            json.dumps(upload_status)
        )
        
        raise self.retry(exc=e, countdown=300, max_retries=3)

# Helper functions
def generate_bias_recommendations(bias_report: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on bias analysis"""
    recommendations = []
    
    if 'gender_stats' in bias_report:
        recommendations.append("Monitor gender distribution in matches")
    
    if 'location_stats' in bias_report:
        recommendations.append("Review location-based matching criteria")
    
    if 'experience_stats' in bias_report:
        recommendations.append("Ensure experience requirements are fair")
    
    return recommendations

def send_email_notification(user_id: str, job_id: str, match_count: int):
    """Send email notification"""
    # Implement email sending logic
    # This would integrate with your email service (SendGrid, SES, etc.)
    pass

def process_resume_data(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process and validate resume data"""
    # Implement resume processing logic
    return resume_data

def process_job_data(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process and validate job data"""
    # Implement job processing logic
    return job_data

def store_resume_data(resume_data: Dict[str, Any]):
    """Store resume data in BigQuery"""
    # Implement storage logic
    pass

def store_job_data(job_data: Dict[str, Any]):
    """Store job data in BigQuery"""
    # Implement storage logic
    pass

# Worker health check
@celery_app.task
def health_check():
    """Worker health check"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'redis_connected': redis_client.ping(),
        'bigquery_connected': True  # Add actual check
    }

if __name__ == '__main__':
    # Start the worker
    celery_app.start()

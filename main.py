"""
AI-Powered Resume & Job Matcher - Main Entry Point
BigQuery AI Hackathon Project
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import traceback
from typing import Optional, Dict, Any

# Set environment variables first
os.environ['GOOGLE_CLOUD_PROJECT'] = 'divine-catalyst-459423-j5'

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.bigquery_client import BigQueryAIClient
from src.data_processor import DataProcessor
from src.embedding_generator import EmbeddingGenerator
from src.semantic_matcher import SemanticMatcher
from src.feedback_generator import FeedbackGenerator
from src.visualizer import Visualizer
from src.dataset_loader import DatasetLoader

def setup_logging():
    """Setup enhanced logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler('resume_matcher.log', mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger('google.cloud').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def initialize_system() -> bool:
    """Initialize BigQuery AI system and create necessary tables"""
    print("🚀 Initializing AI-Powered Resume Matcher...")
    logger = logging.getLogger(__name__)
    
    try:
        # Validate environment variables
        required_env_vars = ['GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_CLOUD_PROJECT']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
        
        # Initialize BigQuery client with error handling
        logger.info("Initializing BigQuery client...")
        client = BigQueryAIClient()
        
        # Create dataset and tables with validation
        logger.info("Creating dataset if not exists...")
        client.create_dataset_if_not_exists()
        
        logger.info("Creating tables...")
        client.create_tables()
        
        # Verify system is working
        logger.info("Verifying system functionality...")
        test_query = f"SELECT 1 as test FROM `{client.project_id}.{client.dataset_id}.INFORMATION_SCHEMA.TABLES` LIMIT 1"
        client.client.query(test_query).result()
        
        print("✅ BigQuery AI system initialized successfully")
        logger.info("System initialization completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error initializing system: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False

def load_sample_data() -> bool:
    """Load real resume and job data from datasets into BigQuery"""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize dataset loader
        logger.info("Initializing dataset loader...")
        loader = DatasetLoader()
        
        print("📊 Loading real datasets...")
        
        # Load datasets with reasonable sample sizes for demonstration
        logger.info("Loading job dataset...")
        jobs_df = loader.load_job_dataset(sample_size=200)  # Load 200 jobs
        
        logger.info("Loading resume dataset...")
        resumes_df = loader.load_resume_dataset(sample_size=100)  # Load 100 resumes
        
        if jobs_df.empty or resumes_df.empty:
            error_msg = f"Failed to load datasets - Jobs: {len(jobs_df)}, Resumes: {len(resumes_df)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            return False
        
        print(f"📈 Dataset Statistics:")
        print(f"  • Jobs loaded: {len(jobs_df)}")
        print(f"  • Resumes loaded: {len(resumes_df)}")
        print(f"  • Job categories: {jobs_df['category'].value_counts().to_dict()}")
        print(f"  • Resume categories: {resumes_df['category'].value_counts().to_dict()}")
        
        # The datasets are already processed by the DatasetLoader
        # No need for additional processing since they're already structured
        
        # Store in BigQuery
        client = BigQueryAIClient()
        
        # Prepare data for BigQuery storage
        resumes_storage = resumes_df.copy()
        jobs_storage = jobs_df.copy()
        
        # Convert list and dict columns to strings for BigQuery storage
        if 'contact_info' in resumes_storage.columns:
            resumes_storage['contact_info'] = resumes_storage['contact_info'].astype(str)
        if 'skills' in resumes_storage.columns:
            resumes_storage['skills'] = resumes_storage['skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            
        if 'skills' in jobs_storage.columns:
            jobs_storage['skills'] = jobs_storage['skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        if 'requirements' in jobs_storage.columns:
            jobs_storage['requirements'] = jobs_storage['requirements'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # Add timestamps
        resumes_storage['created_at'] = datetime.now()
        jobs_storage['created_at'] = datetime.now()
        
        # Store resumes
        table_ref = f"{client.project_id}.{client.dataset_id}.{client.tables['resumes']}"
        resumes_storage.to_gbq(table_ref, if_exists='replace', progress_bar=False)
        
        # Store jobs  
        table_ref = f"{client.project_id}.{client.dataset_id}.{client.tables['jobs']}"
        jobs_storage.to_gbq(table_ref, if_exists='replace', progress_bar=False)
        
        print(f"✅ Loaded {len(resumes_df)} resumes and {len(jobs_df)} job descriptions")
        logger.info(f"Successfully loaded {len(resumes_df)} resumes and {len(jobs_df)} jobs")
        return True
        
    except Exception as e:
        error_msg = f"Error loading sample data: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False

def run_matching_pipeline() -> bool:
    """Run the complete matching pipeline"""
    print("🔄 Running AI-powered matching pipeline...")
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components with error handling
        logger.info("Initializing pipeline components...")
        embedding_gen = EmbeddingGenerator()
        matcher = SemanticMatcher()
        feedback_gen = FeedbackGenerator()
        
        # Step 1: Generate embeddings
        print("1️⃣ Generating embeddings...")
        
        # Get data from BigQuery
        client = BigQueryAIClient()
        
        resumes_query = f"SELECT * FROM `{client.project_id}.{client.dataset_id}.{client.tables['resumes']}`"
        jobs_query = f"SELECT * FROM `{client.project_id}.{client.dataset_id}.{client.tables['jobs']}`"
        
        resumes_df = client.client.query(resumes_query).to_dataframe()
        jobs_df = client.client.query(jobs_query).to_dataframe()
        
        # Generate embeddings
        embedding_gen.generate_resume_embeddings(resumes_df)
        embedding_gen.generate_job_embeddings(jobs_df)
        
        # Step 2: Perform semantic matching
        print("2️⃣ Performing semantic matching...")
        matches_df = matcher.batch_match_all()
        
        # Step 3: Generate feedback for top matches
        print("3️⃣ Generating AI feedback...")
        top_matches = matches_df.head(10) if not matches_df.empty else matches_df
        feedback_df = feedback_gen.generate_batch_feedback(top_matches)
        
        print(f"✅ Pipeline completed: {len(matches_df)} matches, {len(feedback_df)} feedback generated")
        logger.info(f"Pipeline completed successfully - {len(matches_df)} matches, {len(feedback_df)} feedback")
        return True
        
    except Exception as e:
        error_msg = f"Error in matching pipeline: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False

def generate_analytics() -> bool:
    """Generate analytics and visualizations"""
    print("📈 Generating analytics and visualizations...")
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing visualizer...")
        visualizer = Visualizer()
        
        # Create comprehensive dashboard
        visualizer.create_comprehensive_dashboard("analytics_dashboard.html")
        
        # Generate summary report
        report = visualizer.generate_summary_report()
        
        print("📊 Analytics Summary:")
        print(f"   • Total Resumes: {report['total_resumes']}")
        print(f"   • Total Jobs: {report['total_jobs']}")
        print(f"   • Total Matches: {report['total_matches']}")
        print(f"   • Average Match Score: {report['avg_match_score']:.3f}")
        
        if report['recommendations']:
            print("💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("✅ Analytics dashboard saved to 'analytics_dashboard.html'")
        logger.info("Analytics generation completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error generating analytics: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False

def main() -> bool:
    """Main execution function with enhanced error handling"""
    print("🏆 AI-Powered Resume & Job Matcher")
    print("   BigQuery AI Hackathon Project")
    print("=" * 50)
    
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Verify environment variables are set
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        print(f"✅ GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
        print(f"✅ GOOGLE_CLOUD_PROJECT: {project_id}")
        
        # Validate credentials file exists
        if credentials_path and not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
        
        logger.info(f"Starting main pipeline with project: {project_id}")
    
    # Run pipeline steps
    steps = [
        ("Initialize System", initialize_system),
        ("Load Sample Data", load_sample_data),
        ("Run Matching Pipeline", run_matching_pipeline),
        ("Generate Analytics", generate_analytics)
    ]
    
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            logger.info(f"Starting step: {step_name}")
            
            step_start_time = datetime.now()
            success = step_func()
            step_duration = (datetime.now() - step_start_time).total_seconds()
            
            if not success:
                error_msg = f"Failed at step: {step_name} (duration: {step_duration:.2f}s)"
                print(f"❌ {error_msg}")
                logger.error(error_msg)
                return False
            
            logger.info(f"Completed step: {step_name} (duration: {step_duration:.2f}s)")
        
        print("\n🎉 AI-Powered Resume Matcher completed successfully!")
        print("📂 Check the following outputs:")
        print("   • analytics_dashboard.html - Interactive dashboard")
        print("   • resume_matcher.log - Execution logs")
        print("\n🚀 Ready for hackathon demonstration!")
        
        logger.info("Main pipeline completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Critical error in main execution: {str(e)}"
        print(f"❌ {error_msg}")
        logger.critical(f"{error_msg}\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    main()
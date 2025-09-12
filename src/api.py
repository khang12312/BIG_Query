"""
REST API for AI-Powered Resume Matcher
Comprehensive API with authentication, rate limiting, and advanced endpoints
"""

from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import jwt
import hashlib
import secrets
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from bigquery_client import BigQueryAIClient
from advanced_ml import AdvancedMLProcessor
from semantic_matcher import SemanticMatcher
from feedback_generator import FeedbackGenerator
from visualizer import Visualizer

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('API_SECRET_KEY', secrets.token_hex(32))

# Initialize rate limiter
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]
)

# Initialize components
bq_client = BigQueryAIClient()
ml_processor = AdvancedMLProcessor()
matcher = SemanticMatcher()
feedback_gen = FeedbackGenerator()
visualizer = Visualizer()

# API Keys and Users (in production, use proper database)
api_keys = {
    'admin_key_123': {'role': 'admin', 'user_id': 'admin', 'rate_limit': 'unlimited'},
    'recruiter_key_456': {'role': 'recruiter', 'user_id': 'recruiter', 'rate_limit': '1000/hour'},
    'candidate_key_789': {'role': 'candidate', 'user_id': 'candidate', 'rate_limit': '100/hour'}
}

users = {
    'admin': {'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(), 'role': 'admin'},
    'recruiter': {'password_hash': hashlib.sha256('recruiter123'.encode()).hexdigest(), 'role': 'recruiter'},
    'candidate': {'password_hash': hashlib.sha256('candidate123'.encode()).hexdigest(), 'role': 'candidate'}
}

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in api_keys:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        g.current_user = api_keys[api_key]
        return f(*args, **kwargs)
    return decorated_function

def require_role(required_role):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if g.current_user['role'] != required_role and g.current_user['role'] != 'admin':
                return jsonify({'error': f'Insufficient permissions. Required role: {required_role}'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def generate_jwt_token(user_id: str, role: str) -> str:
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Authentication endpoints
@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """User login endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if username in users and users[username]['password_hash'] == password_hash:
        token = generate_jwt_token(username, users[username]['role'])
        return jsonify({
            'token': token,
            'user_id': username,
            'role': users[username]['role'],
            'expires_in': 86400  # 24 hours
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    """User registration endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'candidate')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if username in users:
        return jsonify({'error': 'Username already exists'}), 409
    
    if role not in ['candidate', 'recruiter']:
        return jsonify({'error': 'Invalid role'}), 400
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    users[username] = {'password_hash': password_hash, 'role': role}
    
    token = generate_jwt_token(username, role)
    return jsonify({
        'token': token,
        'user_id': username,
        'role': role,
        'message': 'User registered successfully'
    }), 201

@app.route('/api/auth/refresh', methods=['POST'])
@limiter.limit("20 per hour")
def refresh_token():
    """Refresh JWT token"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    payload = verify_jwt_token(token)
    
    if not payload:
        return jsonify({'error': 'Invalid token'}), 401
    
    new_token = generate_jwt_token(payload['user_id'], payload['role'])
    return jsonify({
        'token': new_token,
        'expires_in': 86400
    })

# Resume endpoints
@app.route('/api/resumes', methods=['GET'])
@require_api_key
@limiter.limit("100 per hour")
def get_resumes():
    """Get all resumes with pagination and filtering"""
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 20)), 100)
    search = request.args.get('search', '')
    skills = request.args.get('skills', '')
    location = request.args.get('location', '')
    
    try:
        query = f"""
        SELECT * FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE 1=1
        """
        
        if search:
            query += f" AND (candidate_name LIKE '%{search}%' OR resume_text LIKE '%{search}%')"
        
        if skills:
            skill_list = skills.split(',')
            skill_conditions = " OR ".join([f"skills LIKE '%{skill.strip()}%'" for skill in skill_list])
            query += f" AND ({skill_conditions})"
        
        if location:
            query += f" AND location LIKE '%{location}%'"
        
        query += f" ORDER BY created_at DESC LIMIT {per_page} OFFSET {(page-1)*per_page}"
        
        resumes_df = bq_client.client.query(query).to_dataframe()
        
        # Get total count
        count_query = query.replace(f"ORDER BY created_at DESC LIMIT {per_page} OFFSET {(page-1)*per_page}", "")
        count_query = f"SELECT COUNT(*) as total FROM ({count_query})"
        total_count = bq_client.client.query(count_query).to_dataframe().iloc[0]['total']
        
        return jsonify({
            'resumes': resumes_df.to_dict('records'),
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': int(total_count),
                'pages': (int(total_count) + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resumes/<resume_id>', methods=['GET'])
@require_api_key
@limiter.limit("200 per hour")
def get_resume(resume_id):
    """Get specific resume by ID"""
    try:
        query = f"""
        SELECT * FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE resume_id = '{resume_id}'
        """
        
        resume_df = bq_client.client.query(query).to_dataframe()
        
        if resume_df.empty:
            return jsonify({'error': 'Resume not found'}), 404
        
        resume_data = resume_df.iloc[0].to_dict()
        
        # Add advanced analysis
        ml_analysis = ml_processor.extract_skills_advanced(resume_data.get('resume_text', ''))
        experience_analysis = ml_processor.calculate_experience_score(resume_data.get('resume_text', ''))
        
        resume_data.update({
            'advanced_analysis': {
                'skills': ml_analysis,
                'experience': experience_analysis
            }
        })
        
        return jsonify(resume_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resumes', methods=['POST'])
@require_api_key
@require_role('recruiter')
@limiter.limit("50 per hour")
def create_resume():
    """Create new resume"""
    data = request.get_json()
    
    required_fields = ['candidate_name', 'resume_text']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        # Generate resume ID
        resume_id = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
        
        # Prepare resume data
        resume_data = {
            'resume_id': resume_id,
            'candidate_name': data['candidate_name'],
            'resume_text': data['resume_text'],
            'skills': data.get('skills', ''),
            'experience_years': data.get('experience_years', 0),
            'education': data.get('education', ''),
            'location': data.get('location', ''),
            'created_at': datetime.now()
        }
        
        # Store in BigQuery
        resume_df = pd.DataFrame([resume_data])
        table_ref = f"{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}"
        resume_df.to_gbq(table_ref, if_exists='append', progress_bar=False)
        
        return jsonify({
            'resume_id': resume_id,
            'message': 'Resume created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Job endpoints
@app.route('/api/jobs', methods=['GET'])
@require_api_key
@limiter.limit("100 per hour")
def get_jobs():
    """Get all job postings with pagination and filtering"""
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 20)), 100)
    search = request.args.get('search', '')
    company = request.args.get('company', '')
    location = request.args.get('location', '')
    
    try:
        query = f"""
        SELECT * FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['jobs']}`
        WHERE 1=1
        """
        
        if search:
            query += f" AND (title LIKE '%{search}%' OR description LIKE '%{search}%')"
        
        if company:
            query += f" AND company LIKE '%{company}%'"
        
        if location:
            query += f" AND location LIKE '%{location}%'"
        
        query += f" ORDER BY created_at DESC LIMIT {per_page} OFFSET {(page-1)*per_page}"
        
        jobs_df = bq_client.client.query(query).to_dataframe()
        
        # Get total count
        count_query = query.replace(f"ORDER BY created_at DESC LIMIT {per_page} OFFSET {(page-1)*per_page}", "")
        count_query = f"SELECT COUNT(*) as total FROM ({count_query})"
        total_count = bq_client.client.query(count_query).to_dataframe().iloc[0]['total']
        
        return jsonify({
            'jobs': jobs_df.to_dict('records'),
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': int(total_count),
                'pages': (int(total_count) + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
@require_api_key
@limiter.limit("200 per hour")
def get_job(job_id):
    """Get specific job by ID"""
    try:
        query = f"""
        SELECT * FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['jobs']}`
        WHERE job_id = '{job_id}'
        """
        
        job_df = bq_client.client.query(query).to_dataframe()
        
        if job_df.empty:
            return jsonify({'error': 'Job not found'}), 404
        
        job_data = job_df.iloc[0].to_dict()
        
        # Add advanced analysis
        ml_analysis = ml_processor.extract_skills_advanced(job_data.get('description', ''))
        experience_analysis = ml_processor.calculate_experience_score(job_data.get('description', ''))
        
        job_data.update({
            'advanced_analysis': {
                'skills': ml_analysis,
                'experience': experience_analysis
            }
        })
        
        return jsonify(job_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Matching endpoints
@app.route('/api/matches', methods=['POST'])
@require_api_key
@limiter.limit("50 per hour")
def find_matches():
    """Find matches between resumes and jobs"""
    data = request.get_json()
    job_id = data.get('job_id')
    resume_id = data.get('resume_id')
    top_k = min(data.get('top_k', 10), 50)
    
    if not job_id and not resume_id:
        return jsonify({'error': 'Either job_id or resume_id required'}), 400
    
    try:
        if job_id:
            # Find best candidates for a job
            matches_df = matcher.find_best_candidates(job_id, top_k=top_k)
            match_type = 'candidates_for_job'
        else:
            # Find best jobs for a candidate
            matches_df = matcher.find_best_jobs(resume_id, top_k=top_k)
            match_type = 'jobs_for_candidate'
        
        if matches_df.empty:
            return jsonify({
                'matches': [],
                'match_type': match_type,
                'message': 'No matches found'
            })
        
        # Add advanced ML analysis for each match
        matches_list = []
        for _, match in matches_df.iterrows():
            match_data = match.to_dict()
            
            # Get detailed data for advanced analysis
            if job_id:
                resume_data = get_resume_data(match['resume_id'])
                job_data = get_job_data(job_id)
            else:
                resume_data = get_resume_data(resume_id)
                job_data = get_job_data(match['job_id'])
            
            if resume_data and job_data:
                # Perform ensemble matching
                ml_analysis = ml_processor.ensemble_matching(resume_data, job_data)
                
                # Generate skill gap analysis
                skill_gap = ml_processor.generate_skill_gap_analysis(
                    ml_analysis['resume_features']['skills']['skills'],
                    ml_analysis['job_features']['skills']['skills']
                )
                
                # Predict salary range
                salary_prediction = ml_processor.predict_salary_range(
                    ml_analysis['resume_features'],
                    ml_analysis['job_features']
                )
                
                match_data.update({
                    'ensemble_score': ml_analysis['ensemble_score'],
                    'individual_scores': ml_analysis['individual_scores'],
                    'skill_gap_analysis': skill_gap,
                    'salary_prediction': salary_prediction,
                    'location_compatibility': ml_analysis['location_score']
                })
            
            matches_list.append(match_data)
        
        return jsonify({
            'matches': matches_list,
            'match_type': match_type,
            'total_matches': len(matches_list),
            'avg_score': matches_df['similarity_score'].mean() if not matches_df.empty else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/matches/batch', methods=['POST'])
@require_api_key
@require_role('recruiter')
@limiter.limit("10 per hour")
def batch_match():
    """Perform batch matching for multiple jobs/resumes"""
    data = request.get_json()
    job_ids = data.get('job_ids', [])
    resume_ids = data.get('resume_ids', [])
    
    if not job_ids and not resume_ids:
        return jsonify({'error': 'Either job_ids or resume_ids required'}), 400
    
    try:
        results = []
        
        if job_ids:
            for job_id in job_ids:
                matches_df = matcher.find_best_candidates(job_id, top_k=5)
                results.append({
                    'job_id': job_id,
                    'matches': matches_df.to_dict('records') if not matches_df.empty else [],
                    'match_count': len(matches_df)
                })
        else:
            for resume_id in resume_ids:
                matches_df = matcher.find_best_jobs(resume_id, top_k=5)
                results.append({
                    'resume_id': resume_id,
                    'matches': matches_df.to_dict('records') if not matches_df.empty else [],
                    'match_count': len(matches_df)
                })
        
        return jsonify({
            'batch_results': results,
            'total_processed': len(results),
            'total_matches': sum(r['match_count'] for r in results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Analytics endpoints
@app.route('/api/analytics/bias', methods=['GET'])
@require_api_key
@require_role('admin')
@limiter.limit("20 per hour")
def bias_analysis():
    """Perform bias analysis on matching results"""
    try:
        # Get recent matches for bias analysis
        matches_query = f"""
        SELECT m.*, r.candidate_name, r.location, r.experience_years
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['matches']}` m
        JOIN `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}` r
        ON m.resume_id = r.resume_id
        ORDER BY m.created_at DESC
        LIMIT 1000
        """
        
        matches_df = bq_client.client.query(matches_query).to_dataframe()
        
        if matches_df.empty:
            return jsonify({'bias_report': {}, 'message': 'No matches found for analysis'})
        
        # Perform bias analysis
        bias_report = ml_processor.detect_bias_patterns(matches_df)
        
        return jsonify({
            'bias_report': bias_report,
            'analysis_date': datetime.now().isoformat(),
            'sample_size': len(matches_df)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/trends', methods=['GET'])
@require_api_key
@limiter.limit("50 per hour")
def market_trends():
    """Get market trends and insights"""
    try:
        # Skills trends
        skills_query = f"""
        SELECT skills, COUNT(*) as frequency
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE skills IS NOT NULL AND skills != ''
        GROUP BY skills
        ORDER BY frequency DESC
        LIMIT 50
        """
        
        skills_df = bq_client.client.query(skills_query).to_dataframe()
        
        # Experience trends
        exp_query = f"""
        SELECT 
            CASE 
                WHEN experience_years <= 2 THEN 'Entry (0-2 years)'
                WHEN experience_years <= 5 THEN 'Junior (2-5 years)'
                WHEN experience_years <= 10 THEN 'Mid (5-10 years)'
                ELSE 'Senior (10+ years)'
            END as experience_level,
            COUNT(*) as count
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE experience_years IS NOT NULL
        GROUP BY experience_level
        """
        
        exp_df = bq_client.client.query(exp_query).to_dataframe()
        
        # Location trends
        location_query = f"""
        SELECT location, COUNT(*) as count
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE location IS NOT NULL AND location != ''
        GROUP BY location
        ORDER BY count DESC
        LIMIT 20
        """
        
        location_df = bq_client.client.query(location_query).to_dataframe()
        
        return jsonify({
            'skills_trends': skills_df.to_dict('records'),
            'experience_trends': exp_df.to_dict('records'),
            'location_trends': location_df.to_dict('records'),
            'analysis_date': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Feedback endpoints
@app.route('/api/feedback', methods=['POST'])
@require_api_key
@limiter.limit("100 per hour")
def generate_feedback():
    """Generate AI feedback for a match"""
    data = request.get_json()
    resume_id = data.get('resume_id')
    job_id = data.get('job_id')
    
    if not resume_id or not job_id:
        return jsonify({'error': 'Both resume_id and job_id required'}), 400
    
    try:
        # Get resume and job data
        resume_data = get_resume_data(resume_id)
        job_data = get_job_data(job_id)
        
        if not resume_data or not job_data:
            return jsonify({'error': 'Resume or job not found'}), 404
        
        # Generate feedback
        feedback = feedback_gen.generate_candidate_feedback(resume_id, job_id)
        
        return jsonify(feedback)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Utility functions
def get_resume_data(resume_id):
    """Get resume data by ID"""
    try:
        query = f"""
        SELECT * FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE resume_id = '{resume_id}'
        """
        result = bq_client.client.query(query).to_dataframe()
        return result.iloc[0].to_dict() if not result.empty else None
    except:
        return None

def get_job_data(job_id):
    """Get job data by ID"""
    try:
        query = f"""
        SELECT * FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['jobs']}`
        WHERE job_id = '{job_id}'
        """
        result = bq_client.client.query(query).to_dataframe()
        return result.iloc[0].to_dict() if not result.empty else None
    except:
        return None

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test BigQuery connection
        test_query = f"SELECT 1 as test FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}` LIMIT 1"
        bq_client.client.query(test_query).result()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'bigquery': 'connected',
                'ai_models': 'available',
                'database': 'accessible'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded', 'retry_after': str(e.retry_after)}), 429

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error_handler(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the API server
    app.run(debug=True, host='0.0.0.0', port=5001)

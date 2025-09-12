"""
Modern Web Interface for AI-Powered Resume Matcher
Flask-based web application with real-time matching and analytics
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
import logging
from functools import wraps
import hashlib
import secrets
import re
from werkzeug.security import check_password_hash, generate_password_hash

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.bigquery_client import BigQueryAIClient
from src.advanced_ml import AdvancedMLProcessor
from src.semantic_matcher import SemanticMatcher
from src.feedback_generator import FeedbackGenerator
from src.visualizer import Visualizer

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize rate limiter
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure CORS more securely
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000", "https://yourdomain.com"])

# Initialize components
bq_client = BigQueryAIClient()
ml_processor = AdvancedMLProcessor()
matcher = SemanticMatcher()
feedback_gen = FeedbackGenerator()
visualizer = Visualizer()

# Enhanced user management with secure password hashing
users = {
    'admin': {'password': generate_password_hash('admin123'), 'role': 'admin', 'failed_attempts': 0, 'locked_until': None},
    'recruiter': {'password': generate_password_hash('recruiter123'), 'role': 'recruiter', 'failed_attempts': 0, 'locked_until': None},
    'candidate': {'password': generate_password_hash('candidate123'), 'role': 'candidate', 'failed_attempts': 0, 'locked_until': None}
}

def is_account_locked(username):
    """Check if account is locked due to failed attempts"""
    user = users.get(username)
    if not user:
        return False
    
    if user['locked_until'] and datetime.now() < user['locked_until']:
        return True
    
    # Reset lock if time has passed
    if user['locked_until'] and datetime.now() >= user['locked_until']:
        user['locked_until'] = None
        user['failed_attempts'] = 0
    
    return False

def sanitize_input(text):
    """Sanitize user input to prevent XSS"""
    if not text:
        return text
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\'\/]', '', str(text))
    return text.strip()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or users.get(session['user_id'], {}).get('role') != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    """Enhanced user login with security measures"""
    if request.method == 'POST':
        username = sanitize_input(request.form.get('username', ''))
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('login.html')
        
        if is_account_locked(username):
            flash('Account is temporarily locked due to failed login attempts', 'error')
            return render_template('login.html')
        
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            # Reset failed attempts on successful login
            user['failed_attempts'] = 0
            user['locked_until'] = None
            
            session['user_id'] = username
            session['role'] = user['role']
            session['login_time'] = datetime.now().isoformat()
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            # Increment failed attempts
            if user:
                user['failed_attempts'] += 1
                if user['failed_attempts'] >= 5:
                    user['locked_until'] = datetime.now() + timedelta(minutes=15)
                    flash('Account locked for 15 minutes due to multiple failed attempts', 'error')
                else:
                    flash(f'Invalid credentials. {5 - user["failed_attempts"]} attempts remaining', 'error')
            else:
                flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    user_role = session.get('role', 'candidate')
    
    # Get dashboard statistics
    try:
        stats = get_dashboard_stats()
    except Exception as e:
        stats = {'error': str(e)}
    
    return render_template('dashboard.html', stats=stats, user_role=user_role)

@app.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for dashboard statistics"""
    try:
        stats = get_dashboard_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    stats = {}
    
    try:
        # Get basic counts
        resumes_query = f"SELECT COUNT(*) as count FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`"
        jobs_query = f"SELECT COUNT(*) as count FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['jobs']}`"
        
        resumes_count = bq_client.client.query(resumes_query).to_dataframe().iloc[0]['count']
        jobs_count = bq_client.client.query(jobs_query).to_dataframe().iloc[0]['count']
        
        stats.update({
            'total_resumes': int(resumes_count),
            'total_jobs': int(jobs_count),
            'total_matches': 0,  # Will be calculated from matches table
            'avg_match_score': 0.0
        })
        
        # Get recent activity
        recent_resumes_query = f"""
        SELECT COUNT(*) as count 
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}` 
        WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        """
        recent_resumes = bq_client.client.query(recent_resumes_query).to_dataframe().iloc[0]['count']
        stats['recent_resumes'] = int(recent_resumes)
        
    except Exception as e:
        stats['error'] = str(e)
    
    return stats

@app.route('/match')
@login_required
def match_page():
    """Resume matching interface"""
    return render_template('match.html')

@app.route('/api/match', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def api_match():
    """API endpoint for resume matching"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        top_k = data.get('top_k', 10)
        
        if not job_id:
            return jsonify({'error': 'Job ID required'}), 400
        
        # Perform matching
        matches = matcher.find_best_candidates(job_id, top_k=top_k)
        
        if matches.empty:
            return jsonify({'matches': [], 'message': 'No matches found'})
        
        # Convert to JSON-serializable format
        matches_list = matches.to_dict('records')
        
        # Add advanced ML analysis for each match
        for match in matches_list:
            resume_id = match['resume_id']
            job_id = match['job_id']
            
            # Get detailed data for advanced analysis
            resume_data = get_resume_data(resume_id)
            job_data = get_job_data(job_id)
            
            if resume_data and job_data:
                # Perform ensemble matching
                ml_analysis = ml_processor.ensemble_matching(resume_data, job_data)
                match.update({
                    'ensemble_score': ml_analysis['ensemble_score'],
                    'skill_gap_analysis': ml_processor.generate_skill_gap_analysis(
                        ml_analysis['resume_features']['skills']['skills'],
                        ml_analysis['job_features']['skills']['skills']
                    ),
                    'salary_prediction': ml_processor.predict_salary_range(
                        ml_analysis['resume_features'],
                        ml_analysis['job_features']
                    )
                })
        
        return jsonify({
            'matches': matches_list,
            'total_matches': len(matches_list),
            'avg_score': matches['similarity_score'].mean() if not matches.empty else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
@login_required
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html')

@app.route('/api/analytics/bias')
@login_required
def api_bias_analysis():
    """API endpoint for bias analysis"""
    try:
        # Get recent matches for bias analysis
        matches_query = f"""
        SELECT m.*, r.candidate_name, r.location, r.experience_years
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['matches']}` m
        JOIN `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}` r
        ON m.resume_id = r.resume_id
        ORDER BY m.created_at DESC
        LIMIT 100
        """
        
        matches_df = bq_client.client.query(matches_query).to_dataframe()
        
        if matches_df.empty:
            return jsonify({'bias_report': {}, 'message': 'No matches found for analysis'})
        
        # Perform bias analysis
        bias_report = ml_processor.detect_bias_patterns(matches_df)
        
        return jsonify({'bias_report': bias_report})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/trends')
@login_required
def api_trends():
    """API endpoint for market trends"""
    try:
        # Get skills trends
        skills_query = f"""
        SELECT skills, COUNT(*) as frequency
        FROM `{bq_client.project_id}.{bq_client.dataset_id}.{bq_client.tables['resumes']}`
        WHERE skills IS NOT NULL
        GROUP BY skills
        ORDER BY frequency DESC
        LIMIT 20
        """
        
        skills_df = bq_client.client.query(skills_query).to_dataframe()
        
        # Get experience trends
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
        
        return jsonify({
            'skills_trends': skills_df.to_dict('records'),
            'experience_trends': exp_df.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload')
@login_required
def upload_page():
    """File upload interface"""
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def api_upload():
    """API endpoint for file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_type = request.form.get('type', 'resume')  # resume or job
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process file based on type
        if file_type == 'resume':
            result = process_resume_upload(file)
        else:
            result = process_job_upload(file)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_resume_upload(file):
    """Process uploaded resume file"""
    # This would integrate with your data processor
    # For now, return a mock response
    return {
        'message': 'Resume uploaded successfully',
        'resume_id': f'resume_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'status': 'processed'
    }

def process_job_upload(file):
    """Process uploaded job description file"""
    # This would integrate with your data processor
    # For now, return a mock response
    return {
        'message': 'Job description uploaded successfully',
        'job_id': f'job_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'status': 'processed'
    }

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

# WebSocket events for real-time updates
@socketio.on('join_room')
def on_join(data):
    """Join a room for real-time updates"""
    room = data['room']
    join_room(room)
    emit('status', {'msg': f'Joined room {room}'}, room=room)

@socketio.on('leave_room')
def on_leave(data):
    """Leave a room"""
    room = data['room']
    leave_room(room)
    emit('status', {'msg': f'Left room {room}'}, room=room)

@socketio.on('request_match')
def on_match_request(data):
    """Handle real-time match requests"""
    job_id = data.get('job_id')
    user_id = session.get('user_id')
    
    if not job_id or not user_id:
        emit('match_error', {'error': 'Invalid request'})
        return
    
    # Perform matching in background
    try:
        matches = matcher.find_best_candidates(job_id, top_k=5)
        
        if not matches.empty:
            emit('match_results', {
                'matches': matches.to_dict('records'),
                'job_id': job_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('match_results', {
                'matches': [],
                'job_id': job_id,
                'message': 'No matches found'
            })
    except Exception as e:
        emit('match_error', {'error': str(e)})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('web_app/templates', exist_ok=True)
    os.makedirs('web_app/static/css', exist_ok=True)
    os.makedirs('web_app/static/js', exist_ok=True)
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

"""
Advanced ML Features for Resume Matching System
Includes ensemble matching, skill extraction, experience scoring, and bias detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import spacy
from collections import Counter
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLProcessor:
    """Advanced ML processor for enhanced resume matching"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.skill_patterns = self._load_skill_patterns()
        self.experience_keywords = self._load_experience_keywords()
        self.location_weights = self._load_location_weights()
        
        # Initialize spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _load_skill_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive skill patterns for extraction"""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'kotlin', 'swift', 'php', 'ruby', 'scala', 'r', 'matlab', 'sql',
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'sqlite', 'dynamodb', 'neo4j', 'influxdb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'amazon web services',
                'microsoft azure', 'kubernetes', 'docker', 'terraform'
            ],
            'ml_ai': [
                'machine learning', 'deep learning', 'neural networks', 'tensorflow',
                'pytorch', 'scikit-learn', 'pandas', 'numpy', 'opencv', 'nlp',
                'natural language processing', 'computer vision', 'reinforcement learning'
            ],
            'tools_frameworks': [
                'git', 'jenkins', 'jira', 'confluence', 'slack', 'figma', 'sketch',
                'tableau', 'power bi', 'excel', 'powerpoint', 'word'
            ]
        }
    
    def _load_experience_keywords(self) -> Dict[str, int]:
        """Load experience level keywords with weights"""
        return {
            'junior': 1, 'entry': 1, 'associate': 2, 'mid': 3, 'senior': 4,
            'lead': 5, 'principal': 5, 'staff': 5, 'architect': 6, 'director': 7,
            'vp': 8, 'vice president': 8, 'cto': 9, 'ceo': 10, 'founder': 10
        }
    
    def _load_location_weights(self) -> Dict[str, float]:
        """Load location-based matching weights"""
        return {
            'remote': 1.0,
            'hybrid': 0.8,
            'onsite': 0.6,
            'same_city': 1.0,
            'same_state': 0.8,
            'same_country': 0.6,
            'international': 0.4
        }
    
    def extract_skills_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced skill extraction using multiple techniques"""
        if not text:
            return {'skills': [], 'skill_categories': {}, 'skill_confidence': 0.0}
        
        text_lower = text.lower()
        extracted_skills = []
        skill_categories = {category: [] for category in self.skill_patterns.keys()}
        
        # Extract skills using pattern matching
        for category, skills in self.skill_patterns.items():
            for skill in skills:
                if skill in text_lower:
                    extracted_skills.append(skill)
                    skill_categories[category].append(skill)
        
        # Use spaCy for additional skill extraction
        if self.nlp:
            doc = self.nlp(text)
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                    if any(keyword in token.text.lower() for keyword in ['api', 'sdk', 'framework', 'library']):
                        extracted_skills.append(token.text.lower())
        
        # Calculate skill confidence based on context
        skill_confidence = min(len(extracted_skills) / 10, 1.0)  # Normalize to 0-1
        
        return {
            'skills': list(set(extracted_skills)),
            'skill_categories': skill_categories,
            'skill_confidence': skill_confidence,
            'total_skills': len(set(extracted_skills))
        }
    
    def calculate_experience_score(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive experience score"""
        if not text:
            return {'experience_score': 0, 'years_experience': 0, 'level': 'entry'}
        
        text_lower = text.lower()
        
        # Extract years of experience
        years_pattern = r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
        years_matches = re.findall(years_pattern, text_lower)
        
        years_experience = 0
        if years_matches:
            years_experience = max([int(match) for match in years_matches])
        
        # Extract experience level keywords
        level_score = 0
        detected_level = 'entry'
        
        for level, score in self.experience_keywords.items():
            if level in text_lower:
                if score > level_score:
                    level_score = score
                    detected_level = level
        
        # Calculate overall experience score
        experience_score = (years_experience * 0.6) + (level_score * 0.4)
        
        # Determine experience level
        if experience_score >= 8:
            level = 'senior'
        elif experience_score >= 5:
            level = 'mid'
        elif experience_score >= 2:
            level = 'junior'
        else:
            level = 'entry'
        
        return {
            'experience_score': min(experience_score, 10),  # Cap at 10
            'years_experience': years_experience,
            'level': level,
            'level_score': level_score
        }
    
    def calculate_location_compatibility(self, resume_location: str, job_location: str) -> float:
        """Calculate location compatibility score"""
        if not resume_location or not job_location:
            return 0.5  # Default neutral score
        
        resume_loc = resume_location.lower()
        job_loc = job_location.lower()
        
        # Check for remote work compatibility
        if 'remote' in job_loc or 'remote' in resume_loc:
            return self.location_weights['remote']
        
        # Check for same city/state
        if resume_loc == job_loc:
            return self.location_weights['same_city']
        
        # Extract city and state for comparison
        resume_parts = resume_loc.split(',')
        job_parts = job_loc.split(',')
        
        if len(resume_parts) > 1 and len(job_parts) > 1:
            resume_state = resume_parts[1].strip()
            job_state = job_parts[1].strip()
            
            if resume_state == job_state:
                return self.location_weights['same_state']
        
        return self.location_weights['same_country']
    
    def ensemble_matching(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ensemble matching using multiple algorithms"""
        
        # Extract features
        resume_skills = self.extract_skills_advanced(resume_data.get('text', ''))
        job_skills = self.extract_skills_advanced(job_data.get('description', ''))
        
        resume_experience = self.calculate_experience_score(resume_data.get('text', ''))
        job_experience = self.calculate_experience_score(job_data.get('description', ''))
        
        location_score = self.calculate_location_compatibility(
            resume_data.get('location', ''), 
            job_data.get('location', '')
        )
        
        # Calculate different similarity scores
        scores = {}
        
        # 1. Skill overlap score
        resume_skill_set = set(resume_skills['skills'])
        job_skill_set = set(job_skills['skills'])
        
        if job_skill_set:
            skill_overlap = len(resume_skill_set.intersection(job_skill_set)) / len(job_skill_set)
        else:
            skill_overlap = 0
        
        scores['skill_overlap'] = skill_overlap
        
        # 2. TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            texts = [resume_data.get('text', ''), job_data.get('description', '')]
            tfidf_matrix = vectorizer.fit_transform(texts)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            scores['tfidf_similarity'] = tfidf_similarity
        except:
            scores['tfidf_similarity'] = 0
        
        # 3. Experience compatibility
        exp_diff = abs(resume_experience['experience_score'] - job_experience['experience_score'])
        exp_compatibility = max(0, 1 - (exp_diff / 10))  # Normalize to 0-1
        scores['experience_compatibility'] = exp_compatibility
        
        # 4. Location compatibility
        scores['location_compatibility'] = location_score
        
        # 5. Skill category matching
        category_scores = []
        for category in resume_skills['skill_categories']:
            resume_cat_skills = set(resume_skills['skill_categories'][category])
            job_cat_skills = set(job_skills['skill_categories'][category])
            
            if job_cat_skills:
                cat_score = len(resume_cat_skills.intersection(job_cat_skills)) / len(job_cat_skills)
                category_scores.append(cat_score)
        
        scores['category_average'] = np.mean(category_scores) if category_scores else 0
        
        # Calculate weighted ensemble score
        weights = {
            'skill_overlap': 0.3,
            'tfidf_similarity': 0.25,
            'experience_compatibility': 0.2,
            'location_compatibility': 0.15,
            'category_average': 0.1
        }
        
        ensemble_score = sum(scores[metric] * weights[metric] for metric in weights)
        
        return {
            'ensemble_score': ensemble_score,
            'individual_scores': scores,
            'weights': weights,
            'resume_features': {
                'skills': resume_skills,
                'experience': resume_experience
            },
            'job_features': {
                'skills': job_skills,
                'experience': job_experience
            },
            'location_score': location_score
        }
    
    def detect_bias_patterns(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential bias patterns in matching results"""
        bias_report = {}
        
        if matches_df.empty:
            return bias_report
        
        # Gender bias detection (based on names - simplified)
        gender_patterns = {
            'female': ['sarah', 'jane', 'mary', 'lisa', 'emma', 'olivia', 'sophia', 'ava'],
            'male': ['john', 'michael', 'david', 'james', 'robert', 'william', 'richard', 'charles']
        }
        
        # Extract gender from names (simplified approach)
        matches_df['gender'] = matches_df['candidate_name'].str.lower().str.split().str[0].map(
            lambda x: 'female' if x in gender_patterns['female'] 
            else 'male' if x in gender_patterns['male'] 
            else 'unknown'
        )
        
        # Calculate match rates by gender
        gender_stats = matches_df.groupby('gender')['similarity_score'].agg(['mean', 'count']).to_dict()
        bias_report['gender_stats'] = gender_stats
        
        # Location bias detection
        location_stats = matches_df.groupby('location')['similarity_score'].agg(['mean', 'count']).to_dict()
        bias_report['location_stats'] = location_stats
        
        # Experience level bias
        if 'experience_years' in matches_df.columns:
            matches_df['exp_level'] = pd.cut(
                matches_df['experience_years'], 
                bins=[0, 2, 5, 10, float('inf')], 
                labels=['entry', 'junior', 'mid', 'senior']
            )
            exp_stats = matches_df.groupby('exp_level')['similarity_score'].agg(['mean', 'count']).to_dict()
            bias_report['experience_stats'] = exp_stats
        
        return bias_report
    
    def generate_skill_gap_analysis(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Generate detailed skill gap analysis"""
        resume_set = set(resume_skills)
        job_set = set(job_skills)
        
        # Skills the candidate has that match the job
        matching_skills = resume_set.intersection(job_set)
        
        # Skills the candidate is missing
        missing_skills = job_set - resume_set
        
        # Skills the candidate has that are extra
        extra_skills = resume_set - job_set
        
        # Calculate gap score
        gap_score = len(missing_skills) / len(job_set) if job_set else 0
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'extra_skills': list(extra_skills),
            'gap_score': gap_score,
            'match_percentage': len(matching_skills) / len(job_set) if job_set else 0,
            'total_job_skills': len(job_set),
            'total_resume_skills': len(resume_set)
        }
    
    def predict_salary_range(self, candidate_features: Dict[str, Any], job_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict salary range based on candidate and job features"""
        # Simplified salary prediction based on experience and skills
        base_salary = 50000  # Base salary
        
        # Experience multiplier
        exp_multiplier = 1 + (candidate_features.get('experience_score', 0) * 0.1)
        
        # Skill multiplier
        skill_count = candidate_features.get('total_skills', 0)
        skill_multiplier = 1 + (skill_count * 0.05)
        
        # Location multiplier (simplified)
        location_multiplier = 1.2 if 'san francisco' in job_features.get('location', '').lower() else 1.0
        
        predicted_salary = base_salary * exp_multiplier * skill_multiplier * location_multiplier
        
        # Add some variance
        variance = predicted_salary * 0.2
        min_salary = max(predicted_salary - variance, base_salary)
        max_salary = predicted_salary + variance
        
        return {
            'predicted_min': int(min_salary),
            'predicted_max': int(max_salary),
            'predicted_median': int(predicted_salary),
            'confidence': 0.7,  # Simplified confidence score
            'factors': {
                'experience_multiplier': exp_multiplier,
                'skill_multiplier': skill_multiplier,
                'location_multiplier': location_multiplier
            }
        }

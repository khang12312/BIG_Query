"""
Working Demo of AI-Powered Resume & Job Matcher
Demonstrates the core functionality without requiring BigQuery AI models
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\chand computer\Desktop\My WorkSpace4\Kaggel\divine-catalyst-459423-j5-6b5f13aeff7c.json'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'divine-catalyst-459423-j5'

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dataset_loader import DatasetLoader
from src.bigquery_client import BigQueryAIClient

class SimpleResumeMatcher:
    """Simplified resume matcher using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.job_vectors = None
        self.resume_vectors = None
        self.jobs_df = None
        self.resumes_df = None
    
    def load_data(self, sample_size=100):
        """Load and process data"""
        print("📊 Loading datasets...")
        
        loader = DatasetLoader()
        self.jobs_df = loader.load_job_dataset(sample_size=sample_size)
        self.resumes_df = loader.load_resume_dataset(sample_size=sample_size//2)
        
        print(f"✅ Loaded {len(self.jobs_df)} jobs and {len(self.resumes_df)} resumes")
        return self.jobs_df, self.resumes_df
    
    def prepare_texts(self):
        """Prepare texts for vectorization"""
        print("🔄 Preparing texts for matching...")
        
        # Combine job title, description, and skills
        job_texts = []
        for _, job in self.jobs_df.iterrows():
            text_parts = [
                str(job.get('title', '')),
                str(job.get('description', '')),
                ', '.join(job.get('skills', [])) if job.get('skills') else ''
            ]
            job_texts.append(' '.join(text_parts))
        
        # Combine resume text and skills
        resume_texts = []
        for _, resume in self.resumes_df.iterrows():
            text_parts = [
                str(resume.get('resume_text', '')),
                ', '.join(resume.get('skills', [])) if resume.get('skills') else ''
            ]
            resume_texts.append(' '.join(text_parts))
        
        return job_texts, resume_texts
    
    def generate_embeddings(self, job_texts, resume_texts):
        """Generate TF-IDF embeddings"""
        print("🤖 Generating embeddings...")
        
        # Combine all texts for vocabulary
        all_texts = job_texts + resume_texts
        
        # Fit vectorizer
        self.vectorizer.fit(all_texts)
        
        # Transform texts
        self.job_vectors = self.vectorizer.transform(job_texts)
        self.resume_vectors = self.vectorizer.transform(resume_texts)
        
        print(f"✅ Generated embeddings: {self.job_vectors.shape[1]} features")
    
    def find_matches(self, top_k=5):
        """Find matches between jobs and resumes"""
        print("🎯 Finding matches...")
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(self.resume_vectors, self.job_vectors)
        
        matches = []
        
        for resume_idx, resume in self.resumes_df.iterrows():
            # Get top matches for this resume
            similarities = similarity_matrix[resume_idx]
            top_job_indices = np.argsort(similarities)[-top_k:][::-1]
            
            for job_idx in top_job_indices:
                similarity_score = similarities[job_idx]
                
                if similarity_score > 0.1:  # Threshold for meaningful matches
                    match = {
                        'resume_id': resume['resume_id'],
                        'candidate_name': resume['candidate_name'],
                        'resume_category': resume['category'],
                        'job_id': self.jobs_df.iloc[job_idx]['job_id'],
                        'job_title': self.jobs_df.iloc[job_idx]['title'],
                        'company': self.jobs_df.iloc[job_idx]['company'],
                        'job_category': self.jobs_df.iloc[job_idx]['category'],
                        'similarity_score': similarity_score,
                        'match_quality': self._categorize_match(similarity_score)
                    }
                    matches.append(match)
        
        matches_df = pd.DataFrame(matches)
        matches_df = matches_df.sort_values('similarity_score', ascending=False)
        
        print(f"✅ Found {len(matches_df)} matches")
        return matches_df
    
    def _categorize_match(self, score):
        """Categorize match quality"""
        if score >= 0.7:
            return "Excellent"
        elif score >= 0.5:
            return "Good"
        elif score >= 0.3:
            return "Fair"
        else:
            return "Poor"
    
    def generate_analytics(self, matches_df):
        """Generate analytics and visualizations"""
        print("📈 Generating analytics...")
        
        # Basic statistics
        total_matches = len(matches_df)
        avg_score = matches_df['similarity_score'].mean()
        
        print(f"\n📊 MATCHING STATISTICS")
        print(f"Total matches: {total_matches}")
        print(f"Average similarity score: {avg_score:.3f}")
        
        # Match quality distribution
        quality_dist = matches_df['match_quality'].value_counts()
        print(f"\nMatch quality distribution:")
        for quality, count in quality_dist.items():
            print(f"  {quality}: {count}")
        
        # Top job categories
        job_categories = matches_df['job_category'].value_counts().head(5)
        print(f"\nTop job categories:")
        for category, count in job_categories.items():
            print(f"  {category}: {count}")
        
        # Top resume categories
        resume_categories = matches_df['resume_category'].value_counts().head(5)
        print(f"\nTop resume categories:")
        for category, count in resume_categories.items():
            print(f"  {category}: {count}")
        
        return {
            'total_matches': total_matches,
            'avg_score': avg_score,
            'quality_distribution': quality_dist.to_dict(),
            'top_job_categories': job_categories.to_dict(),
            'top_resume_categories': resume_categories.to_dict()
        }
    
    def show_top_matches(self, matches_df, n=10):
        """Display top matches"""
        print(f"\n🏆 TOP {n} MATCHES")
        print("=" * 50)
        
        top_matches = matches_df.head(n)
        
        for idx, match in top_matches.iterrows():
            print(f"\n#{idx + 1} - {match['match_quality']} Match (Score: {match['similarity_score']:.3f})")
            print(f"👤 Candidate: {match['candidate_name']} ({match['resume_category']})")
            print(f"💼 Job: {match['job_title']} at {match['company']}")
            print(f"🏷️  Category: {match['job_category']}")
            print("-" * 50)

def main():
    """Main demonstration function"""
    print("🚀 AI-Powered Resume & Job Matcher - Working Demo")
    print("=" * 60)
    
    # Initialize matcher
    matcher = SimpleResumeMatcher()
    
    # Load data
    jobs_df, resumes_df = matcher.load_data(sample_size=100)
    
    # Prepare texts
    job_texts, resume_texts = matcher.prepare_texts()
    
    # Generate embeddings
    matcher.generate_embeddings(job_texts, resume_texts)
    
    # Find matches
    matches_df = matcher.find_matches(top_k=3)
    
    # Generate analytics
    analytics = matcher.generate_analytics(matches_df)
    
    # Show top matches
    matcher.show_top_matches(matches_df, n=15)
    
    # Store results in BigQuery (if connection works)
    try:
        print(f"\n💾 Storing results in BigQuery...")
        client = BigQueryAIClient()
        client.create_dataset_if_not_exists()
        
        # Store matches
        table_ref = f"{client.project_id}.{client.dataset_id}.simple_matches"
        matches_df.to_gbq(table_ref, if_exists='replace', progress_bar=False)
        print(f"✅ Stored {len(matches_df)} matches in BigQuery")
        
    except Exception as e:
        print(f"⚠️  Could not store in BigQuery: {e}")
        print("💡 Results are available locally in matches_df")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📊 Processed {len(jobs_df)} jobs and {len(resumes_df)} resumes")
    print(f"🎯 Generated {len(matches_df)} matches")
    
    return matches_df, analytics

if __name__ == "__main__":
    matches_df, analytics = main()

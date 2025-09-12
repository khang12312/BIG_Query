"""
Dataset Loader for Real Resume and Job Data
Loads and processes the Naukri.com job dataset and resume dataset
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

class DatasetLoader:
    """Load and process real resume and job datasets"""
    
    def __init__(self, data_dir: str = "src/DataSets"):
        self.data_dir = Path(data_dir)
        self.jobs_file = self.data_dir / "JobsSample" / "naukri_com-job_sample.csv"
        self.resumes_file = self.data_dir / "ResumeSample" / "UpdatedResumeDataSet.csv"
        
    def load_job_dataset(self, sample_size: Optional[int] = 1000) -> pd.DataFrame:
        """Load and process the Naukri.com job dataset"""
        try:
            # Load the dataset
            df = pd.read_csv(self.jobs_file)
            
            # Sample if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Clean and standardize the data
            processed_jobs = []
            
            for idx, row in df.iterrows():
                try:
                    job_data = {
                        'job_id': str(row.get('jobid', f'job_{idx}')),
                        'title': self._clean_text(str(row.get('jobtitle', ''))),
                        'company': self._clean_text(str(row.get('company', ''))),
                        'description': self._clean_text(str(row.get('jobdescription', ''))),
                        'skills': self._extract_skills(str(row.get('skills', ''))),
                        'experience': self._clean_text(str(row.get('experience', ''))),
                        'education': self._clean_text(str(row.get('education', ''))),
                        'location': self._clean_text(str(row.get('joblocation_address', ''))),
                        'industry': self._clean_text(str(row.get('industry', ''))),
                        'salary': self._clean_text(str(row.get('payrate', ''))),
                        'requirements': self._extract_requirements(str(row.get('jobdescription', ''))),
                        'category': self._categorize_job(str(row.get('jobtitle', '')), str(row.get('skills', '')))
                    }
                    processed_jobs.append(job_data)
                except Exception as e:
                    logging.warning(f"Error processing job {idx}: {e}")
                    continue
            
            jobs_df = pd.DataFrame(processed_jobs)
            logging.info(f"Loaded {len(jobs_df)} jobs from dataset")
            return jobs_df
            
        except Exception as e:
            logging.error(f"Error loading job dataset: {e}")
            return pd.DataFrame()
    
    def load_resume_dataset(self, sample_size: Optional[int] = 500) -> pd.DataFrame:
        """Load and process the resume dataset"""
        try:
            # Load the dataset
            df = pd.read_csv(self.resumes_file)
            
            # Sample if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Clean and standardize the data
            processed_resumes = []
            
            for idx, row in df.iterrows():
                try:
                    resume_text = str(row.get('Resume', ''))
                    category = str(row.get('Category', 'General'))
                    
                    resume_data = {
                        'resume_id': f'resume_{idx}',
                        'candidate_name': self._extract_name(resume_text),
                        'category': category,
                        'resume_text': self._clean_text(resume_text),
                        'skills': self._extract_skills_from_resume(resume_text),
                        'experience': self._extract_experience(resume_text),
                        'education': self._extract_education(resume_text),
                        'contact_info': self._extract_contact_info(resume_text),
                        'summary': self._extract_summary(resume_text)
                    }
                    processed_resumes.append(resume_data)
                except Exception as e:
                    logging.warning(f"Error processing resume {idx}: {e}")
                    continue
            
            resumes_df = pd.DataFrame(processed_resumes)
            logging.info(f"Loaded {len(resumes_df)} resumes from dataset")
            return resumes_df
            
        except Exception as e:
            logging.error(f"Error loading resume dataset: {e}")
            return pd.DataFrame()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or text == 'nan':
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,()&/]', ' ', text)
        
        return text
    
    def _extract_skills(self, skills_text: str) -> List[str]:
        """Extract skills from text"""
        if pd.isna(skills_text) or skills_text == 'nan':
            return []
        
        # Split by common delimiters
        skills = re.split(r'[,;|]', skills_text)
        
        # Clean and filter skills
        cleaned_skills = []
        for skill in skills:
            skill = skill.strip()
            if len(skill) > 1 and len(skill) < 50:  # Filter out very short or long items
                cleaned_skills.append(skill)
        
        return cleaned_skills[:20]  # Limit to top 20 skills
    
    def _extract_skills_from_resume(self, resume_text: str) -> List[str]:
        """Extract skills from resume text"""
        skills = []
        
        # Common skill patterns
        skill_patterns = [
            r'Skills?\s*:?\s*([^.]+)',
            r'Technical Skills?\s*:?\s*([^.]+)',
            r'Programming Languages?\s*:?\s*([^.]+)',
            r'Technologies?\s*:?\s*([^.]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                extracted_skills = self._extract_skills(match)
                skills.extend(extracted_skills)
        
        # Remove duplicates and return
        return list(set(skills))[:15]
    
    def _extract_requirements(self, job_description: str) -> List[str]:
        """Extract key requirements from job description"""
        requirements = []
        
        # Look for requirement patterns
        req_patterns = [
            r'Requirements?\s*:?\s*([^.]+)',
            r'Qualifications?\s*:?\s*([^.]+)',
            r'Must have\s*:?\s*([^.]+)',
            r'Required\s*:?\s*([^.]+)'
        ]
        
        for pattern in req_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            for match in matches:
                req_items = re.split(r'[,;|]', match)
                for item in req_items:
                    item = item.strip()
                    if len(item) > 5 and len(item) < 100:
                        requirements.append(item)
        
        return requirements[:10]
    
    def _categorize_job(self, title: str, skills: str) -> str:
        """Categorize job based on title and skills"""
        title_lower = title.lower()
        skills_lower = skills.lower()
        
        categories = {
            'Data Science': ['data scientist', 'data analyst', 'machine learning', 'ai', 'analytics'],
            'Software Development': ['developer', 'programmer', 'software engineer', 'full stack'],
            'Web Development': ['web developer', 'frontend', 'backend', 'javascript', 'react'],
            'DevOps': ['devops', 'cloud', 'aws', 'docker', 'kubernetes'],
            'Marketing': ['marketing', 'digital marketing', 'seo', 'content'],
            'Sales': ['sales', 'business development', 'account manager'],
            'HR': ['hr', 'human resources', 'recruiter', 'talent'],
            'Finance': ['finance', 'accounting', 'financial analyst']
        }
        
        for category, keywords in categories.items():
            if any(keyword in title_lower or keyword in skills_lower for keyword in keywords):
                return category
        
        return 'General'
    
    def _extract_name(self, resume_text: str) -> str:
        """Extract candidate name from resume"""
        # Simple name extraction - look for patterns at the beginning
        lines = resume_text.split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            # Look for name patterns (2-4 words, proper case)
            name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', line)
            if name_match:
                return name_match.group(1)
        
        return f"Candidate_{np.random.randint(1000, 9999)}"
    
    def _extract_experience(self, resume_text: str) -> str:
        """Extract experience information"""
        exp_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} years"
        
        return "Not specified"
    
    def _extract_education(self, resume_text: str) -> str:
        """Extract education information"""
        edu_patterns = [
            r'(B\.?Tech|Bachelor|B\.?E\.?|B\.?S\.?|M\.?Tech|Master|M\.?S\.?|PhD|Doctorate)',
            r'Education\s*:?\s*([^\n]+)'
        ]
        
        for pattern in edu_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Not specified"
    
    def _extract_contact_info(self, resume_text: str) -> Dict[str, str]:
        """Extract contact information"""
        contact = {}
        
        # Email pattern
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', resume_text)
        if email_match:
            contact['email'] = email_match.group(1)
        
        # Phone pattern
        phone_match = re.search(r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', resume_text)
        if phone_match:
            contact['phone'] = phone_match.group(1)
        
        return contact
    
    def _extract_summary(self, resume_text: str) -> str:
        """Extract professional summary"""
        # Look for summary sections
        summary_patterns = [
            r'Summary\s*:?\s*([^.]+(?:\.[^.]+){0,2})',
            r'Objective\s*:?\s*([^.]+(?:\.[^.]+){0,2})',
            r'Profile\s*:?\s*([^.]+(?:\.[^.]+){0,2})'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))
        
        # If no explicit summary, take first few sentences
        sentences = resume_text.split('.')[:3]
        return self._clean_text('. '.join(sentences))
    
    def get_dataset_stats(self) -> Dict[str, any]:
        """Get statistics about the datasets"""
        stats = {}
        
        try:
            # Job dataset stats
            jobs_df = pd.read_csv(self.jobs_file)
            stats['jobs'] = {
                'total_records': len(jobs_df),
                'columns': list(jobs_df.columns),
                'file_size_mb': self.jobs_file.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            stats['jobs'] = {'error': str(e)}
        
        try:
            # Resume dataset stats
            resumes_df = pd.read_csv(self.resumes_file)
            stats['resumes'] = {
                'total_records': len(resumes_df),
                'columns': list(resumes_df.columns),
                'categories': resumes_df['Category'].value_counts().to_dict() if 'Category' in resumes_df.columns else {},
                'file_size_mb': self.resumes_file.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            stats['resumes'] = {'error': str(e)}
        
        return stats

if __name__ == "__main__":
    # Test the loader
    loader = DatasetLoader()
    
    print("Dataset Statistics:")
    stats = loader.get_dataset_stats()
    for dataset, info in stats.items():
        print(f"\n{dataset.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\nLoading sample data...")
    jobs_df = loader.load_job_dataset(sample_size=100)
    resumes_df = loader.load_resume_dataset(sample_size=50)
    
    print(f"Loaded {len(jobs_df)} jobs and {len(resumes_df)} resumes")
    
    if not jobs_df.empty:
        print("\nSample Job:")
        print(jobs_df.iloc[0][['title', 'company', 'skills', 'category']])
    
    if not resumes_df.empty:
        print("\nSample Resume:")
        print(resumes_df.iloc[0][['candidate_name', 'category', 'skills']])

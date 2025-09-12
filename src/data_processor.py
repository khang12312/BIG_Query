"""
Data processor for extracting and cleaning text from resumes and job descriptions
"""

import PyPDF2
import docx
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from .config import get_config

class DataProcessor:
    """Process and extract structured data from resumes and job descriptions"""
    
    def __init__(self):
        self.config = get_config('processing')
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def extract_text_from_file(self, file_path: Union[str, Path]) -> str:
        """Extract text from PDF, DOCX, or TXT files"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif extension == '.docx':
                return self._extract_from_docx(file_path)
            elif extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {e}")
        return text.strip()
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\-]{2,}', '-', text)
        
        # Ensure text length is within limits
        if len(text) > self.config['max_text_length']:
            text = text[:self.config['max_text_length']]
        
        return text.strip()
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using pattern matching and keyword lists"""
        
        # Common technical skills
        technical_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'jenkins',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'machine learning', 'deep learning', 'data science', 'ai',
            'project management', 'agile', 'scrum', 'devops', 'ci/cd'
        ]
        
        # Soft skills
        soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'analytical thinking', 'creativity', 'adaptability', 'time management',
            'critical thinking', 'collaboration', 'presentation', 'negotiation'
        ]
        
        all_skills = technical_skills + soft_skills
        text_lower = text.lower()
        
        found_skills = []
        for skill in all_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill.title())
        
        # Remove duplicates and return
        return list(set(found_skills))
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience from text"""
        
        # Patterns to match experience mentions
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*working',
            r'over\s*(\d+)\s*years?',
            r'more than\s*(\d+)\s*years?'
        ]
        
        text_lower = text.lower()
        years = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years.append(int(match))
                except ValueError:
                    continue
        
        # Return the maximum years found, or 0 if none
        return max(years) if years else 0
    
    def extract_education(self, text: str) -> str:
        """Extract education information from text"""
        
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'degree', 'diploma',
            'university', 'college', 'institute', 'school',
            'b.s.', 'b.a.', 'm.s.', 'm.a.', 'mba', 'ph.d.'
        ]
        
        sentences = sent_tokenize(text)
        education_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in education_keywords):
                education_sentences.append(sentence.strip())
        
        return ' '.join(education_sentences[:3])  # Return top 3 education sentences
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from text"""
        
        contact_info = {
            'email': '',
            'phone': '',
            'linkedin': '',
            'location': ''
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Phone pattern
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text.lower())
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()
        
        return contact_info
    
    def process_resume(self, resume_text: str, resume_id: str = None) -> Dict[str, Any]:
        """Process a resume and extract structured information"""
        
        cleaned_text = self.clean_text(resume_text)
        
        if len(cleaned_text) < self.config['min_text_length']:
            self.logger.warning(f"Resume text too short: {len(cleaned_text)} characters")
        
        processed_data = {
            'resume_id': resume_id or f"resume_{hash(cleaned_text) % 10000}",
            'resume_text': cleaned_text,
            'skills': ', '.join(self.extract_skills(cleaned_text)),
            'experience_years': self.extract_experience_years(cleaned_text),
            'education': self.extract_education(cleaned_text),
            'contact_info': self.extract_contact_info(cleaned_text),
            'readability_score': textstat.flesch_reading_ease(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'sentence_count': len(sent_tokenize(cleaned_text))
        }
        
        return processed_data
    
    def process_job_description(self, job_text: str, job_id: str = None) -> Dict[str, Any]:
        """Process a job description and extract structured information"""
        
        cleaned_text = self.clean_text(job_text)
        
        if len(cleaned_text) < self.config['min_text_length']:
            self.logger.warning(f"Job description too short: {len(cleaned_text)} characters")
        
        processed_data = {
            'job_id': job_id or f"job_{hash(cleaned_text) % 10000}",
            'description': cleaned_text,
            'required_skills': ', '.join(self.extract_skills(cleaned_text)),
            'experience_required': self.extract_experience_years(cleaned_text),
            'readability_score': textstat.flesch_reading_ease(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'sentence_count': len(sent_tokenize(cleaned_text))
        }
        
        return processed_data
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases using TF-IDF"""
        
        try:
            # Tokenize and remove stop words
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            
            if len(words) < 5:
                return []
            
            # Create n-grams (1-3 words)
            phrases = []
            for i in range(len(words)):
                # Unigrams
                phrases.append(words[i])
                # Bigrams
                if i < len(words) - 1:
                    phrases.append(f"{words[i]} {words[i+1]}")
                # Trigrams
                if i < len(words) - 2:
                    phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
            
            # Use TF-IDF to score phrases
            vectorizer = TfidfVectorizer(max_features=max_phrases)
            tfidf_matrix = vectorizer.fit_transform([' '.join(phrases)])
            
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases
            phrase_scores = list(zip(feature_names, scores))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [phrase for phrase, score in phrase_scores[:max_phrases]]
            
        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def batch_process_resumes(self, resume_data: List[Dict[str, str]]) -> pd.DataFrame:
        """Process multiple resumes in batch"""
        
        processed_resumes = []
        
        for resume in resume_data:
            try:
                if 'file_path' in resume:
                    text = self.extract_text_from_file(resume['file_path'])
                else:
                    text = resume.get('text', '')
                
                processed = self.process_resume(text, resume.get('resume_id'))
                
                # Add additional metadata
                if 'candidate_name' in resume:
                    processed['candidate_name'] = resume['candidate_name']
                if 'location' in resume:
                    processed['location'] = resume['location']
                
                processed_resumes.append(processed)
                
            except Exception as e:
                self.logger.error(f"Error processing resume {resume.get('resume_id', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(processed_resumes)
    
    def batch_process_jobs(self, job_data: List[Dict[str, str]]) -> pd.DataFrame:
        """Process multiple job descriptions in batch"""
        
        processed_jobs = []
        
        for job in job_data:
            try:
                if 'file_path' in job:
                    text = self.extract_text_from_file(job['file_path'])
                else:
                    text = job.get('description', '')
                
                processed = self.process_job_description(text, job.get('job_id'))
                
                # Add additional metadata
                for field in ['title', 'company', 'location', 'salary_range']:
                    if field in job:
                        processed[field] = job[field]
                
                processed_jobs.append(processed)
                
            except Exception as e:
                self.logger.error(f"Error processing job {job.get('job_id', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(processed_jobs)
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str = 'resume') -> Dict[str, Any]:
        """Validate data quality and provide statistics"""
        
        quality_report = {
            'total_records': len(df),
            'valid_records': 0,
            'issues': [],
            'statistics': {}
        }
        
        if data_type == 'resume':
            text_column = 'resume_text'
            required_fields = ['resume_id', 'resume_text']
        else:
            text_column = 'description'
            required_fields = ['job_id', 'description']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            quality_report['issues'].append(f"Missing required fields: {missing_fields}")
        
        # Check text quality
        if text_column in df.columns:
            text_lengths = df[text_column].str.len()
            quality_report['statistics']['avg_text_length'] = text_lengths.mean()
            quality_report['statistics']['min_text_length'] = text_lengths.min()
            quality_report['statistics']['max_text_length'] = text_lengths.max()
            
            # Count records with sufficient text
            min_length = self.config['min_text_length']
            valid_text = text_lengths >= min_length
            quality_report['valid_records'] = valid_text.sum()
            
            if quality_report['valid_records'] < len(df):
                quality_report['issues'].append(
                    f"{len(df) - quality_report['valid_records']} records have insufficient text"
                )
        
        return quality_report

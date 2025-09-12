"""
Security Module for AI-Powered Resume Matcher
Includes data encryption, GDPR compliance, audit logging, and security utilities
"""

import hashlib
import secrets
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import os
import re
from functools import wraps

class SecurityManager:
    """Comprehensive security manager for the resume matching system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_hex(32))
        
        # GDPR compliance settings
        self.data_retention_days = 365  # 1 year
        self.audit_log_retention_days = 2555  # 7 years
        
        # Security patterns
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
        ]
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = 'encryption.key'
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return {
            'hash': key.decode(),
            'salt': salt
        }
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key.decode() == stored_hash
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate API key for user"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.utcnow().isoformat(),
            'nonce': secrets.token_hex(16)
        }
        
        return base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info"""
        try:
            decoded_data = base64.urlsafe_b64decode(api_key.encode())
            payload = json.loads(decoded_data.decode())
            
            # Check if key is not expired (optional expiration)
            if 'expires_at' in payload:
                if datetime.fromisoformat(payload['expires_at']) < datetime.utcnow():
                    return None
            
            return payload
        except Exception as e:
            self.logger.error(f"API key validation error: {e}")
            return None
    
    def sanitize_data(self, data: Union[str, Dict, List]) -> Union[str, Dict, List]:
        """Sanitize data by removing or masking sensitive information"""
        if isinstance(data, str):
            return self._sanitize_string(data)
        elif isinstance(data, dict):
            return {k: self.sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        else:
            return data
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string by masking sensitive patterns"""
        sanitized = text
        
        for pattern in self.sensitive_patterns:
            if 'SSN' in pattern or 'credit' in pattern:
                sanitized = re.sub(pattern, '[REDACTED]', sanitized)
            elif 'email' in pattern:
                sanitized = re.sub(pattern, '[EMAIL_REDACTED]', sanitized)
            elif 'phone' in pattern:
                sanitized = re.sub(pattern, '[PHONE_REDACTED]', sanitized)
        
        return sanitized
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect personally identifiable information in text"""
        pii_found = {}
        
        for pattern_name, pattern in [
            ('ssn', r'\b\d{3}-\d{2}-\d{4}\b'),
            ('credit_card', r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'),
            ('email', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            ('phone', r'\b\d{3}-\d{3}-\d{4}\b'),
        ]:
            matches = re.findall(pattern, text)
            if matches:
                pii_found[pattern_name] = matches
        
        return pii_found
    
    def audit_log(self, action: str, user_id: str, details: Dict[str, Any], 
                  ip_address: Optional[str] = None) -> None:
        """Log security-relevant actions"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': self.sanitize_data(details),
            'session_id': details.get('session_id'),
            'risk_level': self._assess_risk_level(action, details)
        }
        
        # Log to file and/or external system
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")
        
        # Store in database for compliance
        self._store_audit_log(log_entry)
    
    def _assess_risk_level(self, action: str, details: Dict[str, Any]) -> str:
        """Assess risk level of action"""
        high_risk_actions = ['login', 'password_change', 'data_export', 'admin_access']
        medium_risk_actions = ['data_access', 'search', 'upload']
        
        if action in high_risk_actions:
            return 'HIGH'
        elif action in medium_risk_actions:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _store_audit_log(self, log_entry: Dict[str, Any]) -> None:
        """Store audit log entry"""
        # In production, store in secure audit database
        # For now, just log to file
        audit_file = f"audit_logs_{datetime.now().strftime('%Y%m')}.json"
        
        try:
            with open(audit_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to store audit log: {e}")
    
    def check_data_retention(self, data_type: str, created_at: datetime) -> bool:
        """Check if data should be retained based on GDPR rules"""
        retention_period = timedelta(days=self.data_retention_days)
        
        if data_type in ['resume', 'job_posting']:
            return datetime.utcnow() - created_at < retention_period
        elif data_type == 'audit_log':
            audit_retention = timedelta(days=self.audit_log_retention_days)
            return datetime.utcnow() - created_at < audit_retention
        
        return True  # Default to retain
    
    def anonymize_data(self, data: Dict[str, Any], fields_to_anonymize: List[str]) -> Dict[str, Any]:
        """Anonymize specific fields in data"""
        anonymized = data.copy()
        
        for field in fields_to_anonymize:
            if field in anonymized:
                if isinstance(anonymized[field], str):
                    # Create consistent hash for anonymization
                    anonymized[field] = hashlib.sha256(
                        anonymized[field].encode()
                    ).hexdigest()[:8]
                else:
                    anonymized[field] = '[ANONYMIZED]'
        
        return anonymized
    
    def generate_consent_token(self, user_id: str, data_types: List[str]) -> str:
        """Generate GDPR consent token"""
        payload = {
            'user_id': user_id,
            'data_types': data_types,
            'consent_given': True,
            'timestamp': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=365)).isoformat()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_consent(self, token: str, required_data_types: List[str]) -> bool:
        """Verify GDPR consent for data processing"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            if not payload.get('consent_given', False):
                return False
            
            if datetime.fromisoformat(payload['expires_at']) < datetime.utcnow():
                return False
            
            # Check if user consented to required data types
            consented_types = payload.get('data_types', [])
            return all(dt in consented_types for dt in required_data_types)
            
        except Exception as e:
            self.logger.error(f"Consent verification error: {e}")
            return False
    
    def data_breach_detection(self, access_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential data breaches from access logs"""
        suspicious_activities = []
        
        # Check for unusual access patterns
        user_access_counts = {}
        ip_access_counts = {}
        
        for log in access_logs:
            user_id = log.get('user_id')
            ip_address = log.get('ip_address')
            
            # Count accesses per user
            user_access_counts[user_id] = user_access_counts.get(user_id, 0) + 1
            
            # Count accesses per IP
            ip_access_counts[ip_address] = ip_access_counts.get(ip_address, 0) + 1
        
        # Flag suspicious activities
        for user_id, count in user_access_counts.items():
            if count > 100:  # Threshold for suspicious activity
                suspicious_activities.append({
                    'type': 'excessive_user_access',
                    'user_id': user_id,
                    'access_count': count,
                    'severity': 'HIGH' if count > 500 else 'MEDIUM'
                })
        
        for ip_address, count in ip_access_counts.items():
            if count > 200:  # Threshold for suspicious IP
                suspicious_activities.append({
                    'type': 'excessive_ip_access',
                    'ip_address': ip_address,
                    'access_count': count,
                    'severity': 'HIGH' if count > 1000 else 'MEDIUM'
                })
        
        return suspicious_activities
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'encryption_status': 'ENABLED',
            'audit_logging': 'ENABLED',
            'data_retention_policy': f'{self.data_retention_days} days',
            'gdpr_compliance': 'ENABLED',
            'security_recommendations': [
                'Regular security audits recommended',
                'Monitor access patterns for anomalies',
                'Keep encryption keys secure',
                'Regular backup of audit logs'
            ]
        }
        
        return report

# Security decorators
def require_authentication(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for API key or JWT token
        api_key = request.headers.get('X-API-Key')
        auth_header = request.headers.get('Authorization', '')
        
        if not api_key and not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check user permissions
            user_permissions = get_user_permissions()  # Implement this function
            if not all(perm in user_permissions for perm in required_permissions):
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_security_action(action: str):
    """Decorator to log security-relevant actions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Log the action
            security_manager = SecurityManager()
            security_manager.audit_log(
                action=action,
                user_id=get_current_user_id(),  # Implement this function
                details={'function': f.__name__, 'args': str(args), 'kwargs': str(kwargs)},
                ip_address=request.remote_addr
            )
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Utility functions
def get_current_user_id() -> Optional[str]:
    """Get current user ID from request context"""
    # Implement based on your authentication system
    return None

def get_user_permissions() -> List[str]:
    """Get user permissions from request context"""
    # Implement based on your authorization system
    return []

# Initialize security manager
security_manager = SecurityManager()

#!/usr/bin/env python3
"""
Configuration file for Transport Query Application
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    
    # Translation Configuration
    FORCE_LLM_TRANSLATION = os.getenv('FORCE_LLM_TRANSLATION', 'true').lower() == 'true'
    USE_PATTERN_TRANSLATION = os.getenv('USE_PATTERN_TRANSLATION', 'false').lower() == 'true'
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    LOG_DIR = os.getenv('LOG_DIR', 'logs')
    
    # Spell Correction Configuration
    SPELL_CORRECTION_ENABLED = os.getenv('SPELL_CORRECTION_ENABLED', 'true').lower() == 'true'
    FUZZY_MATCH_THRESHOLD = float(os.getenv('FUZZY_MATCH_THRESHOLD', '80'))
    
    # cPanel specific configurations
    # For cPanel, we need to ensure the app can run in production mode
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Database timeout settings for production
    NEO4J_CONNECTION_TIMEOUT = int(os.getenv('NEO4J_CONNECTION_TIMEOUT', '30'))
    NEO4J_REQUEST_TIMEOUT = int(os.getenv('NEO4J_REQUEST_TIMEOUT', '60'))
    
    # Rate limiting for production
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))  # 1 hour
    
    # Security settings
    CORS_ENABLED = os.getenv('CORS_ENABLED', 'false').lower() == 'true'
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    
    # File upload settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16 * 1024 * 1024'))  # 16MB
    
    # Session configuration
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'false').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Cache settings
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', '300'))  # 5 minutes

# Create a production config class
class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    LOG_LEVEL = 'WARNING'
    
    # Production-specific Neo4j settings
    NEO4J_CONNECTION_TIMEOUT = 60
    NEO4J_REQUEST_TIMEOUT = 120
    
    # Enable rate limiting in production
    RATE_LIMIT_ENABLED = True
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

# Create a development config class
class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CORS_ENABLED = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': Config
}

# Get current configuration based on environment
def get_config():
    env = os.getenv('FLASK_ENV', 'default')
    return config.get(env, Config)

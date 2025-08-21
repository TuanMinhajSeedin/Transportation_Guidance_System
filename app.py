#!/usr/bin/env python3
"""
Main Flask Application for Transport Query System
"""

import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import traceback
from datetime import datetime

# Import configuration
from config import get_config

# Import services
from translation_service import TranslationService
from enhanced_nlp_processor import EnhancedNLPProcessor
from spell_corrector import SpellCorrector
from neo4j_service import Neo4jService
from llm_query_processor import LLMQueryProcessor
from logger import LOG

def create_app(config_class=None):
    """Application factory pattern for Flask"""
    app = Flask(__name__)
    
    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)
    
    # Initialize services
    app.translation_service = TranslationService()
    app.nlp_processor = EnhancedNLPProcessor()
    app.spell_corrector = SpellCorrector()
    app.neo4j_service = Neo4jService()
    app.llm_processor = LLMQueryProcessor()
    
    # Enable CORS if configured
    if app.config.get('CORS_ENABLED', False):
        CORS(app, origins=app.config.get('ALLOWED_ORIGINS', ['*']))
    
    # Proxy fix for cPanel deployment
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    
    # Security headers for production
    @app.after_request
    def add_security_headers(response):
        if app.config.get('SECURITY_HEADERS'):
            for header, value in app.config['SECURITY_HEADERS'].items():
                response.headers[header] = value
        return response
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        LOG.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        LOG.error(f"Unhandled exception: {str(e)}")
        LOG.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500
    
    return app

# Create the Flask app
app = create_app()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process natural language queries"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        language = data.get('language', 'en')
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        LOG.info(f"Processing query: '{query}' (language: {language})")
        
        # Detect language if not provided
        if language == 'auto':
            language = app.translation_service.detect_language(query)
        
        # Translate if needed
        translation_info = None
        if language == 'si':
            translation_info = app.translation_service.translate_query(query)
            if translation_info['success']:
                translated_query = translation_info['translated_query']
                LOG.info(f"Translation: si->en method={translation_info['translation_method']} original='{translation_info['original_query']}' translated='{translation_info['translated_query']}'")
            else:
                LOG.warning(f"Translation failed: {translation_info['error']}")
                return jsonify({
                    'error': 'Translation failed',
                    'details': translation_info['error']
                }), 400
        else:
            translated_query = query
        
        # Process the query
        LOG.info(f"Processing translated query: '{translated_query}'")
        
        # Use enhanced NLP processor
        nlp_result = app.nlp_processor.process_query(translated_query)
        
        if nlp_result['success']:
            response = nlp_result['response']
            confidence = nlp_result.get('confidence', 0.8)
            
            LOG.info(f"Query processed successfully with confidence: {confidence}")
            
            return jsonify({
                'query': query,
                'translated_query': translated_query if language == 'si' else None,
                'response': response,
                'confidence': confidence,
                'intent': nlp_result.get('intent', 'unknown'),
                'entities': nlp_result.get('entities', {})
            })
        else:
            LOG.warning(f"NLP processing failed: {nlp_result['error']}")
            return jsonify({
                'error': 'Could not process query',
                'details': nlp_result['error']
            }), 400
            
    except Exception as e:
        LOG.error(f"Error processing query: {str(e)}")
        LOG.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for cPanel monitoring"""
    try:
        # Check Neo4j connection
        neo4j_status = app.neo4j_service.test_connection()
        
        return jsonify({
            'status': 'healthy',
            'neo4j_connection': neo4j_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        LOG.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Application status endpoint"""
    return jsonify({
        'app_name': 'Transport Query System',
        'version': '1.0.0',
        'environment': os.getenv('FLASK_ENV', 'production'),
        'features': {
            'translation': True,
            'nlp_processing': True,
            'spell_correction': True,
            'neo4j_integration': True
        }
    })

if __name__ == '__main__':
    # Get configuration
    config = get_config()
    
    # Set up logging
    LOG.info(f"Starting Transport Query System in {os.getenv('FLASK_ENV', 'production')} mode")
    
    # Run the app
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )



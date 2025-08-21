#!/usr/bin/env python3
"""
Main Flask Application for Transport Query System
"""

from flask import Flask, render_template, request, jsonify, session
import socket
import os
from llm_query_processor import LLMQueryProcessor
from enhanced_nlp_processor import EnhancedNLPProcessor
from spell_corrector import SpellCorrector
from neo4j_service import Neo4jService
from translation_service import TranslationService
from logger import get_logger
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
logger = get_logger("FlaskApp")

# Initialize services
query_processor = LLMQueryProcessor()
enhanced_nlp_processor = EnhancedNLPProcessor()
spell_corrector = SpellCorrector()
neo4j_service = Neo4jService()
translation_service = TranslationService()

def find_free_port():
    """Find a free port to run the application"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process user query with enhanced NLP and translation support"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        use_enhanced_nlp = data.get('enhanced_nlp', True)  # Default to enhanced NLP
        
        if not user_query:
            return jsonify({
                'success': False,
                'message': 'Please enter a query.'
            })
        
        # Check if query is in Sinhala and translate if needed
        translation_info = translation_service.translate_query(user_query)
        
        # Use translated query for processing
        query_to_process = translation_info['translated_query']
        
        # Log translation info to console
        if translation_info['is_sinhala']:
            logger.info(f"Translation: si->en method={translation_info['translation_method']} original='{translation_info['original_query']}' translated='{translation_info['translated_query']}'")
        else:
            logger.info(f"Processing English Query: '{user_query}'")
        
        # Process the query with enhanced NLP or fallback to basic processor
        if use_enhanced_nlp:
            result = enhanced_nlp_processor.process_query(query_to_process)
        else:
            result = query_processor.process_query(query_to_process)
        
        # If original query was in Sinhala, translate the response back
        if translation_info['is_sinhala']:
            print(f"   English Response: {result.get('message', 'No message')}")
            result = translation_service.translate_response(result)
            result['translation_info'] = translation_info
            print(f"   Sinhala Response: {result.get('message', 'No message')}")
            print(f"   Translation Complete ✅")
        
        logger.info(f"Response success={result.get('success')} type={result.get('query_type','n/a')} message='{result.get('message','')[:120]}'")
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing query: {str(e)}'
        })

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """Get location suggestions for autocomplete"""
    try:
        data = request.get_json()
        partial_location = data.get('location', '').strip()
        
        if not partial_location:
            return jsonify({'suggestions': []})
        
        suggestions = spell_corrector.get_suggestions(partial_location)
        
        return jsonify({
            'suggestions': [{'name': name, 'confidence': conf} for name, conf in suggestions]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting suggestions: {str(e)}'
        })

@app.route('/api/status')
def get_status():
    """Get system status"""
    try:
        neo4j_connected = neo4j_service.is_connected()
        places = neo4j_service.get_all_places() if neo4j_connected else []
        stats = neo4j_service.get_route_statistics() if neo4j_connected else {}
        
        return jsonify({
            'neo4j_connected': neo4j_connected,
            'total_places': len(places),
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting status: {str(e)}'
        })

@app.route('/api/places')
def get_places():
    """Get all available places"""
    try:
        places = neo4j_service.get_all_places()
        return jsonify({
            'success': True,
            'places': places
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting places: {str(e)}'
        })

@app.route('/api/sinhala/examples')
def get_sinhala_examples():
    """Get example queries in Sinhala"""
    try:
        sinhala_examples = translation_service.get_sinhala_examples()
        return jsonify({
            'success': True,
            'examples': sinhala_examples
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting Sinhala examples: {str(e)}'
        })

@app.route('/api/translation/test')
def test_translation():
    """Test translation functionality"""
    try:
        test_results = translation_service.test_translation()
        return jsonify({
            'success': True,
            'test_results': test_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error testing translation: {str(e)}'
        })

@app.route('/api/translation/translate', methods=['POST'])
def translate_text():
    """Translate text between Sinhala and English"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        target_lang = data.get('target_lang', 'en')  # 'en' or 'si'
        source_lang = data.get('source_lang', 'auto')
        
        if not text:
            return jsonify({
                'success': False,
                'message': 'Please provide text to translate.'
            })
        
        translated_text = translation_service.translate_text(text, target_lang, source_lang)
        is_sinhala = translation_service.is_sinhala_text(text)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translated_text': translated_text,
            'source_language': 'si' if is_sinhala else 'en',
            'target_language': target_lang,
            'translation_method': 'google' if translation_service.google_translate_api_key else 'dictionary'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error translating text: {str(e)}'
        })

@app.route('/api/nlp/capabilities')
def get_nlp_capabilities():
    """Get information about natural language processing capabilities with live examples"""
    
    # Test queries for each type to demonstrate actual results
    test_queries = [
        {
            'type': 'fare_inquiry',
            'description': 'Find fare between two specific locations',
            'examples': [
                'What is the fare from Colombo to Kandy?',
                'fare of anuradhapura to kandy',
                'price from panadura to galle',
                'Colombo to Kandy fare'
            ]
        },
        {
            'type': 'comparison',
            'description': 'Compare fares between different routes',
            'examples': [
                'Compare fares from Colombo to Kandy vs Colombo to Galle',
                'Which is cheaper between Colombo to Kandy and Colombo to Anuradapura?',
                'What is the difference in fare between Panadura to Galle and Panadura to Matara?'
            ]
        },
        {
            'type': 'range_search',
            'description': 'Find routes within specific price ranges',
            'examples': [
                'Find routes under 500 rupees',
                'Show me routes between 200 and 800 rupees',
                'Routes over 1000 rupees'
            ]
        },
        {
            'type': 'recommendation',
            'description': 'Get route recommendations based on criteria',
            'examples': [
                'Recommend cheap routes',
                'Show me popular destinations',
                'What are the best routes from Colombo?'
            ]
        },
        {
            'type': 'route_inquiry',
            'description': 'Find routes from/to specific locations',
            'examples': [
                'Routes from Colombo',
                'Routes to Galle',
                'What routes depart from Kandy?'
            ]
        },
        {
            'type': 'statistics',
            'description': 'Get database overview and statistics',
            'examples': [
                'What is the average fare?',
                'Database statistics',
                'How many routes are there?'
            ]
        }
    ]
    
    # Process each test query to get actual results
    live_examples = []
    for query_type in test_queries:
        type_examples = []
        for example_query in query_type['examples'][:2]:  # Test first 2 examples
            try:
                result = enhanced_nlp_processor.process_query(example_query)
                type_examples.append({
                    'query': example_query,
                    'result': result
                })
            except Exception as e:
                type_examples.append({
                    'query': example_query,
                    'result': {
                        'success': False,
                        'message': f'Error: {str(e)}'
                    }
                })
        
        live_examples.append({
            'type': query_type['type'],
            'description': query_type['description'],
            'examples': type_examples
        })
    
    capabilities = {
        'natural_language_processing': {
            'description': 'Advanced NLP for transport queries with enhanced understanding',
            'features': [
                'Multiple query formats (fare, price, cost)',
                'Natural language patterns (from X to Y, X to Y fare, etc.)',
                'Question formats (What is, How much, Show me, etc.)',
                'Compact formats (Colombo to Kandy fare)',
                'Spell correction and fuzzy matching',
                'Automatic location name correction',
                'LLM-powered query interpretation',
                'Fallback keyword-based processing',
                'Advanced intent classification',
                'Entity extraction and normalization',
                'Confidence scoring for query understanding'
            ]
        },
        'query_types': test_queries,
        'live_examples': live_examples,
        'spell_correction': {
            'description': 'Automatic location name correction',
            'methods': [
                'Direct mapping (exact matches)',
                'Fuzzy matching (similar names)',
                'LLM correction (AI-powered)',
                'Partial matching (substring matching)'
            ],
            'examples': [
                'panadra → Panadura',
                'gale → Galle',
                'colmbo → Colombo',
                'kandee → Kandy'
            ]
        },
        'llm_integration': {
            'description': 'AI-powered query interpretation with LLM Cypher generation',
            'features': [
                'Automatic query type detection',
                'LLM-powered Cypher query generation',
                'Natural language understanding',
                'Fallback to keyword-based processing',
                'Advanced entity extraction',
                'Intent classification with confidence scoring',
                'Real-time database querying'
            ]
        },
        'enhanced_features': {
            'description': 'Advanced NLP capabilities',
            'features': [
                'Multi-intent query understanding',
                'Context-aware responses',
                'Query preprocessing and normalization',
                'Advanced pattern matching',
                'Confidence-based result ranking',
                'Comprehensive query analysis',
                'Live database results for all query types'
            ]
        }
    }
    
    return jsonify({
        'success': True,
        'capabilities': capabilities
    })

@app.route('/api/nlp/test', methods=['POST'])
def test_nlp_query():
    """Test a natural language query and return detailed analysis"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        use_enhanced_nlp = data.get('enhanced_nlp', True)
        
        if not user_query:
            return jsonify({
                'success': False,
                'message': 'Please provide a query to test.'
            })
        
        # Get detailed analysis
        analysis = {
            'original_query': user_query,
            'processing_steps': []
        }
        
        # Step 1: Extract locations
        locations = spell_corrector.extract_locations_from_query(user_query)
        analysis['processing_steps'].append({
            'step': 'Location Extraction',
            'locations_found': len(locations),
            'details': [
                {
                    'original': loc[0],
                    'corrected': loc[1],
                    'confidence': loc[2],
                    'method': loc[3]
                } for loc in locations
            ]
        })
        
        # Step 2: Process query with enhanced NLP
        if use_enhanced_nlp:
            result = enhanced_nlp_processor.process_query(user_query)
            analysis['processing_steps'].append({
                'step': 'Enhanced NLP Processing',
                'success': result.get('success', False),
                'query_type': result.get('query_type', 'unknown'),
                'message': result.get('message', ''),
                'confidence': result.get('query_analysis', {}).get('confidence', 0),
                'intent': result.get('query_analysis', {}).get('intent', {}),
                'entities': result.get('query_analysis', {}).get('entities', {})
            })
        else:
            result = query_processor.process_query(user_query)
            analysis['processing_steps'].append({
                'step': 'Basic Query Processing',
                'success': result.get('success', False),
                'query_type': result.get('query_type', 'unknown'),
                'message': result.get('message', ''),
                'cypher_query': result.get('cypher_query', ''),
                'corrections': result.get('corrections', [])
            })
        
        # Step 3: Results
        if result.get('success') and result.get('data'):
            analysis['processing_steps'].append({
                'step': 'Database Results',
                'results_count': len(result['data']),
                'sample_results': result['data'][:3]  # Show first 3 results
            })
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error testing NLP query: {str(e)}'
        })

@app.route('/api/nlp/demo')
def get_nlp_demo():
    """Get a comprehensive demo of natural language capabilities"""
    demo_queries = [
        {
            'category': 'Basic Fare Queries',
            'queries': [
                'What is the fare from Colombo to Kandy?',
                'fare of anuradhapura to kandy',
                'price from panadura to galle',
                'Colombo to Kandy fare'
            ]
        },
        {
            'category': 'Comparison Queries',
            'queries': [
                'Compare fares from Colombo to Kandy vs Colombo to Galle',
                'Which is cheaper between Colombo to Kandy and Colombo to Anuradapura?',
                'What is the difference in fare between Panadura to Galle and Panadura to Matara?'
            ]
        },
        {
            'category': 'Range Search Queries',
            'queries': [
                'Find routes under 500 rupees',
                'Show me routes between 200 and 800 rupees',
                'Routes over 1000 rupees'
            ]
        },
        {
            'category': 'Recommendation Queries',
            'queries': [
                'Recommend cheap routes',
                'Show me popular destinations',
                'What are the best routes from Colombo?'
            ]
        },
        {
            'category': 'Statistical Queries',
            'queries': [
                'What is the average fare?',
                'Database statistics',
                'How many routes are there?'
            ]
        },
        {
            'category': 'Route Queries',
            'queries': [
                'Show me the cheapest routes',
                'Routes from Colombo',
                'Routes to Galle',
                'What routes depart from Kandy?'
            ]
        },
        {
            'category': 'Spell Correction Tests',
            'queries': [
                'price from panadra to gale',
                'fare of colmbo to kandee',
                'cost from anuradapura to kandy'
            ]
        }
    ]
    
    return jsonify({
        'success': True,
        'demo': {
            'title': 'Enhanced Natural Language Transport Query Demo',
            'description': 'Advanced NLP capabilities with comparison, range search, and recommendations',
            'categories': demo_queries
        }
    })

@app.route('/api/examples')
def get_examples():
    """Get comprehensive example queries showcasing natural language capabilities"""
    examples = [
        # === FARE QUERIES (Various Natural Language Formats) ===
        {
            'category': 'Fare Queries',
            'examples': [
                {
                    # 'query': 'What is the fare from Colombo to Kandy?',
                    'query': 'කොළඹ සිට මහනුවරට ගාස්තුව කීයද?',
                    'description': 'Standard fare query format'
                },
                {
                    'query': 'පානදුරේ ඉඳන් ගාල්ලට කීයක් යනවද?',
                    'description': 'Alternative way to ask for fare'
                },
                {
                    'query': 'අනුරාධපුර සිට මහනුවර දක්වා ගාස්තුව',
                    'description': 'Natural language format'
                },
                {
                    # 'query': 'price from panadura to galle',
                    'query': 'පානදුරේ ඉඳන් ගාල්ලට කීයක් යනවද?',
                    'description': 'Using "price" instead of "fare"'
                },
                {
                    # 'query': 'Colombo to nuwara eliya fare',
                    'query': 'බදුල්ල සිට කොළඹට ගාස්තුව කීයද?',
                    'description': 'Compact format'
                },
                {
                    # 'query': 'How much is the fare from matara to kandy?',
                    'query': 'මහනුවර සිට මාතරට ගාස්තුව කීයද?',
                    'description': 'Question format'
                }
            ]
        },
        
        # === COMPARISON QUERIES ===
        {
            'category': 'Comparison Queries',
            'examples': [
                {
                    # 'query': 'Compare fares from Colombo to Kandy vs Colombo to Galle',
                    'query': 'කොළඹ සිට මහනුවර දක්වා සහ කොළඹ සිට ගාල්ල දක්වා ගාස්තු සංසන්දනය කරන්න.',
                    'description': 'Compare two different routes'
                },
                {
                    # 'query': 'Which is cheaper between Colombo to Kandy and Colombo to Anuradapura?',
                    'query': 'කොළඹ සිට මහනුවර දක්වා සහ කොළඹ සිට අනුරාධපුර දක්වා ලාභදායී වන්නේ කුමක්ද?',
                    'description': 'Find the cheaper option'
                },
                {
                    # 'query': 'What is the difference in fare between Panadura to Galle and Panadura to Matara?',
                    'query': 'පානදුර සිට ගාල්ල දක්වා සහ පානදුර සිට මාතර දක්වා ගාස්තුවේ වෙනස කීයද?',
                    'description': 'Calculate fare difference'
                }
            ]
        },
        
        # === RANGE SEARCH QUERIES ===
        {
            'category': 'Range Search Queries',
            'examples': [
                {
                    # 'query': 'Find routes under 500 rupees',
                    'query': 'රුපියල් 500ට අඩු මාර්ග සොයා ගන්න',
                    'description': 'Find affordable routes'
                },
                {
                    # 'query': 'Show me routes between 200 and 800 rupees',
                    'query': 'රුපියල් 200 සහ 800 අතර මාර්ග සොයා ගන්න',
                    'description': 'Find routes in price range'
                },
                {
                    # 'query': 'Routes over 1000 rupees',
                    'query': 'රුපියල් 1000ට ඉහළ මාර්ග සොයා ගන්න',
                    'description': 'Find expensive routes'
                }
            ]
        },
        
        # === RECOMMENDATION QUERIES ===
        {
            'category': 'Recommendation Queries',
            'examples': [
                {
                    # 'query': 'Recommend cheap routes',
                    'query': 'ලාභ මාර්ග නිර්දේශ කරන්න',
                    'description': 'Get budget-friendly recommendations'
                },
                {
                    # 'query': 'Show me popular destinations',
                    'query': 'මට ජනප්‍රිය ගමනාන්ත පෙන්වන්න',
                    'description': 'Find frequently traveled routes'
                },
                {
                    # 'query': 'What are the best routes from Colombo?',
                    'query': 'කොළඹ සිට යාමට හොඳම මාර්ග මොනවාද?',
                    'description': 'Get optimal route suggestions'
                }
            ]
        },
        
        # === STATISTICAL QUERIES ===
        {
            'category': 'Statistical Queries',
            'examples': [
                {
                    # 'query': 'What is the average fare?',
                    'query': 'සාමාන්‍ය ගාස්තුව කීයද?',
                    'description': 'Get average fare statistics'
                },
                {
                    # 'query': 'Database statistics',
                    'query': 'දත්ත සමුදා සංඛ්යා ලේඛන',
                    'description': 'Get comprehensive database overview'
                },
                {
                    'query': 'මාර්ග කීයක් තිබේද?',
                    'description': 'Count total routes'
                }
            ]
        },
        
        # === ROUTE QUERIES ===
        {
            'category': 'Route Queries',
            'examples': [
                {
                    # 'query': 'Show me the cheapest routes',
                    'query': 'මට ලාභදායී  මාර්ග 10ක්  පෙන්වන්න',
                    'description': 'Find top 10 cheapest routes'
                },
                {
                    # 'query': 'Routes from Colombo',
                    'query': 'කොළඹ සිට යාමට මාර්ග මොනවාද?',
                    'description': 'Find all routes departing from a location'
                },
                {
                    # 'query': 'Routes to Galle',
                    'query': 'ගාල්ල යාමට මාර්ග මොනවාද?',
                    'description': 'Find all routes going to a location'
                },
                {
                    # 'query': 'What routes depart from Kandy?',
                    'query': 'මහනුවර සිට යාමට මාර්ග මොනවාද?',
                    'description': 'Question format for routes'
                }
            ]
        },
        
        # === SPELLING ERROR EXAMPLES ===
        {
            'category': 'Spell Correction Examples',
            'examples': [
                {
                    # 'query': 'price from panadra to gale',
                    'query': 'පාන්දුරේ ඉඳන් ගාල්ල්ට කීයක් යනවද?',
                    'description': 'Test spell correction (Panadura, Galle)'
                },
                {
                    # 'query': 'fare of colmbo to kandee',
                    'query': 'කොළ්බ්හ  සිට මහනුවර්ට ගාස්තුව කීයද?',
                    'description': 'Test spell correction (Colombo, Kandy)'
                },
                {
                    # 'query': 'cost from anuradapura to kandy',
                    'query': 'අනුරපුර සිට මහනුවර්රට ගාස්තුව කීයද?',
                    'description': 'Natural format with correct spelling'
                }
            ]
        }
    ]
    
    return jsonify({
        'success': True,
        'examples': examples
    })

@app.route('/api/nlp/advanced', methods=['POST'])
def advanced_nlp_query():
    """Advanced NLP query processing with detailed analysis"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({
                'success': False,
                'message': 'Please provide a query to process.'
            })
        
        # Process with enhanced NLP
        result = enhanced_nlp_processor.process_query(user_query)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing advanced NLP query: {str(e)}'
        })

@app.route('/api/nlp/compare', methods=['POST'])
def compare_routes():
    """Compare multiple routes"""
    try:
        data = request.get_json()
        routes = data.get('routes', [])
        
        if len(routes) < 2:
            return jsonify({
                'success': False,
                'message': 'Please provide at least 2 routes to compare.'
            })
        
        # Build comparison query
        comparison_query = "MATCH "
        for i, route in enumerate(routes):
            from_loc = route.get('from')
            to_loc = route.get('to')
            if from_loc and to_loc:
                if i > 0:
                    comparison_query += ", "
                comparison_query += f"(a{i}:Place {{name: '{from_loc}'}})-[r{i}:Fare]->(b{i}:Place {{name: '{to_loc}'}})"
        
        comparison_query += " RETURN "
        for i, route in enumerate(routes):
            if i > 0:
                comparison_query += ", "
            comparison_query += f"a{i}.name + ' to ' + b{i}.name as route{i+1}, r{i}.fare as fare{i+1}"
        
        # Execute query
        with neo4j_service.driver.session() as session:
            result = session.run(comparison_query)
            results = [dict(record) for record in result]
        
        return jsonify({
            'success': True,
            'data': results,
            'message': f'Comparison of {len(routes)} routes completed'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error comparing routes: {str(e)}'
        })

@app.route('/api/nlp/range', methods=['POST'])
def search_by_range():
    """Search routes by price range"""
    try:
        data = request.get_json()
        min_price = data.get('min_price')
        max_price = data.get('max_price')
        
        if min_price is None and max_price is None:
            return jsonify({
                'success': False,
                'message': 'Please provide min_price or max_price or both.'
            })
        
        # Build range query
        range_query = "MATCH (a:Place)-[r:Fare]->(b:Place) WHERE "
        conditions = []
        
        if min_price is not None:
            conditions.append(f"r.fare >= {min_price}")
        if max_price is not None:
            conditions.append(f"r.fare <= {max_price}")
        
        range_query += " AND ".join(conditions)
        range_query += " RETURN a.name as from_place, b.name as to_place, r.fare as fare ORDER BY r.fare"
        
        # Execute query
        with neo4j_service.driver.session() as session:
            result = session.run(range_query)
            results = [dict(record) for record in result]
        
        return jsonify({
            'success': True,
            'data': results,
            'message': f'Found {len(results)} routes in the specified range'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error searching by range: {str(e)}'
        })

@app.route('/api/nlp/test-all-types')
def test_all_query_types():
    """Test all query types with live results from Neo4j database"""
    try:
        # Define test queries for each type
        test_queries = {
            'fare_inquiry': [
                'What is the fare from Colombo to Kandy?',
                'fare of anuradhapura to kandy',
                'price from panadura to galle'
            ],
            'comparison': [
                'Compare fares from Colombo to Kandy vs Colombo to Galle',
                'Which is cheaper between Colombo to Kandy and Colombo to Anuradapura?'
            ],
            'range_search': [
                'Find routes under 500 rupees',
                'Show me routes between 200 and 800 rupees',
                'Routes over 1000 rupees'
            ],
            'recommendation': [
                'Recommend cheap routes',
                'Show me popular destinations',
                'What are the best routes from Colombo?'
            ],
            'route_inquiry': [
                'Routes from Colombo',
                'Routes to Galle',
                'What routes depart from Kandy?'
            ],
            'statistics': [
                'What is the average fare?',
                'Database statistics',
                'How many routes are there?'
            ]
        }
        
        results = {}
        
        for query_type, queries in test_queries.items():
            type_results = []
            for query in queries:
                try:
                    # Process with enhanced NLP (uses LLM for Cypher generation)
                    result = enhanced_nlp_processor.process_query(query)
                    type_results.append({
                        'query': query,
                        'result': result,
                        'success': result.get('success', False)
                    })
                except Exception as e:
                    type_results.append({
                        'query': query,
                        'result': {
                            'success': False,
                            'message': f'Error processing query: {str(e)}'
                        },
                        'success': False
                    })
            
            results[query_type] = {
                'description': f'Test results for {query_type} queries',
                'total_queries': len(queries),
                'successful_queries': sum(1 for r in type_results if r['success']),
                'examples': type_results
            }
        
        # Summary statistics
        total_queries = sum(len(queries) for queries in test_queries.values())
        total_successful = sum(
            results[query_type]['successful_queries'] 
            for query_type in results
        )
        
        return jsonify({
            'success': True,
            'message': f'Tested {total_queries} queries across {len(test_queries)} types. {total_successful} successful.',
            'summary': {
                'total_query_types': len(test_queries),
                'total_queries_tested': total_queries,
                'successful_queries': total_successful,
                'success_rate': round((total_successful / total_queries) * 100, 2) if total_queries > 0 else 0
            },
            'results': results,
            'neo4j_connected': neo4j_service.is_connected()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error testing query types: {str(e)}',
            'neo4j_connected': neo4j_service.is_connected()
        })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    
    print("🚌 Natural Language Transport Query System")
    print("=" * 60)
    print(f"🚀 Starting on port {port}")
    print(f"🌐 Open your browser and go to: http://localhost:{port}")
    
    # Check Neo4j connection
    if neo4j_service.is_connected():
        print("✅ Connected to Neo4j database")
        stats = neo4j_service.get_route_statistics()
        if stats:
            print(f"📊 Database: {stats.get('total_places', 0)} places, {stats.get('total_routes', 0)} routes")
    else:
        print("⚠️  Neo4j not connected - some features may not work")
    
    # Check LLM availability
    if spell_corrector.llm_available:
        print("🤖 LLM integration available for spell correction")
    else:
        print("⚠️  LLM not available - using fuzzy matching only")
    
    print("\n🎯 Enhanced Natural Language Capabilities:")
    print("   • Multiple query formats (fare, price, cost)")
    print("   • Natural language patterns (from X to Y, X to Y fare)")
    print("   • Question formats (What is, How much, Show me)")
    print("   • Compact formats (Colombo to Kandy fare)")
    print("   • Spell correction and fuzzy matching")
    print("   • LLM-powered query interpretation")
    print("   • Automatic Cypher query generation")
    print("   • Advanced intent classification")
    print("   • Entity extraction and normalization")
    print("   • Comparison queries (vs, versus, compare)")
    print("   • Range search queries (under, over, between)")
    print("   • Recommendation queries (recommend, suggest)")
    print("   • Confidence scoring for query understanding")
    print("   • Sinhala language support with translation")
    print("   • Automatic Sinhala-English translation")
    print("   • Dictionary-based and Google Translate fallback")
    
    print("\n🔗 Available API Endpoints:")
    print("   • /api/query - Process natural language queries (enhanced NLP)")
    print("   • /api/nlp/capabilities - View enhanced NLP capabilities with live examples")
    print("   • /api/nlp/test-all-types - Test all query types with live results")
    print("   • /api/nlp/test - Test queries with detailed analysis")
    print("   • /api/nlp/demo - Get comprehensive demo queries")
    print("   • /api/examples - Get categorized example queries")
    print("   • /api/sinhala/examples - Get Sinhala example queries")
    print("   • /api/translation/test - Test translation functionality")
    print("   • /api/translation/translate - Translate text between languages")
    print("   • /api/status - System status and statistics")
    print("   • /api/suggestions - Get location suggestions")
    print("   • /api/places - Get all available places")
    
    print("=" * 60)
    
    try:
        app.run(debug=True, port=port, host='0.0.0.0')
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Try running as administrator or check if another application is using the port")



#!/usr/bin/env python3
"""
Enhanced NLP Processor for Transport Query Application
Advanced natural language understanding and query processing
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from spell_corrector import SpellCorrector
from neo4j_service import Neo4jService
from config import Config
from logger import get_logger

class EnhancedNLPProcessor:
    """Advanced NLP processor with sophisticated query understanding"""
    
    def __init__(self):
        self.config = Config()
        self.spell_corrector = SpellCorrector()
        self.neo4j_service = Neo4jService()
        self.logger = get_logger(self.__class__.__name__)
        
        # Query patterns and templates
        self.query_patterns = {
            'fare_queries': [
                r'(?:what\s+is\s+)?(?:the\s+)?(?:fare|price|cost)(?:\s+of)?(?:\s+from)?\s+([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'(?:what\s+is\s+)?(?:the\s+)?(?:bus\s+)?(?:fare|price|cost)(?:\s+of)?(?:\s+from)?\s+([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'(?:how\s+much\s+)?(?:is|does)\s+(?:the\s+)?(?:bus\s+)?(?:fare|price|cost)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)\s+(?:fare|price|cost)',
                r'(?:fare|price|cost)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'(?:travel|transport)\s+(?:cost|price|fare)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'(?:bus|train)\s+(?:fare|price|cost)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'(?:ticket\s+price|ticket\s+fare)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)'
            ],
            'comparison_queries': [
                r'(?:compare|difference)\s+(?:between\s+)?(?:fares?|prices?|costs?)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)\s+(?:and|vs|versus)\s+([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'(?:which\s+is\s+)?(?:cheaper|more\s+expensive)\s+(?:between\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)\s+(?:and|vs|versus)\s+([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)'
            ],
            'range_queries': [
                r'(?:routes?|fares?|prices?)\s+(?:between|from)\s+([0-9,]+)\s+(?:and|to)\s+([0-9,]+)\s+(?:rupees?|rs?)',
                r'(?:find|show)\s+(?:routes?|fares?|prices?)\s+(?:under|below|less\s+than)\s+([0-9,]+)\s+(?:rupees?|rs?)',
                r'(?:find|show)\s+(?:routes?|fares?|prices?)\s+(?:over|above|more\s+than)\s+([0-9,]+)\s+(?:rupees?|rs?)'
            ],
            'route_queries': [
                r'(?:routes?|buses?|trains?)\s+(?:from|departing\s+from)\s+([a-zA-Z\s]+)',
                r'(?:routes?|buses?|trains?)\s+(?:to|arriving\s+at)\s+([a-zA-Z\s]+)',
                r'(?:how\s+many\s+)?(?:routes?|buses?|trains?)\s+(?:connect|go\s+to|from)\s+([a-zA-Z\s]+)',
                r'(?:direct|non-stop)\s+(?:routes?|buses?|trains?)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)'
            ],
            'statistical_queries': [
                r'(?:average|mean|median)\s+(?:fare|price|cost)',
                r'(?:total|sum)\s+(?:of\s+)?(?:all\s+)?(?:fares?|prices?|costs?)',
                r'(?:how\s+many\s+)?(?:routes?|places?|locations?)',
                r'(?:database|system)\s+(?:statistics?|stats?|overview)',
                r'(?:summary|overview)\s+(?:of\s+)?(?:transport|fare)\s+(?:data|database)'
            ],
            'recommendation_queries': [
                r'(?:recommend|suggest)\s+(?:cheap|budget|affordable)\s+(?:routes?|options?)',
                r'(?:best|optimal)\s+(?:route|way)\s+(?:from\s+)?([a-zA-Z\s]+)\s+(?:to|→|->)\s+([a-zA-Z\s]+)',
                r'(?:popular|frequent)\s+(?:routes?|destinations?)',
                r'(?:hidden|secret|unknown)\s+(?:routes?|destinations?)'
            ]
        }
        
        # Query intent classification
        self.intent_keywords = {
            'fare_inquiry': ['fare', 'price', 'cost', 'how much', 'what is the cost'],
            'route_inquiry': ['route', 'bus', 'train', 'transport', 'how to get', 'way to'],
            'comparison': ['compare', 'difference', 'vs', 'versus', 'which is', 'better'],
            'statistics': ['statistics', 'stats', 'overview', 'summary', 'total', 'average'],
            'recommendation': ['recommend', 'suggest', 'best', 'optimal', 'popular'],
            'range_search': ['between', 'under', 'over', 'above', 'below', 'range'],
            'availability': ['available', 'exist', 'have', 'is there', 'can i']
        }
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process natural language query with advanced NLP understanding
        
        Args:
            user_query: Natural language query string
            
        Returns:
            Dictionary with comprehensive query analysis and results
        """
        try:
            # Step 1: Preprocess query
            processed_query = self._preprocess_query(user_query)
            self.logger.info(f"Processing query: original='{user_query}', preprocessed='{processed_query}'")
            
            # Step 2: Extract entities and intent
            entities = self._extract_entities(processed_query)
            intent = self._classify_intent(processed_query, entities)
            
            # Step 3: Generate Cypher query
            cypher_query = self._generate_cypher_query(intent, entities, processed_query)
            self.logger.debug(f"Intent: {intent}; Entities: {entities}; Cypher: {str(cypher_query).strip()[:200]}")
            

            
            # Step 4: Execute query and format results
            if cypher_query:
                results = self._execute_query(cypher_query)
                self.logger.info(f"Query results count: {len(results)}")
                response = self._format_response(intent, entities, results, processed_query)
            else:
                response = self._handle_unclear_query(processed_query)
            
            # Step 5: Add metadata
            response.update({
                'query_analysis': {
                    'original_query': user_query,
                    'processed_query': processed_query,
                    'intent': intent,
                    'entities': entities,
                    'confidence': self._calculate_confidence(intent, entities)
                }
            })
            
            return response
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing query: {str(e)}',
                'suggestions': self._get_suggestions()
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and normalize the query"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Normalize common variations
        replacements = {
            'rs.': 'rupees',
            'rs': 'rupees',
            'lkr': 'rupees',
            '→': 'to',
            '->': 'to',
            'vs': 'versus',
            '&': 'and',
            'w/': 'with',
            'w/o': 'without'
        }
        
        for old, new in replacements.items():
            query = query.replace(old, new)
        
        return query
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from the query"""
        entities = {
            'locations': [],
            'numbers': [],
            'currencies': [],
            'comparators': [],
            'time_expressions': []
        }
        
        # Extract locations with priority for different query types
        comparison_patterns = [
            r'(?:which\s+is\s+)?(?:cheaper|more\s+expensive)\s+(?:between\s+)?([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)\s+(?:and|vs|versus)\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            r'(?:what\s+is\s+)?(?:the\s+)?(?:difference|compare)\s+(?:in\s+)?(?:fare|price|cost)\s+(?:between\s+)?([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)\s+(?:and|vs|versus)\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            r'(?:compare|difference)\s+(?:between\s+)?(?:fares?|prices?|costs?)\s+(?:from\s+)?([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)\s+(?:and|vs|versus)\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            # Simpler patterns for comparison
            r'([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)\s+(?:and|vs|versus)\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            r'([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)\s+(?:and|vs|versus)\s+([a-zA-Z\s]+?)(?:\s|$|\?)'
        ]
        
        fare_patterns = [
            r'(?:fare|price|cost)\s+(?:of|from)?\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            r'(?:what\s+is\s+)?(?:the\s+)?(?:fare|price|cost)(?:\s+of)?(?:\s+from)?\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            r'(?:how\s+much\s+)?(?:is|does)\s+(?:the\s+)?(?:fare|price|cost)\s+(?:from\s+)?([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)'
        ]
        
        general_patterns = [
            r'from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            r'([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s|$|\?)',
            r'between\s+([a-zA-Z\s]+?)\s+and\s+([a-zA-Z\s]+?)(?:\s|$|\?)'
        ]
        
        # Use a set to avoid duplicates
        seen_locations = set()
        
        # Try comparison patterns first (highest priority)
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                locations = [loc.strip() for loc in match.groups() if loc.strip()]
                for loc in locations:
                    # Skip if we've already processed this location
                    if loc.lower() in seen_locations:
                        continue
                    seen_locations.add(loc.lower())
                    
                    corrected, confidence, method = self.spell_corrector.correct_location(loc)
                    if confidence > 0.5:
                        entities['locations'].append({
                            'original': loc,
                            'corrected': corrected,
                            'confidence': confidence,
                            'method': method
                        })
        
        # If no locations found with comparison patterns, try fare patterns
        if not entities['locations']:
            for pattern in fare_patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    locations = [loc.strip() for loc in match.groups() if loc.strip()]
                    for loc in locations:
                        # Skip if we've already processed this location
                        if loc.lower() in seen_locations:
                            continue
                        seen_locations.add(loc.lower())
                        
                        corrected, confidence, method = self.spell_corrector.correct_location(loc)
                        if confidence > 0.5:
                            entities['locations'].append({
                                'original': loc,
                                'corrected': corrected,
                                'confidence': confidence,
                                'method': method
                            })
        
        # If no locations found with fare patterns, try general patterns
        if not entities['locations']:
            for pattern in general_patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    locations = [loc.strip() for loc in match.groups() if loc.strip()]
                    for loc in locations:
                        # Skip if we've already processed this location
                        if loc.lower() in seen_locations:
                            continue
                        seen_locations.add(loc.lower())
                        
                        corrected, confidence, method = self.spell_corrector.correct_location(loc)
                        if confidence > 0.5:
                            entities['locations'].append({
                                'original': loc,
                                'corrected': corrected,
                                'confidence': confidence,
                                'method': method
                            })
        

        

        
        # Extract numbers and currencies
        number_patterns = [
            r'(under|below|less\s+than|over|above|more\s+than)\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(rupees?|rs?|lkr)?',
            r'between\s+(\d+(?:,\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:,\d+)*(?:\.\d+)?)\s*(rupees?|rs?|lkr)?',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(rupees?|rs?|lkr)?'
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    if groups[0] in ['under', 'below', 'less than', 'over', 'above', 'more than']:
                        # Pattern: (under|below|less than|over|above|more than) (number) (currency)
                        comparator = groups[0]
                        number = groups[1]
                        currency = groups[2] if len(groups) >= 3 else 'rupees'
                        
                        entities['numbers'].append({
                            'value': float(number.replace(',', '')),
                            'currency': currency,
                            'comparator': comparator
                        })
                    elif 'between' in pattern:
                        # Pattern: between (number1) and (number2) (currency)
                        min_number = groups[0]
                        max_number = groups[1]
                        currency = groups[2] if len(groups) >= 3 else 'rupees'
                        
                        entities['numbers'].append({
                            'value': float(min_number.replace(',', '')),
                            'currency': currency,
                            'comparator': 'between_min'
                        })
                        entities['numbers'].append({
                            'value': float(max_number.replace(',', '')),
                            'currency': currency,
                            'comparator': 'between_max'
                        })
                    else:
                        # Pattern: (number) (currency)
                        number = groups[0]
                        currency = groups[1] if len(groups) >= 2 else 'rupees'
                        
                        entities['numbers'].append({
                            'value': float(number.replace(',', '')),
                            'currency': currency,
                            'comparator': None
                        })
        
        # Extract comparators
        comparator_patterns = [
            r'(cheaper|more\s+expensive|better|worse|faster|slower)',
            r'(compare|difference|vs|versus)',
            r'(under|below|less\s+than|over|above|more\s+than)'
        ]
        
        for pattern in comparator_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities['comparators'].append(match.group(1).lower())
        
        return entities
    
    def _classify_intent(self, query: str, entities: Dict = None) -> Dict[str, Any]:
        """Classify the intent of the query"""
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            intent_scores[intent] = score
        
        # Get primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Check for specific patterns with priority
        if any(pattern in query for pattern in ['compare', 'difference', 'vs', 'versus', 'cheaper', 'more expensive']):
            primary_intent = ('comparison', 10)
        elif any(pattern in query for pattern in ['recommend', 'suggest', 'best', 'optimal', 'popular']):
            primary_intent = ('recommendation', 10)
        elif any(pattern in query for pattern in ['between', 'under', 'over', 'above', 'below', 'range']):
            primary_intent = ('range_search', 10)
        elif any(pattern in query for pattern in ['fare', 'price', 'cost', 'how much']):
            # Check if we have at least 2 locations
            if entities and len(entities.get('locations', [])) >= 2:
                primary_intent = ('fare_inquiry', 10)
        elif any(pattern in query for pattern in ['route', 'bus', 'train', 'transport']):
            primary_intent = ('route_inquiry', 10)
        
        return {
            'primary': primary_intent[0],
            'confidence': primary_intent[1] / 10,
            'all_scores': intent_scores
        }
    
    def _generate_cypher_query(self, intent: Dict, entities: Dict, query: str) -> Optional[str]:
        """Generate Cypher query using LLM for better understanding"""
        try:
            # Try LLM-based query generation first
            llm_query = self._generate_cypher_with_llm(query, intent, entities)
            if llm_query:
                return llm_query
        except Exception as e:
            print(f"LLM query generation failed: {e}")
        
        # Fallback to rule-based generation
        primary_intent = intent['primary']
        
        if primary_intent == 'fare_inquiry':
            return self._generate_fare_query(entities)
        elif primary_intent == 'comparison':
            return self._generate_comparison_query(entities)
        elif primary_intent == 'route_inquiry':
            return self._generate_route_query(entities, query)
        elif primary_intent == 'statistics':
            return self._generate_statistics_query(entities)
        elif primary_intent == 'recommendation':
            return self._generate_recommendation_query(entities, query)
        elif primary_intent == 'range_search':
            return self._generate_range_query(entities)
        else:
            return self._generate_fallback_query(query)
    
    def _generate_fare_query(self, entities: Dict) -> Optional[str]:
        """Generate fare inquiry Cypher query"""
        locations = entities.get('locations', [])
        
        if len(locations) >= 2:
            from_loc = locations[0]['corrected']
            to_loc = locations[1]['corrected']
            
            return f"""
            MATCH (a:Place {{name: '{from_loc}'}})-[r:Fare]->(b:Place {{name: '{to_loc}'}})
            RETURN 
                a.name as from_place,
                b.name as to_place,
                r.fare as fare,
                'Direct route' as route_type
            """
        
        return None
    
    def _generate_comparison_query(self, entities: Dict) -> Optional[str]:
        """Generate comparison Cypher query"""
        locations = entities.get('locations', [])
        
        if len(locations) >= 3:
            # Handle case where we have same origin, different destinations
            if len(locations) == 3:
                # Pattern: "Colombo to Kandy and Colombo to Anuradapura"
                route1_from = locations[0]['corrected']
                route1_to = locations[1]['corrected']
                route2_from = locations[0]['corrected']  # Same origin
                route2_to = locations[2]['corrected']
            elif len(locations) >= 4:
                # Pattern: "Colombo to Kandy and Anuradapura to Galle"
                route1_from = locations[0]['corrected']
                route1_to = locations[1]['corrected']
                route2_from = locations[2]['corrected']
                route2_to = locations[3]['corrected']
            else:
                return None
            
            return f"""
            MATCH (a1:Place {{name: '{route1_from}'}})-[r1:Fare]->(b1:Place {{name: '{route1_to}'}})
            MATCH (a2:Place {{name: '{route2_from}'}})-[r2:Fare]->(b2:Place {{name: '{route2_to}'}})
            RETURN 
                a1.name + ' to ' + b1.name as route1,
                r1.fare as fare1,
                a2.name + ' to ' + b2.name as route2,
                r2.fare as fare2,
                r1.fare - r2.fare as difference,
                CASE 
                    WHEN r1.fare < r2.fare THEN 'Route 1 is cheaper'
                    WHEN r1.fare > r2.fare THEN 'Route 2 is cheaper'
                    ELSE 'Both routes have the same fare'
                END as comparison
            """
        
        return None
    
    def _generate_route_query(self, entities: Dict, query: str) -> Optional[str]:
        """Generate route inquiry Cypher query"""
        locations = entities.get('locations', [])
        
        if 'from' in query and locations:
            location = locations[0]['corrected']
            return f"""
            MATCH (a:Place {{name: '{location}'}})-[r:Fare]->(b:Place)
            RETURN 
                a.name as from_place,
                b.name as to_place,
                r.fare as fare
            ORDER BY r.fare
            """
        elif 'to' in query and locations:
            location = locations[0]['corrected']
            return f"""
            MATCH (a:Place)-[r:Fare]->(b:Place {{name: '{location}'}})
            RETURN 
                a.name as from_place,
                b.name as to_place,
                r.fare as fare
            ORDER BY r.fare
            """
        
        return None
    
    def _generate_statistics_query(self, entities: Dict) -> str:
        """Generate statistics Cypher query"""
        return """
        MATCH (p:Place)
        MATCH ()-[r:Fare]->()
        RETURN 
            count(DISTINCT p) as total_places,
            count(r) as total_routes,
            round(avg(r.fare), 2) as average_fare,
            min(r.fare) as minimum_fare,
            max(r.fare) as maximum_fare,
            round(stdDev(r.fare), 2) as fare_standard_deviation
        """
    
    def _generate_recommendation_query(self, entities: Dict, query: str) -> str:
        """Generate recommendation Cypher query"""
        if 'cheap' in query or 'budget' in query or 'affordable' in query:
            return """
            MATCH (a:Place)-[r:Fare]->(b:Place)
            RETURN 
                a.name as from_place,
                b.name as to_place,
                r.fare as fare
            ORDER BY r.fare ASC
            LIMIT 10
            """
        elif 'popular' in query or 'frequent' in query:
            return """
            MATCH (a:Place)-[r:Fare]->(b:Place)
            RETURN 
                a.name as from_place,
                b.name as to_place,
                r.fare as fare
            ORDER BY r.fare DESC
            LIMIT 10
            """
        else:
            return """
            MATCH (a:Place)-[r:Fare]->(b:Place)
            RETURN 
                a.name as from_place,
                b.name as to_place,
                r.fare as fare
            ORDER BY r.fare ASC
            LIMIT 5
            """
    
    def _generate_range_query(self, entities: Dict) -> Optional[str]:
        """Generate range search Cypher query"""
        numbers = entities.get('numbers', [])
        
        if numbers:
            # Check for between range
            between_min = None
            between_max = None
            single_value = None
            single_comparator = None
            
            for number in numbers:
                comparator = number.get('comparator', '')
                value = number['value']
                
                if comparator == 'between_min':
                    between_min = value
                elif comparator == 'between_max':
                    between_max = value
                elif comparator in ['under', 'below', 'less than', 'over', 'above', 'more than']:
                    single_value = value
                    single_comparator = comparator
            
            # Generate query based on type
            if between_min is not None and between_max is not None:
                return f"""
                MATCH (a:Place)-[r:Fare]->(b:Place)
                WHERE r.fare >= {between_min} AND r.fare <= {between_max}
                RETURN 
                    a.name as from_place,
                    b.name as to_place,
                    r.fare as fare
                ORDER BY r.fare ASC
                """
            elif single_value is not None and single_comparator is not None:
                if single_comparator in ['under', 'below', 'less than']:
                    return f"""
                    MATCH (a:Place)-[r:Fare]->(b:Place)
                    WHERE r.fare < {single_value}
                    RETURN 
                        a.name as from_place,
                        b.name as to_place,
                        r.fare as fare
                    ORDER BY r.fare ASC
                    """
                elif single_comparator in ['over', 'above', 'more than']:
                    return f"""
                    MATCH (a:Place)-[r:Fare]->(b:Place)
                    WHERE r.fare > {single_value}
                    RETURN 
                        a.name as from_place,
                        b.name as to_place,
                        r.fare as fare
                    ORDER BY r.fare DESC
                    """
        
        return None
    
    def _generate_cypher_with_llm(self, query: str, intent: Dict, entities: Dict) -> Optional[str]:
        """Generate Cypher query using LLM for better understanding"""
        try:
            if not self.config.OPENAI_API_KEY:
                return None
            
            # Get available places for context
            available_places = list(self.neo4j_service.get_all_places())
            
            # Create comprehensive prompt for Cypher generation
            prompt = f"""
            You are a Neo4j Cypher query generator for a transport database.
            
            Database Schema:
            - Nodes: Place (with property 'name')
            - Relationships: Fare (with property 'fare')
            
            Available Places: {', '.join(available_places[:50])}... (total: {len(available_places)})
            
            User Query: "{query}"
            Detected Intent: {intent.get('primary', 'unknown')}
            Extracted Entities: {entities}
            
            Your task is to generate a valid Cypher query that answers the user's question.
            
            Query Types and Examples:
            
            1. FARE INQUIRY:
               - "What is the fare from Colombo to Kandy?"
               - Cypher: MATCH (a:Place {{name: 'Colombo'}})-[r:Fare]->(b:Place {{name: 'Kandy'}}) RETURN a.name as from_place, b.name as to_place, r.fare as fare
            
            2. COMPARISON:
               - "Compare fares from Colombo to Kandy vs Colombo to Galle"
               - Cypher: MATCH (a1:Place {{name: 'Colombo'}})-[r1:Fare]->(b1:Place {{name: 'Kandy'}}) MATCH (a2:Place {{name: 'Colombo'}})-[r2:Fare]->(b2:Place {{name: 'Galle'}}) RETURN a1.name + ' to ' + b1.name as route1, r1.fare as fare1, a2.name + ' to ' + b2.name as route2, r2.fare as fare2, r1.fare - r2.fare as difference
            
            3. RANGE SEARCH:
               - "Find routes under 500 rupees"
               - Cypher: MATCH (a:Place)-[r:Fare]->(b:Place) WHERE r.fare < 500 RETURN a.name as from_place, b.name as to_place, r.fare as fare ORDER BY r.fare ASC
            
            4. RECOMMENDATION:
               - "Recommend cheap routes"
               - Cypher: MATCH (a:Place)-[r:Fare]->(b:Place) RETURN a.name as from_place, b.name as to_place, r.fare as fare ORDER BY r.fare ASC LIMIT 10
            
            5. STATISTICS:
               - "What is the average fare?"
               - Cypher: MATCH ()-[r:Fare]->() RETURN round(avg(r.fare), 2) as average_fare, min(r.fare) as min_fare, max(r.fare) as max_fare
            
            6. ROUTE INQUIRY:
               - "Routes from Colombo"
               - Cypher: MATCH (a:Place {{name: 'Colombo'}})-[r:Fare]->(b:Place) RETURN a.name as from_place, b.name as to_place, r.fare as fare ORDER BY r.fare
            
            Important Rules:
            1. Always use proper Cypher syntax
            2. Use exact place names from the available places list
            3. For comparisons, use multiple MATCH clauses
            4. For ranges, use WHERE clauses with appropriate operators
            5. For statistics, use aggregation functions
            6. Always include meaningful column aliases
            7. Use ORDER BY for sorted results
            8. Use LIMIT for large result sets
            
            Return ONLY the Cypher query, nothing else. If you cannot generate a valid query, return "FALLBACK".
            """
            
            cypher_query = None
            # Prefer new SDK
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a Cypher query generator. Return only valid Cypher queries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                cypher_query = response.choices[0].message.content.strip()
            except Exception as sdk_err:
                import openai
                try:
                    openai.api_key = self.config.OPENAI_API_KEY
                    response = openai.ChatCompletion.create(
                        model=self.config.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a Cypher query generator. Return only valid Cypher queries."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        temperature=0.1
                    )
                    cypher_query = response.choices[0].message.content.strip()
                except Exception:
                    raise sdk_err
            
            # Validate the response
            if cypher_query.upper() == "FALLBACK":
                return None
            
            # Basic validation - check if it starts with MATCH
            if cypher_query.upper().startswith('MATCH'):
                return cypher_query
            
            return None
            
        except Exception as e:
            print(f"LLM Cypher generation error: {e}")
            return None
    
    def _generate_fallback_query(self, query: str) -> Optional[str]:
        """Generate fallback query when intent is unclear"""
        # Try to extract locations using spell corrector
        locations = self.spell_corrector.extract_locations_from_query(query)
        
        if len(locations) >= 2:
            from_loc = locations[0][1]
            to_loc = locations[1][1]
            return f"""
            MATCH (a:Place {{name: '{from_loc}'}})-[r:Fare]->(b:Place {{name: '{to_loc}'}})
            RETURN 
                a.name as from_place,
                b.name as to_place,
                r.fare as fare
            """
        
        # Additional fallback: direct pattern matching for fare queries
        if 'fare' in query.lower() or 'price' in query.lower() or 'cost' in query.lower():
            import re
            fare_patterns = [
                r'fare\s+(?:of|from)?\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)',
                r'price\s+(?:of|from)?\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)',
                r'cost\s+(?:of|from)?\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)',
                r'(?:what\s+is\s+)?(?:the\s+)?(?:fare|price|cost)(?:\s+of)?(?:\s+from)?\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)',
                r'(?:how\s+much\s+)?(?:is|does)\s+(?:the\s+)?(?:fare|price|cost)\s+(?:from\s+)?([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)'
            ]
            
            for pattern in fare_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    from_loc = match.group(1).strip()
                    to_loc = match.group(2).strip()
                    
                    # Correct locations
                    from_corrected, from_conf, _ = self.spell_corrector.correct_location(from_loc)
                    to_corrected, to_conf, _ = self.spell_corrector.correct_location(to_loc)
                    
                    if from_conf > 0.5 and to_conf > 0.5:
                        return f"""
                        MATCH (a:Place {{name: '{from_corrected}'}})-[r:Fare]->(b:Place {{name: '{to_corrected}'}})
                        RETURN a.name as from_place, b.name as to_place, r.fare as fare
                        """
        
        return None
    
    def _execute_query(self, cypher_query: str) -> List[Dict]:
        """Execute Cypher query and return results"""
        try:
            with self.neo4j_service.driver.session() as session:
                result = session.run(cypher_query)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Query execution error: {e}")
            return []
    
    def _format_response(self, intent: Dict, entities: Dict, results: List[Dict], query: str) -> Dict[str, Any]:
        """Format the response based on intent and results"""
        primary_intent = intent['primary']
        
        if not results:
            return {
                'success': False,
                'message': 'No results found for your query.',
                'suggestions': self._get_suggestions()
            }
        
        if primary_intent == 'fare_inquiry':
            return self._format_fare_response(results, entities)
        elif primary_intent == 'comparison':
            return self._format_comparison_response(results, entities)
        elif primary_intent == 'route_inquiry':
            return self._format_route_response(results, entities)
        elif primary_intent == 'statistics':
            return self._format_statistics_response(results)
        elif primary_intent == 'recommendation':
            return self._format_recommendation_response(results, query)
        elif primary_intent == 'range_search':
            return self._format_range_response(results, entities)
        else:
            return self._format_generic_response(results)
    
    def _format_fare_response(self, results: List[Dict], entities: Dict) -> Dict[str, Any]:
        """Format fare inquiry response"""
        if results:
            result = results[0]
            return {
                'success': True,
                'message': f"The fare from {result['from_place']} to {result['to_place']} is Rs. {result['fare']}",
                'data': results,
                'query_type': 'fare_inquiry',
                'summary': {
                    'from_place': result['from_place'],
                    'to_place': result['to_place'],
                    'fare': result['fare']
                }
            }
        return {'success': False, 'message': 'Fare information not found.'}
    
    def _format_comparison_response(self, results: List[Dict], entities: Dict) -> Dict[str, Any]:
        """Format comparison response"""
        if results:
            result = results[0]
            return {
                'success': True,
                'message': result.get('comparison', 'Comparison completed'),
                'data': results,
                'query_type': 'comparison',
                'summary': {
                    'route1': result.get('route1'),
                    'route2': result.get('route2'),
                    'difference': result.get('difference')
                }
            }
        return {'success': False, 'message': 'Comparison not possible.'}
    
    def _format_route_response(self, results: List[Dict], entities: Dict) -> Dict[str, Any]:
        """Format route inquiry response"""
        return {
            'success': True,
            'message': f"Found {len(results)} routes",
            'data': results,
            'query_type': 'route_inquiry',
            'summary': {
                'total_routes': len(results),
                'fare_range': f"Rs. {min(r['fare'] for r in results)} - Rs. {max(r['fare'] for r in results)}" if results else "N/A"
            }
        }
    
    def _format_statistics_response(self, results: List[Dict]) -> Dict[str, Any]:
        """Format statistics response"""
        if results:
            stats = results[0]
            return {
                'success': True,
                'message': f"Database contains {stats['total_places']} places and {stats['total_routes']} routes",
                'data': results,
                'query_type': 'statistics',
                'summary': {
                    'total_places': stats['total_places'],
                    'total_routes': stats['total_routes'],
                    'average_fare': stats['average_fare'],
                    'fare_range': f"Rs. {stats['minimum_fare']} - Rs. {stats['maximum_fare']}"
                }
            }
        return {'success': False, 'message': 'Statistics not available.'}
    
    def _format_recommendation_response(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """Format recommendation response"""
        return {
            'success': True,
            'message': f"Here are {len(results)} recommended routes",
            'data': results,
            'query_type': 'recommendation',
            'summary': {
                'recommendations_count': len(results),
                'fare_range': f"Rs. {min(r['fare'] for r in results)} - Rs. {max(r['fare'] for r in results)}" if results else "N/A"
            }
        }
    
    def _format_range_response(self, results: List[Dict], entities: Dict) -> Dict[str, Any]:
        """Format range search response"""
        return {
            'success': True,
            'message': f"Found {len(results)} routes in your specified range",
            'data': results,
            'query_type': 'range_search',
            'summary': {
                'routes_found': len(results),
                'fare_range': f"Rs. {min(r['fare'] for r in results)} - Rs. {max(r['fare'] for r in results)}" if results else "N/A"
            }
        }
    
    def _format_generic_response(self, results: List[Dict]) -> Dict[str, Any]:
        """Format generic response"""
        return {
            'success': True,
            'message': f"Found {len(results)} results",
            'data': results,
            'query_type': 'generic'
        }
    
    def _handle_unclear_query(self, query: str) -> Dict[str, Any]:
        """Handle unclear or ambiguous queries"""
        return {
            'success': False,
            'message': 'I could not understand your query. Please try rephrasing it.',
            'suggestions': self._get_suggestions(),
            'query_type': 'unclear'
        }
    
    def _calculate_confidence(self, intent: Dict, entities: Dict) -> float:
        """Calculate confidence score for the query interpretation"""
        confidence = 0.0
        
        # Intent confidence
        confidence += intent.get('confidence', 0) * 0.4
        
        # Entity confidence
        locations = entities.get('locations', [])
        if locations:
            avg_location_confidence = sum(loc['confidence'] for loc in locations) / len(locations)
            confidence += avg_location_confidence * 0.4
        
        # Query complexity bonus
        if len(locations) >= 2:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _get_suggestions(self) -> List[str]:
        """Get query suggestions"""
        return [
            "What is the fare from Colombo to Kandy?",
            "Compare fares from Colombo to Kandy vs Colombo to Galle",
            "Show me routes from Panadura",
            "Find routes under 500 rupees",
            "What are the cheapest routes?",
            "Show me popular destinations",
            "Give me database statistics",
            "Recommend affordable routes"
        ]

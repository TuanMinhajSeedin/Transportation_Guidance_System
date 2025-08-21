#!/usr/bin/env python3
"""
LLM-Based Query Processor for Transport Query Application
Uses AI to interpret queries and generate Cypher queries
"""

import re
from typing import Dict, List, Tuple, Optional
from spell_corrector import SpellCorrector
from neo4j_service import Neo4jService
from config import Config

class LLMQueryProcessor:
    """Process natural language queries using LLM for interpretation and Cypher generation"""
    
    def __init__(self):
        self.config = Config()
        self.spell_corrector = SpellCorrector()
        self.neo4j_service = Neo4jService()
    
    def process_query(self, user_query: str) -> Dict:
        """
        Process a natural language query using LLM for interpretation
        
        Returns:
            Dictionary with query results and metadata
        """
        try:
            # First, extract and correct locations from the query
            locations = self.spell_corrector.extract_locations_from_query(user_query)
            
            # Use LLM to interpret the query and generate Cypher
            interpretation = self._interpret_query_with_llm(user_query, locations)
            
            if interpretation['success']:
                # Execute the generated Cypher query
                result = self._execute_cypher_query(interpretation['cypher_query'])
                
                return {
                    'success': True,
                    'message': interpretation['message'],
                    'cypher_query': interpretation['cypher_query'],
                    'data': result,
                    'corrections': self._format_corrections(locations),
                    'query_type': interpretation['query_type']
                }
            else:
                return {
                    'success': False,
                    'message': interpretation['message'],
                    'suggestions': self._get_query_suggestions()
                }
                
        except Exception as e:
            print(f"Query processing error: {e}")
            return {
                'success': False,
                'message': 'An error occurred while processing your query.',
                'suggestions': self._get_query_suggestions()
            }
    
    def _interpret_query_with_llm(self, query: str, locations: List[Tuple]) -> Dict:
        """Use LLM to interpret the query and generate appropriate Cypher"""
        try:
            if not self.config.OPENAI_API_KEY:
                return self._fallback_interpretation(query, locations)
            
            # Get available places for context
            available_places = list(self.neo4j_service.get_all_places())
            
            # Create comprehensive prompt for query interpretation
            prompt = f"""
            You are an intelligent transport query interpreter for a Neo4j database containing Sri Lankan transport data.
            
            Database Schema:
            - Nodes: Place (with property 'name')
            - Relationships: Fare (with property 'fare')
            
            Available Places: {', '.join(available_places[:50])}... (total: {len(available_places)})
            
            User Query: "{query}"
            
            Extracted Locations: {[f"{orig}->{corr}" for orig, corr, conf, method in locations]}
            
            Your task is to:
            1. Determine the query type (fare, cheapest, expensive, places, routes_from, routes_to, statistics, lowest_fare)
            2. Generate the appropriate Cypher query
            3. Provide a clear response message
            
                         Query Types:
             - fare: Find fare between two specific locations
             - cheapest: Find cheapest routes (top 10)
             - expensive: Find most expensive routes (top 10)
             - places: List all places
             - routes_from: Find routes departing from a location
             - routes_to: Find routes arriving at a location
             - statistics: Get database statistics
             - lowest_fare: Find the single lowest fare with route details
            
            Return your response in this exact JSON format:
            {{
                "query_type": "fare|cheapest|expensive|places|routes_from|routes_to|statistics|lowest_fare",
                "cypher_query": "MATCH ... RETURN ...",
                "message": "Clear response message for the user"
            }}
            
                         Examples:
             - "What is the fare from Colombo to Kandy?" → fare query: MATCH (a:Place {name: 'Colombo'})-[r:Fare]->(b:Place {name: 'Kandy'}) RETURN a.name as from_place, b.name as to_place, r.fare as fare
             - "fare of anuradhapura to kandy?" → fare query: MATCH (a:Place {name: 'Anuradapura'})-[r:Fare]->(b:Place {name: 'Kandy'}) RETURN a.name as from_place, b.name as to_place, r.fare as fare
             - "Show me the cheapest routes" → cheapest query: MATCH (a:Place)-[r:Fare]->(b:Place) RETURN a.name as from_place, b.name as to_place, r.fare as fare ORDER BY r.fare ASC LIMIT 10
             - "What is the lowest fare?" → lowest_fare query: MATCH (a:Place)-[r:Fare]->(b:Place) RETURN a.name as from_place, b.name as to_place, r.fare as fare ORDER BY r.fare ASC LIMIT 1
             - "List all places" → places query: MATCH (p:Place) RETURN DISTINCT p.name as place ORDER BY p.name
             - "Routes from Colombo" → routes_from query: MATCH (a:Place {name: 'Colombo'})-[r:Fare]->(b:Place) RETURN a.name as from_place, b.name as to_place, r.fare as fare ORDER BY r.fare
             - "Database statistics" → statistics query: MATCH (p:Place) MATCH ()-[r:Fare]->() RETURN count(DISTINCT p) as total_places, count(r) as total_routes, avg(r.fare) as average_fare, min(r.fare) as min_fare, max(r.fare) as max_fare
             
             Keep Cypher queries simple and avoid complex functions like shortestPath. Use direct relationships only.
             
             For fare queries, recognize various formats like "fare of X to Y", "fare from X to Y", "price from X to Y", etc.
            """
            
            # Call LLM using new SDK first, legacy as fallback
            import json
            interpretation = None
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a transport query interpreter. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                interpretation = json.loads(response.choices[0].message.content.strip())
            except Exception as sdk_err:
                try:
                    import openai
                    openai.api_key = self.config.OPENAI_API_KEY
                    response = openai.ChatCompletion.create(
                        model=self.config.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a transport query interpreter. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.1
                    )
                    interpretation = json.loads(response.choices[0].message.content.strip())
                except Exception:
                    raise sdk_err

            # Validate the response
            if interpretation and 'query_type' in interpretation and 'cypher_query' in interpretation and 'message' in interpretation:
                return {
                    'success': True,
                    'query_type': interpretation['query_type'],
                    'cypher_query': interpretation['cypher_query'],
                    'message': interpretation['message']
                }
            else:
                return self._fallback_interpretation(query, locations)
                
        except Exception as e:
            print(f"LLM interpretation error: {e}")
            return self._fallback_interpretation(query, locations)
    
    def _fallback_interpretation(self, query: str, locations: List[Tuple]) -> Dict:
        """Fallback interpretation when LLM is not available"""
        query_lower = query.lower()
        
        # Simple keyword-based interpretation
        if 'lowest' in query_lower or 'minimum' in query_lower or 'cheapest' in query_lower:
            if 'lowest fare' in query_lower or 'minimum fare' in query_lower:
                                 return {
                     'success': True,
                     'query_type': 'lowest_fare',
                     'cypher_query': """
                     MATCH (a:Place)-[r:Fare]->(b:Place)
                     WITH a, b, r, r.fare as fare
                     ORDER BY r.fare ASC
                     LIMIT 1
                     RETURN a.name as from_place, b.name as to_place, fare
                     """,
                     'message': 'Finding the lowest fare in the database...'
                 }
            else:
                return {
                    'success': True,
                    'query_type': 'cheapest',
                    'cypher_query': """
                    MATCH (a:Place)-[r:Fare]->(b:Place)
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare ASC
                    LIMIT 10
                    """,
                    'message': 'Finding the cheapest routes...'
                }
        elif 'expensive' in query_lower or 'highest' in query_lower or 'maximum' in query_lower:
            return {
                'success': True,
                'query_type': 'expensive',
                'cypher_query': """
                MATCH (a:Place)-[r:Fare]->(b:Place)
                RETURN a.name as from_place, b.name as to_place, r.fare as fare
                ORDER BY r.fare DESC
                LIMIT 10
                """,
                'message': 'Finding the most expensive routes...'
            }
        elif 'places' in query_lower or 'locations' in query_lower or 'list all' in query_lower:
            return {
                'success': True,
                'query_type': 'places',
                'cypher_query': """
                MATCH (p:Place)
                RETURN DISTINCT p.name as place
                ORDER BY p.name
                """,
                'message': 'Listing all places...'
            }
        elif 'statistics' in query_lower or 'stats' in query_lower:
            return {
                'success': True,
                'query_type': 'statistics',
                'cypher_query': """
                MATCH (p:Place)
                MATCH ()-[r:Fare]->()
                RETURN 
                    count(DISTINCT p) as total_places,
                    count(r) as total_routes,
                    avg(r.fare) as average_fare,
                    min(r.fare) as min_fare,
                    max(r.fare) as max_fare
                """,
                'message': 'Getting database statistics...'
            }
        elif len(locations) >= 2:
            # Fare query between two locations
            from_location = locations[0][1]
            to_location = locations[1][1]
            return {
                'success': True,
                'query_type': 'fare',
                'cypher_query': f"""
                MATCH (a:Place {{name: '{from_location}'}})-[r:Fare]->(b:Place {{name: '{to_location}'}})
                RETURN a.name as from_place, b.name as to_place, r.fare as fare
                """,
                'message': f'Finding fare from {from_location} to {to_location}...'
            }
        elif 'fare' in query_lower and 'to' in query_lower:
             # Handle queries like "fare of X to Y" where locations might not be extracted properly
             # Try to extract locations using a simpler pattern
             import re
             fare_patterns = [
                 r'fare\s+(?:of|from)?\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
                 r'price\s+(?:of|from)?\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
                 r'cost\s+(?:of|from)?\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
                 r'how\s+much\s+(?:is|does)\s+(?:the\s+)?(?:fare|price|cost)\s+(?:from\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
                 r'what\s+(?:is|are)\s+(?:the\s+)?(?:fare|price|cost)s?\s+(?:from\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
                 r'([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+(?:fare|price|cost)',
                 r'(?:fare|price|cost)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)'
             ]
             
             for pattern in fare_patterns:
                 match = re.search(pattern, query_lower)
                 if match:
                     from_loc = match.group(1).strip()
                     to_loc = match.group(2).strip()
                     
                     # Correct the locations
                     from_corrected, from_conf, _ = self.spell_corrector.correct_location(from_loc)
                     to_corrected, to_conf, _ = self.spell_corrector.correct_location(to_loc)
                     
                     if from_conf > 0.5 and to_conf > 0.5:
                         return {
                             'success': True,
                             'query_type': 'fare',
                             'cypher_query': f"""
                             MATCH (a:Place {{name: '{from_corrected}'}})-[r:Fare]->(b:Place {{name: '{to_corrected}'}})
                             RETURN a.name as from_place, b.name as to_place, r.fare as fare
                             """,
                             'message': f'Finding fare from {from_corrected} to {to_corrected}...'
                         }
        elif len(locations) == 1:
            # Routes from/to a single location
            location = locations[0][1]
            if 'from' in query_lower:
                return {
                    'success': True,
                    'query_type': 'routes_from',
                    'cypher_query': f"""
                    MATCH (a:Place {{name: '{location}'}})-[r:Fare]->(b:Place)
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare
                    """,
                    'message': f'Finding routes from {location}...'
                }
            else:
                return {
                    'success': True,
                    'query_type': 'routes_to',
                    'cypher_query': f"""
                    MATCH (a:Place)-[r:Fare]->(b:Place {{name: '{location}'}})
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare
                    """,
                    'message': f'Finding routes to {location}...'
                }
        else:
            return {
                'success': False,
                'message': 'I could not understand your query. Please try rephrasing it.'
            }
    
    def _execute_cypher_query(self, cypher_query: str) -> List[Dict]:
        """Execute the generated Cypher query"""
        try:
            with self.neo4j_service.driver.session() as session:
                result = session.run(cypher_query)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Cypher execution error: {e}")
            return []
    
    def _format_corrections(self, locations: List[Tuple]) -> List[Dict]:
        """Format location corrections for display"""
        corrections = []
        for original, corrected, confidence, method in locations:
            if original.lower() != corrected.lower():
                corrections.append({
                    'original': original,
                    'corrected': corrected,
                    'confidence': confidence,
                    'method': method
                })
        return corrections
    
    def _get_query_suggestions(self) -> List[str]:
        """Get query suggestions"""
        return [
            "What is the fare from Colombo to Kandy?",
            "What is the lowest fare price?",
            "Show me the cheapest routes",
            "Show me the most expensive routes",
            "List all places",
            "Routes from Panadura",
            "Routes to Galle",
            "Database statistics"
        ]

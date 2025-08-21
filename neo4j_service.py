#!/usr/bin/env python3
"""
Neo4j Service for Transport Query Application
Handles all database operations
"""

from neo4j import GraphDatabase
from typing import List, Dict, Optional, Tuple
from config import Config

class Neo4jService:
    """Neo4j database service"""
    
    def __init__(self):
        self.config = Config()
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.NEO4J_URI,
                auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Connected to Neo4j database")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        return self.driver is not None
    
    def get_fare(self, from_location: str, to_location: str) -> Optional[Dict]:
        """Get fare between two locations"""
        if not self.is_connected():
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Place {name: $from_location})-[r:Fare]->(b:Place {name: $to_location})
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                """, from_location=from_location, to_location=to_location)
                
                record = result.single()
                if record:
                    return {
                        'from_place': record['from_place'],
                        'to_place': record['to_place'],
                        'fare': record['fare']
                    }
                return None
                
        except Exception as e:
            print(f"Error getting fare: {e}")
            return None
    
    def get_all_places(self) -> List[str]:
        """Get all available places"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (p:Place)
                    RETURN DISTINCT p.name as place
                    ORDER BY p.name
                """)
                
                return [record['place'] for record in result]
                
        except Exception as e:
            print(f"Error getting places: {e}")
            return []
    
    def get_routes_from_location(self, from_location: str) -> List[Dict]:
        """Get all routes from a specific location"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Place {name: $from_location})-[r:Fare]->(b:Place)
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare
                """, from_location=from_location)
                
                return [dict(record) for record in result]
                
        except Exception as e:
            print(f"Error getting routes from location: {e}")
            return []
    
    def get_routes_to_location(self, to_location: str) -> List[Dict]:
        """Get all routes to a specific location"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Place)-[r:Fare]->(b:Place {name: $to_location})
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare
                """, to_location=to_location)
                
                return [dict(record) for record in result]
                
        except Exception as e:
            print(f"Error getting routes to location: {e}")
            return []
    
    def get_cheapest_routes(self, limit: int = 10) -> List[Dict]:
        """Get cheapest routes"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Place)-[r:Fare]->(b:Place)
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare ASC
                    LIMIT $limit
                """, limit=limit)
                
                return [dict(record) for record in result]
                
        except Exception as e:
            print(f"Error getting cheapest routes: {e}")
            return []
    
    def get_most_expensive_routes(self, limit: int = 10) -> List[Dict]:
        """Get most expensive routes"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Place)-[r:Fare]->(b:Place)
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare DESC
                    LIMIT $limit
                """, limit=limit)
                
                return [dict(record) for record in result]
                
        except Exception as e:
            print(f"Error getting most expensive routes: {e}")
            return []
    
    def search_routes_by_fare_range(self, min_fare: float, max_fare: float) -> List[Dict]:
        """Search routes within a fare range"""
        if not self.is_connected():
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Place)-[r:Fare]->(b:Place)
                    WHERE r.fare >= $min_fare AND r.fare <= $max_fare
                    RETURN a.name as from_place, b.name as to_place, r.fare as fare
                    ORDER BY r.fare
                """, min_fare=min_fare, max_fare=max_fare)
                
                return [dict(record) for record in result]
                
        except Exception as e:
            print(f"Error searching routes by fare range: {e}")
            return []
    
    def get_route_statistics(self) -> Dict:
        """Get database statistics"""
        if not self.is_connected():
            return {}
        
        try:
            with self.driver.session() as session:
                # Count places
                places_result = session.run("MATCH (p:Place) RETURN count(p) as place_count")
                place_count = places_result.single()['place_count']
                
                # Count routes
                routes_result = session.run("MATCH ()-[r:Fare]->() RETURN count(r) as route_count")
                route_count = routes_result.single()['route_count']
                
                # Average fare
                avg_result = session.run("MATCH ()-[r:Fare]->() RETURN avg(r.fare) as avg_fare")
                avg_fare = avg_result.single()['avg_fare']
                
                # Min and max fares
                fare_range_result = session.run("""
                    MATCH ()-[r:Fare]->()
                    RETURN min(r.fare) as min_fare, max(r.fare) as max_fare
                """)
                fare_range = fare_range_result.single()
                
                return {
                    'total_places': place_count,
                    'total_routes': route_count,
                    'average_fare': round(avg_fare, 2) if avg_fare else 0,
                    'min_fare': fare_range['min_fare'],
                    'max_fare': fare_range['max_fare']
                }
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            self.driver = None

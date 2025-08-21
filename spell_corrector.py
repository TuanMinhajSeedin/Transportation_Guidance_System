#!/usr/bin/env python3
"""
Spell Correction Module for Transport Query Application
Handles location name corrections using fuzzy matching and LLM
"""

import re
from fuzzywuzzy import fuzz
from typing import List, Tuple, Optional
import openai
from config import Config

class SpellCorrector:
    """Spell correction for location names"""
    
    def __init__(self):
        self.config = Config()
        self.location_mapping = self.config.LOCATION_MAPPING
        self.available_locations = set(self.location_mapping.values())
        
        # Initialize OpenAI if API key is available
        if self.config.OPENAI_API_KEY:
            try:
                # Prefer new SDK client if installed; otherwise set legacy api key
                try:
                    from openai import OpenAI  # noqa: F401
                    self.llm_available = True
                except Exception:
                    openai.api_key = self.config.OPENAI_API_KEY
                    self.llm_available = True
            except Exception:
                self.llm_available = False
        else:
            self.llm_available = False
    
    def correct_location(self, location: str) -> Tuple[str, float, str]:
        """
        Correct a location name using multiple methods
        
        Returns:
            Tuple of (corrected_name, confidence_score, correction_method)
        """
        location = location.strip().lower()
        
        # Method 1: Direct mapping
        if location in self.location_mapping:
            corrected = self.location_mapping[location]
            return corrected, 1.0, "direct_mapping"
        
        # Method 2: Fuzzy matching
        best_match, confidence = self._fuzzy_match(location)
        if confidence >= self.config.SIMILARITY_THRESHOLD:
            return best_match, confidence, "fuzzy_matching"
        
        # Method 3: LLM correction (if available)
        if self.llm_available:
            llm_corrected = self._llm_correct(location)
            if llm_corrected:
                # Verify LLM suggestion with fuzzy matching
                llm_confidence = fuzz.ratio(location.lower(), llm_corrected.lower()) / 100
                if llm_confidence >= 0.6:  # Lower threshold for LLM suggestions
                    return llm_corrected, llm_confidence, "llm_correction"
        
        # Method 4: Partial matching
        partial_match = self._partial_match(location)
        if partial_match:
            return partial_match, 0.7, "partial_matching"
        
        # No correction found
        return location.title(), 0.0, "no_correction"
    
    def _fuzzy_match(self, location: str) -> Tuple[str, float]:
        """Find best fuzzy match for location"""
        best_match = None
        best_score = 0
        
        for available_location in self.available_locations:
            score = fuzz.ratio(location.lower(), available_location.lower()) / 100
            if score > best_score:
                best_score = score
                best_match = available_location
        
        return best_match, best_score
    
    def _partial_match(self, location: str) -> Optional[str]:
        """Find partial matches (substring matching)"""
        location_lower = location.lower()
        
        for available_location in self.available_locations:
            available_lower = available_location.lower()
            
            # Check if location is contained in available location
            if location_lower in available_lower or available_lower in location_lower:
                return available_location
        
        return None
    
    def _llm_correct(self, location: str) -> Optional[str]:
        """Use LLM to correct location name"""
        try:
            prompt = f"""
            You are a location name correction system for Sri Lankan cities and towns.
            Given a potentially misspelled location name, return the correct spelling.
            
            Available locations include: {', '.join(sorted(self.available_locations))}
            
            Input location: "{location}"
            
            Return only the corrected location name, nothing else. If no correction is possible, return "UNKNOWN".
            """
            
            corrected = None
            # Try new SDK first
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that corrects location names."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1
                )
                corrected = response.choices[0].message.content.strip()
            except Exception as sdk_err:
                # Fallback to legacy API if present
                import openai
                try:
                    openai.api_key = self.config.OPENAI_API_KEY
                    response = openai.ChatCompletion.create(
                        model=self.config.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that corrects location names."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.1
                    )
                    corrected = response.choices[0].message.content.strip()
                except Exception:
                    raise sdk_err
            
            # Validate LLM response
            if corrected.upper() == "UNKNOWN":
                return None
            
            # Check if corrected location exists in our database
            if corrected in self.available_locations:
                return corrected
            
            # Try fuzzy matching on LLM response
            llm_fuzzy_match, confidence = self._fuzzy_match(corrected)
            if confidence >= 0.8:
                return llm_fuzzy_match
            
            return None
            
        except Exception as e:
            print(f"LLM correction error: {e}")
            return None
    
    def extract_locations_from_query(self, query: str) -> List[Tuple[str, str, float, str]]:
        """
        Extract and correct locations from a natural language query
        
        Returns:
            List of tuples: (original, corrected, confidence, method)
        """
        # Common patterns for location extraction
        patterns = [
             r'from\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'between\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+and\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'fare\s+(?:of|from\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'price\s+(?:of|from\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'cost\s+(?:of|from\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'how\s+much\s+(?:is|does)\s+(?:the\s+)?(?:fare|price|cost)\s+(?:from\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'what\s+(?:is|are)\s+(?:the\s+)?(?:fare|price|cost)s?\s+(?:from\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)',
             r'([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+(?:fare|price|cost)(?:\s|$|\?)',
             r'(?:fare|price|cost)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+to\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)(?:\s|$|\?)'
         ]
        
        locations = []
        
        # Try all patterns to find locations
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Extract locations from the match
                groups = match.groups()
                if len(groups) >= 2:
                    from_location = groups[0].strip()
                    to_location = groups[1].strip()
                    
                    # Skip if locations are too short or common words
                    if len(from_location) >= 2 and from_location.lower() not in ['to', 'from', 'and', 'the', 'a', 'an']:
                        from_corrected, from_confidence, from_method = self.correct_location(from_location)
                        if from_confidence > 0.5:
                            locations.append((
                                from_location,
                                from_corrected,
                                from_confidence,
                                from_method
                            ))
                    
                    if len(to_location) >= 2 and to_location.lower() not in ['to', 'from', 'and', 'the', 'a', 'an']:
                        to_corrected, to_confidence, to_method = self.correct_location(to_location)
                        if to_confidence > 0.5:
                            locations.append((
                                to_location,
                                to_corrected,
                                to_confidence,
                                to_method
                            ))
                    
                    # If we found locations, break to avoid duplicates
                    if len(locations) >= 2:
                        break
        
        return locations
    
    def get_suggestions(self, partial_location: str) -> List[Tuple[str, float]]:
        """Get location suggestions for autocomplete"""
        suggestions = []
        partial_lower = partial_location.lower()
        
        for location in self.available_locations:
            location_lower = location.lower()
            
            # Check if partial location is a prefix
            if location_lower.startswith(partial_lower):
                suggestions.append((location, 1.0))
            # Check fuzzy similarity
            elif fuzz.ratio(partial_lower, location_lower) / 100 >= 0.6:
                suggestions.append((location, fuzz.ratio(partial_lower, location_lower) / 100))
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:self.config.MAX_SUGGESTIONS]
    
    def validate_route(self, from_location: str, to_location: str) -> Tuple[bool, str]:
        """Validate if a route exists in the database"""
        from_corrected, from_confidence, _ = self.correct_location(from_location)
        to_corrected, to_confidence, _ = self.correct_location(to_location)
        
        if from_confidence < 0.5:
            return False, f"Could not identify departure location: '{from_location}'"
        
        if to_confidence < 0.5:
            return False, f"Could not identify destination location: '{to_location}'"
        
        if from_corrected == to_corrected:
            return False, f"Departure and destination cannot be the same: '{from_corrected}'"
        
        return True, f"Route: {from_corrected} → {to_corrected}"

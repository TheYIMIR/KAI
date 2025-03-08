"""
RecursiveKnowledgeExplorer: Comprehensively explores and researches concepts
with optimized performance using Wikipedia and DBpedia as knowledge sources.
Includes GPU acceleration for improved performance on similarity calculations
and text processing.
"""

import time
import logging
import re
import requests
import json
import os
from typing import Dict, List, Tuple, Set, Optional, Any
from urllib.parse import quote_plus
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import wikipedia

# GPU support imports
import torch
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_GPU_SUPPORT = True
except ImportError:
    HAS_GPU_SUPPORT = False

try:
    lemmatizer = WordNetLemmatizer()
except:
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    
# Get the logger
logger = logging.getLogger("KAI")

# Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Pre-load stopwords for faster access
STOP_WORDS = set(stopwords.words('english'))

class SourceCache:
    """Cache for storing retrieved information to avoid redundant API calls"""
    
    def __init__(self, max_size=2000):
        self.cache = {}
        self.max_size = max_size
        self.lock = None  # Will be set in RecursiveKnowledgeExplorer
        self.cache_file = "data/source_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Load cache from disk if available"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                    logger.info(f"Loaded {len(self.cache)} items from cache")
            else:
                self.cache = {}
                logger.info("No cache file found, starting with empty cache")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save cache to disk periodically"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def get(self, key):
        """Get value from cache if it exists"""
        if self.lock:
            with self.lock:
                return self.cache.get(key)
        else:
            return self.cache.get(key)
    
    def set(self, key, value):
        """Set value in cache, evicting old items if needed"""
        if self.lock:
            with self.lock:
                self._set_value(key, value)
        else:
            self._set_value(key, value)
    
    def _set_value(self, key, value):
        """Internal method to set cache value"""
        # Simple eviction strategy: remove random items when cache is full
        if len(self.cache) >= self.max_size:
            # Remove 10% of items
            items_to_remove = max(1, self.max_size // 10)
            for _ in range(items_to_remove):
                if self.cache:
                    self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = value
        
        # Periodically save cache to disk (every 50 new items)
        if len(self.cache) % 50 == 0:
            self.save_cache()


class SourceVerifier:
    """Handles verification of information from Wikipedia and DBpedia"""
    
    def __init__(self):
        self.cache = SourceCache()
        
        # Source weights (prioritize more reliable sources)
        self.source_weights = {
            "Wikipedia": 1.0,
            "DBpedia": 0.8
        }
        
        # Initialize GPU-accelerated embedding model if available
        self.use_gpu = False
        self.embedding_model = None
        
        if HAS_GPU_SUPPORT:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to('cuda')
                    self.use_gpu = True
                    logger.info("Using GPU acceleration for text processing")
                else:
                    logger.info("GPU detected but CUDA not available, using CPU")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU support: {e}")
                self.use_gpu = False
    
    def get_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using GPU acceleration when available
        Returns a score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Use GPU-accelerated embeddings if available
        if self.use_gpu and self.embedding_model:
            try:
                # Compute embeddings
                embeddings = self.embedding_model.encode([text1, text2], convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    embeddings[0].cpu().numpy().reshape(1, -1),
                    embeddings[1].cpu().numpy().reshape(1, -1)
                )[0][0]
                
                return float(similarity)
            except Exception as e:
                logger.warning(f"GPU similarity calculation failed: {e}, falling back to CPU method")
                # Fall back to traditional method
        
        # Traditional method (when GPU not available)
        tokens1 = set(word_tokenize(text1.lower()))
        tokens2 = set(word_tokenize(text2.lower()))
        
        # Remove stopwords
        tokens1 = {t for t in tokens1 if t.isalnum() and t not in STOP_WORDS}
        tokens2 = {t for t in tokens2 if t.isalnum() and t not in STOP_WORDS}
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def batch_calculate_similarities(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Calculate similarities for multiple text pairs efficiently using GPU batching"""
        if not self.use_gpu or not self.embedding_model or not text_pairs:
            # Fall back to individual processing
            return [self.get_similarity_score(t1, t2) for t1, t2 in text_pairs]
        
        try:
            # Extract all texts
            all_texts = []
            for t1, t2 in text_pairs:
                all_texts.append(t1)
                all_texts.append(t2)
            
            # Compute all embeddings in one batch
            with torch.no_grad():
                all_embeddings = self.embedding_model.encode(all_texts, convert_to_tensor=True)
            
            # Calculate similarities
            results = []
            for i in range(len(text_pairs)):
                emb1 = all_embeddings[i*2].cpu().numpy().reshape(1, -1)
                emb2 = all_embeddings[i*2+1].cpu().numpy().reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0][0]
                results.append(float(similarity))
            
            return results
        except Exception as e:
            logger.warning(f"Batch GPU processing failed: {e}")
            # Fall back to individual processing
            return [self.get_similarity_score(t1, t2) for t1, t2 in text_pairs]
    
    def get_multiple_meanings(self, concept: str) -> List[Dict]:
        """
        Find multiple meanings/senses for a concept
        Returns a list of meanings, each with type, definition, and confidence
        """
        meanings = []
        
        # 1. Check if it's a plural form
        singular = lemmatizer.lemmatize(concept.lower(), 'n')
        if singular != concept.lower():
            # Try to get definition of singular form
            singular_result = self.get_wikipedia_definition(singular)
            if singular_result["definition"]:
                # Add the "plural form" meaning
                plural_meaning = {
                    "type": "plural_form",
                    "related_to": singular,
                    "definition": f"Plural form of '{singular}'. " + singular_result["definition"],
                    "url": singular_result["url"],
                    "confidence": 0.9,
                }
                meanings.append(plural_meaning)
        
        # 2. Get Wikipedia search results
        try:
            search_results = wikipedia.search(concept, results=5)
            
            # Process each search result
            for result in search_results:
                try:
                    # Skip if exactly same as singular we already processed
                    if result.lower() == singular.lower() and any(m["type"] == "plural_form" for m in meanings):
                        continue
                        
                    page = wikipedia.page(result, auto_suggest=False)
                    # Check if this is about the concept itself
                    relevance = self._calculate_meaning_relevance(concept, page.title, page.summary)
                    
                    if relevance > 0.3:  # Minimum relevance threshold
                        meaning_type = "entity"
                        # Check if it's a title/named entity vs. a general concept
                        if self._is_likely_title(page.title, page.summary):
                            meaning_type = "named_entity"
                        elif page.title.lower() != concept.lower():
                            meaning_type = "related_concept"
                        
                        meaning = {
                            "type": meaning_type,
                            "title": page.title,
                            "definition": page.summary,
                            "url": page.url,
                            "confidence": relevance,
                        }
                        meanings.append(meaning)
                except Exception as e:
                    logger.debug(f"Error processing search result '{result}': {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error getting search results for '{concept}': {e}")
        
        # 3. Try DBpedia for additional meanings
        try:
            dbpedia_result = self.get_dbpedia_definition(concept)
            if dbpedia_result["definition"] and dbpedia_result["confidence"] > 0.4:
                # Check if this meaning is distinct from what we already have
                is_distinct = True
                for meaning in meanings:
                    if "definition" in meaning:
                        similarity = self.get_similarity_score(
                            dbpedia_result["definition"], 
                            meaning["definition"]
                        )
                        if similarity > 0.7:  # High similarity means it's probably the same meaning
                            is_distinct = False
                            break
                
                if is_distinct:
                    meaning = {
                        "type": "concept",
                        "title": concept,
                        "definition": dbpedia_result["definition"],
                        "url": dbpedia_result["url"],
                        "confidence": dbpedia_result["confidence"],
                    }
                    meanings.append(meaning)
        except Exception as e:
            logger.warning(f"Error getting DBpedia information for '{concept}': {e}")
        
        # 4. Sort by relevance/confidence
        meanings.sort(key=lambda x: x["confidence"], reverse=True)
        
        return meanings
    
    def _calculate_meaning_relevance(self, concept: str, title: str, definition: str) -> float:
        """Calculate how relevant a meaning is to the original concept"""
        concept_lower = concept.lower()
        title_lower = title.lower()
        
        # Exact match with concept gets high relevance
        if concept_lower == title_lower:
            return 0.95
        
        # If title contains the concept
        if concept_lower in title_lower:
            # Calculate what fraction of the title is the concept
            return 0.8 * (len(concept_lower) / len(title_lower))
        
        # Check for concept in first paragraph of definition
        first_paragraph = definition.split('\n')[0].lower()
        if concept_lower in first_paragraph:
            # Position matters - earlier is better
            position = first_paragraph.find(concept_lower)
            if position < 20:
                return 0.7
            elif position < 100:
                return 0.6
            else:
                return 0.5
        
        # If concept appears multiple times in definition
        count = definition.lower().count(concept_lower)
        if count > 2:
            return 0.4
        elif count > 0:
            return 0.3
            
        # Fallback relevance
        return 0.2
    
    def verify_sources_with_meanings(self, concept: str) -> Dict:
        """
        Enhanced version of verify_sources that handles multiple meanings
        Returns a unified definition with metadata about multiple meanings
        """
        # Get multiple meanings for the concept
        meanings = self.get_multiple_meanings(concept)
        
        if not meanings:
            logger.warning(f"No meanings found for '{concept}'")
            return {
                "definition": f"Information about '{concept}' could not be found in reliable sources.",
                "source": "",
                "url": "",
                "confidence": 0.1,
                "meanings": []
            }
        
        # Create a unified definition with all meanings
        definition = f"'{concept}' has multiple meanings:\n\n"
        
        # Add each meaning to the definition
        for i, meaning in enumerate(meanings[:5]):  # Limit to top 5 meanings
            if meaning["type"] == "plural_form":
                definition += f"{i+1}. {meaning['definition'].split('.')[0]}.\n"
            else:
                # Extract first sentence/short summary
                summary = meaning["definition"].split('.')[0]
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                definition += f"{i+1}. {meaning['title']}: {summary}.\n"
        
        # Return unified result with all meanings in metadata
        return {
            "definition": definition,
            "source": "Multiple sources",
            "url": meanings[0]["url"] if "url" in meanings[0] else "",
            "confidence": max(m["confidence"] for m in meanings),
            "meanings": meanings  # Store all meanings in metadata
        }
    
    def _is_likely_title(self, title: str, definition: str) -> bool:
        """Check if definition is about a title/named entity rather than a general concept"""
        # Common patterns for titles and named entities
        if " is a " in definition[:len(title) + 30]:
            entity_indicators = [
                "film", "movie", "novel", "book", "song", "album", 
                "game", "series", "show", "event", "company", "organization",
                "band", "artist", "actor", "actress", "director", "producer"
            ]
            first_sentence = definition.split(".")[0].lower()
            
            # Check for entity indicators in first sentence
            for indicator in entity_indicators:
                if indicator in first_sentence:
                    return True
        
        # Check if title is capitalized (not at beginning of sentence)
        words = title.split()
        if len(words) > 1 and all(w[0].isupper() for w in words if len(w) > 1):
            return True
            
        return False
    
    def concept_appears_in_definition(self, concept: str, definition: str) -> bool:
        """
        Check if the concept actually appears in the definition,
        accounting for variations and forms of the word
        """
        if not concept or not definition:
            return False
            
        # Normalize concept and definition
        concept_lower = concept.lower()
        definition_lower = definition.lower()
        
        # Check for exact concept match
        if re.search(r'\b' + re.escape(concept_lower) + r'\b', definition_lower):
            return True
            
        # Check for plural/singular variations (simple check)
        variations = []
        if concept_lower.endswith('s'):
            variations.append(concept_lower[:-1])  # Remove 's'
        else:
            variations.append(concept_lower + 's')  # Add 's'
            
        # Add other common variations (could be expanded)
        if concept_lower.endswith('y'):
            variations.append(concept_lower[:-1] + 'ies')  # e.g., "technology" -> "technologies"
            
        # Check for variations
        for var in variations:
            if re.search(r'\b' + re.escape(var) + r'\b', definition_lower):
                return True
                
        # If we have a compound concept, check if parts appear (e.g., "artificial intelligence")
        if ' ' in concept_lower:
            parts = concept_lower.split()
            # Check if all significant parts appear
            significant_parts = [p for p in parts if p not in STOP_WORDS]
            if all(re.search(r'\b' + re.escape(p) + r'\b', definition_lower) for p in significant_parts):
                return True
                
        return False
    
    def find_similar_concept(self, concept: str) -> Optional[str]:
        """
        Check if a concept might be a misspelling or variant of a known concept
        Returns the corrected concept if found, or None if no close match
        """
        # Simple edit distance function
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        # Try Wikipedia's search suggestions
        try:
            search_results = wikipedia.search(concept, results=3)
            if search_results:
                # Check edit distance to identify possible corrections
                candidates = []
                for result in search_results:
                    distance = levenshtein_distance(concept.lower(), result.lower())
                    max_allowed_distance = min(3, len(concept) // 3)  # Allow more edits for longer words
                    if distance <= max_allowed_distance:
                        candidates.append((result, distance))
                
                if candidates:
                    # Return the closest match
                    candidates.sort(key=lambda x: x[1])
                    return candidates[0][0]
        except:
            pass
        
        return None
    
    # === WIKIPEDIA SOURCE ===
    
    def get_wikipedia_definition(self, concept: str) -> Dict:
        """Get definition from Wikipedia with caching"""
        # Check cache first
        cache_key = f"wiki_{concept}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # Set timeout 
            wikipedia.timeout = 3.0
            
            # Try to get Wikipedia page
            try:
                page = wikipedia.page(concept, auto_suggest=True)
                summary = page.summary
                url = page.url
                confidence = 0.8  # Wikipedia has decent reliability
                
                # For deeper understanding, get more than just the summary
                try:
                    # Get first few sections of content for a more thorough understanding
                    content = summary
                    sections = page.sections[:3]  # Get first 3 sections
                    for section in sections:
                        section_content = page.section(section)
                        if section_content:
                            content += f"\n\n{section} - {section_content}"
                except:
                    # If we can't get sections, just use the summary
                    content = summary
                    
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation page, try to find the most relevant option
                try:
                    options = e.options
                    best_option = None
                    highest_match = 0
                    
                    for option in options[:5]:  # Check first 5 options
                        match_score = self._calculate_option_match(concept, option)
                        if match_score > highest_match:
                            highest_match = match_score
                            best_option = option
                    
                    if best_option:
                        page = wikipedia.page(best_option)
                        content = page.summary
                        url = page.url
                        confidence = 0.6  # Less confidence due to disambiguation
                    else:
                        raise Exception("No suitable disambiguation option found")
                except Exception as inner_e:
                    logger.warning(f"Failed to resolve disambiguation for '{concept}': {inner_e}")
                    empty_result = {"source": "Wikipedia", "definition": "", "url": "", "confidence": 0.0}
                    self.cache.set(cache_key, empty_result)
                    return empty_result
            except wikipedia.exceptions.PageError:
                logger.warning(f"No Wikipedia page found for '{concept}'")
                empty_result = {"source": "Wikipedia", "definition": "", "url": "", "confidence": 0.0}
                self.cache.set(cache_key, empty_result)
                return empty_result
            
            # Verify the definition is actually about the concept
            if not self.concept_appears_in_definition(concept, content):
                logger.warning(f"Wikipedia definition may not be about '{concept}'")
                confidence = 0.4  # Lower confidence if concept doesn't clearly appear
            
            result = {
                "source": "Wikipedia",
                "definition": content,
                "url": url,
                "confidence": confidence
            }
            
            # Cache the result
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Error getting Wikipedia definition for '{concept}': {e}")
            return {"source": "Wikipedia", "definition": "", "url": "", "confidence": 0.0}
    
    # === DBPEDIA SOURCE ===
    
    def get_dbpedia_definition(self, concept: str) -> Dict:
        """Get definition from DBpedia with caching"""
        # Check cache first
        cache_key = f"dbpedia_{concept}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # Format concept for DBpedia lookup (capitalize first letter of each word)
            formatted_concept = "_".join(word.capitalize() for word in concept.split())
            
            # Query DBpedia
            resource_url = f"https://dbpedia.org/data/{formatted_concept}.json"
            
            response = requests.get(resource_url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                # The resource URI is the key in the JSON
                resource_uri = f"http://dbpedia.org/resource/{formatted_concept}"
                
                if resource_uri in data:
                    resource_data = data[resource_uri]
                    
                    # Extract information
                    definition = ""
                    
                    # Abstract
                    if 'http://dbpedia.org/ontology/abstract' in resource_data:
                        for abstract in resource_data['http://dbpedia.org/ontology/abstract']:
                            if abstract['lang'] == 'en':
                                definition += abstract['value'] + "\n\n"
                    
                    # Type information
                    if 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' in resource_data:
                        types = []
                        for type_info in resource_data['http://www.w3.org/1999/02/22-rdf-syntax-ns#type']:
                            type_uri = type_info['value']
                            # Clean up the URI to get just the type name
                            type_name = type_uri.split('/')[-1]
                            if 'owl#Thing' not in type_name and 'schema.org' not in type_name:
                                types.append(type_name)
                        
                        if types:
                            definition += f"Types: {', '.join(types[:5])}\n"
                    
                    # Other properties
                    interesting_properties = [
                        'http://dbpedia.org/ontology/birthDate',
                        'http://dbpedia.org/ontology/birthPlace',
                        'http://dbpedia.org/ontology/deathDate',
                        'http://dbpedia.org/ontology/field',
                        'http://dbpedia.org/ontology/genre',
                        'http://dbpedia.org/ontology/occupation'
                    ]
                    
                    for prop in interesting_properties:
                        if prop in resource_data:
                            prop_name = prop.split('/')[-1]
                            prop_values = []
                            
                            for value_info in resource_data[prop]:
                                if 'value' in value_info:
                                    value = value_info['value']
                                    if 'http://' in value:
                                        value = value.split('/')[-1].replace('_', ' ')
                                    prop_values.append(value)
                            
                            if prop_values:
                                definition += f"{prop_name}: {', '.join(prop_values[:3])}\n"
                    
                    # Check if we found useful information
                    if definition:
                        result = {
                            "source": "DBpedia",
                            "definition": definition.strip(),
                            "url": f"https://dbpedia.org/page/{formatted_concept}",
                            "confidence": 0.7
                        }
                        
                        # Cache the result
                        self.cache.set(cache_key, result)
                        return result
            
            # If we get here, either the request failed or we didn't find useful information
            # Try a search approach instead
            search_url = f"https://lookup.dbpedia.org/api/search?query={quote_plus(concept)}&format=json"
            
            search_response = requests.get(search_url, timeout=3)
            if search_response.status_code == 200:
                search_data = search_response.json()
                
                if 'docs' in search_data and search_data['docs']:
                    # Get the first result
                    result_data = search_data['docs'][0]
                    
                    # Extract information
                    definition = ""
                    
                    if 'label' in result_data:
                        definition += f"{result_data['label']}: "
                    
                    if 'comment' in result_data:
                        definition += f"{result_data['comment']}\n\n"
                    
                    if 'category' in result_data:
                        definition += f"Categories: {', '.join(result_data['category'])}\n"
                    
                    if 'type' in result_data:
                        definition += f"Types: {', '.join(result_data['type'])}\n"
                    
                    if definition:
                        result = {
                            "source": "DBpedia",
                            "definition": definition.strip(),
                            "url": result_data.get('resource', f"https://dbpedia.org/"),
                            "confidence": 0.6
                        }
                        
                        # Cache the result
                        self.cache.set(cache_key, result)
                        return result
            
            empty_result = {"source": "DBpedia", "definition": "", "url": "", "confidence": 0.0}
            self.cache.set(cache_key, empty_result)
            return empty_result
        except Exception as e:
            logger.warning(f"Error getting DBpedia definition for '{concept}': {e}")
            return {"source": "DBpedia", "definition": "", "url": "", "confidence": 0.0}
    
    # === MAIN VERIFICATION METHOD ===
    
    def _calculate_option_match(self, concept: str, option: str) -> float:
        """Calculate how well a disambiguation option matches the original concept"""
        concept_lower = concept.lower()
        option_lower = option.lower()
        
        # Exact match gets highest score
        if concept_lower == option_lower:
            return 1.0
            
        # If option starts with concept, good match
        if option_lower.startswith(concept_lower):
            return 0.9
            
        # If concept is wholly contained in option
        if concept_lower in option_lower:
            return 0.7
            
        # Calculate word overlap
        concept_words = set(concept_lower.split())
        option_words = set(option_lower.split())
        
        if not concept_words or not option_words:
            return 0.0
            
        overlap = concept_words.intersection(option_words)
        return len(overlap) / max(len(concept_words), len(option_words))
    
    def merge_definitions(self, sources: List[Dict]) -> Dict:
        """Merge multiple source definitions into a comprehensive description"""
        if not sources:
            return {"source": "", "definition": "", "url": "", "confidence": 0.0}
        
        if len(sources) == 1:
            return sources[0]
        
        # For this simplified version, prioritize Wikipedia but incorporate DBpedia data
        # Find Wikipedia source
        wiki_source = next((s for s in sources if s["source"] == "Wikipedia" and s["confidence"] > 0.3), None)
        dbpedia_source = next((s for s in sources if s["source"] == "DBpedia" and s["confidence"] > 0.3), None)
        
        if wiki_source and dbpedia_source:
            # Use Wikipedia as primary with DBpedia supplementary data
            definition = wiki_source["definition"]
            
            # Check if DBpedia adds meaningful information
            wiki_dbpedia_similarity = self.get_similarity_score(wiki_source["definition"], dbpedia_source["definition"])
            
            # Only add DBpedia info if it's not too similar to Wikipedia (avoid redundancy)
            if wiki_dbpedia_similarity < 0.6:
                # Extract additional information from DBpedia that's not already in Wikipedia
                dbpedia_lines = [line for line in dbpedia_source["definition"].split('\n') 
                                if line.startswith(('Types:', 'Categories:', 'birthDate:', 'birthPlace:', 'deathDate:'))]
                
                if dbpedia_lines:
                    definition += "\n\nAdditional information:\n" + "\n".join(dbpedia_lines)
            
            return {
                "source": "Wikipedia and DBpedia",
                "definition": definition,
                "url": wiki_source["url"],
                "confidence": max(wiki_source["confidence"], 0.1 + dbpedia_source["confidence"] * 0.5)
            }
        elif wiki_source:
            return wiki_source
        elif dbpedia_source:
            return dbpedia_source
        else:
            return sources[0] if sources else {"source": "", "definition": "", "url": "", "confidence": 0.0}
    
    def extract_key_concepts(self, text: str, max_concepts: int = 7) -> List[str]:
        """Extract important concepts from text"""
        if not text:
            return []
        
        # Use GPU for embeddings-based keyword extraction if available
        if self.use_gpu and self.embedding_model and len(text) > 500:
            try:
                # Get sentences
                sentences = sent_tokenize(text)
                
                if len(sentences) <= 1:
                    # Fall back to traditional method for very short texts
                    raise ValueError("Text too short for embedding-based extraction")
                
                # Compute embeddings for each sentence
                with torch.no_grad():
                    sentence_embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
                
                # Compute centroid (document embedding)
                centroid = torch.mean(sentence_embeddings, dim=0)
                
                # Calculate similarity of each sentence to the centroid
                similarities = []
                for i, sent_emb in enumerate(sentence_embeddings):
                    sim = cosine_similarity(
                        sent_emb.cpu().numpy().reshape(1, -1),
                        centroid.cpu().numpy().reshape(1, -1)
                    )[0][0]
                    similarities.append((i, sim))
                
                # Get the most representative sentences
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_sentence_indices = [idx for idx, _ in similarities[:min(3, len(similarities))]]
                
                # Extract keywords from these sentences
                important_sentences = [sentences[i] for i in top_sentence_indices]
                important_text = " ".join(important_sentences)
                
                # Extract keywords from important sentences using traditional method
                words = word_tokenize(important_text.lower())
                filtered_words = [word for word in words if word.isalnum() and word not in STOP_WORDS and len(word) > 3]
                word_counts = Counter(filtered_words)
                
                # Return most common words
                return [word for word, _ in word_counts.most_common(max_concepts)]
                
            except Exception as e:
                logger.warning(f"GPU-based keyword extraction failed: {e}, falling back to traditional method")
        
        # Traditional method
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in STOP_WORDS and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        # Return most common words
        return [word for word, _ in word_counts.most_common(max_concepts)]
    
    def verify_sources(self, concept: str, use_multiple_meanings: bool = True) -> Dict:
        """
        Verify a concept using available sources
        With use_multiple_meanings=True, it will find and distinguish between different meanings
        """
        if use_multiple_meanings:
            return self.verify_sources_with_meanings(concept)
        
        # Check for potential typos/misspellings
        similar_concept = self.find_similar_concept(concept)
        if similar_concept and similar_concept.lower() != concept.lower():
            logger.info(f"Found similar concept '{similar_concept}' for '{concept}'")
            # Add a note about the correction
            correction_note = f"Note: Information for '{similar_concept}' (similar to '{concept}'):\n\n"
            # Continue with the similar concept but keep track that it was corrected
        else:
            similar_concept = None
            correction_note = ""
        
        # Check all available sources in parallel for the original concept
        source_functions = [
            self.get_wikipedia_definition,
            self.get_dbpedia_definition
        ]
        
        # Gather definitions from all sources in parallel
        all_results = []
        
        # Use ThreadPoolExecutor to query sources in parallel
        with ThreadPoolExecutor(max_workers=len(source_functions)) as executor:
            all_results = list(executor.map(lambda f: f(concept), source_functions))
        
        # Filter out empty results and sort by confidence
        valid_sources = [s for s in all_results if s["definition"] and s["confidence"] > 0.2]
        
        # If no valid sources found and we have a similar concept, try that instead
        if not valid_sources and similar_concept:
            with ThreadPoolExecutor(max_workers=len(source_functions)) as executor:
                similar_results = list(executor.map(lambda f: f(similar_concept), source_functions))
            
            valid_sources = [s for s in similar_results if s["definition"] and s["confidence"] > 0.2]
            
            # If using similar concept was successful, add the correction note
            if valid_sources:
                for s in valid_sources:
                    s["definition"] = correction_note + s["definition"]
                    # Slightly reduce confidence since it's not an exact match
                    s["confidence"] = max(0.3, s["confidence"] * 0.9)
        
        valid_sources.sort(key=lambda x: x["confidence"], reverse=True)
        
        if not valid_sources:
            logger.warning(f"No valid sources found for '{concept}'")
            return {
                "definition": f"Information about '{concept}' could not be found in reliable sources.",
                "source": "",
                "url": "",
                "confidence": 0.1
            }
        
        # Merge definitions for a unified result
        return self.merge_definitions(valid_sources)


class RecursiveKnowledgeExplorer:
    """Comprehensively explores and learns about concepts without sacrificing speed"""
    
    def __init__(self, knowledge_base, max_depth: int = 3, max_workers: int = 4):
        self.knowledge_base = knowledge_base
        self.max_depth = max_depth  # Keep depth high for comprehensive learning
        self.max_workers = max_workers  # Parallel workers for performance
        self.explored_concepts: Set[str] = set()  # Track explored concepts
        self.exploration_queue = deque()  # Queue for breadth-first exploration
        self.lock = threading.Lock()  # For thread safety
        
        self.verifier = SourceVerifier()
        self.verifier.cache.lock = self.lock  # Share the lock with the cache
        
        self.max_concept_length = 100
        
        # Adaptive timing settings
        self.min_delay = 0  # No minimum delay
        self.current_delay = 0.2  # Start with a small delay
        self.error_count = 0
        self.success_count = 0
        
        # Exploration management
        self.max_concepts_per_depth = {
            0: 1,    # Seed concept
            1: 7,    # First level: top 7 related concepts
            2: 5,    # Second level: top 5 related concepts per first-level
            3: 3     # Third level: top 3 related concepts per second-level
        }
        
        # Track exploration progress
        self.exploration_status = {
            "total_concepts": 0,
            "explored_concepts": 0,
            "successful_concepts": 0,
            "failed_concepts": 0
        }
        
        # Check for GPU
        self.has_gpu = hasattr(self.verifier, 'use_gpu') and self.verifier.use_gpu
        if self.has_gpu:
            logger.info("RecursiveKnowledgeExplorer will use GPU acceleration")
    
    def normalize_concept(self, concept: str) -> str:
        """Normalize a concept string"""
        if not concept:
            return ""
            
        # Convert to lowercase and strip whitespace
        normalized = concept.lower().strip()
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters that might cause issues
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        # Truncate if too long
        if len(normalized) > self.max_concept_length:
            normalized = normalized[:self.max_concept_length]
        
        return normalized
    
    def adaptive_sleep(self, success=True):
        """Adaptive delay based on success/failure patterns"""
        with self.lock:
            if success:
                self.success_count += 1
                self.error_count = 0
                
                # After several consecutive successes, reduce delay
                if self.success_count >= 5:
                    self.current_delay = max(self.min_delay, self.current_delay * 0.8)
                    self.success_count = 0
            else:
                self.error_count += 1
                self.success_count = 0
                
                # After errors, increase delay
                if self.error_count >= 2:
                    self.current_delay = min(1.0, self.current_delay * 1.5)
                    self.error_count = 0
            
            delay = self.current_delay
        
        if delay > 0:
            time.sleep(delay)
    
    def explore_concept(self, concept: str, current_depth: int = 0, use_multiple_meanings: bool = True) -> Optional[object]:
        """
        Explore a concept comprehensively with support for multiple meanings
        Returns the KnowledgeNode created/updated for this concept
        """
        # Normalize concept
        original_concept = concept
        concept = self.normalize_concept(concept)
        
        if not concept:
            logger.warning(f"Unable to normalize concept: '{original_concept}'")
            return None
        
        # Thread-safe check if already explored
        with self.lock:
            # Update exploration status
            self.exploration_status["total_concepts"] += 1
            
            if concept in self.explored_concepts:
                return self.knowledge_base.get_knowledge(concept)
                
            # Mark as explored to prevent loops
            self.explored_concepts.add(concept)
            self.exploration_status["explored_concepts"] += 1
        
        # Check if already in knowledge base with sufficient confidence
        existing_node = self.knowledge_base.get_knowledge(concept)
        if existing_node and existing_node.confidence > 0.7:
            logger.info(f"Concept '{concept}' already known with high confidence")
            return existing_node
        
        success = False
        try:
            # Log exploration
            logger.info(f"Exploring concept: '{concept}' (depth {current_depth})")
            
            # Get comprehensive information about the concept
            if use_multiple_meanings:
                result = self.verifier.verify_sources_with_meanings(concept)
            else:
                result = self.verifier.verify_sources(concept, use_multiple_meanings=False)
            
            if result["confidence"] <= 0.2:
                logger.warning(f"Low confidence for concept '{concept}'")
                self.adaptive_sleep(success=False)
                
                with self.lock:
                    self.exploration_status["failed_concepts"] += 1
                    
                return self._handle_exploration_failure(concept)
            
            # Create or update knowledge node
            if use_multiple_meanings and "meanings" in result:
                # Use the enhanced method for multiple meanings
                node = self.knowledge_base.add_knowledge_with_meanings(concept, result)
            else:
                # Use regular method for single meaning
                node = self.knowledge_base.add_knowledge(
                    concept=concept,
                    definition=result["definition"],
                    source=result["url"] or result["source"],
                    confidence=result["confidence"]
                )
            
            # Extract key concepts for further exploration
            if current_depth < self.max_depth:
                # Get max concepts based on current depth
                max_related = self.max_concepts_per_depth.get(current_depth + 1, 3)
                
                # Extract related concepts - if we have multiple meanings, extract from primary meaning
                if use_multiple_meanings and "meanings" in result and result["meanings"]:
                    primary_meaning = result["meanings"][0]
                    if "definition" in primary_meaning:
                        related_concepts = self.verifier.extract_key_concepts(primary_meaning["definition"], max_related)
                    else:
                        related_concepts = self.verifier.extract_key_concepts(result["definition"], max_related)
                else:
                    related_concepts = self.verifier.extract_key_concepts(result["definition"], max_related)
                
                # Add to exploration queue with thread safety
                with self.lock:
                    for related_concept in related_concepts:
                        if related_concept != concept and related_concept not in self.explored_concepts:
                            # Add connection
                            node.add_connection(related_concept, 0.7)
                            
                            # Queue for exploration (concept, depth)
                            self.exploration_queue.append((related_concept, current_depth + 1))
            
            success = True
            self.adaptive_sleep(success=True)
            
            with self.lock:
                self.exploration_status["successful_concepts"] += 1
                
            return node
            
        except Exception as e:
            logger.error(f"Error exploring concept '{concept}': {e}")
            self.adaptive_sleep(success=False)
            
            with self.lock:
                self.exploration_status["failed_concepts"] += 1
                
            return self._handle_exploration_failure(concept)
    
    def _handle_exploration_failure(self, concept: str) -> object:
        """Handle the case where exploration fails"""
        # Create a placeholder node with low confidence
        return self.knowledge_base.add_knowledge(
            concept=concept,
            definition=f"Information about '{concept}' could not be found or verified from reliable sources.",
            source="",
            confidence=0.1
        )
    
    def start_exploration(self, seed_concept: str):
        """Start comprehensive exploration from a seed concept"""
        # Reset exploration tracking
        with self.lock:
            self.explored_concepts.clear()
            self.exploration_queue.clear()
            self.exploration_queue.append((seed_concept, 0))
            
            self.exploration_status = {
                "total_concepts": 0,
                "explored_concepts": 0,
                "successful_concepts": 0,
                "failed_concepts": 0
            }
        
        # Process the first concept immediately (synchronously)
        self.explore_concept(seed_concept)
        
        # Then process the rest of the queue in parallel
        self._process_queue_parallel()
        
        # Report exploration statistics
        logger.info(f"Exploration complete: {self.exploration_status}")
    
    def _process_queue_parallel(self):
        """Process exploration queue with multiple threads"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                # Get batch of concepts to process
                batch = []
                with self.lock:
                    # Take up to max_workers concepts from the queue
                    while self.exploration_queue and len(batch) < self.max_workers:
                        if not self.exploration_queue:
                            break
                            
                        concept, depth = self.exploration_queue.popleft()
                        if concept not in self.explored_concepts:
                            batch.append((concept, depth))
                            # Mark as explored to prevent duplicates
                            self.explored_concepts.add(concept)
                
                # If queue is empty, we're done
                if not batch:
                    break
                
                # Submit batch to executor
                list(executor.map(lambda args: self.explore_concept(*args), batch))
    
    def get_exploration_status(self):
        """Get current exploration status"""
        with self.lock:
            return dict(self.exploration_status)
            
    def batch_explore_concepts(self, concepts: List[str], depth: int = 0, use_multiple_meanings: bool = True):
        """Explore multiple concepts in parallel efficiently with multiple meanings support"""
        if not concepts:
            return []
            
        # Normalize all concepts first
        normalized_concepts = [self.normalize_concept(c) for c in concepts]
        
        # Filter out already explored concepts
        to_explore = []
        with self.lock:
            for i, concept in enumerate(normalized_concepts):
                if concept and concept not in self.explored_concepts:
                    to_explore.append((concept, depth, use_multiple_meanings))
                    self.explored_concepts.add(concept)
        
        if not to_explore:
            return []
            
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(lambda args: self.explore_concept(*args), to_explore))
            
        return results
        
    def explore_with_gpu_batching(self, seed_concept: str):
        """
        Enhanced exploration that uses GPU for batch processing
        Similar to start_exploration but with optimized batch processing
        """
        if not self.has_gpu:
            # Fall back to regular exploration if GPU not available
            logger.info("GPU not available, using standard exploration")
            return self.start_exploration(seed_concept)
            
        # Reset exploration tracking
        with self.lock:
            self.explored_concepts.clear()
            self.exploration_queue.clear()
            self.exploration_queue.append((seed_concept, 0))
            
            self.exploration_status = {
                "total_concepts": 0,
                "explored_concepts": 0,
                "successful_concepts": 0,
                "failed_concepts": 0
            }
        
        # Process the first concept immediately
        node = self.explore_concept(seed_concept)
        
        # Process the rest in optimized batches
        while True:
            # Get batch of concepts to process
            batch = []
            with self.lock:
                # Take up to max_workers*2 concepts from the queue (larger batches for GPU)
                batch_size = self.max_workers * 2
                while self.exploration_queue and len(batch) < batch_size:
                    if not self.exploration_queue:
                        break
                        
                    concept, depth = self.exploration_queue.popleft()
                    if concept not in self.explored_concepts:
                        batch.append((concept, depth))
                        # Mark as explored to prevent duplicates
                        self.explored_concepts.add(concept)
            
            # If queue is empty, we're done
            if not batch:
                break
                
            # First, gather all the concepts to look up
            concepts = [c for c, _ in batch]
            
            # Pre-fetch multiple definitions in parallel for concepts
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Process concepts with depths
                list(executor.map(lambda args: self.explore_concept(*args), batch))
        
        # Report exploration statistics
        logger.info(f"GPU-optimized exploration complete: {self.exploration_status}")
        
        return node
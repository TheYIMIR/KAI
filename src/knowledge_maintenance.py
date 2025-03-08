"""
Knowledge Maintenance Utility for KAI with GPU Acceleration

This script performs maintenance and improvement on KAI's knowledge base:
1. Verifies existing knowledge and corrects inaccuracies
2. Improves definitions with more detailed information
3. Adds missing connections between related concepts
4. Proactively learns about new concepts without requiring questions
5. Identifies and learns unknown terms within existing definitions

GPU acceleration is used for:
- Similarity comparisons between definitions
- Text processing and analysis
- Batch operations for improved efficiency

Usage:
    python knowledge_maintenance.py [options]

Options:
    --verify-all             Verify all existing concepts in the knowledge base
    --learn-topic X          Learn about topic X and related concepts
    --fix-concept X          Re-learn and correct information about concept X
    --learn-unknown-terms    Scan knowledge base for unknown terms mentioned in definitions
    --batch-size N           Process N concepts at a time (default: 10)
    --depth N                Exploration depth for learning (default: 2)
    --analyze                Analyze knowledge quality and statistics
"""

import os
import sys
import time
import argparse
import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Set, Tuple, Optional
import re
import nltk

# GPU support imports
import torch
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_GPU_SUPPORT = True
except ImportError:
    HAS_GPU_SUPPORT = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kai_maintenance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KAI_Maintenance")

# Import stopwords from NLTK
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except:
    # Fallback if NLTK is not available
    STOP_WORDS = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
                 "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", 
                 "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
                 "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
                 "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", 
                 "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", 
                 "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
                 "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", 
                 "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
                 "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", 
                 "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
                 "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", 
                 "yourselves"}

# Initialize NLTK resources if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("NLTK initialization failed, some features may be limited")

# Import KAI components
# This assumes your project structure with main.py in the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import KAI modules
try:
    from src.knowledge_base import KnowledgeBase, KnowledgeNode
    from src.explorer import RecursiveKnowledgeExplorer, SourceVerifier
except ImportError:
    logger.error("Failed to import KAI modules. Make sure you're running this from the project root.")
    sys.exit(1)

class KnowledgeMaintenance:
    """Maintains and improves KAI's knowledge base with GPU acceleration"""
    
    def __init__(self, storage_path: str = "data/knowledge", batch_size: int = 10, depth: int = 2):
        self.storage_path = storage_path
        self.batch_size = batch_size
        self.knowledge_base = KnowledgeBase(storage_path=storage_path)
        self.explorer = RecursiveKnowledgeExplorer(self.knowledge_base, max_depth=depth)
        self.verifier = self.explorer.verifier
        self.lock = threading.Lock()
        
        # Create directories if they don't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize GPU-accelerated embedding model if available
        self.use_gpu = hasattr(self.verifier, 'use_gpu') and self.verifier.use_gpu
        if self.use_gpu:
            logger.info("Using GPU acceleration for knowledge maintenance")
        else:
            logger.info("GPU acceleration not available, using CPU")
        
        # Statistics
        self.stats = {
            "concepts_processed": 0,
            "concepts_improved": 0,
            "concepts_corrected": 0,
            "new_connections_added": 0,
            "new_concepts_learned": 0
        }
    
    def verify_all_concepts(self):
        """Verify and correct all concepts in the knowledge base"""
        # Get all concepts
        concepts = list(self.knowledge_base.nodes.keys())
        total_concepts = len(concepts)
        
        logger.info(f"Starting verification of {total_concepts} concepts")
        
        # Process in batches
        for i in range(0, total_concepts, self.batch_size):
            batch = concepts[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_concepts + self.batch_size - 1)//self.batch_size}")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                executor.map(self.verify_concept, batch)
            
            # Print progress
            logger.info(f"Processed {min(i + self.batch_size, total_concepts)}/{total_concepts} concepts")
            logger.info(f"Stats: {self.stats}")
        
        logger.info("Verification complete")
        logger.info(f"Final stats: {self.stats}")
    
    def verify_concept(self, concept: str):
        """Verify and potentially correct a single concept"""
        # Get existing node
        existing_node = self.knowledge_base.get_knowledge(concept)
        
        if not existing_node:
            logger.warning(f"Concept '{concept}' not found in knowledge base")
            return
        
        with self.lock:
            self.stats["concepts_processed"] += 1
        
        # Check if verification is needed
        if existing_node.confidence >= 0.8:
            logger.debug(f"Concept '{concept}' already has high confidence, skipping verification")
            return
        
        logger.info(f"Verifying concept: '{concept}'")
        
        # Get fresh information
        result = self.verifier.verify_sources(concept)
        
        if result["confidence"] <= 0.2:
            logger.warning(f"Could not verify concept '{concept}'")
            return
        
        # Check if the new information is better
        is_improved = False
        is_corrected = False
        
        # Longer definition is considered an improvement
        if len(result["definition"]) > len(existing_node.definition) * 1.2:
            is_improved = True
        
        # Higher confidence is considered a correction
        if result["confidence"] > existing_node.confidence * 1.2:
            is_corrected = True
        
        # Different definition with similar confidence is also a potential correction
        similarity = 0.0
        if self.use_gpu:
            # Use GPU-accelerated similarity comparison
            similarity = self.verifier.get_similarity_score(result["definition"], existing_node.definition)
        else:
            # Simple word overlap similarity as fallback
            words1 = set(word for word in existing_node.definition.lower().split() if word not in STOP_WORDS)
            words2 = set(word for word in result["definition"].lower().split() if word not in STOP_WORDS)
            if words1 and words2:
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
        
        if similarity < 0.7 and result["confidence"] >= existing_node.confidence * 0.8:
            is_corrected = True
        
        # Update if improved or corrected
        if is_improved or is_corrected:
            # Update the knowledge node
            updated_node = self.knowledge_base.add_knowledge(
                concept=concept,
                definition=result["definition"],
                source=result["url"] or result["source"],
                confidence=result["confidence"]
            )
            
            # Update connections
            self._update_connections(concept, result["definition"])
            
            # Update stats
            with self.lock:
                if is_improved:
                    self.stats["concepts_improved"] += 1
                if is_corrected:
                    self.stats["concepts_corrected"] += 1
            
            logger.info(f"Concept '{concept}' {'improved' if is_improved else ''} {'corrected' if is_corrected else ''}")
        else:
            logger.debug(f"No improvements needed for concept '{concept}'")
    
    def learn_topic(self, topic: str):
        """Learn about a topic and its related concepts using GPU acceleration if available"""
        logger.info(f"Learning about topic: '{topic}'")
        
        # First check if we already know about this
        existing_node = self.knowledge_base.get_knowledge(topic)
        
        # Start exploration - use GPU-optimized exploration if available
        if self.use_gpu and hasattr(self.explorer, 'explore_with_gpu_batching'):
            self.explorer.explore_with_gpu_batching(topic)
        else:
            self.explorer.start_exploration(topic)
        
        # Get updated node
        updated_node = self.knowledge_base.get_knowledge(topic)
        
        # Update stats
        with self.lock:
            if not existing_node and updated_node:
                self.stats["new_concepts_learned"] += 1
            elif existing_node and updated_node and updated_node.confidence > existing_node.confidence:
                self.stats["concepts_improved"] += 1
        
        # Print status
        status = self.explorer.get_exploration_status()
        logger.info(f"Learning complete: {status}")
        self.stats["new_concepts_learned"] += status["successful_concepts"]
    
    def fix_concept(self, concept: str):
        """Fix information about a specific concept"""
        logger.info(f"Fixing concept: '{concept}'")
        
        # First check if we know about this
        existing_node = self.knowledge_base.get_knowledge(concept)
        
        if not existing_node:
            logger.warning(f"Concept '{concept}' not found in knowledge base")
            return
        
        # Remove from knowledge base to force re-learning
        with self.lock:
            if concept in self.knowledge_base.nodes:
                del self.knowledge_base.nodes[concept]
        
        # Start exploration with GPU acceleration if available
        if self.use_gpu and hasattr(self.explorer, 'explore_with_gpu_batching'):
            self.explorer.explore_with_gpu_batching(concept)
        else:
            self.explorer.explore_concept(concept)
        
        # Get updated node
        updated_node = self.knowledge_base.get_knowledge(concept)
        
        if updated_node:
            # Update connections
            self._update_connections(concept, updated_node.definition)
            
            # Update stats
            with self.lock:
                self.stats["concepts_corrected"] += 1
            
            logger.info(f"Concept '{concept}' fixed with confidence {updated_node.confidence:.2f}")
        else:
            logger.warning(f"Failed to fix concept '{concept}'")
    
    def _update_connections(self, concept: str, definition: str):
        """Update connections between concepts based on definition"""
        # Extract key concepts from the definition
        if self.use_gpu:
            related_concepts = self.verifier.extract_key_concepts(definition, max_concepts=10)
        else:
            # Simpler extraction method as fallback
            words = [w.lower() for w in definition.split() if w.lower() not in STOP_WORDS and len(w) > 3]
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            related_concepts = [word for word, _ in sorted_words[:10]]
        
        # Get the node
        node = self.knowledge_base.get_knowledge(concept)
        if not node:
            return
        
        # Add connections
        connections_added = 0
        for related in related_concepts:
            if related != concept and self.knowledge_base.get_knowledge(related):
                if related not in node.connections or node.connections[related] < 0.7:
                    node.add_connection(related, 0.7)
                    connections_added += 1
        
        # Update stats
        with self.lock:
            self.stats["new_connections_added"] += connections_added
    
    def learn_unknown_terms_in_knowledge_base(self):
        """
        Scan through existing knowledge base and identify unknown terms
        within definitions, then learn about those terms
        """
        logger.info("Scanning knowledge base for unknown terms within definitions")
        
        # Get all concepts
        all_concepts = list(self.knowledge_base.nodes.keys())
        total_concepts = len(all_concepts)
        
        if total_concepts == 0:
            logger.info("Knowledge base is empty")
            return
        
        logger.info(f"Scanning {total_concepts} concepts for unknown terms")
        
        # Track unknown terms found and learned
        all_unknown_terms = set()
        learned_terms = set()
        
        # Process in batches
        for i in range(0, total_concepts, self.batch_size):
            batch = all_concepts[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_concepts + self.batch_size - 1)//self.batch_size}")
            
            for concept in batch:
                # Get the definition
                node = self.knowledge_base.get_knowledge(concept)
                if not node or not node.definition:
                    continue
                    
                # Extract terms from the definition
                terms = self._extract_potential_keywords(node.definition)
                
                # Check which terms are unknown
                for term in terms:
                    # Skip the concept itself
                    if term.lower() == concept.lower():
                        continue
                        
                    # Skip very common words and very short terms
                    if term.lower() in STOP_WORDS or len(term) < 4:
                        continue
                    
                    # Check if term is known
                    term_node = self.knowledge_base.get_knowledge(term)
                    if not term_node or term_node.confidence < 0.4:
                        all_unknown_terms.add(term)
                        
                        # Use the original concept as context for learning
                        logger.info(f"Found unknown term '{term}' in definition of '{concept}'")
                    
            # Report progress
            logger.info(f"Processed {min(i + self.batch_size, total_concepts)}/{total_concepts} concepts")
            logger.info(f"Found {len(all_unknown_terms)} unknown terms so far")
        
        # Now learn about all the unknown terms
        logger.info(f"Learning about {len(all_unknown_terms)} unknown terms found in definitions")
        
        # Convert to list and sort by length (shortest first for efficiency)
        unknown_terms_list = sorted(list(all_unknown_terms), key=len)
        
        # Process in batches
        for i in range(0, len(unknown_terms_list), self.batch_size):
            batch = unknown_terms_list[i:i+self.batch_size]
            
            # Process batch with GPU acceleration if available
            if self.use_gpu and hasattr(self.explorer, 'batch_explore_concepts'):
                results = self.explorer.batch_explore_concepts(batch)
                
                # Check results
                for term, result in zip(batch, results):
                    term_node = self.knowledge_base.get_knowledge(term)
                    if result and term_node and term_node.confidence > 0.3:
                        learned_terms.add(term)
                        with self.lock:
                            self.stats["new_concepts_learned"] += 1
            else:
                # Process batch in parallel without GPU optimization
                with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                    results = list(executor.map(self.explorer.explore_concept, batch))
                
                # Check which terms were learned
                for term, result in zip(batch, results):
                    term_node = self.knowledge_base.get_knowledge(term)
                    if result and term_node and term_node.confidence > 0.3:
                        learned_terms.add(term)
                        with self.lock:
                            self.stats["new_concepts_learned"] += 1
            
            # Report progress
            logger.info(f"Learned about {len(learned_terms)}/{len(all_unknown_terms)} unknown terms")
            
        # Final report
        logger.info(f"Knowledge base scan complete")
        logger.info(f"Found {len(all_unknown_terms)} unknown terms in existing definitions")
        logger.info(f"Successfully learned about {len(learned_terms)} terms")
        
        return list(learned_terms)
    
    def _extract_potential_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text"""
        # Use GPU-accelerated extraction if available
        if self.use_gpu and hasattr(self.verifier, 'extract_key_concepts'):
            return self.verifier.extract_key_concepts(text, max_concepts=15)
        
        # Use NLTK for part-of-speech tagging
        try:
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            
            # Extract nouns and proper nouns
            keywords = [word for word, tag in tagged if tag.startswith(('NN', 'NNP')) and len(word) > 3]
            
            # Remove duplicates and normalize
            normalized_keywords = []
            seen = set()
            for keyword in keywords:
                normalized = keyword.lower()
                if normalized not in seen and normalized not in STOP_WORDS:
                    seen.add(normalized)
                    normalized_keywords.append(normalized)
            
            return normalized_keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            
            # Fallback method if NLTK fails
            words = text.split()
            filtered_words = [w.lower() for w in words if len(w) > 3 and w.lower() not in STOP_WORDS]
            return list(set(filtered_words))
    
    def extract_and_learn_unknown_keywords(self, text: str, context: str = ""):
        """Extract keywords from text and learn about any unknown ones within the given context"""
        logger.info("Extracting and learning unknown keywords from text")
        
        # Extract potential keywords (nouns, proper nouns, technical terms)
        keywords = self._extract_potential_keywords(text)
        
        if not keywords:
            logger.info("No significant keywords found in text")
            return []
        
        # Filter to keywords not already in knowledge base or with low confidence
        unknown_keywords = []
        for keyword in keywords:
            node = self.knowledge_base.get_knowledge(keyword)
            if not node or node.confidence < 0.4:
                unknown_keywords.append(keyword)
        
        if not unknown_keywords:
            logger.info("All keywords are already known with sufficient confidence")
            return []
        
        logger.info(f"Found {len(unknown_keywords)} unknown keywords: {', '.join(unknown_keywords[:10])}" + 
                  (f"... and {len(unknown_keywords) - 10} more" if len(unknown_keywords) > 10 else ""))
        
        # Learn about unknown keywords
        learned_keywords = []
        
        # Process batch with GPU acceleration if available
        if self.use_gpu and hasattr(self.explorer, 'batch_explore_concepts'):
            # If context provided, create contextual exploration terms
            if context:
                contextual_terms = [f"{keyword} {context}" for keyword in unknown_keywords[:20]]
                results = self.explorer.batch_explore_concepts(contextual_terms)
                
                # Check results and fall back to direct keyword for failed ones
                fallback_keywords = []
                for keyword, result in zip(unknown_keywords[:20], results):
                    concept = f"{keyword} {context}"
                    node = self.knowledge_base.get_knowledge(concept)
                    if not node or node.confidence < 0.3:
                        fallback_keywords.append(keyword)
                    else:
                        learned_keywords.append(keyword)
                        with self.lock:
                            self.stats["new_concepts_learned"] += 1
                
                # Try fallbacks
                if fallback_keywords:
                    fallback_results = self.explorer.batch_explore_concepts(fallback_keywords)
                    for keyword, result in zip(fallback_keywords, fallback_results):
                        node = self.knowledge_base.get_knowledge(keyword)
                        if node and node.confidence > 0.3:
                            learned_keywords.append(keyword)
                            with self.lock:
                                self.stats["new_concepts_learned"] += 1
            else:
                # No context, just learn the keywords directly
                results = self.explorer.batch_explore_concepts(unknown_keywords[:20])
                for keyword, result in zip(unknown_keywords[:20], results):
                    node = self.knowledge_base.get_knowledge(keyword)
                    if node and node.confidence > 0.3:
                        learned_keywords.append(keyword)
                        with self.lock:
                            self.stats["new_concepts_learned"] += 1
        else:
            # Process individually without GPU optimization
            for keyword in unknown_keywords[:20]:  # Limit to first 20 for efficiency
                logger.info(f"Learning about keyword: '{keyword}' in context: '{context}'")
                
                # Create contextual exploration by combining the keyword with context
                if context:
                    # Attempt to learn with context
                    contextual_term = f"{keyword} {context}"
                    self.explorer.explore_concept(contextual_term)
                    node = self.knowledge_base.get_knowledge(contextual_term)
                    
                    # If contextual learning failed, try just the keyword
                    if not node or node.confidence < 0.3:
                        self.explorer.explore_concept(keyword)
                else:
                    # No context, just learn the keyword directly
                    self.explorer.explore_concept(keyword)
                
                # Check if keyword was learned
                node = self.knowledge_base.get_knowledge(keyword)
                if node and node.confidence > 0.3:
                    learned_keywords.append(keyword)
                    with self.lock:
                        self.stats["new_concepts_learned"] += 1
        
        logger.info(f"Successfully learned about {len(learned_keywords)} keywords")
        return learned_keywords
    
    def analyze_knowledge_quality(self):
        """Analyze the quality of the knowledge base"""
        # Get all concepts
        concepts = list(self.knowledge_base.nodes.keys())
        total_concepts = len(concepts)
        
        if total_concepts == 0:
            logger.info("Knowledge base is empty")
            return
        
        # Calculate statistics
        confidence_sum = 0
        definition_length_sum = 0
        connections_sum = 0
        low_confidence_count = 0
        high_confidence_count = 0
        
        for concept in concepts:
            node = self.knowledge_base.get_knowledge(concept)
            if node:
                confidence_sum += node.confidence
                definition_length_sum += len(node.definition)
                connections_sum += len(node.connections)
                
                if node.confidence < 0.4:
                    low_confidence_count += 1
                if node.confidence > 0.7:
                    high_confidence_count += 1
        
        # Calculate averages
        avg_confidence = confidence_sum / total_concepts
        avg_definition_length = definition_length_sum / total_concepts
        avg_connections = connections_sum / total_concepts
        
        # Print report
        logger.info("Knowledge Base Quality Report:")
        logger.info(f"Total concepts: {total_concepts}")
        logger.info(f"Average confidence: {avg_confidence:.2f}")
        logger.info(f"Average definition length: {avg_definition_length:.1f} characters")
        logger.info(f"Average connections per concept: {avg_connections:.1f}")
        logger.info(f"Low confidence concepts (<0.4): {low_confidence_count} ({low_confidence_count/total_concepts*100:.1f}%)")
        logger.info(f"High confidence concepts (>0.7): {high_confidence_count} ({high_confidence_count/total_concepts*100:.1f}%)")
        
        # List top concepts that need improvement
        concepts_to_improve = [(c, self.knowledge_base.get_knowledge(c).confidence) 
                              for c in concepts if self.knowledge_base.get_knowledge(c).confidence < 0.4]
        concepts_to_improve.sort(key=lambda x: x[1])
        
        if concepts_to_improve:
            logger.info("Top 10 concepts needing improvement:")
            for concept, confidence in concepts_to_improve[:10]:
                logger.info(f"  - '{concept}' (confidence: {confidence:.2f})")
            
        return {
            "total_concepts": total_concepts,
            "avg_confidence": avg_confidence,
            "avg_definition_length": avg_definition_length,
            "avg_connections": avg_connections,
            "low_confidence_count": low_confidence_count,
            "high_confidence_count": high_confidence_count
        }
        
    def clean_unreliable_knowledge(self):
        """
        Scan the knowledge base and remove concepts that don't meet reliability standards
        - Concepts with no source
        - Concepts with very low confidence
        - Concepts with no meaningful definition
        """
        logger.info("Starting reliability cleanup of knowledge base")
        
        # Get all concepts
        all_concepts = list(self.knowledge_base.nodes.keys())
        total_concepts = len(all_concepts)
        
        if total_concepts == 0:
            logger.info("Knowledge base is empty")
            return
        
        logger.info(f"Scanning {total_concepts} concepts for reliability issues")
        
        # Track deleted and fixed concepts
        deleted_concepts = []
        verified_concepts = []
        
        for concept in all_concepts:
            node = self.knowledge_base.get_knowledge(concept)
            if not node:
                continue
                
            should_delete = False
            
            # Check for reliability issues
            if not node.source:
                logger.info(f"Concept '{concept}' has no source")
                should_delete = True
                
            elif isinstance(node.source, str) and node.source.strip() == "":
                logger.info(f"Concept '{concept}' has empty source string")
                should_delete = True
                
            elif isinstance(node.source, list) and (not node.source or all(not s for s in node.source)):
                logger.info(f"Concept '{concept}' has empty source list")
                should_delete = True
                
            elif node.confidence < 0.2:
                logger.info(f"Concept '{concept}' has very low confidence: {node.confidence}")
                should_delete = True
                
            elif not node.definition or len(node.definition.strip()) < 20:
                logger.info(f"Concept '{concept}' has insufficient definition: '{node.definition}'")
                should_delete = True
                
            elif "could not be found" in node.definition.lower() or "could not be verified" in node.definition.lower():
                logger.info(f"Concept '{concept}' has placeholder definition")
                should_delete = True
            
            if should_delete:
                # Try to verify before deleting
                logger.info(f"Attempting to verify concept '{concept}' before deletion")
                result = self.verifier.verify_sources_with_meanings(concept)
                
                if result["confidence"] > 0.3 and result["definition"] and len(result["definition"]) > 50:
                    # We found better information, update instead of deleting
                    if "meanings" in result:
                        node = self.knowledge_base.add_knowledge_with_meanings(concept, result)
                    else:
                        node = self.knowledge_base.add_knowledge(
                            concept=concept,
                            definition=result["definition"],
                            source=result["url"] or result["source"],
                            confidence=result["confidence"]
                        )
                    verified_concepts.append(concept)
                    logger.info(f"Successfully verified concept '{concept}' with confidence {result['confidence']:.2f}")
                else:
                    # Delete the concept
                    with self.lock:
                        if concept in self.knowledge_base.nodes:
                            del self.knowledge_base.nodes[concept]
                    deleted_concepts.append(concept)
                    logger.info(f"Deleted unreliable concept '{concept}'")
        
        # Final report
        logger.info(f"Knowledge reliability cleanup complete")
        logger.info(f"Deleted {len(deleted_concepts)} unreliable concepts")
        logger.info(f"Verified and updated {len(verified_concepts)} concepts")
        
        if deleted_concepts:
            logger.info(f"Deleted concepts: {', '.join(deleted_concepts[:20])}" + 
                    (f"... and {len(deleted_concepts) - 20} more" if len(deleted_concepts) > 20 else ""))
        
        return {
            "deleted_concepts": deleted_concepts,
            "verified_concepts": verified_concepts
        }

    def enforce_reliability_standards(self, min_confidence: float = 0.3, min_definition_length: int = 50):
        """
        Modify KAI's knowledge base to enforce reliability standards
        - Updates the confidence threshold for accepting information
        - Adds verification steps to all knowledge acquisition
        """
        # 1. Set minimum confidence threshold in explorer
        if hasattr(self.explorer, 'min_confidence_threshold'):
            self.explorer.min_confidence_threshold = min_confidence
        
        # 2. Apply reliability filter to existing knowledge
        total_concepts = len(self.knowledge_base.nodes)
        fixed_count = 0
        removed_count = 0
        
        for concept in list(self.knowledge_base.nodes.keys()):
            node = self.knowledge_base.get_knowledge(concept)
            if not node:
                continue
                
            if node.confidence < min_confidence or len(node.definition) < min_definition_length:
                # Try to improve the concept
                result = self.verifier.verify_sources_with_meanings(concept)
                
                if result["confidence"] >= min_confidence and len(result["definition"]) >= min_definition_length:
                    # Update with better information
                    if "meanings" in result:
                        self.knowledge_base.add_knowledge_with_meanings(concept, result)
                    else:
                        self.knowledge_base.add_knowledge(
                            concept=concept,
                            definition=result["definition"],
                            source=result["url"] or result["source"],
                            confidence=result["confidence"]
                        )
                    fixed_count += 1
                else:
                    # Remove if can't meet standards
                    with self.lock:
                        if concept in self.knowledge_base.nodes:
                            del self.knowledge_base.nodes[concept]
                    removed_count += 1
        
        # Report results
        logger.info(f"Reliability standards enforced: min_confidence={min_confidence}, min_definition_length={min_definition_length}")
        logger.info(f"Total concepts: {total_concepts}, Fixed: {fixed_count}, Removed: {removed_count}")
        
        return {
            "total_concepts": total_concepts,
            "fixed_count": fixed_count,
            "removed_count": removed_count
        }

    def validate_source_reliability(self, source_url: str) -> float:
        """
        Validate the reliability of a source URL
        Returns a reliability score between 0 and 1
        """
        # Trusted domain patterns with reliability scores
        trusted_domains = {
            r'wikipedia\.org': 0.9,
            r'dbpedia\.org': 0.85,
            r'britannica\.com': 0.9,
            r'merriam-webster\.com': 0.9,
            r'dictionary\.com': 0.85,
            r'nytimes\.com': 0.8,
            r'washingtonpost\.com': 0.8,
            r'bbc\.co\.uk': 0.85,
            r'nature\.com': 0.95,
            r'science\.org': 0.95,
            r'ieee\.org': 0.9,
            r'mit\.edu': 0.9,
            r'stanford\.edu': 0.9,
            r'harvard\.edu': 0.9,
            r'nasa\.gov': 0.95,
            r'cdc\.gov': 0.95,
            r'nih\.gov': 0.95,
            r'edu$': 0.8,  # Educational institutions generally
            r'gov$': 0.85,  # Government domains generally
        }
        
        # Check for known reliable domains
        for domain_pattern, score in trusted_domains.items():
            if re.search(domain_pattern, source_url, re.IGNORECASE):
                return score
        
        # Default reliability for unknown sources
        return 0.5


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="KAI Knowledge Maintenance Utility")
    parser.add_argument("--verify-all", action="store_true", help="Verify all existing concepts")
    parser.add_argument("--learn-topic", type=str, help="Learn about a specific topic")
    parser.add_argument("--fix-concept", type=str, help="Fix a specific concept")
    parser.add_argument("--learn-unknown-terms", action="store_true", 
                       help="Scan knowledge base to find and learn terms mentioned in definitions")
    parser.add_argument("--process-text", type=str, help="Extract and learn unknown keywords from text file")
    parser.add_argument("--context", type=str, default="", help="Context for keyword learning")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--depth", type=int, default=2, help="Exploration depth")
    parser.add_argument("--analyze", action="store_true", help="Analyze knowledge quality")
    parser.add_argument("--clean-unreliable", action="store_true", 
                       help="Clean up unreliable knowledge from the knowledge base")
    parser.add_argument("--enforce-standards", action="store_true",
                       help="Enforce reliability standards on all knowledge")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                       help="Minimum confidence threshold for knowledge (default: 0.3)")
    parser.add_argument("--min-def-length", type=int, default=50,
                       help="Minimum definition length in characters (default: 50)")
    args = parser.parse_args()
    
    # Create maintenance object
    maintenance = KnowledgeMaintenance(batch_size=args.batch_size, depth=args.depth)
    
    # Execute requested operation
    if args.verify_all:
        maintenance.verify_all_concepts()
    elif args.learn_topic:
        maintenance.learn_topic(args.learn_topic)
    elif args.fix_concept:
        maintenance.fix_concept(args.fix_concept)
    elif args.learn_unknown_terms:
        maintenance.learn_unknown_terms_in_knowledge_base()
    elif args.process_text:
        try:
            with open(args.process_text, 'r', encoding='utf-8') as f:
                text = f.read()
            maintenance.extract_and_learn_unknown_keywords(text, args.context)
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
    elif args.analyze:
        maintenance.analyze_knowledge_quality()
    elif args.clean_unreliable:
        maintenance.clean_unreliable_knowledge()
    elif args.enforce_standards:
        maintenance.enforce_reliability_standards(
            min_confidence=args.min_confidence,
            min_definition_length=args.min_def_length
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
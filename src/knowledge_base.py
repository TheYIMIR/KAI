"""
Cognitive Knowledge Base for KAI

A brain-inspired knowledge representation system that models:
- Multidimensional concepts with multiple meanings
- Contextual flexibility and associations
- Dynamic neural-like connection networks
- Hierarchical meaning structures
- Temporal and confidence-based learning
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict

# Configure logging
logger = logging.getLogger("KAI")

class ConnectionType(Enum):
    """Types of connections between concepts"""
    SYNONYM = "synonym"          # Same/similar meaning
    ANTONYM = "antonym"          # Opposite meaning
    HYPERNYM = "hypernym"        # General category (dog → animal)
    HYPONYM = "hyponym"          # Specific instance (animal → dog)
    MERONYM = "meronym"          # Part-of relationship (wheel → car)
    HOLONYM = "holonym"          # Whole-of relationship (car → wheel)
    TEMPORAL = "temporal"        # Time relationship (morning → afternoon)
    CAUSAL = "causal"            # Cause-effect (rain → wet)
    METAPHORICAL = "metaphorical" # Figurative connection
    CONTEXTUAL = "contextual"    # Connected in specific context
    VARIATION = "variation"      # Variation of concept (plural, linguistic)
    MEANING = "meaning"          # Connection to a specific meaning
    ASSOCIATION = "association"  # General association

class MeaningType(Enum):
    """Types of meanings a concept can have"""
    PRIMARY = "primary"          # Main/most common meaning
    ENTITY = "entity"            # Named entity (person, place, etc)
    PLURAL_FORM = "plural_form"  # Plural variation
    METAPHORICAL = "metaphorical" # Figurative meaning
    TECHNICAL = "technical"      # Domain-specific technical meaning
    CULTURAL = "cultural"        # Cultural variation
    HISTORICAL = "historical"    # Historical meaning
    NAMED_ENTITY = "named_entity" # Title/proper noun
    RELATED_CONCEPT = "related_concept" # Related but distinct concept
    VARIATION = "variation"      # Linguistic variation

class Context(Enum):
    """Common context domains for meanings and connections"""
    GENERAL = "general"          # General/common usage
    SCIENCE = "science"          # Scientific context
    MEDICINE = "medicine"        # Medical context
    TECHNOLOGY = "technology"    # Technology context
    MATHEMATICS = "mathematics"  # Mathematical context
    ARTS = "arts"                # Arts/aesthetic context
    LAW = "law"                  # Legal context
    FINANCE = "finance"          # Financial context
    COLLOQUIAL = "colloquial"    # Everyday speech
    SLANG = "slang"              # Informal/slang usage
    HISTORICAL = "historical"    # Historical context
    CULTURAL = "cultural"        # Cultural context
    GEOGRAPHICAL = "geographical" # Location-specific meaning
    LINGUISTIC = "linguistic"    # Language-specific

class Connection:
    """Represents a neural-like connection between concepts/meanings"""
    
    def __init__(self, 
                 target_id: str, 
                 strength: float = 0.5, 
                 connection_type: Union[str, ConnectionType] = ConnectionType.ASSOCIATION,
                 context: Union[str, Context] = Context.GENERAL):
        self.target_id = target_id  # ID of the target node
        self.strength = strength    # Connection strength (0-1)
        
        # Handle string or enum for connection_type
        if isinstance(connection_type, str):
            try:
                self.connection_type = ConnectionType(connection_type)
            except ValueError:
                self.connection_type = ConnectionType.ASSOCIATION
        else:
            self.connection_type = connection_type
            
        # Handle string or enum for context
        if isinstance(context, str):
            try:
                self.context = Context(context)
            except ValueError:
                self.context = Context.GENERAL
        else:
            self.context = context
            
        self.created_at = time.time()
        self.last_activated = time.time()
        self.activation_count = 1  # Track usage for reinforcement
        
    def activate(self, strength_boost: float = 0.01):
        """Strengthen connection through activation (Hebbian learning)"""
        self.last_activated = time.time()
        self.activation_count += 1
        
        # Strengthen connection slightly with usage (up to a max of 1.0)
        self.strength = min(1.0, self.strength + strength_boost)
        
    def decay(self, decay_rate: float = 0.001, time_factor: float = 0.1):
        """Weaken connection over time (forgetting curve)"""
        time_since_activation = time.time() - self.last_activated
        decay_amount = decay_rate * np.exp(-time_factor * self.activation_count) * time_since_activation
        self.strength = max(0.1, self.strength - decay_amount)  # Don't decay below 0.1
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "target_id": self.target_id,
            "strength": self.strength,
            "connection_type": self.connection_type.value,
            "context": self.context.value,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "activation_count": self.activation_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Connection':
        """Create a Connection from a dictionary"""
        conn = cls(
            target_id=data["target_id"],
            strength=data["strength"],
            connection_type=data["connection_type"],
            context=data["context"]
        )
        conn.created_at = data["created_at"]
        conn.last_activated = data["last_activated"]
        conn.activation_count = data["activation_count"]
        return conn


class Meaning:
    """Represents a specific meaning or sense of a concept"""
    
    def __init__(self, 
                 meaning_id: str,
                 title: str,
                 definition: str = "",
                 meaning_type: Union[str, MeaningType] = MeaningType.PRIMARY,
                 context: Union[str, Context] = Context.GENERAL,
                 parent_concept: str = None,
                 source: str = "",
                 confidence: float = 0.5):
        
        self.meaning_id = meaning_id  # Unique identifier
        self.title = title            # Human-readable title
        self.definition = definition  # Full definition
        
        # Handle string or enum for meaning_type
        if isinstance(meaning_type, str):
            try:
                self.meaning_type = MeaningType(meaning_type)
            except ValueError:
                self.meaning_type = MeaningType.PRIMARY
        else:
            self.meaning_type = meaning_type
            
        # Handle string or enum for context
        if isinstance(context, str):
            try:
                self.context = Context(context)
            except ValueError:
                self.context = Context.GENERAL
        else:
            self.context = context
            
        self.parent_concept = parent_concept  # Parent concept ID
        self.source = source                  # Information source
        self.confidence = confidence          # Confidence in accuracy (0-1)
        self.created_at = time.time()
        self.last_updated = time.time()
        self.access_count = 1                 # Track usage frequency
        self.connections: Dict[str, Connection] = {}  # Connections to other concepts/meanings
        self.metadata: Dict[str, Any] = {}    # Additional flexible metadata
        
    def add_connection(self, 
                      target_id: str, 
                      strength: float = 0.5, 
                      connection_type: Union[str, ConnectionType] = ConnectionType.ASSOCIATION,
                      context: Union[str, Context] = Context.GENERAL):
        """Add or update a connection to another concept/meaning"""
        if target_id in self.connections:
            # Activate existing connection
            self.connections[target_id].activate()
            # Update connection type and context if provided
            if connection_type:
                self.connections[target_id].connection_type = connection_type
            if context:
                self.connections[target_id].context = context
        else:
            # Create new connection
            self.connections[target_id] = Connection(
                target_id=target_id,
                strength=strength,
                connection_type=connection_type,
                context=context
            )
            
    def access(self):
        """Record an access to this meaning"""
        self.access_count += 1
        self.last_updated = time.time()
        
    def update_confidence(self, new_confidence: float):
        """Update confidence based on confirmation/validation"""
        # Apply a weighted update based on existing confidence and new information
        weight = 0.2 if self.access_count < 5 else 0.1  # Less weight for established meanings
        self.confidence = (1 - weight) * self.confidence + weight * new_confidence
        self.last_updated = time.time()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "meaning_id": self.meaning_id,
            "title": self.title,
            "definition": self.definition,
            "meaning_type": self.meaning_type.value,
            "context": self.context.value,
            "parent_concept": self.parent_concept,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "access_count": self.access_count,
            "connections": {k: v.to_dict() for k, v in self.connections.items()},
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Meaning':
        """Create a Meaning from a dictionary"""
        meaning = cls(
            meaning_id=data["meaning_id"],
            title=data["title"],
            definition=data["definition"],
            meaning_type=data["meaning_type"],
            context=data["context"],
            parent_concept=data["parent_concept"],
            source=data["source"],
            confidence=data["confidence"]
        )
        meaning.created_at = data["created_at"]
        meaning.last_updated = data["last_updated"]
        meaning.access_count = data["access_count"]
        meaning.metadata = data["metadata"]
        
        # Load connections
        for target_id, conn_data in data["connections"].items():
            meaning.connections[target_id] = Connection.from_dict(conn_data)
            
        return meaning


class ConceptNode:
    """
    Represents a concept in the cognitive knowledge base.
    A concept acts as a hub for multiple meanings and connections.
    """
    
    def __init__(self, concept_id: str, primary_definition: str = ""):
        self.concept_id = concept_id.lower()  # Normalize concept ID
        self.primary_definition = primary_definition
        self.created_at = time.time()
        self.last_updated = time.time()
        self.access_count = 1
        self.meanings: Dict[str, Meaning] = {}  # Meanings for this concept
        self.connections: Dict[str, Connection] = {}  # Connections to other concepts
        self.metadata: Dict[str, Any] = {}  # Additional flexible metadata
        
    def add_meaning(self, meaning: Meaning):
        """Add a meaning to this concept"""
        self.meanings[meaning.meaning_id] = meaning
        self.last_updated = time.time()
        
        # Create a connection to the meaning
        self.add_connection(
            meaning.meaning_id, 
            strength=0.9, 
            connection_type=ConnectionType.MEANING,
            context=meaning.context
        )
        
    def add_connection(self, 
                      target_id: str, 
                      strength: float = 0.5, 
                      connection_type: Union[str, ConnectionType] = ConnectionType.ASSOCIATION,
                      context: Union[str, Context] = Context.GENERAL):
        """Add or update a connection to another concept"""
        if target_id in self.connections:
            # Activate existing connection
            self.connections[target_id].activate()
            # Update connection type and context if provided
            if connection_type:
                self.connections[target_id].connection_type = connection_type
            if context:
                self.connections[target_id].context = context
        else:
            # Create new connection
            self.connections[target_id] = Connection(
                target_id=target_id,
                strength=strength,
                connection_type=connection_type,
                context=context
            )
    
    def get_primary_meaning(self) -> Optional[Meaning]:
        """Get the primary meaning of this concept"""
        # First look for a meaning with PRIMARY type
        for meaning in self.meanings.values():
            if meaning.meaning_type == MeaningType.PRIMARY:
                return meaning
                
        # If none found, return the most accessed meaning
        if self.meanings:
            return max(self.meanings.values(), key=lambda m: m.access_count)
            
        return None
        
    def get_meanings_by_type(self, meaning_type: Union[str, MeaningType]) -> List[Meaning]:
        """Get all meanings of a particular type"""
        # Normalize meaning_type to enum
        if isinstance(meaning_type, str):
            try:
                meaning_type = MeaningType(meaning_type)
            except ValueError:
                return []
                
        return [m for m in self.meanings.values() if m.meaning_type == meaning_type]
        
    def get_meanings_by_context(self, context: Union[str, Context]) -> List[Meaning]:
        """Get all meanings relevant to a particular context"""
        # Normalize context to enum
        if isinstance(context, str):
            try:
                context = Context(context)
            except ValueError:
                return []
                
        return [m for m in self.meanings.values() if m.context == context]
        
    def search_meanings(self, query: str) -> List[Tuple[Meaning, float]]:
        """
        Search through meanings for those matching a query
        Returns a list of (meaning, relevance_score) tuples
        """
        query = query.lower()
        results = []
        
        for meaning in self.meanings.values():
            # Calculate relevance based on multiple factors
            relevance = 0.0
            
            # Title match
            if query in meaning.title.lower():
                relevance += 0.5
                if query == meaning.title.lower():
                    relevance += 0.3  # Exact title match bonus
                    
            # Definition match
            if query in meaning.definition.lower():
                relevance += 0.3
                
            # Weight by access count (popularity)
            relevance *= (0.7 + 0.3 * min(1.0, meaning.access_count / 10))
            
            # Weight by confidence
            relevance *= meaning.confidence
            
            if relevance > 0:
                results.append((meaning, relevance))
                
        # Sort by relevance
        return sorted(results, key=lambda x: x[1], reverse=True)
        
    def access(self):
        """Record an access to this concept"""
        self.access_count += 1
        self.last_updated = time.time()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "concept_id": self.concept_id,
            "primary_definition": self.primary_definition,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "access_count": self.access_count,
            "connections": {k: v.to_dict() for k, v in self.connections.items()},
            "metadata": self.metadata,
            # Don't store meanings here - they're stored separately
            "meaning_ids": list(self.meanings.keys())
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptNode':
        """Create a ConceptNode from a dictionary"""
        concept = cls(
            concept_id=data["concept_id"],
            primary_definition=data["primary_definition"]
        )
        concept.created_at = data["created_at"]
        concept.last_updated = data["last_updated"]
        concept.access_count = data["access_count"]
        concept.metadata = data["metadata"]
        
        # Load connections
        for target_id, conn_data in data.get("connections", {}).items():
            concept.connections[target_id] = Connection.from_dict(conn_data)
            
        return concept


class CognitiveKnowledgeBase:
    """
    A brain-inspired knowledge representation system that models:
    - Multidimensional concepts with multiple meanings
    - Contextual flexibility
    - Neural-like connection networks
    - Dynamic learning and evolution
    """
    
    def __init__(self, storage_path: str = "data/cognitive_knowledge"):
        self.concepts: Dict[str, ConceptNode] = {}
        self.meanings: Dict[str, Meaning] = {}
        self.storage_path = storage_path
        self.ensure_storage_exists()
        self.load_knowledge()
        
        # Activation thresholds
        self.min_activation_threshold = 0.3  # Minimum activation to consider a connection
        self.similarity_threshold = 0.7      # Threshold for considering meanings similar
        
        # Statistics tracking
        self.stats = {
            "concept_count": 0,
            "meaning_count": 0,
            "connection_count": 0,
            "access_count": 0,
            "last_maintenance": time.time()
        }
        
    def ensure_storage_exists(self):
        """Ensure required storage directories exist"""
        # Main storage directory
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            
        # Subdirectories for concepts and meanings
        concepts_dir = os.path.join(self.storage_path, "concepts")
        meanings_dir = os.path.join(self.storage_path, "meanings")
        
        if not os.path.exists(concepts_dir):
            os.makedirs(concepts_dir)
        if not os.path.exists(meanings_dir):
            os.makedirs(meanings_dir)
            
    def save_knowledge(self):
        """Save the knowledge base to storage"""
        try:
            # Save concepts
            concepts_dir = os.path.join(self.storage_path, "concepts")
            for concept_id, concept in self.concepts.items():
                file_path = os.path.join(concepts_dir, f"{concept_id.replace(' ', '_')}.json")
                with open(file_path, 'w') as f:
                    json.dump(concept.to_dict(), f, indent=2)
                    
            # Save meanings
            meanings_dir = os.path.join(self.storage_path, "meanings")
            for meaning_id, meaning in self.meanings.items():
                file_path = os.path.join(meanings_dir, f"{meaning_id.replace(':', '__')}.json")
                with open(file_path, 'w') as f:
                    json.dump(meaning.to_dict(), f, indent=2)
                    
            # Save statistics
            stats_path = os.path.join(self.storage_path, "stats.json")
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
            logger.info(f"Knowledge base saved with {len(self.concepts)} concepts and {len(self.meanings)} meanings")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            
    def load_knowledge(self):
        """Load the knowledge base from storage"""
        try:
            self.concepts = {}
            self.meanings = {}
            
            # Load concepts
            concepts_dir = os.path.join(self.storage_path, "concepts")
            if os.path.exists(concepts_dir):
                concept_files = [f for f in os.listdir(concepts_dir) if f.endswith('.json')]
                for file in concept_files:
                    try:
                        with open(os.path.join(concepts_dir, file), 'r') as f:
                            data = json.load(f)
                            concept = ConceptNode.from_dict(data)
                            self.concepts[concept.concept_id] = concept
                    except Exception as e:
                        logger.error(f"Error loading concept {file}: {e}")
                        
            # Load meanings
            meanings_dir = os.path.join(self.storage_path, "meanings")
            if os.path.exists(meanings_dir):
                meaning_files = [f for f in os.listdir(meanings_dir) if f.endswith('.json')]
                for file in meaning_files:
                    try:
                        with open(os.path.join(meanings_dir, file), 'r') as f:
                            data = json.load(f)
                            meaning = Meaning.from_dict(data)
                            self.meanings[meaning.meaning_id] = meaning
                    except Exception as e:
                        logger.error(f"Error loading meaning {file}: {e}")
                        
            # Load statistics
            stats_path = os.path.join(self.storage_path, "stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
            else:
                # Calculate statistics
                self.stats = {
                    "concept_count": len(self.concepts),
                    "meaning_count": len(self.meanings),
                    "connection_count": sum(len(c.connections) for c in self.concepts.values()) +
                                       sum(len(m.connections) for m in self.meanings.values()),
                    "access_count": sum(c.access_count for c in self.concepts.values()) +
                                   sum(m.access_count for m in self.meanings.values()),
                    "last_maintenance": time.time()
                }
                
            # Reconnect meanings to their parent concepts
            self._reconnect_meanings()
            
            logger.info(f"Loaded {len(self.concepts)} concepts and {len(self.meanings)} meanings")
            
            # Check for legacy data to migrate
            self._check_for_migration()
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            
    def _reconnect_meanings(self):
        """Reconnect meanings to their parent concepts"""
        for meaning_id, meaning in self.meanings.items():
            if meaning.parent_concept and meaning.parent_concept in self.concepts:
                self.concepts[meaning.parent_concept].meanings[meaning_id] = meaning
                
    def _check_for_migration(self):
        """Check for legacy data in the old format and migrate if needed"""
        legacy_dir = "data/knowledge"
        if not os.path.exists(legacy_dir):
            return
            
        try:
            # Check for legacy knowledge files
            files = [f for f in os.listdir(legacy_dir) if f.endswith('.json')]
            
            if not files:
                return
                
            logger.info(f"Found {len(files)} legacy knowledge files, starting migration")
            
            migrated_count = 0
            for file in files:
                try:
                    with open(os.path.join(legacy_dir, file), 'r') as f:
                        data = json.load(f)
                        
                    # Extract concept info
                    concept = data.get("concept", "").lower()
                    if not concept:
                        continue
                        
                    # Check if we already have this concept
                    if concept in self.concepts:
                        continue
                        
                    # Create concept node
                    concept_node = ConceptNode(concept_id=concept, primary_definition=data.get("definition", ""))
                    concept_node.metadata["source"] = data.get("source", "")
                    concept_node.metadata["legacy_confidence"] = data.get("confidence", 0.5)
                    
                    # Check for meanings in metadata
                    if "metadata" in data and "meanings" in data["metadata"] and data["metadata"]["meanings"]:
                        for meaning_data in data["metadata"]["meanings"]:
                            # Create meaning
                            meaning_type = meaning_data.get("type", "entity")
                            meaning_title = meaning_data.get("title", concept)
                            
                            # Create unique ID
                            meaning_id = f"{concept}:{meaning_title.lower().replace(' ', '_')}"
                            
                            meaning = Meaning(
                                meaning_id=meaning_id,
                                title=meaning_title,
                                definition=meaning_data.get("definition", ""),
                                meaning_type=meaning_type,
                                parent_concept=concept,
                                source=meaning_data.get("url", "") or meaning_data.get("source", ""),
                                confidence=meaning_data.get("confidence", 0.5)
                            )
                            
                            # Add meaning
                            self.meanings[meaning_id] = meaning
                            concept_node.add_meaning(meaning)
                            
                    # Add concept
                    self.concepts[concept] = concept_node
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating legacy file {file}: {e}")
                    
            if migrated_count > 0:
                logger.info(f"Successfully migrated {migrated_count} concepts from legacy format")
                self.save_knowledge()
                
        except Exception as e:
            logger.error(f"Error checking for legacy data: {e}")
            
    def create_concept(self, concept: str, definition: str = "", source: str = "") -> ConceptNode:
        """Create a new concept in the knowledge base"""
        concept_id = concept.lower()
        
        # Check if concept already exists
        if concept_id in self.concepts:
            return self.concepts[concept_id]
            
        # Create new concept
        concept_node = ConceptNode(concept_id=concept_id, primary_definition=definition)
        concept_node.metadata["source"] = source
        
        # Add to knowledge base
        self.concepts[concept_id] = concept_node
        
        # Create primary meaning
        meaning_id = f"{concept_id}:primary"
        meaning = Meaning(
            meaning_id=meaning_id,
            title=concept,
            definition=definition,
            meaning_type=MeaningType.PRIMARY,
            parent_concept=concept_id,
            source=source,
            confidence=0.5
        )
        
        # Add meaning
        self.meanings[meaning_id] = meaning
        concept_node.add_meaning(meaning)
        
        # Update stats
        self.stats["concept_count"] += 1
        self.stats["meaning_count"] += 1
        
        # Save knowledge base
        self.save_knowledge()
        
        return concept_node
        
    def add_meaning(self, 
                   parent_concept: str, 
                   title: str, 
                   definition: str, 
                   meaning_type: Union[str, MeaningType] = MeaningType.VARIATION,
                   context: Union[str, Context] = Context.GENERAL,
                   source: str = "",
                   confidence: float = 0.5) -> Optional[Meaning]:
        """Add a new meaning to an existing concept"""
        parent_id = parent_concept.lower()
        
        # Ensure parent concept exists
        if parent_id not in self.concepts:
            # Auto-create parent concept
            parent = self.create_concept(parent_concept)
        else:
            parent = self.concepts[parent_id]
            
        # Create unique meaning ID
        meaning_id = f"{parent_id}:{title.lower().replace(' ', '_')}"
        
        # Check if meaning already exists
        if meaning_id in self.meanings:
            # Update existing meaning
            meaning = self.meanings[meaning_id]
            meaning.definition = definition
            meaning.source = source
            meaning.update_confidence(confidence)
            meaning.context = context
            meaning.meaning_type = meaning_type
            meaning.last_updated = time.time()
            return meaning
            
        # Create new meaning
        meaning = Meaning(
            meaning_id=meaning_id,
            title=title,
            definition=definition,
            meaning_type=meaning_type,
            context=context,
            parent_concept=parent_id,
            source=source,
            confidence=confidence
        )
        
        # Add to knowledge base
        self.meanings[meaning_id] = meaning
        parent.add_meaning(meaning)
        
        # Update stats
        self.stats["meaning_count"] += 1
        
        # Save knowledge base
        self.save_knowledge()
        
        return meaning
        
    def add_connection(self,
                      source_id: str,
                      target_id: str,
                      strength: float = 0.5,
                      connection_type: Union[str, ConnectionType] = ConnectionType.ASSOCIATION,
                      context: Union[str, Context] = Context.GENERAL,
                      bidirectional: bool = True) -> bool:
        """Add a connection between concepts or meanings"""
        # Check if source exists
        source_node = None
        if source_id in self.concepts:
            source_node = self.concepts[source_id]
        elif source_id in self.meanings:
            source_node = self.meanings[source_id]
        else:
            return False
            
        # Check if target exists
        target_exists = (target_id in self.concepts) or (target_id in self.meanings)
        if not target_exists:
            return False
            
        # Add connection from source to target
        if isinstance(source_node, ConceptNode):
            source_node.add_connection(target_id, strength, connection_type, context)
        else:
            source_node.add_connection(target_id, strength, connection_type, context)
            
        # Add reverse connection if bidirectional
        if bidirectional:
            if target_id in self.concepts:
                self.concepts[target_id].add_connection(
                    source_id, 
                    strength, 
                    self._get_reverse_connection_type(connection_type), 
                    context
                )
            elif target_id in self.meanings:
                self.meanings[target_id].add_connection(
                    source_id, 
                    strength, 
                    self._get_reverse_connection_type(connection_type), 
                    context
                )
                
        # Update stats
        self.stats["connection_count"] += 1 + (1 if bidirectional else 0)
        
        return True
        
    def _get_reverse_connection_type(self, connection_type: Union[str, ConnectionType]) -> ConnectionType:
        """Get the reverse connection type for bidirectional connections"""
        # Normalize to enum
        if isinstance(connection_type, str):
            try:
                connection_type = ConnectionType(connection_type)
            except ValueError:
                return ConnectionType.ASSOCIATION
                
        # Define pairs of connection types that are opposites
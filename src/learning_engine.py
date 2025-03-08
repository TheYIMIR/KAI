"""
LearningEngine: Processes and integrates new information into KAI's knowledge base.
Identifies knowledge gaps and manages the learning process.
"""

from typing import List
from nltk.tokenize import sent_tokenize
import logging

# Get the logger
logger = logging.getLogger("KAI")

class LearningEngine:
    """Processes and integrates new information into KAI's knowledge base"""
    
    def __init__(self, knowledge_base, explorer):
        self.knowledge_base = knowledge_base
        self.explorer = explorer
    
    def learn_from_text(self, text: str) -> List[str]:
        """
        Learn from a piece of text, identifying new concepts and relationships
        Returns a list of concepts that were learned or updated
        """
        # Extract sentences for processing
        sentences = sent_tokenize(text)
        learned_concepts = []
        
        # Extract key concepts from the whole text
        key_concepts = self.knowledge_base.extract_key_concepts(text)
        
        # For each key concept, explore if not already known
        for concept in key_concepts:
            existing = self.knowledge_base.get_knowledge(concept)
            if not existing or existing.confidence < 0.5:
                # Trigger exploration for this concept
                node = self.explorer.explore_concept(concept)
                if node:
                    learned_concepts.append(concept)
        
        return learned_concepts
    
    def identify_unknown_concepts(self, text: str) -> List[str]:
        """Identify concepts in the text that are not in the knowledge base"""
        # Extract key concepts
        candidates = self.knowledge_base.extract_key_concepts(text)
        
        # Filter to those not in knowledge base or with low confidence
        unknown = []
        for concept in candidates:
            node = self.knowledge_base.get_knowledge(concept)
            if not node or node.confidence < 0.5:
                unknown.append(concept)
        
        return unknown
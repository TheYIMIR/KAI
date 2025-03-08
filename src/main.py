#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KAI: Knowledge Adaptive Intelligence - Main Module
This module contains the main KAI class and console interface.
"""

import logging
import os
import nltk
from typing import List

# Import our modules
from knowledge_base import KnowledgeBase
from explorer import RecursiveKnowledgeExplorer
from learning_engine import LearningEngine
from chat_interface import ChatInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kai.log", encoding='utf-8'),  # Specify utf-8 encoding
        logging.StreamHandler()  # Console handler
    ]
)
logging.getLogger().handlers[1].setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger("KAI")

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')

class KAI:
    """Main KAI class that integrates all components"""
    
    # In main.py, update your KAI class __init__ method:

    def __init__(self, storage_path: str = "data/knowledge"):
        # Ensure NLTK data is available
        ensure_nltk_data()
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs("data", exist_ok=True)  # Ensure data directory exists for cache
        
        # Initialize components
        self.knowledge_base = KnowledgeBase(storage_path=storage_path)
        
        # Create the new comprehensive explorer with preferred settings
        self.explorer = RecursiveKnowledgeExplorer(
            self.knowledge_base,
            max_depth=3,       # How deep to explore related concepts
            max_workers=4      # Number of parallel workers (adjust based on your CPU/connection)
        )
        
        self.learning_engine = LearningEngine(self.knowledge_base, self.explorer)
        self.chat_interface = ChatInterface(self.knowledge_base, self.learning_engine)
        
        logger.info("KAI initialized successfully")
    
    def chat(self, message: str) -> str:
        """Process a message from the user"""
        return self.chat_interface.process_message(message)
    
    def bootstrap_knowledge(self, seed_concepts: List[str]):
        """Bootstrap the knowledge base with initial concepts"""
        logger.info(f"Bootstrapping knowledge with seeds: {seed_concepts}")
        for concept in seed_concepts:
            logger.info(f"Starting exploration for seed concept: {concept}")
            self.explorer.start_exploration(concept)
            
            # Get and display exploration statistics
            status = self.explorer.get_exploration_status()
            logger.info(f"Exploration results for '{concept}': {status}")
            print(f"Learned about {status['successful_concepts']} concepts related to '{concept}'")


def run_console_interface():
    """Run a simple console interface for interacting with KAI"""
    print("====================================")
    print("KAI: Knowledge Adaptive Intelligence")
    print("====================================")
    print("Initializing...")
    
    # Initialize KAI
    kai = KAI()
    
    # Bootstrap with some seed knowledge if the knowledge base is empty
    if len(kai.knowledge_base.nodes) == 0:
        print("Bootstrapping initial knowledge...")
        kai.bootstrap_knowledge(["artificial intelligence", "machine learning", "knowledge representation"])
    
    print(f"KAI initialized with {len(kai.knowledge_base.nodes)} concepts")
    print("You can start chatting now. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("KAI: Goodbye!")
            break
        
        response = kai.chat(user_input)
        print(f"\nKAI: {response}")


if __name__ == "__main__":
    run_console_interface()
"""
KAI: Knowledge Adaptive Intelligence
This package contains the components for a self-learning AI system.
"""

# This file makes the src directory a Python package
# It can be empty or can contain package-level imports and variables

# Import main components to make them available at package level
from .knowledge_base import KnowledgeBase, KnowledgeNode
from .explorer import RecursiveKnowledgeExplorer
from .learning_engine import LearningEngine
from .chat_interface import ChatInterface

__version__ = "0.1.0"
__author__ = "Your Name"
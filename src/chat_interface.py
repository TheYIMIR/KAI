"""
ChatInterface: Provides a text-based interface for interacting with KAI.
Processes user messages and generates appropriate responses.
Features advanced question analysis with context awareness and intent recognition.
"""

import logging
import re
import nltk
from typing import List, Dict, Tuple, Optional, Set
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Get the logger
logger = logging.getLogger("KAI")

# Ensure all required NLTK data is downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {e}")

# Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

# Pre-load stopwords for faster access
STOP_WORDS = set(stopwords.words('english'))

class ChatInterface:
    """Provides an intelligent text-based interface for interacting with KAI"""
    
    def __init__(self, knowledge_base, learning_engine):
        self.knowledge_base = knowledge_base
        self.learning_engine = learning_engine
        self.conversation_history = []
        
        # Track recent topics for context awareness
        self.recent_topics = []
        self.max_recent_topics = 5
        
        # Common question words and phrases to ignore when extracting topics
        self.question_words = {
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose',
            'can', 'could', 'would', 'should', 'is', 'are', 'was', 'were',
            'do', 'does', 'did', 'have', 'has', 'had', 'tell', 'explain',
            'describe', 'define', 'elaborate', 'clarify', 'about'
        }
        
        # Words that indicate a desire for information
        self.information_seeking_words = {
            'know', 'learn', 'understand', 'information', 'details', 'explanation',
            'tell', 'explain', 'describe', 'what', 'who', 'where', 'when', 'why', 'how'
        }
        
        # Domain-specific prefixes to handle separately
        self.domain_prefixes = {
            'game': ['videogame', 'video game', 'computer game', 'mobile game', 'console game'],
            'movie': ['film', 'motion picture', 'cinema'],
            'book': ['novel', 'publication', 'text', 'textbook'],
            'music': ['song', 'album', 'track', 'band', 'artist'],
            'technology': ['tech', 'software', 'hardware', 'application', 'app'],
            'science': ['scientific', 'biological', 'physical', 'chemical'],
        }
        
        # Question intent classification
        self.question_intents = {
            'what': ['what', 'define', 'describe', 'explain', 'tell me about'],
            'why': ['why', 'reason', 'cause', 'purpose'],
            'how': ['how', 'method', 'way', 'means', 'process'],
            'when': ['when', 'time', 'date', 'period'],
            'where': ['where', 'location', 'place'],
            'who': ['who', 'person', 'people'],
            'which': ['which', 'option', 'alternative'],
            'compare': ['compare', 'versus', 'vs', 'difference between']
        }
        
        # Common compound topic patterns
        self.compound_patterns = [
            r"([\w\s]+)\s+(?:of|in|and|with|between|vs|versus)\s+([\w\s]+)",
            r"([\w\s]+)\s+(?:cause|effect|impact|influence|role)\s+(?:of|on|in)\s+([\w\s]+)",
            r"(?:relationship|connection|correlation)\s+(?:between|of)\s+([\w\s]+)\s+(?:and|with)\s+([\w\s]+)"
        ]
        
        # Topic pairs that should be treated as a single concept
        self.related_concept_pairs = [
            ('war', 'states', 'interstate warfare'),
            ('war', 'cause', 'causes of war'),
            ('war', 'reason', 'reasons for war'),
            ('states', 'war', 'interstate warfare'),
            ('artificial', 'intelligence', 'artificial intelligence'),
            ('quantum', 'physics', 'quantum physics'),
            ('machine', 'learning', 'machine learning'),
            ('climate', 'change', 'climate change'),
            ('global', 'warming', 'global warming')
        ]
    
    def process_message(self, message: str) -> str:
        """Process a message from the user and generate a response"""
        # Record in conversation history
        self.conversation_history.append({"role": "user", "message": message})
        
        # Determine message type and handle accordingly
        if self._is_question(message):
            response = self._handle_question(message)
        else:
            # If not a question, treat as information to learn from
            response = self._handle_information(message)
        
        # Record response
        self.conversation_history.append({"role": "kai", "message": response})
        return response
    
    def _is_question(self, message: str) -> bool:
        """
        Determine if a message is a question
        Checks for question marks, question words, and question structures
        """
        message = message.strip().lower()
        
        # Check for question mark
        if '?' in message:
            return True
        
        # Check for common question starters
        question_starters = [
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 'can',
            'could', 'would', 'should', 'is', 'are', 'do', 'does', 'did',
            'have', 'has', 'tell me', 'explain', 'define', 'elaborate on'
        ]
        
        for starter in question_starters:
            if message.startswith(starter + ' '):
                return True
        
        # Check for inverted question structure
        inverted_structures = [
            r'\b(is|are|was|were|do|does|did|have|has|had|can|could|should|would) (the|a|an|it|they|he|she|we|you|i)\b',
            r'\b(can|could|would|should) you\b'
        ]
        
        for pattern in inverted_structures:
            if re.search(pattern, message):
                return True
                
        # Check if message contains information-seeking words and is relatively short
        if len(message.split()) < 15:  # Short messages are more likely to be questions
            tokens = set(word_tokenize(message.lower()))
            if any(word in tokens for word in self.information_seeking_words):
                return True
        
        return False
    
    def _handle_question(self, question: str) -> str:
        """Handle a question from the user with support for multiple meanings"""
        # Extract the main subject/topic of the question
        topics = self._extract_question_topics(question)
        
        if not topics:
            return "I'm not sure what you're asking about. Could you rephrase your question or be more specific?"
        
        # Get the primary topic
        primary_topic = topics[0]
        logger.info(f"Identified primary topic: '{primary_topic}'")
        
        # Check knowledge base for the primary topic
        node = self.knowledge_base.get_knowledge(primary_topic)
        
        if node and "meanings" in node.metadata and len(node.metadata["meanings"]) > 1:
            # Check if question directly references a specific meaning
            specified_meaning = self._identify_specific_meaning(question, node.metadata["meanings"])
            
            if specified_meaning:
                # User is asking about a specific meaning - return that instead of the general definition
                response = f"Regarding {specified_meaning['title']}: {specified_meaning['definition']}"
                return response
        
        if node:
            # Check if this node has multiple meanings
            has_multiple_meanings = "meanings" in node.metadata and len(node.metadata["meanings"]) > 1
            
            # Determine confidence level messaging
            confidence_msg = ""
            if node.confidence < 0.3:
                confidence_msg = "I'm not very confident about this information, but here's what I found: "
            elif node.confidence < 0.6:
                confidence_msg = "I'm somewhat confident in this information: "
            
            # Handle multiple meanings differently
            if has_multiple_meanings:
                # Check if the question specifies a particular meaning
                specified_meaning = self._identify_specific_meaning(question, node.metadata["meanings"])
                
                if specified_meaning:
                    # User is asking about a specific meaning
                    response = f"{confidence_msg}Regarding {primary_topic} ({specified_meaning['type']}): "
                    
                    if specified_meaning["type"] == "plural_form":
                        # For plural forms, give definition of singular
                        response += specified_meaning["definition"]
                    else:
                        # For other meanings, use their specific definition
                        response += f"{specified_meaning['definition']}"
                else:
                    # No specific meaning requested, show overview of all meanings
                    response = f"{confidence_msg}'{primary_topic}' has multiple meanings:\n\n"
                    
                    for i, meaning in enumerate(node.metadata["meanings"][:5]):
                        if meaning["type"] == "plural_form":
                            response += f"{i+1}. Plural form of '{meaning['related_to']}'.\n"
                        else:
                            # Show title and first sentence of definition
                            first_sentence = meaning["definition"].split('.')[0] + '.'
                            response += f"{i+1}. {meaning['title']}: {first_sentence}\n"
                    
                    response += "\nYou can ask me specifically about any of these meanings."
            else:
                # Standard single-meaning response
                response = f"{confidence_msg}Regarding {primary_topic}: {node.definition}\n\n"
                
                # Add information about related concepts
                if node.confidence > 0.3:
                    related = self.knowledge_base.find_related_concepts(primary_topic, min_strength=0.5)
                    if related:
                        response += "Related concepts you might be interested in:\n"
                        for concept, _ in related[:3]:  # Top 3 related concepts
                            related_node = self.knowledge_base.get_knowledge(concept)
                            if related_node:
                                response += f"- {concept.title()}: {related_node.definition[:100]}...\n"
            
            return response
        else:
            # We don't know about this topic yet
            response = f"I don't know much about '{primary_topic}' yet. Let me research that for you...\n\n"
            
            # Start exploration for the primary topic
            logger.info(f"Starting exploration for unknown topic: '{primary_topic}'")
            self.learning_engine.explorer.start_exploration(primary_topic)
            
            # Report exploration results
            status = self.learning_engine.explorer.get_exploration_status()
            logger.info(f"Exploration results: {status}")
            
            # Now check if we learned something
            node = self.knowledge_base.get_knowledge(primary_topic)
            if node:
                # Always provide what we found, but indicate confidence
                if node.confidence < 0.3:
                    response += f"I found some information about {primary_topic}, but I'm not very confident it's accurate: {node.definition}"
                elif node.confidence < 0.6:
                    response += f"I found information about {primary_topic}, though I'm only moderately confident: {node.definition}"
                else:
                    response += f"I've learned about {primary_topic}: {node.definition}"
                    
                # Handle multiple meanings if present
                if "meanings" in node.metadata and len(node.metadata["meanings"]) > 1:
                    response += "\n\nI found multiple meanings for this term. You can ask me about a specific meaning if needed."
            else:
                # Nothing found at all
                response += f"I couldn't find specific information about {primary_topic}, but I'll continue learning."
            
            return response
    
    def _identify_specific_meaning(self, question: str, meanings: List[Dict]) -> Optional[Dict]:
        """
        Identify if the question is asking about a specific meaning
        Returns the specific meaning dictionary if identified, otherwise None
        """
        question_lower = question.lower()
        
        # First check for exact title matches in meanings
        for meaning in meanings:
            if "title" in meaning and meaning["title"].lower() in question_lower:
                # If the meaning title appears in the question, this is likely the specific meaning they want
                logger.info(f"Found specific meaning match: {meaning['title']}")
                return meaning
        
        # Check for plural form indicators
        plural_indicators = ["plural", "plural form", "multiple", "many"]
        if any(indicator in question_lower for indicator in plural_indicators):
            # Look for plural_form meaning
            for meaning in meanings:
                if meaning["type"] == "plural_form":
                    return meaning
        
        # Check for movie/film/book indicators
        media_indicators = ["movie", "film", "show", "series", "book", "novel", "game"]
        if any(indicator in question_lower for indicator in media_indicators):
            for meaning in meanings:
                if meaning["type"] == "named_entity" and any(indicator in meaning["definition"].lower() 
                                                        for indicator in media_indicators):
                    return meaning
        
        # Check for exact title matches
        for meaning in meanings:
            if "title" in meaning and meaning["title"].lower() in question_lower:
                return meaning
        
        # No specific meaning identified
        return None

    # Optional: Add this method for follow-up questions about specific meanings
    def _handle_meaning_followup(self, question: str, previous_topic: str) -> Optional[str]:
        """
        Handle follow-up questions about specific meanings of a previously discussed topic
        Returns a response if it can identify a meaning-specific follow-up, otherwise None
        """
        question_lower = question.lower()
        
        # Check for meaning number references like "tell me about the first meaning" or "what's meaning #2?"
        meaning_number_match = re.search(r'(?:meaning|definition|sense)\s+(?:number\s+)?#?(\d+)', question_lower)
        if meaning_number_match:
            try:
                meaning_index = int(meaning_number_match.group(1)) - 1  # Convert to 0-based index
                
                # Get the node for the previous topic
                node = self.knowledge_base.get_knowledge(previous_topic)
                if node and "meanings" in node.metadata and meaning_index < len(node.metadata["meanings"]):
                    meaning = node.metadata["meanings"][meaning_index]
                    
                    response = f"Regarding meaning #{meaning_index+1} of '{previous_topic}'"
                    if meaning["type"] == "plural_form":
                        response += f" (plural form of '{meaning['related_to']}'):\n\n"
                    elif "title" in meaning:
                        response += f" ('{meaning['title']}'):\n\n"
                    
                    response += meaning["definition"]
                    return response
            except:
                pass
        
        # Check for specific meaning type references
        meaning_type_indicators = {
            "plural": "plural_form",
            "movie": "named_entity",
            "film": "named_entity",
            "book": "named_entity",
            "general": "concept"
        }
        
        for indicator, meaning_type in meaning_type_indicators.items():
            if indicator in question_lower:
                node = self.knowledge_base.get_knowledge(previous_topic)
                if node and "meanings" in node.metadata:
                    for meaning in node.metadata["meanings"]:
                        if meaning["type"] == meaning_type:
                            response = f"Regarding the {indicator} meaning of '{previous_topic}':\n\n"
                            response += meaning["definition"]
                            return response
        
        # No specific meaning follow-up identified
        return None
    
    def _extract_causes_from_definition(self, definition: str) -> str:
        """Extract information about causes from a definition"""
        # Look for sentences containing cause-related words
        sentences = sent_tokenize(definition)
        cause_sentences = []
        
        cause_words = ['cause', 'reason', 'because', 'due to', 'result of', 'lead to', 'factor']
        for sentence in sentences:
            if any(word in sentence.lower() for word in cause_words):
                cause_sentences.append(sentence)
        
        if cause_sentences:
            return ' '.join(cause_sentences)
        return ""
    
    def _identify_question_intent(self, question: str) -> str:
        """Identify the intent of a question (what, why, how, etc.)"""
        question_lower = question.lower()
        
        for intent, keywords in self.question_intents.items():
            for keyword in keywords:
                if question_lower.startswith(keyword + ' ') or f' {keyword} ' in question_lower:
                    return intent
        
        return 'what'  # Default intent
    
    def _extract_question_topics(self, question: str) -> List[str]:
        """
        Extract the main topics of a question using multiple techniques
        Returns a list of topics in order of relevance
        """
        question_clean = question.strip().lower()
        if not question_clean:
            return []
        
        # Get question intent
        intent = self._identify_question_intent(question_clean)
        
        # Check if this is a follow-up question about a recent topic
        follow_up_topic = self._check_for_follow_up(question_clean, intent)
        
        candidate_topics = []
        
        # If follow-up detected, prioritize that topic
        if follow_up_topic:
            candidate_topics.append(follow_up_topic)
        
        # Check for compound topics first
        compound_topic = self._extract_compound_topic(question_clean)
        if compound_topic:
            candidate_topics.append(compound_topic)
        
        # Method 1: Pattern matching for common question formats
        pattern_topics = self._extract_topics_by_patterns(question_clean)
        if pattern_topics:
            candidate_topics.extend(pattern_topics)
        
        # Method 2: Named Entity Recognition (simplified)
        entity_topics = self._extract_entities(question_clean)
        if entity_topics:
            candidate_topics.extend(entity_topics)
        
        # Method 3: POS tagging to find noun phrases
        np_topics = self._extract_noun_phrases(question_clean)
        if np_topics:
            candidate_topics.extend(np_topics)
        
        # Method 4: Extract keywords (least precise method)
        keyword_topics = self._extract_keywords(question_clean)
        if keyword_topics:
            candidate_topics.extend(keyword_topics)
        
        # Process all candidate topics
        processed_topics = []
        seen = set()
        
        for topic in candidate_topics:
            # Normalize
            topic = topic.lower().strip()
            
            # Skip very short topics
            if len(topic) < 3:
                continue
                
            # Skip topics that are just question words
            if topic in self.question_words:
                continue
            
            # Clean up topic
            topic = self._clean_topic(topic)
            
            # Skip if already processed or empty
            if not topic or topic in seen:
                continue
                
            # Add to processed list
            processed_topics.append(topic)
            seen.add(topic)
        
        # Special handling for 'why' questions about causes
        if intent == 'why' and processed_topics and not follow_up_topic:
            # Check if this is a "why" question about a known topic
            topic = processed_topics[0]
            if 'cause' in question_clean or 'reason' in question_clean:
                for pair in self.related_concept_pairs:
                    if pair[0] == topic and pair[1] in ['cause', 'reason']:
                        processed_topics.insert(0, pair[2])
                        break
        
        # Update recent topics for context
        if processed_topics:
            # Only store actual topics, not questions or phrases
            topic_to_store = processed_topics[0]
            if len(topic_to_store.split()) <= 3:  # Only store reasonably short topics
                if topic_to_store not in self.recent_topics:
                    self.recent_topics.insert(0, topic_to_store)
                    self.recent_topics = self.recent_topics[:self.max_recent_topics]
        
        # Limit to reasonable number of topics
        return processed_topics[:3]
    
    def _check_for_follow_up(self, question: str, intent: str) -> Optional[str]:
        """Check if this is a follow-up question about a recent topic"""
        if not self.recent_topics:
            return None
            
        # For why/how questions that don't contain their own clear topic
        if intent in ['why', 'how'] and len(question.split()) < 10:
            latest_topic = self.recent_topics[0]
            
            # Check for pronouns that might refer to the previous topic
            pronouns = ['it', 'they', 'this', 'that', 'these', 'those', 'them']
            if any(pronoun in question.split() for pronoun in pronouns):
                return latest_topic
                
            # Check if the question is very short and doesn't contain any nouns
            tokens = word_tokenize(question)
            tagged = nltk.pos_tag(tokens)
            has_nouns = any(tag.startswith('NN') for _, tag in tagged)
            
            if not has_nouns:
                return latest_topic
                
            # Special case for questions like "Why?" or "How come?"
            if question == "why?" or question == "how?" or question == "how come?":
                return latest_topic
        
        # For questions that explicitly mention recent topics
        for topic in self.recent_topics:
            if topic in question:
                return topic
                
        return None
    
    def _extract_compound_topic(self, question: str) -> Optional[str]:
        """Extract compound topics from questions"""
        # First check our predefined related pairs
        tokens = word_tokenize(question.lower())
        for pair in self.related_concept_pairs:
            if pair[0] in tokens and pair[1] in tokens:
                return pair[2]
        
        # Then check compound patterns
        for pattern in self.compound_patterns:
            matches = re.findall(pattern, question)
            if matches:
                # Take the first match
                if isinstance(matches[0], tuple):
                    terms = [term.strip() for term in matches[0] if term.strip()]
                    if len(terms) >= 2:
                        # Form compound topic
                        # Special case for "causes of X", "effects of X", etc.
                        if terms[0] in ['cause', 'causes', 'effect', 'effects', 'impact']:
                            return f"{terms[0]} of {terms[1]}"
                        elif "between" in pattern:
                            return f"{terms[0]} and {terms[1]}"
                        else:
                            return f"{terms[0]} {terms[1]}"
                else:
                    return matches[0].strip()
        
        # Special handling for certain question types
        intent = self._identify_question_intent(question)
        if intent == 'why' and 'war' in tokens and any(word in tokens for word in ['state', 'states', 'country', 'countries']):
            return 'causes of war'
            
        return None
    
    def _extract_topics_by_patterns(self, question: str) -> List[str]:
        """Extract topics using common question patterns"""
        patterns = [
            # What is X?
            r'what\s+(?:is|are)\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$|when|where|why|how)',
            # Who is X?
            r'who\s+(?:is|are)\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$|when|where|why|how)',
            # Tell me about X
            r'tell\s+(?:me|us)\s+about\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$|when|where|why|how)',
            # Define X
            r'define\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$|when|where|why|how)',
            # Explain X
            r'explain\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$|when|where|why|how)',
            # Do you know about X?
            r'do\s+you\s+know\s+about\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$|when|where|why|how)',
            # X is what?
            r'(?:a|an|the)?\s*([\w\s\-\']+)\s+is\s+what(?:\?|$)',
            # What do you know about X?
            r'what\s+do\s+you\s+know\s+about\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$)',
            # I want to learn about X
            r'i\s+want\s+to\s+(?:learn|know)\s+about\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$)',
            # Can you tell me about X?
            r'can\s+you\s+tell\s+(?:me|us)\s+about\s+(?:a|an|the)?\s*([\w\s\-\']+?)(?:\?|$)',
            # Why do states have war?
            r'why\s+do\s+(?:states|countries|nations)\s+have\s+([\w\s\-\']+?)(?:\?|$)',
            # Why is there war?
            r'why\s+is\s+there\s+([\w\s\-\']+?)(?:\?|$)',
        ]
        
        topics = []
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                if match and len(match) > 2:  # Avoid very short matches
                    # Check for domain-specific prefixes and extract the actual topic
                    topic = self._process_domain_specific_topic(match)
                    topics.append(topic)
        
        return topics
    
    def _process_domain_specific_topic(self, topic: str) -> str:
        """Process topics with domain-specific prefixes like 'the game X'"""
        topic = topic.strip().lower()
        
        # Check all domain prefixes
        for domain, prefixes in self.domain_prefixes.items():
            for prefix in prefixes:
                if topic.startswith(f"the {prefix} "):
                    return topic.replace(f"the {prefix} ", "")
                if topic.startswith(f"{prefix} "):
                    return topic.replace(f"{prefix} ", "")
        
        return topic
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract potential named entities from the question"""
        # Simple heuristic for named entities: sequences of capitalized words
        # In a real system, you'd use a proper NER model
        entities = []
        
        # Remove punctuation for cleaner extraction
        clean_q = re.sub(r'[^\w\s]', ' ', question)
        
        # Find sequences of capitalized words in the original question
        capitalized_sequences = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
        if capitalized_sequences:
            entities.extend(capitalized_sequences)
        
        return entities
    
    def _extract_noun_phrases(self, question: str) -> List[str]:
        """Extract noun phrases using POS tagging"""
        try:
            # Tokenize and tag
            tokens = word_tokenize(question)
            tagged = nltk.pos_tag(tokens)
            
            # Extract nouns and noun phrases
            phrases = []
            current_phrase = []
            
            for word, tag in tagged:
                # Skip question words and stopwords
                if word.lower() in self.question_words or word.lower() in STOP_WORDS:
                    if current_phrase:
                        phrases.append(' '.join(current_phrase))
                        current_phrase = []
                    continue
                
                # Build noun phrases
                if tag.startswith('NN') or tag == 'JJ':  # Nouns and adjectives
                    current_phrase.append(word)
                elif current_phrase:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
            
            # Add the last phrase if there is one
            if current_phrase:
                phrases.append(' '.join(current_phrase))
            
            return phrases
        except Exception as e:
            logger.warning(f"Error extracting noun phrases: {e}")
            return []
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords as a fallback method"""
        # Tokenize and remove stopwords
        tokens = word_tokenize(question)
        tokens = [t.lower() for t in tokens if t.isalnum() and len(t) > 3]
        
        # Remove question words and stopwords
        tokens = [t for t in tokens if t not in self.question_words and t not in STOP_WORDS]
        
        # Count and return most frequent
        counter = Counter(tokens)
        return [word for word, _ in counter.most_common(3)]
    
    def _clean_topic(self, topic: str) -> str:
        """Clean up an extracted topic"""
        # Remove any leading/trailing punctuation or whitespace
        topic = topic.strip()
        topic = re.sub(r'^[\s\.,;:!?\'\"]+|[\s\.,;:!?\'\"]+$', '', topic)
        
        # Remove articles at the beginning
        topic = re.sub(r'^(?:a|an|the)\s+', '', topic, flags=re.IGNORECASE)
        
        # Remove question words at the beginning
        for word in self.question_words:
            if topic.lower().startswith(word + ' '):
                topic = topic[len(word)+1:]
                
        # Final cleanup of whitespace
        topic = topic.strip()
        
        return topic
    
    def _handle_information(self, text: str) -> str:
        """Handle information provided by the user"""
        # Extract key topics from the information
        topics = self._extract_topics_from_information(text)
        
        # Learn from the text
        learned_concepts = self.learning_engine.learn_from_text(text)
        
        # Combine explicitly learned concepts with extracted topics
        all_concepts = set(learned_concepts)
        for topic in topics:
            all_concepts.add(topic)
            
        # Explore key topics in depth if they're new
        for topic in topics[:2]:  # Limit to top 2 topics to avoid excessive exploration
            existing = self.knowledge_base.get_knowledge(topic)
            if not existing or existing.confidence < 0.5:
                # Trigger exploration for this topic
                logger.info(f"Exploring key topic from information: '{topic}'")
                self.learning_engine.explorer.explore_concept(topic)
        
        # Generate a response based on what was learned
        if all_concepts:
            response = f"Thank you for the information. I've learned more about: {', '.join(all_concepts)}."
            
            # Add detail about one of the concepts
            if topics:
                primary = topics[0]
                node = self.knowledge_base.get_knowledge(primary)
                if node and node.confidence > 0.5:
                    response += f"\n\nMy understanding of {primary} is now:\n{node.definition[:200]}..."
            
            return response
        else:
            # Nothing specific was learned, give a generic response
            return "Thank you for sharing. I've integrated this information with what I already know."
    
    def _extract_topics_from_information(self, text: str) -> List[str]:
        """Extract key topics from informational text"""
        # This is similar to question topic extraction but with some differences
        
        # First, split into sentences for better processing
        sentences = sent_tokenize(text)
        
        # Extract noun phrases from each sentence
        all_phrases = []
        for sentence in sentences:
            phrases = self._extract_noun_phrases(sentence)
            all_phrases.extend(phrases)
        
        # Extract potential named entities
        entities = []
        for sentence in sentences:
            sentence_entities = self._extract_entities(sentence)
            entities.extend(sentence_entities)
        
        # Combine and count occurrences
        all_candidates = all_phrases + entities
        
        # Clean and normalize
        cleaned_candidates = []
        for candidate in all_candidates:
            clean = self._clean_topic(candidate)
            if clean and len(clean) > 2:
                cleaned_candidates.append(clean)
        
        # Count occurrences
        counter = Counter(cleaned_candidates)
        
        # Return most frequent
        return [topic for topic, _ in counter.most_common(5)]
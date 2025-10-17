"""
Advanced NLP Engine
High-accuracy natural language processing for command interpretation.
Supports intent recognition, entity extraction, and context understanding.
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .context_manager import ContextManager
from .command_parser import CommandParser


@dataclass
class NLPResult:
    """Result of NLP processing"""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    command: Dict[str, Any]
    context: Dict[str, Any]
    raw_text: str
    processed_text: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IntentDefinition:
    """Definition of an intent"""
    name: str
    description: str
    examples: List[str]
    required_entities: List[str]
    optional_entities: List[str]
    action_type: str
    confidence_threshold: float = 0.8


class AdvancedNLPEngine:
    """Advanced Natural Language Processing Engine"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.intent_classifier: Optional[IntentClassifier] = None
        self.entity_extractor: Optional[EntityExtractor] = None
        self.context_manager: Optional[ContextManager] = None
        self.command_parser: Optional[CommandParser] = None
        
        # Language models
        self.tokenizer: Optional[AutoTokenizer] = None
        self.language_model: Optional[AutoModel] = None
        self.spacy_model: Optional[spacy.Language] = None
        
        # Intent definitions
        self.intents: Dict[str, IntentDefinition] = {}
        
        # Model paths
        self.model_dir = Path("models/nlp")
        self.intents_path = self.model_dir / "intents.json"
        self.model_path = self.model_dir / "nlp_model.pth"
        
        # Performance tracking
        self.accuracy_history: List[float] = []
        self.processing_times: List[float] = []
        
        # Text preprocessing
        self.text_normalizer = TextNormalizer()
        
        # Similarity matching
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.intent_vectors: Optional[np.ndarray] = None
    
    async def initialize(self):
        """Initialize the NLP engine"""
        try:
            self.logger.info("Initializing Advanced NLP Engine...")
            
            # Create model directory
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize components
            await self._initialize_language_models()
            await self._load_intents()
            await self._initialize_components()
            
            # Build intent vectors for similarity matching
            await self._build_intent_vectors()
            
            self.logger.info("NLP Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP Engine: {e}")
            return False
    
    async def _initialize_language_models(self):
        """Initialize language models"""
        try:
            # Load transformer model for embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.language_model = AutoModel.from_pretrained(model_name)
            
            # Load spaCy model for NER and linguistic features
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found, using basic NLP")
                # Create a basic nlp pipeline
                self.spacy_model = None
            
            self.logger.info("Language models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Language model initialization error: {e}")
            raise
    
    async def _load_intents(self):
        """Load intent definitions"""
        try:
            if self.intents_path.exists():
                with open(self.intents_path, 'r', encoding='utf-8') as f:
                    intents_data = json.load(f)
                    
                for intent_data in intents_data:
                    intent = IntentDefinition(**intent_data)
                    self.intents[intent.name] = intent
            else:
                await self._create_default_intents()
            
            self.logger.info(f"Loaded {len(self.intents)} intent definitions")
            
        except Exception as e:
            self.logger.error(f"Intent loading error: {e}")
            await self._create_default_intents()
    
    async def _create_default_intents(self):
        """Create default intent definitions"""
        default_intents = [
            IntentDefinition(
                name="system_control",
                description="Control system functions",
                examples=[
                    "open calculator",
                    "close notepad",
                    "minimize window",
                    "maximize browser",
                    "switch to desktop"
                ],
                required_entities=["action", "target"],
                optional_entities=["window_name"],
                action_type="system"
            ),
            IntentDefinition(
                name="file_management",
                description="File and folder operations",
                examples=[
                    "create new folder",
                    "delete this file",
                    "copy document to desktop",
                    "move photos to pictures folder",
                    "rename file to report"
                ],
                required_entities=["action", "file_path"],
                optional_entities=["destination", "new_name"],
                action_type="file"
            ),
            IntentDefinition(
                name="web_search",
                description="Web search and browsing",
                examples=[
                    "search for python tutorials",
                    "open google",
                    "navigate to github",
                    "find weather forecast",
                    "look up stock prices"
                ],
                required_entities=["query"],
                optional_entities=["website", "search_type"],
                action_type="web"
            ),
            IntentDefinition(
                name="text_processing",
                description="Text manipulation and editing",
                examples=[
                    "type hello world",
                    "select all text",
                    "copy this paragraph",
                    "paste clipboard content",
                    "find and replace text"
                ],
                required_entities=["action"],
                optional_entities=["text_content", "find_text", "replace_text"],
                action_type="text"
            ),
            IntentDefinition(
                name="information_query",
                description="Information requests and queries",
                examples=[
                    "what time is it",
                    "what's the weather like",
                    "tell me about machine learning",
                    "explain quantum computing",
                    "show system information"
                ],
                required_entities=["query_type"],
                optional_entities=["topic", "location"],
                action_type="query"
            ),
            IntentDefinition(
                name="code_assistance",
                description="Programming and code-related tasks",
                examples=[
                    "write python function",
                    "debug this code",
                    "explain this algorithm",
                    "generate unit tests",
                    "refactor this method"
                ],
                required_entities=["task_type"],
                optional_entities=["language", "code_snippet", "function_name"],
                action_type="code"
            ),
            IntentDefinition(
                name="financial_analysis",
                description="Financial data and analysis",
                examples=[
                    "show stock price for apple",
                    "analyze market trends",
                    "calculate portfolio value",
                    "track cryptocurrency prices",
                    "generate financial report"
                ],
                required_entities=["analysis_type"],
                optional_entities=["symbol", "timeframe", "metric"],
                action_type="financial"
            ),
            IntentDefinition(
                name="automation_task",
                description="Automated task execution",
                examples=[
                    "schedule email reminder",
                    "automate file backup",
                    "set up recurring task",
                    "create workflow",
                    "monitor system performance"
                ],
                required_entities=["task_type"],
                optional_entities=["schedule", "parameters", "conditions"],
                action_type="automation"
            )
        ]
        
        # Save default intents
        intents_data = [asdict(intent) for intent in default_intents]
        with open(self.intents_path, 'w', encoding='utf-8') as f:
            json.dump(intents_data, f, indent=2)
        
        # Store in memory
        for intent in default_intents:
            self.intents[intent.name] = intent
        
        self.logger.info(f"Created {len(default_intents)} default intents")
    
    async def _initialize_components(self):
        """Initialize NLP components"""
        try:
            # Initialize intent classifier
            self.intent_classifier = IntentClassifier(
                intents=self.intents,
                language_model=self.language_model,
                tokenizer=self.tokenizer
            )
            await self.intent_classifier.initialize()
            
            # Initialize entity extractor
            self.entity_extractor = EntityExtractor(
                spacy_model=self.spacy_model,
                language_model=self.language_model,
                tokenizer=self.tokenizer
            )
            await self.entity_extractor.initialize()
            
            # Initialize context manager
            self.context_manager = ContextManager()
            await self.context_manager.initialize()
            
            # Initialize command parser
            self.command_parser = CommandParser(intents=self.intents)
            await self.command_parser.initialize()
            
            self.logger.info("NLP components initialized")
            
        except Exception as e:
            self.logger.error(f"Component initialization error: {e}")
            raise
    
    async def _build_intent_vectors(self):
        """Build TF-IDF vectors for intent similarity matching"""
        try:
            # Collect all intent examples
            all_examples = []
            intent_labels = []
            
            for intent_name, intent_def in self.intents.items():
                for example in intent_def.examples:
                    all_examples.append(self.text_normalizer.normalize(example))
                    intent_labels.append(intent_name)
            
            # Build TF-IDF vectors
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            self.intent_vectors = self.tfidf_vectorizer.fit_transform(all_examples)
            self.intent_labels = intent_labels
            
            self.logger.info("Intent vectors built successfully")
            
        except Exception as e:
            self.logger.error(f"Intent vector building error: {e}")
    
    async def process(self, text: str, context: Optional[Dict] = None) -> NLPResult:
        """Process natural language input"""
        start_time = datetime.now()
        
        try:
            # Normalize text
            processed_text = self.text_normalizer.normalize(text)
            
            # Update context
            if context:
                await self.context_manager.update_context(context)
            
            # Intent classification
            intent_result = await self.intent_classifier.classify(processed_text)
            
            # Entity extraction
            entities = await self.entity_extractor.extract(processed_text, intent_result['intent'])
            
            # Command parsing
            command = await self.command_parser.parse(
                text=processed_text,
                intent=intent_result['intent'],
                entities=entities
            )
            
            # Get current context
            current_context = await self.context_manager.get_context()
            
            # Create result
            result = NLPResult(
                intent=intent_result['intent'],
                confidence=intent_result['confidence'],
                entities=entities,
                command=command,
                context=current_context,
                raw_text=text,
                processed_text=processed_text,
                timestamp=datetime.now()
            )
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_times.append(processing_time)
            
            # Update context with result
            await self.context_manager.add_interaction(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"NLP processing error: {e}")
            
            # Return fallback result
            return NLPResult(
                intent="unknown",
                confidence=0.0,
                entities={},
                command={},
                context={},
                raw_text=text,
                processed_text=text,
                timestamp=datetime.now()
            )
    
    async def add_intent(self, intent_def: IntentDefinition):
        """Add new intent definition"""
        try:
            self.intents[intent_def.name] = intent_def
            
            # Retrain classifier
            await self.intent_classifier.add_intent(intent_def)
            
            # Rebuild vectors
            await self._build_intent_vectors()
            
            # Save intents
            await self._save_intents()
            
            self.logger.info(f"Added intent: {intent_def.name}")
            
        except Exception as e:
            self.logger.error(f"Intent addition error: {e}")
    
    async def _save_intents(self):
        """Save intent definitions to file"""
        try:
            intents_data = [asdict(intent) for intent in self.intents.values()]
            with open(self.intents_path, 'w', encoding='utf-8') as f:
                json.dump(intents_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Intent saving error: {e}")
    
    async def get_similar_intents(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find similar intents using TF-IDF similarity"""
        try:
            if self.tfidf_vectorizer is None or self.intent_vectors is None:
                return []
            
            # Normalize and vectorize input text
            normalized_text = self.text_normalizer.normalize(text)
            text_vector = self.tfidf_vectorizer.transform([normalized_text])
            
            # Calculate similarities
            similarities = cosine_similarity(text_vector, self.intent_vectors)[0]
            
            # Get top similar intents
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            seen_intents = set()
            
            for idx in top_indices:
                intent_name = self.intent_labels[idx]
                similarity = similarities[idx]
                
                if intent_name not in seen_intents and similarity > 0.1:
                    results.append((intent_name, similarity))
                    seen_intents.add(intent_name)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Similar intent search error: {e}")
            return []
    
    async def validate_command(self, result: NLPResult) -> bool:
        """Validate if command has required entities"""
        try:
            if result.intent not in self.intents:
                return False
            
            intent_def = self.intents[result.intent]
            
            # Check required entities
            for required_entity in intent_def.required_entities:
                if required_entity not in result.entities:
                    return False
            
            # Check confidence threshold
            if result.confidence < intent_def.confidence_threshold:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Command validation error: {e}")
            return False
    
    async def get_suggestions(self, partial_text: str) -> List[str]:
        """Get command suggestions based on partial input"""
        try:
            suggestions = []
            
            # Find similar intents
            similar_intents = await self.get_similar_intents(partial_text, top_k=5)
            
            for intent_name, similarity in similar_intents:
                if intent_name in self.intents:
                    intent_def = self.intents[intent_name]
                    # Add example commands as suggestions
                    for example in intent_def.examples[:2]:  # Limit to 2 examples per intent
                        if example.lower().startswith(partial_text.lower()):
                            suggestions.append(example)
            
            return suggestions[:10]  # Limit total suggestions
            
        except Exception as e:
            self.logger.error(f"Suggestion generation error: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get NLP engine performance metrics"""
        try:
            metrics = {
                "total_processed": len(self.processing_times),
                "average_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
                "max_processing_time": np.max(self.processing_times) if self.processing_times else 0,
                "min_processing_time": np.min(self.processing_times) if self.processing_times else 0,
                "accuracy_history": self.accuracy_history[-10:],  # Last 10 accuracy measurements
                "average_accuracy": np.mean(self.accuracy_history) if self.accuracy_history else 0,
                "total_intents": len(self.intents),
                "context_size": len(self.context_manager.get_context_sync()) if self.context_manager else 0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {e}")
            return {}
    
    async def update_accuracy(self, accuracy: float):
        """Update accuracy tracking"""
        self.accuracy_history.append(accuracy)
        
        # Keep only last 100 measurements
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]
    
    async def cleanup(self):
        """Cleanup NLP engine resources"""
        self.logger.info("Cleaning up NLP Engine...")
        
        if self.intent_classifier:
            await self.intent_classifier.cleanup()
        
        if self.entity_extractor:
            await self.entity_extractor.cleanup()
        
        if self.context_manager:
            await self.context_manager.cleanup()
        
        if self.command_parser:
            await self.command_parser.cleanup()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("NLP Engine cleanup completed")


class TextNormalizer:
    """Text normalization utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def normalize(self, text: str) -> str:
        """Normalize text for processing"""
        try:
            # Convert to lowercase
            text = text.lower().strip()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Expand contractions
            contractions = {
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would",
                "'m": " am"
            }
            
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
            
            # Remove punctuation except important ones
            text = re.sub(r'[^\w\s\-\.]', ' ', text)
            
            # Remove extra spaces again
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Text normalization error: {e}")
            return text
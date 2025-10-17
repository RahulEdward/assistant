"""
Entity Extractor
Advanced entity extraction using NER models and custom patterns.
Extracts relevant entities for command execution.
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from datetime import datetime, timedelta
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class CustomEntityExtractor:
    """Custom entity extraction with pattern matching and NER"""
    
    def __init__(self, spacy_model, language_model, tokenizer):
        self.spacy_model = spacy_model
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Entity patterns
        self.patterns: Dict[str, List[Dict]] = {}
        self.matchers: Dict[str, Matcher] = {}
        self.phrase_matchers: Dict[str, PhraseMatcher] = {}
        
        # NER pipeline
        self.ner_pipeline: Optional[Any] = None
        
        # Entity types
        self.entity_types = {
            'action': ['open', 'close', 'minimize', 'maximize', 'create', 'delete', 'copy', 'move', 'rename'],
            'application': ['notepad', 'calculator', 'browser', 'chrome', 'firefox', 'word', 'excel'],
            'file_type': ['folder', 'file', 'document', 'image', 'video', 'audio', 'text'],
            'location': ['desktop', 'documents', 'downloads', 'pictures', 'music', 'videos'],
            'time': ['now', 'today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'evening'],
            'number': [],  # Will be extracted using regex
            'url': [],     # Will be extracted using regex
            'email': [],   # Will be extracted using regex
            'path': []     # Will be extracted using regex
        }
        
        # Regex patterns
        self.regex_patterns = {
            'number': r'\b\d+(?:\.\d+)?\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'path': r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
            'time_12h': r'\b(?:1[0-2]|0?[1-9]):(?:[0-5][0-9])\s*(?:AM|PM|am|pm)\b',
            'time_24h': r'\b(?:2[0-3]|[01]?[0-9]):(?:[0-5][0-9])\b',
            'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|usd)\b'
        }
        
        # Model paths
        self.model_dir = Path("models/nlp/entity_extractor")
        self.patterns_path = self.model_dir / "entity_patterns.json"
    
    async def initialize(self):
        """Initialize the entity extractor"""
        try:
            self.logger.info("Initializing Entity Extractor...")
            
            # Create model directory
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize NER pipeline
            await self._initialize_ner_pipeline()
            
            # Load or create patterns
            await self._load_patterns()
            
            # Initialize matchers
            await self._initialize_matchers()
            
            self.logger.info("Entity Extractor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Entity Extractor initialization error: {e}")
            return False
    
    async def _initialize_ner_pipeline(self):
        """Initialize NER pipeline"""
        try:
            # Use a pre-trained NER model
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple"
            )
            
            self.logger.info("NER pipeline initialized")
            
        except Exception as e:
            self.logger.warning(f"NER pipeline initialization failed: {e}")
            self.ner_pipeline = None
    
    async def _load_patterns(self):
        """Load entity patterns"""
        try:
            if self.patterns_path.exists():
                with open(self.patterns_path, 'r', encoding='utf-8') as f:
                    self.patterns = json.load(f)
            else:
                await self._create_default_patterns()
            
            self.logger.info(f"Loaded patterns for {len(self.patterns)} entity types")
            
        except Exception as e:
            self.logger.error(f"Pattern loading error: {e}")
            await self._create_default_patterns()
    
    async def _create_default_patterns(self):
        """Create default entity patterns"""
        try:
            self.patterns = {
                'action': [
                    {'LOWER': {'IN': ['open', 'launch', 'start', 'run']}},
                    {'LOWER': {'IN': ['close', 'exit', 'quit', 'terminate']}},
                    {'LOWER': {'IN': ['minimize', 'hide']}},
                    {'LOWER': {'IN': ['maximize', 'expand', 'fullscreen']}},
                    {'LOWER': {'IN': ['create', 'make', 'new']}},
                    {'LOWER': {'IN': ['delete', 'remove', 'erase']}},
                    {'LOWER': {'IN': ['copy', 'duplicate']}},
                    {'LOWER': {'IN': ['move', 'transfer', 'relocate']}},
                    {'LOWER': {'IN': ['rename', 'change name']}},
                    {'LOWER': {'IN': ['search', 'find', 'look for']}},
                    {'LOWER': {'IN': ['type', 'write', 'enter']}},
                    {'LOWER': {'IN': ['click', 'press', 'select']}},
                    {'LOWER': {'IN': ['scroll', 'navigate']}},
                    {'LOWER': {'IN': ['save', 'store']}},
                    {'LOWER': {'IN': ['load', 'open file']}}
                ],
                'application': [
                    {'LOWER': {'IN': ['notepad', 'text editor']}},
                    {'LOWER': {'IN': ['calculator', 'calc']}},
                    {'LOWER': {'IN': ['browser', 'chrome', 'firefox', 'edge', 'safari']}},
                    {'LOWER': {'IN': ['word', 'microsoft word', 'ms word']}},
                    {'LOWER': {'IN': ['excel', 'microsoft excel', 'ms excel']}},
                    {'LOWER': {'IN': ['powerpoint', 'microsoft powerpoint', 'ms powerpoint']}},
                    {'LOWER': {'IN': ['outlook', 'microsoft outlook', 'ms outlook']}},
                    {'LOWER': {'IN': ['explorer', 'file explorer', 'windows explorer']}},
                    {'LOWER': {'IN': ['cmd', 'command prompt', 'terminal']}},
                    {'LOWER': {'IN': ['powershell', 'ps']}},
                    {'LOWER': {'IN': ['task manager', 'taskmgr']}},
                    {'LOWER': {'IN': ['control panel', 'settings']}}
                ],
                'file_type': [
                    {'LOWER': {'IN': ['folder', 'directory', 'dir']}},
                    {'LOWER': {'IN': ['file', 'document', 'doc']}},
                    {'LOWER': {'IN': ['image', 'picture', 'photo', 'img']}},
                    {'LOWER': {'IN': ['video', 'movie', 'clip']}},
                    {'LOWER': {'IN': ['audio', 'music', 'sound']}},
                    {'LOWER': {'IN': ['text', 'txt', 'note']}},
                    {'LOWER': {'IN': ['pdf', 'document']}},
                    {'LOWER': {'IN': ['zip', 'archive', 'compressed']}},
                    {'LOWER': {'IN': ['exe', 'executable', 'program']}},
                    {'LOWER': {'IN': ['script', 'batch', 'code']}}
                ],
                'location': [
                    {'LOWER': {'IN': ['desktop', 'desktop folder']}},
                    {'LOWER': {'IN': ['documents', 'my documents']}},
                    {'LOWER': {'IN': ['downloads', 'download folder']}},
                    {'LOWER': {'IN': ['pictures', 'photos', 'images folder']}},
                    {'LOWER': {'IN': ['music', 'audio folder']}},
                    {'LOWER': {'IN': ['videos', 'movies folder']}},
                    {'LOWER': {'IN': ['temp', 'temporary', 'tmp']}},
                    {'LOWER': {'IN': ['recycle bin', 'trash']}},
                    {'LOWER': {'IN': ['program files', 'programs']}},
                    {'LOWER': {'IN': ['system', 'windows', 'system32']}}
                ],
                'direction': [
                    {'LOWER': {'IN': ['up', 'down', 'left', 'right']}},
                    {'LOWER': {'IN': ['top', 'bottom', 'center', 'middle']}},
                    {'LOWER': {'IN': ['next', 'previous', 'forward', 'back']}},
                    {'LOWER': {'IN': ['first', 'last', 'beginning', 'end']}}
                ],
                'modifier': [
                    {'LOWER': {'IN': ['all', 'everything', 'entire']}},
                    {'LOWER': {'IN': ['selected', 'highlighted', 'chosen']}},
                    {'LOWER': {'IN': ['current', 'active', 'focused']}},
                    {'LOWER': {'IN': ['new', 'fresh', 'blank']}},
                    {'LOWER': {'IN': ['existing', 'current', 'old']}}
                ]
            }
            
            # Save patterns
            with open(self.patterns_path, 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, indent=2)
            
            self.logger.info("Created default entity patterns")
            
        except Exception as e:
            self.logger.error(f"Default pattern creation error: {e}")
    
    async def _initialize_matchers(self):
        """Initialize spaCy matchers"""
        try:
            if not self.spacy_model:
                return
            
            # Initialize matchers for each entity type
            for entity_type, patterns in self.patterns.items():
                matcher = Matcher(self.spacy_model.vocab)
                
                # Add patterns to matcher
                for i, pattern in enumerate(patterns):
                    pattern_name = f"{entity_type}_{i}"
                    matcher.add(pattern_name, [pattern])
                
                self.matchers[entity_type] = matcher
            
            # Initialize phrase matchers for entity lists
            for entity_type, entity_list in self.entity_types.items():
                if entity_list:
                    phrase_matcher = PhraseMatcher(self.spacy_model.vocab)
                    patterns = [self.spacy_model(text) for text in entity_list]
                    phrase_matcher.add(entity_type, patterns)
                    self.phrase_matchers[entity_type] = phrase_matcher
            
            self.logger.info("Matchers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Matcher initialization error: {e}")
    
    async def extract(self, text: str, intent: str = None) -> Dict[str, Any]:
        """Extract entities from text"""
        try:
            entities = {}
            
            # Extract using different methods
            spacy_entities = await self._extract_with_spacy(text)
            pattern_entities = await self._extract_with_patterns(text)
            regex_entities = await self._extract_with_regex(text)
            ner_entities = await self._extract_with_ner(text)
            
            # Merge all entities
            entities.update(spacy_entities)
            entities.update(pattern_entities)
            entities.update(regex_entities)
            entities.update(ner_entities)
            
            # Intent-specific entity extraction
            if intent:
                intent_entities = await self._extract_intent_specific(text, intent)
                entities.update(intent_entities)
            
            # Post-process entities
            entities = await self._post_process_entities(entities, text)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction error: {e}")
            return {}
    
    async def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy NER"""
        try:
            if not self.spacy_model:
                return {}
            
            entities = {}
            doc = self.spacy_model(text)
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = ent.label_.lower()
                entity_text = ent.text
                
                # Map spaCy labels to our entity types
                mapped_type = self._map_spacy_label(entity_type)
                
                if mapped_type:
                    if mapped_type not in entities:
                        entities[mapped_type] = []
                    
                    entities[mapped_type].append({
                        'text': entity_text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8  # Default confidence for spaCy
                    })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"spaCy entity extraction error: {e}")
            return {}
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'person': 'person',
            'org': 'organization',
            'gpe': 'location',
            'date': 'date',
            'time': 'time',
            'money': 'currency',
            'percent': 'percentage',
            'cardinal': 'number',
            'ordinal': 'number'
        }
        
        return mapping.get(spacy_label)
    
    async def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Extract entities using pattern matching"""
        try:
            if not self.spacy_model:
                return {}
            
            entities = {}
            doc = self.spacy_model(text)
            
            # Use pattern matchers
            for entity_type, matcher in self.matchers.items():
                matches = matcher(doc)
                
                if matches:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    for match_id, start, end in matches:
                        span = doc[start:end]
                        entities[entity_type].append({
                            'text': span.text,
                            'start': span.start_char,
                            'end': span.end_char,
                            'confidence': 0.9  # High confidence for pattern matches
                        })
            
            # Use phrase matchers
            for entity_type, phrase_matcher in self.phrase_matchers.items():
                matches = phrase_matcher(doc)
                
                if matches:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    for match_id, start, end in matches:
                        span = doc[start:end]
                        entities[entity_type].append({
                            'text': span.text,
                            'start': span.start_char,
                            'end': span.end_char,
                            'confidence': 0.95  # Very high confidence for exact matches
                        })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Pattern entity extraction error: {e}")
            return {}
    
    async def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract entities using regex patterns"""
        try:
            entities = {}
            
            for entity_type, pattern in self.regex_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    entities[entity_type].append({
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.85  # Good confidence for regex matches
                    })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Regex entity extraction error: {e}")
            return {}
    
    async def _extract_with_ner(self, text: str) -> Dict[str, Any]:
        """Extract entities using transformer NER pipeline"""
        try:
            if not self.ner_pipeline:
                return {}
            
            entities = {}
            ner_results = self.ner_pipeline(text)
            
            for result in ner_results:
                entity_type = result['entity_group'].lower()
                entity_text = result['word']
                confidence = result['score']
                
                # Map NER labels to our entity types
                mapped_type = self._map_ner_label(entity_type)
                
                if mapped_type and confidence > 0.5:  # Confidence threshold
                    if mapped_type not in entities:
                        entities[mapped_type] = []
                    
                    entities[mapped_type].append({
                        'text': entity_text,
                        'start': result['start'],
                        'end': result['end'],
                        'confidence': confidence
                    })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"NER entity extraction error: {e}")
            return {}
    
    def _map_ner_label(self, ner_label: str) -> Optional[str]:
        """Map NER labels to our entity types"""
        mapping = {
            'per': 'person',
            'org': 'organization',
            'loc': 'location',
            'misc': 'miscellaneous'
        }
        
        return mapping.get(ner_label)
    
    async def _extract_intent_specific(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities specific to the intent"""
        try:
            entities = {}
            
            # Intent-specific extraction rules
            if intent == 'file_management':
                # Extract file paths more aggressively
                file_patterns = [
                    r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.[a-zA-Z0-9]+',
                    r'\.\\[^\\/:*?"<>|\r\n]*',
                    r'\\\\[^\\/:*?"<>|\r\n]+\\[^\\/:*?"<>|\r\n]*'
                ]
                
                for pattern in file_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        if 'file_path' not in entities:
                            entities['file_path'] = []
                        
                        entities['file_path'].append({
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': 0.9
                        })
            
            elif intent == 'web_search':
                # Extract search queries
                query_patterns = [
                    r'search for (.+?)(?:\s+on|\s+in|\s*$)',
                    r'find (.+?)(?:\s+on|\s+in|\s*$)',
                    r'look up (.+?)(?:\s+on|\s+in|\s*$)'
                ]
                
                for pattern in query_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if 'query' not in entities:
                            entities['query'] = []
                        
                        entities['query'].append({
                            'text': match.group(1).strip(),
                            'start': match.start(1),
                            'end': match.end(1),
                            'confidence': 0.85
                        })
            
            elif intent == 'system_control':
                # Extract window names
                window_patterns = [
                    r'(?:window|app|application)\s+(?:named|called)\s+([^,.\s]+)',
                    r'([A-Z][a-zA-Z0-9\s]+)(?:\s+window|\s+app)'
                ]
                
                for pattern in window_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if 'window_name' not in entities:
                            entities['window_name'] = []
                        
                        entities['window_name'].append({
                            'text': match.group(1).strip(),
                            'start': match.start(1),
                            'end': match.end(1),
                            'confidence': 0.8
                        })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Intent-specific extraction error: {e}")
            return {}
    
    async def _post_process_entities(self, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Post-process extracted entities"""
        try:
            processed = {}
            
            for entity_type, entity_list in entities.items():
                if not entity_list:
                    continue
                
                # Remove duplicates and overlaps
                unique_entities = self._remove_overlapping_entities(entity_list)
                
                # Normalize entity values
                normalized_entities = []
                for entity in unique_entities:
                    normalized = self._normalize_entity(entity, entity_type)
                    if normalized:
                        normalized_entities.append(normalized)
                
                if normalized_entities:
                    processed[entity_type] = normalized_entities
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Entity post-processing error: {e}")
            return entities
    
    def _remove_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping the one with highest confidence"""
        if not entities:
            return []
        
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda x: x['confidence'], reverse=True)
        
        non_overlapping = []
        
        for entity in sorted_entities:
            overlaps = False
            
            for existing in non_overlapping:
                # Check for overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
    
    def _normalize_entity(self, entity: Dict, entity_type: str) -> Optional[Dict]:
        """Normalize entity value based on type"""
        try:
            text = entity['text'].strip()
            
            if entity_type == 'number':
                try:
                    # Convert to number
                    if '.' in text:
                        entity['value'] = float(text)
                    else:
                        entity['value'] = int(text)
                except ValueError:
                    return None
            
            elif entity_type == 'time':
                # Normalize time format
                entity['value'] = self._normalize_time(text)
            
            elif entity_type == 'date':
                # Normalize date format
                entity['value'] = self._normalize_date(text)
            
            elif entity_type == 'path':
                # Normalize file path
                entity['value'] = text.replace('/', '\\')  # Windows path format
            
            elif entity_type == 'url':
                # Ensure URL has protocol
                if not text.startswith(('http://', 'https://')):
                    entity['value'] = 'https://' + text
                else:
                    entity['value'] = text
            
            else:
                # Default: lowercase and strip
                entity['value'] = text.lower()
            
            return entity
            
        except Exception as e:
            self.logger.error(f"Entity normalization error: {e}")
            return entity
    
    def _normalize_time(self, time_str: str) -> str:
        """Normalize time string"""
        try:
            # Simple time normalization
            time_str = time_str.lower().strip()
            
            # Convert 12-hour to 24-hour format
            if 'am' in time_str or 'pm' in time_str:
                # Extract time and AM/PM
                time_part = re.search(r'(\d{1,2}):(\d{2})', time_str)
                am_pm = 'pm' if 'pm' in time_str else 'am'
                
                if time_part:
                    hour = int(time_part.group(1))
                    minute = int(time_part.group(2))
                    
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    return f"{hour:02d}:{minute:02d}"
            
            return time_str
            
        except Exception:
            return time_str
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string"""
        try:
            # Simple date normalization to YYYY-MM-DD format
            date_str = date_str.strip()
            
            # Handle different date formats
            patterns = [
                (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', lambda m: f"{m.group(3)}-{m.group(1):0>2}-{m.group(2):0>2}"),
                (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2):0>2}-{m.group(3):0>2}")
            ]
            
            for pattern, formatter in patterns:
                match = re.match(pattern, date_str)
                if match:
                    return formatter(match)
            
            return date_str
            
        except Exception:
            return date_str
    
    async def add_entity_pattern(self, entity_type: str, pattern: Dict):
        """Add new entity pattern"""
        try:
            if entity_type not in self.patterns:
                self.patterns[entity_type] = []
            
            self.patterns[entity_type].append(pattern)
            
            # Reinitialize matchers
            await self._initialize_matchers()
            
            # Save patterns
            with open(self.patterns_path, 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, indent=2)
            
            self.logger.info(f"Added pattern for entity type: {entity_type}")
            
        except Exception as e:
            self.logger.error(f"Pattern addition error: {e}")
    
    async def cleanup(self):
        """Cleanup entity extractor resources"""
        self.logger.info("Cleaning up Entity Extractor...")
        
        self.ner_pipeline = None
        self.matchers.clear()
        self.phrase_matchers.clear()
        
        self.logger.info("Entity Extractor cleanup completed")


# Alias for backward compatibility
EntityExtractor = CustomEntityExtractor
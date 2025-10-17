"""
Context Manager
Manages conversation context, user preferences, and session state.
Maintains context for multi-turn conversations and personalization.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import pickle


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    timestamp: datetime
    user_input: str
    intent: str
    entities: Dict[str, Any]
    response: str
    success: bool
    execution_time: float
    context_used: Dict[str, Any]


@dataclass
class UserPreference:
    """User preference item"""
    key: str
    value: Any
    category: str
    timestamp: datetime
    confidence: float = 1.0


@dataclass
class ContextState:
    """Current context state"""
    active_application: Optional[str] = None
    current_directory: Optional[str] = None
    selected_files: List[str] = None
    clipboard_content: Optional[str] = None
    last_search_query: Optional[str] = None
    active_windows: List[str] = None
    user_focus: Optional[str] = None
    task_in_progress: Optional[str] = None
    
    def __post_init__(self):
        if self.selected_files is None:
            self.selected_files = []
        if self.active_windows is None:
            self.active_windows = []


class AdvancedContextManager:
    """Advanced context management with learning capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Context storage
        self.conversation_history: deque = deque(maxlen=100)  # Last 100 turns
        self.user_preferences: Dict[str, UserPreference] = {}
        self.context_state = ContextState()
        self.session_data: Dict[str, Any] = {}
        
        # Context analysis
        self.entity_memory: Dict[str, List[Dict]] = {}  # Remember entities by type
        self.pattern_memory: Dict[str, int] = {}  # Track usage patterns
        self.success_patterns: Dict[str, float] = {}  # Track successful patterns
        
        # Temporal context
        self.short_term_memory: deque = deque(maxlen=10)  # Last 10 interactions
        self.medium_term_memory: Dict[str, Any] = {}  # Session-level memory
        self.long_term_memory: Dict[str, Any] = {}  # Persistent memory
        
        # Context weights
        self.context_weights = {
            'recent_interaction': 0.4,
            'user_preference': 0.3,
            'current_state': 0.2,
            'historical_pattern': 0.1
        }
        
        # Storage paths
        self.data_dir = Path("data/context")
        self.conversation_path = self.data_dir / "conversation_history.pkl"
        self.preferences_path = self.data_dir / "user_preferences.json"
        self.memory_path = self.data_dir / "context_memory.pkl"
        
        # Context categories
        self.preference_categories = {
            'interface': ['voice_speed', 'response_format', 'verbosity_level'],
            'behavior': ['confirmation_required', 'auto_execute', 'learning_enabled'],
            'applications': ['default_browser', 'preferred_editor', 'file_manager'],
            'automation': ['automation_level', 'safety_checks', 'batch_operations'],
            'personalization': ['greeting_style', 'error_handling', 'suggestion_level']
        }
    
    async def initialize(self):
        """Initialize context manager"""
        try:
            self.logger.info("Initializing Context Manager...")
            
            # Create data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing data
            await self._load_conversation_history()
            await self._load_user_preferences()
            await self._load_context_memory()
            
            # Initialize session
            await self._initialize_session()
            
            self.logger.info("Context Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Context Manager initialization error: {e}")
            return False
    
    async def _load_conversation_history(self):
        """Load conversation history"""
        try:
            if self.conversation_path.exists():
                with open(self.conversation_path, 'rb') as f:
                    history_data = pickle.load(f)
                    self.conversation_history = deque(history_data, maxlen=100)
                
                self.logger.info(f"Loaded {len(self.conversation_history)} conversation turns")
            
        except Exception as e:
            self.logger.error(f"Conversation history loading error: {e}")
    
    async def _load_user_preferences(self):
        """Load user preferences"""
        try:
            if self.preferences_path.exists():
                with open(self.preferences_path, 'r', encoding='utf-8') as f:
                    prefs_data = json.load(f)
                
                # Convert to UserPreference objects
                for key, data in prefs_data.items():
                    self.user_preferences[key] = UserPreference(
                        key=data['key'],
                        value=data['value'],
                        category=data['category'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        confidence=data.get('confidence', 1.0)
                    )
                
                self.logger.info(f"Loaded {len(self.user_preferences)} user preferences")
            
        except Exception as e:
            self.logger.error(f"User preferences loading error: {e}")
    
    async def _load_context_memory(self):
        """Load context memory"""
        try:
            if self.memory_path.exists():
                with open(self.memory_path, 'rb') as f:
                    memory_data = pickle.load(f)
                    
                    self.entity_memory = memory_data.get('entity_memory', {})
                    self.pattern_memory = memory_data.get('pattern_memory', {})
                    self.success_patterns = memory_data.get('success_patterns', {})
                    self.long_term_memory = memory_data.get('long_term_memory', {})
                
                self.logger.info("Loaded context memory")
            
        except Exception as e:
            self.logger.error(f"Context memory loading error: {e}")
    
    async def _initialize_session(self):
        """Initialize new session"""
        try:
            self.session_data = {
                'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'start_time': datetime.now(),
                'interaction_count': 0,
                'successful_commands': 0,
                'failed_commands': 0,
                'user_satisfaction': 0.0,
                'learning_events': []
            }
            
            # Reset short-term memory for new session
            self.short_term_memory.clear()
            self.medium_term_memory.clear()
            
            self.logger.info(f"Initialized session: {self.session_data['session_id']}")
            
        except Exception as e:
            self.logger.error(f"Session initialization error: {e}")
    
    async def add_conversation_turn(self, turn: ConversationTurn):
        """Add conversation turn to history"""
        try:
            # Add to conversation history
            self.conversation_history.append(turn)
            
            # Add to short-term memory
            self.short_term_memory.append({
                'timestamp': turn.timestamp,
                'intent': turn.intent,
                'entities': turn.entities,
                'success': turn.success,
                'context_state': asdict(self.context_state)
            })
            
            # Update session statistics
            self.session_data['interaction_count'] += 1
            if turn.success:
                self.session_data['successful_commands'] += 1
            else:
                self.session_data['failed_commands'] += 1
            
            # Learn from interaction
            await self._learn_from_interaction(turn)
            
            # Update context state based on interaction
            await self._update_context_state(turn)
            
            self.logger.debug(f"Added conversation turn: {turn.intent}")
            
        except Exception as e:
            self.logger.error(f"Conversation turn addition error: {e}")
    
    async def _learn_from_interaction(self, turn: ConversationTurn):
        """Learn patterns from interaction"""
        try:
            # Update entity memory
            for entity_type, entities in turn.entities.items():
                if entity_type not in self.entity_memory:
                    self.entity_memory[entity_type] = []
                
                for entity in entities:
                    # Add entity with context
                    entity_record = {
                        'value': entity.get('value', entity.get('text')),
                        'timestamp': turn.timestamp,
                        'intent': turn.intent,
                        'success': turn.success,
                        'frequency': 1
                    }
                    
                    # Check if entity already exists
                    existing = None
                    for existing_entity in self.entity_memory[entity_type]:
                        if existing_entity['value'] == entity_record['value']:
                            existing = existing_entity
                            break
                    
                    if existing:
                        existing['frequency'] += 1
                        existing['timestamp'] = turn.timestamp
                    else:
                        self.entity_memory[entity_type].append(entity_record)
            
            # Update pattern memory
            pattern_key = f"{turn.intent}_{len(turn.entities)}"
            self.pattern_memory[pattern_key] = self.pattern_memory.get(pattern_key, 0) + 1
            
            # Update success patterns
            if turn.success:
                success_key = f"{turn.intent}_{turn.execution_time:.2f}"
                current_success = self.success_patterns.get(turn.intent, 0.0)
                self.success_patterns[turn.intent] = (current_success + 1.0) / 2.0
            
            # Learn user preferences
            await self._infer_preferences(turn)
            
        except Exception as e:
            self.logger.error(f"Learning from interaction error: {e}")
    
    async def _infer_preferences(self, turn: ConversationTurn):
        """Infer user preferences from interaction"""
        try:
            # Infer response format preference
            if turn.success and turn.execution_time < 1.0:
                await self._update_preference(
                    'response_speed', 'fast', 'behavior',
                    confidence=0.1
                )
            
            # Infer verbosity preference
            if len(turn.response) > 200:
                await self._update_preference(
                    'verbosity_level', 'detailed', 'interface',
                    confidence=0.05
                )
            elif len(turn.response) < 50:
                await self._update_preference(
                    'verbosity_level', 'concise', 'interface',
                    confidence=0.05
                )
            
            # Infer automation preference
            if turn.intent in ['system_control', 'file_management'] and turn.success:
                await self._update_preference(
                    'automation_level', 'high', 'automation',
                    confidence=0.1
                )
            
        except Exception as e:
            self.logger.error(f"Preference inference error: {e}")
    
    async def _update_context_state(self, turn: ConversationTurn):
        """Update context state based on interaction"""
        try:
            # Update based on intent
            if turn.intent == 'file_management':
                # Extract file paths from entities
                if 'file_path' in turn.entities:
                    paths = [e.get('value', e.get('text')) for e in turn.entities['file_path']]
                    if paths:
                        self.context_state.current_directory = str(Path(paths[0]).parent)
                        self.context_state.selected_files = paths
            
            elif turn.intent == 'system_control':
                # Update active application
                if 'application' in turn.entities:
                    apps = [e.get('value', e.get('text')) for e in turn.entities['application']]
                    if apps:
                        self.context_state.active_application = apps[0]
            
            elif turn.intent == 'web_search':
                # Update last search query
                if 'query' in turn.entities:
                    queries = [e.get('value', e.get('text')) for e in turn.entities['query']]
                    if queries:
                        self.context_state.last_search_query = queries[0]
            
            # Update task in progress
            if not turn.success:
                self.context_state.task_in_progress = turn.intent
            else:
                self.context_state.task_in_progress = None
            
        except Exception as e:
            self.logger.error(f"Context state update error: {e}")
    
    async def get_context_for_intent(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context for intent processing"""
        try:
            context = {
                'current_state': asdict(self.context_state),
                'recent_interactions': list(self.short_term_memory),
                'user_preferences': {},
                'relevant_entities': {},
                'suggested_values': {},
                'confidence_scores': {}
            }
            
            # Get relevant user preferences
            for pref_key, preference in self.user_preferences.items():
                if self._is_preference_relevant(preference, intent):
                    context['user_preferences'][pref_key] = {
                        'value': preference.value,
                        'confidence': preference.confidence
                    }
            
            # Get relevant entities from memory
            for entity_type in entities.keys():
                if entity_type in self.entity_memory:
                    # Get most frequent entities of this type
                    sorted_entities = sorted(
                        self.entity_memory[entity_type],
                        key=lambda x: x['frequency'],
                        reverse=True
                    )[:5]  # Top 5
                    
                    context['relevant_entities'][entity_type] = sorted_entities
            
            # Get suggested values based on context
            context['suggested_values'] = await self._get_suggested_values(intent, entities)
            
            # Calculate confidence scores
            context['confidence_scores'] = await self._calculate_confidence_scores(intent, entities)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context retrieval error: {e}")
            return {}
    
    def _is_preference_relevant(self, preference: UserPreference, intent: str) -> bool:
        """Check if preference is relevant to intent"""
        relevance_map = {
            'system_control': ['automation_level', 'safety_checks', 'confirmation_required'],
            'file_management': ['default_browser', 'preferred_editor', 'batch_operations'],
            'web_search': ['default_browser', 'verbosity_level'],
            'text_processing': ['preferred_editor', 'verbosity_level'],
            'information_query': ['response_format', 'verbosity_level'],
            'code_assistance': ['preferred_editor', 'automation_level'],
            'financial_analysis': ['verbosity_level', 'response_format'],
            'automation_task': ['automation_level', 'safety_checks', 'batch_operations']
        }
        
        relevant_prefs = relevance_map.get(intent, [])
        return preference.key in relevant_prefs
    
    async def _get_suggested_values(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Get suggested values based on context"""
        try:
            suggestions = {}
            
            # Suggest based on recent interactions
            for memory_item in self.short_term_memory:
                if memory_item['intent'] == intent and memory_item['success']:
                    for entity_type, entity_list in memory_item.get('entities', {}).items():
                        if entity_type not in entities:  # Only suggest missing entities
                            if entity_type not in suggestions:
                                suggestions[entity_type] = []
                            
                            for entity in entity_list:
                                value = entity.get('value', entity.get('text'))
                                if value not in [s['value'] for s in suggestions[entity_type]]:
                                    suggestions[entity_type].append({
                                        'value': value,
                                        'source': 'recent_interaction',
                                        'confidence': 0.7
                                    })
            
            # Suggest based on current context state
            if intent == 'file_management' and 'file_path' not in entities:
                if self.context_state.selected_files:
                    suggestions['file_path'] = [{
                        'value': path,
                        'source': 'current_selection',
                        'confidence': 0.9
                    } for path in self.context_state.selected_files]
            
            if intent == 'system_control' and 'application' not in entities:
                if self.context_state.active_application:
                    suggestions['application'] = [{
                        'value': self.context_state.active_application,
                        'source': 'current_focus',
                        'confidence': 0.8
                    }]
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Suggestion generation error: {e}")
            return {}
    
    async def _calculate_confidence_scores(self, intent: str, entities: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for context elements"""
        try:
            scores = {}
            
            # Base confidence from success patterns
            base_confidence = self.success_patterns.get(intent, 0.5)
            scores['intent_confidence'] = base_confidence
            
            # Entity confidence based on frequency
            for entity_type, entity_list in entities.items():
                entity_confidence = 0.0
                
                if entity_type in self.entity_memory:
                    for entity in entity_list:
                        value = entity.get('value', entity.get('text'))
                        
                        # Find matching entity in memory
                        for memory_entity in self.entity_memory[entity_type]:
                            if memory_entity['value'] == value:
                                # Higher frequency = higher confidence
                                freq_confidence = min(memory_entity['frequency'] / 10.0, 1.0)
                                entity_confidence = max(entity_confidence, freq_confidence)
                                break
                
                scores[f'{entity_type}_confidence'] = entity_confidence
            
            # Context state confidence
            state_confidence = 0.5
            if self.context_state.active_application:
                state_confidence += 0.2
            if self.context_state.current_directory:
                state_confidence += 0.2
            if self.context_state.selected_files:
                state_confidence += 0.1
            
            scores['context_confidence'] = min(state_confidence, 1.0)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return {'intent_confidence': 0.5}
    
    async def update_preference(self, key: str, value: Any, category: str = 'general', confidence: float = 1.0):
        """Update user preference"""
        await self._update_preference(key, value, category, confidence)
    
    async def _update_preference(self, key: str, value: Any, category: str, confidence: float):
        """Internal preference update"""
        try:
            # Update or create preference
            if key in self.user_preferences:
                existing = self.user_preferences[key]
                # Weighted average of confidences
                new_confidence = (existing.confidence + confidence) / 2.0
                self.user_preferences[key] = UserPreference(
                    key=key,
                    value=value,
                    category=category,
                    timestamp=datetime.now(),
                    confidence=new_confidence
                )
            else:
                self.user_preferences[key] = UserPreference(
                    key=key,
                    value=value,
                    category=category,
                    timestamp=datetime.now(),
                    confidence=confidence
                )
            
            self.logger.debug(f"Updated preference: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Preference update error: {e}")
    
    async def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference value"""
        try:
            if key in self.user_preferences:
                return self.user_preferences[key].value
            return default
            
        except Exception as e:
            self.logger.error(f"Preference retrieval error: {e}")
            return default
    
    async def update_context_state(self, **kwargs):
        """Update context state"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.context_state, key):
                    setattr(self.context_state, key, value)
                    self.logger.debug(f"Updated context state: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Context state update error: {e}")
    
    async def get_conversation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get conversation summary"""
        try:
            recent_turns = list(self.conversation_history)[-last_n:]
            
            summary = {
                'total_interactions': len(self.conversation_history),
                'recent_interactions': len(recent_turns),
                'success_rate': 0.0,
                'common_intents': {},
                'average_response_time': 0.0,
                'user_satisfaction': self.session_data.get('user_satisfaction', 0.0)
            }
            
            if recent_turns:
                successful = sum(1 for turn in recent_turns if turn.success)
                summary['success_rate'] = successful / len(recent_turns)
                
                # Count intents
                for turn in recent_turns:
                    intent = turn.intent
                    summary['common_intents'][intent] = summary['common_intents'].get(intent, 0) + 1
                
                # Average response time
                total_time = sum(turn.execution_time for turn in recent_turns)
                summary['average_response_time'] = total_time / len(recent_turns)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Conversation summary error: {e}")
            return {}
    
    async def save_context_data(self):
        """Save context data to disk"""
        try:
            # Save conversation history
            with open(self.conversation_path, 'wb') as f:
                pickle.dump(list(self.conversation_history), f)
            
            # Save user preferences
            prefs_data = {}
            for key, pref in self.user_preferences.items():
                prefs_data[key] = {
                    'key': pref.key,
                    'value': pref.value,
                    'category': pref.category,
                    'timestamp': pref.timestamp.isoformat(),
                    'confidence': pref.confidence
                }
            
            with open(self.preferences_path, 'w', encoding='utf-8') as f:
                json.dump(prefs_data, f, indent=2)
            
            # Save context memory
            memory_data = {
                'entity_memory': self.entity_memory,
                'pattern_memory': self.pattern_memory,
                'success_patterns': self.success_patterns,
                'long_term_memory': self.long_term_memory
            }
            
            with open(self.memory_path, 'wb') as f:
                pickle.dump(memory_data, f)
            
            self.logger.info("Context data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Context data saving error: {e}")
    
    async def cleanup(self):
        """Cleanup context manager"""
        self.logger.info("Cleaning up Context Manager...")
        
        # Save data before cleanup
        await self.save_context_data()
        
        # Clear memory structures
        self.conversation_history.clear()
        self.short_term_memory.clear()
        self.medium_term_memory.clear()
        self.entity_memory.clear()
        self.pattern_memory.clear()
        
        self.logger.info("Context Manager cleanup completed")


# Alias for backward compatibility
ContextManager = AdvancedContextManager
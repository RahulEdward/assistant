"""
Adaptive Learning System for Computer Assistant

This module provides adaptive learning capabilities including:
- Continuous learning from user interactions
- Pattern recognition and adaptation
- Performance optimization
- Behavioral learning and adjustment
- Real-time model updates
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class LearningType(Enum):
    """Types of learning"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    ONLINE = "online"
    BATCH = "batch"
    INCREMENTAL = "incremental"


class AdaptationType(Enum):
    """Types of adaptation"""
    PERFORMANCE = "performance"
    BEHAVIOR = "behavior"
    PREFERENCE = "preference"
    CONTEXT = "context"
    USAGE_PATTERN = "usage_pattern"
    ERROR_CORRECTION = "error_correction"


class LearningStatus(Enum):
    """Learning status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LearningEvent:
    """Learning event data structure"""
    id: str
    event_type: str
    timestamp: datetime
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'context': self.context,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'feedback': self.feedback,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class LearningPattern:
    """Identified learning pattern"""
    id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    first_seen: datetime
    last_seen: datetime
    examples: List[str] = field(default_factory=list)
    adaptations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'examples': self.examples,
            'adaptations': self.adaptations,
            'metadata': self.metadata
        }


@dataclass
class AdaptationRule:
    """Adaptation rule"""
    id: str
    name: str
    adaptation_type: AdaptationType
    condition: str
    action: str
    priority: int = 1
    enabled: bool = True
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'adaptation_type': self.adaptation_type.value,
            'condition': self.condition,
            'action': self.action,
            'priority': self.priority,
            'enabled': self.enabled,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'created_at': self.created_at.isoformat(),
            'last_applied': self.last_applied.isoformat() if self.last_applied else None,
            'metadata': self.metadata
        }


@dataclass
class LearningSession:
    """Learning session data"""
    id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    learning_type: LearningType = LearningType.ONLINE
    events_processed: int = 0
    patterns_discovered: int = 0
    adaptations_applied: int = 0
    performance_improvement: float = 0.0
    status: LearningStatus = LearningStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'learning_type': self.learning_type.value,
            'events_processed': self.events_processed,
            'patterns_discovered': self.patterns_discovered,
            'adaptations_applied': self.adaptations_applied,
            'performance_improvement': self.performance_improvement,
            'status': self.status.value,
            'metadata': self.metadata
        }


class AdaptiveLearner:
    """
    Adaptive learning system for continuous improvement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the adaptive learner"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Learning data storage
        self.learning_events: deque = deque(maxlen=self.config.get('max_events', 10000))
        self.patterns: Dict[str, LearningPattern] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.learning_sessions: Dict[str, LearningSession] = {}
        
        # Configuration
        self.learning_dir = Path(self.config.get('learning_dir', 'adaptive_learning'))
        self.learning_dir.mkdir(exist_ok=True)
        
        # Learning settings
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.pattern_threshold = self.config.get('pattern_threshold', 0.7)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.8)
        self.max_patterns = self.config.get('max_patterns', 1000)
        self.max_rules = self.config.get('max_rules', 500)
        
        # Real-time learning
        self.real_time_learning = self.config.get('real_time_learning', True)
        self.batch_size = self.config.get('batch_size', 100)
        self.learning_interval = self.config.get('learning_interval', 300)  # 5 minutes
        
        # Performance tracking
        self.performance_metrics = {
            'events_processed': 0,
            'patterns_discovered': 0,
            'adaptations_applied': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'learning_accuracy': 0.0,
            'adaptation_success_rate': 0.0,
            'average_response_time': 0.0,
            'user_satisfaction': 0.0,
            'daily_stats': {}
        }
        
        # User profiles and preferences
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.context_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Pattern recognition
        self.pattern_detectors = {
            'usage_frequency': self._detect_usage_patterns,
            'error_patterns': self._detect_error_patterns,
            'performance_patterns': self._detect_performance_patterns,
            'preference_patterns': self._detect_preference_patterns,
            'context_patterns': self._detect_context_patterns
        }
        
        # Adaptation strategies
        self.adaptation_strategies = {
            AdaptationType.PERFORMANCE: self._adapt_performance,
            AdaptationType.BEHAVIOR: self._adapt_behavior,
            AdaptationType.PREFERENCE: self._adapt_preferences,
            AdaptationType.CONTEXT: self._adapt_context,
            AdaptationType.USAGE_PATTERN: self._adapt_usage_patterns,
            AdaptationType.ERROR_CORRECTION: self._adapt_error_correction
        }
        
        # Current learning session
        self.current_session: Optional[LearningSession] = None
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Learning lock for thread safety
        self.learning_lock = threading.Lock()
        
        # Initialize built-in adaptation rules
        self._initialize_adaptation_rules()
        
        # Load existing data
        self._load_learning_data()
        
        # Start learning session
        if self.real_time_learning:
            self._start_learning_session()
        
        self.logger.info("Adaptive Learner initialized")
    
    def _initialize_adaptation_rules(self) -> None:
        """Initialize built-in adaptation rules"""
        try:
            # Performance optimization rules
            self.add_adaptation_rule(
                "optimize_slow_operations",
                AdaptationType.PERFORMANCE,
                "response_time > 5.0",
                "cache_result_and_optimize_algorithm",
                priority=1
            )
            
            self.add_adaptation_rule(
                "reduce_memory_usage",
                AdaptationType.PERFORMANCE,
                "memory_usage > 0.8",
                "clear_cache_and_optimize_memory",
                priority=2
            )
            
            # Behavior adaptation rules
            self.add_adaptation_rule(
                "adapt_to_user_preferences",
                AdaptationType.PREFERENCE,
                "user_feedback_negative > 3",
                "adjust_behavior_based_on_feedback",
                priority=1
            )
            
            self.add_adaptation_rule(
                "learn_from_corrections",
                AdaptationType.ERROR_CORRECTION,
                "error_rate > 0.1",
                "update_model_with_corrections",
                priority=1
            )
            
            # Context adaptation rules
            self.add_adaptation_rule(
                "adapt_to_context_changes",
                AdaptationType.CONTEXT,
                "context_change_detected",
                "adjust_behavior_for_new_context",
                priority=2
            )
            
            # Usage pattern rules
            self.add_adaptation_rule(
                "optimize_frequent_operations",
                AdaptationType.USAGE_PATTERN,
                "operation_frequency > 10_per_day",
                "preload_and_optimize_frequent_operations",
                priority=2
            )
            
            self.logger.info("Built-in adaptation rules initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing adaptation rules: {e}")
    
    def _load_learning_data(self) -> None:
        """Load existing learning data"""
        try:
            # Load patterns
            patterns_file = self.learning_dir / "patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_data in patterns_data:
                    pattern = LearningPattern(
                        id=pattern_data['id'],
                        pattern_type=pattern_data['pattern_type'],
                        description=pattern_data['description'],
                        frequency=pattern_data['frequency'],
                        confidence=pattern_data['confidence'],
                        first_seen=datetime.fromisoformat(pattern_data['first_seen']),
                        last_seen=datetime.fromisoformat(pattern_data['last_seen']),
                        examples=pattern_data.get('examples', []),
                        adaptations=pattern_data.get('adaptations', []),
                        metadata=pattern_data.get('metadata', {})
                    )
                    self.patterns[pattern.id] = pattern
            
            # Load adaptation rules
            rules_file = self.learning_dir / "adaptation_rules.json"
            if rules_file.exists():
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data:
                    rule = AdaptationRule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        adaptation_type=AdaptationType(rule_data['adaptation_type']),
                        condition=rule_data['condition'],
                        action=rule_data['action'],
                        priority=rule_data.get('priority', 1),
                        enabled=rule_data.get('enabled', True),
                        success_count=rule_data.get('success_count', 0),
                        failure_count=rule_data.get('failure_count', 0),
                        created_at=datetime.fromisoformat(rule_data['created_at']),
                        last_applied=datetime.fromisoformat(rule_data['last_applied']) if rule_data.get('last_applied') else None,
                        metadata=rule_data.get('metadata', {})
                    )
                    self.adaptation_rules[rule.id] = rule
            
            # Load user profiles
            profiles_file = self.learning_dir / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    self.user_profiles = json.load(f)
            
            # Load performance metrics
            metrics_file = self.learning_dir / "performance_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.performance_metrics.update(json.load(f))
            
            self.logger.info(f"Loaded {len(self.patterns)} patterns, {len(self.adaptation_rules)} rules")
            
        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")
    
    async def save_learning_data(self) -> None:
        """Save learning data to disk"""
        try:
            # Save patterns
            patterns_file = self.learning_dir / "patterns.json"
            with open(patterns_file, 'w') as f:
                patterns_data = [pattern.to_dict() for pattern in self.patterns.values()]
                json.dump(patterns_data, f, indent=2)
            
            # Save adaptation rules
            rules_file = self.learning_dir / "adaptation_rules.json"
            with open(rules_file, 'w') as f:
                rules_data = [rule.to_dict() for rule in self.adaptation_rules.values()]
                json.dump(rules_data, f, indent=2)
            
            # Save user profiles
            profiles_file = self.learning_dir / "user_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(self.user_profiles, f, indent=2)
            
            # Save performance metrics
            metrics_file = self.learning_dir / "performance_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            self.logger.info("Learning data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
    
    def _start_learning_session(self) -> None:
        """Start a new learning session"""
        try:
            session_id = str(uuid.uuid4())
            self.current_session = LearningSession(
                id=session_id,
                start_time=datetime.now(),
                learning_type=LearningType.ONLINE
            )
            
            self.learning_sessions[session_id] = self.current_session
            
            self.logger.info(f"Started learning session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting learning session: {e}")
    
    async def record_event(self, event_type: str, input_data: Dict[str, Any],
                          output_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None,
                          user_id: Optional[str] = None, feedback: Optional[Dict[str, Any]] = None,
                          success: Optional[bool] = None) -> str:
        """Record a learning event"""
        try:
            event_id = str(uuid.uuid4())
            
            event = LearningEvent(
                id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                context=context or {},
                input_data=input_data,
                output_data=output_data,
                feedback=feedback,
                success=success
            )
            
            # Add to event queue
            self.learning_events.append(event)
            
            # Update current session
            if self.current_session:
                self.current_session.events_processed += 1
            
            # Update performance metrics
            self.performance_metrics['events_processed'] += 1
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            if today not in self.performance_metrics['daily_stats']:
                self.performance_metrics['daily_stats'][today] = {'events': 0}
            self.performance_metrics['daily_stats'][today]['events'] += 1
            
            # Update user profile
            if user_id:
                await self._update_user_profile(user_id, event)
            
            # Real-time learning
            if self.real_time_learning:
                await self._process_event_for_learning(event)
            
            self.logger.debug(f"Recorded learning event: {event_type}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Error recording event: {e}")
            raise
    
    async def _update_user_profile(self, user_id: str, event: LearningEvent) -> None:
        """Update user profile based on event"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    'created_at': datetime.now().isoformat(),
                    'total_events': 0,
                    'preferences': {},
                    'behavior_patterns': {},
                    'performance_history': [],
                    'feedback_history': [],
                    'context_preferences': {}
                }
            
            profile = self.user_profiles[user_id]
            profile['total_events'] += 1
            profile['last_activity'] = event.timestamp.isoformat()
            
            # Update preferences based on feedback
            if event.feedback:
                profile['feedback_history'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'feedback': event.feedback,
                    'event_type': event.event_type
                })
                
                # Keep only last 100 feedback entries
                if len(profile['feedback_history']) > 100:
                    profile['feedback_history'] = profile['feedback_history'][-100:]
            
            # Update context preferences
            if event.context:
                for key, value in event.context.items():
                    if key not in profile['context_preferences']:
                        profile['context_preferences'][key] = {}
                    
                    if str(value) not in profile['context_preferences'][key]:
                        profile['context_preferences'][key][str(value)] = 0
                    
                    profile['context_preferences'][key][str(value)] += 1
            
            # Update behavior patterns
            if event.event_type not in profile['behavior_patterns']:
                profile['behavior_patterns'][event.event_type] = {
                    'frequency': 0,
                    'success_rate': 0.0,
                    'average_response_time': 0.0,
                    'last_used': None
                }
            
            pattern = profile['behavior_patterns'][event.event_type]
            pattern['frequency'] += 1
            pattern['last_used'] = event.timestamp.isoformat()
            
            if event.success is not None:
                # Update success rate
                total_attempts = pattern['frequency']
                current_successes = pattern['success_rate'] * (total_attempts - 1)
                if event.success:
                    current_successes += 1
                pattern['success_rate'] = current_successes / total_attempts
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
    
    async def _process_event_for_learning(self, event: LearningEvent) -> None:
        """Process event for real-time learning"""
        try:
            with self.learning_lock:
                # Detect patterns
                await self._detect_patterns([event])
                
                # Apply adaptations
                await self._apply_adaptations(event)
                
                # Update context history
                if event.context:
                    context_key = f"{event.event_type}_{event.user_id or 'anonymous'}"
                    self.context_history[context_key].append({
                        'timestamp': event.timestamp.isoformat(),
                        'context': event.context,
                        'success': event.success
                    })
                    
                    # Keep only last 50 context entries per key
                    if len(self.context_history[context_key]) > 50:
                        self.context_history[context_key] = self.context_history[context_key][-50:]
            
        except Exception as e:
            self.logger.error(f"Error processing event for learning: {e}")
    
    async def _detect_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Detect patterns in learning events"""
        try:
            new_patterns = []
            
            # Run pattern detectors
            for detector_name, detector_func in self.pattern_detectors.items():
                try:
                    detected_patterns = await detector_func(events)
                    new_patterns.extend(detected_patterns)
                except Exception as e:
                    self.logger.error(f"Error in pattern detector {detector_name}: {e}")
            
            # Update existing patterns or create new ones
            for pattern in new_patterns:
                if pattern.id in self.patterns:
                    # Update existing pattern
                    existing = self.patterns[pattern.id]
                    existing.frequency += pattern.frequency
                    existing.last_seen = pattern.last_seen
                    existing.confidence = max(existing.confidence, pattern.confidence)
                    existing.examples.extend(pattern.examples)
                    
                    # Keep only last 10 examples
                    if len(existing.examples) > 10:
                        existing.examples = existing.examples[-10:]
                else:
                    # Add new pattern
                    if len(self.patterns) < self.max_patterns:
                        self.patterns[pattern.id] = pattern
                        
                        # Update session stats
                        if self.current_session:
                            self.current_session.patterns_discovered += 1
                        
                        # Update performance metrics
                        self.performance_metrics['patterns_discovered'] += 1
                        
                        self.logger.info(f"New pattern discovered: {pattern.description}")
            
            return new_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    async def _detect_usage_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Detect usage frequency patterns"""
        try:
            patterns = []
            
            # Count event types
            event_counts = defaultdict(int)
            for event in events:
                event_counts[event.event_type] += 1
            
            # Identify high-frequency patterns
            for event_type, count in event_counts.items():
                if count >= 5:  # Threshold for frequent usage
                    pattern_id = f"usage_freq_{event_type}"
                    
                    pattern = LearningPattern(
                        id=pattern_id,
                        pattern_type="usage_frequency",
                        description=f"High frequency usage of {event_type}",
                        frequency=count,
                        confidence=min(count / 10.0, 1.0),
                        first_seen=min(e.timestamp for e in events if e.event_type == event_type),
                        last_seen=max(e.timestamp for e in events if e.event_type == event_type),
                        examples=[event_type]
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting usage patterns: {e}")
            return []
    
    async def _detect_error_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Detect error patterns"""
        try:
            patterns = []
            
            # Find error events
            error_events = [e for e in events if e.success is False or e.error_message]
            
            if len(error_events) >= 3:  # Threshold for error pattern
                # Group by error type or context
                error_groups = defaultdict(list)
                
                for event in error_events:
                    if event.error_message:
                        # Group by error message similarity
                        error_key = event.error_message[:50]  # First 50 chars
                    else:
                        # Group by event type
                        error_key = event.event_type
                    
                    error_groups[error_key].append(event)
                
                # Create patterns for significant error groups
                for error_key, group_events in error_groups.items():
                    if len(group_events) >= 2:
                        pattern_id = f"error_pattern_{hash(error_key) % 10000}"
                        
                        pattern = LearningPattern(
                            id=pattern_id,
                            pattern_type="error_pattern",
                            description=f"Recurring error: {error_key}",
                            frequency=len(group_events),
                            confidence=min(len(group_events) / 5.0, 1.0),
                            first_seen=min(e.timestamp for e in group_events),
                            last_seen=max(e.timestamp for e in group_events),
                            examples=[e.event_type for e in group_events[:3]],
                            metadata={'error_key': error_key}
                        )
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting error patterns: {e}")
            return []
    
    async def _detect_performance_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Detect performance patterns"""
        try:
            patterns = []
            
            # Analyze response times from context
            response_times = []
            for event in events:
                if 'response_time' in event.context:
                    response_times.append((event, event.context['response_time']))
            
            if len(response_times) >= 5:
                # Find slow operations
                avg_time = sum(rt[1] for rt in response_times) / len(response_times)
                slow_operations = [rt for rt in response_times if rt[1] > avg_time * 2]
                
                if len(slow_operations) >= 3:
                    pattern_id = "performance_slow_operations"
                    
                    pattern = LearningPattern(
                        id=pattern_id,
                        pattern_type="performance_pattern",
                        description="Slow operation performance pattern",
                        frequency=len(slow_operations),
                        confidence=len(slow_operations) / len(response_times),
                        first_seen=min(rt[0].timestamp for rt in slow_operations),
                        last_seen=max(rt[0].timestamp for rt in slow_operations),
                        examples=[rt[0].event_type for rt in slow_operations[:3]],
                        metadata={'average_time': avg_time, 'slow_threshold': avg_time * 2}
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting performance patterns: {e}")
            return []
    
    async def _detect_preference_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Detect user preference patterns"""
        try:
            patterns = []
            
            # Analyze feedback patterns
            feedback_events = [e for e in events if e.feedback]
            
            if len(feedback_events) >= 3:
                # Group by user
                user_feedback = defaultdict(list)
                for event in feedback_events:
                    user_id = event.user_id or 'anonymous'
                    user_feedback[user_id].append(event)
                
                # Analyze preferences for each user
                for user_id, user_events in user_feedback.items():
                    positive_feedback = [e for e in user_events if e.feedback.get('rating', 0) > 3]
                    negative_feedback = [e for e in user_events if e.feedback.get('rating', 0) < 3]
                    
                    if len(positive_feedback) >= 2:
                        pattern_id = f"preference_positive_{user_id}"
                        
                        pattern = LearningPattern(
                            id=pattern_id,
                            pattern_type="preference_pattern",
                            description=f"Positive preferences for user {user_id}",
                            frequency=len(positive_feedback),
                            confidence=len(positive_feedback) / len(user_events),
                            first_seen=min(e.timestamp for e in positive_feedback),
                            last_seen=max(e.timestamp for e in positive_feedback),
                            examples=[e.event_type for e in positive_feedback[:3]],
                            metadata={'user_id': user_id, 'feedback_type': 'positive'}
                        )
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting preference patterns: {e}")
            return []
    
    async def _detect_context_patterns(self, events: List[LearningEvent]) -> List[LearningPattern]:
        """Detect context-based patterns"""
        try:
            patterns = []
            
            # Analyze context combinations
            context_combinations = defaultdict(list)
            
            for event in events:
                if event.context:
                    # Create context signature
                    context_items = sorted(event.context.items())
                    context_sig = tuple((k, str(v)) for k, v in context_items if k != 'timestamp')
                    
                    if context_sig:
                        context_combinations[context_sig].append(event)
            
            # Find frequent context patterns
            for context_sig, context_events in context_combinations.items():
                if len(context_events) >= 3:
                    pattern_id = f"context_pattern_{hash(context_sig) % 10000}"
                    
                    pattern = LearningPattern(
                        id=pattern_id,
                        pattern_type="context_pattern",
                        description=f"Context pattern: {dict(context_sig)}",
                        frequency=len(context_events),
                        confidence=min(len(context_events) / 5.0, 1.0),
                        first_seen=min(e.timestamp for e in context_events),
                        last_seen=max(e.timestamp for e in context_events),
                        examples=[e.event_type for e in context_events[:3]],
                        metadata={'context_signature': dict(context_sig)}
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting context patterns: {e}")
            return []
    
    async def _apply_adaptations(self, event: LearningEvent) -> None:
        """Apply adaptations based on event and patterns"""
        try:
            # Get applicable rules
            applicable_rules = []
            
            for rule in self.adaptation_rules.values():
                if rule.enabled and self._evaluate_condition(rule.condition, event):
                    applicable_rules.append(rule)
            
            # Sort by priority
            applicable_rules.sort(key=lambda r: r.priority)
            
            # Apply rules
            for rule in applicable_rules:
                try:
                    success = await self._execute_adaptation(rule, event)
                    
                    if success:
                        rule.success_count += 1
                        rule.last_applied = datetime.now()
                        
                        # Update session stats
                        if self.current_session:
                            self.current_session.adaptations_applied += 1
                        
                        # Update performance metrics
                        self.performance_metrics['adaptations_applied'] += 1
                        self.performance_metrics['successful_adaptations'] += 1
                        
                        self.logger.info(f"Applied adaptation rule: {rule.name}")
                    else:
                        rule.failure_count += 1
                        self.performance_metrics['failed_adaptations'] += 1
                        
                except Exception as e:
                    rule.failure_count += 1
                    self.performance_metrics['failed_adaptations'] += 1
                    self.logger.error(f"Error applying adaptation rule {rule.name}: {e}")
            
            # Update adaptation success rate
            total_adaptations = (self.performance_metrics['successful_adaptations'] + 
                               self.performance_metrics['failed_adaptations'])
            if total_adaptations > 0:
                self.performance_metrics['adaptation_success_rate'] = (
                    self.performance_metrics['successful_adaptations'] / total_adaptations
                )
            
        except Exception as e:
            self.logger.error(f"Error applying adaptations: {e}")
    
    def _evaluate_condition(self, condition: str, event: LearningEvent) -> bool:
        """Evaluate adaptation rule condition"""
        try:
            # Create evaluation context
            context = {
                'event_type': event.event_type,
                'success': event.success,
                'user_id': event.user_id,
                'timestamp': event.timestamp,
                'response_time': event.context.get('response_time', 0),
                'memory_usage': event.context.get('memory_usage', 0),
                'error_rate': event.context.get('error_rate', 0),
                'user_feedback_negative': event.feedback.get('negative_count', 0) if event.feedback else 0,
                'context_change_detected': event.context.get('context_changed', False),
                'operation_frequency': event.context.get('frequency', 0)
            }
            
            # Simple condition evaluation (in practice, you might use a more sophisticated parser)
            try:
                # Replace variables in condition
                eval_condition = condition
                for key, value in context.items():
                    eval_condition = eval_condition.replace(key, str(value))
                
                # Evaluate simple conditions
                if '>' in eval_condition:
                    parts = eval_condition.split('>')
                    if len(parts) == 2:
                        left = float(parts[0].strip())
                        right = float(parts[1].strip())
                        return left > right
                
                elif '<' in eval_condition:
                    parts = eval_condition.split('<')
                    if len(parts) == 2:
                        left = float(parts[0].strip())
                        right = float(parts[1].strip())
                        return left < right
                
                elif '==' in eval_condition:
                    parts = eval_condition.split('==')
                    if len(parts) == 2:
                        left = parts[0].strip().strip('"\'')
                        right = parts[1].strip().strip('"\'')
                        return left == right
                
                elif eval_condition.lower() in ['true', 'false']:
                    return eval_condition.lower() == 'true'
                
                return False
                
            except Exception:
                return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _execute_adaptation(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Execute adaptation action"""
        try:
            adaptation_type = rule.adaptation_type
            
            if adaptation_type in self.adaptation_strategies:
                return await self.adaptation_strategies[adaptation_type](rule, event)
            else:
                self.logger.warning(f"Unknown adaptation type: {adaptation_type}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error executing adaptation: {e}")
            return False
    
    async def _adapt_performance(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Adapt performance based on rule"""
        try:
            action = rule.action
            
            if "cache_result" in action:
                # Implement result caching
                cache_key = f"{event.event_type}_{hash(str(event.input_data))}"
                # In practice, you would implement actual caching logic
                self.logger.info(f"Caching result for key: {cache_key}")
                return True
            
            elif "optimize_algorithm" in action:
                # Implement algorithm optimization
                self.logger.info(f"Optimizing algorithm for event type: {event.event_type}")
                return True
            
            elif "clear_cache" in action:
                # Implement cache clearing
                self.logger.info("Clearing cache to reduce memory usage")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in performance adaptation: {e}")
            return False
    
    async def _adapt_behavior(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Adapt behavior based on rule"""
        try:
            # Implement behavior adaptation logic
            self.logger.info(f"Adapting behavior based on rule: {rule.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in behavior adaptation: {e}")
            return False
    
    async def _adapt_preferences(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Adapt preferences based on rule"""
        try:
            if event.user_id and event.feedback:
                # Update user preferences based on feedback
                if event.user_id in self.user_profiles:
                    profile = self.user_profiles[event.user_id]
                    
                    # Adjust preferences based on feedback
                    if 'preferences' not in profile:
                        profile['preferences'] = {}
                    
                    feedback_score = event.feedback.get('rating', 3)
                    if feedback_score > 3:
                        # Positive feedback - reinforce this behavior
                        pref_key = f"{event.event_type}_preference"
                        profile['preferences'][pref_key] = profile['preferences'].get(pref_key, 0) + 0.1
                    elif feedback_score < 3:
                        # Negative feedback - reduce this behavior
                        pref_key = f"{event.event_type}_preference"
                        profile['preferences'][pref_key] = profile['preferences'].get(pref_key, 0) - 0.1
                    
                    self.logger.info(f"Updated preferences for user: {event.user_id}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in preference adaptation: {e}")
            return False
    
    async def _adapt_context(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Adapt context handling based on rule"""
        try:
            # Implement context adaptation logic
            self.logger.info(f"Adapting context handling for event: {event.event_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in context adaptation: {e}")
            return False
    
    async def _adapt_usage_patterns(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Adapt usage patterns based on rule"""
        try:
            # Implement usage pattern adaptation
            self.logger.info(f"Adapting usage patterns for: {event.event_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in usage pattern adaptation: {e}")
            return False
    
    async def _adapt_error_correction(self, rule: AdaptationRule, event: LearningEvent) -> bool:
        """Adapt error correction based on rule"""
        try:
            if event.error_message or event.success is False:
                # Implement error correction learning
                error_key = event.error_message or f"failure_{event.event_type}"
                
                # In practice, you would update models or rules to prevent this error
                self.logger.info(f"Learning from error: {error_key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in error correction adaptation: {e}")
            return False
    
    def add_adaptation_rule(self, name: str, adaptation_type: AdaptationType,
                           condition: str, action: str, priority: int = 1) -> str:
        """Add a new adaptation rule"""
        try:
            rule_id = str(uuid.uuid4())
            
            rule = AdaptationRule(
                id=rule_id,
                name=name,
                adaptation_type=adaptation_type,
                condition=condition,
                action=action,
                priority=priority
            )
            
            self.adaptation_rules[rule_id] = rule
            
            self.logger.info(f"Added adaptation rule: {name}")
            return rule_id
            
        except Exception as e:
            self.logger.error(f"Error adding adaptation rule: {e}")
            raise
    
    async def batch_learn(self, events: Optional[List[LearningEvent]] = None) -> Dict[str, Any]:
        """Perform batch learning on events"""
        try:
            if events is None:
                events = list(self.learning_events)
            
            if not events:
                return {'patterns_discovered': 0, 'adaptations_applied': 0}
            
            # Start batch learning session
            session_id = str(uuid.uuid4())
            session = LearningSession(
                id=session_id,
                start_time=datetime.now(),
                learning_type=LearningType.BATCH
            )
            
            self.learning_sessions[session_id] = session
            
            # Detect patterns
            new_patterns = await self._detect_patterns(events)
            session.patterns_discovered = len(new_patterns)
            
            # Apply adaptations to recent events
            adaptations_applied = 0
            for event in events[-self.batch_size:]:  # Process last batch_size events
                await self._apply_adaptations(event)
                adaptations_applied += 1
            
            session.adaptations_applied = adaptations_applied
            session.end_time = datetime.now()
            session.status = LearningStatus.STOPPED
            
            # Save learning data
            await self.save_learning_data()
            
            result = {
                'session_id': session_id,
                'events_processed': len(events),
                'patterns_discovered': session.patterns_discovered,
                'adaptations_applied': session.adaptations_applied,
                'learning_time': (session.end_time - session.start_time).total_seconds()
            }
            
            self.logger.info(f"Batch learning completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in batch learning: {e}")
            raise
    
    def get_patterns(self, pattern_type: Optional[str] = None,
                    min_confidence: float = 0.0) -> List[LearningPattern]:
        """Get discovered patterns"""
        patterns = list(self.patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if min_confidence > 0:
            patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)
    
    def get_adaptation_rules(self, adaptation_type: Optional[AdaptationType] = None,
                           enabled_only: bool = True) -> List[AdaptationRule]:
        """Get adaptation rules"""
        rules = list(self.adaptation_rules.values())
        
        if adaptation_type:
            rules = [r for r in rules if r.adaptation_type == adaptation_type]
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        return sorted(rules, key=lambda r: r.priority)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        return self.user_profiles.get(user_id)
    
    def get_learning_sessions(self) -> List[LearningSession]:
        """Get learning sessions"""
        return list(self.learning_sessions.values())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    async def enable_rule(self, rule_id: str) -> None:
        """Enable adaptation rule"""
        if rule_id in self.adaptation_rules:
            self.adaptation_rules[rule_id].enabled = True
            self.logger.info(f"Enabled adaptation rule: {rule_id}")
        else:
            raise ValueError(f"Rule {rule_id} not found")
    
    async def disable_rule(self, rule_id: str) -> None:
        """Disable adaptation rule"""
        if rule_id in self.adaptation_rules:
            self.adaptation_rules[rule_id].enabled = False
            self.logger.info(f"Disabled adaptation rule: {rule_id}")
        else:
            raise ValueError(f"Rule {rule_id} not found")
    
    async def delete_pattern(self, pattern_id: str) -> None:
        """Delete a learning pattern"""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            self.logger.info(f"Deleted pattern: {pattern_id}")
        else:
            raise ValueError(f"Pattern {pattern_id} not found")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        
        # Update specific settings
        if 'learning_rate' in new_config:
            self.learning_rate = new_config['learning_rate']
        
        if 'pattern_threshold' in new_config:
            self.pattern_threshold = new_config['pattern_threshold']
        
        if 'adaptation_threshold' in new_config:
            self.adaptation_threshold = new_config['adaptation_threshold']
        
        if 'real_time_learning' in new_config:
            self.real_time_learning = new_config['real_time_learning']
        
        if 'batch_size' in new_config:
            self.batch_size = new_config['batch_size']
        
        if 'learning_interval' in new_config:
            self.learning_interval = new_config['learning_interval']
        
        self.logger.info("Adaptive Learner configuration updated")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # End current session
            if self.current_session and self.current_session.status == LearningStatus.ACTIVE:
                self.current_session.end_time = datetime.now()
                self.current_session.status = LearningStatus.STOPPED
            
            # Save learning data
            await self.save_learning_data()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear old sessions (keep last 20)
            if len(self.learning_sessions) > 20:
                sorted_sessions = sorted(
                    self.learning_sessions.items(),
                    key=lambda x: x[1].start_time,
                    reverse=True
                )
                self.learning_sessions = dict(sorted_sessions[:20])
            
            # Clear old daily stats (keep last 30 days)
            if 'daily_stats' in self.performance_metrics:
                cutoff_date = (datetime.now() - timedelta(days=30)).date()
                self.performance_metrics['daily_stats'] = {
                    date: stats for date, stats in self.performance_metrics['daily_stats'].items()
                    if datetime.fromisoformat(date).date() >= cutoff_date
                }
            
            # Clear old context history
            for key in list(self.context_history.keys()):
                if len(self.context_history[key]) > 50:
                    self.context_history[key] = self.context_history[key][-50:]
            
            self.logger.info("Adaptive Learner cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
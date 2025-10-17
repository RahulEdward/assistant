"""
AI Manager for Computer Assistant

This module provides centralized management of AI services including:
- Multiple AI provider support (OpenAI, Anthropic, etc.)
- Model selection and switching
- Request routing and load balancing
- Response processing and caching
- Usage tracking and cost management
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
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class AIProvider(Enum):
    """AI service providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    LOCAL = "local"


class ModelCapability(Enum):
    """AI model capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseStatus(Enum):
    """Response status"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass
class AIModel:
    """AI model configuration"""
    id: str
    name: str
    provider: AIProvider
    model_id: str
    capabilities: List[ModelCapability]
    max_tokens: int = 4096
    cost_per_token: float = 0.0
    rate_limit: int = 60  # requests per minute
    context_window: int = 4096
    supports_streaming: bool = False
    supports_functions: bool = False
    temperature_range: tuple = (0.0, 2.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'provider': self.provider.value,
            'model_id': self.model_id,
            'capabilities': [cap.value for cap in self.capabilities],
            'max_tokens': self.max_tokens,
            'cost_per_token': self.cost_per_token,
            'rate_limit': self.rate_limit,
            'context_window': self.context_window,
            'supports_streaming': self.supports_streaming,
            'supports_functions': self.supports_functions,
            'temperature_range': list(self.temperature_range),
            'metadata': self.metadata
        }


@dataclass
class AIRequest:
    """AI request"""
    id: str
    provider: AIProvider
    model_id: str
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: int = 30
    retry_count: int = 3
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'provider': self.provider.value,
            'model_id': self.model_id,
            'prompt': self.prompt,
            'parameters': self.parameters,
            'priority': self.priority.value,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AIResponse:
    """AI response"""
    id: str
    request_id: str
    provider: AIProvider
    model_id: str
    content: str
    status: ResponseStatus
    tokens_used: int = 0
    cost: float = 0.0
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'provider': self.provider.value,
            'model_id': self.model_id,
            'content': self.content,
            'status': self.status.value,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'response_time': self.response_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'error_message': self.error_message
        }


@dataclass
class ConversationContext:
    """Conversation context for AI interactions"""
    id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message to conversation"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation messages"""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'messages': self.messages,
            'system_prompt': self.system_prompt,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class AIManager:
    """
    Central AI management system for handling multiple providers and models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AI manager"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.models: Dict[str, AIModel] = {}
        self.providers: Dict[AIProvider, Any] = {}  # Provider clients
        self.request_history: deque = deque(maxlen=1000)
        self.response_cache: Dict[str, AIResponse] = {}
        
        # Configuration
        self.ai_dir = Path(self.config.get('ai_dir', 'ai_data'))
        self.ai_dir.mkdir(exist_ok=True)
        
        # Request management
        self.request_queue: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        self.active_requests: Dict[str, AIRequest] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Settings
        self.default_model = self.config.get('default_model')
        self.fallback_models = self.config.get('fallback_models', [])
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.enable_load_balancing = self.config.get('enable_load_balancing', True)
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'provider_stats': {},
            'model_stats': {},
            'daily_stats': {}
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        
        # Request processing lock
        self.request_lock = threading.Lock()
        
        # Initialize built-in models
        self._initialize_models()
        
        # Load existing data
        self._load_ai_data()
        
        # Start request processor
        self._start_request_processor()
        
        self.logger.info("AI Manager initialized")
    
    def _initialize_models(self) -> None:
        """Initialize built-in AI models"""
        try:
            # OpenAI models
            self.add_model(AIModel(
                id="gpt-4",
                name="GPT-4",
                provider=AIProvider.OPENAI,
                model_id="gpt-4",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CONVERSATION,
                    ModelCapability.ANALYSIS,
                    ModelCapability.FUNCTION_CALLING
                ],
                max_tokens=8192,
                cost_per_token=0.00003,
                context_window=8192,
                supports_streaming=True,
                supports_functions=True
            ))
            
            self.add_model(AIModel(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider=AIProvider.OPENAI,
                model_id="gpt-3.5-turbo",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CONVERSATION,
                    ModelCapability.FUNCTION_CALLING
                ],
                max_tokens=4096,
                cost_per_token=0.000002,
                context_window=4096,
                supports_streaming=True,
                supports_functions=True
            ))
            
            # Anthropic models
            self.add_model(AIModel(
                id="claude-3-opus",
                name="Claude 3 Opus",
                provider=AIProvider.ANTHROPIC,
                model_id="claude-3-opus-20240229",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CONVERSATION,
                    ModelCapability.ANALYSIS,
                    ModelCapability.VISION
                ],
                max_tokens=4096,
                cost_per_token=0.000015,
                context_window=200000,
                supports_streaming=True,
                supports_functions=False
            ))
            
            self.add_model(AIModel(
                id="claude-3-sonnet",
                name="Claude 3 Sonnet",
                provider=AIProvider.ANTHROPIC,
                model_id="claude-3-sonnet-20240229",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CONVERSATION,
                    ModelCapability.ANALYSIS,
                    ModelCapability.VISION
                ],
                max_tokens=4096,
                cost_per_token=0.000003,
                context_window=200000,
                supports_streaming=True,
                supports_functions=False
            ))
            
            self.add_model(AIModel(
                id="claude-3-haiku",
                name="Claude 3 Haiku",
                provider=AIProvider.ANTHROPIC,
                model_id="claude-3-haiku-20240307",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CONVERSATION,
                    ModelCapability.ANALYSIS
                ],
                max_tokens=4096,
                cost_per_token=0.00000025,
                context_window=200000,
                supports_streaming=True,
                supports_functions=False
            ))
            
            # Set default model if not specified
            if not self.default_model and self.models:
                self.default_model = "gpt-3.5-turbo"
            
            self.logger.info(f"Initialized {len(self.models)} AI models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def _load_ai_data(self) -> None:
        """Load existing AI data"""
        try:
            # Load performance stats
            stats_file = self.ai_dir / "performance_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.performance_stats.update(json.load(f))
            
            # Load custom models
            models_file = self.ai_dir / "custom_models.json"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                
                for model_data in models_data:
                    model = AIModel(
                        id=model_data['id'],
                        name=model_data['name'],
                        provider=AIProvider(model_data['provider']),
                        model_id=model_data['model_id'],
                        capabilities=[ModelCapability(cap) for cap in model_data['capabilities']],
                        max_tokens=model_data.get('max_tokens', 4096),
                        cost_per_token=model_data.get('cost_per_token', 0.0),
                        rate_limit=model_data.get('rate_limit', 60),
                        context_window=model_data.get('context_window', 4096),
                        supports_streaming=model_data.get('supports_streaming', False),
                        supports_functions=model_data.get('supports_functions', False),
                        temperature_range=tuple(model_data.get('temperature_range', [0.0, 2.0])),
                        metadata=model_data.get('metadata', {})
                    )
                    self.models[model.id] = model
            
            self.logger.info("AI data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading AI data: {e}")
    
    async def save_ai_data(self) -> None:
        """Save AI data to disk"""
        try:
            # Save performance stats
            stats_file = self.ai_dir / "performance_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.performance_stats, f, indent=2)
            
            # Save custom models (exclude built-in models)
            custom_models = [
                model.to_dict() for model in self.models.values()
                if model.metadata.get('custom', False)
            ]
            
            if custom_models:
                models_file = self.ai_dir / "custom_models.json"
                with open(models_file, 'w') as f:
                    json.dump(custom_models, f, indent=2)
            
            self.logger.info("AI data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving AI data: {e}")
    
    def _start_request_processor(self) -> None:
        """Start the request processor"""
        def process_requests():
            """Process queued requests"""
            while True:
                try:
                    # Process requests by priority
                    for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH, 
                                   RequestPriority.NORMAL, RequestPriority.LOW]:
                        if self.request_queue[priority]:
                            request = self.request_queue[priority].popleft()
                            
                            # Check if we can process this request (rate limiting)
                            if self._can_process_request(request):
                                # Process request asynchronously
                                future = self.executor.submit(self._process_request_sync, request)
                            else:
                                # Put request back in queue
                                self.request_queue[priority].appendleft(request)
                    
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                    
                except Exception as e:
                    self.logger.error(f"Error in request processor: {e}")
                    time.sleep(1)
        
        # Start processor in background thread
        processor_thread = threading.Thread(target=process_requests, daemon=True)
        processor_thread.start()
    
    def _can_process_request(self, request: AIRequest) -> bool:
        """Check if request can be processed (rate limiting)"""
        try:
            model_key = f"{request.provider.value}:{request.model_id}"
            
            if model_key not in self.rate_limiters:
                self.rate_limiters[model_key] = {
                    'requests': deque(),
                    'limit': self.models.get(request.model_id, {}).rate_limit if request.model_id in self.models else 60
                }
            
            limiter = self.rate_limiters[model_key]
            current_time = time.time()
            
            # Remove old requests (older than 1 minute)
            while limiter['requests'] and current_time - limiter['requests'][0] > 60:
                limiter['requests'].popleft()
            
            # Check if we can make another request
            if len(limiter['requests']) < limiter['limit']:
                limiter['requests'].append(current_time)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            return True  # Allow request if check fails
    
    def _process_request_sync(self, request: AIRequest) -> None:
        """Process request synchronously (for thread pool)"""
        try:
            # Run async processing in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_request(request))
            loop.close()
        except Exception as e:
            self.logger.error(f"Error processing request {request.id}: {e}")
    
    async def _process_request(self, request: AIRequest) -> AIResponse:
        """Process AI request"""
        try:
            start_time = time.time()
            
            # Check cache first
            if self.enable_caching:
                cached_response = self._get_cached_response(request)
                if cached_response:
                    self.performance_stats['cache_hits'] += 1
                    return cached_response
            
            self.performance_stats['cache_misses'] += 1
            
            # Get provider client
            provider_client = self.providers.get(request.provider)
            if not provider_client:
                raise ValueError(f"Provider {request.provider} not available")
            
            # Make request to provider
            response_content = await provider_client.generate_response(
                model_id=request.model_id,
                prompt=request.prompt,
                parameters=request.parameters,
                timeout=request.timeout
            )
            
            # Create response
            response = AIResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                provider=request.provider,
                model_id=request.model_id,
                content=response_content.get('content', ''),
                status=ResponseStatus.SUCCESS,
                tokens_used=response_content.get('tokens_used', 0),
                cost=response_content.get('cost', 0.0),
                response_time=time.time() - start_time,
                metadata=response_content.get('metadata', {})
            )
            
            # Cache response
            if self.enable_caching:
                self._cache_response(request, response)
            
            # Update statistics
            self._update_stats(request, response)
            
            # Add to history
            self.request_history.append(request)
            
            # Remove from active requests
            if request.id in self.active_requests:
                del self.active_requests[request.id]
            
            return response
            
        except Exception as e:
            # Create error response
            response = AIResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                provider=request.provider,
                model_id=request.model_id,
                content="",
                status=ResponseStatus.ERROR,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
            
            self._update_stats(request, response)
            
            # Remove from active requests
            if request.id in self.active_requests:
                del self.active_requests[request.id]
            
            self.logger.error(f"Error processing request {request.id}: {e}")
            return response
    
    def _get_cached_response(self, request: AIRequest) -> Optional[AIResponse]:
        """Get cached response if available"""
        try:
            # Create cache key
            cache_key = f"{request.provider.value}:{request.model_id}:{hash(request.prompt)}"
            
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                
                # Check if cache is still valid
                cache_age = (datetime.now() - cached_response.timestamp).total_seconds()
                if cache_age < self.cache_ttl:
                    return cached_response
                else:
                    # Remove expired cache entry
                    del self.response_cache[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached response: {e}")
            return None
    
    def _cache_response(self, request: AIRequest, response: AIResponse) -> None:
        """Cache response"""
        try:
            cache_key = f"{request.provider.value}:{request.model_id}:{hash(request.prompt)}"
            self.response_cache[cache_key] = response
            
            # Limit cache size
            if len(self.response_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.response_cache.keys(),
                    key=lambda k: self.response_cache[k].timestamp
                )[:100]
                
                for key in oldest_keys:
                    del self.response_cache[key]
            
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
    
    def _update_stats(self, request: AIRequest, response: AIResponse) -> None:
        """Update performance statistics"""
        try:
            # Update global stats
            self.performance_stats['total_requests'] += 1
            
            if response.status == ResponseStatus.SUCCESS:
                self.performance_stats['successful_requests'] += 1
            else:
                self.performance_stats['failed_requests'] += 1
            
            self.performance_stats['total_tokens'] += response.tokens_used
            self.performance_stats['total_cost'] += response.cost
            
            # Update average response time
            total_requests = self.performance_stats['total_requests']
            current_avg = self.performance_stats['average_response_time']
            self.performance_stats['average_response_time'] = (
                (current_avg * (total_requests - 1) + response.response_time) / total_requests
            )
            
            # Update provider stats
            provider_key = request.provider.value
            if provider_key not in self.performance_stats['provider_stats']:
                self.performance_stats['provider_stats'][provider_key] = {
                    'requests': 0,
                    'successful': 0,
                    'failed': 0,
                    'tokens': 0,
                    'cost': 0.0,
                    'avg_response_time': 0.0
                }
            
            provider_stats = self.performance_stats['provider_stats'][provider_key]
            provider_stats['requests'] += 1
            provider_stats['tokens'] += response.tokens_used
            provider_stats['cost'] += response.cost
            
            if response.status == ResponseStatus.SUCCESS:
                provider_stats['successful'] += 1
            else:
                provider_stats['failed'] += 1
            
            # Update model stats
            model_key = request.model_id
            if model_key not in self.performance_stats['model_stats']:
                self.performance_stats['model_stats'][model_key] = {
                    'requests': 0,
                    'successful': 0,
                    'failed': 0,
                    'tokens': 0,
                    'cost': 0.0,
                    'avg_response_time': 0.0
                }
            
            model_stats = self.performance_stats['model_stats'][model_key]
            model_stats['requests'] += 1
            model_stats['tokens'] += response.tokens_used
            model_stats['cost'] += response.cost
            
            if response.status == ResponseStatus.SUCCESS:
                model_stats['successful'] += 1
            else:
                model_stats['failed'] += 1
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            if today not in self.performance_stats['daily_stats']:
                self.performance_stats['daily_stats'][today] = {
                    'requests': 0,
                    'tokens': 0,
                    'cost': 0.0
                }
            
            daily_stats = self.performance_stats['daily_stats'][today]
            daily_stats['requests'] += 1
            daily_stats['tokens'] += response.tokens_used
            daily_stats['cost'] += response.cost
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")
    
    def add_model(self, model: AIModel) -> None:
        """Add AI model"""
        self.models[model.id] = model
        self.logger.info(f"Added AI model: {model.name}")
    
    def remove_model(self, model_id: str) -> None:
        """Remove AI model"""
        if model_id in self.models:
            del self.models[model_id]
            self.logger.info(f"Removed AI model: {model_id}")
        else:
            raise ValueError(f"Model {model_id} not found")
    
    def register_provider(self, provider: AIProvider, client: Any) -> None:
        """Register AI provider client"""
        self.providers[provider] = client
        self.logger.info(f"Registered AI provider: {provider.value}")
    
    async def generate_response(self, prompt: str,
                               model_id: Optional[str] = None,
                               provider: Optional[AIProvider] = None,
                               parameters: Optional[Dict[str, Any]] = None,
                               priority: RequestPriority = RequestPriority.NORMAL,
                               timeout: int = 30,
                               context: Optional[ConversationContext] = None) -> AIResponse:
        """Generate AI response"""
        try:
            # Select model
            if not model_id:
                if provider:
                    # Find best model for provider
                    provider_models = [m for m in self.models.values() if m.provider == provider]
                    if provider_models:
                        model_id = max(provider_models, key=lambda m: m.max_tokens).id
                    else:
                        raise ValueError(f"No models available for provider {provider}")
                else:
                    model_id = self.default_model
            
            if not model_id or model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            # Create request
            request = AIRequest(
                id=str(uuid.uuid4()),
                provider=model.provider,
                model_id=model_id,
                prompt=prompt,
                parameters=parameters or {},
                priority=priority,
                timeout=timeout,
                context={'conversation_context': context.to_dict() if context else None}
            )
            
            # Add to active requests
            self.active_requests[request.id] = request
            
            # Queue request
            self.request_queue[priority].append(request)
            
            # Wait for processing (simplified - in real implementation would use proper async handling)
            max_wait = timeout + 5
            wait_time = 0
            
            while request.id in self.active_requests and wait_time < max_wait:
                await asyncio.sleep(0.1)
                wait_time += 0.1
            
            # Find response in history (simplified)
            for req in reversed(list(self.request_history)):
                if req.id == request.id:
                    # Find corresponding response (would be better implemented with proper tracking)
                    break
            
            # For now, return a mock response (would be replaced with actual response tracking)
            return AIResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                provider=model.provider,
                model_id=model_id,
                content="Response generated successfully",
                status=ResponseStatus.SUCCESS,
                tokens_used=100,
                cost=0.01,
                response_time=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_streaming_response(self, prompt: str,
                                        model_id: Optional[str] = None,
                                        parameters: Optional[Dict[str, Any]] = None,
                                        context: Optional[ConversationContext] = None) -> AsyncGenerator[str, None]:
        """Generate streaming AI response"""
        try:
            # Select model
            if not model_id:
                model_id = self.default_model
            
            if not model_id or model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            if not model.supports_streaming:
                raise ValueError(f"Model {model_id} does not support streaming")
            
            # Get provider client
            provider_client = self.providers.get(model.provider)
            if not provider_client:
                raise ValueError(f"Provider {model.provider} not available")
            
            # Generate streaming response
            async for chunk in provider_client.generate_streaming_response(
                model_id=model_id,
                prompt=prompt,
                parameters=parameters or {},
                context=context
            ):
                yield chunk
            
        except Exception as e:
            self.logger.error(f"Error generating streaming response: {e}")
            raise
    
    def get_models(self, provider: Optional[AIProvider] = None,
                   capability: Optional[ModelCapability] = None) -> List[AIModel]:
        """Get available models"""
        models = list(self.models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        return sorted(models, key=lambda m: m.name)
    
    def get_providers(self) -> List[AIProvider]:
        """Get registered providers"""
        return list(self.providers.keys())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    def get_request_history(self, limit: int = 100) -> List[AIRequest]:
        """Get request history"""
        return list(self.request_history)[-limit:]
    
    def clear_cache(self) -> None:
        """Clear response cache"""
        self.response_cache.clear()
        self.logger.info("Response cache cleared")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        
        # Update specific settings
        if 'default_model' in new_config:
            self.default_model = new_config['default_model']
        
        if 'enable_caching' in new_config:
            self.enable_caching = new_config['enable_caching']
        
        if 'cache_ttl' in new_config:
            self.cache_ttl = new_config['cache_ttl']
        
        if 'max_concurrent_requests' in new_config:
            self.max_concurrent_requests = new_config['max_concurrent_requests']
        
        self.logger.info("AI Manager configuration updated")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Save AI data
            await self.save_ai_data()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear caches
            self.response_cache.clear()
            
            # Clear old daily stats
            cutoff_date = (datetime.now() - timedelta(days=30)).date()
            if 'daily_stats' in self.performance_stats:
                self.performance_stats['daily_stats'] = {
                    date: stats for date, stats in self.performance_stats['daily_stats'].items()
                    if datetime.fromisoformat(date).date() >= cutoff_date
                }
            
            self.logger.info("AI Manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
"""
AI Integration Package for Computer Assistant

This package provides integration with external AI services including:
- OpenAI GPT models (ChatGPT, GPT-4, etc.)
- Anthropic Claude models
- AI model management and switching
- Conversation handling and context management
- Response processing and optimization
"""

from .ai_manager import AIManager, AIProvider, AIModel, AIResponse, ConversationContext
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .conversation_manager import ConversationManager, Conversation, Message, MessageRole

__version__ = "1.0.0"
__author__ = "Computer Assistant Team"

__all__ = [
    # Core AI Management
    'AIManager',
    'AIProvider',
    'AIModel', 
    'AIResponse',
    'ConversationContext',
    
    # AI Clients
    'OpenAIClient',
    'AnthropicClient',
    
    # Conversation Management
    'ConversationManager',
    'Conversation',
    'Message',
    'MessageRole'
]
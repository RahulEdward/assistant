"""
Assistant Manager - Core orchestrator for the Desktop Assistant
Coordinates all components including voice processing, NLP, system automation, and AI integration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import threading
import queue

from .config_manager import ConfigManager
from ..audio.voice_processor import VoiceProcessor
from ..nlp.command_processor import CommandProcessor
from ..automation.system_controller import SystemController
from ..ocr.screen_reader import ScreenReader
from ..ai.external_ai_manager import ExternalAIManager
from ..modules.financial_analyzer import FinancialAnalyzer
from ..modules.code_assistant import CodeAssistant
from ..modules.test_automation import TestAutomation
from ..ml.performance_optimizer import PerformanceOptimizer
from ..utils.response_formatter import ResponseFormatter


class AssistantManager:
    """Main coordinator for all assistant components"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.voice_processor: Optional[VoiceProcessor] = None
        self.command_processor: Optional[CommandProcessor] = None
        self.system_controller: Optional[SystemController] = None
        self.screen_reader: Optional[ScreenReader] = None
        self.external_ai: Optional[ExternalAIManager] = None
        
        # Specialized modules
        self.financial_analyzer: Optional[FinancialAnalyzer] = None
        self.code_assistant: Optional[CodeAssistant] = None
        self.test_automation: Optional[TestAutomation] = None
        
        # ML and optimization
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.response_formatter: Optional[ResponseFormatter] = None
        
        # State management
        self.is_listening = False
        self.is_processing = False
        self.command_queue = queue.Queue()
        self.response_callbacks: List[Callable] = []
        
        # Performance metrics
        self.metrics = {
            'commands_processed': 0,
            'successful_executions': 0,
            'average_response_time': 0.0,
            'accuracy_rate': 0.0
        }
    
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing Assistant Manager...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize specialized modules
            await self._initialize_specialized_modules()
            
            # Initialize ML components
            await self._initialize_ml_components()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Assistant Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Assistant Manager: {e}")
            return False
    
    async def _initialize_core_components(self):
        """Initialize core system components"""
        # Voice processing
        self.voice_processor = VoiceProcessor(self.config.voice)
        await self.voice_processor.initialize()
        
        # Natural language processing
        self.command_processor = CommandProcessor(self.config.nlp)
        await self.command_processor.initialize()
        
        # System automation
        self.system_controller = SystemController(self.config.system)
        await self.system_controller.initialize()
        
        # Screen reading and OCR
        self.screen_reader = ScreenReader(self.config.system)
        await self.screen_reader.initialize()
        
        # External AI integration
        self.external_ai = ExternalAIManager(self.config.ai_integration)
        await self.external_ai.initialize()
        
        # Response formatting
        self.response_formatter = ResponseFormatter()
    
    async def _initialize_specialized_modules(self):
        """Initialize specialized functionality modules"""
        # Financial analysis
        self.financial_analyzer = FinancialAnalyzer()
        await self.financial_analyzer.initialize()
        
        # Code assistance
        self.code_assistant = CodeAssistant()
        await self.code_assistant.initialize()
        
        # Test automation
        self.test_automation = TestAutomation()
        await self.test_automation.initialize()
    
    async def _initialize_ml_components(self):
        """Initialize machine learning components"""
        self.performance_optimizer = PerformanceOptimizer()
        await self.performance_optimizer.initialize()
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        # Start command processing loop
        asyncio.create_task(self._command_processing_loop())
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
    
    async def start_listening(self):
        """Start voice input listening"""
        if not self.voice_processor:
            raise RuntimeError("Voice processor not initialized")
            
        self.is_listening = True
        await self.voice_processor.start_listening(self._on_voice_command)
        self.logger.info("Started voice listening")
    
    async def stop_listening(self):
        """Stop voice input listening"""
        self.is_listening = False
        if self.voice_processor:
            await self.voice_processor.stop_listening()
        self.logger.info("Stopped voice listening")
    
    async def process_text_command(self, text: str) -> Dict[str, Any]:
        """Process a text command"""
        return await self._process_command(text, input_type="text")
    
    async def _on_voice_command(self, audio_data: bytes):
        """Handle voice command input"""
        try:
            # Convert speech to text
            text = await self.voice_processor.speech_to_text(audio_data)
            if text:
                await self._process_command(text, input_type="voice")
        except Exception as e:
            self.logger.error(f"Error processing voice command: {e}")
    
    async def _process_command(self, text: str, input_type: str = "text") -> Dict[str, Any]:
        """Process a command (text or voice)"""
        start_time = datetime.now()
        
        try:
            self.is_processing = True
            self.metrics['commands_processed'] += 1
            
            # Parse and understand the command
            command_data = await self.command_processor.process_command(text)
            
            # Execute the command
            result = await self._execute_command(command_data)
            
            # Format response
            response = await self.response_formatter.format_response(
                result, input_type, command_data
            )
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._update_metrics(execution_time, result.get('success', False))
            
            # Send response to callbacks
            await self._send_response(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing command '{text}': {e}")
            error_response = {
                'success': False,
                'error': str(e),
                'text': f"Sorry, I encountered an error: {e}",
                'timestamp': datetime.now().isoformat()
            }
            await self._send_response(error_response)
            return error_response
            
        finally:
            self.is_processing = False
    
    async def _execute_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a parsed command"""
        command_type = command_data.get('type', 'unknown')
        intent = command_data.get('intent', '')
        parameters = command_data.get('parameters', {})
        
        try:
            # Route to appropriate handler
            if command_type == 'system':
                return await self._handle_system_command(intent, parameters)
            elif command_type == 'application':
                return await self._handle_application_command(intent, parameters)
            elif command_type == 'financial':
                return await self._handle_financial_command(intent, parameters)
            elif command_type == 'code':
                return await self._handle_code_command(intent, parameters)
            elif command_type == 'test':
                return await self._handle_test_command(intent, parameters)
            elif command_type == 'ai':
                return await self._handle_ai_command(intent, parameters)
            elif command_type == 'screen':
                return await self._handle_screen_command(intent, parameters)
            else:
                # Fallback to external AI
                return await self.external_ai.process_query(
                    command_data.get('original_text', ''), parameters
                )
                
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _handle_system_command(self, intent: str, parameters: Dict) -> Dict[str, Any]:
        """Handle system-level commands"""
        return await self.system_controller.execute_command(intent, parameters)
    
    async def _handle_application_command(self, intent: str, parameters: Dict) -> Dict[str, Any]:
        """Handle application control commands"""
        return await self.system_controller.control_application(intent, parameters)
    
    async def _handle_financial_command(self, intent: str, parameters: Dict) -> Dict[str, Any]:
        """Handle financial analysis commands"""
        return await self.financial_analyzer.process_request(intent, parameters)
    
    async def _handle_code_command(self, intent: str, parameters: Dict) -> Dict[str, Any]:
        """Handle code generation/debugging commands"""
        return await self.code_assistant.process_request(intent, parameters)
    
    async def _handle_test_command(self, intent: str, parameters: Dict) -> Dict[str, Any]:
        """Handle test automation commands"""
        return await self.test_automation.process_request(intent, parameters)
    
    async def _handle_ai_command(self, intent: str, parameters: Dict) -> Dict[str, Any]:
        """Handle external AI integration commands"""
        return await self.external_ai.process_query(intent, parameters)
    
    async def _handle_screen_command(self, intent: str, parameters: Dict) -> Dict[str, Any]:
        """Handle screen reading and OCR commands"""
        return await self.screen_reader.process_request(intent, parameters)
    
    async def _command_processing_loop(self):
        """Background loop for processing queued commands"""
        while True:
            try:
                if not self.command_queue.empty():
                    command_data = self.command_queue.get_nowait()
                    await self._process_command(**command_data)
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in command processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _performance_monitoring_loop(self):
        """Background loop for performance monitoring and optimization"""
        while True:
            try:
                # Update performance metrics
                await self.performance_optimizer.analyze_performance(self.metrics)
                
                # Apply optimizations if needed
                optimizations = await self.performance_optimizer.get_optimizations()
                if optimizations:
                    await self._apply_optimizations(optimizations)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _apply_optimizations(self, optimizations: List[Dict]):
        """Apply performance optimizations"""
        for optimization in optimizations:
            try:
                component = optimization.get('component')
                action = optimization.get('action')
                parameters = optimization.get('parameters', {})
                
                if component == 'voice_processor' and self.voice_processor:
                    await self.voice_processor.apply_optimization(action, parameters)
                elif component == 'command_processor' and self.command_processor:
                    await self.command_processor.apply_optimization(action, parameters)
                # Add more component optimizations as needed
                    
            except Exception as e:
                self.logger.error(f"Error applying optimization: {e}")
    
    async def _update_metrics(self, execution_time: float, success: bool):
        """Update performance metrics"""
        if success:
            self.metrics['successful_executions'] += 1
        
        # Update average response time
        total_commands = self.metrics['commands_processed']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = (
            (current_avg * (total_commands - 1) + execution_time) / total_commands
        )
        
        # Update accuracy rate
        self.metrics['accuracy_rate'] = (
            self.metrics['successful_executions'] / total_commands
        )
    
    async def _send_response(self, response: Dict[str, Any]):
        """Send response to registered callbacks"""
        for callback in self.response_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(response)
                else:
                    callback(response)
            except Exception as e:
                self.logger.error(f"Error in response callback: {e}")
    
    def add_response_callback(self, callback: Callable):
        """Add a response callback"""
        self.response_callbacks.append(callback)
    
    def remove_response_callback(self, callback: Callable):
        """Remove a response callback"""
        if callback in self.response_callbacks:
            self.response_callbacks.remove(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    async def cleanup(self):
        """Cleanup all components"""
        self.logger.info("Cleaning up Assistant Manager...")
        
        # Stop listening
        await self.stop_listening()
        
        # Cleanup components
        components = [
            self.voice_processor,
            self.command_processor,
            self.system_controller,
            self.screen_reader,
            self.external_ai,
            self.financial_analyzer,
            self.code_assistant,
            self.test_automation,
            self.performance_optimizer
        ]
        
        for component in components:
            if component and hasattr(component, 'cleanup'):
                try:
                    await component.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up component: {e}")
        
        self.logger.info("Assistant Manager cleanup completed")
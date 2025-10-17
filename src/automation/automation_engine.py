"""
Automation Engine
Main orchestrator for system-level automation with human-like execution speed and machine precision.
Coordinates system control, application management, and browser automation.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .system_controller import SystemController
from .application_controller import ApplicationController
from .browser_controller import BrowserController
from .window_manager import WindowManager


class ExecutionStatus(Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionPriority(Enum):
    """Execution priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class ExecutionResult:
    """Execution result"""
    task_id: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AutomationTask:
    """Automation task definition"""
    id: str
    command_type: str
    action: str
    parameters: Dict[str, Any]
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    timeout: float = 30.0
    retry_count: int = 3
    dependencies: List[str] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class AdvancedAutomationEngine:
    """Advanced automation engine with performance optimization"""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.system_controller = SystemController()
        self.app_controller = ApplicationController()
        self.browser_controller = BrowserController()
        self.window_manager = WindowManager()
        
        # Execution management
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue: asyncio.Queue = None
        self.active_tasks: Dict[str, AutomationTask] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.execution_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'fastest_execution': float('inf'),
            'slowest_execution': 0.0
        }
        
        # Optimization settings
        self.performance_targets = {
            'max_response_time': 1.0,  # Sub-second target
            'min_accuracy': 0.99,      # 99% accuracy target
            'max_concurrent_tasks': max_workers
        }
        
        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        
        # Configuration
        self.config_dir = Path("config/automation")
        self.config_path = self.config_dir / "automation_config.json"
        self.stats_path = self.config_dir / "execution_stats.json"
        
        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()
    
    async def initialize(self):
        """Initialize automation engine"""
        try:
            self.logger.info("Initializing Automation Engine...")
            
            # Create config directory
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize task queue
            self.task_queue = asyncio.Queue()
            
            # Initialize components
            await self.system_controller.initialize()
            await self.app_controller.initialize()
            await self.browser_controller.initialize()
            await self.window_manager.initialize()
            
            # Register task handlers
            await self._register_task_handlers()
            
            # Load configuration and stats
            await self._load_configuration()
            await self._load_execution_stats()
            
            # Start task processor
            self.is_running = True
            asyncio.create_task(self._process_task_queue())
            
            self.logger.info("Automation Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Automation Engine initialization error: {e}")
            return False
    
    async def _register_task_handlers(self):
        """Register task handlers for different command types"""
        try:
            self.task_handlers = {
                # System control handlers
                'system_control': {
                    'open_application': self._handle_open_application,
                    'close_application': self._handle_close_application,
                    'minimize_window': self._handle_minimize_window,
                    'maximize_window': self._handle_maximize_window,
                    'switch_window': self._handle_switch_window,
                    'get_system_info': self._handle_get_system_info,
                    'set_system_setting': self._handle_set_system_setting
                },
                
                # File management handlers
                'file_management': {
                    'create_file': self._handle_create_file,
                    'delete_file': self._handle_delete_file,
                    'copy_file': self._handle_copy_file,
                    'move_file': self._handle_move_file,
                    'search_files': self._handle_search_files,
                    'open_file': self._handle_open_file,
                    'rename_file': self._handle_rename_file
                },
                
                # Web automation handlers
                'web_automation': {
                    'open_url': self._handle_open_url,
                    'web_search': self._handle_web_search,
                    'click_element': self._handle_click_element,
                    'fill_form': self._handle_fill_form,
                    'extract_data': self._handle_extract_data,
                    'navigate_page': self._handle_navigate_page,
                    'take_screenshot': self._handle_take_screenshot
                },
                
                # Text processing handlers
                'text_processing': {
                    'extract_text': self._handle_extract_text,
                    'translate_text': self._handle_translate_text,
                    'summarize_text': self._handle_summarize_text,
                    'format_text': self._handle_format_text
                },
                
                # Information query handlers
                'information_query': {
                    'get_weather': self._handle_get_weather,
                    'get_time': self._handle_get_time,
                    'get_calendar': self._handle_get_calendar,
                    'get_news': self._handle_get_news
                }
            }
            
            self.logger.info(f"Registered {sum(len(handlers) for handlers in self.task_handlers.values())} task handlers")
            
        except Exception as e:
            self.logger.error(f"Task handler registration error: {e}")
    
    async def execute_task(self, task: AutomationTask) -> ExecutionResult:
        """Execute automation task with performance optimization"""
        try:
            start_time = datetime.now()
            task_start = time.time()
            
            self.logger.info(f"Executing task: {task.id} ({task.command_type}.{task.action})")
            
            # Check dependencies
            if not await self._check_dependencies(task):
                return ExecutionResult(
                    task_id=task.id,
                    status=ExecutionStatus.FAILED,
                    error="Dependencies not met",
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            # Add to active tasks
            self.active_tasks[task.id] = task
            
            # Get task handler
            handler = self._get_task_handler(task.command_type, task.action)
            if not handler:
                return ExecutionResult(
                    task_id=task.id,
                    status=ExecutionStatus.FAILED,
                    error=f"No handler found for {task.command_type}.{task.action}",
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            # Execute with timeout and retry logic
            result = await self._execute_with_retry(handler, task)
            
            # Calculate execution time
            execution_time = time.time() - task_start
            
            # Update result
            result.execution_time = execution_time
            result.start_time = start_time
            result.end_time = datetime.now()
            
            # Update statistics
            await self._update_execution_stats(result)
            
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            # Store completed task
            self.completed_tasks[task.id] = result
            
            # Execute callback if provided
            if task.callback:
                try:
                    await task.callback(result)
                except Exception as e:
                    self.logger.warning(f"Task callback error: {e}")
            
            self.logger.info(f"Task completed: {task.id} ({result.status.value}) in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
    
    async def _execute_with_retry(self, handler: Callable, task: AutomationTask) -> ExecutionResult:
        """Execute task with retry logic"""
        last_error = None
        
        for attempt in range(task.retry_count + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(task.parameters),
                    timeout=task.timeout
                )
                
                return ExecutionResult(
                    task_id=task.id,
                    status=ExecutionStatus.COMPLETED,
                    result=result
                )
                
            except asyncio.TimeoutError:
                last_error = f"Task timeout after {task.timeout}s"
                self.logger.warning(f"Task {task.id} timeout (attempt {attempt + 1})")
                
                if attempt < task.retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Task {task.id} error (attempt {attempt + 1}): {e}")
                
                if attempt < task.retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        return ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.FAILED,
            error=last_error
        )
    
    def _get_task_handler(self, command_type: str, action: str) -> Optional[Callable]:
        """Get task handler for command type and action"""
        try:
            if command_type in self.task_handlers:
                return self.task_handlers[command_type].get(action)
            return None
            
        except Exception as e:
            self.logger.error(f"Task handler retrieval error: {e}")
            return None
    
    async def _check_dependencies(self, task: AutomationTask) -> bool:
        """Check if task dependencies are satisfied"""
        try:
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    return False
                
                dep_result = self.completed_tasks[dep_id]
                if dep_result.status != ExecutionStatus.COMPLETED:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dependency check error: {e}")
            return False
    
    async def _process_task_queue(self):
        """Process task queue continuously"""
        try:
            while self.is_running:
                try:
                    # Get task from queue with timeout
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    
                    # Execute task asynchronously
                    asyncio.create_task(self.execute_task(task))
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Task queue processing error: {e}")
                    await asyncio.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"Task queue processor error: {e}")
    
    async def queue_task(self, task: AutomationTask):
        """Add task to execution queue"""
        try:
            await self.task_queue.put(task)
            self.logger.info(f"Task queued: {task.id}")
            
        except Exception as e:
            self.logger.error(f"Task queueing error: {e}")
    
    async def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        try:
            self.execution_stats['total_tasks'] += 1
            
            if result.status == ExecutionStatus.COMPLETED:
                self.execution_stats['successful_tasks'] += 1
            else:
                self.execution_stats['failed_tasks'] += 1
            
            # Update timing stats
            if result.execution_time > 0:
                current_avg = self.execution_stats['average_execution_time']
                total_tasks = self.execution_stats['total_tasks']
                
                # Calculate new average
                new_avg = ((current_avg * (total_tasks - 1)) + result.execution_time) / total_tasks
                self.execution_stats['average_execution_time'] = new_avg
                
                # Update min/max
                self.execution_stats['fastest_execution'] = min(
                    self.execution_stats['fastest_execution'],
                    result.execution_time
                )
                self.execution_stats['slowest_execution'] = max(
                    self.execution_stats['slowest_execution'],
                    result.execution_time
                )
            
            # Save stats periodically
            if self.execution_stats['total_tasks'] % 10 == 0:
                await self._save_execution_stats()
            
        except Exception as e:
            self.logger.error(f"Stats update error: {e}")
    
    async def _load_configuration(self):
        """Load automation configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.performance_targets.update(config.get('performance_targets', {}))
                    self.max_workers = config.get('max_workers', self.max_workers)
            
            self.logger.info("Configuration loaded")
            
        except Exception as e:
            self.logger.error(f"Configuration loading error: {e}")
    
    async def _load_execution_stats(self):
        """Load execution statistics"""
        try:
            if self.stats_path.exists():
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    saved_stats = json.load(f)
                    self.execution_stats.update(saved_stats)
            
            self.logger.info("Execution stats loaded")
            
        except Exception as e:
            self.logger.error(f"Stats loading error: {e}")
    
    async def _save_execution_stats(self):
        """Save execution statistics"""
        try:
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.execution_stats, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Stats saving error: {e}")
    
    # Task handler implementations
    async def _handle_open_application(self, parameters: Dict[str, Any]) -> Any:
        """Handle open application task"""
        application = parameters.get('application')
        arguments = parameters.get('arguments', [])
        wait_for_load = parameters.get('wait_for_load', True)
        
        return await self.app_controller.open_application(application, arguments, wait_for_load)
    
    async def _handle_close_application(self, parameters: Dict[str, Any]) -> Any:
        """Handle close application task"""
        application = parameters.get('application')
        force_close = parameters.get('force_close', False)
        save_before_close = parameters.get('save_before_close', True)
        
        return await self.app_controller.close_application(application, force_close, save_before_close)
    
    async def _handle_minimize_window(self, parameters: Dict[str, Any]) -> Any:
        """Handle minimize window task"""
        window_title = parameters.get('window_title')
        application = parameters.get('application')
        
        return await self.window_manager.minimize_window(window_title, application)
    
    async def _handle_maximize_window(self, parameters: Dict[str, Any]) -> Any:
        """Handle maximize window task"""
        window_title = parameters.get('window_title')
        application = parameters.get('application')
        
        return await self.window_manager.maximize_window(window_title, application)
    
    async def _handle_switch_window(self, parameters: Dict[str, Any]) -> Any:
        """Handle switch window task"""
        window_title = parameters.get('window_title')
        application = parameters.get('application')
        
        return await self.window_manager.switch_to_window(window_title, application)
    
    async def _handle_get_system_info(self, parameters: Dict[str, Any]) -> Any:
        """Handle get system info task"""
        info_type = parameters.get('info_type', 'general')
        
        return await self.system_controller.get_system_info(info_type)
    
    async def _handle_set_system_setting(self, parameters: Dict[str, Any]) -> Any:
        """Handle set system setting task"""
        setting_name = parameters.get('setting_name')
        setting_value = parameters.get('setting_value')
        
        return await self.system_controller.set_system_setting(setting_name, setting_value)
    
    async def _handle_create_file(self, parameters: Dict[str, Any]) -> Any:
        """Handle create file task"""
        file_path = parameters.get('file_path')
        content = parameters.get('content', '')
        overwrite = parameters.get('overwrite', False)
        
        return await self.system_controller.create_file(file_path, content, overwrite)
    
    async def _handle_delete_file(self, parameters: Dict[str, Any]) -> Any:
        """Handle delete file task"""
        file_path = parameters.get('file_path')
        permanent = parameters.get('permanent', False)
        
        return await self.system_controller.delete_file(file_path, permanent)
    
    async def _handle_copy_file(self, parameters: Dict[str, Any]) -> Any:
        """Handle copy file task"""
        source_path = parameters.get('source_path')
        destination_path = parameters.get('destination_path')
        overwrite = parameters.get('overwrite', False)
        
        return await self.system_controller.copy_file(source_path, destination_path, overwrite)
    
    async def _handle_move_file(self, parameters: Dict[str, Any]) -> Any:
        """Handle move file task"""
        source_path = parameters.get('source_path')
        destination_path = parameters.get('destination_path')
        overwrite = parameters.get('overwrite', False)
        
        return await self.system_controller.move_file(source_path, destination_path, overwrite)
    
    async def _handle_search_files(self, parameters: Dict[str, Any]) -> Any:
        """Handle search files task"""
        search_path = parameters.get('search_path', '.')
        pattern = parameters.get('pattern')
        file_type = parameters.get('file_type')
        recursive = parameters.get('recursive', True)
        
        return await self.system_controller.search_files(search_path, pattern, file_type, recursive)
    
    async def _handle_open_file(self, parameters: Dict[str, Any]) -> Any:
        """Handle open file task"""
        file_path = parameters.get('file_path')
        application = parameters.get('application')
        
        return await self.app_controller.open_file(file_path, application)
    
    async def _handle_rename_file(self, parameters: Dict[str, Any]) -> Any:
        """Handle rename file task"""
        old_path = parameters.get('old_path')
        new_path = parameters.get('new_path')
        
        return await self.system_controller.rename_file(old_path, new_path)
    
    async def _handle_open_url(self, parameters: Dict[str, Any]) -> Any:
        """Handle open URL task"""
        url = parameters.get('url')
        browser = parameters.get('browser')
        new_tab = parameters.get('new_tab', True)
        
        return await self.browser_controller.open_url(url, browser, new_tab)
    
    async def _handle_web_search(self, parameters: Dict[str, Any]) -> Any:
        """Handle web search task"""
        query = parameters.get('query')
        search_engine = parameters.get('search_engine', 'google')
        results_count = parameters.get('results_count', 10)
        
        return await self.browser_controller.web_search(query, search_engine, results_count)
    
    async def _handle_click_element(self, parameters: Dict[str, Any]) -> Any:
        """Handle click element task"""
        selector = parameters.get('selector')
        wait_for_element = parameters.get('wait_for_element', True)
        timeout = parameters.get('timeout', 5.0)
        
        return await self.browser_controller.click_element(selector, wait_for_element, timeout)
    
    async def _handle_fill_form(self, parameters: Dict[str, Any]) -> Any:
        """Handle fill form task"""
        form_data = parameters.get('form_data', {})
        submit = parameters.get('submit', False)
        
        return await self.browser_controller.fill_form(form_data, submit)
    
    async def _handle_extract_data(self, parameters: Dict[str, Any]) -> Any:
        """Handle extract data task"""
        selector = parameters.get('selector')
        data_type = parameters.get('data_type', 'text')
        
        return await self.browser_controller.extract_data(selector, data_type)
    
    async def _handle_navigate_page(self, parameters: Dict[str, Any]) -> Any:
        """Handle navigate page task"""
        direction = parameters.get('direction', 'forward')
        
        return await self.browser_controller.navigate_page(direction)
    
    async def _handle_take_screenshot(self, parameters: Dict[str, Any]) -> Any:
        """Handle take screenshot task"""
        save_path = parameters.get('save_path')
        element_selector = parameters.get('element_selector')
        
        return await self.browser_controller.take_screenshot(save_path, element_selector)
    
    async def _handle_extract_text(self, parameters: Dict[str, Any]) -> Any:
        """Handle extract text task"""
        source = parameters.get('source')
        format_type = parameters.get('format', 'plain')
        
        # This would integrate with OCR module when implemented
        return f"Text extracted from {source} in {format_type} format"
    
    async def _handle_translate_text(self, parameters: Dict[str, Any]) -> Any:
        """Handle translate text task"""
        text = parameters.get('text')
        target_language = parameters.get('target_language')
        source_language = parameters.get('source_language', 'auto')
        
        # This would integrate with translation service
        return f"Translated text from {source_language} to {target_language}"
    
    async def _handle_summarize_text(self, parameters: Dict[str, Any]) -> Any:
        """Handle summarize text task"""
        text = parameters.get('text')
        max_length = parameters.get('max_length', 200)
        style = parameters.get('style', 'concise')
        
        # This would integrate with NLP summarization
        return f"Text summarized in {style} style (max {max_length} chars)"
    
    async def _handle_format_text(self, parameters: Dict[str, Any]) -> Any:
        """Handle format text task"""
        text = parameters.get('text')
        format_type = parameters.get('format_type', 'plain')
        
        return f"Text formatted as {format_type}"
    
    async def _handle_get_weather(self, parameters: Dict[str, Any]) -> Any:
        """Handle get weather task"""
        location = parameters.get('location')
        forecast_days = parameters.get('forecast_days', 1)
        
        # This would integrate with weather API
        return f"Weather for {location} ({forecast_days} days)"
    
    async def _handle_get_time(self, parameters: Dict[str, Any]) -> Any:
        """Handle get time task"""
        timezone = parameters.get('timezone', 'local')
        format_type = parameters.get('format', '12h')
        
        current_time = datetime.now()
        if format_type == '12h':
            return current_time.strftime("%I:%M %p")
        else:
            return current_time.strftime("%H:%M")
    
    async def _handle_get_calendar(self, parameters: Dict[str, Any]) -> Any:
        """Handle get calendar task"""
        date_range = parameters.get('date_range', 'today')
        
        return f"Calendar events for {date_range}"
    
    async def _handle_get_news(self, parameters: Dict[str, Any]) -> Any:
        """Handle get news task"""
        category = parameters.get('category', 'general')
        count = parameters.get('count', 5)
        
        return f"Top {count} news articles in {category}"
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            total_tasks = self.execution_stats['total_tasks']
            successful_tasks = self.execution_stats['successful_tasks']
            
            accuracy = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
            avg_response_time = self.execution_stats['average_execution_time']
            
            return {
                'total_tasks_executed': total_tasks,
                'success_rate': accuracy,
                'average_response_time': avg_response_time,
                'fastest_execution': self.execution_stats['fastest_execution'],
                'slowest_execution': self.execution_stats['slowest_execution'],
                'active_tasks': len(self.active_tasks),
                'meets_response_target': avg_response_time <= self.performance_targets['max_response_time'],
                'meets_accuracy_target': accuracy >= self.performance_targets['min_accuracy'] * 100,
                'performance_targets': self.performance_targets
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics error: {e}")
            return {}
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific task"""
        try:
            if task_id in self.active_tasks:
                return {
                    'status': 'running',
                    'task': asdict(self.active_tasks[task_id])
                }
            elif task_id in self.completed_tasks:
                return {
                    'status': 'completed',
                    'result': asdict(self.completed_tasks[task_id])
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Task status error: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel active task"""
        try:
            if task_id in self.active_tasks:
                # Mark task as cancelled
                result = ExecutionResult(
                    task_id=task_id,
                    status=ExecutionStatus.CANCELLED,
                    end_time=datetime.now()
                )
                
                self.completed_tasks[task_id] = result
                del self.active_tasks[task_id]
                
                self.logger.info(f"Task cancelled: {task_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Task cancellation error: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup automation engine"""
        try:
            self.logger.info("Cleaning up Automation Engine...")
            
            # Stop task processing
            self.is_running = False
            self.shutdown_event.set()
            
            # Save final stats
            await self._save_execution_stats()
            
            # Cleanup components
            await self.system_controller.cleanup()
            await self.app_controller.cleanup()
            await self.browser_controller.cleanup()
            await self.window_manager.cleanup()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Automation Engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Automation Engine cleanup error: {e}")


# Alias for backward compatibility
AutomationEngine = AdvancedAutomationEngine
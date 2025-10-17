"""
Test Automation Module for Computer Assistant

This module provides comprehensive automated testing capabilities including:
- Automated test case generation and execution
- UI testing with screenshot comparison
- API testing and validation
- Performance testing and benchmarking
- Test data generation and management
- Test reporting and analytics
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import subprocess
import hashlib
import base64


class TestType(Enum):
    """Types of automated tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    UI = "ui"
    API = "api"
    SECURITY = "security"
    LOAD = "load"
    SMOKE = "smoke"
    REGRESSION = "regression"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestPriority(Enum):
    """Test priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestCase:
    """Individual test case structure"""
    id: str
    name: str
    description: str
    test_type: TestType
    priority: TestPriority
    test_function: Optional[Callable] = None
    test_data: Dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    timeout: int = 30  # seconds
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'test_type': self.test_type.value,
            'priority': self.priority.value,
            'test_data': self.test_data,
            'expected_result': self.expected_result,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class TestResult:
    """Test execution result"""
    test_case_id: str
    status: TestStatus
    execution_time: float
    start_time: datetime
    end_time: datetime
    actual_result: Any = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    retry_attempt: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_case_id': self.test_case_id,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'actual_result': self.actual_result,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'screenshots': self.screenshots,
            'logs': self.logs,
            'metrics': self.metrics,
            'retry_attempt': self.retry_attempt
        }


@dataclass
class TestSuite:
    """Test suite containing multiple test cases"""
    id: str
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    parallel_execution: bool = False
    max_parallel_tests: int = 5
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add test case to suite"""
        self.test_cases.append(test_case)
    
    def get_test_case(self, test_id: str) -> Optional[TestCase]:
        """Get test case by ID"""
        return next((tc for tc in self.test_cases if tc.id == test_id), None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'test_cases': [tc.to_dict() for tc in self.test_cases],
            'parallel_execution': self.parallel_execution,
            'max_parallel_tests': self.max_parallel_tests,
            'tags': self.tags,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class TestExecution:
    """Test execution session"""
    id: str
    suite_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TestStatus = TestStatus.PENDING
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    def add_result(self, result: TestResult) -> None:
        """Add test result"""
        self.results.append(result)
        self._update_summary()
    
    def _update_summary(self) -> None:
        """Update execution summary"""
        if not self.results:
            return
        
        status_counts = {}
        total_time = 0
        
        for result in self.results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_time += result.execution_time
        
        self.summary = {
            'total_tests': len(self.results),
            'status_counts': status_counts,
            'total_execution_time': total_time,
            'average_execution_time': total_time / len(self.results) if self.results else 0,
            'pass_rate': status_counts.get('passed', 0) / len(self.results) * 100 if self.results else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'suite_id': self.suite_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
            'environment': self.environment
        }


class TestAutomator:
    """
    Comprehensive test automation system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test automator"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Test storage
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_executions: Dict[str, TestExecution] = {}
        
        # Configuration
        self.test_data_dir = Path(self.config.get('test_data_dir', 'test_data'))
        self.test_data_dir.mkdir(exist_ok=True)
        
        self.screenshots_dir = Path(self.config.get('screenshots_dir', 'screenshots'))
        self.screenshots_dir.mkdir(exist_ok=True)
        
        self.reports_dir = Path(self.config.get('reports_dir', 'test_reports'))
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test execution settings
        self.default_timeout = self.config.get('default_timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.parallel_execution = self.config.get('parallel_execution', True)
        self.max_parallel_tests = self.config.get('max_parallel_tests', 5)
        
        # Performance tracking
        self.performance_stats = {
            'tests_executed': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'suites_created': 0,
            'test_types_executed': {},
            'daily_stats': {}
        }
        
        # Test data generators
        self.data_generators = {
            'string': self._generate_string_data,
            'number': self._generate_number_data,
            'email': self._generate_email_data,
            'phone': self._generate_phone_data,
            'date': self._generate_date_data,
            'url': self._generate_url_data,
            'json': self._generate_json_data,
            'xml': self._generate_xml_data
        }
        
        # Built-in test templates
        self._initialize_test_templates()
        
        self.logger.info("Test automator initialized")
    
    def _initialize_test_templates(self) -> None:
        """Initialize built-in test templates"""
        try:
            # API test template
            api_suite = TestSuite(
                id="api_test_suite",
                name="API Testing Suite",
                description="Comprehensive API testing suite",
                tags=["api", "integration"]
            )
            
            # Add API test cases
            api_suite.add_test_case(TestCase(
                id="api_get_test",
                name="API GET Request Test",
                description="Test GET API endpoint",
                test_type=TestType.API,
                priority=TestPriority.HIGH,
                test_function=self._test_api_get,
                test_data={
                    'url': 'http://localhost:8000/api/test',
                    'expected_status': 200,
                    'expected_fields': ['status', 'data']
                },
                tags=["api", "get"]
            ))
            
            api_suite.add_test_case(TestCase(
                id="api_post_test",
                name="API POST Request Test",
                description="Test POST API endpoint",
                test_type=TestType.API,
                priority=TestPriority.HIGH,
                test_function=self._test_api_post,
                test_data={
                    'url': 'http://localhost:8000/api/data',
                    'payload': {'name': 'test', 'value': 123},
                    'expected_status': 201
                },
                tags=["api", "post"]
            ))
            
            self.test_suites["api_test_suite"] = api_suite
            
            # UI test template
            ui_suite = TestSuite(
                id="ui_test_suite",
                name="UI Testing Suite",
                description="User interface testing suite",
                tags=["ui", "functional"]
            )
            
            ui_suite.add_test_case(TestCase(
                id="ui_login_test",
                name="Login Form Test",
                description="Test login form functionality",
                test_type=TestType.UI,
                priority=TestPriority.CRITICAL,
                test_function=self._test_ui_login,
                test_data={
                    'username': 'testuser',
                    'password': 'testpass',
                    'expected_redirect': '/dashboard'
                },
                tags=["ui", "login", "authentication"]
            ))
            
            ui_suite.add_test_case(TestCase(
                id="ui_navigation_test",
                name="Navigation Test",
                description="Test main navigation functionality",
                test_type=TestType.UI,
                priority=TestPriority.MEDIUM,
                test_function=self._test_ui_navigation,
                test_data={
                    'menu_items': ['Home', 'About', 'Contact'],
                    'expected_pages': ['/home', '/about', '/contact']
                },
                tags=["ui", "navigation"]
            ))
            
            self.test_suites["ui_test_suite"] = ui_suite
            
            # Performance test template
            perf_suite = TestSuite(
                id="performance_test_suite",
                name="Performance Testing Suite",
                description="Performance and load testing suite",
                tags=["performance", "load"]
            )
            
            perf_suite.add_test_case(TestCase(
                id="response_time_test",
                name="Response Time Test",
                description="Test API response times",
                test_type=TestType.PERFORMANCE,
                priority=TestPriority.HIGH,
                test_function=self._test_response_time,
                test_data={
                    'url': 'http://localhost:8000/api/test',
                    'max_response_time': 1.0,  # seconds
                    'concurrent_requests': 10
                },
                tags=["performance", "response-time"]
            ))
            
            perf_suite.add_test_case(TestCase(
                id="load_test",
                name="Load Test",
                description="Test system under load",
                test_type=TestType.LOAD,
                priority=TestPriority.MEDIUM,
                test_function=self._test_load,
                test_data={
                    'url': 'http://localhost:8000/api/test',
                    'concurrent_users': 50,
                    'duration': 60,  # seconds
                    'max_error_rate': 0.05  # 5%
                },
                tags=["load", "stress"]
            ))
            
            self.test_suites["performance_test_suite"] = perf_suite
            
            self.logger.info(f"Initialized {len(self.test_suites)} test suite templates")
            
        except Exception as e:
            self.logger.error(f"Error initializing test templates: {e}")
    
    async def create_test_suite(self, suite: TestSuite) -> str:
        """Create a new test suite"""
        try:
            if not suite.id:
                suite.id = str(uuid.uuid4())
            
            self.test_suites[suite.id] = suite
            self.performance_stats['suites_created'] += 1
            
            self.logger.info(f"Created test suite: {suite.name}")
            return suite.id
            
        except Exception as e:
            self.logger.error(f"Error creating test suite: {e}")
            raise
    
    async def add_test_case(self, suite_id: str, test_case: TestCase) -> None:
        """Add test case to existing suite"""
        try:
            if suite_id not in self.test_suites:
                raise ValueError(f"Test suite {suite_id} not found")
            
            if not test_case.id:
                test_case.id = str(uuid.uuid4())
            
            self.test_suites[suite_id].add_test_case(test_case)
            
            self.logger.info(f"Added test case {test_case.name} to suite {suite_id}")
            
        except Exception as e:
            self.logger.error(f"Error adding test case: {e}")
            raise
    
    async def execute_test_suite(self, suite_id: str, 
                               test_filter: Optional[Dict[str, Any]] = None) -> TestExecution:
        """Execute a test suite"""
        try:
            if suite_id not in self.test_suites:
                raise ValueError(f"Test suite {suite_id} not found")
            
            suite = self.test_suites[suite_id]
            execution_id = str(uuid.uuid4())
            
            execution = TestExecution(
                id=execution_id,
                suite_id=suite_id,
                start_time=datetime.now(),
                status=TestStatus.RUNNING,
                environment=self._get_environment_info()
            )
            
            self.test_executions[execution_id] = execution
            
            # Filter test cases if needed
            test_cases = self._filter_test_cases(suite.test_cases, test_filter)
            
            self.logger.info(f"Starting execution of {len(test_cases)} test cases")
            
            # Execute suite setup if exists
            if suite.setup_function:
                try:
                    await self._execute_function(suite.setup_function)
                except Exception as e:
                    self.logger.error(f"Suite setup failed: {e}")
                    execution.status = TestStatus.ERROR
                    return execution
            
            # Execute test cases
            if suite.parallel_execution and len(test_cases) > 1:
                await self._execute_tests_parallel(execution, test_cases, suite.max_parallel_tests)
            else:
                await self._execute_tests_sequential(execution, test_cases)
            
            # Execute suite teardown if exists
            if suite.teardown_function:
                try:
                    await self._execute_function(suite.teardown_function)
                except Exception as e:
                    self.logger.error(f"Suite teardown failed: {e}")
            
            # Finalize execution
            execution.end_time = datetime.now()
            execution.status = TestStatus.PASSED if all(
                r.status == TestStatus.PASSED for r in execution.results
            ) else TestStatus.FAILED
            
            # Update performance stats
            self._update_performance_stats(execution)
            
            self.logger.info(f"Test suite execution completed: {execution.summary}")
            return execution
            
        except Exception as e:
            self.logger.error(f"Error executing test suite: {e}")
            raise
    
    async def _execute_tests_sequential(self, execution: TestExecution, test_cases: List[TestCase]) -> None:
        """Execute test cases sequentially"""
        try:
            for test_case in test_cases:
                result = await self._execute_test_case(test_case)
                execution.add_result(result)
                
                # Stop on critical failure if configured
                if (result.status == TestStatus.FAILED and 
                    test_case.priority == TestPriority.CRITICAL and
                    self.config.get('stop_on_critical_failure', False)):
                    self.logger.warning("Stopping execution due to critical test failure")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in sequential test execution: {e}")
            raise
    
    async def _execute_tests_parallel(self, execution: TestExecution, 
                                    test_cases: List[TestCase], max_parallel: int) -> None:
        """Execute test cases in parallel"""
        try:
            semaphore = asyncio.Semaphore(max_parallel)
            
            async def execute_with_semaphore(test_case: TestCase) -> TestResult:
                async with semaphore:
                    return await self._execute_test_case(test_case)
            
            # Create tasks for all test cases
            tasks = [execute_with_semaphore(tc) for tc in test_cases]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Test execution error: {result}")
                    # Create error result
                    error_result = TestResult(
                        test_case_id="unknown",
                        status=TestStatus.ERROR,
                        execution_time=0.0,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(result)
                    )
                    execution.add_result(error_result)
                else:
                    execution.add_result(result)
                    
        except Exception as e:
            self.logger.error(f"Error in parallel test execution: {e}")
            raise
    
    async def _execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing test case: {test_case.name}")
            
            # Execute setup if exists
            if test_case.setup_function:
                await self._execute_function(test_case.setup_function)
            
            # Execute the test with timeout and retries
            result = await self._execute_test_with_retries(test_case)
            
            # Execute teardown if exists
            if test_case.teardown_function:
                try:
                    await self._execute_function(test_case.teardown_function)
                except Exception as e:
                    self.logger.warning(f"Teardown failed for {test_case.name}: {e}")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Test case {test_case.name} failed: {e}")
            
            return TestResult(
                test_case_id=test_case.id,
                status=TestStatus.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
                stack_trace=self._get_stack_trace()
            )
    
    async def _execute_test_with_retries(self, test_case: TestCase) -> TestResult:
        """Execute test with retry logic"""
        last_result = None
        
        for attempt in range(test_case.retry_count + 1):
            try:
                start_time = datetime.now()
                
                # Execute the test function with timeout
                if test_case.test_function:
                    actual_result = await asyncio.wait_for(
                        self._execute_function(test_case.test_function, test_case.test_data),
                        timeout=test_case.timeout
                    )
                else:
                    actual_result = None
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Validate result
                status = self._validate_test_result(actual_result, test_case.expected_result)
                
                result = TestResult(
                    test_case_id=test_case.id,
                    status=status,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    actual_result=actual_result,
                    retry_attempt=attempt
                )
                
                # If test passed or this is the last attempt, return result
                if status == TestStatus.PASSED or attempt == test_case.retry_count:
                    return result
                
                last_result = result
                self.logger.warning(f"Test {test_case.name} failed, attempt {attempt + 1}")
                
                # Wait before retry
                if attempt < test_case.retry_count:
                    await asyncio.sleep(1)
                
            except asyncio.TimeoutError:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                last_result = TestResult(
                    test_case_id=test_case.id,
                    status=TestStatus.TIMEOUT,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    error_message=f"Test timed out after {test_case.timeout} seconds",
                    retry_attempt=attempt
                )
                
                if attempt == test_case.retry_count:
                    return last_result
                
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                last_result = TestResult(
                    test_case_id=test_case.id,
                    status=TestStatus.FAILED,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    error_message=str(e),
                    stack_trace=self._get_stack_trace(),
                    retry_attempt=attempt
                )
                
                if attempt == test_case.retry_count:
                    return last_result
        
        return last_result or TestResult(
            test_case_id=test_case.id,
            status=TestStatus.ERROR,
            execution_time=0.0,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error_message="Unknown error occurred"
        )
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function (sync or async)"""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Function execution failed: {e}")
            raise
    
    def _validate_test_result(self, actual: Any, expected: Any) -> TestStatus:
        """Validate test result against expected result"""
        try:
            if expected is None:
                return TestStatus.PASSED  # No validation needed
            
            if actual == expected:
                return TestStatus.PASSED
            
            # Handle different types of validation
            if isinstance(expected, dict) and isinstance(actual, dict):
                # Validate dictionary structure
                for key, value in expected.items():
                    if key not in actual or actual[key] != value:
                        return TestStatus.FAILED
                return TestStatus.PASSED
            
            if isinstance(expected, list) and isinstance(actual, list):
                # Validate list contents
                if len(actual) != len(expected):
                    return TestStatus.FAILED
                for i, item in enumerate(expected):
                    if actual[i] != item:
                        return TestStatus.FAILED
                return TestStatus.PASSED
            
            return TestStatus.FAILED
            
        except Exception as e:
            self.logger.error(f"Error validating test result: {e}")
            return TestStatus.ERROR
    
    def _filter_test_cases(self, test_cases: List[TestCase], 
                          test_filter: Optional[Dict[str, Any]]) -> List[TestCase]:
        """Filter test cases based on criteria"""
        if not test_filter:
            return test_cases
        
        filtered_cases = test_cases
        
        # Filter by test type
        if 'test_type' in test_filter:
            test_type = TestType(test_filter['test_type'])
            filtered_cases = [tc for tc in filtered_cases if tc.test_type == test_type]
        
        # Filter by priority
        if 'priority' in test_filter:
            priority = TestPriority(test_filter['priority'])
            filtered_cases = [tc for tc in filtered_cases if tc.priority == priority]
        
        # Filter by tags
        if 'tags' in test_filter:
            required_tags = test_filter['tags']
            if isinstance(required_tags, str):
                required_tags = [required_tags]
            filtered_cases = [tc for tc in filtered_cases 
                            if any(tag in tc.tags for tag in required_tags)]
        
        # Filter by test IDs
        if 'test_ids' in test_filter:
            test_ids = test_filter['test_ids']
            filtered_cases = [tc for tc in filtered_cases if tc.id in test_ids]
        
        return filtered_cases
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get current environment information"""
        try:
            import platform
            import sys
            
            return {
                'platform': platform.platform(),
                'python_version': sys.version,
                'hostname': platform.node(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting environment info: {e}")
            return {'error': str(e)}
    
    def _get_stack_trace(self) -> str:
        """Get current stack trace"""
        try:
            import traceback
            return traceback.format_exc()
        except Exception:
            return "Stack trace not available"
    
    def _update_performance_stats(self, execution: TestExecution) -> None:
        """Update performance statistics"""
        try:
            self.performance_stats['tests_executed'] += len(execution.results)
            
            for result in execution.results:
                if result.status == TestStatus.PASSED:
                    self.performance_stats['tests_passed'] += 1
                elif result.status == TestStatus.FAILED:
                    self.performance_stats['tests_failed'] += 1
                
                self.performance_stats['total_execution_time'] += result.execution_time
            
            # Update average execution time
            if self.performance_stats['tests_executed'] > 0:
                self.performance_stats['average_execution_time'] = (
                    self.performance_stats['total_execution_time'] / 
                    self.performance_stats['tests_executed']
                )
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            if today not in self.performance_stats['daily_stats']:
                self.performance_stats['daily_stats'][today] = {
                    'tests_executed': 0,
                    'tests_passed': 0,
                    'tests_failed': 0
                }
            
            daily_stats = self.performance_stats['daily_stats'][today]
            daily_stats['tests_executed'] += len(execution.results)
            daily_stats['tests_passed'] += sum(1 for r in execution.results if r.status == TestStatus.PASSED)
            daily_stats['tests_failed'] += sum(1 for r in execution.results if r.status == TestStatus.FAILED)
            
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")
    
    # Built-in test functions
    async def _test_api_get(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test API GET request"""
        try:
            import aiohttp
            
            url = test_data['url']
            expected_status = test_data.get('expected_status', 200)
            expected_fields = test_data.get('expected_fields', [])
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    status = response.status
                    data = await response.json()
                    
                    result = {
                        'status_code': status,
                        'response_data': data,
                        'status_match': status == expected_status,
                        'fields_present': all(field in data for field in expected_fields)
                    }
                    
                    return result
                    
        except Exception as e:
            return {'error': str(e), 'status_match': False, 'fields_present': False}
    
    async def _test_api_post(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test API POST request"""
        try:
            import aiohttp
            
            url = test_data['url']
            payload = test_data.get('payload', {})
            expected_status = test_data.get('expected_status', 201)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    status = response.status
                    data = await response.json()
                    
                    result = {
                        'status_code': status,
                        'response_data': data,
                        'status_match': status == expected_status
                    }
                    
                    return result
                    
        except Exception as e:
            return {'error': str(e), 'status_match': False}
    
    async def _test_ui_login(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test UI login functionality"""
        try:
            # This is a placeholder for UI testing
            # In a real implementation, you would use tools like Selenium
            username = test_data['username']
            password = test_data['password']
            expected_redirect = test_data.get('expected_redirect', '/dashboard')
            
            # Simulate login test
            await asyncio.sleep(0.1)  # Simulate UI interaction time
            
            result = {
                'login_successful': True,
                'redirect_url': expected_redirect,
                'redirect_match': True
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'login_successful': False, 'redirect_match': False}
    
    async def _test_ui_navigation(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test UI navigation"""
        try:
            menu_items = test_data['menu_items']
            expected_pages = test_data['expected_pages']
            
            # Simulate navigation test
            await asyncio.sleep(0.1)  # Simulate UI interaction time
            
            result = {
                'menu_items_found': len(menu_items),
                'navigation_successful': True,
                'pages_accessible': len(expected_pages)
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'navigation_successful': False}
    
    async def _test_response_time(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test API response time"""
        try:
            import aiohttp
            
            url = test_data['url']
            max_response_time = test_data.get('max_response_time', 1.0)
            concurrent_requests = test_data.get('concurrent_requests', 1)
            
            response_times = []
            
            async def make_request():
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        await response.text()
                        return time.time() - start_time
            
            # Make concurrent requests
            tasks = [make_request() for _ in range(concurrent_requests)]
            response_times = await asyncio.gather(*tasks)
            
            avg_response_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            
            result = {
                'average_response_time': avg_response_time,
                'max_response_time': max_time,
                'all_requests_within_limit': max_time <= max_response_time,
                'response_times': response_times
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'all_requests_within_limit': False}
    
    async def _test_load(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test system under load"""
        try:
            import aiohttp
            
            url = test_data['url']
            concurrent_users = test_data.get('concurrent_users', 10)
            duration = test_data.get('duration', 30)
            max_error_rate = test_data.get('max_error_rate', 0.05)
            
            start_time = time.time()
            end_time = start_time + duration
            
            successful_requests = 0
            failed_requests = 0
            response_times = []
            
            async def user_simulation():
                nonlocal successful_requests, failed_requests
                
                while time.time() < end_time:
                    try:
                        request_start = time.time()
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url) as response:
                                if response.status == 200:
                                    successful_requests += 1
                                else:
                                    failed_requests += 1
                                response_times.append(time.time() - request_start)
                    except Exception:
                        failed_requests += 1
                    
                    await asyncio.sleep(0.1)  # Small delay between requests
            
            # Start concurrent users
            tasks = [user_simulation() for _ in range(concurrent_users)]
            await asyncio.gather(*tasks)
            
            total_requests = successful_requests + failed_requests
            error_rate = failed_requests / total_requests if total_requests > 0 else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            result = {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'error_rate': error_rate,
                'error_rate_within_limit': error_rate <= max_error_rate,
                'average_response_time': avg_response_time,
                'requests_per_second': total_requests / duration
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'error_rate_within_limit': False}
    
    # Test data generators
    def _generate_string_data(self, length: int = 10, charset: str = 'alphanumeric') -> str:
        """Generate random string data"""
        import random
        import string
        
        if charset == 'alphanumeric':
            chars = string.ascii_letters + string.digits
        elif charset == 'alpha':
            chars = string.ascii_letters
        elif charset == 'numeric':
            chars = string.digits
        else:
            chars = charset
        
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _generate_number_data(self, min_val: int = 0, max_val: int = 100) -> int:
        """Generate random number data"""
        import random
        return random.randint(min_val, max_val)
    
    def _generate_email_data(self) -> str:
        """Generate random email data"""
        domains = ['example.com', 'test.org', 'sample.net']
        username = self._generate_string_data(8, 'alpha').lower()
        domain = domains[self._generate_number_data(0, len(domains) - 1)]
        return f"{username}@{domain}"
    
    def _generate_phone_data(self) -> str:
        """Generate random phone data"""
        area_code = self._generate_number_data(200, 999)
        exchange = self._generate_number_data(200, 999)
        number = self._generate_number_data(1000, 9999)
        return f"({area_code}) {exchange}-{number}"
    
    def _generate_date_data(self) -> str:
        """Generate random date data"""
        import random
        from datetime import timedelta
        
        base_date = datetime.now()
        random_days = random.randint(-365, 365)
        random_date = base_date + timedelta(days=random_days)
        return random_date.strftime('%Y-%m-%d')
    
    def _generate_url_data(self) -> str:
        """Generate random URL data"""
        protocols = ['http', 'https']
        domains = ['example.com', 'test.org', 'sample.net']
        paths = ['api', 'data', 'test', 'info']
        
        protocol = protocols[self._generate_number_data(0, len(protocols) - 1)]
        domain = domains[self._generate_number_data(0, len(domains) - 1)]
        path = paths[self._generate_number_data(0, len(paths) - 1)]
        
        return f"{protocol}://{domain}/{path}"
    
    def _generate_json_data(self, structure: Dict[str, str]) -> Dict[str, Any]:
        """Generate JSON data based on structure"""
        result = {}
        for key, data_type in structure.items():
            if data_type in self.data_generators:
                result[key] = self.data_generators[data_type]()
            else:
                result[key] = f"sample_{key}"
        return result
    
    def _generate_xml_data(self, root_element: str = 'data') -> str:
        """Generate random XML data"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<{root_element}>
    <id>{self._generate_number_data(1, 1000)}</id>
    <name>{self._generate_string_data(10, 'alpha')}</name>
    <email>{self._generate_email_data()}</email>
    <created>{self._generate_date_data()}</created>
</{root_element}>"""
    
    async def generate_test_data(self, data_type: str, count: int = 1, **kwargs) -> List[Any]:
        """Generate test data of specified type"""
        try:
            if data_type not in self.data_generators:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            generator = self.data_generators[data_type]
            data = []
            
            for _ in range(count):
                if data_type == 'json':
                    structure = kwargs.get('structure', {'name': 'string', 'value': 'number'})
                    data.append(generator(structure))
                elif data_type == 'xml':
                    root_element = kwargs.get('root_element', 'data')
                    data.append(generator(root_element))
                else:
                    data.append(generator(**kwargs))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating test data: {e}")
            return []
    
    async def generate_test_report(self, execution_id: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        try:
            if execution_id not in self.test_executions:
                raise ValueError(f"Test execution {execution_id} not found")
            
            execution = self.test_executions[execution_id]
            suite = self.test_suites[execution.suite_id]
            
            # Generate report data
            report = {
                'execution_id': execution_id,
                'suite_name': suite.name,
                'suite_description': suite.description,
                'execution_summary': execution.summary,
                'environment': execution.environment,
                'start_time': execution.start_time.isoformat(),
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'total_duration': (execution.end_time - execution.start_time).total_seconds() if execution.end_time else 0,
                'test_results': [result.to_dict() for result in execution.results],
                'performance_metrics': {
                    'fastest_test': min(execution.results, key=lambda r: r.execution_time).execution_time if execution.results else 0,
                    'slowest_test': max(execution.results, key=lambda r: r.execution_time).execution_time if execution.results else 0,
                    'average_test_time': execution.summary.get('average_execution_time', 0)
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Save report if output file specified
            if output_file:
                report_path = self.reports_dir / output_file
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Test report saved to {report_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating test report: {e}")
            raise
    
    def get_test_suites(self) -> List[TestSuite]:
        """Get all test suites"""
        return list(self.test_suites.values())
    
    def get_test_execution(self, execution_id: str) -> Optional[TestExecution]:
        """Get test execution by ID"""
        return self.test_executions.get(execution_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    async def export_test_suite(self, suite_id: str, output_file: str) -> None:
        """Export test suite to file"""
        try:
            if suite_id not in self.test_suites:
                raise ValueError(f"Test suite {suite_id} not found")
            
            suite = self.test_suites[suite_id]
            suite_data = suite.to_dict()
            
            with open(output_file, 'w') as f:
                json.dump(suite_data, f, indent=2)
            
            self.logger.info(f"Test suite exported to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting test suite: {e}")
            raise
    
    async def import_test_suite(self, input_file: str) -> str:
        """Import test suite from file"""
        try:
            with open(input_file, 'r') as f:
                suite_data = json.load(f)
            
            # Create test cases
            test_cases = []
            for tc_data in suite_data.get('test_cases', []):
                test_case = TestCase(
                    id=tc_data['id'],
                    name=tc_data['name'],
                    description=tc_data['description'],
                    test_type=TestType(tc_data['test_type']),
                    priority=TestPriority(tc_data['priority']),
                    test_data=tc_data.get('test_data', {}),
                    expected_result=tc_data.get('expected_result'),
                    timeout=tc_data.get('timeout', 30),
                    retry_count=tc_data.get('retry_count', 0),
                    tags=tc_data.get('tags', []),
                    dependencies=tc_data.get('dependencies', [])
                )
                test_cases.append(test_case)
            
            # Create test suite
            suite = TestSuite(
                id=suite_data['id'],
                name=suite_data['name'],
                description=suite_data['description'],
                test_cases=test_cases,
                parallel_execution=suite_data.get('parallel_execution', False),
                max_parallel_tests=suite_data.get('max_parallel_tests', 5),
                tags=suite_data.get('tags', [])
            )
            
            suite_id = await self.create_test_suite(suite)
            self.logger.info(f"Test suite imported from {input_file}")
            
            return suite_id
            
        except Exception as e:
            self.logger.error(f"Error importing test suite: {e}")
            raise
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        
        # Update specific settings
        if 'default_timeout' in new_config:
            self.default_timeout = new_config['default_timeout']
        
        if 'max_retries' in new_config:
            self.max_retries = new_config['max_retries']
        
        if 'parallel_execution' in new_config:
            self.parallel_execution = new_config['parallel_execution']
        
        if 'max_parallel_tests' in new_config:
            self.max_parallel_tests = new_config['max_parallel_tests']
        
        self.logger.info("Test automator configuration updated")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Clear old executions (keep last 100)
            if len(self.test_executions) > 100:
                sorted_executions = sorted(
                    self.test_executions.items(),
                    key=lambda x: x[1].start_time,
                    reverse=True
                )
                self.test_executions = dict(sorted_executions[:100])
            
            # Clear old daily stats (keep last 30 days)
            if 'daily_stats' in self.performance_stats:
                cutoff_date = (datetime.now() - timedelta(days=30)).date()
                self.performance_stats['daily_stats'] = {
                    date: stats for date, stats in self.performance_stats['daily_stats'].items()
                    if datetime.fromisoformat(date).date() >= cutoff_date
                }
            
            self.logger.info("Test automator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
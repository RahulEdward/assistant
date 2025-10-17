"""
Code Generation Module for Computer Assistant

This module provides comprehensive code generation capabilities including:
- Template-based code generation for multiple languages
- AI-assisted code completion and refactoring
- Documentation generation and code analysis
- Project scaffolding and boilerplate generation
- Code quality assessment and improvement suggestions
"""

import ast
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import subprocess


class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    POWERSHELL = "powershell"


class CodeType(Enum):
    """Types of code to generate"""
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"
    INTERFACE = "interface"
    COMPONENT = "component"
    SERVICE = "service"
    MODEL = "model"
    CONTROLLER = "controller"
    TEST = "test"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    SCRIPT = "script"


class QualityLevel(Enum):
    """Code quality levels"""
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"
    PRODUCTION = "production"


@dataclass
class CodeTemplate:
    """Code template structure"""
    name: str
    language: CodeLanguage
    code_type: CodeType
    template: str
    variables: Dict[str, str]
    description: str
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'language': self.language.value,
            'code_type': self.code_type.value,
            'template': self.template,
            'variables': self.variables,
            'description': self.description,
            'tags': self.tags,
            'author': self.author,
            'version': self.version,
            'dependencies': self.dependencies
        }


@dataclass
class GeneratedCode:
    """Generated code structure"""
    content: str
    language: CodeLanguage
    code_type: CodeType
    filename: str
    template_used: Optional[str]
    variables_used: Dict[str, str]
    timestamp: datetime
    quality_score: float
    suggestions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tests_generated: bool = False
    documentation_generated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'content': self.content,
            'language': self.language.value,
            'code_type': self.code_type.value,
            'filename': self.filename,
            'template_used': self.template_used,
            'variables_used': self.variables_used,
            'timestamp': self.timestamp.isoformat(),
            'quality_score': self.quality_score,
            'suggestions': self.suggestions,
            'dependencies': self.dependencies,
            'tests_generated': self.tests_generated,
            'documentation_generated': self.documentation_generated
        }


@dataclass
class CodeAnalysis:
    """Code analysis results"""
    filename: str
    language: CodeLanguage
    lines_of_code: int
    complexity_score: float
    quality_score: float
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime


class CodeGenerator:
    """
    Advanced code generation and development assistance system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the code generator"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Templates storage
        self.templates: Dict[str, CodeTemplate] = {}
        self.template_dir = Path(self.config.get('template_dir', 'templates'))
        self.template_dir.mkdir(exist_ok=True)
        
        # Generated code history
        self.generation_history: List[GeneratedCode] = []
        self.max_history = self.config.get('max_history', 1000)
        
        # Code analysis cache
        self.analysis_cache: Dict[str, CodeAnalysis] = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # Performance tracking
        self.performance_stats = {
            'code_generated': 0,
            'templates_used': 0,
            'analyses_performed': 0,
            'average_generation_time': 0.0,
            'quality_scores': [],
            'languages_used': {},
            'code_types_generated': {}
        }
        
        # Language-specific configurations
        self.language_configs = {
            CodeLanguage.PYTHON: {
                'file_extension': '.py',
                'comment_style': '#',
                'indent': '    ',
                'line_ending': '\n'
            },
            CodeLanguage.JAVASCRIPT: {
                'file_extension': '.js',
                'comment_style': '//',
                'indent': '  ',
                'line_ending': '\n'
            },
            CodeLanguage.TYPESCRIPT: {
                'file_extension': '.ts',
                'comment_style': '//',
                'indent': '  ',
                'line_ending': '\n'
            },
            CodeLanguage.JAVA: {
                'file_extension': '.java',
                'comment_style': '//',
                'indent': '    ',
                'line_ending': '\n'
            },
            CodeLanguage.CSHARP: {
                'file_extension': '.cs',
                'comment_style': '//',
                'indent': '    ',
                'line_ending': '\n'
            }
        }
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
        
        self.logger.info("Code generator initialized")
    
    def _initialize_builtin_templates(self) -> None:
        """Initialize built-in code templates"""
        try:
            # Python class template
            python_class_template = CodeTemplate(
                name="python_class",
                language=CodeLanguage.PYTHON,
                code_type=CodeType.CLASS,
                template='''"""
{description}
"""

import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class {class_name}:
    """
    {class_description}
    """
    
    def __init__(self, {init_params}):
        """Initialize the {class_name}"""
        {init_body}
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"{class_name} initialized")
    
    {methods}
    
    def __str__(self) -> str:
        """String representation"""
        return f"{class_name}({self.__dict__})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return self.__str__()
''',
                variables={
                    'class_name': 'MyClass',
                    'class_description': 'A sample class',
                    'description': 'Module description',
                    'init_params': 'name: str',
                    'init_body': 'self.name = name',
                    'methods': '''def get_name(self) -> str:
        """Get the name"""
        return self.name
    
    def set_name(self, name: str) -> None:
        """Set the name"""
        self.name = name'''
                },
                description="Python class template with logging and type hints",
                tags=["python", "class", "oop"],
                dependencies=["logging", "typing", "dataclasses"]
            )
            self.templates["python_class"] = python_class_template
            
            # Python function template
            python_function_template = CodeTemplate(
                name="python_function",
                language=CodeLanguage.PYTHON,
                code_type=CodeType.FUNCTION,
                template='''def {function_name}({parameters}) -> {return_type}:
    """
    {function_description}
    
    Args:
        {args_description}
    
    Returns:
        {return_description}
    
    Raises:
        {raises_description}
    """
    try:
        {function_body}
        
    except Exception as e:
        logging.error(f"Error in {function_name}: {{e}}")
        raise
''',
                variables={
                    'function_name': 'my_function',
                    'parameters': 'param1: str, param2: int = 0',
                    'return_type': 'str',
                    'function_description': 'Function description',
                    'args_description': 'param1: Description of param1\n        param2: Description of param2',
                    'return_description': 'Description of return value',
                    'raises_description': 'Exception: When something goes wrong',
                    'function_body': 'result = f"{param1}_{param2}"\n        return result'
                },
                description="Python function template with documentation and error handling",
                tags=["python", "function"],
                dependencies=["logging"]
            )
            self.templates["python_function"] = python_function_template
            
            # JavaScript class template
            js_class_template = CodeTemplate(
                name="javascript_class",
                language=CodeLanguage.JAVASCRIPT,
                code_type=CodeType.CLASS,
                template='''/**
 * {class_description}
 */
class {class_name} {
  /**
   * Initialize the {class_name}
   * @param {{Object}} options - Configuration options
   */
  constructor({constructor_params}) {
    {constructor_body}
    this.logger = console;
    this.logger.info(`{class_name} initialized`);
  }
  
  {methods}
  
  /**
   * String representation
   * @returns {{string}} String representation
   */
  toString() {
    return `{class_name}(${{JSON.stringify(this)}})`;
  }
}

module.exports = {class_name};
''',
                variables={
                    'class_name': 'MyClass',
                    'class_description': 'A sample JavaScript class',
                    'constructor_params': 'options = {}',
                    'constructor_body': 'this.name = options.name || "default";',
                    'methods': '''/**
   * Get the name
   * @returns {string} The name
   */
  getName() {
    return this.name;
  }
  
  /**
   * Set the name
   * @param {string} name - The new name
   */
  setName(name) {
    this.name = name;
  }'''
                },
                description="JavaScript class template with JSDoc documentation",
                tags=["javascript", "class", "oop"],
                dependencies=[]
            )
            self.templates["javascript_class"] = js_class_template
            
            # REST API endpoint template
            api_endpoint_template = CodeTemplate(
                name="rest_api_endpoint",
                language=CodeLanguage.PYTHON,
                code_type=CodeType.FUNCTION,
                template='''from flask import Flask, request, jsonify
from typing import Dict, Any
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('{endpoint_path}', methods=['{http_method}'])
def {function_name}():
    """
    {endpoint_description}
    
    Returns:
        JSON response with {return_description}
    """
    try:
        # Get request data
        {request_handling}
        
        # Process the request
        {processing_logic}
        
        # Return response
        return jsonify({{
            'success': True,
            'data': result,
            'message': '{success_message}'
        }}), 200
        
    except ValueError as e:
        logger.error(f"Validation error in {function_name}: {{e}}")
        return jsonify({{
            'success': False,
            'error': str(e),
            'message': 'Invalid input data'
        }}), 400
        
    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}")
        return jsonify({{
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }}), 500

if __name__ == '__main__':
    app.run(debug=True)
''',
                variables={
                    'endpoint_path': '/api/data',
                    'http_method': 'GET',
                    'function_name': 'get_data',
                    'endpoint_description': 'API endpoint description',
                    'return_description': 'the requested data',
                    'request_handling': 'data = request.get_json() if request.is_json else {}',
                    'processing_logic': 'result = {"message": "Hello, World!"}',
                    'success_message': 'Data retrieved successfully'
                },
                description="REST API endpoint template with error handling",
                tags=["python", "api", "flask", "rest"],
                dependencies=["flask"]
            )
            self.templates["rest_api_endpoint"] = api_endpoint_template
            
            # Unit test template
            unit_test_template = CodeTemplate(
                name="python_unit_test",
                language=CodeLanguage.PYTHON,
                code_type=CodeType.TEST,
                template='''"""
Unit tests for {module_name}
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from {module_import} import {class_name}


class Test{class_name}(unittest.TestCase):
    """Test cases for {class_name}"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        {setup_code}
    
    def tearDown(self):
        """Clean up after each test method"""
        {teardown_code}
    
    {test_methods}
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        {edge_case_tests}
    
    def test_error_handling(self):
        """Test error handling and exceptions"""
        {error_handling_tests}

if __name__ == '__main__':
    unittest.main()
''',
                variables={
                    'module_name': 'my_module',
                    'module_import': 'my_module',
                    'class_name': 'MyClass',
                    'setup_code': 'self.instance = MyClass("test")',
                    'teardown_code': 'pass',
                    'test_methods': '''def test_initialization(self):
        """Test object initialization"""
        self.assertIsNotNone(self.instance)
        self.assertEqual(self.instance.name, "test")
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.instance.get_name()
        self.assertEqual(result, "test")''',
                    'edge_case_tests': 'pass  # Add edge case tests here',
                    'error_handling_tests': 'pass  # Add error handling tests here'
                },
                description="Python unit test template with setUp and tearDown",
                tags=["python", "test", "unittest"],
                dependencies=["unittest"]
            )
            self.templates["python_unit_test"] = unit_test_template
            
            self.logger.info(f"Initialized {len(self.templates)} built-in templates")
            
        except Exception as e:
            self.logger.error(f"Error initializing built-in templates: {e}")
    
    async def generate_code(self, template_name: str, variables: Dict[str, str],
                          filename: Optional[str] = None) -> GeneratedCode:
        """Generate code from template"""
        try:
            start_time = time.time()
            
            if template_name not in self.templates:
                raise ValueError(f"Template '{template_name}' not found")
            
            template = self.templates[template_name]
            
            # Merge provided variables with template defaults
            merged_variables = {**template.variables, **variables}
            
            # Generate code content
            content = template.template
            for var_name, var_value in merged_variables.items():
                placeholder = f"{{{var_name}}}"
                content = content.replace(placeholder, str(var_value))
            
            # Generate filename if not provided
            if not filename:
                lang_config = self.language_configs.get(template.language, {})
                extension = lang_config.get('file_extension', '.txt')
                base_name = merged_variables.get('class_name', merged_variables.get('function_name', 'generated'))
                filename = f"{base_name.lower()}{extension}"
            
            # Analyze code quality
            quality_score = await self._analyze_code_quality(content, template.language)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(content, template.language, template.code_type)
            
            # Create generated code object
            generated_code = GeneratedCode(
                content=content,
                language=template.language,
                code_type=template.code_type,
                filename=filename,
                template_used=template_name,
                variables_used=merged_variables,
                timestamp=datetime.now(),
                quality_score=quality_score,
                suggestions=suggestions,
                dependencies=template.dependencies.copy()
            )
            
            # Update performance stats
            generation_time = time.time() - start_time
            self._update_generation_stats(template.language, template.code_type, quality_score, generation_time)
            
            # Add to history
            self.generation_history.append(generated_code)
            if len(self.generation_history) > self.max_history:
                self.generation_history = self.generation_history[-self.max_history:]
            
            self.logger.info(f"Generated {template.code_type.value} code for {template.language.value}")
            return generated_code
            
        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            raise
    
    async def create_template(self, template: CodeTemplate) -> None:
        """Create a new code template"""
        try:
            # Validate template
            if not template.name or not template.template:
                raise ValueError("Template name and content are required")
            
            # Store template
            self.templates[template.name] = template
            
            # Save to file if template directory exists
            template_file = self.template_dir / f"{template.name}.json"
            with open(template_file, 'w') as f:
                json.dump(template.to_dict(), f, indent=2)
            
            self.logger.info(f"Created template: {template.name}")
            
        except Exception as e:
            self.logger.error(f"Error creating template: {e}")
            raise
    
    async def load_templates_from_directory(self, directory: str) -> int:
        """Load templates from directory"""
        try:
            template_dir = Path(directory)
            if not template_dir.exists():
                raise ValueError(f"Directory {directory} does not exist")
            
            loaded_count = 0
            for template_file in template_dir.glob("*.json"):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                    
                    template = CodeTemplate(
                        name=template_data['name'],
                        language=CodeLanguage(template_data['language']),
                        code_type=CodeType(template_data['code_type']),
                        template=template_data['template'],
                        variables=template_data['variables'],
                        description=template_data['description'],
                        tags=template_data.get('tags', []),
                        author=template_data.get('author'),
                        version=template_data.get('version', '1.0.0'),
                        dependencies=template_data.get('dependencies', [])
                    )
                    
                    self.templates[template.name] = template
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error loading template from {template_file}: {e}")
                    continue
            
            self.logger.info(f"Loaded {loaded_count} templates from {directory}")
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"Error loading templates: {e}")
            return 0
    
    async def analyze_code(self, code: str, language: CodeLanguage, filename: str = "code.py") -> CodeAnalysis:
        """Analyze code quality and complexity"""
        try:
            # Check cache first
            cache_key = f"{filename}_{hash(code)}"
            if cache_key in self.analysis_cache:
                cached_analysis = self.analysis_cache[cache_key]
                if (datetime.now() - cached_analysis.timestamp).seconds < self.cache_ttl:
                    return cached_analysis
            
            # Count lines of code
            lines = code.strip().split('\n')
            loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # Calculate complexity (simplified)
            complexity_score = await self._calculate_complexity(code, language)
            
            # Calculate quality score
            quality_score = await self._analyze_code_quality(code, language)
            
            # Find issues
            issues = await self._find_code_issues(code, language)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(code, language, CodeType.MODULE)
            
            # Calculate metrics
            metrics = await self._calculate_code_metrics(code, language)
            
            analysis = CodeAnalysis(
                filename=filename,
                language=language,
                lines_of_code=loc,
                complexity_score=complexity_score,
                quality_score=quality_score,
                issues=issues,
                suggestions=suggestions,
                metrics=metrics,
                timestamp=datetime.now()
            )
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            self.performance_stats['analyses_performed'] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing code: {e}")
            raise
    
    async def _analyze_code_quality(self, code: str, language: CodeLanguage) -> float:
        """Analyze code quality and return score (0.0 to 1.0)"""
        try:
            score = 0.0
            factors = 0
            
            # Check for documentation
            if language == CodeLanguage.PYTHON:
                if '"""' in code or "'''" in code:
                    score += 0.2
                    factors += 1
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                if '/**' in code and '*/' in code:
                    score += 0.2
                    factors += 1
            
            # Check for error handling
            error_keywords = ['try', 'catch', 'except', 'finally', 'throw', 'raise']
            if any(keyword in code.lower() for keyword in error_keywords):
                score += 0.2
                factors += 1
            
            # Check for type hints (Python) or types (TypeScript)
            if language == CodeLanguage.PYTHON:
                if '->' in code or ': str' in code or ': int' in code:
                    score += 0.15
                    factors += 1
            elif language == CodeLanguage.TYPESCRIPT:
                if ': string' in code or ': number' in code or ': boolean' in code:
                    score += 0.15
                    factors += 1
            
            # Check for logging
            logging_keywords = ['log', 'logger', 'console.log', 'print']
            if any(keyword in code.lower() for keyword in logging_keywords):
                score += 0.1
                factors += 1
            
            # Check for constants (uppercase variables)
            if re.search(r'\b[A-Z_]{2,}\b', code):
                score += 0.1
                factors += 1
            
            # Check for proper naming conventions
            if language == CodeLanguage.PYTHON:
                # Snake_case for functions and variables
                if re.search(r'def [a-z_]+\(', code):
                    score += 0.1
                    factors += 1
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                # CamelCase for functions
                if re.search(r'function [a-z][a-zA-Z]*\(', code):
                    score += 0.1
                    factors += 1
            
            # Check for comments
            comment_chars = {'#', '//', '/*', '*'}
            lines = code.split('\n')
            comment_lines = sum(1 for line in lines if any(line.strip().startswith(char) for char in comment_chars))
            if comment_lines > 0:
                comment_ratio = comment_lines / len(lines)
                score += min(0.15, comment_ratio * 0.5)
                factors += 1
            
            # Normalize score
            if factors > 0:
                return min(1.0, score)
            else:
                return 0.5  # Default score if no factors found
            
        except Exception as e:
            self.logger.error(f"Error analyzing code quality: {e}")
            return 0.5
    
    async def _calculate_complexity(self, code: str, language: CodeLanguage) -> float:
        """Calculate code complexity score"""
        try:
            complexity = 1.0  # Base complexity
            
            # Count control flow statements
            control_keywords = ['if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'try', 'catch']
            for keyword in control_keywords:
                complexity += code.lower().count(keyword) * 0.1
            
            # Count nested levels (simplified)
            lines = code.split('\n')
            max_indent = 0
            for line in lines:
                if line.strip():
                    indent_level = (len(line) - len(line.lstrip())) // 4  # Assuming 4-space indentation
                    max_indent = max(max_indent, indent_level)
            
            complexity += max_indent * 0.2
            
            # Count function definitions
            if language == CodeLanguage.PYTHON:
                complexity += code.count('def ') * 0.1
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                complexity += code.count('function ') * 0.1
                complexity += code.count('=>') * 0.1  # Arrow functions
            
            return min(10.0, complexity)  # Cap at 10.0
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {e}")
            return 1.0
    
    async def _find_code_issues(self, code: str, language: CodeLanguage) -> List[Dict[str, Any]]:
        """Find potential issues in code"""
        try:
            issues = []
            lines = code.split('\n')
            
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for long lines
                if len(line) > 120:
                    issues.append({
                        'type': 'style',
                        'severity': 'warning',
                        'line': i,
                        'message': f'Line too long ({len(line)} characters)',
                        'suggestion': 'Consider breaking this line into multiple lines'
                    })
                
                # Check for TODO/FIXME comments
                if 'todo' in line_stripped.lower() or 'fixme' in line_stripped.lower():
                    issues.append({
                        'type': 'maintenance',
                        'severity': 'info',
                        'line': i,
                        'message': 'TODO/FIXME comment found',
                        'suggestion': 'Address this comment before production'
                    })
                
                # Language-specific checks
                if language == CodeLanguage.PYTHON:
                    # Check for bare except
                    if line_stripped == 'except:':
                        issues.append({
                            'type': 'error_handling',
                            'severity': 'error',
                            'line': i,
                            'message': 'Bare except clause',
                            'suggestion': 'Specify exception type or use "except Exception:"'
                        })
                    
                    # Check for print statements (should use logging)
                    if 'print(' in line_stripped and 'logger' not in code:
                        issues.append({
                            'type': 'logging',
                            'severity': 'warning',
                            'line': i,
                            'message': 'Using print() instead of logging',
                            'suggestion': 'Consider using logging instead of print()'
                        })
                
                elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                    # Check for console.log in production code
                    if 'console.log(' in line_stripped:
                        issues.append({
                            'type': 'logging',
                            'severity': 'warning',
                            'line': i,
                            'message': 'console.log() found',
                            'suggestion': 'Remove console.log() statements before production'
                        })
                    
                    # Check for == instead of ===
                    if ' == ' in line_stripped and ' === ' not in line_stripped:
                        issues.append({
                            'type': 'comparison',
                            'severity': 'warning',
                            'line': i,
                            'message': 'Using == instead of ===',
                            'suggestion': 'Use === for strict equality comparison'
                        })
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error finding code issues: {e}")
            return []
    
    async def _generate_suggestions(self, code: str, language: CodeLanguage, code_type: CodeType) -> List[str]:
        """Generate improvement suggestions"""
        try:
            suggestions = []
            
            # General suggestions
            if 'TODO' in code or 'FIXME' in code:
                suggestions.append("Address TODO and FIXME comments before finalizing the code")
            
            if len(code.split('\n')) > 100:
                suggestions.append("Consider breaking this large code block into smaller, more manageable functions")
            
            # Language-specific suggestions
            if language == CodeLanguage.PYTHON:
                if 'import *' in code:
                    suggestions.append("Avoid wildcard imports; import specific functions/classes instead")
                
                if '"""' not in code and "'''" not in code:
                    suggestions.append("Add docstrings to document your functions and classes")
                
                if 'logging' not in code and 'print(' in code:
                    suggestions.append("Consider using the logging module instead of print statements")
                
                if code_type == CodeType.CLASS and '__str__' not in code:
                    suggestions.append("Consider implementing __str__ and __repr__ methods for better object representation")
            
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                if 'var ' in code:
                    suggestions.append("Use 'let' or 'const' instead of 'var' for better scoping")
                
                if '/**' not in code:
                    suggestions.append("Add JSDoc comments to document your functions and classes")
                
                if language == CodeLanguage.JAVASCRIPT and code_type == CodeType.CLASS:
                    suggestions.append("Consider migrating to TypeScript for better type safety")
            
            # Code type specific suggestions
            if code_type == CodeType.FUNCTION:
                if 'try' not in code and 'catch' not in code:
                    suggestions.append("Consider adding error handling with try-catch blocks")
            
            elif code_type == CodeType.CLASS:
                if 'test' not in code.lower():
                    suggestions.append("Consider writing unit tests for this class")
            
            elif code_type == CodeType.API:
                if 'validation' not in code.lower():
                    suggestions.append("Add input validation for API endpoints")
                
                if 'rate' not in code.lower() and 'limit' not in code.lower():
                    suggestions.append("Consider implementing rate limiting for API endpoints")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []
    
    async def _calculate_code_metrics(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Calculate various code metrics"""
        try:
            lines = code.split('\n')
            
            metrics = {
                'total_lines': len(lines),
                'blank_lines': len([line for line in lines if not line.strip()]),
                'comment_lines': 0,
                'code_lines': 0,
                'function_count': 0,
                'class_count': 0,
                'max_line_length': max(len(line) for line in lines) if lines else 0,
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
            }
            
            # Count comment lines and code lines
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                elif line_stripped.startswith('#') or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                    metrics['comment_lines'] += 1
                else:
                    metrics['code_lines'] += 1
            
            # Language-specific metrics
            if language == CodeLanguage.PYTHON:
                metrics['function_count'] = code.count('def ')
                metrics['class_count'] = code.count('class ')
                metrics['import_count'] = code.count('import ') + code.count('from ')
            
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                metrics['function_count'] = code.count('function ') + code.count('=>')
                metrics['class_count'] = code.count('class ')
                metrics['import_count'] = code.count('import ') + code.count('require(')
            
            # Calculate ratios
            if metrics['total_lines'] > 0:
                metrics['comment_ratio'] = metrics['comment_lines'] / metrics['total_lines']
                metrics['code_ratio'] = metrics['code_lines'] / metrics['total_lines']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _update_generation_stats(self, language: CodeLanguage, code_type: CodeType, 
                               quality_score: float, generation_time: float) -> None:
        """Update performance statistics"""
        try:
            self.performance_stats['code_generated'] += 1
            self.performance_stats['templates_used'] += 1
            self.performance_stats['quality_scores'].append(quality_score)
            
            # Update average generation time
            current_avg = self.performance_stats['average_generation_time']
            count = self.performance_stats['code_generated']
            self.performance_stats['average_generation_time'] = (current_avg * (count - 1) + generation_time) / count
            
            # Update language usage
            lang_key = language.value
            if lang_key not in self.performance_stats['languages_used']:
                self.performance_stats['languages_used'][lang_key] = 0
            self.performance_stats['languages_used'][lang_key] += 1
            
            # Update code type usage
            type_key = code_type.value
            if type_key not in self.performance_stats['code_types_generated']:
                self.performance_stats['code_types_generated'][type_key] = 0
            self.performance_stats['code_types_generated'][type_key] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating generation stats: {e}")
    
    async def generate_documentation(self, code: str, language: CodeLanguage) -> str:
        """Generate documentation for code"""
        try:
            # Analyze the code first
            analysis = await self.analyze_code(code, language)
            
            # Generate documentation based on language
            if language == CodeLanguage.PYTHON:
                return await self._generate_python_docs(code, analysis)
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                return await self._generate_js_docs(code, analysis)
            else:
                return await self._generate_generic_docs(code, analysis)
            
        except Exception as e:
            self.logger.error(f"Error generating documentation: {e}")
            return "# Documentation\n\nError generating documentation."
    
    async def _generate_python_docs(self, code: str, analysis: CodeAnalysis) -> str:
        """Generate Python documentation"""
        try:
            docs = f"""# {analysis.filename} Documentation

## Overview
This module contains {analysis.metrics.get('function_count', 0)} functions and {analysis.metrics.get('class_count', 0)} classes.

## Code Metrics
- Lines of Code: {analysis.lines_of_code}
- Complexity Score: {analysis.complexity_score:.2f}
- Quality Score: {analysis.quality_score:.2f}

## Functions and Classes

"""
            
            # Extract functions and classes (simplified)
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    func_name = line.strip().split('(')[0].replace('def ', '')
                    docs += f"### {func_name}\n\n"
                    
                    # Look for docstring
                    if i + 1 < len(lines) and '"""' in lines[i + 1]:
                        j = i + 2
                        while j < len(lines) and '"""' not in lines[j]:
                            docs += f"{lines[j].strip()}\n"
                            j += 1
                        docs += "\n"
                    else:
                        docs += "No documentation available.\n\n"
                
                elif line.strip().startswith('class '):
                    class_name = line.strip().split('(')[0].replace('class ', '').replace(':', '')
                    docs += f"## {class_name}\n\n"
                    
                    # Look for docstring
                    if i + 1 < len(lines) and '"""' in lines[i + 1]:
                        j = i + 2
                        while j < len(lines) and '"""' not in lines[j]:
                            docs += f"{lines[j].strip()}\n"
                            j += 1
                        docs += "\n"
                    else:
                        docs += "No documentation available.\n\n"
            
            return docs
            
        except Exception as e:
            self.logger.error(f"Error generating Python docs: {e}")
            return "# Documentation\n\nError generating Python documentation."
    
    async def _generate_js_docs(self, code: str, analysis: CodeAnalysis) -> str:
        """Generate JavaScript/TypeScript documentation"""
        try:
            docs = f"""# {analysis.filename} Documentation

## Overview
This module contains {analysis.metrics.get('function_count', 0)} functions and {analysis.metrics.get('class_count', 0)} classes.

## Code Metrics
- Lines of Code: {analysis.lines_of_code}
- Complexity Score: {analysis.complexity_score:.2f}
- Quality Score: {analysis.quality_score:.2f}

## Functions and Classes

"""
            
            # Extract functions and classes (simplified)
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'function ' in line or '=>' in line:
                    if 'function ' in line:
                        func_name = line.split('function ')[1].split('(')[0].strip()
                    else:
                        func_name = line.split('=>')[0].split('=')[0].strip()
                    
                    docs += f"### {func_name}\n\n"
                    
                    # Look for JSDoc
                    if i > 0 and '/**' in lines[i - 1]:
                        j = i - 1
                        while j >= 0 and '/**' not in lines[j]:
                            j -= 1
                        while j < i and '*/' not in lines[j]:
                            if lines[j].strip().startswith('*'):
                                docs += f"{lines[j].strip()[1:].strip()}\n"
                            j += 1
                        docs += "\n"
                    else:
                        docs += "No documentation available.\n\n"
                
                elif line.strip().startswith('class '):
                    class_name = line.strip().split(' ')[1].split('{')[0].strip()
                    docs += f"## {class_name}\n\n"
                    docs += "No documentation available.\n\n"
            
            return docs
            
        except Exception as e:
            self.logger.error(f"Error generating JS docs: {e}")
            return "# Documentation\n\nError generating JavaScript documentation."
    
    async def _generate_generic_docs(self, code: str, analysis: CodeAnalysis) -> str:
        """Generate generic documentation"""
        try:
            docs = f"""# {analysis.filename} Documentation

## Overview
Code written in {analysis.language.value}.

## Code Metrics
- Lines of Code: {analysis.lines_of_code}
- Complexity Score: {analysis.complexity_score:.2f}
- Quality Score: {analysis.quality_score:.2f}

## Analysis Results

### Issues Found
"""
            
            for issue in analysis.issues:
                docs += f"- Line {issue['line']}: {issue['message']} ({issue['severity']})\n"
            
            docs += "\n### Suggestions\n"
            for suggestion in analysis.suggestions:
                docs += f"- {suggestion}\n"
            
            return docs
            
        except Exception as e:
            self.logger.error(f"Error generating generic docs: {e}")
            return "# Documentation\n\nError generating documentation."
    
    async def save_generated_code(self, generated_code: GeneratedCode, output_dir: str) -> str:
        """Save generated code to file"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            file_path = output_path / generated_code.filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(generated_code.content)
            
            self.logger.info(f"Saved generated code to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving generated code: {e}")
            raise
    
    def get_templates(self, language: Optional[CodeLanguage] = None, 
                     code_type: Optional[CodeType] = None) -> List[CodeTemplate]:
        """Get available templates with optional filtering"""
        try:
            templates = list(self.templates.values())
            
            if language:
                templates = [t for t in templates if t.language == language]
            
            if code_type:
                templates = [t for t in templates if t.code_type == code_type]
            
            return templates
            
        except Exception as e:
            self.logger.error(f"Error getting templates: {e}")
            return []
    
    def get_generation_history(self, limit: int = 50) -> List[GeneratedCode]:
        """Get recent code generation history"""
        try:
            return self.generation_history[-limit:]
        except Exception as e:
            self.logger.error(f"Error getting generation history: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            stats = self.performance_stats.copy()
            
            # Calculate average quality score
            if stats['quality_scores']:
                stats['average_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            else:
                stats['average_quality_score'] = 0.0
            
            # Add template count
            stats['templates_available'] = len(self.templates)
            stats['cache_size'] = len(self.analysis_cache)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}
    
    async def export_templates(self, output_file: str) -> None:
        """Export all templates to file"""
        try:
            templates_data = {
                name: template.to_dict() 
                for name, template in self.templates.items()
            }
            
            with open(output_file, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            self.logger.info(f"Exported {len(self.templates)} templates to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting templates: {e}")
            raise
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        
        # Update specific settings
        if 'max_history' in new_config:
            self.max_history = new_config['max_history']
        
        if 'cache_ttl' in new_config:
            self.cache_ttl = new_config['cache_ttl']
        
        self.logger.info("Configuration updated")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Clear caches
            self.analysis_cache.clear()
            
            # Trim history if needed
            if len(self.generation_history) > self.max_history:
                self.generation_history = self.generation_history[-self.max_history:]
            
            self.logger.info("Code generator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
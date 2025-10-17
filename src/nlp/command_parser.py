"""
Command Parser
Converts NLP results into structured, executable commands.
Handles command validation, parameter extraction, and execution planning.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class CommandType(Enum):
    """Command types"""
    SYSTEM_CONTROL = "system_control"
    FILE_MANAGEMENT = "file_management"
    WEB_AUTOMATION = "web_automation"
    TEXT_PROCESSING = "text_processing"
    INFORMATION_QUERY = "information_query"
    CODE_ASSISTANCE = "code_assistance"
    FINANCIAL_ANALYSIS = "financial_analysis"
    AUTOMATION_TASK = "automation_task"
    VOICE_CONTROL = "voice_control"
    CUSTOM = "custom"


class ExecutionMode(Enum):
    """Command execution modes"""
    IMMEDIATE = "immediate"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    CONDITIONAL = "conditional"
    INTERACTIVE = "interactive"


@dataclass
class CommandParameter:
    """Command parameter definition"""
    name: str
    value: Any
    type: str
    required: bool = True
    validated: bool = False
    source: str = "user_input"  # user_input, context, default, inferred


@dataclass
class ExecutableCommand:
    """Structured executable command"""
    id: str
    type: CommandType
    action: str
    parameters: Dict[str, CommandParameter]
    execution_mode: ExecutionMode = ExecutionMode.IMMEDIATE
    priority: int = 5  # 1-10, 10 is highest
    timeout: float = 30.0
    retry_count: int = 3
    validation_required: bool = True
    confirmation_required: bool = False
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class AdvancedCommandParser:
    """Advanced command parser with validation and optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Command templates
        self.command_templates: Dict[str, Dict] = {}
        self.action_mappings: Dict[str, str] = {}
        self.parameter_validators: Dict[str, callable] = {}
        
        # Parsing rules
        self.parsing_rules: Dict[str, List[Dict]] = {}
        self.entity_mappings: Dict[str, str] = {}
        self.context_resolvers: Dict[str, callable] = {}
        
        # Command optimization
        self.command_patterns: Dict[str, int] = {}
        self.optimization_rules: List[Dict] = []
        
        # Storage paths
        self.templates_dir = Path("config/command_templates")
        self.templates_path = self.templates_dir / "command_templates.json"
        self.rules_path = self.templates_dir / "parsing_rules.json"
        
        # Command counter for unique IDs
        self.command_counter = 0
    
    async def initialize(self):
        """Initialize command parser"""
        try:
            self.logger.info("Initializing Command Parser...")
            
            # Create templates directory
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Load command templates
            await self._load_command_templates()
            
            # Load parsing rules
            await self._load_parsing_rules()
            
            # Initialize validators
            await self._initialize_validators()
            
            # Initialize context resolvers
            await self._initialize_context_resolvers()
            
            self.logger.info("Command Parser initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Command Parser initialization error: {e}")
            return False
    
    async def _load_command_templates(self):
        """Load command templates"""
        try:
            if self.templates_path.exists():
                with open(self.templates_path, 'r', encoding='utf-8') as f:
                    self.command_templates = json.load(f)
            else:
                await self._create_default_templates()
            
            self.logger.info(f"Loaded {len(self.command_templates)} command templates")
            
        except Exception as e:
            self.logger.error(f"Template loading error: {e}")
            await self._create_default_templates()
    
    async def _create_default_templates(self):
        """Create default command templates"""
        try:
            self.command_templates = {
                "system_control": {
                    "open_application": {
                        "action": "open_application",
                        "parameters": {
                            "application": {"type": "string", "required": True},
                            "arguments": {"type": "list", "required": False, "default": []},
                            "wait_for_load": {"type": "boolean", "required": False, "default": True}
                        },
                        "timeout": 10.0,
                        "retry_count": 2
                    },
                    "close_application": {
                        "action": "close_application",
                        "parameters": {
                            "application": {"type": "string", "required": True},
                            "force_close": {"type": "boolean", "required": False, "default": False},
                            "save_before_close": {"type": "boolean", "required": False, "default": True}
                        },
                        "timeout": 5.0,
                        "confirmation_required": True
                    },
                    "minimize_window": {
                        "action": "minimize_window",
                        "parameters": {
                            "window_title": {"type": "string", "required": False},
                            "application": {"type": "string", "required": False}
                        },
                        "timeout": 2.0
                    },
                    "maximize_window": {
                        "action": "maximize_window",
                        "parameters": {
                            "window_title": {"type": "string", "required": False},
                            "application": {"type": "string", "required": False}
                        },
                        "timeout": 2.0
                    }
                },
                "file_management": {
                    "create_file": {
                        "action": "create_file",
                        "parameters": {
                            "file_path": {"type": "path", "required": True},
                            "content": {"type": "string", "required": False, "default": ""},
                            "overwrite": {"type": "boolean", "required": False, "default": False}
                        },
                        "timeout": 5.0,
                        "validation_required": True
                    },
                    "delete_file": {
                        "action": "delete_file",
                        "parameters": {
                            "file_path": {"type": "path", "required": True},
                            "permanent": {"type": "boolean", "required": False, "default": False}
                        },
                        "timeout": 5.0,
                        "confirmation_required": True
                    },
                    "copy_file": {
                        "action": "copy_file",
                        "parameters": {
                            "source_path": {"type": "path", "required": True},
                            "destination_path": {"type": "path", "required": True},
                            "overwrite": {"type": "boolean", "required": False, "default": False}
                        },
                        "timeout": 10.0
                    },
                    "move_file": {
                        "action": "move_file",
                        "parameters": {
                            "source_path": {"type": "path", "required": True},
                            "destination_path": {"type": "path", "required": True},
                            "overwrite": {"type": "boolean", "required": False, "default": False}
                        },
                        "timeout": 10.0
                    },
                    "search_files": {
                        "action": "search_files",
                        "parameters": {
                            "search_path": {"type": "path", "required": False, "default": "."},
                            "pattern": {"type": "string", "required": True},
                            "file_type": {"type": "string", "required": False},
                            "recursive": {"type": "boolean", "required": False, "default": True}
                        },
                        "timeout": 30.0
                    }
                },
                "web_automation": {
                    "open_url": {
                        "action": "open_url",
                        "parameters": {
                            "url": {"type": "url", "required": True},
                            "browser": {"type": "string", "required": False},
                            "new_tab": {"type": "boolean", "required": False, "default": True}
                        },
                        "timeout": 10.0
                    },
                    "web_search": {
                        "action": "web_search",
                        "parameters": {
                            "query": {"type": "string", "required": True},
                            "search_engine": {"type": "string", "required": False, "default": "google"},
                            "results_count": {"type": "integer", "required": False, "default": 10}
                        },
                        "timeout": 15.0
                    },
                    "click_element": {
                        "action": "click_element",
                        "parameters": {
                            "selector": {"type": "string", "required": True},
                            "wait_for_element": {"type": "boolean", "required": False, "default": True},
                            "timeout": {"type": "float", "required": False, "default": 5.0}
                        },
                        "timeout": 10.0
                    },
                    "fill_form": {
                        "action": "fill_form",
                        "parameters": {
                            "form_data": {"type": "dict", "required": True},
                            "submit": {"type": "boolean", "required": False, "default": False}
                        },
                        "timeout": 15.0
                    }
                },
                "text_processing": {
                    "extract_text": {
                        "action": "extract_text",
                        "parameters": {
                            "source": {"type": "string", "required": True},
                            "format": {"type": "string", "required": False, "default": "plain"}
                        },
                        "timeout": 10.0
                    },
                    "translate_text": {
                        "action": "translate_text",
                        "parameters": {
                            "text": {"type": "string", "required": True},
                            "target_language": {"type": "string", "required": True},
                            "source_language": {"type": "string", "required": False, "default": "auto"}
                        },
                        "timeout": 10.0
                    },
                    "summarize_text": {
                        "action": "summarize_text",
                        "parameters": {
                            "text": {"type": "string", "required": True},
                            "max_length": {"type": "integer", "required": False, "default": 200},
                            "style": {"type": "string", "required": False, "default": "concise"}
                        },
                        "timeout": 15.0
                    }
                },
                "information_query": {
                    "get_system_info": {
                        "action": "get_system_info",
                        "parameters": {
                            "info_type": {"type": "string", "required": True}
                        },
                        "timeout": 5.0
                    },
                    "get_weather": {
                        "action": "get_weather",
                        "parameters": {
                            "location": {"type": "string", "required": True},
                            "forecast_days": {"type": "integer", "required": False, "default": 1}
                        },
                        "timeout": 10.0
                    },
                    "get_time": {
                        "action": "get_time",
                        "parameters": {
                            "timezone": {"type": "string", "required": False, "default": "local"},
                            "format": {"type": "string", "required": False, "default": "12h"}
                        },
                        "timeout": 2.0
                    }
                }
            }
            
            # Save templates
            with open(self.templates_path, 'w', encoding='utf-8') as f:
                json.dump(self.command_templates, f, indent=2)
            
            self.logger.info("Created default command templates")
            
        except Exception as e:
            self.logger.error(f"Default template creation error: {e}")
    
    async def _load_parsing_rules(self):
        """Load parsing rules"""
        try:
            if self.rules_path.exists():
                with open(self.rules_path, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                    self.parsing_rules = rules_data.get('parsing_rules', {})
                    self.entity_mappings = rules_data.get('entity_mappings', {})
                    self.action_mappings = rules_data.get('action_mappings', {})
            else:
                await self._create_default_rules()
            
            self.logger.info("Loaded parsing rules")
            
        except Exception as e:
            self.logger.error(f"Parsing rules loading error: {e}")
            await self._create_default_rules()
    
    async def _create_default_rules(self):
        """Create default parsing rules"""
        try:
            self.parsing_rules = {
                "system_control": [
                    {
                        "pattern": r"open|launch|start|run",
                        "action": "open_application",
                        "entity_mapping": {"application": "application"}
                    },
                    {
                        "pattern": r"close|exit|quit|terminate",
                        "action": "close_application",
                        "entity_mapping": {"application": "application"}
                    },
                    {
                        "pattern": r"minimize|hide",
                        "action": "minimize_window",
                        "entity_mapping": {"application": "application", "window_name": "window_title"}
                    },
                    {
                        "pattern": r"maximize|expand|fullscreen",
                        "action": "maximize_window",
                        "entity_mapping": {"application": "application", "window_name": "window_title"}
                    }
                ],
                "file_management": [
                    {
                        "pattern": r"create|make|new",
                        "action": "create_file",
                        "entity_mapping": {"file_path": "file_path", "path": "file_path"}
                    },
                    {
                        "pattern": r"delete|remove|erase",
                        "action": "delete_file",
                        "entity_mapping": {"file_path": "file_path", "path": "file_path"}
                    },
                    {
                        "pattern": r"copy|duplicate",
                        "action": "copy_file",
                        "entity_mapping": {"file_path": "source_path", "path": "source_path"}
                    },
                    {
                        "pattern": r"move|transfer|relocate",
                        "action": "move_file",
                        "entity_mapping": {"file_path": "source_path", "path": "source_path"}
                    },
                    {
                        "pattern": r"search|find|look for",
                        "action": "search_files",
                        "entity_mapping": {"query": "pattern", "file_type": "file_type"}
                    }
                ],
                "web_automation": [
                    {
                        "pattern": r"open|go to|navigate to",
                        "action": "open_url",
                        "entity_mapping": {"url": "url"}
                    },
                    {
                        "pattern": r"search|google|find",
                        "action": "web_search",
                        "entity_mapping": {"query": "query"}
                    },
                    {
                        "pattern": r"click|press|select",
                        "action": "click_element",
                        "entity_mapping": {"element": "selector"}
                    }
                ],
                "text_processing": [
                    {
                        "pattern": r"extract|get text",
                        "action": "extract_text",
                        "entity_mapping": {"source": "source"}
                    },
                    {
                        "pattern": r"translate",
                        "action": "translate_text",
                        "entity_mapping": {"text": "text", "language": "target_language"}
                    },
                    {
                        "pattern": r"summarize|summary",
                        "action": "summarize_text",
                        "entity_mapping": {"text": "text"}
                    }
                ],
                "information_query": [
                    {
                        "pattern": r"system info|computer info",
                        "action": "get_system_info",
                        "entity_mapping": {"info_type": "info_type"}
                    },
                    {
                        "pattern": r"weather|forecast",
                        "action": "get_weather",
                        "entity_mapping": {"location": "location"}
                    },
                    {
                        "pattern": r"time|clock",
                        "action": "get_time",
                        "entity_mapping": {}
                    }
                ]
            }
            
            self.entity_mappings = {
                "application": "application",
                "file_path": "file_path",
                "path": "file_path",
                "url": "url",
                "query": "query",
                "text": "text",
                "location": "location",
                "number": "number",
                "time": "time",
                "date": "date"
            }
            
            self.action_mappings = {
                "open": "open_application",
                "launch": "open_application",
                "start": "open_application",
                "run": "open_application",
                "close": "close_application",
                "exit": "close_application",
                "quit": "close_application",
                "create": "create_file",
                "make": "create_file",
                "new": "create_file",
                "delete": "delete_file",
                "remove": "delete_file",
                "search": "web_search",
                "find": "search_files",
                "copy": "copy_file",
                "move": "move_file"
            }
            
            # Save rules
            rules_data = {
                'parsing_rules': self.parsing_rules,
                'entity_mappings': self.entity_mappings,
                'action_mappings': self.action_mappings
            }
            
            with open(self.rules_path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2)
            
            self.logger.info("Created default parsing rules")
            
        except Exception as e:
            self.logger.error(f"Default rules creation error: {e}")
    
    async def _initialize_validators(self):
        """Initialize parameter validators"""
        try:
            self.parameter_validators = {
                'string': self._validate_string,
                'integer': self._validate_integer,
                'float': self._validate_float,
                'boolean': self._validate_boolean,
                'path': self._validate_path,
                'url': self._validate_url,
                'email': self._validate_email,
                'list': self._validate_list,
                'dict': self._validate_dict
            }
            
            self.logger.info("Parameter validators initialized")
            
        except Exception as e:
            self.logger.error(f"Validator initialization error: {e}")
    
    async def _initialize_context_resolvers(self):
        """Initialize context resolvers"""
        try:
            self.context_resolvers = {
                'current_directory': self._resolve_current_directory,
                'active_application': self._resolve_active_application,
                'selected_files': self._resolve_selected_files,
                'clipboard_content': self._resolve_clipboard_content,
                'default_browser': self._resolve_default_browser
            }
            
            self.logger.info("Context resolvers initialized")
            
        except Exception as e:
            self.logger.error(f"Context resolver initialization error: {e}")
    
    async def parse_command(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any], user_input: str) -> Optional[ExecutableCommand]:
        """Parse NLP result into executable command"""
        try:
            # Determine command type
            command_type = self._map_intent_to_command_type(intent)
            
            # Find matching action
            action = await self._determine_action(intent, entities, user_input)
            if not action:
                self.logger.warning(f"No action found for intent: {intent}")
                return None
            
            # Get command template
            template = self._get_command_template(command_type.value, action)
            if not template:
                self.logger.warning(f"No template found for {command_type.value}.{action}")
                return None
            
            # Extract and validate parameters
            parameters = await self._extract_parameters(template, entities, context)
            
            # Create executable command
            command = ExecutableCommand(
                id=self._generate_command_id(),
                type=command_type,
                action=action,
                parameters=parameters,
                timeout=template.get('timeout', 30.0),
                retry_count=template.get('retry_count', 3),
                validation_required=template.get('validation_required', True),
                confirmation_required=template.get('confirmation_required', False),
                metadata={
                    'original_input': user_input,
                    'intent': intent,
                    'entities': entities,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Validate command
            if await self._validate_command(command):
                self.logger.info(f"Parsed command: {command.type.value}.{command.action}")
                return command
            else:
                self.logger.warning(f"Command validation failed: {command.action}")
                return None
            
        except Exception as e:
            self.logger.error(f"Command parsing error: {e}")
            return None
    
    def _map_intent_to_command_type(self, intent: str) -> CommandType:
        """Map intent to command type"""
        mapping = {
            'system_control': CommandType.SYSTEM_CONTROL,
            'file_management': CommandType.FILE_MANAGEMENT,
            'web_search': CommandType.WEB_AUTOMATION,
            'text_processing': CommandType.TEXT_PROCESSING,
            'information_query': CommandType.INFORMATION_QUERY,
            'code_assistance': CommandType.CODE_ASSISTANCE,
            'financial_analysis': CommandType.FINANCIAL_ANALYSIS,
            'automation_task': CommandType.AUTOMATION_TASK,
            'voice_control': CommandType.VOICE_CONTROL
        }
        
        return mapping.get(intent, CommandType.CUSTOM)
    
    async def _determine_action(self, intent: str, entities: Dict[str, Any], user_input: str) -> Optional[str]:
        """Determine action from intent and entities"""
        try:
            # Check parsing rules for intent
            if intent in self.parsing_rules:
                for rule in self.parsing_rules[intent]:
                    pattern = rule['pattern']
                    if re.search(pattern, user_input, re.IGNORECASE):
                        return rule['action']
            
            # Check action entities
            if 'action' in entities:
                for action_entity in entities['action']:
                    action_text = action_entity.get('value', action_entity.get('text', '')).lower()
                    if action_text in self.action_mappings:
                        return self.action_mappings[action_text]
            
            # Default actions for intents
            default_actions = {
                'system_control': 'open_application',
                'file_management': 'search_files',
                'web_search': 'web_search',
                'text_processing': 'extract_text',
                'information_query': 'get_system_info'
            }
            
            return default_actions.get(intent)
            
        except Exception as e:
            self.logger.error(f"Action determination error: {e}")
            return None
    
    def _get_command_template(self, command_type: str, action: str) -> Optional[Dict]:
        """Get command template"""
        try:
            if command_type in self.command_templates:
                return self.command_templates[command_type].get(action)
            return None
            
        except Exception as e:
            self.logger.error(f"Template retrieval error: {e}")
            return None
    
    async def _extract_parameters(self, template: Dict, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, CommandParameter]:
        """Extract and validate command parameters"""
        try:
            parameters = {}
            template_params = template.get('parameters', {})
            
            for param_name, param_config in template_params.items():
                param_type = param_config['type']
                required = param_config.get('required', True)
                default_value = param_config.get('default')
                
                # Try to extract from entities
                value = await self._extract_parameter_from_entities(param_name, entities)
                source = "user_input"
                
                # Try to extract from context if not found in entities
                if value is None:
                    value = await self._extract_parameter_from_context(param_name, context)
                    if value is not None:
                        source = "context"
                
                # Use default value if still not found
                if value is None and default_value is not None:
                    value = default_value
                    source = "default"
                
                # Validate parameter
                if value is not None:
                    validated_value, is_valid = await self._validate_parameter(value, param_type)
                    if is_valid:
                        parameters[param_name] = CommandParameter(
                            name=param_name,
                            value=validated_value,
                            type=param_type,
                            required=required,
                            validated=True,
                            source=source
                        )
                    else:
                        self.logger.warning(f"Parameter validation failed: {param_name} = {value}")
                elif required:
                    # Try to infer missing required parameter
                    inferred_value = await self._infer_parameter(param_name, param_type, entities, context)
                    if inferred_value is not None:
                        parameters[param_name] = CommandParameter(
                            name=param_name,
                            value=inferred_value,
                            type=param_type,
                            required=required,
                            validated=False,
                            source="inferred"
                        )
                    else:
                        self.logger.warning(f"Required parameter missing: {param_name}")
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Parameter extraction error: {e}")
            return {}
    
    async def _extract_parameter_from_entities(self, param_name: str, entities: Dict[str, Any]) -> Any:
        """Extract parameter value from entities"""
        try:
            # Direct mapping
            if param_name in entities:
                entity_list = entities[param_name]
                if entity_list:
                    return entity_list[0].get('value', entity_list[0].get('text'))
            
            # Check entity mappings
            for entity_type, entity_list in entities.items():
                if entity_type in self.entity_mappings:
                    mapped_param = self.entity_mappings[entity_type]
                    if mapped_param == param_name and entity_list:
                        return entity_list[0].get('value', entity_list[0].get('text'))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Entity parameter extraction error: {e}")
            return None
    
    async def _extract_parameter_from_context(self, param_name: str, context: Dict[str, Any]) -> Any:
        """Extract parameter value from context"""
        try:
            current_state = context.get('current_state', {})
            
            # Context mappings
            context_mappings = {
                'application': 'active_application',
                'file_path': 'selected_files',
                'source_path': 'selected_files',
                'search_path': 'current_directory',
                'window_title': 'active_application'
            }
            
            if param_name in context_mappings:
                context_key = context_mappings[param_name]
                value = current_state.get(context_key)
                
                if value:
                    # Handle list values (e.g., selected_files)
                    if isinstance(value, list) and value:
                        return value[0]
                    elif not isinstance(value, list):
                        return value
            
            # Check context resolvers
            if param_name in self.context_resolvers:
                resolver = self.context_resolvers[param_name]
                return await resolver(context)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Context parameter extraction error: {e}")
            return None
    
    async def _infer_parameter(self, param_name: str, param_type: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Infer missing parameter value"""
        try:
            # Inference rules based on parameter name and type
            if param_name == 'application' and param_type == 'string':
                # Try to infer from other entities
                if 'window_name' in entities:
                    return entities['window_name'][0].get('value')
            
            elif param_name == 'destination_path' and param_type == 'path':
                # Use current directory as destination
                current_state = context.get('current_state', {})
                current_dir = current_state.get('current_directory')
                if current_dir:
                    return current_dir
            
            elif param_name == 'search_engine' and param_type == 'string':
                # Default search engine
                return 'google'
            
            elif param_name == 'browser' and param_type == 'string':
                # Get default browser from preferences
                preferences = context.get('user_preferences', {})
                return preferences.get('default_browser', {}).get('value', 'chrome')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Parameter inference error: {e}")
            return None
    
    async def _validate_parameter(self, value: Any, param_type: str) -> Tuple[Any, bool]:
        """Validate parameter value"""
        try:
            if param_type in self.parameter_validators:
                validator = self.parameter_validators[param_type]
                return await validator(value)
            else:
                # Default validation
                return value, True
                
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return value, False
    
    async def _validate_string(self, value: Any) -> Tuple[str, bool]:
        """Validate string parameter"""
        try:
            if isinstance(value, str):
                return value.strip(), len(value.strip()) > 0
            else:
                return str(value), True
        except:
            return "", False
    
    async def _validate_integer(self, value: Any) -> Tuple[int, bool]:
        """Validate integer parameter"""
        try:
            if isinstance(value, int):
                return value, True
            elif isinstance(value, str):
                return int(value), True
            elif isinstance(value, float):
                return int(value), True
            else:
                return 0, False
        except:
            return 0, False
    
    async def _validate_float(self, value: Any) -> Tuple[float, bool]:
        """Validate float parameter"""
        try:
            if isinstance(value, (int, float)):
                return float(value), True
            elif isinstance(value, str):
                return float(value), True
            else:
                return 0.0, False
        except:
            return 0.0, False
    
    async def _validate_boolean(self, value: Any) -> Tuple[bool, bool]:
        """Validate boolean parameter"""
        try:
            if isinstance(value, bool):
                return value, True
            elif isinstance(value, str):
                return value.lower() in ['true', 'yes', '1', 'on', 'enable'], True
            elif isinstance(value, (int, float)):
                return bool(value), True
            else:
                return False, False
        except:
            return False, False
    
    async def _validate_path(self, value: Any) -> Tuple[str, bool]:
        """Validate path parameter"""
        try:
            if isinstance(value, str):
                # Basic path validation
                path_str = value.strip()
                if len(path_str) > 0:
                    # Convert to Path object to validate format
                    try:
                        Path(path_str)
                        return path_str, True
                    except:
                        return path_str, False
            return "", False
        except:
            return "", False
    
    async def _validate_url(self, value: Any) -> Tuple[str, bool]:
        """Validate URL parameter"""
        try:
            if isinstance(value, str):
                url = value.strip()
                # Basic URL validation
                if url.startswith(('http://', 'https://', 'ftp://')):
                    return url, True
                elif '.' in url and ' ' not in url:
                    # Assume it's a domain without protocol
                    return f"https://{url}", True
            return "", False
        except:
            return "", False
    
    async def _validate_email(self, value: Any) -> Tuple[str, bool]:
        """Validate email parameter"""
        try:
            if isinstance(value, str):
                email = value.strip()
                # Basic email validation
                if '@' in email and '.' in email.split('@')[-1]:
                    return email, True
            return "", False
        except:
            return "", False
    
    async def _validate_list(self, value: Any) -> Tuple[List, bool]:
        """Validate list parameter"""
        try:
            if isinstance(value, list):
                return value, True
            elif isinstance(value, str):
                # Try to parse as comma-separated values
                items = [item.strip() for item in value.split(',')]
                return items, True
            else:
                return [value], True
        except:
            return [], False
    
    async def _validate_dict(self, value: Any) -> Tuple[Dict, bool]:
        """Validate dictionary parameter"""
        try:
            if isinstance(value, dict):
                return value, True
            elif isinstance(value, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return parsed, True
                except:
                    pass
            return {}, False
        except:
            return {}, False
    
    # Context resolvers
    async def _resolve_current_directory(self, context: Dict[str, Any]) -> Optional[str]:
        """Resolve current directory from context"""
        current_state = context.get('current_state', {})
        return current_state.get('current_directory')
    
    async def _resolve_active_application(self, context: Dict[str, Any]) -> Optional[str]:
        """Resolve active application from context"""
        current_state = context.get('current_state', {})
        return current_state.get('active_application')
    
    async def _resolve_selected_files(self, context: Dict[str, Any]) -> Optional[List[str]]:
        """Resolve selected files from context"""
        current_state = context.get('current_state', {})
        return current_state.get('selected_files')
    
    async def _resolve_clipboard_content(self, context: Dict[str, Any]) -> Optional[str]:
        """Resolve clipboard content from context"""
        current_state = context.get('current_state', {})
        return current_state.get('clipboard_content')
    
    async def _resolve_default_browser(self, context: Dict[str, Any]) -> Optional[str]:
        """Resolve default browser from context"""
        preferences = context.get('user_preferences', {})
        browser_pref = preferences.get('default_browser', {})
        return browser_pref.get('value', 'chrome')
    
    async def _validate_command(self, command: ExecutableCommand) -> bool:
        """Validate complete command"""
        try:
            # Check required parameters
            template = self._get_command_template(command.type.value, command.action)
            if not template:
                return False
            
            template_params = template.get('parameters', {})
            
            for param_name, param_config in template_params.items():
                if param_config.get('required', True):
                    if param_name not in command.parameters:
                        self.logger.warning(f"Required parameter missing: {param_name}")
                        return False
                    
                    param = command.parameters[param_name]
                    if not param.validated:
                        # Try to validate now
                        validated_value, is_valid = await self._validate_parameter(param.value, param.type)
                        if not is_valid:
                            self.logger.warning(f"Parameter validation failed: {param_name}")
                            return False
                        param.value = validated_value
                        param.validated = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Command validation error: {e}")
            return False
    
    def _generate_command_id(self) -> str:
        """Generate unique command ID"""
        self.command_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cmd_{timestamp}_{self.command_counter:04d}"
    
    async def optimize_command(self, command: ExecutableCommand, context: Dict[str, Any]) -> ExecutableCommand:
        """Optimize command based on context and patterns"""
        try:
            # Track command pattern
            pattern_key = f"{command.type.value}_{command.action}"
            self.command_patterns[pattern_key] = self.command_patterns.get(pattern_key, 0) + 1
            
            # Apply optimization rules
            for rule in self.optimization_rules:
                if self._matches_optimization_rule(command, rule):
                    command = self._apply_optimization_rule(command, rule)
            
            # Context-based optimizations
            if command.type == CommandType.FILE_MANAGEMENT:
                # Optimize file operations based on current directory
                current_state = context.get('current_state', {})
                current_dir = current_state.get('current_directory')
                
                if current_dir and 'file_path' in command.parameters:
                    file_path = command.parameters['file_path'].value
                    if not Path(file_path).is_absolute():
                        # Make relative path absolute
                        absolute_path = str(Path(current_dir) / file_path)
                        command.parameters['file_path'].value = absolute_path
            
            return command
            
        except Exception as e:
            self.logger.error(f"Command optimization error: {e}")
            return command
    
    def _matches_optimization_rule(self, command: ExecutableCommand, rule: Dict) -> bool:
        """Check if command matches optimization rule"""
        # Implementation would depend on rule structure
        return False
    
    def _apply_optimization_rule(self, command: ExecutableCommand, rule: Dict) -> ExecutableCommand:
        """Apply optimization rule to command"""
        # Implementation would depend on rule structure
        return command
    
    async def cleanup(self):
        """Cleanup command parser"""
        self.logger.info("Cleaning up Command Parser...")
        
        # Save command patterns for future optimization
        patterns_path = self.templates_dir / "command_patterns.json"
        with open(patterns_path, 'w', encoding='utf-8') as f:
            json.dump(self.command_patterns, f, indent=2)
        
        self.logger.info("Command Parser cleanup completed")


# Alias for backward compatibility
CommandParser = AdvancedCommandParser
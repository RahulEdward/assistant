"""
Configuration Manager for Desktop Assistant
Handles all application settings, user preferences, and system configurations.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging


@dataclass
class VoiceConfig:
    """Voice processing configuration"""
    stt_engine: str = "custom"
    tts_engine: str = "custom"
    voice_activation_threshold: float = 0.7
    noise_reduction: bool = True
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class NLPConfig:
    """Natural Language Processing configuration"""
    model_path: str = "models/nlp"
    accuracy_threshold: float = 0.95
    context_window: int = 512
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class SystemConfig:
    """System automation configuration"""
    response_timeout: float = 1.0
    max_retries: int = 3
    automation_delay: float = 0.1
    screen_capture_quality: int = 95
    ocr_confidence_threshold: float = 0.8


@dataclass
class AIIntegrationConfig:
    """External AI integration configuration"""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    max_requests_per_minute: int = 60
    fallback_model: str = "local"


@dataclass
class AppConfig:
    """Main application configuration"""
    voice: VoiceConfig
    nlp: NLPConfig
    system: SystemConfig
    ai_integration: AIIntegrationConfig
    debug_mode: bool = False
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"


class ConfigManager:
    """Manages application configuration and settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Set default config path
        if config_path is None:
            self.config_path = Path("config/config.yaml")
        else:
            self.config_path = Path(config_path)
            
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create default configuration
        self.config = self._load_config()
        
    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    
                # Convert nested dicts to dataclass instances
                voice_config = VoiceConfig(**config_data.get('voice', {}))
                nlp_config = NLPConfig(**config_data.get('nlp', {}))
                system_config = SystemConfig(**config_data.get('system', {}))
                ai_config = AIIntegrationConfig(**config_data.get('ai_integration', {}))
                
                return AppConfig(
                    voice=voice_config,
                    nlp=nlp_config,
                    system=system_config,
                    ai_integration=ai_config,
                    debug_mode=config_data.get('debug_mode', False),
                    log_level=config_data.get('log_level', 'INFO'),
                    data_dir=config_data.get('data_dir', 'data'),
                    models_dir=config_data.get('models_dir', 'models')
                )
            else:
                # Create default configuration
                config = AppConfig(
                    voice=VoiceConfig(),
                    nlp=NLPConfig(),
                    system=SystemConfig(),
                    ai_integration=AIIntegrationConfig()
                )
                self._save_config(config)
                return config
                
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            # Return default config on error
            return AppConfig(
                voice=VoiceConfig(),
                nlp=NLPConfig(),
                system=SystemConfig(),
                ai_integration=AIIntegrationConfig()
            )
    
    def _save_config(self, config: AppConfig):
        """Save configuration to file"""
        try:
            config_dict = asdict(config)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values"""
        try:
            # Update configuration attributes
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Save updated configuration
            self._save_config(self.config)
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
    
    def get_data_dir(self) -> Path:
        """Get data directory path"""
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_models_dir(self) -> Path:
        """Get models directory path"""
        models_dir = Path(self.config.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    def get_logs_dir(self) -> Path:
        """Get logs directory path"""
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        try:
            # Check required directories
            self.get_data_dir()
            self.get_models_dir()
            self.get_logs_dir()
            
            # Validate thresholds
            if not (0.0 <= self.config.voice.voice_activation_threshold <= 1.0):
                self.logger.warning("Invalid voice activation threshold")
                return False
                
            if not (0.0 <= self.config.nlp.accuracy_threshold <= 1.0):
                self.logger.warning("Invalid NLP accuracy threshold")
                return False
                
            if not (0.0 <= self.config.system.ocr_confidence_threshold <= 1.0):
                self.logger.warning("Invalid OCR confidence threshold")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Config validation error: {e}")
            return False
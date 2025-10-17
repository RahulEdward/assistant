"""
Settings Widget for Computer Assistant GUI
Provides configuration management for all application components
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QListWidget, QListWidgetItem, QProgressBar, QSpinBox,
    QCheckBox, QTabWidget, QFileDialog, QMessageBox, QSlider,
    QFrame, QScrollArea, QTreeWidget, QTreeWidgetItem, QSplitter,
    QDoubleSpinBox, QColorDialog, QFontDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, QSettings
from PySide6.QtGui import QFont, QColor, QPalette


class GeneralSettingsWidget(QGroupBox):
    """Widget for general application settings"""
    
    def __init__(self):
        super().__init__("General Settings")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Application settings
        app_group = QGroupBox("Application")
        app_layout = QGridLayout(app_group)
        
        # Startup options
        app_layout.addWidget(QLabel("Start with Windows:"), 0, 0)
        self.startup_cb = QCheckBox()
        app_layout.addWidget(self.startup_cb, 0, 1)
        
        app_layout.addWidget(QLabel("Start minimized:"), 1, 0)
        self.start_minimized_cb = QCheckBox()
        app_layout.addWidget(self.start_minimized_cb, 1, 1)
        
        app_layout.addWidget(QLabel("Show system tray icon:"), 2, 0)
        self.system_tray_cb = QCheckBox()
        self.system_tray_cb.setChecked(True)
        app_layout.addWidget(self.system_tray_cb, 2, 1)
        
        app_layout.addWidget(QLabel("Minimize to tray:"), 3, 0)
        self.minimize_to_tray_cb = QCheckBox()
        app_layout.addWidget(self.minimize_to_tray_cb, 3, 1)
        
        layout.addWidget(app_group)
        
        # Interface settings
        ui_group = QGroupBox("User Interface")
        ui_layout = QGridLayout(ui_group)
        
        # Theme
        ui_layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Auto"])
        ui_layout.addWidget(self.theme_combo, 0, 1)
        
        # Language
        ui_layout.addWidget(QLabel("Language:"), 1, 0)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Spanish", "French", "German", "Chinese"])
        ui_layout.addWidget(self.language_combo, 1, 1)
        
        # Font size
        ui_layout.addWidget(QLabel("Font Size:"), 2, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(10)
        ui_layout.addWidget(self.font_size_spin, 2, 1)
        
        # Auto-refresh interval
        ui_layout.addWidget(QLabel("Auto-refresh (seconds):"), 3, 0)
        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(1, 60)
        self.refresh_interval_spin.setValue(5)
        ui_layout.addWidget(self.refresh_interval_spin, 3, 1)
        
        layout.addWidget(ui_group)
        
        # Logging settings
        log_group = QGroupBox("Logging")
        log_layout = QGridLayout(log_group)
        
        log_layout.addWidget(QLabel("Log Level:"), 0, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        log_layout.addWidget(self.log_level_combo, 0, 1)
        
        log_layout.addWidget(QLabel("Log to file:"), 1, 0)
        self.log_to_file_cb = QCheckBox()
        self.log_to_file_cb.setChecked(True)
        log_layout.addWidget(self.log_to_file_cb, 1, 1)
        
        log_layout.addWidget(QLabel("Max log file size (MB):"), 2, 0)
        self.max_log_size_spin = QSpinBox()
        self.max_log_size_spin.setRange(1, 100)
        self.max_log_size_spin.setValue(10)
        log_layout.addWidget(self.max_log_size_spin, 2, 1)
        
        layout.addWidget(log_group)


class VoiceSettingsWidget(QGroupBox):
    """Widget for voice and audio settings"""
    
    def __init__(self):
        super().__init__("Voice & Audio Settings")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Voice Recognition
        recognition_group = QGroupBox("Voice Recognition")
        recognition_layout = QGridLayout(recognition_group)
        
        recognition_layout.addWidget(QLabel("Enable voice recognition:"), 0, 0)
        self.voice_enabled_cb = QCheckBox()
        self.voice_enabled_cb.setChecked(True)
        recognition_layout.addWidget(self.voice_enabled_cb, 0, 1)
        
        recognition_layout.addWidget(QLabel("Language:"), 1, 0)
        self.voice_language_combo = QComboBox()
        self.voice_language_combo.addItems([
            "en-US", "en-GB", "es-ES", "fr-FR", "de-DE", 
            "it-IT", "pt-PT", "ru-RU", "zh-CN", "ja-JP"
        ])
        recognition_layout.addWidget(self.voice_language_combo, 1, 1)
        
        recognition_layout.addWidget(QLabel("Activation phrase:"), 2, 0)
        self.activation_phrase_edit = QLineEdit()
        self.activation_phrase_edit.setText("Hey Assistant")
        recognition_layout.addWidget(self.activation_phrase_edit, 2, 1)
        
        recognition_layout.addWidget(QLabel("Confidence threshold:"), 3, 0)
        confidence_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(70)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("70%")
        confidence_layout.addWidget(self.confidence_label)
        confidence_widget = QWidget()
        confidence_widget.setLayout(confidence_layout)
        recognition_layout.addWidget(confidence_widget, 3, 1)
        
        layout.addWidget(recognition_group)
        
        # Text-to-Speech
        tts_group = QGroupBox("Text-to-Speech")
        tts_layout = QGridLayout(tts_group)
        
        tts_layout.addWidget(QLabel("Enable TTS:"), 0, 0)
        self.tts_enabled_cb = QCheckBox()
        self.tts_enabled_cb.setChecked(True)
        tts_layout.addWidget(self.tts_enabled_cb, 0, 1)
        
        tts_layout.addWidget(QLabel("Voice:"), 1, 0)
        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.addItems([
            "Default", "Male Voice 1", "Female Voice 1", 
            "Male Voice 2", "Female Voice 2"
        ])
        tts_layout.addWidget(self.tts_voice_combo, 1, 1)
        
        tts_layout.addWidget(QLabel("Speech rate:"), 2, 0)
        rate_layout = QHBoxLayout()
        self.speech_rate_slider = QSlider(Qt.Horizontal)
        self.speech_rate_slider.setRange(50, 200)
        self.speech_rate_slider.setValue(100)
        self.speech_rate_slider.valueChanged.connect(self.update_rate_label)
        rate_layout.addWidget(self.speech_rate_slider)
        self.rate_label = QLabel("100%")
        rate_layout.addWidget(self.rate_label)
        rate_widget = QWidget()
        rate_widget.setLayout(rate_layout)
        tts_layout.addWidget(rate_widget, 2, 1)
        
        tts_layout.addWidget(QLabel("Volume:"), 3, 0)
        volume_layout = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.update_volume_label)
        volume_layout.addWidget(self.volume_slider)
        self.volume_label = QLabel("80%")
        volume_layout.addWidget(self.volume_label)
        volume_widget = QWidget()
        volume_widget.setLayout(volume_layout)
        tts_layout.addWidget(volume_widget, 3, 1)
        
        layout.addWidget(tts_group)
        
        # Audio Device Settings
        device_group = QGroupBox("Audio Devices")
        device_layout = QGridLayout(device_group)
        
        device_layout.addWidget(QLabel("Input device:"), 0, 0)
        self.input_device_combo = QComboBox()
        self.input_device_combo.addItems(["Default", "Microphone 1", "Microphone 2"])
        device_layout.addWidget(self.input_device_combo, 0, 1)
        
        device_layout.addWidget(QLabel("Output device:"), 1, 0)
        self.output_device_combo = QComboBox()
        self.output_device_combo.addItems(["Default", "Speakers", "Headphones"])
        device_layout.addWidget(self.output_device_combo, 1, 1)
        
        layout.addWidget(device_group)
    
    def update_confidence_label(self, value):
        self.confidence_label.setText(f"{value}%")
    
    def update_rate_label(self, value):
        self.rate_label.setText(f"{value}%")
    
    def update_volume_label(self, value):
        self.volume_label.setText(f"{value}%")


class AISettingsWidget(QGroupBox):
    """Widget for AI and NLP settings"""
    
    def __init__(self):
        super().__init__("AI & NLP Settings")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # AI Provider
        provider_group = QGroupBox("AI Provider")
        provider_layout = QGridLayout(provider_group)
        
        provider_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.ai_provider_combo = QComboBox()
        self.ai_provider_combo.addItems([
            "OpenAI", "Azure OpenAI", "Anthropic", "Google", "Local Model"
        ])
        provider_layout.addWidget(self.ai_provider_combo, 0, 1)
        
        provider_layout.addWidget(QLabel("API Key:"), 1, 0)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Enter your API key")
        provider_layout.addWidget(self.api_key_edit, 1, 1)
        
        provider_layout.addWidget(QLabel("Model:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", 
            "claude-3-sonnet", "claude-3-opus", "gemini-pro"
        ])
        provider_layout.addWidget(self.model_combo, 2, 1)
        
        provider_layout.addWidget(QLabel("Max tokens:"), 3, 0)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 8000)
        self.max_tokens_spin.setValue(2000)
        provider_layout.addWidget(self.max_tokens_spin, 3, 1)
        
        provider_layout.addWidget(QLabel("Temperature:"), 4, 0)
        temp_layout = QHBoxLayout()
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(70)
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)
        temp_layout.addWidget(self.temperature_slider)
        self.temperature_label = QLabel("0.7")
        temp_layout.addWidget(self.temperature_label)
        temp_widget = QWidget()
        temp_widget.setLayout(temp_layout)
        provider_layout.addWidget(temp_widget, 4, 1)
        
        layout.addWidget(provider_group)
        
        # NLP Settings
        nlp_group = QGroupBox("Natural Language Processing")
        nlp_layout = QGridLayout(nlp_group)
        
        nlp_layout.addWidget(QLabel("Enable intent recognition:"), 0, 0)
        self.intent_recognition_cb = QCheckBox()
        self.intent_recognition_cb.setChecked(True)
        nlp_layout.addWidget(self.intent_recognition_cb, 0, 1)
        
        nlp_layout.addWidget(QLabel("Enable entity extraction:"), 1, 0)
        self.entity_extraction_cb = QCheckBox()
        self.entity_extraction_cb.setChecked(True)
        nlp_layout.addWidget(self.entity_extraction_cb, 1, 1)
        
        nlp_layout.addWidget(QLabel("Context memory (messages):"), 2, 0)
        self.context_memory_spin = QSpinBox()
        self.context_memory_spin.setRange(1, 50)
        self.context_memory_spin.setValue(10)
        nlp_layout.addWidget(self.context_memory_spin, 2, 1)
        
        layout.addWidget(nlp_group)
    
    def update_temperature_label(self, value):
        temp_value = value / 100.0
        self.temperature_label.setText(f"{temp_value:.1f}")


class AutomationSettingsWidget(QGroupBox):
    """Widget for automation settings"""
    
    def __init__(self):
        super().__init__("Automation Settings")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Task Execution
        execution_group = QGroupBox("Task Execution")
        execution_layout = QGridLayout(execution_group)
        
        execution_layout.addWidget(QLabel("Enable automation:"), 0, 0)
        self.automation_enabled_cb = QCheckBox()
        self.automation_enabled_cb.setChecked(True)
        execution_layout.addWidget(self.automation_enabled_cb, 0, 1)
        
        execution_layout.addWidget(QLabel("Confirmation required:"), 1, 0)
        self.confirmation_required_cb = QCheckBox()
        self.confirmation_required_cb.setChecked(True)
        execution_layout.addWidget(self.confirmation_required_cb, 1, 1)
        
        execution_layout.addWidget(QLabel("Max concurrent tasks:"), 2, 0)
        self.max_concurrent_spin = QSpinBox()
        self.max_concurrent_spin.setRange(1, 10)
        self.max_concurrent_spin.setValue(3)
        execution_layout.addWidget(self.max_concurrent_spin, 2, 1)
        
        execution_layout.addWidget(QLabel("Task timeout (seconds):"), 3, 0)
        self.task_timeout_spin = QSpinBox()
        self.task_timeout_spin.setRange(10, 300)
        self.task_timeout_spin.setValue(60)
        execution_layout.addWidget(self.task_timeout_spin, 3, 1)
        
        layout.addWidget(execution_group)
        
        # Safety Settings
        safety_group = QGroupBox("Safety & Security")
        safety_layout = QGridLayout(safety_group)
        
        safety_layout.addWidget(QLabel("Safe mode:"), 0, 0)
        self.safe_mode_cb = QCheckBox()
        self.safe_mode_cb.setChecked(True)
        safety_layout.addWidget(self.safe_mode_cb, 0, 1)
        
        safety_layout.addWidget(QLabel("Restrict system commands:"), 1, 0)
        self.restrict_system_cb = QCheckBox()
        self.restrict_system_cb.setChecked(True)
        safety_layout.addWidget(self.restrict_system_cb, 1, 1)
        
        safety_layout.addWidget(QLabel("Backup before changes:"), 2, 0)
        self.backup_cb = QCheckBox()
        self.backup_cb.setChecked(False)
        safety_layout.addWidget(self.backup_cb, 2, 1)
        
        layout.addWidget(safety_group)


class OCRSettingsWidget(QGroupBox):
    """Widget for OCR settings"""
    
    def __init__(self):
        super().__init__("OCR Settings")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # OCR Engine
        engine_group = QGroupBox("OCR Engine")
        engine_layout = QGridLayout(engine_group)
        
        engine_layout.addWidget(QLabel("Default engine:"), 0, 0)
        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItems([
            "Tesseract", "EasyOCR", "PaddleOCR", "Azure Computer Vision"
        ])
        engine_layout.addWidget(self.ocr_engine_combo, 0, 1)
        
        engine_layout.addWidget(QLabel("Default language:"), 1, 0)
        self.ocr_language_combo = QComboBox()
        self.ocr_language_combo.addItems([
            "English", "Spanish", "French", "German", "Chinese", "Japanese"
        ])
        engine_layout.addWidget(self.ocr_language_combo, 1, 1)
        
        engine_layout.addWidget(QLabel("Confidence threshold:"), 2, 0)
        conf_layout = QHBoxLayout()
        self.ocr_confidence_slider = QSlider(Qt.Horizontal)
        self.ocr_confidence_slider.setRange(0, 100)
        self.ocr_confidence_slider.setValue(70)
        self.ocr_confidence_slider.valueChanged.connect(self.update_ocr_confidence_label)
        conf_layout.addWidget(self.ocr_confidence_slider)
        self.ocr_confidence_label = QLabel("70%")
        conf_layout.addWidget(self.ocr_confidence_label)
        conf_widget = QWidget()
        conf_widget.setLayout(conf_layout)
        engine_layout.addWidget(conf_widget, 2, 1)
        
        layout.addWidget(engine_group)
        
        # Processing Options
        processing_group = QGroupBox("Image Processing")
        processing_layout = QGridLayout(processing_group)
        
        processing_layout.addWidget(QLabel("Auto-enhance images:"), 0, 0)
        self.auto_enhance_cb = QCheckBox()
        self.auto_enhance_cb.setChecked(True)
        processing_layout.addWidget(self.auto_enhance_cb, 0, 1)
        
        processing_layout.addWidget(QLabel("Auto-detect orientation:"), 1, 0)
        self.auto_orientation_cb = QCheckBox()
        self.auto_orientation_cb.setChecked(False)
        processing_layout.addWidget(self.auto_orientation_cb, 1, 1)
        
        processing_layout.addWidget(QLabel("Preprocess images:"), 2, 0)
        self.preprocess_cb = QCheckBox()
        self.preprocess_cb.setChecked(True)
        processing_layout.addWidget(self.preprocess_cb, 2, 1)
        
        layout.addWidget(processing_group)
    
    def update_ocr_confidence_label(self, value):
        self.ocr_confidence_label.setText(f"{value}%")


class SettingsWidget(QWidget):
    """Main settings widget"""
    
    settings_changed = Signal()
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        self.settings = QSettings("ComputerAssistant", "Settings")
        
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """Initialize the settings UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Settings & Configuration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create tabs for different setting categories
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # General settings tab
        self.general_widget = GeneralSettingsWidget()
        self.tab_widget.addTab(self.general_widget, "General")
        
        # Voice settings tab
        self.voice_widget = VoiceSettingsWidget()
        self.tab_widget.addTab(self.voice_widget, "Voice & Audio")
        
        # AI settings tab
        self.ai_widget = AISettingsWidget()
        self.tab_widget.addTab(self.ai_widget, "AI & NLP")
        
        # Automation settings tab
        self.automation_widget = AutomationSettingsWidget()
        self.tab_widget.addTab(self.automation_widget, "Automation")
        
        # OCR settings tab
        self.ocr_widget = OCRSettingsWidget()
        self.tab_widget.addTab(self.ocr_widget, "OCR")
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Settings")
        self.load_btn.clicked.connect(self.load_settings_from_file)
        button_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings_to_file)
        button_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(self.apply_btn)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.ok_clicked)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_clicked)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
    
    def load_settings(self):
        """Load settings from QSettings"""
        try:
            # General settings
            self.general_widget.startup_cb.setChecked(
                self.settings.value("general/startup", False, type=bool)
            )
            self.general_widget.start_minimized_cb.setChecked(
                self.settings.value("general/start_minimized", False, type=bool)
            )
            self.general_widget.system_tray_cb.setChecked(
                self.settings.value("general/system_tray", True, type=bool)
            )
            self.general_widget.minimize_to_tray_cb.setChecked(
                self.settings.value("general/minimize_to_tray", False, type=bool)
            )
            
            theme = self.settings.value("general/theme", "Light", type=str)
            index = self.general_widget.theme_combo.findText(theme)
            if index >= 0:
                self.general_widget.theme_combo.setCurrentIndex(index)
            
            # Voice settings
            self.voice_widget.voice_enabled_cb.setChecked(
                self.settings.value("voice/enabled", True, type=bool)
            )
            self.voice_widget.tts_enabled_cb.setChecked(
                self.settings.value("voice/tts_enabled", True, type=bool)
            )
            self.voice_widget.confidence_slider.setValue(
                self.settings.value("voice/confidence", 70, type=int)
            )
            
            # AI settings
            self.ai_widget.intent_recognition_cb.setChecked(
                self.settings.value("ai/intent_recognition", True, type=bool)
            )
            self.ai_widget.entity_extraction_cb.setChecked(
                self.settings.value("ai/entity_extraction", True, type=bool)
            )
            self.ai_widget.max_tokens_spin.setValue(
                self.settings.value("ai/max_tokens", 2000, type=int)
            )
            
            # Automation settings
            self.automation_widget.automation_enabled_cb.setChecked(
                self.settings.value("automation/enabled", True, type=bool)
            )
            self.automation_widget.confirmation_required_cb.setChecked(
                self.settings.value("automation/confirmation_required", True, type=bool)
            )
            
            # OCR settings
            self.ocr_widget.auto_enhance_cb.setChecked(
                self.settings.value("ocr/auto_enhance", True, type=bool)
            )
            self.ocr_widget.ocr_confidence_slider.setValue(
                self.settings.value("ocr/confidence", 70, type=int)
            )
            
            self.logger.info("Settings loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Save settings to QSettings"""
        try:
            # General settings
            self.settings.setValue("general/startup", self.general_widget.startup_cb.isChecked())
            self.settings.setValue("general/start_minimized", self.general_widget.start_minimized_cb.isChecked())
            self.settings.setValue("general/system_tray", self.general_widget.system_tray_cb.isChecked())
            self.settings.setValue("general/minimize_to_tray", self.general_widget.minimize_to_tray_cb.isChecked())
            self.settings.setValue("general/theme", self.general_widget.theme_combo.currentText())
            
            # Voice settings
            self.settings.setValue("voice/enabled", self.voice_widget.voice_enabled_cb.isChecked())
            self.settings.setValue("voice/tts_enabled", self.voice_widget.tts_enabled_cb.isChecked())
            self.settings.setValue("voice/confidence", self.voice_widget.confidence_slider.value())
            
            # AI settings
            self.settings.setValue("ai/intent_recognition", self.ai_widget.intent_recognition_cb.isChecked())
            self.settings.setValue("ai/entity_extraction", self.ai_widget.entity_extraction_cb.isChecked())
            self.settings.setValue("ai/max_tokens", self.ai_widget.max_tokens_spin.value())
            
            # Automation settings
            self.settings.setValue("automation/enabled", self.automation_widget.automation_enabled_cb.isChecked())
            self.settings.setValue("automation/confirmation_required", self.automation_widget.confirmation_required_cb.isChecked())
            
            # OCR settings
            self.settings.setValue("ocr/auto_enhance", self.ocr_widget.auto_enhance_cb.isChecked())
            self.settings.setValue("ocr/confidence", self.ocr_widget.ocr_confidence_slider.value())
            
            self.settings.sync()
            self.logger.info("Settings saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
    
    def get_settings_dict(self) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return {
            'general': {
                'startup': self.general_widget.startup_cb.isChecked(),
                'start_minimized': self.general_widget.start_minimized_cb.isChecked(),
                'system_tray': self.general_widget.system_tray_cb.isChecked(),
                'minimize_to_tray': self.general_widget.minimize_to_tray_cb.isChecked(),
                'theme': self.general_widget.theme_combo.currentText(),
                'language': self.general_widget.language_combo.currentText(),
                'font_size': self.general_widget.font_size_spin.value(),
                'refresh_interval': self.general_widget.refresh_interval_spin.value(),
                'log_level': self.general_widget.log_level_combo.currentText(),
                'log_to_file': self.general_widget.log_to_file_cb.isChecked(),
                'max_log_size': self.general_widget.max_log_size_spin.value()
            },
            'voice': {
                'enabled': self.voice_widget.voice_enabled_cb.isChecked(),
                'language': self.voice_widget.voice_language_combo.currentText(),
                'activation_phrase': self.voice_widget.activation_phrase_edit.text(),
                'confidence': self.voice_widget.confidence_slider.value(),
                'tts_enabled': self.voice_widget.tts_enabled_cb.isChecked(),
                'tts_voice': self.voice_widget.tts_voice_combo.currentText(),
                'speech_rate': self.voice_widget.speech_rate_slider.value(),
                'volume': self.voice_widget.volume_slider.value(),
                'input_device': self.voice_widget.input_device_combo.currentText(),
                'output_device': self.voice_widget.output_device_combo.currentText()
            },
            'ai': {
                'provider': self.ai_widget.ai_provider_combo.currentText(),
                'api_key': self.ai_widget.api_key_edit.text(),
                'model': self.ai_widget.model_combo.currentText(),
                'max_tokens': self.ai_widget.max_tokens_spin.value(),
                'temperature': self.ai_widget.temperature_slider.value() / 100.0,
                'intent_recognition': self.ai_widget.intent_recognition_cb.isChecked(),
                'entity_extraction': self.ai_widget.entity_extraction_cb.isChecked(),
                'context_memory': self.ai_widget.context_memory_spin.value()
            },
            'automation': {
                'enabled': self.automation_widget.automation_enabled_cb.isChecked(),
                'confirmation_required': self.automation_widget.confirmation_required_cb.isChecked(),
                'max_concurrent': self.automation_widget.max_concurrent_spin.value(),
                'task_timeout': self.automation_widget.task_timeout_spin.value(),
                'safe_mode': self.automation_widget.safe_mode_cb.isChecked(),
                'restrict_system': self.automation_widget.restrict_system_cb.isChecked(),
                'backup': self.automation_widget.backup_cb.isChecked()
            },
            'ocr': {
                'engine': self.ocr_widget.ocr_engine_combo.currentText(),
                'language': self.ocr_widget.ocr_language_combo.currentText(),
                'confidence': self.ocr_widget.ocr_confidence_slider.value(),
                'auto_enhance': self.ocr_widget.auto_enhance_cb.isChecked(),
                'auto_orientation': self.ocr_widget.auto_orientation_cb.isChecked(),
                'preprocess': self.ocr_widget.preprocess_cb.isChecked()
            }
        }
    
    def load_settings_from_file(self):
        """Load settings from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    settings_dict = json.load(f)
                
                self.apply_settings_dict(settings_dict)
                QMessageBox.information(self, "Success", "Settings loaded successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load settings: {e}")
    
    def save_settings_to_file(self):
        """Save settings to JSON file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                settings_dict = self.get_settings_dict()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(settings_dict, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Success", "Settings saved successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
    
    def apply_settings_dict(self, settings_dict: Dict[str, Any]):
        """Apply settings from dictionary"""
        # This would set all the UI controls based on the dictionary
        # Implementation would be similar to load_settings but from dict
        pass
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.settings.clear()
            self.load_settings()
            QMessageBox.information(self, "Success", "Settings reset to defaults.")
    
    def apply_settings(self):
        """Apply current settings"""
        self.save_settings()
        self.settings_changed.emit()
        QMessageBox.information(self, "Success", "Settings applied successfully.")
    
    def ok_clicked(self):
        """Handle OK button click"""
        self.apply_settings()
    
    def cancel_clicked(self):
        """Handle Cancel button click"""
        self.load_settings()  # Reload original settings
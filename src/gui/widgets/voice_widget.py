"""
Voice Control Widget for Computer Assistant GUI
Provides interface for speech recognition, text-to-speech, and voice commands
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QListWidget, QListWidgetItem, QProgressBar, QSpinBox,
    QCheckBox, QSlider, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QIcon, QPixmap


class VoiceRecognitionThread(QThread):
    """Thread for voice recognition processing"""
    
    recognition_started = Signal()
    recognition_stopped = Signal()
    text_recognized = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, voice_processor):
        super().__init__()
        self.voice_processor = voice_processor
        self.is_listening = False
        
    def run(self):
        """Run voice recognition"""
        try:
            self.is_listening = True
            self.recognition_started.emit()
            
            # Simulate voice recognition
            # In real implementation, this would use the voice processor
            import time
            time.sleep(2)  # Simulate listening time
            
            if self.is_listening:
                # Simulate recognized text
                self.text_recognized.emit("Hello, how can I help you?")
            
            self.recognition_stopped.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.is_listening = False
    
    def stop_listening(self):
        """Stop voice recognition"""
        self.is_listening = False


class VoiceControlsWidget(QGroupBox):
    """Widget for voice control buttons and settings"""
    
    start_listening = Signal()
    stop_listening = Signal()
    speak_text = Signal(str)
    
    def __init__(self):
        super().__init__("Voice Controls")
        self.is_listening = False
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Main control buttons
        button_layout = QHBoxLayout()
        
        self.listen_btn = QPushButton("Start Listening")
        self.listen_btn.clicked.connect(self.toggle_listening)
        self.listen_btn.setMinimumHeight(50)
        button_layout.addWidget(self.listen_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_voice_recognition)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(50)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Voice settings
        settings_group = QGroupBox("Voice Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Microphone sensitivity
        settings_layout.addWidget(QLabel("Microphone Sensitivity:"), 0, 0)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        settings_layout.addWidget(self.sensitivity_slider, 0, 1)
        
        self.sensitivity_label = QLabel("5")
        settings_layout.addWidget(self.sensitivity_label, 0, 2)
        
        # Language selection
        settings_layout.addWidget(QLabel("Language:"), 1, 0)
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "English (US)",
            "English (UK)",
            "Spanish",
            "French",
            "German",
            "Chinese",
            "Japanese"
        ])
        settings_layout.addWidget(self.language_combo, 1, 1, 1, 2)
        
        # Voice activation
        self.voice_activation_cb = QCheckBox("Voice Activation")
        self.voice_activation_cb.setChecked(True)
        settings_layout.addWidget(self.voice_activation_cb, 2, 0, 1, 3)
        
        # Continuous listening
        self.continuous_cb = QCheckBox("Continuous Listening")
        settings_layout.addWidget(self.continuous_cb, 3, 0, 1, 3)
        
        layout.addWidget(settings_group)
        
        # Text-to-Speech section
        tts_group = QGroupBox("Text-to-Speech")
        tts_layout = QVBoxLayout(tts_group)
        
        # Text input
        self.tts_text_edit = QTextEdit()
        self.tts_text_edit.setPlaceholderText("Enter text to speak...")
        self.tts_text_edit.setMaximumHeight(80)
        tts_layout.addWidget(self.tts_text_edit)
        
        # TTS controls
        tts_control_layout = QHBoxLayout()
        
        self.speak_btn = QPushButton("Speak")
        self.speak_btn.clicked.connect(self.speak_text_input)
        tts_control_layout.addWidget(self.speak_btn)
        
        self.clear_tts_btn = QPushButton("Clear")
        self.clear_tts_btn.clicked.connect(self.clear_tts_text)
        tts_control_layout.addWidget(self.clear_tts_btn)
        
        tts_layout.addLayout(tts_control_layout)
        
        # TTS settings
        tts_settings_layout = QGridLayout()
        
        tts_settings_layout.addWidget(QLabel("Voice:"), 0, 0)
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["Default", "Male", "Female", "Robotic"])
        tts_settings_layout.addWidget(self.voice_combo, 0, 1)
        
        tts_settings_layout.addWidget(QLabel("Speed:"), 1, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.update_speed)
        tts_settings_layout.addWidget(self.speed_slider, 1, 1)
        
        self.speed_label = QLabel("5")
        tts_settings_layout.addWidget(self.speed_label, 1, 2)
        
        tts_layout.addLayout(tts_settings_layout)
        layout.addWidget(tts_group)
    
    def toggle_listening(self):
        """Toggle voice listening state"""
        if not self.is_listening:
            self.start_listening.emit()
            self.listen_btn.setText("Listening...")
            self.listen_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.is_listening = True
        else:
            self.stop_listening.emit()
    
    def stop_voice_recognition(self):
        """Stop voice recognition"""
        self.stop_listening.emit()
        self.listen_btn.setText("Start Listening")
        self.listen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.is_listening = False
    
    def speak_text_input(self):
        """Speak the text in the input field"""
        text = self.tts_text_edit.toPlainText().strip()
        if text:
            self.speak_text.emit(text)
    
    def clear_tts_text(self):
        """Clear the TTS text input"""
        self.tts_text_edit.clear()
    
    def update_sensitivity(self, value):
        """Update microphone sensitivity display"""
        self.sensitivity_label.setText(str(value))
    
    def update_speed(self, value):
        """Update TTS speed display"""
        self.speed_label.setText(str(value))


class CommandHistoryWidget(QGroupBox):
    """Widget showing voice command history"""
    
    def __init__(self):
        super().__init__("Command History")
        self.commands = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Command list
        self.command_list = QListWidget()
        self.command_list.setMaximumHeight(200)
        layout.addWidget(self.command_list)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self.clear_history)
        button_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("Export History")
        self.export_btn.clicked.connect(self.export_history)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
    
    def add_command(self, command: str, response: str = ""):
        """Add a command to the history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if response:
            item_text = f"[{timestamp}] Command: {command} → Response: {response}"
        else:
            item_text = f"[{timestamp}] Command: {command}"
        
        item = QListWidgetItem(item_text)
        self.command_list.insertItem(0, item)
        
        # Keep only last 100 commands
        if self.command_list.count() > 100:
            self.command_list.takeItem(self.command_list.count() - 1)
        
        # Store in internal list
        self.commands.insert(0, {
            'timestamp': timestamp,
            'command': command,
            'response': response
        })
        if len(self.commands) > 100:
            self.commands.pop()
    
    def clear_history(self):
        """Clear command history"""
        self.command_list.clear()
        self.commands.clear()
    
    def export_history(self):
        """Export command history to file"""
        if not self.commands:
            return
        
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Command History", "", "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Voice Command History\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for cmd in reversed(self.commands):
                        f.write(f"[{cmd['timestamp']}] {cmd['command']}\n")
                        if cmd['response']:
                            f.write(f"Response: {cmd['response']}\n")
                        f.write("\n")
                
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", "History exported successfully.")
                
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to export history: {e}")


class VoiceStatusWidget(QGroupBox):
    """Widget showing voice system status"""
    
    def __init__(self):
        super().__init__("Voice Status")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Status indicators
        status_layout = QGridLayout()
        
        # Microphone status
        status_layout.addWidget(QLabel("Microphone:"), 0, 0)
        self.mic_status = QLabel("Ready")
        self.mic_status.setStyleSheet("color: green;")
        status_layout.addWidget(self.mic_status, 0, 1)
        
        # Recognition engine status
        status_layout.addWidget(QLabel("Recognition:"), 1, 0)
        self.recognition_status = QLabel("Ready")
        self.recognition_status.setStyleSheet("color: green;")
        status_layout.addWidget(self.recognition_status, 1, 1)
        
        # TTS engine status
        status_layout.addWidget(QLabel("Text-to-Speech:"), 2, 0)
        self.tts_status = QLabel("Ready")
        self.tts_status.setStyleSheet("color: green;")
        status_layout.addWidget(self.tts_status, 2, 1)
        
        # Voice activation status
        status_layout.addWidget(QLabel("Voice Activation:"), 3, 0)
        self.activation_status = QLabel("Enabled")
        self.activation_status.setStyleSheet("color: green;")
        status_layout.addWidget(self.activation_status, 3, 1)
        
        layout.addLayout(status_layout)
        
        # Audio level indicator
        level_layout = QVBoxLayout()
        level_layout.addWidget(QLabel("Audio Level:"))
        
        self.audio_level = QProgressBar()
        self.audio_level.setRange(0, 100)
        self.audio_level.setValue(0)
        level_layout.addWidget(self.audio_level)
        
        layout.addLayout(level_layout)
    
    def update_status(self, component: str, status: str, color: str = "gray"):
        """Update status for a specific component"""
        if component == "microphone":
            self.mic_status.setText(status)
            self.mic_status.setStyleSheet(f"color: {color};")
        elif component == "recognition":
            self.recognition_status.setText(status)
            self.recognition_status.setStyleSheet(f"color: {color};")
        elif component == "tts":
            self.tts_status.setText(status)
            self.tts_status.setStyleSheet(f"color: {color};")
        elif component == "activation":
            self.activation_status.setText(status)
            self.activation_status.setStyleSheet(f"color: {color};")
    
    def update_audio_level(self, level: int):
        """Update audio level indicator"""
        self.audio_level.setValue(level)


class VoiceWidget(QWidget):
    """Main voice control widget"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        self.recognition_thread = None
        
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        """Initialize the voice control UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Voice Control & Speech Recognition")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for layout
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Voice controls
        self.voice_controls = VoiceControlsWidget()
        self.voice_controls.start_listening.connect(self.start_voice_recognition)
        self.voice_controls.stop_listening.connect(self.stop_voice_recognition)
        self.voice_controls.speak_text.connect(self.speak_text)
        left_layout.addWidget(self.voice_controls)
        
        # Voice status
        self.voice_status = VoiceStatusWidget()
        left_layout.addWidget(self.voice_status)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # Right panel - History and feedback
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Recognition feedback
        feedback_group = QGroupBox("Recognition Feedback")
        feedback_layout = QVBoxLayout(feedback_group)
        
        self.recognition_text = QTextEdit()
        self.recognition_text.setPlaceholderText("Recognized speech will appear here...")
        self.recognition_text.setMaximumHeight(100)
        self.recognition_text.setReadOnly(True)
        feedback_layout.addWidget(self.recognition_text)
        
        right_layout.addWidget(feedback_group)
        
        # Command history
        self.command_history = CommandHistoryWidget()
        right_layout.addWidget(self.command_history)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 350])
    
    def setup_timers(self):
        """Setup update timers"""
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.update_audio_level)
        self.audio_timer.start(100)  # Update every 100ms
    
    def start_voice_recognition(self):
        """Start voice recognition"""
        try:
            if self.recognition_thread and self.recognition_thread.isRunning():
                return
            
            # Get voice processor from assistant manager
            voice_processor = None
            if self.assistant_manager and hasattr(self.assistant_manager, 'voice_processor'):
                voice_processor = self.assistant_manager.voice_processor
            
            # Create and start recognition thread
            self.recognition_thread = VoiceRecognitionThread(voice_processor)
            self.recognition_thread.recognition_started.connect(self.on_recognition_started)
            self.recognition_thread.recognition_stopped.connect(self.on_recognition_stopped)
            self.recognition_thread.text_recognized.connect(self.on_text_recognized)
            self.recognition_thread.error_occurred.connect(self.on_recognition_error)
            
            self.recognition_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting voice recognition: {e}")
            self.voice_status.update_status("recognition", "Error", "red")
    
    def stop_voice_recognition(self):
        """Stop voice recognition"""
        try:
            if self.recognition_thread and self.recognition_thread.isRunning():
                self.recognition_thread.stop_listening()
                self.recognition_thread.wait(3000)  # Wait up to 3 seconds
            
            self.voice_controls.stop_voice_recognition()
            
        except Exception as e:
            self.logger.error(f"Error stopping voice recognition: {e}")
    
    def speak_text(self, text: str):
        """Speak the given text using TTS"""
        try:
            self.voice_status.update_status("tts", "Speaking", "blue")
            
            # In real implementation, this would use the TTS engine
            self.logger.info(f"Speaking: {text}")
            
            # Simulate TTS delay
            QTimer.singleShot(2000, lambda: self.voice_status.update_status("tts", "Ready", "green"))
            
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {e}")
            self.voice_status.update_status("tts", "Error", "red")
    
    def on_recognition_started(self):
        """Handle recognition started signal"""
        self.voice_status.update_status("recognition", "Listening", "blue")
        self.recognition_text.setText("Listening...")
    
    def on_recognition_stopped(self):
        """Handle recognition stopped signal"""
        self.voice_status.update_status("recognition", "Ready", "green")
        self.voice_controls.stop_voice_recognition()
    
    def on_text_recognized(self, text: str):
        """Handle recognized text"""
        self.recognition_text.setText(f"Recognized: {text}")
        self.command_history.add_command(text)
        
        # Process the command if assistant manager is available
        if self.assistant_manager and hasattr(self.assistant_manager, 'command_processor'):
            try:
                # In real implementation, this would process the voice command
                response = f"Processing command: {text}"
                self.command_history.add_command(text, response)
            except Exception as e:
                self.logger.error(f"Error processing voice command: {e}")
    
    def on_recognition_error(self, error: str):
        """Handle recognition error"""
        self.voice_status.update_status("recognition", "Error", "red")
        self.recognition_text.setText(f"Error: {error}")
        self.voice_controls.stop_voice_recognition()
    
    def update_audio_level(self):
        """Update audio level indicator"""
        try:
            # Simulate audio level (in real implementation, get from microphone)
            import random
            level = random.randint(0, 30) if not (self.recognition_thread and self.recognition_thread.is_listening) else random.randint(20, 80)
            self.voice_status.update_audio_level(level)
            
        except Exception as e:
            self.logger.error(f"Error updating audio level: {e}")
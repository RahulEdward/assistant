"""
AI Chat Widget for Computer Assistant GUI
Provides interface for AI conversation, NLP processing, and intelligent assistance
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QListWidget, QListWidgetItem, QProgressBar, QSpinBox,
    QCheckBox, QTabWidget, QSplitter, QScrollArea, QFrame,
    QMessageBox, QFileDialog, QSlider
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QIcon, QPixmap, QTextCharFormat, QColor


class AIResponseThread(QThread):
    """Thread for processing AI responses"""
    
    response_started = Signal()
    response_received = Signal(str)
    response_error = Signal(str)
    
    def __init__(self, message: str, ai_manager):
        super().__init__()
        self.message = message
        self.ai_manager = ai_manager
        
    def run(self):
        """Process AI response"""
        try:
            self.response_started.emit()
            
            if self.ai_manager:
                # Use actual AI manager
                response = self.ai_manager.process_message(self.message)
                self.response_received.emit(response)
            else:
                # Simulate AI response
                import time
                time.sleep(2)  # Simulate processing time
                
                # Generate a mock response based on the message
                if "hello" in self.message.lower():
                    response = "Hello! How can I assist you today?"
                elif "weather" in self.message.lower():
                    response = "I'd be happy to help with weather information, but I don't have access to real-time weather data at the moment."
                elif "time" in self.message.lower():
                    response = f"The current time is {datetime.now().strftime('%H:%M:%S')}."
                elif "help" in self.message.lower():
                    response = "I can help you with various tasks including automation, system control, file management, and answering questions. What would you like to do?"
                else:
                    response = f"I understand you said: '{self.message}'. How can I help you with that?"
                
                self.response_received.emit(response)
                
        except Exception as e:
            self.response_error.emit(str(e))


class ChatDisplayWidget(QGroupBox):
    """Widget for displaying chat conversation"""
    
    def __init__(self):
        super().__init__("Conversation")
        self.conversation_history = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(300)
        layout.addWidget(self.chat_display)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.clicked.connect(self.clear_chat)
        control_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("Save Chat")
        self.save_btn.clicked.connect(self.save_chat)
        control_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Chat")
        self.load_btn.clicked.connect(self.load_chat)
        control_layout.addWidget(self.load_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
    
    def add_message(self, sender: str, message: str, timestamp: Optional[datetime] = None):
        """Add a message to the chat display"""
        if timestamp is None:
            timestamp = datetime.now()
        
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Store in history
        self.conversation_history.append({
            'sender': sender,
            'message': message,
            'timestamp': timestamp.isoformat()
        })
        
        # Format message for display
        if sender == "User":
            formatted_message = f"<div style='margin: 5px 0; padding: 8px; background-color: #e3f2fd; border-radius: 8px;'><b>[{time_str}] You:</b><br>{message}</div>"
        else:
            formatted_message = f"<div style='margin: 5px 0; padding: 8px; background-color: #f3e5f5; border-radius: 8px;'><b>[{time_str}] Assistant:</b><br>{message}</div>"
        
        # Add to display
        self.chat_display.append(formatted_message)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_system_message(self, message: str):
        """Add a system message to the chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"<div style='margin: 5px 0; padding: 5px; color: #666; font-style: italic; text-align: center;'>[{timestamp}] {message}</div>"
        self.chat_display.append(formatted_message)
    
    def clear_chat(self):
        """Clear the chat display"""
        reply = QMessageBox.question(
            self, "Clear Chat",
            "Are you sure you want to clear the conversation?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.chat_display.clear()
            self.conversation_history.clear()
            self.add_system_message("Conversation cleared")
    
    def save_chat(self):
        """Save chat to file"""
        if not self.conversation_history:
            QMessageBox.information(self, "Info", "No conversation to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Conversation", "", "JSON Files (*.json);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("AI Assistant Conversation\n")
                        f.write("=" * 50 + "\n\n")
                        
                        for entry in self.conversation_history:
                            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                            f.write(f"[{timestamp}] {entry['sender']}: {entry['message']}\n\n")
                
                QMessageBox.information(self, "Success", "Conversation saved successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save conversation: {e}")
    
    def load_chat(self):
        """Load chat from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Conversation", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                self.chat_display.clear()
                self.conversation_history = history
                
                for entry in history:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    self.add_message(entry['sender'], entry['message'], timestamp)
                
                QMessageBox.information(self, "Success", "Conversation loaded successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load conversation: {e}")


class ChatInputWidget(QGroupBox):
    """Widget for chat input and controls"""
    
    message_sent = Signal(str)
    
    def __init__(self):
        super().__init__("Message Input")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Message input
        input_layout = QVBoxLayout()
        
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(100)
        self.message_input.setPlaceholderText("Type your message here...")
        input_layout.addWidget(self.message_input)
        
        # Input controls
        control_layout = QHBoxLayout()
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setDefault(True)
        control_layout.addWidget(self.send_btn)
        
        self.clear_input_btn = QPushButton("Clear")
        self.clear_input_btn.clicked.connect(self.clear_input)
        control_layout.addWidget(self.clear_input_btn)
        
        control_layout.addStretch()
        
        # Quick actions
        self.quick_help_btn = QPushButton("Help")
        self.quick_help_btn.clicked.connect(lambda: self.send_quick_message("help"))
        control_layout.addWidget(self.quick_help_btn)
        
        self.quick_time_btn = QPushButton("Time")
        self.quick_time_btn.clicked.connect(lambda: self.send_quick_message("What time is it?"))
        control_layout.addWidget(self.quick_time_btn)
        
        input_layout.addLayout(control_layout)
        layout.addLayout(input_layout)
        
        # Message templates
        templates_group = QGroupBox("Quick Templates")
        templates_layout = QVBoxLayout(templates_group)
        
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Custom message...",
            "How can you help me?",
            "What are your capabilities?",
            "Show me system information",
            "Help me automate a task",
            "What's the weather like?",
            "Set a reminder for me",
            "Open an application",
            "Search for files"
        ])
        self.template_combo.currentTextChanged.connect(self.on_template_selected)
        templates_layout.addWidget(self.template_combo)
        
        layout.addWidget(templates_group)
        
        # Connect Enter key to send
        self.message_input.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Handle key events for message input"""
        if obj == self.message_input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)
    
    def send_message(self):
        """Send the current message"""
        message = self.message_input.toPlainText().strip()
        if message:
            self.message_sent.emit(message)
            self.message_input.clear()
    
    def send_quick_message(self, message: str):
        """Send a quick message"""
        self.message_sent.emit(message)
    
    def clear_input(self):
        """Clear the message input"""
        self.message_input.clear()
    
    def on_template_selected(self, text: str):
        """Handle template selection"""
        if text != "Custom message...":
            self.message_input.setPlainText(text)
            self.template_combo.setCurrentIndex(0)  # Reset to "Custom message..."


class AISettingsWidget(QGroupBox):
    """Widget for AI settings and configuration"""
    
    def __init__(self):
        super().__init__("AI Settings")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # AI Provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("AI Provider:"))
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "OpenAI GPT",
            "Local AI Model",
            "Azure OpenAI",
            "Anthropic Claude",
            "Google Bard"
        ])
        provider_layout.addWidget(self.provider_combo)
        
        layout.addLayout(provider_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "claude-3-sonnet",
            "claude-3-opus"
        ])
        model_layout.addWidget(self.model_combo)
        
        layout.addLayout(model_layout)
        
        # Temperature setting
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Creativity:"))
        
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(70)
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)
        temp_layout.addWidget(self.temperature_slider)
        
        self.temperature_label = QLabel("0.7")
        temp_layout.addWidget(self.temperature_label)
        
        layout.addLayout(temp_layout)
        
        # Max tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max Response Length:"))
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 4000)
        self.max_tokens_spin.setValue(1000)
        tokens_layout.addWidget(self.max_tokens_spin)
        
        layout.addLayout(tokens_layout)
        
        # Options
        self.stream_response_cb = QCheckBox("Stream Responses")
        self.stream_response_cb.setChecked(True)
        layout.addWidget(self.stream_response_cb)
        
        self.context_memory_cb = QCheckBox("Remember Conversation Context")
        self.context_memory_cb.setChecked(True)
        layout.addWidget(self.context_memory_cb)
        
        self.auto_suggestions_cb = QCheckBox("Show Auto Suggestions")
        self.auto_suggestions_cb.setChecked(False)
        layout.addWidget(self.auto_suggestions_cb)
    
    def update_temperature_label(self, value):
        """Update temperature label"""
        temp_value = value / 100.0
        self.temperature_label.setText(f"{temp_value:.1f}")


class AIChatWidget(QWidget):
    """Main AI chat widget"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        self.response_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the AI chat UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("AI Assistant Chat")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Chat
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Chat display
        self.chat_display = ChatDisplayWidget()
        left_layout.addWidget(self.chat_display)
        
        # Chat input
        self.chat_input = ChatInputWidget()
        self.chat_input.message_sent.connect(self.send_message)
        left_layout.addWidget(self.chat_input)
        
        main_splitter.addWidget(left_panel)
        
        # Right panel - Settings and status
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # AI Settings
        self.ai_settings = AISettingsWidget()
        right_layout.addWidget(self.ai_settings)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        right_layout.addWidget(status_group)
        
        right_layout.addStretch()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 250])
        
        # Welcome message
        self.chat_display.add_system_message("AI Assistant initialized. How can I help you today?")
    
    def send_message(self, message: str):
        """Send a message to the AI"""
        try:
            if self.response_thread and self.response_thread.isRunning():
                QMessageBox.warning(self, "Warning", "Please wait for the current response to complete.")
                return
            
            # Add user message to chat
            self.chat_display.add_message("User", message)
            
            # Update status
            self.status_label.setText("Processing...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Get AI manager from assistant manager
            ai_manager = None
            if self.assistant_manager and hasattr(self.assistant_manager, 'external_ai'):
                ai_manager = self.assistant_manager.external_ai
            
            # Create and start response thread
            self.response_thread = AIResponseThread(message, ai_manager)
            self.response_thread.response_started.connect(self.on_response_started)
            self.response_thread.response_received.connect(self.on_response_received)
            self.response_thread.response_error.connect(self.on_response_error)
            
            self.response_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            self.on_response_error(str(e))
    
    def on_response_started(self):
        """Handle response started signal"""
        self.status_label.setText("AI is thinking...")
    
    def on_response_received(self, response: str):
        """Handle AI response"""
        self.status_label.setText("Ready")
        self.progress_bar.setVisible(False)
        
        # Add AI response to chat
        self.chat_display.add_message("Assistant", response)
    
    def on_response_error(self, error: str):
        """Handle response error"""
        self.status_label.setText("Error")
        self.progress_bar.setVisible(False)
        
        self.chat_display.add_system_message(f"Error: {error}")
        QMessageBox.critical(self, "AI Error", f"Failed to get AI response: {error}")
    
    def get_ai_settings(self) -> Dict[str, Any]:
        """Get current AI settings"""
        return {
            'provider': self.ai_settings.provider_combo.currentText(),
            'model': self.ai_settings.model_combo.currentText(),
            'temperature': self.ai_settings.temperature_slider.value() / 100.0,
            'max_tokens': self.ai_settings.max_tokens_spin.value(),
            'stream_response': self.ai_settings.stream_response_cb.isChecked(),
            'context_memory': self.ai_settings.context_memory_cb.isChecked(),
            'auto_suggestions': self.ai_settings.auto_suggestions_cb.isChecked()
        }
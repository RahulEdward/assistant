"""
Dashboard Widget for Computer Assistant GUI
Provides system overview, quick stats, and main feature access
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QFrame, QScrollArea,
    QListWidget, QListWidgetItem, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPixmap, QIcon
import psutil


class SystemStatsWidget(QGroupBox):
    """Widget displaying real-time system statistics"""
    
    def __init__(self):
        super().__init__("System Statistics")
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout(self)
        
        # CPU Usage
        self.cpu_label = QLabel("CPU Usage:")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximum(100)
        self.cpu_value = QLabel("0%")
        
        layout.addWidget(self.cpu_label, 0, 0)
        layout.addWidget(self.cpu_progress, 0, 1)
        layout.addWidget(self.cpu_value, 0, 2)
        
        # Memory Usage
        self.memory_label = QLabel("Memory Usage:")
        self.memory_progress = QProgressBar()
        self.memory_progress.setMaximum(100)
        self.memory_value = QLabel("0%")
        
        layout.addWidget(self.memory_label, 1, 0)
        layout.addWidget(self.memory_progress, 1, 1)
        layout.addWidget(self.memory_value, 1, 2)
        
        # Disk Usage
        self.disk_label = QLabel("Disk Usage:")
        self.disk_progress = QProgressBar()
        self.disk_progress.setMaximum(100)
        self.disk_value = QLabel("0%")
        
        layout.addWidget(self.disk_label, 2, 0)
        layout.addWidget(self.disk_progress, 2, 1)
        layout.addWidget(self.disk_value, 2, 2)
        
        # Network
        self.network_label = QLabel("Network:")
        self.network_status = QLabel("Disconnected")
        
        layout.addWidget(self.network_label, 3, 0)
        layout.addWidget(self.network_status, 3, 1, 1, 2)
    
    def update_stats(self):
        """Update system statistics"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent()
            self.cpu_progress.setValue(int(cpu_percent))
            self.cpu_value.setText(f"{cpu_percent:.1f}%")
            
            # Memory
            memory = psutil.virtual_memory()
            self.memory_progress.setValue(int(memory.percent))
            self.memory_value.setText(f"{memory.percent:.1f}%")
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_progress.setValue(int(disk_percent))
            self.disk_value.setText(f"{disk_percent:.1f}%")
            
            # Network
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                self.network_status.setText("Connected")
                self.network_status.setStyleSheet("color: green;")
            except OSError:
                self.network_status.setText("Disconnected")
                self.network_status.setStyleSheet("color: red;")
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating system stats: {e}")


class QuickActionsWidget(QGroupBox):
    """Widget with quick action buttons"""
    
    action_triggered = Signal(str)
    
    def __init__(self):
        super().__init__("Quick Actions")
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout(self)
        
        # Voice Control
        self.voice_btn = QPushButton("Start Voice Control")
        self.voice_btn.clicked.connect(lambda: self.action_triggered.emit("voice_control"))
        layout.addWidget(self.voice_btn, 0, 0)
        
        # Window Manager
        self.window_btn = QPushButton("Window Manager")
        self.window_btn.clicked.connect(lambda: self.action_triggered.emit("window_manager"))
        layout.addWidget(self.window_btn, 0, 1)
        
        # AI Chat
        self.ai_btn = QPushButton("AI Assistant")
        self.ai_btn.clicked.connect(lambda: self.action_triggered.emit("ai_chat"))
        layout.addWidget(self.ai_btn, 1, 0)
        
        # Automation
        self.automation_btn = QPushButton("Automation Tasks")
        self.automation_btn.clicked.connect(lambda: self.action_triggered.emit("automation"))
        layout.addWidget(self.automation_btn, 1, 1)
        
        # System Monitor
        self.monitor_btn = QPushButton("System Monitor")
        self.monitor_btn.clicked.connect(lambda: self.action_triggered.emit("system_monitor"))
        layout.addWidget(self.monitor_btn, 2, 0)
        
        # Settings
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(lambda: self.action_triggered.emit("settings"))
        layout.addWidget(self.settings_btn, 2, 1)


class RecentActivityWidget(QGroupBox):
    """Widget showing recent activity and logs"""
    
    def __init__(self):
        super().__init__("Recent Activity")
        self.init_ui()
        self.activities = []
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.activity_list = QListWidget()
        self.activity_list.setMaximumHeight(200)
        layout.addWidget(self.activity_list)
        
        # Clear button
        clear_btn = QPushButton("Clear Activity")
        clear_btn.clicked.connect(self.clear_activity)
        layout.addWidget(clear_btn)
    
    def add_activity(self, message: str):
        """Add new activity to the list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        activity_text = f"[{timestamp}] {message}"
        
        item = QListWidgetItem(activity_text)
        self.activity_list.insertItem(0, item)
        
        # Keep only last 50 items
        if self.activity_list.count() > 50:
            self.activity_list.takeItem(self.activity_list.count() - 1)
    
    def clear_activity(self):
        """Clear all activity"""
        self.activity_list.clear()


class StatusOverviewWidget(QGroupBox):
    """Widget showing overall system status"""
    
    def __init__(self):
        super().__init__("System Status")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Assistant Status
        status_layout = QHBoxLayout()
        self.assistant_status_label = QLabel("Assistant Status:")
        self.assistant_status_value = QLabel("Initializing...")
        self.assistant_status_value.setStyleSheet("color: orange;")
        
        status_layout.addWidget(self.assistant_status_label)
        status_layout.addWidget(self.assistant_status_value)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Voice Recognition Status
        voice_layout = QHBoxLayout()
        self.voice_status_label = QLabel("Voice Recognition:")
        self.voice_status_value = QLabel("Disabled")
        self.voice_status_value.setStyleSheet("color: red;")
        
        voice_layout.addWidget(self.voice_status_label)
        voice_layout.addWidget(self.voice_status_value)
        voice_layout.addStretch()
        layout.addLayout(voice_layout)
        
        # AI Integration Status
        ai_layout = QHBoxLayout()
        self.ai_status_label = QLabel("AI Integration:")
        self.ai_status_value = QLabel("Disconnected")
        self.ai_status_value.setStyleSheet("color: red;")
        
        ai_layout.addWidget(self.ai_status_label)
        ai_layout.addWidget(self.ai_status_value)
        ai_layout.addStretch()
        layout.addLayout(ai_layout)
        
        # Automation Status
        automation_layout = QHBoxLayout()
        self.automation_status_label = QLabel("Automation:")
        self.automation_status_value = QLabel("Idle")
        self.automation_status_value.setStyleSheet("color: gray;")
        
        automation_layout.addWidget(self.automation_status_label)
        automation_layout.addWidget(self.automation_status_value)
        automation_layout.addStretch()
        layout.addLayout(automation_layout)
    
    def update_status(self, component: str, status: str, color: str = "gray"):
        """Update status for a specific component"""
        if component == "assistant":
            self.assistant_status_value.setText(status)
            self.assistant_status_value.setStyleSheet(f"color: {color};")
        elif component == "voice":
            self.voice_status_value.setText(status)
            self.voice_status_value.setStyleSheet(f"color: {color};")
        elif component == "ai":
            self.ai_status_value.setText(status)
            self.ai_status_value.setStyleSheet(f"color: {color};")
        elif component == "automation":
            self.automation_status_value.setText(status)
            self.automation_status_value.setStyleSheet(f"color: {color};")


class DashboardWidget(QWidget):
    """Main dashboard widget providing system overview"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        """Initialize the dashboard UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Computer Assistant Dashboard")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for layout
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # System stats
        self.system_stats = SystemStatsWidget()
        left_layout.addWidget(self.system_stats)
        
        # Status overview
        self.status_overview = StatusOverviewWidget()
        left_layout.addWidget(self.status_overview)
        
        # Quick actions
        self.quick_actions = QuickActionsWidget()
        self.quick_actions.action_triggered.connect(self.handle_quick_action)
        left_layout.addWidget(self.quick_actions)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Recent activity
        self.recent_activity = RecentActivityWidget()
        right_layout.addWidget(self.recent_activity)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 300])
        
        # Add some initial activity
        self.recent_activity.add_activity("Dashboard initialized")
        
    def setup_timers(self):
        """Setup update timers"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(2000)  # Update every 2 seconds
        
    def update_data(self):
        """Update dashboard data"""
        try:
            # Update system stats
            self.system_stats.update_stats()
            
            # Update status based on assistant manager
            if self.assistant_manager:
                if hasattr(self.assistant_manager, 'is_initialized'):
                    if self.assistant_manager.is_initialized:
                        self.status_overview.update_status("assistant", "Running", "green")
                    else:
                        self.status_overview.update_status("assistant", "Stopped", "red")
                
                # Check voice processor status
                if hasattr(self.assistant_manager, 'voice_processor'):
                    if self.assistant_manager.voice_processor:
                        if hasattr(self.assistant_manager.voice_processor, 'is_listening'):
                            if self.assistant_manager.voice_processor.is_listening:
                                self.status_overview.update_status("voice", "Listening", "green")
                            else:
                                self.status_overview.update_status("voice", "Ready", "orange")
                        else:
                            self.status_overview.update_status("voice", "Ready", "orange")
                    else:
                        self.status_overview.update_status("voice", "Disabled", "red")
                
                # Check AI integration status
                if hasattr(self.assistant_manager, 'external_ai'):
                    if self.assistant_manager.external_ai:
                        self.status_overview.update_status("ai", "Connected", "green")
                    else:
                        self.status_overview.update_status("ai", "Disconnected", "red")
                
                # Check automation status
                if hasattr(self.assistant_manager, 'system_controller'):
                    if self.assistant_manager.system_controller:
                        self.status_overview.update_status("automation", "Ready", "green")
                    else:
                        self.status_overview.update_status("automation", "Disabled", "red")
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
    
    def handle_quick_action(self, action: str):
        """Handle quick action button clicks"""
        try:
            self.recent_activity.add_activity(f"Quick action: {action}")
            
            # Get parent tab widget and switch tabs
            parent = self.parent()
            while parent and not hasattr(parent, 'setCurrentIndex'):
                parent = parent.parent()
            
            if parent:
                tab_mapping = {
                    "voice_control": 2,
                    "window_manager": 4,
                    "ai_chat": 3,
                    "automation": 1,
                    "system_monitor": 5,
                    "settings": 6
                }
                
                if action in tab_mapping:
                    parent.setCurrentIndex(tab_mapping[action])
                    
        except Exception as e:
            self.logger.error(f"Error handling quick action {action}: {e}")
    
    def add_activity(self, message: str):
        """Add activity message to the dashboard"""
        self.recent_activity.add_activity(message)
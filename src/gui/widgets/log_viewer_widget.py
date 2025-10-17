"""
Log Viewer Widget for Computer Assistant GUI
Provides real-time log monitoring and analysis capabilities
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QListWidget, QListWidgetItem, QProgressBar, QSpinBox,
    QCheckBox, QTabWidget, QFileDialog, QMessageBox, QSlider,
    QFrame, QScrollArea, QTreeWidget, QTreeWidgetItem, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateTimeEdit
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QDateTime
from PySide6.QtGui import QFont, QColor, QPalette, QTextCursor, QTextCharFormat


class LogMonitorThread(QThread):
    """Thread for monitoring log files"""
    
    log_entry_received = Signal(dict)
    
    def __init__(self, log_file_path: str):
        super().__init__()
        self.log_file_path = log_file_path
        self.running = False
        self.last_position = 0
        
    def run(self):
        """Monitor log file for new entries"""
        self.running = True
        
        while self.running:
            try:
                if os.path.exists(self.log_file_path):
                    with open(self.log_file_path, 'r', encoding='utf-8') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                        
                        for line in new_lines:
                            if line.strip():
                                log_entry = self.parse_log_line(line.strip())
                                if log_entry:
                                    self.log_entry_received.emit(log_entry)
                
                self.msleep(1000)  # Check every second
                
            except Exception as e:
                print(f"Error monitoring log file: {e}")
                self.msleep(5000)  # Wait longer on error
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a log line into structured data"""
        try:
            # Common log format: TIMESTAMP - LEVEL - MODULE - MESSAGE
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - ([^-]+) - (.+)'
            match = re.match(pattern, line)
            
            if match:
                timestamp_str, level, module, message = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                
                return {
                    'timestamp': timestamp,
                    'level': level.strip(),
                    'module': module.strip(),
                    'message': message.strip(),
                    'raw_line': line
                }
            else:
                # Fallback for non-standard format
                return {
                    'timestamp': datetime.now(),
                    'level': 'INFO',
                    'module': 'Unknown',
                    'message': line,
                    'raw_line': line
                }
                
        except Exception as e:
            return None
    
    def stop(self):
        """Stop monitoring"""
        self.running = False


class LogFilterWidget(QGroupBox):
    """Widget for filtering log entries"""
    
    filter_changed = Signal()
    
    def __init__(self):
        super().__init__("Log Filters")
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout(self)
        
        # Level filter
        layout.addWidget(QLabel("Level:"), 0, 0)
        self.level_combo = QComboBox()
        self.level_combo.addItems(["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_combo.currentTextChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.level_combo, 0, 1)
        
        # Module filter
        layout.addWidget(QLabel("Module:"), 0, 2)
        self.module_combo = QComboBox()
        self.module_combo.addItem("All")
        self.module_combo.currentTextChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.module_combo, 0, 3)
        
        # Time range filter
        layout.addWidget(QLabel("From:"), 1, 0)
        self.from_datetime = QDateTimeEdit()
        self.from_datetime.setDateTime(QDateTime.currentDateTime().addDays(-1))
        self.from_datetime.dateTimeChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.from_datetime, 1, 1)
        
        layout.addWidget(QLabel("To:"), 1, 2)
        self.to_datetime = QDateTimeEdit()
        self.to_datetime.setDateTime(QDateTime.currentDateTime())
        self.to_datetime.dateTimeChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.to_datetime, 1, 3)
        
        # Search filter
        layout.addWidget(QLabel("Search:"), 2, 0)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search in messages...")
        self.search_edit.textChanged.connect(self.filter_changed.emit)
        layout.addWidget(self.search_edit, 2, 1, 1, 2)
        
        # Clear filters button
        self.clear_btn = QPushButton("Clear Filters")
        self.clear_btn.clicked.connect(self.clear_filters)
        layout.addWidget(self.clear_btn, 2, 3)
    
    def clear_filters(self):
        """Clear all filters"""
        self.level_combo.setCurrentText("All")
        self.module_combo.setCurrentText("All")
        self.from_datetime.setDateTime(QDateTime.currentDateTime().addDays(-1))
        self.to_datetime.setDateTime(QDateTime.currentDateTime())
        self.search_edit.clear()
    
    def get_filter_criteria(self) -> Dict[str, Any]:
        """Get current filter criteria"""
        return {
            'level': self.level_combo.currentText() if self.level_combo.currentText() != "All" else None,
            'module': self.module_combo.currentText() if self.module_combo.currentText() != "All" else None,
            'from_time': self.from_datetime.dateTime().toPython(),
            'to_time': self.to_datetime.dateTime().toPython(),
            'search_text': self.search_edit.text().strip() if self.search_edit.text().strip() else None
        }


class LogTableWidget(QTableWidget):
    """Custom table widget for displaying log entries"""
    
    def __init__(self):
        super().__init__()
        self.log_entries = []
        self.init_ui()
        
    def init_ui(self):
        """Initialize the table"""
        # Set columns
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Timestamp", "Level", "Module", "Message"])
        
        # Configure headers
        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Timestamp
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Level
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Module
        header.setSectionResizeMode(3, QHeaderView.Stretch)           # Message
        
        # Configure table
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSortingEnabled(True)
        
        # Set row height
        self.verticalHeader().setDefaultSectionSize(25)
        self.verticalHeader().setVisible(False)
    
    def add_log_entry(self, entry: Dict[str, Any]):
        """Add a new log entry to the table"""
        self.log_entries.append(entry)
        
        row = self.rowCount()
        self.insertRow(row)
        
        # Timestamp
        timestamp_item = QTableWidgetItem(entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
        self.setItem(row, 0, timestamp_item)
        
        # Level
        level_item = QTableWidgetItem(entry['level'])
        level_item.setBackground(self.get_level_color(entry['level']))
        self.setItem(row, 1, level_item)
        
        # Module
        module_item = QTableWidgetItem(entry['module'])
        self.setItem(row, 2, module_item)
        
        # Message
        message_item = QTableWidgetItem(entry['message'])
        message_item.setToolTip(entry['message'])  # Full message on hover
        self.setItem(row, 3, message_item)
        
        # Auto-scroll to bottom
        self.scrollToBottom()
    
    def get_level_color(self, level: str) -> QColor:
        """Get color for log level"""
        colors = {
            'DEBUG': QColor(200, 200, 200),
            'INFO': QColor(173, 216, 230),
            'WARNING': QColor(255, 255, 0),
            'ERROR': QColor(255, 182, 193),
            'CRITICAL': QColor(255, 99, 71)
        }
        return colors.get(level, QColor(255, 255, 255))
    
    def clear_entries(self):
        """Clear all log entries"""
        self.log_entries.clear()
        self.setRowCount(0)
    
    def filter_entries(self, criteria: Dict[str, Any]):
        """Filter entries based on criteria"""
        self.setRowCount(0)
        
        for entry in self.log_entries:
            if self.matches_criteria(entry, criteria):
                row = self.rowCount()
                self.insertRow(row)
                
                # Add filtered entry
                timestamp_item = QTableWidgetItem(entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
                self.setItem(row, 0, timestamp_item)
                
                level_item = QTableWidgetItem(entry['level'])
                level_item.setBackground(self.get_level_color(entry['level']))
                self.setItem(row, 1, level_item)
                
                module_item = QTableWidgetItem(entry['module'])
                self.setItem(row, 2, module_item)
                
                message_item = QTableWidgetItem(entry['message'])
                message_item.setToolTip(entry['message'])
                self.setItem(row, 3, message_item)
    
    def matches_criteria(self, entry: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if entry matches filter criteria"""
        # Level filter
        if criteria['level'] and entry['level'] != criteria['level']:
            return False
        
        # Module filter
        if criteria['module'] and entry['module'] != criteria['module']:
            return False
        
        # Time range filter
        if entry['timestamp'] < criteria['from_time'] or entry['timestamp'] > criteria['to_time']:
            return False
        
        # Search filter
        if criteria['search_text']:
            search_text = criteria['search_text'].lower()
            if search_text not in entry['message'].lower():
                return False
        
        return True


class LogStatsWidget(QGroupBox):
    """Widget for displaying log statistics"""
    
    def __init__(self):
        super().__init__("Log Statistics")
        self.stats = {
            'DEBUG': 0,
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0,
            'CRITICAL': 0
        }
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout(self)
        
        # Level counters
        self.level_labels = {}
        for i, level in enumerate(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
            layout.addWidget(QLabel(f"{level}:"), i, 0)
            label = QLabel("0")
            label.setStyleSheet(f"color: {self.get_level_color_name(level)};")
            self.level_labels[level] = label
            layout.addWidget(label, i, 1)
        
        # Total entries
        layout.addWidget(QLabel("Total:"), 5, 0)
        self.total_label = QLabel("0")
        self.total_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.total_label, 5, 1)
    
    def get_level_color_name(self, level: str) -> str:
        """Get CSS color name for log level"""
        colors = {
            'DEBUG': 'gray',
            'INFO': 'blue',
            'WARNING': 'orange',
            'ERROR': 'red',
            'CRITICAL': 'darkred'
        }
        return colors.get(level, 'black')
    
    def update_stats(self, level: str):
        """Update statistics for a new log entry"""
        if level in self.stats:
            self.stats[level] += 1
            self.level_labels[level].setText(str(self.stats[level]))
            
            total = sum(self.stats.values())
            self.total_label.setText(str(total))
    
    def reset_stats(self):
        """Reset all statistics"""
        for level in self.stats:
            self.stats[level] = 0
            self.level_labels[level].setText("0")
        self.total_label.setText("0")


class LogViewerWidget(QWidget):
    """Main log viewer widget"""
    
    def __init__(self, assistant_manager=None):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        
        # Log monitoring
        self.monitor_thread = None
        self.log_file_path = "logs/assistant.log"  # Default log file
        
        # Available modules (will be populated from log entries)
        self.modules = set()
        
        self.init_ui()
        self.setup_monitoring()
        
    def init_ui(self):
        """Initialize the log viewer UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Log Viewer & Monitor")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Log file selection
        control_layout.addWidget(QLabel("Log File:"))
        self.log_file_edit = QLineEdit(self.log_file_path)
        control_layout.addWidget(self.log_file_edit)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_log_file)
        control_layout.addWidget(self.browse_btn)
        
        # Monitoring controls
        self.start_monitor_btn = QPushButton("Start Monitoring")
        self.start_monitor_btn.clicked.connect(self.start_monitoring)
        control_layout.addWidget(self.start_monitor_btn)
        
        self.stop_monitor_btn = QPushButton("Stop Monitoring")
        self.stop_monitor_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitor_btn.setEnabled(False)
        control_layout.addWidget(self.stop_monitor_btn)
        
        # Clear logs
        self.clear_btn = QPushButton("Clear Display")
        self.clear_btn.clicked.connect(self.clear_display)
        control_layout.addWidget(self.clear_btn)
        
        # Export logs
        self.export_btn = QPushButton("Export Logs")
        self.export_btn.clicked.connect(self.export_logs)
        control_layout.addWidget(self.export_btn)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Left panel - Filters and Stats
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Filters
        self.filter_widget = LogFilterWidget()
        self.filter_widget.filter_changed.connect(self.apply_filters)
        left_layout.addWidget(self.filter_widget)
        
        # Statistics
        self.stats_widget = LogStatsWidget()
        left_layout.addWidget(self.stats_widget)
        
        left_layout.addStretch()
        content_splitter.addWidget(left_panel)
        
        # Right panel - Log table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Log table
        self.log_table = LogTableWidget()
        right_layout.addWidget(self.log_table)
        
        content_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        content_splitter.setSizes([300, 700])
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        self.entries_count_label = QLabel("Entries: 0")
        status_layout.addWidget(self.entries_count_label)
        
        main_layout.addLayout(status_layout)
    
    def browse_log_file(self):
        """Browse for log file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Log File", "", "Log Files (*.log);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            self.log_file_edit.setText(file_path)
            self.log_file_path = file_path
    
    def setup_monitoring(self):
        """Setup log file monitoring"""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.log_file_path).parent
        log_dir.mkdir(exist_ok=True)
        
        # Create log file if it doesn't exist
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w') as f:
                f.write(f"# Log file created at {datetime.now()}\n")
    
    def start_monitoring(self):
        """Start monitoring the log file"""
        try:
            self.log_file_path = self.log_file_edit.text()
            
            if not os.path.exists(self.log_file_path):
                QMessageBox.warning(self, "Warning", f"Log file not found: {self.log_file_path}")
                return
            
            # Stop existing monitoring
            self.stop_monitoring()
            
            # Start new monitoring thread
            self.monitor_thread = LogMonitorThread(self.log_file_path)
            self.monitor_thread.log_entry_received.connect(self.add_log_entry)
            self.monitor_thread.start()
            
            # Update UI
            self.start_monitor_btn.setEnabled(False)
            self.stop_monitor_btn.setEnabled(True)
            self.status_label.setText(f"Monitoring: {self.log_file_path}")
            
            self.logger.info(f"Started monitoring log file: {self.log_file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start monitoring: {e}")
            self.logger.error(f"Error starting log monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring the log file"""
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.stop()
            self.monitor_thread.wait()
            self.monitor_thread = None
        
        # Update UI
        self.start_monitor_btn.setEnabled(True)
        self.stop_monitor_btn.setEnabled(False)
        self.status_label.setText("Monitoring stopped")
        
        self.logger.info("Stopped log monitoring")
    
    def add_log_entry(self, entry: Dict[str, Any]):
        """Add a new log entry"""
        try:
            # Add to table
            self.log_table.add_log_entry(entry)
            
            # Update statistics
            self.stats_widget.update_stats(entry['level'])
            
            # Update module filter if new module
            if entry['module'] not in self.modules:
                self.modules.add(entry['module'])
                self.filter_widget.module_combo.addItem(entry['module'])
            
            # Update entry count
            total_entries = len(self.log_table.log_entries)
            self.entries_count_label.setText(f"Entries: {total_entries}")
            
        except Exception as e:
            self.logger.error(f"Error adding log entry: {e}")
    
    def apply_filters(self):
        """Apply current filters to the log display"""
        try:
            criteria = self.filter_widget.get_filter_criteria()
            self.log_table.filter_entries(criteria)
            
            # Update entry count for filtered view
            visible_entries = self.log_table.rowCount()
            total_entries = len(self.log_table.log_entries)
            self.entries_count_label.setText(f"Entries: {visible_entries}/{total_entries}")
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {e}")
    
    def clear_display(self):
        """Clear the log display"""
        reply = QMessageBox.question(
            self, "Clear Display",
            "Are you sure you want to clear the log display?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.log_table.clear_entries()
            self.stats_widget.reset_stats()
            self.entries_count_label.setText("Entries: 0")
            self.status_label.setText("Display cleared")
    
    def export_logs(self):
        """Export logs to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Log Export - {datetime.now()}\n")
                    f.write(f"# Total Entries: {len(self.log_table.log_entries)}\n\n")
                    
                    for entry in self.log_table.log_entries:
                        f.write(f"{entry['timestamp']} - {entry['level']} - {entry['module']} - {entry['message']}\n")
                
                QMessageBox.information(self, "Success", f"Logs exported to: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export logs: {e}")
    
    def closeEvent(self, event):
        """Handle widget close event"""
        self.stop_monitoring()
        event.accept()
"""
Automation Widget for Computer Assistant GUI
Provides interface for automation tasks, system control, and task management
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QListWidget, QListWidgetItem, QProgressBar, QSpinBox,
    QCheckBox, QTabWidget, QSplitter, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QFileDialog, QSlider
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QIcon


class TaskExecutionThread(QThread):
    """Thread for executing automation tasks"""
    
    task_started = Signal(str)
    task_progress = Signal(int)
    task_completed = Signal(str, bool)
    task_error = Signal(str, str)
    
    def __init__(self, task_data: Dict[str, Any], automation_engine):
        super().__init__()
        self.task_data = task_data
        self.automation_engine = automation_engine
        self.is_running = False
        
    def run(self):
        """Execute the automation task"""
        try:
            self.is_running = True
            task_name = self.task_data.get('name', 'Unknown Task')
            self.task_started.emit(task_name)
            
            # Simulate task execution with progress updates
            for i in range(0, 101, 10):
                if not self.is_running:
                    break
                self.task_progress.emit(i)
                self.msleep(100)  # Simulate work
            
            if self.is_running:
                self.task_completed.emit(task_name, True)
            else:
                self.task_completed.emit(task_name, False)
                
        except Exception as e:
            self.task_error.emit(task_name, str(e))
    
    def stop(self):
        """Stop the task execution"""
        self.is_running = False


class TaskManagerWidget(QGroupBox):
    """Widget for managing automation tasks"""
    
    task_executed = Signal(dict)
    
    def __init__(self):
        super().__init__("Task Manager")
        self.tasks = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Task creation section
        create_group = QGroupBox("Create New Task")
        create_layout = QGridLayout(create_group)
        
        # Task name
        create_layout.addWidget(QLabel("Task Name:"), 0, 0)
        self.task_name_edit = QLineEdit()
        create_layout.addWidget(self.task_name_edit, 0, 1)
        
        # Task type
        create_layout.addWidget(QLabel("Task Type:"), 1, 0)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems([
            "Window Management",
            "File Operations",
            "System Commands",
            "Mouse Actions",
            "Keyboard Actions",
            "Screen Capture",
            "Custom Script"
        ])
        create_layout.addWidget(self.task_type_combo, 1, 1)
        
        # Task description
        create_layout.addWidget(QLabel("Description:"), 2, 0)
        self.task_desc_edit = QTextEdit()
        self.task_desc_edit.setMaximumHeight(80)
        create_layout.addWidget(self.task_desc_edit, 2, 1)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.create_btn = QPushButton("Create Task")
        self.create_btn.clicked.connect(self.create_task)
        button_layout.addWidget(self.create_btn)
        
        self.load_btn = QPushButton("Load Tasks")
        self.load_btn.clicked.connect(self.load_tasks)
        button_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Tasks")
        self.save_btn.clicked.connect(self.save_tasks)
        button_layout.addWidget(self.save_btn)
        
        create_layout.addLayout(button_layout, 3, 0, 1, 2)
        layout.addWidget(create_group)
        
        # Task list
        list_group = QGroupBox("Available Tasks")
        list_layout = QVBoxLayout(list_group)
        
        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.execute_selected_task)
        list_layout.addWidget(self.task_list)
        
        # Task control buttons
        control_layout = QHBoxLayout()
        self.execute_btn = QPushButton("Execute Selected")
        self.execute_btn.clicked.connect(self.execute_selected_task)
        control_layout.addWidget(self.execute_btn)
        
        self.edit_btn = QPushButton("Edit Selected")
        self.edit_btn.clicked.connect(self.edit_selected_task)
        control_layout.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_task)
        control_layout.addWidget(self.delete_btn)
        
        list_layout.addLayout(control_layout)
        layout.addWidget(list_group)
    
    def create_task(self):
        """Create a new automation task"""
        name = self.task_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a task name.")
            return
        
        task = {
            'id': len(self.tasks) + 1,
            'name': name,
            'type': self.task_type_combo.currentText(),
            'description': self.task_desc_edit.toPlainText(),
            'created': datetime.now().isoformat(),
            'status': 'Ready'
        }
        
        self.tasks.append(task)
        self.update_task_list()
        
        # Clear form
        self.task_name_edit.clear()
        self.task_desc_edit.clear()
    
    def update_task_list(self):
        """Update the task list display"""
        self.task_list.clear()
        for task in self.tasks:
            item_text = f"{task['name']} ({task['type']}) - {task['status']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, task)
            self.task_list.addItem(item)
    
    def execute_selected_task(self):
        """Execute the selected task"""
        current_item = self.task_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a task to execute.")
            return
        
        task = current_item.data(Qt.UserRole)
        self.task_executed.emit(task)
    
    def edit_selected_task(self):
        """Edit the selected task"""
        current_item = self.task_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a task to edit.")
            return
        
        task = current_item.data(Qt.UserRole)
        # Populate form with task data
        self.task_name_edit.setText(task['name'])
        self.task_desc_edit.setPlainText(task['description'])
        
        # Find and set the task type
        index = self.task_type_combo.findText(task['type'])
        if index >= 0:
            self.task_type_combo.setCurrentIndex(index)
    
    def delete_selected_task(self):
        """Delete the selected task"""
        current_item = self.task_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a task to delete.")
            return
        
        task = current_item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete task '{task['name']}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.tasks = [t for t in self.tasks if t['id'] != task['id']]
            self.update_task_list()
    
    def load_tasks(self):
        """Load tasks from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Tasks", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    self.tasks = json.load(f)
                self.update_task_list()
                QMessageBox.information(self, "Success", "Tasks loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load tasks: {e}")
    
    def save_tasks(self):
        """Save tasks to file"""
        if not self.tasks:
            QMessageBox.warning(self, "Warning", "No tasks to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Tasks", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                import json
                with open(file_path, 'w') as f:
                    json.dump(self.tasks, f, indent=2)
                QMessageBox.information(self, "Success", "Tasks saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save tasks: {e}")


class SystemControlWidget(QGroupBox):
    """Widget for system control operations"""
    
    def __init__(self):
        super().__init__("System Control")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Power management
        power_group = QGroupBox("Power Management")
        power_layout = QGridLayout(power_group)
        
        self.shutdown_btn = QPushButton("Shutdown")
        self.shutdown_btn.clicked.connect(self.shutdown_system)
        power_layout.addWidget(self.shutdown_btn, 0, 0)
        
        self.restart_btn = QPushButton("Restart")
        self.restart_btn.clicked.connect(self.restart_system)
        power_layout.addWidget(self.restart_btn, 0, 1)
        
        self.sleep_btn = QPushButton("Sleep")
        self.sleep_btn.clicked.connect(self.sleep_system)
        power_layout.addWidget(self.sleep_btn, 0, 2)
        
        layout.addWidget(power_group)
        
        # Volume control
        volume_group = QGroupBox("Volume Control")
        volume_layout = QVBoxLayout(volume_group)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.set_volume)
        
        volume_layout.addWidget(QLabel("System Volume:"))
        volume_layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("50%")
        self.volume_label.setAlignment(Qt.AlignCenter)
        volume_layout.addWidget(self.volume_label)
        
        layout.addWidget(volume_group)
        
        # Process management
        process_group = QGroupBox("Process Management")
        process_layout = QVBoxLayout(process_group)
        
        process_control_layout = QHBoxLayout()
        self.process_name_edit = QLineEdit()
        self.process_name_edit.setPlaceholderText("Process name...")
        process_control_layout.addWidget(self.process_name_edit)
        
        self.kill_process_btn = QPushButton("Kill Process")
        self.kill_process_btn.clicked.connect(self.kill_process)
        process_control_layout.addWidget(self.kill_process_btn)
        
        process_layout.addLayout(process_control_layout)
        layout.addWidget(process_group)
    
    def shutdown_system(self):
        """Shutdown the system"""
        reply = QMessageBox.question(
            self, "Confirm Shutdown",
            "Are you sure you want to shutdown the system?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # In a real implementation, this would call the system controller
            QMessageBox.information(self, "Info", "Shutdown command would be executed.")
    
    def restart_system(self):
        """Restart the system"""
        reply = QMessageBox.question(
            self, "Confirm Restart",
            "Are you sure you want to restart the system?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # In a real implementation, this would call the system controller
            QMessageBox.information(self, "Info", "Restart command would be executed.")
    
    def sleep_system(self):
        """Put system to sleep"""
        # In a real implementation, this would call the system controller
        QMessageBox.information(self, "Info", "Sleep command would be executed.")
    
    def set_volume(self, value):
        """Set system volume"""
        self.volume_label.setText(f"{value}%")
        # In a real implementation, this would call the system controller
    
    def kill_process(self):
        """Kill a process by name"""
        process_name = self.process_name_edit.text().strip()
        if not process_name:
            QMessageBox.warning(self, "Warning", "Please enter a process name.")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Kill Process",
            f"Are you sure you want to kill process '{process_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # In a real implementation, this would call the system controller
            QMessageBox.information(self, "Info", f"Kill process '{process_name}' command would be executed.")


class AutomationWidget(QWidget):
    """Main automation widget"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        self.execution_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the automation UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Automation & System Control")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Task Manager tab
        self.task_manager = TaskManagerWidget()
        self.task_manager.task_executed.connect(self.execute_task)
        tab_widget.addTab(self.task_manager, "Task Manager")
        
        # System Control tab
        self.system_control = SystemControlWidget()
        tab_widget.addTab(self.system_control, "System Control")
        
        # Execution status
        status_group = QGroupBox("Execution Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        self.stop_btn = QPushButton("Stop Execution")
        self.stop_btn.clicked.connect(self.stop_execution)
        self.stop_btn.setEnabled(False)
        status_layout.addWidget(self.stop_btn)
        
        main_layout.addWidget(status_group)
        
        # Execution log
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_btn)
        
        main_layout.addWidget(log_group)
    
    def execute_task(self, task_data: Dict[str, Any]):
        """Execute an automation task"""
        try:
            if self.execution_thread and self.execution_thread.isRunning():
                QMessageBox.warning(self, "Warning", "Another task is already running.")
                return
            
            self.log_message(f"Starting task: {task_data['name']}")
            
            # Get automation engine from assistant manager
            automation_engine = None
            if self.assistant_manager and hasattr(self.assistant_manager, 'system_controller'):
                automation_engine = self.assistant_manager.system_controller
            
            # Create and start execution thread
            self.execution_thread = TaskExecutionThread(task_data, automation_engine)
            self.execution_thread.task_started.connect(self.on_task_started)
            self.execution_thread.task_progress.connect(self.on_task_progress)
            self.execution_thread.task_completed.connect(self.on_task_completed)
            self.execution_thread.task_error.connect(self.on_task_error)
            
            self.execution_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            self.log_message(f"Error: {e}")
    
    def on_task_started(self, task_name: str):
        """Handle task started signal"""
        self.status_label.setText(f"Executing: {task_name}")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(True)
        self.log_message(f"Task started: {task_name}")
    
    def on_task_progress(self, progress: int):
        """Handle task progress signal"""
        self.progress_bar.setValue(progress)
    
    def on_task_completed(self, task_name: str, success: bool):
        """Handle task completed signal"""
        self.status_label.setText("Ready")
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.log_message(f"Task completed successfully: {task_name}")
        else:
            self.log_message(f"Task was stopped: {task_name}")
    
    def on_task_error(self, task_name: str, error: str):
        """Handle task error signal"""
        self.status_label.setText("Error")
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.log_message(f"Task error in {task_name}: {error}")
        
        QMessageBox.critical(self, "Task Error", f"Error in task '{task_name}':\n{error}")
    
    def stop_execution(self):
        """Stop the current task execution"""
        if self.execution_thread and self.execution_thread.isRunning():
            self.execution_thread.stop()
            self.log_message("Stopping task execution...")
    
    def log_message(self, message: str):
        """Add a message to the execution log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
    
    def clear_log(self):
        """Clear the execution log"""
        self.log_text.clear()
"""
Window Manager Widget for Computer Assistant GUI
Provides interface for window management, monitoring, and control operations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QProgressBar, QSpinBox, QCheckBox, QTabWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSlider, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QIcon, QPixmap


class WindowRefreshThread(QThread):
    """Thread for refreshing window information"""
    
    windows_updated = Signal(list)
    error_occurred = Signal(str)
    
    def __init__(self, window_manager):
        super().__init__()
        self.window_manager = window_manager
        self.is_running = False
        
    def run(self):
        """Refresh window information"""
        try:
            self.is_running = True
            
            if self.window_manager:
                # Get windows from window manager
                windows = self.window_manager.get_all_windows()
                self.windows_updated.emit(windows)
            else:
                # Simulate window data for testing
                windows = [
                    {'title': 'Notepad', 'class': 'Notepad', 'pid': 1234, 'visible': True, 'hwnd': 12345},
                    {'title': 'Calculator', 'class': 'Calculator', 'pid': 5678, 'visible': True, 'hwnd': 67890},
                    {'title': 'File Explorer', 'class': 'CabinetWClass', 'pid': 9012, 'visible': True, 'hwnd': 34567}
                ]
                self.windows_updated.emit(windows)
                
        except Exception as e:
            self.error_occurred.emit(str(e))


class WindowListWidget(QGroupBox):
    """Widget displaying list of windows"""
    
    window_selected = Signal(dict)
    refresh_requested = Signal()
    
    def __init__(self):
        super().__init__("Active Windows")
        self.windows = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        control_layout.addWidget(self.refresh_btn)
        
        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        control_layout.addWidget(self.auto_refresh_cb)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter windows by title...")
        self.filter_edit.textChanged.connect(self.filter_windows)
        filter_layout.addWidget(self.filter_edit)
        
        layout.addLayout(filter_layout)
        
        # Window table
        self.window_table = QTableWidget()
        self.window_table.setColumnCount(5)
        self.window_table.setHorizontalHeaderLabels([
            "Title", "Class", "PID", "Visible", "Handle"
        ])
        
        # Set column widths
        header = self.window_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        self.window_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.window_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.window_table)
    
    def update_windows(self, windows: List[Dict[str, Any]]):
        """Update the window list"""
        self.windows = windows
        self.populate_table()
    
    def populate_table(self):
        """Populate the window table"""
        filter_text = self.filter_edit.text().lower()
        
        # Filter windows
        filtered_windows = []
        for window in self.windows:
            title = window.get('title', '').lower()
            if not filter_text or filter_text in title:
                filtered_windows.append(window)
        
        # Update table
        self.window_table.setRowCount(len(filtered_windows))
        
        for row, window in enumerate(filtered_windows):
            # Title
            title_item = QTableWidgetItem(window.get('title', 'Unknown'))
            title_item.setData(Qt.UserRole, window)
            self.window_table.setItem(row, 0, title_item)
            
            # Class
            class_item = QTableWidgetItem(window.get('class', 'Unknown'))
            self.window_table.setItem(row, 1, class_item)
            
            # PID
            pid_item = QTableWidgetItem(str(window.get('pid', 0)))
            self.window_table.setItem(row, 2, pid_item)
            
            # Visible
            visible_item = QTableWidgetItem("Yes" if window.get('visible', False) else "No")
            self.window_table.setItem(row, 3, visible_item)
            
            # Handle
            handle_item = QTableWidgetItem(str(window.get('hwnd', 0)))
            self.window_table.setItem(row, 4, handle_item)
    
    def filter_windows(self):
        """Filter windows based on search text"""
        self.populate_table()
    
    def on_selection_changed(self):
        """Handle window selection change"""
        current_row = self.window_table.currentRow()
        if current_row >= 0:
            title_item = self.window_table.item(current_row, 0)
            if title_item:
                window_data = title_item.data(Qt.UserRole)
                if window_data:
                    self.window_selected.emit(window_data)


class WindowControlWidget(QGroupBox):
    """Widget for window control operations"""
    
    def __init__(self):
        super().__init__("Window Control")
        self.selected_window = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Selected window info
        info_group = QGroupBox("Selected Window")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Title:"), 0, 0)
        self.title_label = QLabel("None selected")
        info_layout.addWidget(self.title_label, 0, 1)
        
        info_layout.addWidget(QLabel("Class:"), 1, 0)
        self.class_label = QLabel("-")
        info_layout.addWidget(self.class_label, 1, 1)
        
        info_layout.addWidget(QLabel("PID:"), 2, 0)
        self.pid_label = QLabel("-")
        info_layout.addWidget(self.pid_label, 2, 1)
        
        layout.addWidget(info_group)
        
        # Control buttons
        control_group = QGroupBox("Actions")
        control_layout = QGridLayout(control_group)
        
        # Window state controls
        self.minimize_btn = QPushButton("Minimize")
        self.minimize_btn.clicked.connect(self.minimize_window)
        control_layout.addWidget(self.minimize_btn, 0, 0)
        
        self.maximize_btn = QPushButton("Maximize")
        self.maximize_btn.clicked.connect(self.maximize_window)
        control_layout.addWidget(self.maximize_btn, 0, 1)
        
        self.restore_btn = QPushButton("Restore")
        self.restore_btn.clicked.connect(self.restore_window)
        control_layout.addWidget(self.restore_btn, 1, 0)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_window)
        control_layout.addWidget(self.close_btn, 1, 1)
        
        # Focus and visibility
        self.focus_btn = QPushButton("Bring to Front")
        self.focus_btn.clicked.connect(self.focus_window)
        control_layout.addWidget(self.focus_btn, 2, 0)
        
        self.hide_btn = QPushButton("Hide")
        self.hide_btn.clicked.connect(self.hide_window)
        control_layout.addWidget(self.hide_btn, 2, 1)
        
        # Position and size
        self.move_btn = QPushButton("Move Window")
        self.move_btn.clicked.connect(self.move_window)
        control_layout.addWidget(self.move_btn, 3, 0)
        
        self.resize_btn = QPushButton("Resize Window")
        self.resize_btn.clicked.connect(self.resize_window)
        control_layout.addWidget(self.resize_btn, 3, 1)
        
        layout.addWidget(control_group)
        
        # Position controls
        position_group = QGroupBox("Position & Size")
        position_layout = QGridLayout(position_group)
        
        # X, Y coordinates
        position_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_spin = QSpinBox()
        self.x_spin.setRange(-9999, 9999)
        position_layout.addWidget(self.x_spin, 0, 1)
        
        position_layout.addWidget(QLabel("Y:"), 0, 2)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(-9999, 9999)
        position_layout.addWidget(self.y_spin, 0, 3)
        
        # Width, Height
        position_layout.addWidget(QLabel("Width:"), 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 9999)
        self.width_spin.setValue(800)
        position_layout.addWidget(self.width_spin, 1, 1)
        
        position_layout.addWidget(QLabel("Height:"), 1, 2)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 9999)
        self.height_spin.setValue(600)
        position_layout.addWidget(self.height_spin, 1, 3)
        
        # Apply button
        self.apply_position_btn = QPushButton("Apply Position/Size")
        self.apply_position_btn.clicked.connect(self.apply_position_size)
        position_layout.addWidget(self.apply_position_btn, 2, 0, 1, 4)
        
        layout.addWidget(position_group)
        
        # Initially disable all controls
        self.set_controls_enabled(False)
    
    def set_selected_window(self, window_data: Dict[str, Any]):
        """Set the selected window"""
        self.selected_window = window_data
        
        # Update info labels
        self.title_label.setText(window_data.get('title', 'Unknown'))
        self.class_label.setText(window_data.get('class', 'Unknown'))
        self.pid_label.setText(str(window_data.get('pid', 0)))
        
        # Enable controls
        self.set_controls_enabled(True)
    
    def set_controls_enabled(self, enabled: bool):
        """Enable or disable control buttons"""
        for widget in self.findChildren(QPushButton):
            widget.setEnabled(enabled)
    
    def minimize_window(self):
        """Minimize the selected window"""
        if self.selected_window:
            QMessageBox.information(self, "Action", f"Minimizing window: {self.selected_window['title']}")
    
    def maximize_window(self):
        """Maximize the selected window"""
        if self.selected_window:
            QMessageBox.information(self, "Action", f"Maximizing window: {self.selected_window['title']}")
    
    def restore_window(self):
        """Restore the selected window"""
        if self.selected_window:
            QMessageBox.information(self, "Action", f"Restoring window: {self.selected_window['title']}")
    
    def close_window(self):
        """Close the selected window"""
        if self.selected_window:
            reply = QMessageBox.question(
                self, "Confirm Close",
                f"Are you sure you want to close '{self.selected_window['title']}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                QMessageBox.information(self, "Action", f"Closing window: {self.selected_window['title']}")
    
    def focus_window(self):
        """Bring window to front"""
        if self.selected_window:
            QMessageBox.information(self, "Action", f"Bringing to front: {self.selected_window['title']}")
    
    def hide_window(self):
        """Hide the selected window"""
        if self.selected_window:
            QMessageBox.information(self, "Action", f"Hiding window: {self.selected_window['title']}")
    
    def move_window(self):
        """Move the selected window"""
        if self.selected_window:
            x = self.x_spin.value()
            y = self.y_spin.value()
            QMessageBox.information(self, "Action", f"Moving window to ({x}, {y})")
    
    def resize_window(self):
        """Resize the selected window"""
        if self.selected_window:
            width = self.width_spin.value()
            height = self.height_spin.value()
            QMessageBox.information(self, "Action", f"Resizing window to {width}x{height}")
    
    def apply_position_size(self):
        """Apply position and size changes"""
        if self.selected_window:
            x = self.x_spin.value()
            y = self.y_spin.value()
            width = self.width_spin.value()
            height = self.height_spin.value()
            QMessageBox.information(
                self, "Action", 
                f"Setting window position to ({x}, {y}) and size to {width}x{height}"
            )


class WindowStatsWidget(QGroupBox):
    """Widget showing window statistics"""
    
    def __init__(self):
        super().__init__("Window Statistics")
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout(self)
        
        # Total windows
        layout.addWidget(QLabel("Total Windows:"), 0, 0)
        self.total_label = QLabel("0")
        layout.addWidget(self.total_label, 0, 1)
        
        # Visible windows
        layout.addWidget(QLabel("Visible Windows:"), 1, 0)
        self.visible_label = QLabel("0")
        layout.addWidget(self.visible_label, 1, 1)
        
        # Hidden windows
        layout.addWidget(QLabel("Hidden Windows:"), 2, 0)
        self.hidden_label = QLabel("0")
        layout.addWidget(self.hidden_label, 2, 1)
        
        # Minimized windows
        layout.addWidget(QLabel("Minimized Windows:"), 3, 0)
        self.minimized_label = QLabel("0")
        layout.addWidget(self.minimized_label, 3, 1)
    
    def update_stats(self, windows: List[Dict[str, Any]]):
        """Update window statistics"""
        total = len(windows)
        visible = sum(1 for w in windows if w.get('visible', False))
        hidden = total - visible
        
        self.total_label.setText(str(total))
        self.visible_label.setText(str(visible))
        self.hidden_label.setText(str(hidden))
        # Note: Minimized count would need additional window state info
        self.minimized_label.setText("N/A")


class WindowManagerWidget(QWidget):
    """Main window manager widget"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        self.refresh_thread = None
        
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        """Initialize the window manager UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Window Manager")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for layout
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Window list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Window list
        self.window_list = WindowListWidget()
        self.window_list.window_selected.connect(self.on_window_selected)
        self.window_list.refresh_requested.connect(self.refresh_windows)
        left_layout.addWidget(self.window_list)
        
        # Window statistics
        self.window_stats = WindowStatsWidget()
        left_layout.addWidget(self.window_stats)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Window control
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Window control
        self.window_control = WindowControlWidget()
        right_layout.addWidget(self.window_control)
        
        right_layout.addStretch()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([500, 300])
        
        # Initial window refresh
        self.refresh_windows()
    
    def setup_timers(self):
        """Setup update timers"""
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_check)
        self.auto_refresh_timer.start(5000)  # Check every 5 seconds
    
    def refresh_windows(self):
        """Refresh the window list"""
        try:
            if self.refresh_thread and self.refresh_thread.isRunning():
                return
            
            # Get window manager from assistant manager
            window_manager = None
            if self.assistant_manager and hasattr(self.assistant_manager, 'system_controller'):
                system_controller = self.assistant_manager.system_controller
                if hasattr(system_controller, 'window_manager'):
                    window_manager = system_controller.window_manager
            
            # Create and start refresh thread
            self.refresh_thread = WindowRefreshThread(window_manager)
            self.refresh_thread.windows_updated.connect(self.on_windows_updated)
            self.refresh_thread.error_occurred.connect(self.on_refresh_error)
            
            self.refresh_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error refreshing windows: {e}")
    
    def auto_refresh_check(self):
        """Check if auto refresh is enabled and refresh if needed"""
        if self.window_list.auto_refresh_cb.isChecked():
            self.refresh_windows()
    
    def on_windows_updated(self, windows: List[Dict[str, Any]]):
        """Handle updated window list"""
        self.window_list.update_windows(windows)
        self.window_stats.update_stats(windows)
    
    def on_refresh_error(self, error: str):
        """Handle refresh error"""
        self.logger.error(f"Window refresh error: {error}")
        QMessageBox.warning(self, "Refresh Error", f"Failed to refresh windows: {error}")
    
    def on_window_selected(self, window_data: Dict[str, Any]):
        """Handle window selection"""
        self.window_control.set_selected_window(window_data)
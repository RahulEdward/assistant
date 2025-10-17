"""
Main Window for Computer Assistant GUI
Comprehensive PySide6-based interface with tabbed layout and real-time functionality
"""

import sys
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QStatusBar, QToolBar, QSplitter,
    QTextEdit, QLabel, QPushButton, QProgressBar, QSystemTrayIcon,
    QMenu, QMessageBox, QFrame, QGridLayout, QGroupBox,
    QListWidget, QTreeWidget, QTreeWidgetItem, QTableWidget,
    QTableWidgetItem, QComboBox, QSpinBox, QCheckBox, QSlider,
    QLineEdit, QPlainTextEdit, QScrollArea
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QSize, QSettings, QRect,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
)
from PySide6.QtGui import (
    QIcon, QPixmap, QFont, QColor, QPalette, QAction,
    QKeySequence, QTextCursor, QTextCharFormat, QBrush
)

from .widgets.dashboard_widget import DashboardWidget
from .widgets.automation_widget import AutomationWidget
from .widgets.voice_widget import VoiceWidget
from .widgets.ai_chat_widget import AIChatWidget
from .widgets.window_manager_widget import WindowManagerWidget
from .widgets.system_monitor_widget import SystemMonitorWidget
from .widgets.settings_widget import SettingsWidget
from .widgets.log_viewer_widget import LogViewerWidget
from .widgets.theme_toggle_button import ThemeToggleWidget
from .theme_manager import theme_manager
from .theme_utils import theme_applicator


class MainWindow(QMainWindow):
    """Main application window with comprehensive GUI interface"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        
        # Window properties
        self.setWindowTitle("Computer Assistant - AI Desktop Automation")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Settings
        self.settings = QSettings("ComputerAssistant", "MainApp")
        
        # UI Components
        self.central_widget = None
        self.tab_widget = None
        self.status_bar = None
        self.menu_bar = None
        self.tool_bar = None
        self.system_tray = None
        
        # Tab widgets
        self.dashboard_widget = None
        self.automation_widget = None
        self.voice_widget = None
        self.ai_chat_widget = None
        self.window_manager_widget = None
        self.system_monitor_widget = None
        self.settings_widget = None
        self.log_viewer_widget = None
        
        # Theme toggle widget
        self.theme_toggle_widget = None
        
        # Timers and threads
        self.update_timer = QTimer()
        self.status_timer = QTimer()
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        self.restore_settings()
        
        # Apply initial theme
        self.apply_theme_styling()
        
    def init_ui(self):
        """Initialize the user interface"""
        try:
            # Set application style
            self.setStyleSheet(self.get_application_stylesheet())
            
            # Create central widget and layout
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            
            # Main layout
            main_layout = QVBoxLayout(self.central_widget)
            main_layout.setContentsMargins(5, 5, 5, 5)
            
            # Create menu bar
            self.create_menu_bar()
            
            # Create tool bar
            self.create_tool_bar()
            
            # Create tab widget
            self.create_tab_widget()
            main_layout.addWidget(self.tab_widget)
            
            # Create status bar
            self.create_status_bar()
            
            # Create system tray
            self.create_system_tray()
            
            self.logger.info("Main window UI initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize UI: {e}")
            QMessageBox.critical(self, "Error", f"Failed to initialize interface: {e}")
    
    def create_menu_bar(self):
        """Create the application menu bar"""
        self.menu_bar = self.menuBar()
        
        # File menu
        file_menu = self.menu_bar.addMenu("&File")
        
        new_action = QAction("&New Session", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_session)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Configuration", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_configuration)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Configuration", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = self.menu_bar.addMenu("&Tools")
        
        automation_action = QAction("&Automation Manager", self)
        automation_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        tools_menu.addAction(automation_action)
        
        voice_action = QAction("&Voice Control", self)
        voice_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        tools_menu.addAction(voice_action)
        
        window_action = QAction("&Window Manager", self)
        window_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(4))
        tools_menu.addAction(window_action)
        
        # View menu
        view_menu = self.menu_bar.addMenu("&View")
        
        dashboard_action = QAction("&Dashboard", self)
        dashboard_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        view_menu.addAction(dashboard_action)
        
        logs_action = QAction("&Logs", self)
        logs_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(7))
        view_menu.addAction(logs_action)
        
        fullscreen_action = QAction("&Full Screen", self)
        fullscreen_action.setShortcut(QKeySequence.FullScreen)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = self.menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        help_action = QAction("&Help Documentation", self)
        help_action.setShortcut(QKeySequence.HelpContents)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
    
    def create_tool_bar(self):
        """Create the application tool bar"""
        self.tool_bar = self.addToolBar("Main")
        self.tool_bar.setMovable(False)
        
        # Quick access buttons
        dashboard_btn = QPushButton("Dashboard")
        dashboard_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0))
        self.tool_bar.addWidget(dashboard_btn)
        
        automation_btn = QPushButton("Automation")
        automation_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        self.tool_bar.addWidget(automation_btn)
        
        voice_btn = QPushButton("Voice")
        voice_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        self.tool_bar.addWidget(voice_btn)
        
        ai_btn = QPushButton("AI Chat")
        ai_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
        self.tool_bar.addWidget(ai_btn)
        
        self.tool_bar.addSeparator()
        
        # Status indicators
        self.status_label = QLabel("Ready")
        self.tool_bar.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.tool_bar.addWidget(self.progress_bar)
        
        # Add spacer to push theme toggle to the right
        spacer = QWidget()
        spacer.setSizePolicy(QWidget.SizePolicy.Expanding, QWidget.SizePolicy.Preferred)
        self.tool_bar.addWidget(spacer)
        
        # Theme toggle widget
        self.theme_toggle_widget = ThemeToggleWidget()
        self.theme_toggle_widget.theme_changed.connect(self.on_theme_changed)
        self.tool_bar.addWidget(self.theme_toggle_widget)
    
    def create_tab_widget(self):
        """Create the main tab widget with all functional tabs"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        
        # Dashboard tab
        self.dashboard_widget = DashboardWidget(self.assistant_manager)
        self.tab_widget.addTab(self.dashboard_widget, "Dashboard")
        
        # Automation tab
        self.automation_widget = AutomationWidget(self.assistant_manager)
        self.tab_widget.addTab(self.automation_widget, "Automation")
        
        # Voice Control tab
        self.voice_widget = VoiceWidget(self.assistant_manager)
        self.tab_widget.addTab(self.voice_widget, "Voice Control")
        
        # AI Chat tab
        self.ai_chat_widget = AIChatWidget(self.assistant_manager)
        self.tab_widget.addTab(self.ai_chat_widget, "AI Chat")
        
        # Window Manager tab
        self.window_manager_widget = WindowManagerWidget(self.assistant_manager)
        self.tab_widget.addTab(self.window_manager_widget, "Window Manager")
        
        # System Monitor tab
        self.system_monitor_widget = SystemMonitorWidget(self.assistant_manager)
        self.tab_widget.addTab(self.system_monitor_widget, "System Monitor")
        
        # Settings tab
        self.settings_widget = SettingsWidget(self.assistant_manager)
        self.tab_widget.addTab(self.settings_widget, "Settings")
        
        # Log Viewer tab
        self.log_viewer_widget = LogViewerWidget(self.assistant_manager)
        self.tab_widget.addTab(self.log_viewer_widget, "Logs")
    
    def create_status_bar(self):
        """Create the application status bar"""
        self.status_bar = self.statusBar()
        
        # Status message
        self.status_message = QLabel("Ready")
        self.status_bar.addWidget(self.status_message)
        
        # Connection status
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red;")
        self.status_bar.addPermanentWidget(self.connection_status)
        
        # Performance indicator
        self.performance_label = QLabel("CPU: 0% | RAM: 0%")
        self.status_bar.addPermanentWidget(self.performance_label)
        
        # Time
        self.time_label = QLabel()
        self.update_time()
        self.status_bar.addPermanentWidget(self.time_label)
    
    def create_system_tray(self):
        """Create system tray icon and menu"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.system_tray = QSystemTrayIcon(self)
            
            # Create tray menu
            tray_menu = QMenu()
            
            show_action = QAction("Show", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)
            
            hide_action = QAction("Hide", self)
            hide_action.triggered.connect(self.hide)
            tray_menu.addAction(hide_action)
            
            tray_menu.addSeparator()
            
            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.close)
            tray_menu.addAction(quit_action)
            
            self.system_tray.setContextMenu(tray_menu)
            self.system_tray.activated.connect(self.tray_icon_activated)
            self.system_tray.show()
    
    def setup_connections(self):
        """Setup signal connections and timers"""
        # Update timer for real-time data
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(1000)  # Update every second
        
        # Status timer for time updates
        self.status_timer.timeout.connect(self.update_time)
        self.status_timer.start(1000)
        
        # Tab change connections
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Theme manager connections
        theme_manager.theme_changed.connect(self.on_theme_changed)
    
    def get_application_stylesheet(self):
        """Get the application stylesheet for modern UI"""
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #3c3c3c;
        }
        
        QTabWidget::tab-bar {
            alignment: left;
        }
        
        QTabBar::tab {
            background-color: #404040;
            color: #ffffff;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: #0078d4;
        }
        
        QTabBar::tab:hover {
            background-color: #505050;
        }
        
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #106ebe;
        }
        
        QPushButton:pressed {
            background-color: #005a9e;
        }
        
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 2px;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 4px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QStatusBar {
            background-color: #404040;
            color: #ffffff;
        }
        
        QMenuBar {
            background-color: #404040;
            color: #ffffff;
        }
        
        QMenuBar::item:selected {
            background-color: #0078d4;
        }
        
        QMenu {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
        }
        
        QMenu::item:selected {
            background-color: #0078d4;
        }
        
        QToolBar {
            background-color: #404040;
            border: none;
            spacing: 3px;
        }
        """
    
    def update_data(self):
        """Update real-time data across all widgets"""
        try:
            # Update connection status
            if self.assistant_manager and hasattr(self.assistant_manager, 'is_initialized'):
                if self.assistant_manager.is_initialized:
                    self.connection_status.setText("Connected")
                    self.connection_status.setStyleSheet("color: green;")
                else:
                    self.connection_status.setText("Disconnected")
                    self.connection_status.setStyleSheet("color: red;")
            
            # Update performance info
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            self.performance_label.setText(f"CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}%")
            
            # Update active tab widget
            current_widget = self.tab_widget.currentWidget()
            if hasattr(current_widget, 'update_data'):
                current_widget.update_data()
                
        except Exception as e:
            self.logger.error(f"Error updating data: {e}")
    
    def update_time(self):
        """Update the time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
    
    def on_tab_changed(self, index):
        """Handle tab change events"""
        tab_names = ["Dashboard", "Automation", "Voice Control", "AI Chat", 
                    "Window Manager", "System Monitor", "Settings", "Logs"]
        if 0 <= index < len(tab_names):
            self.status_message.setText(f"Switched to {tab_names[index]}")
    
    def tray_icon_activated(self, reason):
        """Handle system tray icon activation"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.raise_()
                self.activateWindow()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def new_session(self):
        """Start a new session"""
        self.status_message.setText("Starting new session...")
        # Implementation for new session
    
    def open_configuration(self):
        """Open configuration file"""
        self.status_message.setText("Opening configuration...")
        # Implementation for opening configuration
    
    def save_configuration(self):
        """Save current configuration"""
        self.status_message.setText("Saving configuration...")
        # Implementation for saving configuration
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Computer Assistant",
                         "Computer Assistant v1.0\n\n"
                         "AI-powered desktop automation with voice control,\n"
                         "system management, and intelligent task execution.\n\n"
                         "Built with PySide6 and advanced AI integration.")
    
    def show_help(self):
        """Show help documentation"""
        self.status_message.setText("Opening help documentation...")
        # Implementation for showing help
    
    def restore_settings(self):
        """Restore window settings"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def save_settings(self):
        """Save window settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    def on_theme_changed(self, theme_name):
        """Handle theme change events"""
        self.logger.info(f"Theme changed to: {theme_name}")
        
        # Apply theme-specific styling to main window components
        self.apply_theme_styling()
        
        # Update status bar message
        self.status_bar.showMessage(f"Theme switched to {theme_name} mode", 2000)
    
    def apply_theme_styling(self):
        """Apply theme styling to main window and all child widgets"""
        colors = theme_manager.get_current_colors()
        
        # Apply main window background
        self.setStyleSheet(f"""
        QMainWindow {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
        }}
        """)
        
        # Apply toolbar styling
        if hasattr(self, 'tool_bar') and self.tool_bar:
            toolbar_style = f"""
            QToolBar {{
                background-color: {colors['surface']};
                border: none;
                border-bottom: 1px solid {colors['border']};
                spacing: 8px;
                padding: 4px;
            }}
            
            QToolBar QToolButton {{
                background-color: transparent;
                color: {colors['text_primary']};
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
            }}
            
            QToolBar QToolButton:hover {{
                background-color: {colors['hover']};
            }}
            
            QToolBar QToolButton:pressed {{
                background-color: {colors['selected']};
            }}
            """
            self.tool_bar.setStyleSheet(toolbar_style)
        
        # Apply status bar styling
        if hasattr(self, 'status_bar') and self.status_bar:
            status_style = f"""
            QStatusBar {{
                background-color: {colors['surface']};
                color: {colors['text_secondary']};
                border-top: 1px solid {colors['border']};
                padding: 4px;
            }}
            
            QStatusBar QLabel {{
                color: {colors['text_secondary']};
                background-color: transparent;
            }}
            """
            self.status_bar.setStyleSheet(status_style)
        
        # Apply theme to all child widgets using theme applicator
        theme_applicator.apply_theme_to_widget(self, recursive=True)
        
        # Apply theme to tab widget specifically
        if hasattr(self, 'tab_widget') and self.tab_widget:
            tab_style = f"""
            QTabWidget::pane {{
                border: 1px solid {colors['border']};
                background-color: {colors['background']};
                border-radius: 4px;
            }}
            
            QTabBar::tab {{
                background-color: {colors['surface']};
                color: {colors['text_secondary']};
                border: 1px solid {colors['border']};
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {colors['primary']};
                color: {colors['text_on_primary']};
                border-bottom: none;
                font-weight: 500;
            }}
            
            QTabBar::tab:hover:!selected {{
                background-color: {colors['hover']};
                color: {colors['text_primary']};
            }}
            
            QTabBar::tab:focus {{
                outline: 2px solid {colors['focus']};
                outline-offset: -2px;
            }}
            """
            self.tab_widget.setStyleSheet(tab_style)
        
        logger.info(f"Applied {theme_manager.current_theme} theme styling to main window and all widgets")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.system_tray and self.system_tray.isVisible():
            self.hide()
            event.ignore()
        else:
            self.save_settings()
            event.accept()
    
    async def run(self):
        """Run the GUI application"""
        try:
            self.show()
            self.logger.info("GUI application started successfully")
            
            # Keep the application running
            app = QApplication.instance()
            if app:
                while True:
                    app.processEvents()
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            self.logger.error(f"Error running GUI application: {e}")
    
    async def cleanup(self):
        """Cleanup GUI resources"""
        try:
            self.update_timer.stop()
            self.status_timer.stop()
            
            if self.system_tray:
                self.system_tray.hide()
            
            self.save_settings()
            self.logger.info("GUI cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during GUI cleanup: {e}")
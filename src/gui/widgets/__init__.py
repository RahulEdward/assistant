"""GUI Widgets Package for Computer Assistant
Contains all custom widgets used in the PySide6 interface
"""

from .dashboard_widget import DashboardWidget
from .automation_widget import AutomationWidget
from .voice_widget import VoiceWidget
from .ai_chat_widget import AIChatWidget
from .window_manager_widget import WindowManagerWidget
from .system_monitor_widget import SystemMonitorWidget
from .ocr_widget import OCRWidget
from .settings_widget import SettingsWidget
from .log_viewer_widget import LogViewerWidget

__all__ = [
    'DashboardWidget',
    'AutomationWidget', 
    'VoiceWidget',
    'AIChatWidget',
    'WindowManagerWidget',
    'SystemMonitorWidget',
    'OCRWidget',
    'SettingsWidget',
    'LogViewerWidget'
]

__version__ = "1.0.0"
__author__ = "Computer Assistant Team"
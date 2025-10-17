"""
GUI Package for Computer Assistant

This package provides the graphical user interface components for the computer assistant,
including the main window, widgets, and dialogs.
"""

from .main_window import MainWindow
from .widgets import *

__version__ = "1.0.0"
__author__ = "Computer Assistant Team"

__all__ = [
    'MainWindow',
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
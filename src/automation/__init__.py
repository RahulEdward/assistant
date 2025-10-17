"""
Automation Package
System-level automation framework for Windows applications and browser control.
Provides human-like execution speed with machine precision.
"""

__version__ = "1.0.0"
__author__ = "Desktop Assistant"

# Package components
from .system_controller import SystemController
from .application_controller import ApplicationController
from .browser_controller import BrowserController
from .window_manager import WindowManager
from .automation_engine import AutomationEngine

__all__ = [
    'SystemController',
    'ApplicationController', 
    'BrowserController',
    'WindowManager',
    'AutomationEngine'
]
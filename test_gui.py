"""
Test script for the Computer Assistant GUI
This script tests the GUI components independently without complex dependencies
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

# Import GUI components
from src.gui.widgets.dashboard_widget import DashboardWidget
from src.gui.widgets.automation_widget import AutomationWidget
from src.gui.widgets.voice_widget import VoiceWidget
from src.gui.widgets.ai_chat_widget import AIChatWidget
from src.gui.widgets.window_manager_widget import WindowManagerWidget
from src.gui.widgets.system_monitor_widget import SystemMonitorWidget
from src.gui.widgets.ocr_widget import OCRWidget
from src.gui.widgets.settings_widget import SettingsWidget
from src.gui.widgets.log_viewer_widget import LogViewerWidget


class MockAssistantManager:
    """Mock assistant manager for testing"""
    
    def __init__(self):
        self.config = {}
        self.voice_processor = None
        self.ai_manager = None
        self.automation_engine = None
        self.window_manager = None
        self.system_controller = None
        self.ocr_engine = None
        
    def get_system_stats(self):
        """Mock system stats"""
        return {
            'cpu_percent': 45.2,
            'memory_percent': 67.8,
            'disk_percent': 23.1,
            'network_sent': 1024000,
            'network_recv': 2048000
        }
    
    def get_recent_activities(self):
        """Mock recent activities"""
        return [
            "Voice command processed: 'open calculator'",
            "Window minimized: Notepad",
            "AI response generated",
            "System monitoring started",
            "OCR scan completed"
        ]
    
    def get_component_status(self):
        """Mock component status"""
        return {
            'assistant': True,
            'voice_recognition': True,
            'ai_integration': True,
            'automation': True
        }


class TestMainWindow(QMainWindow):
    """Test main window for GUI components"""
    
    def __init__(self):
        super().__init__()
        self.assistant_manager = MockAssistantManager()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the test UI"""
        self.setWindowTitle("Computer Assistant GUI Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add tabs with widgets
        try:
            # Dashboard
            dashboard = DashboardWidget(self.assistant_manager)
            self.tab_widget.addTab(dashboard, "Dashboard")
            
            # Automation
            automation = AutomationWidget(self.assistant_manager)
            self.tab_widget.addTab(automation, "Automation")
            
            # Voice Control
            voice = VoiceWidget(self.assistant_manager)
            self.tab_widget.addTab(voice, "Voice Control")
            
            # AI Chat
            ai_chat = AIChatWidget(self.assistant_manager)
            self.tab_widget.addTab(ai_chat, "AI Chat")
            
            # Window Manager
            window_manager = WindowManagerWidget(self.assistant_manager)
            self.tab_widget.addTab(window_manager, "Window Manager")
            
            # System Monitor
            system_monitor = SystemMonitorWidget(self.assistant_manager)
            self.tab_widget.addTab(system_monitor, "System Monitor")
            
            # OCR
            ocr = OCRWidget(self.assistant_manager)
            self.tab_widget.addTab(ocr, "OCR")
            
            # Settings
            settings = SettingsWidget(self.assistant_manager)
            self.tab_widget.addTab(settings, "Settings")
            
            # Log Viewer
            log_viewer = LogViewerWidget(self.assistant_manager)
            self.tab_widget.addTab(log_viewer, "Log Viewer")
            
            print("✅ All GUI components loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading GUI components: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the GUI test"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Computer Assistant GUI Test")
    
    # Create and show main window
    window = TestMainWindow()
    window.show()
    
    print("🚀 GUI Test Application Started!")
    print("📋 Available tabs:")
    print("   - Dashboard: System overview and quick actions")
    print("   - Automation: Task management and system control")
    print("   - Voice Control: Speech recognition and TTS")
    print("   - AI Chat: Conversation interface")
    print("   - Window Manager: Window control and management")
    print("   - System Monitor: Performance metrics and processes")
    print("   - OCR: Optical character recognition")
    print("   - Settings: Configuration management")
    print("   - Log Viewer: Real-time log monitoring")
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
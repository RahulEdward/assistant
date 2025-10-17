"""
Windows Desktop Assistant - Main Application Entry Point
A comprehensive AI-powered desktop assistant with voice/text input, system automation,
and advanced capabilities including OCR, financial analysis, and code generation.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.assistant_manager import AssistantManager
from src.core.config_manager import ConfigManager
from src.gui.main_window import MainWindow
from src.utils.logger import setup_logger


class DesktopAssistant:
    """Main Desktop Assistant Application"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.logger = setup_logger("DesktopAssistant")
        self.assistant_manager: Optional[AssistantManager] = None
        self.main_window: Optional[MainWindow] = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing Desktop Assistant...")
            
            # Initialize core components
            self.assistant_manager = AssistantManager(self.config_manager)
            await self.assistant_manager.initialize()
            
            # Initialize GUI
            self.main_window = MainWindow(self.assistant_manager)
            
            self.logger.info("Desktop Assistant initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Desktop Assistant: {e}")
            return False
    
    async def run(self):
        """Run the main application"""
        if not await self.initialize():
            self.logger.error("Failed to initialize application")
            return False
            
        try:
            self.logger.info("Starting Desktop Assistant...")
            
            # Start the GUI
            if self.main_window:
                await self.main_window.run()
                
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        if self.assistant_manager:
            await self.assistant_manager.cleanup()
            
        if self.main_window:
            await self.main_window.cleanup()


def main():
    """Main entry point"""
    # Ensure we're running on Windows
    if sys.platform != "win32":
        print("This application is designed for Windows only.")
        sys.exit(1)
    
    # Create and run the application
    app = DesktopAssistant()
    
    try:
        # Run the async application
        asyncio.run(app.run())
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
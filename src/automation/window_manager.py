"""
Window Manager
Advanced window management for Windows desktop applications.
Provides comprehensive window manipulation, monitoring, and automation capabilities.
"""

import logging
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

# Windows API imports
try:
    import win32gui
    import win32con
    import win32api
    import win32process
    import win32clipboard
    import win32ui
    import win32print
    from win32com.shell import shell, shellcon
    import ctypes
    from ctypes import wintypes, windll
    import psutil
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False


@dataclass
class WindowInfo:
    """Window information structure"""
    hwnd: int
    title: str
    class_name: str
    process_id: int
    process_name: str
    executable_path: str
    is_visible: bool
    is_enabled: bool
    is_minimized: bool
    is_maximized: bool
    rect: Tuple[int, int, int, int]  # left, top, right, bottom
    client_rect: Tuple[int, int, int, int]
    parent_hwnd: int
    owner_hwnd: int
    style: int
    ex_style: int
    thread_id: int


@dataclass
class MonitorInfo:
    """Monitor information structure"""
    handle: int
    rect: Tuple[int, int, int, int]  # left, top, right, bottom
    work_rect: Tuple[int, int, int, int]  # work area
    is_primary: bool
    device_name: str
    width: int
    height: int
    dpi: int


@dataclass
class WindowEvent:
    """Window event information"""
    event_type: str
    hwnd: int
    window_title: str
    timestamp: datetime
    details: Dict[str, Any]


class WindowManager:
    """Advanced window management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Window tracking
        self.windows = {}  # hwnd -> WindowInfo
        self.window_history = []  # List of WindowEvent
        self.active_window = None
        
        # Monitor information
        self.monitors = {}  # handle -> MonitorInfo
        
        # Event callbacks
        self.event_callbacks = {}  # event_type -> List[callback]
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # seconds
        
        # Performance tracking
        self.operation_stats = {
            'windows_enumerated': 0,
            'windows_manipulated': 0,
            'events_processed': 0,
            'screenshots_taken': 0
        }
        
        # Window filters
        self.window_filters = {
            'min_title_length': 1,
            'exclude_invisible': True,
            'exclude_system': True,
            'exclude_empty_title': True
        }
        
        # System constants
        self.system_classes = {
            'Shell_TrayWnd',  # Taskbar
            'Progman',        # Desktop
            'WorkerW',        # Desktop worker
            'DV2ControlHost', # Start menu
            'Windows.UI.Core.CoreWindow'  # Modern apps
        }
    
    async def initialize(self):
        """Initialize window manager"""
        try:
            self.logger.info("Initializing Window Manager...")
            
            if not WIN32_AVAILABLE:
                self.logger.error("Windows API not available")
                return False
            
            # Initialize monitors
            await self._enumerate_monitors()
            
            # Initialize windows
            await self._enumerate_windows()
            
            # Start monitoring
            await self.start_monitoring()
            
            self.logger.info("Window Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Window Manager initialization error: {e}")
            return False
    
    async def _enumerate_monitors(self):
        """Enumerate all monitors"""
        try:
            self.monitors.clear()
            
            # Enumerate monitors - using direct call without callback
            monitors = win32api.EnumDisplayMonitors()
            
            for monitor_data in monitors:
                try:
                    hmonitor = monitor_data[0]
                    
                    # Get monitor info
                    monitor_info = win32api.GetMonitorInfo(hmonitor)
                    
                    # Get device name
                    device_name = monitor_info.get('Device', 'Unknown')
                    
                    # Calculate dimensions
                    monitor_rect = monitor_info['Monitor']
                    work_rect = monitor_info['Work']
                    
                    width = monitor_rect[2] - monitor_rect[0]
                    height = monitor_rect[3] - monitor_rect[1]
                    
                    # Get DPI
                    try:
                        hdc = win32gui.GetDC(0)
                        dpi = win32print.GetDeviceCaps(hdc, win32con.LOGPIXELSX)
                        win32gui.ReleaseDC(0, hdc)
                    except:
                        dpi = 96  # Default DPI
                    
                    # Check if primary
                    is_primary = monitor_info['Flags'] & win32con.MONITORINFOF_PRIMARY != 0
                    
                    monitor = MonitorInfo(
                        handle=hmonitor,
                        rect=monitor_rect,
                        work_rect=work_rect,
                        is_primary=is_primary,
                        device_name=device_name,
                        width=width,
                        height=height,
                        dpi=dpi
                    )
                    
                    self.monitors[hmonitor] = monitor
                    
                except Exception as e:
                    self.logger.warning(f"Error processing monitor {hmonitor}: {e}")
            
            self.logger.info(f"Enumerated {len(self.monitors)} monitors")
            
        except Exception as e:
            self.logger.error(f"Monitor enumeration error: {e}")
    
    async def _enumerate_windows(self):
        """Enumerate all windows"""
        try:
            self.windows.clear()
            
            def window_enum_proc(hwnd, data):
                try:
                    window_info = self._get_window_info(hwnd)
                    if window_info and self._should_include_window(window_info):
                        self.windows[hwnd] = window_info
                except Exception as e:
                    self.logger.debug(f"Error processing window {hwnd}: {e}")
                return True
            
            # Enumerate windows
            win32gui.EnumWindows(window_enum_proc, None)
            
            # Update active window
            try:
                active_hwnd = win32gui.GetForegroundWindow()
                if active_hwnd in self.windows:
                    self.active_window = active_hwnd
            except:
                pass
            
            self.operation_stats['windows_enumerated'] += len(self.windows)
            self.logger.debug(f"Enumerated {len(self.windows)} windows")
            
        except Exception as e:
            self.logger.error(f"Window enumeration error: {e}")
    
    def _get_window_info(self, hwnd: int) -> Optional[WindowInfo]:
        """Get detailed window information"""
        try:
            # Basic window properties
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            
            # Window state
            is_visible = win32gui.IsWindowVisible(hwnd)
            is_enabled = win32gui.IsWindowEnabled(hwnd)
            
            # Window placement
            try:
                placement = win32gui.GetWindowPlacement(hwnd)
                is_minimized = placement[1] == win32con.SW_SHOWMINIMIZED
                is_maximized = placement[1] == win32con.SW_SHOWMAXIMIZED
            except:
                is_minimized = False
                is_maximized = False
            
            # Window rectangles
            try:
                rect = win32gui.GetWindowRect(hwnd)
                # Convert client rect to screen coordinates
                client_rect_raw = win32gui.GetClientRect(hwnd)
                # Client rect is relative to window, convert to absolute coordinates
                client_rect = (
                    rect[0] + client_rect_raw[0],
                    rect[1] + client_rect_raw[1], 
                    rect[0] + client_rect_raw[2],
                    rect[1] + client_rect_raw[3]
                )
            except:
                rect = (0, 0, 0, 0)
                client_rect = (0, 0, 0, 0)
            
            # Parent and owner
            try:
                parent_hwnd = win32gui.GetParent(hwnd)
                owner_hwnd = win32gui.GetWindow(hwnd, win32con.GW_OWNER)
            except:
                parent_hwnd = 0
                owner_hwnd = 0
            
            # Window styles
            try:
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            except:
                style = 0
                ex_style = 0
            
            # Process information
            try:
                thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
                
                # Get process info
                try:
                    process = psutil.Process(process_id)
                    process_name = process.name()
                    executable_path = process.exe()
                except:
                    process_name = "Unknown"
                    executable_path = ""
                    
            except:
                thread_id = 0
                process_id = 0
                process_name = "Unknown"
                executable_path = ""
            
            return WindowInfo(
                hwnd=hwnd,
                title=title,
                class_name=class_name,
                process_id=process_id,
                process_name=process_name,
                executable_path=executable_path,
                is_visible=is_visible,
                is_enabled=is_enabled,
                is_minimized=is_minimized,
                is_maximized=is_maximized,
                rect=rect,
                client_rect=client_rect,
                parent_hwnd=parent_hwnd,
                owner_hwnd=owner_hwnd,
                style=style,
                ex_style=ex_style,
                thread_id=thread_id
            )
            
        except Exception as e:
            self.logger.debug(f"Error getting window info for {hwnd}: {e}")
            return None
    
    def _should_include_window(self, window_info: WindowInfo) -> bool:
        """Check if window should be included based on filters"""
        try:
            # Check title length
            if len(window_info.title) < self.window_filters['min_title_length']:
                return False
            
            # Check visibility
            if self.window_filters['exclude_invisible'] and not window_info.is_visible:
                return False
            
            # Check empty title
            if self.window_filters['exclude_empty_title'] and not window_info.title.strip():
                return False
            
            # Check system windows
            if self.window_filters['exclude_system']:
                if window_info.class_name in self.system_classes:
                    return False
                
                # Exclude windows with no visible area
                rect = window_info.rect
                if rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    # Window Enumeration and Search
    async def get_all_windows(self, refresh: bool = False) -> Dict[str, Any]:
        """Get all windows"""
        try:
            if refresh:
                await self._enumerate_windows()
            
            windows_list = []
            for hwnd, window_info in self.windows.items():
                windows_list.append({
                    'hwnd': hwnd,
                    'title': window_info.title,
                    'class_name': window_info.class_name,
                    'process_name': window_info.process_name,
                    'process_id': window_info.process_id,
                    'is_visible': window_info.is_visible,
                    'is_enabled': window_info.is_enabled,
                    'is_minimized': window_info.is_minimized,
                    'is_maximized': window_info.is_maximized,
                    'rect': window_info.rect,
                    'is_active': hwnd == self.active_window
                })
            
            return {
                'success': True,
                'data': {
                    'windows': {str(hwnd): {
                        'hwnd': hwnd,
                        'title': window_info.title,
                        'class_name': window_info.class_name,
                        'process_name': window_info.process_name,
                        'process_id': window_info.process_id,
                        'is_visible': window_info.is_visible,
                        'is_enabled': window_info.is_enabled,
                        'is_minimized': window_info.is_minimized,
                        'is_maximized': window_info.is_maximized,
                        'rect': window_info.rect,
                        'is_active': hwnd == self.active_window
                    } for hwnd, window_info in self.windows.items()},
                    'count': len(self.windows),
                    'active_window': self.active_window
                }
            }
            
        except Exception as e:
            self.logger.error(f"Get all windows error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def find_windows(self, title: Optional[str] = None, class_name: Optional[str] = None, 
                          process_name: Optional[str] = None, partial_match: bool = True) -> Dict[str, Any]:
        """Find windows by criteria"""
        try:
            matching_windows = []
            
            for hwnd, window_info in self.windows.items():
                match = True
                
                # Check title
                if title is not None:
                    if partial_match:
                        match = match and title.lower() in window_info.title.lower()
                    else:
                        match = match and title.lower() == window_info.title.lower()
                
                # Check class name
                if class_name is not None:
                    if partial_match:
                        match = match and class_name.lower() in window_info.class_name.lower()
                    else:
                        match = match and class_name.lower() == window_info.class_name.lower()
                
                # Check process name
                if process_name is not None:
                    if partial_match:
                        match = match and process_name.lower() in window_info.process_name.lower()
                    else:
                        match = match and process_name.lower() == window_info.process_name.lower()
                
                if match:
                    matching_windows.append({
                        'hwnd': hwnd,
                        'title': window_info.title,
                        'class_name': window_info.class_name,
                        'process_name': window_info.process_name,
                        'process_id': window_info.process_id,
                        'rect': window_info.rect,
                        'is_active': hwnd == self.active_window
                    })
            
            return {
                'success': True,
                'data': {
                    'windows': matching_windows,
                    'count': len(matching_windows)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Find windows error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_window_info(self, hwnd: int) -> Dict[str, Any]:
        """Get detailed information about a specific window"""
        try:
            if hwnd not in self.windows:
                # Try to get fresh info
                window_info = self._get_window_info(hwnd)
                if not window_info:
                    return {
                        'success': False,
                        'error': f'Window not found: {hwnd}'
                    }
            else:
                window_info = self.windows[hwnd]
            
            return {
                'success': True,
                'window_info': {
                    'hwnd': window_info.hwnd,
                    'title': window_info.title,
                    'class_name': window_info.class_name,
                    'process_id': window_info.process_id,
                    'process_name': window_info.process_name,
                    'executable_path': window_info.executable_path,
                    'is_visible': window_info.is_visible,
                    'is_enabled': window_info.is_enabled,
                    'is_minimized': window_info.is_minimized,
                    'is_maximized': window_info.is_maximized,
                    'rect': window_info.rect,
                    'client_rect': window_info.client_rect,
                    'parent_hwnd': window_info.parent_hwnd,
                    'owner_hwnd': window_info.owner_hwnd,
                    'style': window_info.style,
                    'ex_style': window_info.ex_style,
                    'thread_id': window_info.thread_id,
                    'is_active': hwnd == self.active_window
                }
            }
            
        except Exception as e:
            self.logger.error(f"Get window info error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Window Manipulation
    async def activate_window(self, hwnd: int) -> Dict[str, Any]:
        """Activate (bring to front) a window"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            # Check if window exists and is valid
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            # Check if window is visible (can't activate invisible windows)
            if not win32gui.IsWindowVisible(hwnd):
                return {
                    'success': False,
                    'error': f'Cannot activate invisible window: {hwnd}'
                }
            
            # Restore if minimized
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            # Bring to foreground
            win32gui.SetForegroundWindow(hwnd)
            
            # Update active window
            self.active_window = hwnd
            
            # Log event
            self._log_window_event('window_activated', hwnd, {'action': 'activated'})
            
            self.operation_stats['windows_manipulated'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'action': 'activated'
            }
            
        except Exception as e:
            self.logger.error(f"Activate window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def minimize_window(self, hwnd: int) -> Dict[str, Any]:
        """Minimize a window"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            
            # Log event
            self._log_window_event('window_minimized', hwnd, {'action': 'minimized'})
            
            self.operation_stats['windows_manipulated'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'action': 'minimized'
            }
            
        except Exception as e:
            self.logger.error(f"Minimize window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def maximize_window(self, hwnd: int) -> Dict[str, Any]:
        """Maximize a window"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            
            # Log event
            self._log_window_event('window_maximized', hwnd, {'action': 'maximized'})
            
            self.operation_stats['windows_manipulated'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'action': 'maximized'
            }
            
        except Exception as e:
            self.logger.error(f"Maximize window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def restore_window(self, hwnd: int) -> Dict[str, Any]:
        """Restore a window to normal size"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            # Log event
            self._log_window_event('window_restored', hwnd, {'action': 'restored'})
            
            self.operation_stats['windows_manipulated'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'action': 'restored'
            }
            
        except Exception as e:
            self.logger.error(f"Restore window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_window(self, hwnd: int) -> Dict[str, Any]:
        """Close a window"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            # Send close message
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            
            # Remove from tracking
            if hwnd in self.windows:
                del self.windows[hwnd]
            
            # Update active window
            if self.active_window == hwnd:
                self.active_window = None
            
            # Log event
            self._log_window_event('window_closed', hwnd, {'action': 'closed'})
            
            self.operation_stats['windows_manipulated'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'action': 'closed'
            }
            
        except Exception as e:
            self.logger.error(f"Close window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def resize_window(self, hwnd: int, width: int, height: int, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        """Resize and optionally move a window"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            # Get current position if not specified
            if x is None or y is None:
                rect = win32gui.GetWindowRect(hwnd)
                if x is None:
                    x = rect[0]
                if y is None:
                    y = rect[1]
            
            # Resize window
            win32gui.SetWindowPos(
                hwnd, 0, x, y, width, height,
                win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
            )
            
            # Log event
            self._log_window_event('window_resized', hwnd, {
                'action': 'resized',
                'new_size': (width, height),
                'new_position': (x, y)
            })
            
            self.operation_stats['windows_manipulated'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'action': 'resized',
                'size': (width, height),
                'position': (x, y)
            }
            
        except Exception as e:
            self.logger.error(f"Resize window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def move_window(self, hwnd: int, x: int, y: int) -> Dict[str, Any]:
        """Move a window to new position"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            # Get current size
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # Move window
            win32gui.SetWindowPos(
                hwnd, 0, x, y, width, height,
                win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE | win32con.SWP_NOSIZE
            )
            
            # Log event
            self._log_window_event('window_moved', hwnd, {
                'action': 'moved',
                'new_position': (x, y)
            })
            
            self.operation_stats['windows_manipulated'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'action': 'moved',
                'position': (x, y)
            }
            
        except Exception as e:
            self.logger.error(f"Move window error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Monitor Management
    async def get_monitors(self) -> Dict[str, Any]:
        """Get all monitor information"""
        try:
            monitors_list = []
            
            for handle, monitor_info in self.monitors.items():
                monitors_list.append({
                    'handle': handle,
                    'rect': monitor_info.rect,
                    'work_rect': monitor_info.work_rect,
                    'is_primary': monitor_info.is_primary,
                    'device_name': monitor_info.device_name,
                    'width': monitor_info.width,
                    'height': monitor_info.height,
                    'dpi': monitor_info.dpi
                })
            
            return {
                'success': True,
                'data': {
                    'monitors': monitors_list,
                    'count': len(monitors_list)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Get monitors error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_primary_monitor(self) -> Dict[str, Any]:
        """Get primary monitor information"""
        try:
            for handle, monitor_info in self.monitors.items():
                if monitor_info.is_primary:
                    return {
                        'success': True,
                        'data': {
                            'monitor': {
                                'handle': handle,
                                'rect': monitor_info.rect,
                                'work_rect': monitor_info.work_rect,
                                'device_name': monitor_info.device_name,
                                'width': monitor_info.width,
                                'height': monitor_info.height,
                                'dpi': monitor_info.dpi
                            }
                        }
                    }
            
            return {
                'success': False,
                'error': 'Primary monitor not found'
            }
            
        except Exception as e:
            self.logger.error(f"Get primary monitor error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Window Monitoring
    async def start_monitoring(self):
        """Start window monitoring"""
        try:
            if self.monitoring_active:
                return {'success': True, 'message': 'Monitoring already active'}
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Window monitoring started")
            
            return {'success': True, 'message': 'Monitoring started'}
            
        except Exception as e:
            self.logger.error(f"Start monitoring error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def stop_monitoring(self):
        """Stop window monitoring"""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            self.logger.info("Window monitoring stopped")
            
            return {'success': True, 'message': 'Monitoring stopped'}
            
        except Exception as e:
            self.logger.error(f"Stop monitoring error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            while self.monitoring_active:
                try:
                    # Check active window changes
                    current_active = win32gui.GetForegroundWindow()
                    if current_active != self.active_window:
                        old_active = self.active_window
                        self.active_window = current_active
                        
                        # Log event
                        self._log_window_event('window_focus_changed', current_active, {
                            'old_active': old_active,
                            'new_active': current_active
                        })
                        
                        # Trigger callbacks
                        self._trigger_callbacks('window_focus_changed', {
                            'old_hwnd': old_active,
                            'new_hwnd': current_active
                        })
                    
                    # Refresh window list periodically
                    if len(self.window_history) % 10 == 0:  # Every 10 events
                        # Run async method in the loop
                        loop.run_until_complete(self._enumerate_windows())
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.debug(f"Monitoring loop error: {e}")
                    time.sleep(1.0)
                    
        except Exception as e:
            self.logger.error(f"Monitoring loop fatal error: {e}")
        finally:
            # Clean up the event loop
            try:
                loop.close()
            except:
                pass
    
    def _log_window_event(self, event_type: str, hwnd: int, details: Dict[str, Any]):
        """Log window event"""
        try:
            # Get window title
            window_title = ""
            try:
                if hwnd and win32gui.IsWindow(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
            except:
                pass
            
            event = WindowEvent(
                event_type=event_type,
                hwnd=hwnd,
                window_title=window_title,
                timestamp=datetime.now(),
                details=details
            )
            
            self.window_history.append(event)
            
            # Keep history size manageable
            if len(self.window_history) > 1000:
                self.window_history = self.window_history[-500:]
            
            self.operation_stats['events_processed'] += 1
            
        except Exception as e:
            self.logger.debug(f"Log window event error: {e}")
    
    def _trigger_callbacks(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger event callbacks"""
        try:
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        callback(event_data)
                    except Exception as e:
                        self.logger.warning(f"Callback error for {event_type}: {e}")
                        
        except Exception as e:
            self.logger.debug(f"Trigger callbacks error: {e}")
    
    # Event Management
    async def register_callback(self, event_type: str, callback: Callable) -> Dict[str, Any]:
        """Register event callback"""
        try:
            if event_type not in self.event_callbacks:
                self.event_callbacks[event_type] = []
            
            self.event_callbacks[event_type].append(callback)
            
            return {
                'success': True,
                'event_type': event_type,
                'callback_count': len(self.event_callbacks[event_type])
            }
            
        except Exception as e:
            self.logger.error(f"Register callback error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_window_history(self, limit: int = 100) -> Dict[str, Any]:
        """Get window event history"""
        try:
            recent_events = self.window_history[-limit:] if limit > 0 else self.window_history
            
            events_list = []
            for event in recent_events:
                events_list.append({
                    'event_type': event.event_type,
                    'hwnd': event.hwnd,
                    'window_title': event.window_title,
                    'timestamp': event.timestamp.isoformat(),
                    'details': event.details
                })
            
            return {
                'success': True,
                'events': events_list,
                'count': len(events_list),
                'total_events': len(self.window_history)
            }
            
        except Exception as e:
            self.logger.error(f"Get window history error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Screenshot and Capture
    async def capture_window(self, hwnd: int, filename: Optional[str] = None) -> Dict[str, Any]:
        """Capture screenshot of specific window"""
        try:
            if not WIN32_AVAILABLE:
                return {'success': False, 'error': 'Windows API not available'}
            
            if not win32gui.IsWindow(hwnd):
                return {
                    'success': False,
                    'error': f'Invalid window handle: {hwnd}'
                }
            
            # Get window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            if width <= 0 or height <= 0:
                return {
                    'success': False,
                    'error': 'Window has no visible area'
                }
            
            # Create device contexts
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Copy window content to bitmap
            result = save_dc.BitBlt(
                (0, 0), (width, height), 
                mfc_dc, (0, 0), 
                win32con.SRCCOPY
            )
            
            if not result:
                # Cleanup on failure
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)
                return {
                    'success': False,
                    'error': 'Failed to capture window content'
                }
            
            # Save to file
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                window_title = ""
                try:
                    window_title = win32gui.GetWindowText(hwnd)
                    # Clean filename from window title
                    window_title = "".join(c for c in window_title if c.isalnum() or c in (' ', '-', '_')).strip()
                    if window_title:
                        filename = f'window_{hwnd}_{window_title[:30]}_{timestamp}.bmp'
                    else:
                        filename = f'window_{hwnd}_{timestamp}.bmp'
                except:
                    filename = f'window_{hwnd}_{timestamp}.bmp'
            
            # Ensure screenshots directory exists
            screenshots_dir = Path.cwd() / 'screenshots'
            screenshots_dir.mkdir(exist_ok=True)
            
            screenshot_path = screenshots_dir / filename
            save_bitmap.SaveBitmapFile(save_dc, str(screenshot_path))
            
            # Cleanup
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            
            self.operation_stats['screenshots_taken'] += 1
            
            return {
                'success': True,
                'filename': filename,
                'path': str(screenshot_path),
                'size': (width, height),
                'hwnd': hwnd
            }
            
        except Exception as e:
            self.logger.error(f"Capture window error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Performance and Statistics
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get window manager performance statistics"""
        return {
            'success': True,
            'data': {
                'stats': {
                    'total_operations': self.operation_stats.get('total_operations', 0),
                    'total_errors': self.operation_stats.get('total_errors', 0),
                    'windows_enumerated': self.operation_stats.get('windows_enumerated', 0),
                    'windows_manipulated': self.operation_stats.get('windows_manipulated', 0)
                },
                'operation_stats': self.operation_stats.copy(),
                'window_stats': {
                    'total_windows': len(self.windows),
                    'active_window': self.active_window,
                    'visible_windows': sum(1 for w in self.windows.values() if w.is_visible),
                    'enabled_windows': sum(1 for w in self.windows.values() if w.is_enabled)
                },
                'monitor_stats': {
                    'total_monitors': len(self.monitors),
                    'primary_monitor': next((h for h, m in self.monitors.items() if m.is_primary), None)
                },
                'monitoring_stats': {
                    'is_active': self.monitoring_active,
                    'events_logged': len(self.window_history),
                    'callbacks_registered': sum(len(callbacks) for callbacks in self.event_callbacks.values())
                }
            }
        }
    
    async def cleanup(self):
        """Cleanup window manager"""
        try:
            self.logger.info("Cleaning up Window Manager...")
            
            # Stop monitoring
            await self.stop_monitoring()
            
            # Clear data
            self.windows.clear()
            self.monitors.clear()
            self.window_history.clear()
            self.event_callbacks.clear()
            
            self.logger.info("Window Manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Window Manager cleanup error: {e}")
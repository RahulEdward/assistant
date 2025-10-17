"""
Application Controller
Manages Windows applications with precise control and automation capabilities.
Handles application launching, window management, and interaction.
"""

import asyncio
import logging
import os
import sys
import subprocess
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Windows API imports
import win32api
import win32con
import win32gui
import win32process
import win32clipboard
import win32event
import win32file
import win32security
import psutil
import ctypes
from ctypes import wintypes


@dataclass
class WindowInfo:
    """Window information structure"""
    hwnd: int
    title: str
    class_name: str
    pid: int
    process_name: str
    rect: Tuple[int, int, int, int]
    is_visible: bool
    is_minimized: bool
    is_maximized: bool
    parent_hwnd: int


@dataclass
class ApplicationInfo:
    """Application information structure"""
    name: str
    path: str
    pid: int
    windows: List[WindowInfo]
    status: str
    cpu_usage: float
    memory_usage: float
    start_time: datetime


class ApplicationController:
    """Advanced Windows application controller"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Application registry
        self.applications = {}  # pid -> ApplicationInfo
        self.window_registry = {}  # hwnd -> WindowInfo
        
        # Application paths cache
        self.app_paths = {}
        
        # Performance monitoring
        self.operation_stats = {
            'applications_launched': 0,
            'windows_managed': 0,
            'automation_actions': 0,
            'clipboard_operations': 0
        }
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Window enumeration lock
        self.enum_lock = threading.Lock()
        
        # Common application paths
        self.common_apps = {
            'notepad': 'notepad.exe',
            'calculator': 'calc.exe',
            'paint': 'mspaint.exe',
            'wordpad': 'write.exe',
            'explorer': 'explorer.exe',
            'cmd': 'cmd.exe',
            'powershell': 'powershell.exe',
            'taskmgr': 'taskmgr.exe',
            'regedit': 'regedit.exe',
            'msconfig': 'msconfig.exe',
            'control': 'control.exe'
        }
        
        # Initialize Windows API functions
        self._init_win32_functions()
    
    def _init_win32_functions(self):
        """Initialize Windows API function prototypes"""
        try:
            # Set up function prototypes for better performance
            self.user32 = ctypes.windll.user32
            self.kernel32 = ctypes.windll.kernel32
            
            # Window enumeration callback type
            self.enum_windows_proc = ctypes.WINFUNCTYPE(
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_void_p
            )
            
        except Exception as e:
            self.logger.error(f"Win32 function initialization error: {e}")
    
    async def initialize(self):
        """Initialize application controller"""
        try:
            self.logger.info("Initializing Application Controller...")
            
            # Discover installed applications
            await self._discover_applications()
            
            # Initialize window registry
            await self._refresh_window_registry()
            
            # Start monitoring thread
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_applications, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("Application Controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Application Controller initialization error: {e}")
            return False
    
    async def _discover_applications(self):
        """Discover installed applications"""
        try:
            # Common Windows application directories
            search_paths = [
                Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')),
                Path(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')),
                Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'System32',
                Path(os.environ.get('USERPROFILE', '')) / 'AppData' / 'Local' / 'Programs'
            ]
            
            # Search for executable files
            for search_path in search_paths:
                if search_path.exists():
                    for exe_file in search_path.rglob('*.exe'):
                        try:
                            app_name = exe_file.stem.lower()
                            if app_name not in self.app_paths:
                                self.app_paths[app_name] = str(exe_file)
                        except Exception:
                            continue
            
            # Add common applications
            for app_name, exe_name in self.common_apps.items():
                if app_name not in self.app_paths:
                    # Try to find in PATH
                    try:
                        result = subprocess.run(
                            ['where', exe_name],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            self.app_paths[app_name] = result.stdout.strip().split('\n')[0]
                    except:
                        continue
            
            self.logger.info(f"Discovered {len(self.app_paths)} applications")
            
        except Exception as e:
            self.logger.error(f"Application discovery error: {e}")
    
    async def _refresh_window_registry(self):
        """Refresh window registry"""
        try:
            with self.enum_lock:
                self.window_registry.clear()
                
                # Enumerate all windows
                def enum_callback(hwnd, lparam):
                    try:
                        if win32gui.IsWindow(hwnd):
                            window_info = self._get_window_info(hwnd)
                            if window_info:
                                self.window_registry[hwnd] = window_info
                    except Exception:
                        pass
                    return True
                
                # Use callback to enumerate windows
                callback = self.enum_windows_proc(enum_callback)
                win32gui.EnumWindows(callback, 0)
            
        except Exception as e:
            self.logger.error(f"Window registry refresh error: {e}")
    
    def _get_window_info(self, hwnd: int) -> Optional[WindowInfo]:
        """Get detailed window information"""
        try:
            # Get window title
            title = win32gui.GetWindowText(hwnd)
            
            # Get window class name
            class_name = win32gui.GetClassName(hwnd)
            
            # Get process ID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            # Get process name
            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except:
                process_name = "Unknown"
            
            # Get window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            
            # Get window state
            is_visible = win32gui.IsWindowVisible(hwnd)
            is_minimized = win32gui.IsIconic(hwnd)
            is_maximized = win32gui.IsZoomed(hwnd)
            
            # Get parent window
            parent_hwnd = win32gui.GetParent(hwnd)
            
            return WindowInfo(
                hwnd=hwnd,
                title=title,
                class_name=class_name,
                pid=pid,
                process_name=process_name,
                rect=rect,
                is_visible=is_visible,
                is_minimized=is_minimized,
                is_maximized=is_maximized,
                parent_hwnd=parent_hwnd
            )
            
        except Exception as e:
            self.logger.debug(f"Window info error for hwnd {hwnd}: {e}")
            return None
    
    def _monitor_applications(self):
        """Monitor running applications"""
        while self.monitoring_active:
            try:
                # Update application registry
                current_pids = set()
                
                for proc in psutil.process_iter(['pid', 'name', 'exe', 'create_time', 'cpu_percent', 'memory_percent']):
                    try:
                        pid = proc.info['pid']
                        current_pids.add(pid)
                        
                        if pid not in self.applications:
                            # New application
                            app_info = ApplicationInfo(
                                name=proc.info['name'],
                                path=proc.info['exe'] or '',
                                pid=pid,
                                windows=[],
                                status='running',
                                cpu_usage=proc.info['cpu_percent'] or 0.0,
                                memory_usage=proc.info['memory_percent'] or 0.0,
                                start_time=datetime.fromtimestamp(proc.info['create_time'])
                            )
                            self.applications[pid] = app_info
                        else:
                            # Update existing application
                            app_info = self.applications[pid]
                            app_info.cpu_usage = proc.info['cpu_percent'] or 0.0
                            app_info.memory_usage = proc.info['memory_percent'] or 0.0
                            app_info.status = 'running'
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Remove terminated applications
                terminated_pids = set(self.applications.keys()) - current_pids
                for pid in terminated_pids:
                    del self.applications[pid]
                
                # Update window registry periodically
                if int(time.time()) % 10 == 0:  # Every 10 seconds
                    asyncio.create_task(self._refresh_window_registry())
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Application monitoring error: {e}")
                time.sleep(5)
    
    # Application Management
    async def launch_application(self, app_name: str, arguments: str = '', working_directory: str = '') -> Dict[str, Any]:
        """Launch application"""
        try:
            # Resolve application path
            app_path = await self._resolve_app_path(app_name)
            if not app_path:
                return {
                    'success': False,
                    'error': f'Application not found: {app_name}',
                    'suggestions': self._get_app_suggestions(app_name)
                }
            
            # Prepare command
            if arguments:
                command = f'"{app_path}" {arguments}'
            else:
                command = f'"{app_path}"'
            
            # Set working directory
            if not working_directory:
                working_directory = str(Path(app_path).parent)
            
            # Launch application
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for the process to start
            await asyncio.sleep(0.5)
            
            # Get process information
            try:
                proc_info = psutil.Process(process.pid)
                app_info = ApplicationInfo(
                    name=proc_info.name(),
                    path=app_path,
                    pid=process.pid,
                    windows=[],
                    status='starting',
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    start_time=datetime.now()
                )
                self.applications[process.pid] = app_info
            except:
                pass
            
            self.operation_stats['applications_launched'] += 1
            
            return {
                'success': True,
                'pid': process.pid,
                'app_name': app_name,
                'app_path': app_path,
                'command': command
            }
            
        except Exception as e:
            self.logger.error(f"Application launch error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _resolve_app_path(self, app_name: str) -> Optional[str]:
        """Resolve application path"""
        try:
            # Check if it's already a full path
            if Path(app_name).exists():
                return app_name
            
            # Check common applications
            app_lower = app_name.lower()
            if app_lower in self.app_paths:
                return self.app_paths[app_lower]
            
            # Check if it's a known executable
            if app_lower in self.common_apps:
                exe_name = self.common_apps[app_lower]
                try:
                    result = subprocess.run(
                        ['where', exe_name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        return result.stdout.strip().split('\n')[0]
                except:
                    pass
            
            # Search in discovered applications
            for name, path in self.app_paths.items():
                if app_lower in name or name in app_lower:
                    return path
            
            # Try to find in PATH
            try:
                result = subprocess.run(
                    ['where', app_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            except:
                pass
            
            return None
            
        except Exception as e:
            self.logger.error(f"App path resolution error: {e}")
            return None
    
    def _get_app_suggestions(self, app_name: str) -> List[str]:
        """Get application name suggestions"""
        try:
            suggestions = []
            app_lower = app_name.lower()
            
            # Find similar names
            for name in self.app_paths.keys():
                if app_lower in name or name in app_lower:
                    suggestions.append(name)
            
            # Add common applications
            for name in self.common_apps.keys():
                if app_lower in name or name in app_lower:
                    suggestions.append(name)
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"App suggestions error: {e}")
            return []
    
    async def terminate_application(self, identifier: str, force: bool = False) -> Dict[str, Any]:
        """Terminate application by name or PID"""
        try:
            # Find application
            target_pids = []
            
            if identifier.isdigit():
                # PID provided
                pid = int(identifier)
                if pid in self.applications:
                    target_pids.append(pid)
            else:
                # Application name provided
                for pid, app_info in self.applications.items():
                    if (identifier.lower() in app_info.name.lower() or 
                        identifier.lower() in Path(app_info.path).stem.lower()):
                        target_pids.append(pid)
            
            if not target_pids:
                return {
                    'success': False,
                    'error': f'Application not found: {identifier}'
                }
            
            terminated = []
            errors = []
            
            for pid in target_pids:
                try:
                    process = psutil.Process(pid)
                    
                    if force:
                        process.kill()
                    else:
                        process.terminate()
                    
                    # Wait for termination
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        if not force:
                            process.kill()
                            process.wait(timeout=2)
                    
                    terminated.append({
                        'pid': pid,
                        'name': self.applications[pid].name
                    })
                    
                    # Remove from registry
                    if pid in self.applications:
                        del self.applications[pid]
                
                except Exception as e:
                    errors.append({
                        'pid': pid,
                        'error': str(e)
                    })
            
            return {
                'success': len(terminated) > 0,
                'terminated': terminated,
                'errors': errors
            }
            
        except Exception as e:
            self.logger.error(f"Application termination error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Window Management
    async def get_windows(self, filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get windows matching criteria"""
        try:
            await self._refresh_window_registry()
            
            windows = list(self.window_registry.values())
            
            # Apply filters
            if filter_criteria:
                filtered_windows = []
                
                for window in windows:
                    match = True
                    
                    if 'title_contains' in filter_criteria:
                        if filter_criteria['title_contains'].lower() not in window.title.lower():
                            match = False
                    
                    if 'class_name' in filter_criteria:
                        if filter_criteria['class_name'] != window.class_name:
                            match = False
                    
                    if 'process_name' in filter_criteria:
                        if filter_criteria['process_name'].lower() not in window.process_name.lower():
                            match = False
                    
                    if 'is_visible' in filter_criteria:
                        if filter_criteria['is_visible'] != window.is_visible:
                            match = False
                    
                    if match:
                        filtered_windows.append(window)
                
                windows = filtered_windows
            
            # Convert to serializable format
            window_data = []
            for window in windows:
                window_data.append({
                    'hwnd': window.hwnd,
                    'title': window.title,
                    'class_name': window.class_name,
                    'pid': window.pid,
                    'process_name': window.process_name,
                    'rect': window.rect,
                    'is_visible': window.is_visible,
                    'is_minimized': window.is_minimized,
                    'is_maximized': window.is_maximized,
                    'parent_hwnd': window.parent_hwnd
                })
            
            return {
                'success': True,
                'windows': window_data,
                'count': len(window_data)
            }
            
        except Exception as e:
            self.logger.error(f"Get windows error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def focus_window(self, identifier: str) -> Dict[str, Any]:
        """Focus window by title, class name, or hwnd"""
        try:
            # Find window
            target_window = None
            
            if identifier.isdigit():
                # HWND provided
                hwnd = int(identifier)
                if hwnd in self.window_registry:
                    target_window = self.window_registry[hwnd]
            else:
                # Search by title or class name
                for window in self.window_registry.values():
                    if (identifier.lower() in window.title.lower() or 
                        identifier.lower() in window.class_name.lower()):
                        target_window = window
                        break
            
            if not target_window:
                return {
                    'success': False,
                    'error': f'Window not found: {identifier}'
                }
            
            # Focus window
            hwnd = target_window.hwnd
            
            # Restore if minimized
            if target_window.is_minimized:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            # Bring to foreground
            win32gui.SetForegroundWindow(hwnd)
            win32gui.BringWindowToTop(hwnd)
            
            self.operation_stats['windows_managed'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'title': target_window.title,
                'action': 'focused'
            }
            
        except Exception as e:
            self.logger.error(f"Focus window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def resize_window(self, identifier: str, width: int, height: int, x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
        """Resize and optionally move window"""
        try:
            # Find window
            target_window = await self._find_window(identifier)
            if not target_window:
                return {
                    'success': False,
                    'error': f'Window not found: {identifier}'
                }
            
            hwnd = target_window.hwnd
            
            # Get current position if not specified
            if x is None or y is None:
                current_rect = win32gui.GetWindowRect(hwnd)
                if x is None:
                    x = current_rect[0]
                if y is None:
                    y = current_rect[1]
            
            # Resize and move window
            win32gui.SetWindowPos(
                hwnd,
                0,  # hwndInsertAfter
                x, y, width, height,
                win32con.SWP_NOZORDER
            )
            
            self.operation_stats['windows_managed'] += 1
            
            return {
                'success': True,
                'hwnd': hwnd,
                'title': target_window.title,
                'new_rect': (x, y, x + width, y + height)
            }
            
        except Exception as e:
            self.logger.error(f"Resize window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def minimize_window(self, identifier: str) -> Dict[str, Any]:
        """Minimize window"""
        try:
            target_window = await self._find_window(identifier)
            if not target_window:
                return {
                    'success': False,
                    'error': f'Window not found: {identifier}'
                }
            
            win32gui.ShowWindow(target_window.hwnd, win32con.SW_MINIMIZE)
            
            self.operation_stats['windows_managed'] += 1
            
            return {
                'success': True,
                'hwnd': target_window.hwnd,
                'title': target_window.title,
                'action': 'minimized'
            }
            
        except Exception as e:
            self.logger.error(f"Minimize window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def maximize_window(self, identifier: str) -> Dict[str, Any]:
        """Maximize window"""
        try:
            target_window = await self._find_window(identifier)
            if not target_window:
                return {
                    'success': False,
                    'error': f'Window not found: {identifier}'
                }
            
            win32gui.ShowWindow(target_window.hwnd, win32con.SW_MAXIMIZE)
            
            self.operation_stats['windows_managed'] += 1
            
            return {
                'success': True,
                'hwnd': target_window.hwnd,
                'title': target_window.title,
                'action': 'maximized'
            }
            
        except Exception as e:
            self.logger.error(f"Maximize window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_window(self, identifier: str) -> Dict[str, Any]:
        """Close window"""
        try:
            target_window = await self._find_window(identifier)
            if not target_window:
                return {
                    'success': False,
                    'error': f'Window not found: {identifier}'
                }
            
            # Send close message
            win32gui.PostMessage(target_window.hwnd, win32con.WM_CLOSE, 0, 0)
            
            self.operation_stats['windows_managed'] += 1
            
            return {
                'success': True,
                'hwnd': target_window.hwnd,
                'title': target_window.title,
                'action': 'closed'
            }
            
        except Exception as e:
            self.logger.error(f"Close window error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _find_window(self, identifier: str) -> Optional[WindowInfo]:
        """Find window by identifier"""
        try:
            await self._refresh_window_registry()
            
            if identifier.isdigit():
                # HWND provided
                hwnd = int(identifier)
                return self.window_registry.get(hwnd)
            else:
                # Search by title or class name
                for window in self.window_registry.values():
                    if (identifier.lower() in window.title.lower() or 
                        identifier.lower() in window.class_name.lower()):
                        return window
            
            return None
            
        except Exception as e:
            self.logger.error(f"Find window error: {e}")
            return None
    
    # Clipboard Operations
    async def get_clipboard_content(self) -> Dict[str, Any]:
        """Get clipboard content"""
        try:
            win32clipboard.OpenClipboard()
            
            try:
                # Try to get text
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
                    data = win32clipboard.GetClipboardData(win32con.CF_TEXT)
                    content_type = 'text'
                elif win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                    content_type = 'unicode_text'
                else:
                    data = None
                    content_type = 'unknown'
                
                self.operation_stats['clipboard_operations'] += 1
                
                return {
                    'success': True,
                    'content': data,
                    'type': content_type,
                    'has_content': data is not None
                }
                
            finally:
                win32clipboard.CloseClipboard()
                
        except Exception as e:
            self.logger.error(f"Get clipboard error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def set_clipboard_content(self, content: str) -> Dict[str, Any]:
        """Set clipboard content"""
        try:
            win32clipboard.OpenClipboard()
            
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(content)
                
                self.operation_stats['clipboard_operations'] += 1
                
                return {
                    'success': True,
                    'content_length': len(content)
                }
                
            finally:
                win32clipboard.CloseClipboard()
                
        except Exception as e:
            self.logger.error(f"Set clipboard error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Application Information
    async def get_running_applications(self) -> Dict[str, Any]:
        """Get list of running applications"""
        try:
            apps = []
            
            for pid, app_info in self.applications.items():
                # Get windows for this application
                app_windows = []
                for window in self.window_registry.values():
                    if window.pid == pid and window.is_visible:
                        app_windows.append({
                            'hwnd': window.hwnd,
                            'title': window.title,
                            'class_name': window.class_name
                        })
                
                apps.append({
                    'pid': pid,
                    'name': app_info.name,
                    'path': app_info.path,
                    'status': app_info.status,
                    'cpu_usage': app_info.cpu_usage,
                    'memory_usage': app_info.memory_usage,
                    'start_time': app_info.start_time.isoformat(),
                    'window_count': len(app_windows),
                    'windows': app_windows
                })
            
            return {
                'success': True,
                'applications': apps,
                'count': len(apps)
            }
            
        except Exception as e:
            self.logger.error(f"Get running applications error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_application_info(self, identifier: str) -> Dict[str, Any]:
        """Get detailed application information"""
        try:
            # Find application
            target_app = None
            
            if identifier.isdigit():
                # PID provided
                pid = int(identifier)
                target_app = self.applications.get(pid)
            else:
                # Application name provided
                for app_info in self.applications.values():
                    if (identifier.lower() in app_info.name.lower() or 
                        identifier.lower() in Path(app_info.path).stem.lower()):
                        target_app = app_info
                        break
            
            if not target_app:
                return {
                    'success': False,
                    'error': f'Application not found: {identifier}'
                }
            
            # Get detailed process information
            try:
                process = psutil.Process(target_app.pid)
                
                # Get memory info
                memory_info = process.memory_info()
                
                # Get CPU times
                cpu_times = process.cpu_times()
                
                # Get open files
                try:
                    open_files = [f.path for f in process.open_files()]
                except:
                    open_files = []
                
                # Get network connections
                try:
                    connections = [
                        {
                            'local_address': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                            'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                            'status': conn.status
                        }
                        for conn in process.connections()
                    ]
                except:
                    connections = []
                
                detailed_info = {
                    'basic_info': {
                        'pid': target_app.pid,
                        'name': target_app.name,
                        'path': target_app.path,
                        'status': target_app.status,
                        'start_time': target_app.start_time.isoformat()
                    },
                    'performance': {
                        'cpu_usage': target_app.cpu_usage,
                        'memory_usage': target_app.memory_usage,
                        'memory_info': {
                            'rss': memory_info.rss,
                            'vms': memory_info.vms,
                            'peak_wset': memory_info.peak_wset,
                            'wset': memory_info.wset
                        },
                        'cpu_times': {
                            'user': cpu_times.user,
                            'system': cpu_times.system
                        }
                    },
                    'resources': {
                        'open_files': open_files,
                        'network_connections': connections
                    },
                    'windows': []
                }
                
                # Get windows for this application
                for window in self.window_registry.values():
                    if window.pid == target_app.pid:
                        detailed_info['windows'].append({
                            'hwnd': window.hwnd,
                            'title': window.title,
                            'class_name': window.class_name,
                            'rect': window.rect,
                            'is_visible': window.is_visible,
                            'is_minimized': window.is_minimized,
                            'is_maximized': window.is_maximized
                        })
                
                return {
                    'success': True,
                    'application': detailed_info
                }
                
            except psutil.NoSuchProcess:
                return {
                    'success': False,
                    'error': 'Process no longer exists'
                }
            
        except Exception as e:
            self.logger.error(f"Get application info error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get application controller performance statistics"""
        return {
            'operation_stats': self.operation_stats.copy(),
            'registry_stats': {
                'applications_tracked': len(self.applications),
                'windows_tracked': len(self.window_registry),
                'discovered_apps': len(self.app_paths)
            },
            'monitoring_status': {
                'monitoring_active': self.monitoring_active,
                'monitor_thread_alive': self.monitor_thread.is_alive() if hasattr(self, 'monitor_thread') else False
            }
        }
    
    async def cleanup(self):
        """Cleanup application controller"""
        try:
            self.logger.info("Cleaning up Application Controller...")
            
            # Stop monitoring
            self.monitoring_active = False
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=5)
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Clear registries
            self.applications.clear()
            self.window_registry.clear()
            self.app_paths.clear()
            
            self.logger.info("Application Controller cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Application Controller cleanup error: {e}")
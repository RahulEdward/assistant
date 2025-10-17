"""
System Controller
Handles Windows system-level operations with high precision and performance.
Provides file management, system information, and low-level system control.
"""

import asyncio
import logging
import os
import sys
import shutil
import subprocess
import psutil
import winreg
import ctypes
from ctypes import wintypes
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import glob
import threading
from concurrent.futures import ThreadPoolExecutor

# Windows API imports
from ctypes import windll
import win32api
import win32con
import win32gui
import win32process
import win32security
import win32service
import win32serviceutil


class SystemController:
    """Advanced Windows system controller"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # System information cache
        self.system_info_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
        # Performance monitoring
        self.operation_stats = {
            'file_operations': 0,
            'system_queries': 0,
            'registry_operations': 0,
            'service_operations': 0
        }
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Windows API handles
        self.kernel32 = ctypes.windll.kernel32
        self.user32 = ctypes.windll.user32
        self.advapi32 = ctypes.windll.advapi32
        
        # System paths
        self.system_paths = {
            'windows': os.environ.get('WINDIR', 'C:\\Windows'),
            'system32': os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'System32'),
            'program_files': os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
            'program_files_x86': os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'),
            'user_profile': os.environ.get('USERPROFILE', ''),
            'temp': os.environ.get('TEMP', ''),
            'desktop': os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop'),
            'documents': os.path.join(os.environ.get('USERPROFILE', ''), 'Documents'),
            'downloads': os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads')
        }
    
    async def initialize(self):
        """Initialize system controller"""
        try:
            self.logger.info("Initializing System Controller...")
            
            # Check Windows version and capabilities
            await self._check_system_capabilities()
            
            # Initialize system information cache
            await self._initialize_system_cache()
            
            # Verify permissions
            await self._check_permissions()
            
            self.logger.info("System Controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System Controller initialization error: {e}")
            return False
    
    async def _check_system_capabilities(self):
        """Check Windows system capabilities"""
        try:
            # Get Windows version
            version_info = sys.getwindowsversion()
            self.windows_version = {
                'major': version_info.major,
                'minor': version_info.minor,
                'build': version_info.build,
                'platform': version_info.platform,
                'service_pack': version_info.service_pack
            }
            
            # Check if running as administrator
            self.is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            
            # Check available APIs
            self.api_available = {
                'win32api': True,
                'psutil': True,
                'winreg': True
            }
            
            self.logger.info(f"Windows {version_info.major}.{version_info.minor} Build {version_info.build}")
            self.logger.info(f"Administrator privileges: {self.is_admin}")
            
        except Exception as e:
            self.logger.error(f"System capability check error: {e}")
    
    async def _initialize_system_cache(self):
        """Initialize system information cache"""
        try:
            # Cache basic system info
            await self._cache_system_info('basic')
            
            self.logger.info("System cache initialized")
            
        except Exception as e:
            self.logger.error(f"System cache initialization error: {e}")
    
    async def _check_permissions(self):
        """Check required permissions"""
        try:
            # Test file system access
            test_path = Path(self.system_paths['temp']) / 'assistant_test.tmp'
            try:
                test_path.write_text('test')
                test_path.unlink()
                self.file_access = True
            except:
                self.file_access = False
            
            # Test registry access
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software")
                winreg.CloseKey(key)
                self.registry_access = True
            except:
                self.registry_access = False
            
            self.logger.info(f"File access: {self.file_access}, Registry access: {self.registry_access}")
            
        except Exception as e:
            self.logger.error(f"Permission check error: {e}")
    
    # File Management Operations
    async def create_file(self, file_path: str, content: str = '', overwrite: bool = False) -> Dict[str, Any]:
        """Create file with content"""
        try:
            path = Path(file_path)
            
            # Check if file exists
            if path.exists() and not overwrite:
                return {
                    'success': False,
                    'error': 'File already exists',
                    'path': str(path)
                }
            
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            path.write_text(content, encoding='utf-8')
            
            self.operation_stats['file_operations'] += 1
            
            return {
                'success': True,
                'path': str(path),
                'size': path.stat().st_size,
                'created': datetime.fromtimestamp(path.stat().st_ctime).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"File creation error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def delete_file(self, file_path: str, permanent: bool = False) -> Dict[str, Any]:
        """Delete file (to recycle bin or permanently)"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {
                    'success': False,
                    'error': 'File not found',
                    'path': str(path)
                }
            
            if permanent:
                # Permanent deletion
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            else:
                # Move to recycle bin using Windows API
                await self._move_to_recycle_bin(str(path))
            
            self.operation_stats['file_operations'] += 1
            
            return {
                'success': True,
                'path': str(path),
                'permanent': permanent
            }
            
        except Exception as e:
            self.logger.error(f"File deletion error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _move_to_recycle_bin(self, file_path: str):
        """Move file to recycle bin using Windows API"""
        try:
            import win32file
            import win32con
            
            # Use SHFileOperation to move to recycle bin
            result = win32file.SHFileOperation((
                0,  # hwnd
                win32file.FO_DELETE,  # operation
                file_path + '\0',  # from (null-terminated)
                None,  # to
                win32file.FOF_ALLOWUNDO | win32file.FOF_NOCONFIRMATION,  # flags
                None,  # progress title
                None   # progress text
            ))
            
            if result[0] != 0:
                raise Exception(f"SHFileOperation failed with code {result[0]}")
                
        except ImportError:
            # Fallback: permanent deletion if win32file not available
            Path(file_path).unlink()
        except Exception as e:
            self.logger.error(f"Recycle bin operation error: {e}")
            raise
    
    async def copy_file(self, source_path: str, destination_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """Copy file or directory"""
        try:
            source = Path(source_path)
            destination = Path(destination_path)
            
            if not source.exists():
                return {
                    'success': False,
                    'error': 'Source not found',
                    'source': str(source)
                }
            
            if destination.exists() and not overwrite:
                return {
                    'success': False,
                    'error': 'Destination already exists',
                    'destination': str(destination)
                }
            
            # Create parent directories
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            if source.is_file():
                shutil.copy2(source, destination)
            elif source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            
            self.operation_stats['file_operations'] += 1
            
            return {
                'success': True,
                'source': str(source),
                'destination': str(destination),
                'size': destination.stat().st_size if destination.is_file() else self._get_directory_size(destination)
            }
            
        except Exception as e:
            self.logger.error(f"File copy error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def move_file(self, source_path: str, destination_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """Move file or directory"""
        try:
            source = Path(source_path)
            destination = Path(destination_path)
            
            if not source.exists():
                return {
                    'success': False,
                    'error': 'Source not found',
                    'source': str(source)
                }
            
            if destination.exists() and not overwrite:
                return {
                    'success': False,
                    'error': 'Destination already exists',
                    'destination': str(destination)
                }
            
            # Create parent directories
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file/directory
            shutil.move(source, destination)
            
            self.operation_stats['file_operations'] += 1
            
            return {
                'success': True,
                'source': str(source),
                'destination': str(destination)
            }
            
        except Exception as e:
            self.logger.error(f"File move error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def rename_file(self, old_path: str, new_path: str) -> Dict[str, Any]:
        """Rename file or directory"""
        try:
            old = Path(old_path)
            new = Path(new_path)
            
            if not old.exists():
                return {
                    'success': False,
                    'error': 'File not found',
                    'path': str(old)
                }
            
            if new.exists():
                return {
                    'success': False,
                    'error': 'Target name already exists',
                    'path': str(new)
                }
            
            old.rename(new)
            
            self.operation_stats['file_operations'] += 1
            
            return {
                'success': True,
                'old_path': str(old),
                'new_path': str(new)
            }
            
        except Exception as e:
            self.logger.error(f"File rename error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def search_files(self, search_path: str, pattern: str, file_type: Optional[str] = None, recursive: bool = True) -> Dict[str, Any]:
        """Search for files matching pattern"""
        try:
            search_dir = Path(search_path)
            
            if not search_dir.exists():
                return {
                    'success': False,
                    'error': 'Search path not found',
                    'path': str(search_dir)
                }
            
            results = []
            
            # Build search pattern
            if file_type:
                search_pattern = f"*.{file_type}"
            else:
                search_pattern = "*"
            
            # Search files
            if recursive:
                glob_pattern = str(search_dir / "**" / search_pattern)
                files = glob.glob(glob_pattern, recursive=True)
            else:
                glob_pattern = str(search_dir / search_pattern)
                files = glob.glob(glob_pattern)
            
            # Filter by pattern
            for file_path in files:
                path = Path(file_path)
                if pattern.lower() in path.name.lower():
                    try:
                        stat = path.stat()
                        results.append({
                            'path': str(path),
                            'name': path.name,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'is_directory': path.is_dir()
                        })
                    except:
                        continue
            
            self.operation_stats['file_operations'] += 1
            
            return {
                'success': True,
                'results': results,
                'count': len(results),
                'search_path': str(search_dir),
                'pattern': pattern
            }
            
        except Exception as e:
            self.logger.error(f"File search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory"""
        try:
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except:
            return 0
    
    # System Information Operations
    async def get_system_info(self, info_type: str = 'general') -> Dict[str, Any]:
        """Get system information"""
        try:
            # Check cache first
            cache_key = f"system_info_{info_type}"
            if self._is_cache_valid(cache_key):
                return self.system_info_cache[cache_key]
            
            info = {}
            
            if info_type in ['general', 'all']:
                info.update(await self._get_general_system_info())
            
            if info_type in ['hardware', 'all']:
                info.update(await self._get_hardware_info())
            
            if info_type in ['network', 'all']:
                info.update(await self._get_network_info())
            
            if info_type in ['processes', 'all']:
                info.update(await self._get_process_info())
            
            if info_type in ['services', 'all']:
                info.update(await self._get_service_info())
            
            # Cache result
            self._cache_result(cache_key, info)
            self.operation_stats['system_queries'] += 1
            
            return {'success': True, 'info': info}
            
        except Exception as e:
            self.logger.error(f"System info error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _get_general_system_info(self) -> Dict[str, Any]:
        """Get general system information"""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            
            return {
                'system': {
                    'platform': sys.platform,
                    'windows_version': self.windows_version,
                    'computer_name': os.environ.get('COMPUTERNAME', 'Unknown'),
                    'username': os.environ.get('USERNAME', 'Unknown'),
                    'boot_time': boot_time.isoformat(),
                    'uptime': str(datetime.now() - boot_time)
                },
                'cpu': {
                    'count': psutil.cpu_count(),
                    'usage': psutil.cpu_percent(interval=1),
                    'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'used': psutil.virtual_memory().used,
                    'percentage': psutil.virtual_memory().percent
                },
                'disk': [
                    {
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': psutil.disk_usage(partition.mountpoint).total,
                        'used': psutil.disk_usage(partition.mountpoint).used,
                        'free': psutil.disk_usage(partition.mountpoint).free,
                        'percentage': (psutil.disk_usage(partition.mountpoint).used / 
                                     psutil.disk_usage(partition.mountpoint).total * 100)
                    }
                    for partition in psutil.disk_partitions()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"General system info error: {e}")
            return {}
    
    async def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        try:
            return {
                'hardware': {
                    'cpu_info': await self._get_cpu_info(),
                    'memory_info': await self._get_memory_info(),
                    'gpu_info': await self._get_gpu_info(),
                    'motherboard_info': await self._get_motherboard_info()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Hardware info error: {e}")
            return {}
    
    async def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information using WMI"""
        try:
            import wmi
            c = wmi.WMI()
            
            cpu_info = []
            for cpu in c.Win32_Processor():
                cpu_info.append({
                    'name': cpu.Name,
                    'manufacturer': cpu.Manufacturer,
                    'cores': cpu.NumberOfCores,
                    'threads': cpu.NumberOfLogicalProcessors,
                    'max_speed': cpu.MaxClockSpeed,
                    'current_speed': cpu.CurrentClockSpeed,
                    'architecture': cpu.Architecture
                })
            
            return {'processors': cpu_info}
            
        except ImportError:
            return {'error': 'WMI not available'}
        except Exception as e:
            self.logger.error(f"CPU info error: {e}")
            return {}
    
    async def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        try:
            import wmi
            c = wmi.WMI()
            
            memory_info = []
            for memory in c.Win32_PhysicalMemory():
                memory_info.append({
                    'capacity': int(memory.Capacity) if memory.Capacity else 0,
                    'speed': memory.Speed,
                    'manufacturer': memory.Manufacturer,
                    'part_number': memory.PartNumber,
                    'serial_number': memory.SerialNumber
                })
            
            return {'memory_modules': memory_info}
            
        except ImportError:
            return {'error': 'WMI not available'}
        except Exception as e:
            self.logger.error(f"Memory info error: {e}")
            return {}
    
    async def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            import wmi
            c = wmi.WMI()
            
            gpu_info = []
            for gpu in c.Win32_VideoController():
                gpu_info.append({
                    'name': gpu.Name,
                    'adapter_ram': gpu.AdapterRAM,
                    'driver_version': gpu.DriverVersion,
                    'driver_date': gpu.DriverDate,
                    'video_processor': gpu.VideoProcessor
                })
            
            return {'graphics_cards': gpu_info}
            
        except ImportError:
            return {'error': 'WMI not available'}
        except Exception as e:
            self.logger.error(f"GPU info error: {e}")
            return {}
    
    async def _get_motherboard_info(self) -> Dict[str, Any]:
        """Get motherboard information"""
        try:
            import wmi
            c = wmi.WMI()
            
            for board in c.Win32_BaseBoard():
                return {
                    'manufacturer': board.Manufacturer,
                    'product': board.Product,
                    'version': board.Version,
                    'serial_number': board.SerialNumber
                }
            
            return {}
            
        except ImportError:
            return {'error': 'WMI not available'}
        except Exception as e:
            self.logger.error(f"Motherboard info error: {e}")
            return {}
    
    async def _get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        try:
            network_info = {
                'interfaces': [],
                'connections': []
            }
            
            # Network interfaces
            for interface, addrs in psutil.net_if_addrs().items():
                interface_info = {
                    'name': interface,
                    'addresses': []
                }
                
                for addr in addrs:
                    interface_info['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
                
                network_info['interfaces'].append(interface_info)
            
            # Network connections
            for conn in psutil.net_connections():
                if conn.status == 'ESTABLISHED':
                    network_info['connections'].append({
                        'local_address': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                        'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                        'status': conn.status,
                        'pid': conn.pid
                    })
            
            return {'network': network_info}
            
        except Exception as e:
            self.logger.error(f"Network info error: {e}")
            return {}
    
    async def _get_process_info(self) -> Dict[str, Any]:
        """Get process information"""
        try:
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            
            return {
                'processes': {
                    'total_count': len(processes),
                    'top_cpu': processes[:10],  # Top 10 by CPU
                    'top_memory': sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:10]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Process info error: {e}")
            return {}
    
    async def _get_service_info(self) -> Dict[str, Any]:
        """Get Windows service information"""
        try:
            services = []
            
            for service in psutil.win_service_iter():
                try:
                    service_info = service.as_dict()
                    services.append({
                        'name': service_info['name'],
                        'display_name': service_info['display_name'],
                        'status': service_info['status'],
                        'start_type': service_info['start_type'],
                        'pid': service_info.get('pid')
                    })
                except Exception:
                    continue
            
            return {
                'services': {
                    'total_count': len(services),
                    'running': [s for s in services if s['status'] == 'running'],
                    'stopped': [s for s in services if s['status'] == 'stopped']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Service info error: {e}")
            return {}
    
    # Registry Operations
    async def get_registry_value(self, key_path: str, value_name: str, hive: str = 'HKEY_CURRENT_USER') -> Dict[str, Any]:
        """Get registry value"""
        try:
            hive_map = {
                'HKEY_CURRENT_USER': winreg.HKEY_CURRENT_USER,
                'HKEY_LOCAL_MACHINE': winreg.HKEY_LOCAL_MACHINE,
                'HKEY_CLASSES_ROOT': winreg.HKEY_CLASSES_ROOT,
                'HKEY_USERS': winreg.HKEY_USERS,
                'HKEY_CURRENT_CONFIG': winreg.HKEY_CURRENT_CONFIG
            }
            
            hive_key = hive_map.get(hive, winreg.HKEY_CURRENT_USER)
            
            with winreg.OpenKey(hive_key, key_path) as key:
                value, reg_type = winreg.QueryValueEx(key, value_name)
                
                self.operation_stats['registry_operations'] += 1
                
                return {
                    'success': True,
                    'value': value,
                    'type': reg_type,
                    'key_path': key_path,
                    'value_name': value_name
                }
                
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'Registry key or value not found',
                'key_path': key_path,
                'value_name': value_name
            }
        except Exception as e:
            self.logger.error(f"Registry read error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def set_registry_value(self, key_path: str, value_name: str, value: Any, value_type: int = winreg.REG_SZ, hive: str = 'HKEY_CURRENT_USER') -> Dict[str, Any]:
        """Set registry value"""
        try:
            if not self.is_admin and hive == 'HKEY_LOCAL_MACHINE':
                return {
                    'success': False,
                    'error': 'Administrator privileges required for HKEY_LOCAL_MACHINE'
                }
            
            hive_map = {
                'HKEY_CURRENT_USER': winreg.HKEY_CURRENT_USER,
                'HKEY_LOCAL_MACHINE': winreg.HKEY_LOCAL_MACHINE,
                'HKEY_CLASSES_ROOT': winreg.HKEY_CLASSES_ROOT,
                'HKEY_USERS': winreg.HKEY_USERS,
                'HKEY_CURRENT_CONFIG': winreg.HKEY_CURRENT_CONFIG
            }
            
            hive_key = hive_map.get(hive, winreg.HKEY_CURRENT_USER)
            
            with winreg.CreateKey(hive_key, key_path) as key:
                winreg.SetValueEx(key, value_name, 0, value_type, value)
                
                self.operation_stats['registry_operations'] += 1
                
                return {
                    'success': True,
                    'key_path': key_path,
                    'value_name': value_name,
                    'value': value
                }
                
        except Exception as e:
            self.logger.error(f"Registry write error: {e}")
            return {'success': False, 'error': str(e)}
    
    # System Settings
    async def set_system_setting(self, setting_name: str, setting_value: Any) -> Dict[str, Any]:
        """Set system setting"""
        try:
            setting_handlers = {
                'volume': self._set_volume,
                'brightness': self._set_brightness,
                'wallpaper': self._set_wallpaper,
                'power_plan': self._set_power_plan,
                'display_timeout': self._set_display_timeout
            }
            
            if setting_name in setting_handlers:
                handler = setting_handlers[setting_name]
                result = await handler(setting_value)
                return result
            else:
                return {
                    'success': False,
                    'error': f'Unknown setting: {setting_name}',
                    'available_settings': list(setting_handlers.keys())
                }
                
        except Exception as e:
            self.logger.error(f"System setting error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _set_volume(self, volume: int) -> Dict[str, Any]:
        """Set system volume (0-100)"""
        try:
            # Use Windows API to set volume
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Set volume (0.0 to 1.0)
            volume_level = max(0, min(100, volume)) / 100.0
            volume_interface.SetMasterScalarVolume(volume_level, None)
            
            return {
                'success': True,
                'setting': 'volume',
                'value': volume
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'pycaw library required for volume control'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _set_brightness(self, brightness: int) -> Dict[str, Any]:
        """Set display brightness (0-100)"""
        try:
            import wmi
            c = wmi.WMI(namespace='wmi')
            
            brightness_level = max(0, min(100, brightness))
            
            for method in c.WmiMonitorBrightnessMethods():
                method.WmiSetBrightness(brightness_level, 0)
            
            return {
                'success': True,
                'setting': 'brightness',
                'value': brightness_level
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'WMI library required for brightness control'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _set_wallpaper(self, image_path: str) -> Dict[str, Any]:
        """Set desktop wallpaper"""
        try:
            import ctypes
            
            # Verify image exists
            if not Path(image_path).exists():
                return {
                    'success': False,
                    'error': 'Image file not found',
                    'path': image_path
                }
            
            # Set wallpaper using Windows API
            SPI_SETDESKWALLPAPER = 20
            result = ctypes.windll.user32.SystemParametersInfoW(
                SPI_SETDESKWALLPAPER, 0, image_path, 3
            )
            
            if result:
                return {
                    'success': True,
                    'setting': 'wallpaper',
                    'value': image_path
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to set wallpaper'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _set_power_plan(self, plan_name: str) -> Dict[str, Any]:
        """Set power plan"""
        try:
            # Use powercfg command
            result = subprocess.run(
                ['powercfg', '/setactive', plan_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'setting': 'power_plan',
                    'value': plan_name
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Failed to set power plan'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _set_display_timeout(self, timeout_minutes: int) -> Dict[str, Any]:
        """Set display timeout"""
        try:
            # Use powercfg command
            result = subprocess.run(
                ['powercfg', '/change', 'monitor-timeout-ac', str(timeout_minutes)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'setting': 'display_timeout',
                    'value': timeout_minutes
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Failed to set display timeout'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # Cache management
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.system_info_cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        return time.time() < self.cache_expiry[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache result with expiry"""
        self.system_info_cache[cache_key] = result
        self.cache_expiry[cache_key] = time.time() + self.cache_duration
    
    async def _cache_system_info(self, info_type: str):
        """Pre-cache system information"""
        try:
            await self.get_system_info(info_type)
        except Exception as e:
            self.logger.error(f"System info caching error: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get system controller performance statistics"""
        return {
            'operation_stats': self.operation_stats.copy(),
            'cache_stats': {
                'cached_entries': len(self.system_info_cache),
                'cache_hit_ratio': 0.0  # Would need to track hits/misses
            },
            'system_capabilities': {
                'is_admin': self.is_admin,
                'file_access': self.file_access,
                'registry_access': self.registry_access,
                'api_available': self.api_available
            }
        }
    
    async def cleanup(self):
        """Cleanup system controller"""
        try:
            self.logger.info("Cleaning up System Controller...")
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Clear caches
            self.system_info_cache.clear()
            self.cache_expiry.clear()
            
            self.logger.info("System Controller cleanup completed")
            
        except Exception as e:
            self.logger.error(f"System Controller cleanup error: {e}")
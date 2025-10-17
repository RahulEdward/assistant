"""
System Monitor Widget for Computer Assistant GUI
Provides real-time system monitoring, performance metrics, and resource usage
"""

import logging
import psutil
import platform
from typing import Dict, Any, List
from datetime import datetime, timedelta

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QListWidget, QListWidgetItem, QTextEdit, QComboBox,
    QCheckBox, QSpinBox, QSlider, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
import pyqtgraph as pg


class SystemMonitorThread(QThread):
    """Thread for collecting system information"""
    
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        
    def run(self):
        """Collect system data"""
        self.running = True
        
        while self.running:
            try:
                data = self.collect_system_data()
                self.data_updated.emit(data)
                self.msleep(1000)  # Update every second
                
            except Exception as e:
                logging.error(f"Error collecting system data: {e}")
                self.msleep(5000)  # Wait longer on error
    
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        self.wait()
    
    def collect_system_data(self) -> Dict[str, Any]:
        """Collect comprehensive system data"""
        data = {}
        
        # CPU Information
        data['cpu'] = {
            'percent': psutil.cpu_percent(interval=0.1),
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True),
            'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True)
        }
        
        # Memory Information
        memory = psutil.virtual_memory()
        data['memory'] = {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'free': memory.free
        }
        
        # Swap Information
        swap = psutil.swap_memory()
        data['swap'] = {
            'total': swap.total,
            'used': swap.used,
            'free': swap.free,
            'percent': swap.percent
        }
        
        # Disk Information
        data['disk'] = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                data['disk'][partition.device] = {
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                }
            except PermissionError:
                continue
        
        # Network Information
        net_io = psutil.net_io_counters()
        data['network'] = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # Process Information
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        data['processes'] = processes[:20]  # Top 20 processes
        
        # System Information
        data['system'] = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'boot_time': psutil.boot_time(),
            'uptime': datetime.now().timestamp() - psutil.boot_time()
        }
        
        # Temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            data['temperature'] = temps
        except AttributeError:
            data['temperature'] = {}
        
        # Battery (if available)
        try:
            battery = psutil.sensors_battery()
            if battery:
                data['battery'] = {
                    'percent': battery.percent,
                    'secsleft': battery.secsleft,
                    'power_plugged': battery.power_plugged
                }
        except AttributeError:
            data['battery'] = None
        
        return data


class CPUWidget(QGroupBox):
    """Widget for CPU monitoring"""
    
    def __init__(self):
        super().__init__("CPU Usage")
        self.cpu_history = []
        self.max_history = 60  # Keep 60 seconds of history
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # CPU overall usage
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.cpu_label.setFont(font)
        layout.addWidget(self.cpu_label)
        
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        layout.addWidget(self.cpu_progress)
        
        # CPU cores
        self.cores_layout = QGridLayout()
        layout.addLayout(self.cores_layout)
        
        # CPU chart
        self.cpu_chart = pg.PlotWidget()
        self.cpu_chart.setLabel('left', 'CPU Usage (%)')
        self.cpu_chart.setLabel('bottom', 'Time (seconds ago)')
        self.cpu_chart.setYRange(0, 100)
        self.cpu_chart.setMaximumHeight(150)
        layout.addWidget(self.cpu_chart)
        
        self.cpu_curve = self.cpu_chart.plot(pen='b')
        
    def update_cpu_data(self, cpu_data: Dict[str, Any]):
        """Update CPU display with new data"""
        cpu_percent = cpu_data.get('percent', 0)
        per_cpu = cpu_data.get('per_cpu', [])
        
        # Update overall CPU
        self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
        self.cpu_progress.setValue(int(cpu_percent))
        
        # Update CPU history
        self.cpu_history.append(cpu_percent)
        if len(self.cpu_history) > self.max_history:
            self.cpu_history.pop(0)
        
        # Update chart
        x_data = list(range(-len(self.cpu_history) + 1, 1))
        self.cpu_curve.setData(x_data, self.cpu_history)
        
        # Update per-core display
        self.update_cpu_cores(per_cpu)
        
        # Color coding based on usage
        if cpu_percent > 80:
            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #ff4444; }")
        elif cpu_percent > 60:
            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #ffaa00; }")
        else:
            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #44ff44; }")
    
    def update_cpu_cores(self, per_cpu: List[float]):
        """Update per-core CPU display"""
        # Clear existing core widgets
        for i in reversed(range(self.cores_layout.count())):
            self.cores_layout.itemAt(i).widget().setParent(None)
        
        # Add core widgets
        cols = 4
        for i, cpu_percent in enumerate(per_cpu):
            row = i // cols
            col = i % cols
            
            core_widget = QFrame()
            core_layout = QVBoxLayout(core_widget)
            core_layout.setContentsMargins(2, 2, 2, 2)
            
            core_label = QLabel(f"Core {i}")
            core_label.setAlignment(Qt.AlignCenter)
            core_layout.addWidget(core_label)
            
            core_progress = QProgressBar()
            core_progress.setRange(0, 100)
            core_progress.setValue(int(cpu_percent))
            core_progress.setMaximumHeight(15)
            core_layout.addWidget(core_progress)
            
            percent_label = QLabel(f"{cpu_percent:.1f}%")
            percent_label.setAlignment(Qt.AlignCenter)
            percent_label.setStyleSheet("font-size: 8pt;")
            core_layout.addWidget(percent_label)
            
            self.cores_layout.addWidget(core_widget, row, col)


class MemoryWidget(QGroupBox):
    """Widget for memory monitoring"""
    
    def __init__(self):
        super().__init__("Memory Usage")
        self.memory_history = []
        self.max_history = 60
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Memory info
        info_layout = QGridLayout()
        
        self.memory_label = QLabel("RAM: 0%")
        self.memory_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.memory_label.setFont(font)
        layout.addWidget(self.memory_label)
        
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        layout.addWidget(self.memory_progress)
        
        # Memory details
        self.total_label = QLabel("Total: 0 GB")
        self.used_label = QLabel("Used: 0 GB")
        self.available_label = QLabel("Available: 0 GB")
        self.free_label = QLabel("Free: 0 GB")
        
        info_layout.addWidget(self.total_label, 0, 0)
        info_layout.addWidget(self.used_label, 0, 1)
        info_layout.addWidget(self.available_label, 1, 0)
        info_layout.addWidget(self.free_label, 1, 1)
        
        layout.addLayout(info_layout)
        
        # Swap info
        swap_group = QGroupBox("Swap Memory")
        swap_layout = QVBoxLayout(swap_group)
        
        self.swap_progress = QProgressBar()
        self.swap_progress.setRange(0, 100)
        swap_layout.addWidget(self.swap_progress)
        
        self.swap_label = QLabel("Swap: 0% (0 GB / 0 GB)")
        swap_layout.addWidget(self.swap_label)
        
        layout.addWidget(swap_group)
        
        # Memory chart
        self.memory_chart = pg.PlotWidget()
        self.memory_chart.setLabel('left', 'Memory Usage (%)')
        self.memory_chart.setLabel('bottom', 'Time (seconds ago)')
        self.memory_chart.setYRange(0, 100)
        self.memory_chart.setMaximumHeight(150)
        layout.addWidget(self.memory_chart)
        
        self.memory_curve = self.memory_chart.plot(pen='g')
    
    def update_memory_data(self, memory_data: Dict[str, Any], swap_data: Dict[str, Any]):
        """Update memory display with new data"""
        # Memory
        total = memory_data.get('total', 0)
        used = memory_data.get('used', 0)
        available = memory_data.get('available', 0)
        free = memory_data.get('free', 0)
        percent = memory_data.get('percent', 0)
        
        # Update labels
        self.memory_label.setText(f"RAM: {percent:.1f}%")
        self.memory_progress.setValue(int(percent))
        
        self.total_label.setText(f"Total: {self.format_bytes(total)}")
        self.used_label.setText(f"Used: {self.format_bytes(used)}")
        self.available_label.setText(f"Available: {self.format_bytes(available)}")
        self.free_label.setText(f"Free: {self.format_bytes(free)}")
        
        # Update history
        self.memory_history.append(percent)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
        
        # Update chart
        x_data = list(range(-len(self.memory_history) + 1, 1))
        self.memory_curve.setData(x_data, self.memory_history)
        
        # Swap
        swap_total = swap_data.get('total', 0)
        swap_used = swap_data.get('used', 0)
        swap_percent = swap_data.get('percent', 0)
        
        self.swap_progress.setValue(int(swap_percent))
        self.swap_label.setText(f"Swap: {swap_percent:.1f}% ({self.format_bytes(swap_used)} / {self.format_bytes(swap_total)})")
        
        # Color coding
        if percent > 85:
            self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: #ff4444; }")
        elif percent > 70:
            self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: #ffaa00; }")
        else:
            self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: #44ff44; }")
    
    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"


class ProcessWidget(QGroupBox):
    """Widget for process monitoring"""
    
    def __init__(self):
        super().__init__("Running Processes")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["CPU Usage", "Memory Usage", "Process Name", "PID"])
        controls_layout.addWidget(QLabel("Sort by:"))
        controls_layout.addWidget(self.sort_combo)
        
        self.refresh_btn = QPushButton("Refresh")
        controls_layout.addWidget(self.refresh_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Process table
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(5)
        self.process_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "Memory %", "Status"])
        self.process_table.setSortingEnabled(True)
        layout.addWidget(self.process_table)
    
    def update_process_data(self, processes: List[Dict[str, Any]]):
        """Update process table with new data"""
        self.process_table.setRowCount(len(processes))
        
        for row, proc in enumerate(processes):
            self.process_table.setItem(row, 0, QTableWidgetItem(str(proc.get('pid', 'N/A'))))
            self.process_table.setItem(row, 1, QTableWidgetItem(proc.get('name', 'N/A')))
            self.process_table.setItem(row, 2, QTableWidgetItem(f"{proc.get('cpu_percent', 0):.1f}"))
            self.process_table.setItem(row, 3, QTableWidgetItem(f"{proc.get('memory_percent', 0):.1f}"))
            self.process_table.setItem(row, 4, QTableWidgetItem(proc.get('status', 'N/A')))


class SystemInfoWidget(QGroupBox):
    """Widget for system information"""
    
    def __init__(self):
        super().__init__("System Information")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        layout.addWidget(self.info_text)
    
    def update_system_info(self, system_data: Dict[str, Any]):
        """Update system information display"""
        info_text = []
        
        info_text.append(f"Platform: {system_data.get('platform', 'Unknown')}")
        info_text.append(f"Processor: {system_data.get('processor', 'Unknown')}")
        
        arch = system_data.get('architecture', ['Unknown', 'Unknown'])
        info_text.append(f"Architecture: {arch[0]} ({arch[1]})")
        
        boot_time = system_data.get('boot_time', 0)
        if boot_time:
            boot_datetime = datetime.fromtimestamp(boot_time)
            info_text.append(f"Boot Time: {boot_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        uptime = system_data.get('uptime', 0)
        if uptime:
            uptime_str = str(timedelta(seconds=int(uptime)))
            info_text.append(f"Uptime: {uptime_str}")
        
        self.info_text.setPlainText('\n'.join(info_text))


class SystemMonitorWidget(QWidget):
    """Main system monitor widget"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        self.monitor_thread = None
        
        self.init_ui()
        self.start_monitoring()
        
    def init_ui(self):
        """Initialize the system monitor UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("System Monitor")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Overview tab
        overview_widget = QWidget()
        overview_layout = QGridLayout(overview_widget)
        
        self.cpu_widget = CPUWidget()
        overview_layout.addWidget(self.cpu_widget, 0, 0)
        
        self.memory_widget = MemoryWidget()
        overview_layout.addWidget(self.memory_widget, 0, 1)
        
        self.system_info_widget = SystemInfoWidget()
        overview_layout.addWidget(self.system_info_widget, 1, 0, 1, 2)
        
        self.tab_widget.addTab(overview_widget, "Overview")
        
        # Processes tab
        self.process_widget = ProcessWidget()
        self.tab_widget.addTab(self.process_widget, "Processes")
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        controls_layout.addWidget(self.auto_refresh_cb)
        
        self.refresh_interval_spin = QSpinBox()
        self.refresh_interval_spin.setRange(1, 60)
        self.refresh_interval_spin.setValue(1)
        self.refresh_interval_spin.setSuffix(" sec")
        controls_layout.addWidget(QLabel("Interval:"))
        controls_layout.addWidget(self.refresh_interval_spin)
        
        controls_layout.addStretch()
        
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        controls_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(controls_layout)
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitor_thread and self.monitor_thread.isRunning():
            return
        
        try:
            self.monitor_thread = SystemMonitorThread()
            self.monitor_thread.data_updated.connect(self.update_displays)
            self.monitor_thread.start()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            self.logger.info("System monitoring started")
            
        except Exception as e:
            self.logger.error(f"Error starting system monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.logger.info("System monitoring stopped")
    
    def update_displays(self, data: Dict[str, Any]):
        """Update all display widgets with new data"""
        try:
            # Update CPU widget
            if 'cpu' in data:
                self.cpu_widget.update_cpu_data(data['cpu'])
            
            # Update memory widget
            if 'memory' in data and 'swap' in data:
                self.memory_widget.update_memory_data(data['memory'], data['swap'])
            
            # Update process widget
            if 'processes' in data:
                self.process_widget.update_process_data(data['processes'])
            
            # Update system info widget
            if 'system' in data:
                self.system_info_widget.update_system_info(data['system'])
                
        except Exception as e:
            self.logger.error(f"Error updating displays: {e}")
    
    def closeEvent(self, event):
        """Handle widget close event"""
        self.stop_monitoring()
        super().closeEvent(event)
"""
OCR Widget for Computer Assistant GUI
Provides optical character recognition, screen analysis, and text extraction functionality
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import base64
from io import BytesIO

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QTextEdit, QComboBox, QListWidget,
    QListWidgetItem, QProgressBar, QSpinBox, QCheckBox,
    QTabWidget, QFileDialog, QMessageBox, QSlider, QFrame,
    QScrollArea, QSplitter, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QRect
from PySide6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QCursor
import PIL.Image
import PIL.ImageGrab


class ScreenCaptureThread(QThread):
    """Thread for capturing screen regions"""
    
    capture_completed = Signal(object)  # PIL Image
    capture_error = Signal(str)
    
    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None):
        super().__init__()
        self.region = region  # (x, y, width, height)
        
    def run(self):
        """Capture screen or region"""
        try:
            if self.region:
                # Capture specific region
                x, y, width, height = self.region
                bbox = (x, y, x + width, y + height)
                image = PIL.ImageGrab.grab(bbox=bbox)
            else:
                # Capture entire screen
                image = PIL.ImageGrab.grab()
            
            self.capture_completed.emit(image)
            
        except Exception as e:
            self.capture_error.emit(str(e))


class OCRProcessingThread(QThread):
    """Thread for OCR processing"""
    
    ocr_completed = Signal(dict)
    ocr_error = Signal(str)
    
    def __init__(self, image: PIL.Image.Image, ocr_engine, language: str = 'eng'):
        super().__init__()
        self.image = image
        self.ocr_engine = ocr_engine
        self.language = language
        
    def run(self):
        """Process OCR on image"""
        try:
            if self.ocr_engine:
                # Use actual OCR engine
                result = self.ocr_engine.extract_text(self.image, language=self.language)
            else:
                # Simulate OCR processing
                import time
                time.sleep(2)
                
                # Mock OCR result
                result = {
                    'text': "This is sample OCR text extracted from the image.\nLine 2 of extracted text.\nLine 3 with more content.",
                    'confidence': 85.5,
                    'words': [
                        {'text': 'This', 'confidence': 90.2, 'bbox': (10, 10, 50, 30)},
                        {'text': 'is', 'confidence': 88.1, 'bbox': (55, 10, 75, 30)},
                        {'text': 'sample', 'confidence': 92.3, 'bbox': (80, 10, 140, 30)},
                        {'text': 'OCR', 'confidence': 87.5, 'bbox': (145, 10, 180, 30)},
                        {'text': 'text', 'confidence': 89.8, 'bbox': (185, 10, 220, 30)}
                    ],
                    'lines': [
                        {'text': 'This is sample OCR text extracted from the image.', 'confidence': 89.2, 'bbox': (10, 10, 400, 30)},
                        {'text': 'Line 2 of extracted text.', 'confidence': 86.7, 'bbox': (10, 35, 200, 55)},
                        {'text': 'Line 3 with more content.', 'confidence': 88.9, 'bbox': (10, 60, 220, 80)}
                    ]
                }
            
            self.ocr_completed.emit(result)
            
        except Exception as e:
            self.ocr_error.emit(str(e))


class ScreenCaptureWidget(QGroupBox):
    """Widget for screen capture functionality"""
    
    capture_requested = Signal(object)  # region tuple or None
    
    def __init__(self):
        super().__init__("Screen Capture")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Capture options
        options_layout = QGridLayout()
        
        # Capture mode
        options_layout.addWidget(QLabel("Capture Mode:"), 0, 0)
        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItems([
            "Full Screen",
            "Active Window",
            "Custom Region",
            "From File"
        ])
        options_layout.addWidget(self.capture_mode_combo, 0, 1)
        
        # Region selection (for custom region)
        options_layout.addWidget(QLabel("Region (x,y,w,h):"), 1, 0)
        region_layout = QHBoxLayout()
        
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 9999)
        self.x_spin.setValue(100)
        region_layout.addWidget(self.x_spin)
        
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 9999)
        self.y_spin.setValue(100)
        region_layout.addWidget(self.y_spin)
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 9999)
        self.width_spin.setValue(800)
        region_layout.addWidget(self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 9999)
        self.height_spin.setValue(600)
        region_layout.addWidget(self.height_spin)
        
        region_widget = QWidget()
        region_widget.setLayout(region_layout)
        options_layout.addWidget(region_widget, 1, 1)
        
        layout.addLayout(options_layout)
        
        # Capture buttons
        button_layout = QHBoxLayout()
        
        self.capture_btn = QPushButton("Capture Screen")
        self.capture_btn.clicked.connect(self.capture_screen)
        button_layout.addWidget(self.capture_btn)
        
        self.select_region_btn = QPushButton("Select Region")
        self.select_region_btn.clicked.connect(self.select_region)
        button_layout.addWidget(self.select_region_btn)
        
        self.load_file_btn = QPushButton("Load Image File")
        self.load_file_btn.clicked.connect(self.load_image_file)
        button_layout.addWidget(self.load_file_btn)
        
        layout.addLayout(button_layout)
        
        # Capture preview
        self.preview_label = QLabel("No image captured")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(150)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        layout.addWidget(self.preview_label)
        
        # Image info
        self.info_label = QLabel("Image info will appear here")
        layout.addWidget(self.info_label)
    
    def capture_screen(self):
        """Capture screen based on selected mode"""
        mode = self.capture_mode_combo.currentText()
        
        if mode == "Full Screen":
            self.capture_requested.emit(None)
        elif mode == "Active Window":
            # For now, treat as full screen - could be enhanced to capture active window
            self.capture_requested.emit(None)
        elif mode == "Custom Region":
            region = (
                self.x_spin.value(),
                self.y_spin.value(),
                self.width_spin.value(),
                self.height_spin.value()
            )
            self.capture_requested.emit(region)
    
    def select_region(self):
        """Interactive region selection (placeholder)"""
        QMessageBox.information(
            self, "Region Selection",
            "Interactive region selection will be implemented in a future version.\n"
            "For now, please use the coordinate inputs above."
        )
    
    def load_image_file(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        
        if file_path:
            try:
                image = PIL.Image.open(file_path)
                self.display_image(image)
                
                # Emit as if it was captured
                self.capture_requested.emit(image)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
    
    def display_image(self, image: PIL.Image.Image):
        """Display captured image in preview"""
        try:
            # Convert PIL image to QPixmap
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for preview
            preview_size = (300, 200)
            image_preview = image.copy()
            image_preview.thumbnail(preview_size, PIL.Image.Resampling.LANCZOS)
            
            # Convert to QImage
            width, height = image_preview.size
            bytes_per_line = 3 * width
            q_image = QImage(
                image_preview.tobytes(), width, height,
                bytes_per_line, QImage.Format_RGB888
            )
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap)
            
            # Update info
            self.info_label.setText(f"Size: {image.size[0]}x{image.size[1]}, Mode: {image.mode}")
            
        except Exception as e:
            logging.error(f"Error displaying image: {e}")
            self.preview_label.setText("Error displaying image")


class OCRResultsWidget(QGroupBox):
    """Widget for displaying OCR results"""
    
    def __init__(self):
        super().__init__("OCR Results")
        self.current_result = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)
        
        # Text tab
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        
        self.extracted_text = QTextEdit()
        self.extracted_text.setPlaceholderText("Extracted text will appear here...")
        text_layout.addWidget(self.extracted_text)
        
        # Text controls
        text_controls = QHBoxLayout()
        
        self.copy_text_btn = QPushButton("Copy Text")
        self.copy_text_btn.clicked.connect(self.copy_text)
        text_controls.addWidget(self.copy_text_btn)
        
        self.save_text_btn = QPushButton("Save Text")
        self.save_text_btn.clicked.connect(self.save_text)
        text_controls.addWidget(self.save_text_btn)
        
        self.clear_text_btn = QPushButton("Clear")
        self.clear_text_btn.clicked.connect(self.clear_results)
        text_controls.addWidget(self.clear_text_btn)
        
        text_controls.addStretch()
        text_layout.addLayout(text_controls)
        
        self.results_tabs.addTab(text_widget, "Extracted Text")
        
        # Details tab
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        # Confidence info
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Overall Confidence:"))
        self.confidence_label = QLabel("N/A")
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addStretch()
        details_layout.addLayout(confidence_layout)
        
        # Word/line details
        self.details_list = QListWidget()
        details_layout.addWidget(self.details_list)
        
        self.results_tabs.addTab(details_widget, "Details")
        
        # Statistics tab
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        self.results_tabs.addTab(stats_widget, "Statistics")
    
    def display_ocr_result(self, result: Dict[str, Any]):
        """Display OCR results"""
        self.current_result = result
        
        # Display extracted text
        text = result.get('text', '')
        self.extracted_text.setPlainText(text)
        
        # Display confidence
        confidence = result.get('confidence', 0)
        self.confidence_label.setText(f"{confidence:.1f}%")
        
        # Display word/line details
        self.details_list.clear()
        
        # Add lines
        lines = result.get('lines', [])
        for i, line in enumerate(lines):
            item_text = f"Line {i+1}: {line.get('text', '')} (Confidence: {line.get('confidence', 0):.1f}%)"
            item = QListWidgetItem(item_text)
            self.details_list.addItem(item)
        
        # Add words
        words = result.get('words', [])
        for i, word in enumerate(words):
            item_text = f"Word {i+1}: {word.get('text', '')} (Confidence: {word.get('confidence', 0):.1f}%)"
            item = QListWidgetItem(item_text)
            self.details_list.addItem(item)
        
        # Display statistics
        self.update_statistics(result)
    
    def update_statistics(self, result: Dict[str, Any]):
        """Update statistics display"""
        stats = []
        
        text = result.get('text', '')
        stats.append(f"Total Characters: {len(text)}")
        stats.append(f"Total Words: {len(text.split())}")
        stats.append(f"Total Lines: {len(text.splitlines())}")
        
        words = result.get('words', [])
        if words:
            confidences = [w.get('confidence', 0) for w in words]
            stats.append(f"Word Count: {len(words)}")
            stats.append(f"Average Word Confidence: {sum(confidences) / len(confidences):.1f}%")
            stats.append(f"Min Word Confidence: {min(confidences):.1f}%")
            stats.append(f"Max Word Confidence: {max(confidences):.1f}%")
        
        lines = result.get('lines', [])
        if lines:
            line_confidences = [l.get('confidence', 0) for l in lines]
            stats.append(f"Line Count: {len(lines)}")
            stats.append(f"Average Line Confidence: {sum(line_confidences) / len(line_confidences):.1f}%")
        
        overall_confidence = result.get('confidence', 0)
        stats.append(f"Overall Confidence: {overall_confidence:.1f}%")
        
        self.stats_text.setPlainText('\n'.join(stats))
    
    def copy_text(self):
        """Copy extracted text to clipboard"""
        text = self.extracted_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            QMessageBox.information(self, "Success", "Text copied to clipboard.")
        else:
            QMessageBox.information(self, "Info", "No text to copy.")
    
    def save_text(self):
        """Save extracted text to file"""
        text = self.extracted_text.toPlainText()
        if not text:
            QMessageBox.information(self, "Info", "No text to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Extracted Text", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "Success", "Text saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save text: {e}")
    
    def clear_results(self):
        """Clear all results"""
        self.extracted_text.clear()
        self.confidence_label.setText("N/A")
        self.details_list.clear()
        self.stats_text.clear()
        self.current_result = None


class OCRSettingsWidget(QGroupBox):
    """Widget for OCR settings and configuration"""
    
    def __init__(self):
        super().__init__("OCR Settings")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Language selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "English (eng)",
            "Spanish (spa)",
            "French (fra)",
            "German (deu)",
            "Italian (ita)",
            "Portuguese (por)",
            "Russian (rus)",
            "Chinese Simplified (chi_sim)",
            "Chinese Traditional (chi_tra)",
            "Japanese (jpn)",
            "Korean (kor)",
            "Arabic (ara)"
        ])
        lang_layout.addWidget(self.language_combo)
        
        layout.addLayout(lang_layout)
        
        # OCR Engine
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("OCR Engine:"))
        
        self.engine_combo = QComboBox()
        self.engine_combo.addItems([
            "Tesseract",
            "EasyOCR",
            "PaddleOCR",
            "Azure Computer Vision",
            "Google Cloud Vision"
        ])
        engine_layout.addWidget(self.engine_combo)
        
        layout.addLayout(engine_layout)
        
        # Processing options
        self.preprocess_cb = QCheckBox("Preprocess Image")
        self.preprocess_cb.setChecked(True)
        layout.addWidget(self.preprocess_cb)
        
        self.enhance_cb = QCheckBox("Enhance Image Quality")
        self.enhance_cb.setChecked(True)
        layout.addWidget(self.enhance_cb)
        
        self.detect_orientation_cb = QCheckBox("Auto-detect Orientation")
        self.detect_orientation_cb.setChecked(False)
        layout.addWidget(self.detect_orientation_cb)
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Min Confidence:"))
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        threshold_layout.addWidget(self.confidence_slider)
        
        self.confidence_value_label = QLabel("50%")
        threshold_layout.addWidget(self.confidence_value_label)
        
        layout.addLayout(threshold_layout)
    
    def update_confidence_label(self, value):
        """Update confidence threshold label"""
        self.confidence_value_label.setText(f"{value}%")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current OCR settings"""
        language_text = self.language_combo.currentText()
        language_code = language_text.split('(')[-1].rstrip(')')
        
        return {
            'language': language_code,
            'engine': self.engine_combo.currentText(),
            'preprocess': self.preprocess_cb.isChecked(),
            'enhance': self.enhance_cb.isChecked(),
            'detect_orientation': self.detect_orientation_cb.isChecked(),
            'min_confidence': self.confidence_slider.value()
        }


class OCRWidget(QWidget):
    """Main OCR widget"""
    
    def __init__(self, assistant_manager):
        super().__init__()
        self.assistant_manager = assistant_manager
        self.logger = logging.getLogger(__name__)
        self.capture_thread = None
        self.ocr_thread = None
        self.current_image = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the OCR UI"""
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Optical Character Recognition (OCR)")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Capture and Settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Screen capture widget
        self.capture_widget = ScreenCaptureWidget()
        self.capture_widget.capture_requested.connect(self.process_capture_request)
        left_layout.addWidget(self.capture_widget)
        
        # OCR settings widget
        self.settings_widget = OCRSettingsWidget()
        left_layout.addWidget(self.settings_widget)
        
        # Process button
        self.process_btn = QPushButton("Process OCR")
        self.process_btn.clicked.connect(self.process_ocr)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)
        
        # Status
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        left_layout.addLayout(status_layout)
        
        main_splitter.addWidget(left_panel)
        
        # Right panel - Results
        self.results_widget = OCRResultsWidget()
        main_splitter.addWidget(self.results_widget)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 600])
    
    def process_capture_request(self, region_or_image):
        """Process screen capture request"""
        if isinstance(region_or_image, PIL.Image.Image):
            # Direct image provided
            self.current_image = region_or_image
            self.capture_widget.display_image(region_or_image)
            self.process_btn.setEnabled(True)
            self.status_label.setText("Image loaded - ready for OCR")
        else:
            # Screen capture requested
            if self.capture_thread and self.capture_thread.isRunning():
                return
            
            self.status_label.setText("Capturing screen...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            self.capture_thread = ScreenCaptureThread(region_or_image)
            self.capture_thread.capture_completed.connect(self.on_capture_completed)
            self.capture_thread.capture_error.connect(self.on_capture_error)
            self.capture_thread.start()
    
    def on_capture_completed(self, image: PIL.Image.Image):
        """Handle completed screen capture"""
        self.current_image = image
        self.capture_widget.display_image(image)
        
        self.status_label.setText("Screen captured - ready for OCR")
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
    
    def on_capture_error(self, error: str):
        """Handle capture error"""
        self.status_label.setText("Capture failed")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Capture Error", f"Failed to capture screen: {error}")
    
    def process_ocr(self):
        """Process OCR on current image"""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "No image to process. Please capture or load an image first.")
            return
        
        if self.ocr_thread and self.ocr_thread.isRunning():
            QMessageBox.warning(self, "Warning", "OCR processing is already in progress.")
            return
        
        # Get OCR settings
        settings = self.settings_widget.get_settings()
        
        # Update status
        self.status_label.setText("Processing OCR...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.process_btn.setEnabled(False)
        
        # Get OCR engine from assistant manager
        ocr_engine = None
        if self.assistant_manager and hasattr(self.assistant_manager, 'screen_reader'):
            ocr_engine = self.assistant_manager.screen_reader
        
        # Start OCR processing
        self.ocr_thread = OCRProcessingThread(
            self.current_image, 
            ocr_engine, 
            settings['language']
        )
        self.ocr_thread.ocr_completed.connect(self.on_ocr_completed)
        self.ocr_thread.ocr_error.connect(self.on_ocr_error)
        self.ocr_thread.start()
    
    def on_ocr_completed(self, result: Dict[str, Any]):
        """Handle completed OCR processing"""
        self.status_label.setText("OCR completed")
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # Display results
        self.results_widget.display_ocr_result(result)
        
        # Log success
        confidence = result.get('confidence', 0)
        text_length = len(result.get('text', ''))
        self.logger.info(f"OCR completed: {text_length} characters extracted with {confidence:.1f}% confidence")
    
    def on_ocr_error(self, error: str):
        """Handle OCR processing error"""
        self.status_label.setText("OCR failed")
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        QMessageBox.critical(self, "OCR Error", f"Failed to process OCR: {error}")
        self.logger.error(f"OCR processing failed: {error}")
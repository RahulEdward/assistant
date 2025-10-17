"""
Screen Analyzer
Advanced screen content analysis and visual understanding.
Provides comprehensive screen reading, layout analysis, and interactive element detection.
"""

import asyncio
import logging
import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import base64
import io

# PIL for image processing
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Windows screen capture
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

# Alternative screen capture
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# OpenCV for image processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class ScreenRegion:
    """Screen region information"""
    x: int
    y: int
    width: int
    height: int
    content_type: str  # text, button, input, image, menu, etc.
    text: str
    confidence: float
    properties: Dict[str, Any]
    children: List['ScreenRegion']


@dataclass
class ScreenElement:
    """Interactive screen element"""
    element_id: str
    element_type: str  # button, textbox, dropdown, checkbox, etc.
    bbox: Tuple[int, int, int, int]
    text: str
    properties: Dict[str, Any]
    clickable: bool
    focusable: bool
    visible: bool
    enabled: bool


@dataclass
class ScreenAnalysis:
    """Complete screen analysis result"""
    screenshot: np.ndarray
    timestamp: datetime
    regions: List[ScreenRegion]
    elements: List[ScreenElement]
    layout_structure: Dict[str, Any]
    text_content: str
    confidence: float
    processing_time: float


class ScreenAnalyzer:
    """Advanced screen content analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Screen capture configuration
        self.capture_config = {
            'method': 'auto',  # auto, win32, mss
            'quality': 'high',  # low, medium, high
            'color_mode': 'RGB',
            'compression': False
        }
        
        # Analysis configuration
        self.analysis_config = {
            'detect_text': True,
            'detect_elements': True,
            'analyze_layout': True,
            'extract_structure': True,
            'confidence_threshold': 0.5,
            'element_types': [
                'button', 'textbox', 'dropdown', 'checkbox', 'radio',
                'menu', 'menuitem', 'tab', 'link', 'image', 'icon'
            ]
        }
        
        # OCR engine reference
        self.ocr_engine = None
        
        # Element detection models
        self.element_detectors = {}
        
        # Performance tracking
        self.analysis_stats = {
            'screens_analyzed': 0,
            'elements_detected': 0,
            'text_regions_found': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Cache for screen analysis
        self.analysis_cache = {}
        self.cache_max_size = 50
        
        # Screen monitoring
        self.monitoring_active = False
        self.monitoring_callbacks = []
        self.last_screenshot = None
        self.change_threshold = 0.1  # 10% change threshold
    
    async def initialize(self, ocr_engine=None):
        """Initialize screen analyzer"""
        try:
            self.logger.info("Initializing Screen Analyzer...")
            
            # Set OCR engine reference
            self.ocr_engine = ocr_engine
            
            # Initialize screen capture
            await self._initialize_screen_capture()
            
            # Initialize element detectors
            await self._initialize_element_detectors()
            
            # Test screen capture
            test_screenshot = await self.capture_screen()
            if test_screenshot is None:
                raise Exception("Screen capture test failed")
            
            self.logger.info("Screen Analyzer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Screen Analyzer initialization error: {e}")
            return False
    
    async def _initialize_screen_capture(self):
        """Initialize screen capture method"""
        try:
            # Determine best capture method
            if WIN32_AVAILABLE and self.capture_config['method'] in ['auto', 'win32']:
                self.capture_method = 'win32'
                self.logger.info("Using Win32 screen capture")
            elif MSS_AVAILABLE and self.capture_config['method'] in ['auto', 'mss']:
                self.capture_method = 'mss'
                self.mss_instance = mss.mss()
                self.logger.info("Using MSS screen capture")
            else:
                raise Exception("No screen capture method available")
            
        except Exception as e:
            self.logger.error(f"Screen capture initialization error: {e}")
            raise
    
    async def _initialize_element_detectors(self):
        """Initialize element detection models"""
        try:
            # Initialize basic element detectors
            self.element_detectors = {
                'button': self._detect_buttons,
                'textbox': self._detect_textboxes,
                'dropdown': self._detect_dropdowns,
                'checkbox': self._detect_checkboxes,
                'menu': self._detect_menus,
                'link': self._detect_links,
                'image': self._detect_images,
                'icon': self._detect_icons
            }
            
            self.logger.info("Element detectors initialized")
            
        except Exception as e:
            self.logger.error(f"Element detector initialization error: {e}")
    
    # Screen Capture
    async def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """Capture screenshot of screen or region"""
        try:
            if self.capture_method == 'win32':
                return await self._capture_win32(region)
            elif self.capture_method == 'mss':
                return await self._capture_mss(region)
            else:
                raise Exception("No capture method available")
                
        except Exception as e:
            self.logger.error(f"Screen capture error: {e}")
            return None
    
    async def _capture_win32(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """Capture screen using Win32 API"""
        try:
            # Get screen dimensions
            if region:
                x, y, width, height = region
            else:
                x, y = 0, 0
                width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
                height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            
            # Create device contexts
            hdesktop = win32gui.GetDesktopWindow()
            hwndDC = win32gui.GetWindowDC(hdesktop)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy screen to bitmap
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (x, y), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (height, width, 4)  # BGRA format
            
            # Convert BGRA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Cleanup
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hdesktop, hwndDC)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Win32 capture error: {e}")
            return None
    
    async def _capture_mss(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """Capture screen using MSS"""
        try:
            if region:
                x, y, width, height = region
                monitor = {"top": y, "left": x, "width": width, "height": height}
            else:
                monitor = self.mss_instance.monitors[1]  # Primary monitor
            
            # Capture screenshot
            screenshot = self.mss_instance.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert BGRA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            return img
            
        except Exception as e:
            self.logger.error(f"MSS capture error: {e}")
            return None
    
    # Screen Analysis
    async def analyze_screen(self, screenshot: Optional[np.ndarray] = None,
                           region: Optional[Tuple[int, int, int, int]] = None) -> ScreenAnalysis:
        """Perform comprehensive screen analysis"""
        try:
            start_time = time.time()
            
            # Capture screenshot if not provided
            if screenshot is None:
                screenshot = await self.capture_screen(region)
                if screenshot is None:
                    raise Exception("Failed to capture screenshot")
            
            # Generate cache key
            cache_key = self._generate_analysis_cache_key(screenshot)
            
            # Check cache
            if cache_key in self.analysis_cache:
                self.logger.debug("Using cached screen analysis")
                return self.analysis_cache[cache_key]
            
            # Initialize analysis result
            analysis = ScreenAnalysis(
                screenshot=screenshot,
                timestamp=datetime.now(),
                regions=[],
                elements=[],
                layout_structure={},
                text_content="",
                confidence=0.0,
                processing_time=0.0
            )
            
            # Perform analysis tasks
            tasks = []
            
            if self.analysis_config['detect_text']:
                tasks.append(self._analyze_text_regions(screenshot))
            
            if self.analysis_config['detect_elements']:
                tasks.append(self._detect_interactive_elements(screenshot))
            
            if self.analysis_config['analyze_layout']:
                tasks.append(self._analyze_layout_structure(screenshot))
            
            # Execute analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            text_regions = []
            elements = []
            layout_structure = {}
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Analysis task {i} failed: {result}")
                    continue
                
                if i == 0 and self.analysis_config['detect_text']:  # Text regions
                    text_regions = result
                elif i == 1 and self.analysis_config['detect_elements']:  # Elements
                    elements = result
                elif i == 2 and self.analysis_config['analyze_layout']:  # Layout
                    layout_structure = result
            
            # Combine results
            analysis.regions = text_regions
            analysis.elements = elements
            analysis.layout_structure = layout_structure
            
            # Extract text content
            analysis.text_content = self._extract_text_content(text_regions)
            
            # Calculate overall confidence
            confidences = []
            for region in text_regions:
                confidences.append(region.confidence)
            for element in elements:
                if 'confidence' in element.properties:
                    confidences.append(element.properties['confidence'])
            
            analysis.confidence = np.mean(confidences) if confidences else 0.0
            
            # Calculate processing time
            analysis.processing_time = time.time() - start_time
            
            # Update statistics
            self.analysis_stats['screens_analyzed'] += 1
            self.analysis_stats['elements_detected'] += len(elements)
            self.analysis_stats['text_regions_found'] += len(text_regions)
            self.analysis_stats['total_processing_time'] += analysis.processing_time
            
            # Update average confidence
            total_analyses = self.analysis_stats['screens_analyzed']
            current_avg = self.analysis_stats['average_confidence']
            self.analysis_stats['average_confidence'] = (
                (current_avg * (total_analyses - 1) + analysis.confidence) / total_analyses
            )
            
            # Cache result
            self._cache_analysis(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Screen analysis error: {e}")
            return ScreenAnalysis(
                screenshot=screenshot or np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=datetime.now(),
                regions=[],
                elements=[],
                layout_structure={},
                text_content="",
                confidence=0.0,
                processing_time=0.0
            )
    
    async def _analyze_text_regions(self, screenshot: np.ndarray) -> List[ScreenRegion]:
        """Analyze text regions in screenshot"""
        try:
            regions = []
            
            if self.ocr_engine:
                # Use OCR engine for text detection
                ocr_result = await self.ocr_engine.extract_text(screenshot)
                
                # Convert OCR results to screen regions
                for word in ocr_result.words:
                    region = ScreenRegion(
                        x=word['bbox'][0],
                        y=word['bbox'][1],
                        width=word['bbox'][2],
                        height=word['bbox'][3],
                        content_type='text',
                        text=word['text'],
                        confidence=word['confidence'],
                        properties={
                            'font_size': self._estimate_font_size(word['bbox']),
                            'is_title': self._is_title_text(word['text']),
                            'is_clickable': self._is_clickable_text(word['text'])
                        },
                        children=[]
                    )
                    regions.append(region)
                
                # Group words into lines and paragraphs
                line_regions = self._create_line_regions(ocr_result.lines)
                paragraph_regions = self._create_paragraph_regions(ocr_result.paragraphs)
                
                regions.extend(line_regions)
                regions.extend(paragraph_regions)
            
            else:
                # Fallback: Basic text region detection using image processing
                regions = await self._detect_text_regions_basic(screenshot)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Text region analysis error: {e}")
            return []
    
    async def _detect_text_regions_basic(self, screenshot: np.ndarray) -> List[ScreenRegion]:
        """Basic text region detection using image processing"""
        try:
            regions = []
            
            if not OPENCV_AVAILABLE:
                return regions
            
            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # Apply morphological operations to find text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Gradient
            grad_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
            gradient = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
            
            # Threshold
            _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours and create regions
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w > 10 and h > 5 and w < screenshot.shape[1] * 0.8:
                    region = ScreenRegion(
                        x=x, y=y, width=w, height=h,
                        content_type='text',
                        text='[Text detected]',
                        confidence=0.5,
                        properties={'method': 'basic_detection'},
                        children=[]
                    )
                    regions.append(region)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Basic text detection error: {e}")
            return []
    
    async def _detect_interactive_elements(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect interactive elements in screenshot"""
        try:
            elements = []
            
            # Run element detectors
            for element_type, detector in self.element_detectors.items():
                if element_type in self.analysis_config['element_types']:
                    detected_elements = await detector(screenshot)
                    elements.extend(detected_elements)
            
            # Remove duplicates and overlapping elements
            elements = self._remove_duplicate_elements(elements)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Interactive element detection error: {e}")
            return []
    
    async def _detect_buttons(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect button elements"""
        try:
            buttons = []
            
            if not OPENCV_AVAILABLE:
                return buttons
            
            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # Detect rectangular shapes (potential buttons)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by size and aspect ratio
                    if (20 <= w <= 200 and 15 <= h <= 50 and 
                        0.2 <= h/w <= 2.0):
                        
                        # Extract region for text analysis
                        button_region = screenshot[y:y+h, x:x+w]
                        
                        # Simple button text detection
                        button_text = await self._extract_button_text(button_region)
                        
                        button = ScreenElement(
                            element_id=f"button_{len(buttons)}",
                            element_type='button',
                            bbox=(x, y, w, h),
                            text=button_text,
                            properties={
                                'confidence': 0.7,
                                'style': 'rectangular'
                            },
                            clickable=True,
                            focusable=True,
                            visible=True,
                            enabled=True
                        )
                        buttons.append(button)
            
            return buttons
            
        except Exception as e:
            self.logger.error(f"Button detection error: {e}")
            return []
    
    async def _detect_textboxes(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect textbox elements"""
        try:
            textboxes = []
            
            if not OPENCV_AVAILABLE:
                return textboxes
            
            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # Detect rectangular shapes with white/light background
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio (textboxes are usually wider than tall)
                if (50 <= w <= 400 and 15 <= h <= 40 and w/h >= 2.0):
                    
                    # Check if it has a border (common for textboxes)
                    border_detected = self._has_border(gray[y:y+h, x:x+w])
                    
                    if border_detected:
                        textbox = ScreenElement(
                            element_id=f"textbox_{len(textboxes)}",
                            element_type='textbox',
                            bbox=(x, y, w, h),
                            text='',  # Textboxes usually don't have visible text
                            properties={
                                'confidence': 0.6,
                                'has_border': True
                            },
                            clickable=True,
                            focusable=True,
                            visible=True,
                            enabled=True
                        )
                        textboxes.append(textbox)
            
            return textboxes
            
        except Exception as e:
            self.logger.error(f"Textbox detection error: {e}")
            return []
    
    async def _detect_dropdowns(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect dropdown elements"""
        try:
            dropdowns = []
            
            # Look for dropdown arrow patterns
            # This is a simplified implementation
            
            return dropdowns
            
        except Exception as e:
            self.logger.error(f"Dropdown detection error: {e}")
            return []
    
    async def _detect_checkboxes(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect checkbox elements"""
        try:
            checkboxes = []
            
            if not OPENCV_AVAILABLE:
                return checkboxes
            
            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # Look for small square shapes
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (checkboxes are small and square)
                if (10 <= w <= 25 and 10 <= h <= 25 and 
                    0.8 <= w/h <= 1.2):  # Nearly square
                    
                    checkbox = ScreenElement(
                        element_id=f"checkbox_{len(checkboxes)}",
                        element_type='checkbox',
                        bbox=(x, y, w, h),
                        text='',
                        properties={
                            'confidence': 0.5,
                            'checked': self._is_checkbox_checked(gray[y:y+h, x:x+w])
                        },
                        clickable=True,
                        focusable=True,
                        visible=True,
                        enabled=True
                    )
                    checkboxes.append(checkbox)
            
            return checkboxes
            
        except Exception as e:
            self.logger.error(f"Checkbox detection error: {e}")
            return []
    
    async def _detect_menus(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect menu elements"""
        try:
            menus = []
            
            # Look for menu patterns (horizontal or vertical lists)
            # This is a simplified implementation
            
            return menus
            
        except Exception as e:
            self.logger.error(f"Menu detection error: {e}")
            return []
    
    async def _detect_links(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect link elements"""
        try:
            links = []
            
            # Look for underlined text or blue text (common link indicators)
            # This would require color analysis and text detection
            
            return links
            
        except Exception as e:
            self.logger.error(f"Link detection error: {e}")
            return []
    
    async def _detect_images(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect image elements"""
        try:
            images = []
            
            # Look for rectangular regions with complex content
            # This is a simplified implementation
            
            return images
            
        except Exception as e:
            self.logger.error(f"Image detection error: {e}")
            return []
    
    async def _detect_icons(self, screenshot: np.ndarray) -> List[ScreenElement]:
        """Detect icon elements"""
        try:
            icons = []
            
            # Look for small square/circular regions with distinct content
            # This is a simplified implementation
            
            return icons
            
        except Exception as e:
            self.logger.error(f"Icon detection error: {e}")
            return []
    
    async def _analyze_layout_structure(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyze overall layout structure"""
        try:
            structure = {
                'width': screenshot.shape[1],
                'height': screenshot.shape[0],
                'regions': [],
                'hierarchy': {},
                'reading_order': []
            }
            
            # Basic layout analysis
            # Divide screen into regions and analyze content density
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Layout structure analysis error: {e}")
            return {}
    
    # Helper Methods
    def _estimate_font_size(self, bbox: Tuple[int, int, int, int]) -> int:
        """Estimate font size from bounding box"""
        return max(8, min(72, bbox[3]))  # Height as rough font size estimate
    
    def _is_title_text(self, text: str) -> bool:
        """Check if text appears to be a title"""
        return (len(text) > 3 and 
                (text.isupper() or text.istitle()) and
                len(text.split()) <= 10)
    
    def _is_clickable_text(self, text: str) -> bool:
        """Check if text appears to be clickable"""
        clickable_keywords = ['click', 'button', 'link', 'menu', 'ok', 'cancel', 'submit', 'save']
        return any(keyword in text.lower() for keyword in clickable_keywords)
    
    def _create_line_regions(self, lines: List[Dict[str, Any]]) -> List[ScreenRegion]:
        """Create screen regions from OCR lines"""
        regions = []
        
        for i, line in enumerate(lines):
            region = ScreenRegion(
                x=line['bbox'][0],
                y=line['bbox'][1],
                width=line['bbox'][2],
                height=line['bbox'][3],
                content_type='text_line',
                text=line['text'],
                confidence=np.mean([w.get('confidence', 0) for w in line.get('words', [])]),
                properties={'line_index': i},
                children=[]
            )
            regions.append(region)
        
        return regions
    
    def _create_paragraph_regions(self, paragraphs: List[Dict[str, Any]]) -> List[ScreenRegion]:
        """Create screen regions from OCR paragraphs"""
        regions = []
        
        for i, paragraph in enumerate(paragraphs):
            region = ScreenRegion(
                x=paragraph['bbox'][0],
                y=paragraph['bbox'][1],
                width=paragraph['bbox'][2],
                height=paragraph['bbox'][3],
                content_type='text_paragraph',
                text=paragraph['text'],
                confidence=np.mean([w.get('confidence', 0) for w in paragraph.get('words', [])]),
                properties={'paragraph_index': i},
                children=[]
            )
            regions.append(region)
        
        return regions
    
    def _extract_text_content(self, regions: List[ScreenRegion]) -> str:
        """Extract combined text content from regions"""
        text_parts = []
        
        for region in regions:
            if region.content_type in ['text', 'text_line', 'text_paragraph'] and region.text.strip():
                text_parts.append(region.text.strip())
        
        return '\n'.join(text_parts)
    
    async def _extract_button_text(self, button_region: np.ndarray) -> str:
        """Extract text from button region"""
        try:
            if self.ocr_engine:
                result = await self.ocr_engine.extract_text(button_region)
                return result.text.strip()
            else:
                return '[Button]'
        except:
            return '[Button]'
    
    def _has_border(self, region: np.ndarray) -> bool:
        """Check if region has a border"""
        try:
            if not OPENCV_AVAILABLE:
                return False
            
            # Check edges for consistent lines
            edges = cv2.Canny(region, 50, 150)
            
            # Check top and bottom edges
            top_edge = np.sum(edges[0, :]) > region.shape[1] * 0.3
            bottom_edge = np.sum(edges[-1, :]) > region.shape[1] * 0.3
            
            # Check left and right edges
            left_edge = np.sum(edges[:, 0]) > region.shape[0] * 0.3
            right_edge = np.sum(edges[:, -1]) > region.shape[0] * 0.3
            
            return sum([top_edge, bottom_edge, left_edge, right_edge]) >= 3
            
        except:
            return False
    
    def _is_checkbox_checked(self, checkbox_region: np.ndarray) -> bool:
        """Check if checkbox is checked"""
        try:
            # Look for checkmark or X pattern
            # Simple implementation: check for dark pixels in center
            center_y, center_x = checkbox_region.shape[0] // 2, checkbox_region.shape[1] // 2
            center_region = checkbox_region[
                center_y-2:center_y+3,
                center_x-2:center_x+3
            ]
            
            # If center region is darker than average, likely checked
            return np.mean(center_region) < np.mean(checkbox_region) * 0.7
            
        except:
            return False
    
    def _remove_duplicate_elements(self, elements: List[ScreenElement]) -> List[ScreenElement]:
        """Remove duplicate and overlapping elements"""
        try:
            unique_elements = []
            
            for element in elements:
                is_duplicate = False
                
                for existing in unique_elements:
                    # Check for overlap
                    overlap = self._calculate_overlap(element.bbox, existing.bbox)
                    if overlap > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_elements.append(element)
            
            return unique_elements
            
        except Exception as e:
            self.logger.error(f"Remove duplicates error: {e}")
            return elements
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        try:
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            # Calculate intersection
            left = max(x1, x2)
            top = max(y1, y2)
            right = min(x1 + w1, x2 + w2)
            bottom = min(y1 + h1, y2 + h2)
            
            if left < right and top < bottom:
                intersection = (right - left) * (bottom - top)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _generate_analysis_cache_key(self, screenshot: np.ndarray) -> str:
        """Generate cache key for screen analysis"""
        try:
            # Create hash from screenshot data (downsampled for efficiency)
            small_screenshot = cv2.resize(screenshot, (64, 64)) if OPENCV_AVAILABLE else screenshot[::8, ::8]
            screenshot_hash = hash(small_screenshot.tobytes())
            return f"analysis_{screenshot_hash}"
        except:
            return f"analysis_unknown_{time.time()}"
    
    def _cache_analysis(self, key: str, analysis: ScreenAnalysis):
        """Cache screen analysis result"""
        try:
            # Manage cache size
            if len(self.analysis_cache) >= self.cache_max_size:
                # Remove oldest entries
                keys_to_remove = list(self.analysis_cache.keys())[:self.cache_max_size // 2]
                for k in keys_to_remove:
                    del self.analysis_cache[k]
            
            # Don't cache the screenshot to save memory
            cached_analysis = ScreenAnalysis(
                screenshot=None,  # Don't cache screenshot
                timestamp=analysis.timestamp,
                regions=analysis.regions,
                elements=analysis.elements,
                layout_structure=analysis.layout_structure,
                text_content=analysis.text_content,
                confidence=analysis.confidence,
                processing_time=analysis.processing_time
            )
            
            self.analysis_cache[key] = cached_analysis
            
        except Exception as e:
            self.logger.debug(f"Cache analysis error: {e}")
    
    # Screen Monitoring
    async def start_monitoring(self, callback: Callable[[ScreenAnalysis], None], 
                             interval: float = 1.0):
        """Start monitoring screen changes"""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_callbacks.append(callback)
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop(interval))
            
            self.logger.info("Screen monitoring started")
            
        except Exception as e:
            self.logger.error(f"Start monitoring error: {e}")
    
    async def stop_monitoring(self):
        """Stop screen monitoring"""
        try:
            self.monitoring_active = False
            self.monitoring_callbacks.clear()
            
            self.logger.info("Screen monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Stop monitoring error: {e}")
    
    async def _monitoring_loop(self, interval: float):
        """Screen monitoring loop"""
        try:
            while self.monitoring_active:
                # Capture current screen
                current_screenshot = await self.capture_screen()
                
                if current_screenshot is not None:
                    # Check for significant changes
                    if self._has_significant_change(current_screenshot):
                        # Analyze screen
                        analysis = await self.analyze_screen(current_screenshot)
                        
                        # Notify callbacks
                        for callback in self.monitoring_callbacks:
                            try:
                                callback(analysis)
                            except Exception as e:
                                self.logger.error(f"Monitoring callback error: {e}")
                        
                        # Update last screenshot
                        self.last_screenshot = current_screenshot
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
    
    def _has_significant_change(self, current_screenshot: np.ndarray) -> bool:
        """Check if screen has changed significantly"""
        try:
            if self.last_screenshot is None:
                return True
            
            if not OPENCV_AVAILABLE:
                return True  # Assume change if can't compare
            
            # Resize for comparison
            current_small = cv2.resize(current_screenshot, (64, 64))
            last_small = cv2.resize(self.last_screenshot, (64, 64))
            
            # Calculate difference
            diff = cv2.absdiff(current_small, last_small)
            change_ratio = np.sum(diff) / (64 * 64 * 3 * 255)
            
            return change_ratio > self.change_threshold
            
        except:
            return True  # Assume change if comparison fails
    
    # Configuration and Management
    async def set_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update screen analyzer configuration"""
        try:
            for key, value in config.items():
                if key in self.capture_config:
                    self.capture_config[key] = value
                elif key in self.analysis_config:
                    self.analysis_config[key] = value
            
            return {
                'success': True,
                'capture_config': self.capture_config.copy(),
                'analysis_config': self.analysis_config.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Set config error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get screen analyzer performance statistics"""
        return {
            'analysis_stats': self.analysis_stats.copy(),
            'capture_stats': {
                'method': self.capture_method,
                'available_methods': []
            },
            'cache_stats': {
                'cache_size': len(self.analysis_cache),
                'cache_max_size': self.cache_max_size
            },
            'monitoring_stats': {
                'active': self.monitoring_active,
                'callbacks': len(self.monitoring_callbacks)
            },
            'config': {
                'capture_config': self.capture_config.copy(),
                'analysis_config': self.analysis_config.copy()
            }
        }
    
    async def cleanup(self):
        """Cleanup screen analyzer"""
        try:
            self.logger.info("Cleaning up Screen Analyzer...")
            
            # Stop monitoring
            await self.stop_monitoring()
            
            # Clear cache
            self.analysis_cache.clear()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Cleanup MSS instance
            if hasattr(self, 'mss_instance'):
                self.mss_instance.close()
            
            self.logger.info("Screen Analyzer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Screen Analyzer cleanup error: {e}")
"""
Element Detector
Advanced UI element detection and interaction mapping.
Provides precise identification of interactive elements and their properties.
"""

import asyncio
import logging
import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import re

# PIL for image processing
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OpenCV for image processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Machine learning libraries
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ElementFeatures:
    """Element visual features"""
    color_histogram: np.ndarray
    texture_features: np.ndarray
    shape_features: Dict[str, float]
    edge_density: float
    corner_count: int
    symmetry_score: float


@dataclass
class InteractionHint:
    """Interaction hint for element"""
    action_type: str  # click, double_click, right_click, drag, type, etc.
    confidence: float
    description: str
    keyboard_shortcut: Optional[str]
    accessibility_info: Dict[str, Any]


@dataclass
class DetectedElement:
    """Detected UI element with full information"""
    element_id: str
    element_type: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    text: str
    confidence: float
    features: ElementFeatures
    interaction_hints: List[InteractionHint]
    properties: Dict[str, Any]
    parent_element: Optional[str]
    child_elements: List[str]
    z_index: int
    visible: bool
    enabled: bool
    focused: bool


class ElementDetector:
    """Advanced UI element detector"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Detection configuration
        self.config = {
            'detection_methods': ['template', 'feature', 'ml', 'heuristic'],
            'element_types': [
                'button', 'textbox', 'dropdown', 'checkbox', 'radio',
                'slider', 'tab', 'menu', 'menuitem', 'toolbar',
                'statusbar', 'scrollbar', 'progressbar', 'label',
                'link', 'image', 'icon', 'window', 'dialog',
                'listbox', 'treeview', 'table', 'cell'
            ],
            'confidence_threshold': 0.6,
            'overlap_threshold': 0.3,
            'feature_extraction': True,
            'interaction_analysis': True,
            'accessibility_analysis': True
        }
        
        # Element templates and patterns
        self.element_templates = {}
        self.element_patterns = {}
        
        # Feature extractors
        self.feature_extractors = {}
        
        # ML models for element classification
        self.ml_models = {}
        
        # Performance tracking
        self.detection_stats = {
            'elements_detected': 0,
            'detection_time': 0.0,
            'accuracy_score': 0.0,
            'method_usage': {}
        }
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache for element detection
        self.detection_cache = {}
        self.cache_max_size = 100
    
    async def initialize(self, ocr_engine=None):
        """Initialize element detector"""
        try:
            self.logger.info("Initializing Element Detector...")
            
            # Set OCR engine reference
            self.ocr_engine = ocr_engine
            
            # Initialize element templates
            await self._initialize_templates()
            
            # Initialize feature extractors
            await self._initialize_feature_extractors()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Initialize element patterns
            await self._initialize_patterns()
            
            self.logger.info("Element Detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Element Detector initialization error: {e}")
            return False
    
    async def _initialize_templates(self):
        """Initialize element templates"""
        try:
            # Create basic templates for common UI elements
            self.element_templates = {
                'button': await self._create_button_templates(),
                'textbox': await self._create_textbox_templates(),
                'checkbox': await self._create_checkbox_templates(),
                'dropdown': await self._create_dropdown_templates(),
                'slider': await self._create_slider_templates(),
                'scrollbar': await self._create_scrollbar_templates()
            }
            
            self.logger.info("Element templates initialized")
            
        except Exception as e:
            self.logger.error(f"Template initialization error: {e}")
    
    async def _create_button_templates(self) -> List[Dict[str, Any]]:
        """Create button templates"""
        templates = []
        
        # Standard button template
        templates.append({
            'name': 'standard_button',
            'min_width': 50,
            'max_width': 200,
            'min_height': 20,
            'max_height': 40,
            'aspect_ratio_range': (1.5, 8.0),
            'has_border': True,
            'has_text': True,
            'background_type': 'solid'
        })
        
        # Icon button template
        templates.append({
            'name': 'icon_button',
            'min_width': 16,
            'max_width': 48,
            'min_height': 16,
            'max_height': 48,
            'aspect_ratio_range': (0.8, 1.2),
            'has_border': False,
            'has_text': False,
            'background_type': 'transparent'
        })
        
        return templates
    
    async def _create_textbox_templates(self) -> List[Dict[str, Any]]:
        """Create textbox templates"""
        templates = []
        
        # Single-line textbox
        templates.append({
            'name': 'single_line_textbox',
            'min_width': 80,
            'max_width': 500,
            'min_height': 18,
            'max_height': 30,
            'aspect_ratio_range': (3.0, 20.0),
            'has_border': True,
            'background_color': 'white',
            'cursor_visible': True
        })
        
        # Multi-line textbox
        templates.append({
            'name': 'multi_line_textbox',
            'min_width': 100,
            'max_width': 600,
            'min_height': 60,
            'max_height': 400,
            'aspect_ratio_range': (1.2, 8.0),
            'has_border': True,
            'has_scrollbar': True,
            'background_color': 'white'
        })
        
        return templates
    
    async def _create_checkbox_templates(self) -> List[Dict[str, Any]]:
        """Create checkbox templates"""
        templates = []
        
        # Standard checkbox
        templates.append({
            'name': 'standard_checkbox',
            'min_width': 12,
            'max_width': 20,
            'min_height': 12,
            'max_height': 20,
            'aspect_ratio_range': (0.8, 1.2),
            'has_border': True,
            'shape': 'square',
            'has_checkmark': None  # Can be True, False, or None (indeterminate)
        })
        
        return templates
    
    async def _create_dropdown_templates(self) -> List[Dict[str, Any]]:
        """Create dropdown templates"""
        templates = []
        
        # Standard dropdown
        templates.append({
            'name': 'standard_dropdown',
            'min_width': 80,
            'max_width': 300,
            'min_height': 18,
            'max_height': 30,
            'aspect_ratio_range': (3.0, 15.0),
            'has_border': True,
            'has_arrow': True,
            'arrow_position': 'right'
        })
        
        return templates
    
    async def _create_slider_templates(self) -> List[Dict[str, Any]]:
        """Create slider templates"""
        templates = []
        
        # Horizontal slider
        templates.append({
            'name': 'horizontal_slider',
            'min_width': 80,
            'max_width': 400,
            'min_height': 15,
            'max_height': 30,
            'orientation': 'horizontal',
            'has_track': True,
            'has_thumb': True
        })
        
        # Vertical slider
        templates.append({
            'name': 'vertical_slider',
            'min_width': 15,
            'max_width': 30,
            'min_height': 80,
            'max_height': 400,
            'orientation': 'vertical',
            'has_track': True,
            'has_thumb': True
        })
        
        return templates
    
    async def _create_scrollbar_templates(self) -> List[Dict[str, Any]]:
        """Create scrollbar templates"""
        templates = []
        
        # Vertical scrollbar
        templates.append({
            'name': 'vertical_scrollbar',
            'min_width': 12,
            'max_width': 25,
            'min_height': 50,
            'max_height': 1000,
            'orientation': 'vertical',
            'has_arrows': True,
            'has_thumb': True
        })
        
        # Horizontal scrollbar
        templates.append({
            'name': 'horizontal_scrollbar',
            'min_width': 50,
            'max_width': 1000,
            'min_height': 12,
            'max_height': 25,
            'orientation': 'horizontal',
            'has_arrows': True,
            'has_thumb': True
        })
        
        return templates
    
    async def _initialize_feature_extractors(self):
        """Initialize feature extractors"""
        try:
            self.feature_extractors = {
                'color': self._extract_color_features,
                'texture': self._extract_texture_features,
                'shape': self._extract_shape_features,
                'edge': self._extract_edge_features,
                'corner': self._extract_corner_features,
                'symmetry': self._extract_symmetry_features
            }
            
            self.logger.info("Feature extractors initialized")
            
        except Exception as e:
            self.logger.error(f"Feature extractor initialization error: {e}")
    
    async def _initialize_ml_models(self):
        """Initialize ML models for element classification"""
        try:
            # Placeholder for ML model initialization
            # In a full implementation, you would load pre-trained models here
            self.ml_models = {
                'element_classifier': None,  # Would be a trained classifier
                'interaction_predictor': None,  # Would predict interaction types
                'accessibility_analyzer': None  # Would analyze accessibility features
            }
            
            self.logger.info("ML models initialized")
            
        except Exception as e:
            self.logger.error(f"ML model initialization error: {e}")
    
    async def _initialize_patterns(self):
        """Initialize element detection patterns"""
        try:
            self.element_patterns = {
                'button_text_patterns': [
                    r'\b(OK|Cancel|Apply|Save|Delete|Edit|Add|Remove|Submit|Reset|Close)\b',
                    r'\b(Yes|No|Accept|Decline|Agree|Disagree)\b',
                    r'\b(Next|Previous|Back|Forward|Continue|Finish)\b',
                    r'\b(Login|Logout|Sign In|Sign Out|Register)\b'
                ],
                'textbox_placeholder_patterns': [
                    r'Enter\s+\w+',
                    r'Type\s+\w+',
                    r'Search\s*\.{3}',
                    r'\w+@\w+\.\w+',  # Email pattern
                    r'Password',
                    r'Username'
                ],
                'menu_patterns': [
                    r'File|Edit|View|Tools|Help',
                    r'Options|Settings|Preferences',
                    r'Window|Format|Insert'
                ],
                'link_patterns': [
                    r'https?://\S+',
                    r'www\.\S+',
                    r'Click\s+here',
                    r'Learn\s+more',
                    r'Read\s+more'
                ]
            }
            
            self.logger.info("Element patterns initialized")
            
        except Exception as e:
            self.logger.error(f"Pattern initialization error: {e}")
    
    # Main Detection Methods
    async def detect_elements(self, screenshot: np.ndarray, 
                            region: Optional[Tuple[int, int, int, int]] = None) -> List[DetectedElement]:
        """Detect all UI elements in screenshot"""
        try:
            start_time = time.time()
            
            # Apply region filter if specified
            if region:
                x, y, w, h = region
                screenshot = screenshot[y:y+h, x:x+w]
                offset = (x, y)
            else:
                offset = (0, 0)
            
            # Generate cache key
            cache_key = self._generate_cache_key(screenshot)
            
            # Check cache
            if cache_key in self.detection_cache:
                self.logger.debug("Using cached element detection")
                return self.detection_cache[cache_key]
            
            # Run detection methods
            all_elements = []
            
            for method in self.config['detection_methods']:
                try:
                    if method == 'template':
                        elements = await self._detect_by_template(screenshot)
                    elif method == 'feature':
                        elements = await self._detect_by_features(screenshot)
                    elif method == 'ml':
                        elements = await self._detect_by_ml(screenshot)
                    elif method == 'heuristic':
                        elements = await self._detect_by_heuristics(screenshot)
                    else:
                        continue
                    
                    # Adjust coordinates for region offset
                    if offset != (0, 0):
                        elements = self._adjust_element_coordinates(elements, offset)
                    
                    all_elements.extend(elements)
                    
                    # Update method usage stats
                    if method not in self.detection_stats['method_usage']:
                        self.detection_stats['method_usage'][method] = 0
                    self.detection_stats['method_usage'][method] += len(elements)
                    
                except Exception as e:
                    self.logger.warning(f"Detection method {method} failed: {e}")
            
            # Remove duplicates and merge overlapping elements
            unique_elements = await self._merge_duplicate_elements(all_elements)
            
            # Extract features for each element
            if self.config['feature_extraction']:
                for element in unique_elements:
                    element.features = await self._extract_element_features(screenshot, element)
            
            # Analyze interactions
            if self.config['interaction_analysis']:
                for element in unique_elements:
                    element.interaction_hints = await self._analyze_element_interactions(element)
            
            # Analyze accessibility
            if self.config['accessibility_analysis']:
                for element in unique_elements:
                    await self._analyze_element_accessibility(element)
            
            # Update statistics
            detection_time = time.time() - start_time
            self.detection_stats['elements_detected'] += len(unique_elements)
            self.detection_stats['detection_time'] += detection_time
            
            # Cache result
            self._cache_detection(cache_key, unique_elements)
            
            return unique_elements
            
        except Exception as e:
            self.logger.error(f"Element detection error: {e}")
            return []
    
    async def _detect_by_template(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements using template matching"""
        try:
            elements = []
            
            for element_type, templates in self.element_templates.items():
                for template in templates:
                    detected = await self._match_template(screenshot, element_type, template)
                    elements.extend(detected)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Template detection error: {e}")
            return []
    
    async def _match_template(self, screenshot: np.ndarray, element_type: str, 
                            template: Dict[str, Any]) -> List[DetectedElement]:
        """Match specific template in screenshot"""
        try:
            elements = []
            
            if not OPENCV_AVAILABLE:
                return elements
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # Find contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if dimensions match template
                if (template['min_width'] <= w <= template['max_width'] and
                    template['min_height'] <= h <= template['max_height']):
                    
                    # Check aspect ratio
                    aspect_ratio = w / h
                    min_ratio, max_ratio = template['aspect_ratio_range']
                    
                    if min_ratio <= aspect_ratio <= max_ratio:
                        # Additional template-specific checks
                        confidence = await self._calculate_template_confidence(
                            screenshot[y:y+h, x:x+w], element_type, template
                        )
                        
                        if confidence >= self.config['confidence_threshold']:
                            element = DetectedElement(
                                element_id=f"{element_type}_{i}",
                                element_type=element_type,
                                bbox=(x, y, w, h),
                                center=(x + w//2, y + h//2),
                                text="",  # Will be filled later
                                confidence=confidence,
                                features=None,  # Will be filled later
                                interaction_hints=[],
                                properties={
                                    'detection_method': 'template',
                                    'template_name': template['name']
                                },
                                parent_element=None,
                                child_elements=[],
                                z_index=0,
                                visible=True,
                                enabled=True,
                                focused=False
                            )
                            elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Template matching error: {e}")
            return []
    
    async def _calculate_template_confidence(self, region: np.ndarray, element_type: str, 
                                          template: Dict[str, Any]) -> float:
        """Calculate confidence score for template match"""
        try:
            confidence = 0.5  # Base confidence
            
            # Check template-specific features
            if element_type == 'button':
                if template.get('has_border') and self._has_border(region):
                    confidence += 0.2
                if template.get('has_text') and await self._has_text(region):
                    confidence += 0.2
            
            elif element_type == 'textbox':
                if template.get('has_border') and self._has_border(region):
                    confidence += 0.2
                if template.get('background_color') == 'white' and self._has_white_background(region):
                    confidence += 0.2
            
            elif element_type == 'checkbox':
                if template.get('shape') == 'square' and self._is_square_shape(region):
                    confidence += 0.3
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Template confidence calculation error: {e}")
            return 0.0
    
    async def _detect_by_features(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements using feature analysis"""
        try:
            elements = []
            
            if not OPENCV_AVAILABLE:
                return elements
            
            # Find potential element regions
            regions = await self._find_element_regions(screenshot)
            
            for i, region in enumerate(regions):
                x, y, w, h = region
                region_image = screenshot[y:y+h, x:x+w]
                
                # Extract features
                features = await self._extract_element_features(screenshot, None, region)
                
                # Classify element type based on features
                element_type, confidence = await self._classify_by_features(features)
                
                if confidence >= self.config['confidence_threshold']:
                    element = DetectedElement(
                        element_id=f"feature_{i}",
                        element_type=element_type,
                        bbox=region,
                        center=(x + w//2, y + h//2),
                        text="",
                        confidence=confidence,
                        features=features,
                        interaction_hints=[],
                        properties={'detection_method': 'feature'},
                        parent_element=None,
                        child_elements=[],
                        z_index=0,
                        visible=True,
                        enabled=True,
                        focused=False
                    )
                    elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Feature detection error: {e}")
            return []
    
    async def _find_element_regions(self, screenshot: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find potential element regions using image processing"""
        try:
            regions = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            
            # Apply different techniques to find regions
            
            # 1. Edge-based detection
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # Minimum size filter
                    regions.append((x, y, w, h))
            
            # 2. Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 15 and h > 15:
                    regions.append((x, y, w, h))
            
            # Remove duplicates and overlapping regions
            regions = self._remove_overlapping_regions(regions)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Find element regions error: {e}")
            return []
    
    async def _detect_by_ml(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements using machine learning models"""
        try:
            elements = []
            
            # Placeholder for ML-based detection
            # In a full implementation, you would use trained models here
            
            if self.ml_models.get('element_classifier'):
                # Use ML model to detect and classify elements
                pass
            
            return elements
            
        except Exception as e:
            self.logger.error(f"ML detection error: {e}")
            return []
    
    async def _detect_by_heuristics(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements using heuristic rules"""
        try:
            elements = []
            
            # Use OCR results to find text-based elements
            if self.ocr_engine:
                ocr_result = await self.ocr_engine.extract_text(screenshot)
                
                # Analyze text for button patterns
                for word in ocr_result.words:
                    text = word['text'].strip()
                    
                    # Check for button text patterns
                    for pattern in self.element_patterns['button_text_patterns']:
                        if re.search(pattern, text, re.IGNORECASE):
                            element = DetectedElement(
                                element_id=f"heuristic_button_{len(elements)}",
                                element_type='button',
                                bbox=word['bbox'],
                                center=(word['bbox'][0] + word['bbox'][2]//2, 
                                       word['bbox'][1] + word['bbox'][3]//2),
                                text=text,
                                confidence=0.7,
                                features=None,
                                interaction_hints=[],
                                properties={
                                    'detection_method': 'heuristic',
                                    'pattern_matched': pattern
                                },
                                parent_element=None,
                                child_elements=[],
                                z_index=0,
                                visible=True,
                                enabled=True,
                                focused=False
                            )
                            elements.append(element)
                            break
                    
                    # Check for link patterns
                    for pattern in self.element_patterns['link_patterns']:
                        if re.search(pattern, text, re.IGNORECASE):
                            element = DetectedElement(
                                element_id=f"heuristic_link_{len(elements)}",
                                element_type='link',
                                bbox=word['bbox'],
                                center=(word['bbox'][0] + word['bbox'][2]//2, 
                                       word['bbox'][1] + word['bbox'][3]//2),
                                text=text,
                                confidence=0.8,
                                features=None,
                                interaction_hints=[],
                                properties={
                                    'detection_method': 'heuristic',
                                    'pattern_matched': pattern
                                },
                                parent_element=None,
                                child_elements=[],
                                z_index=0,
                                visible=True,
                                enabled=True,
                                focused=False
                            )
                            elements.append(element)
                            break
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Heuristic detection error: {e}")
            return []
    
    # Feature Extraction
    async def _extract_element_features(self, screenshot: np.ndarray, element: Optional[DetectedElement],
                                      region: Optional[Tuple[int, int, int, int]] = None) -> ElementFeatures:
        """Extract visual features from element"""
        try:
            if element:
                x, y, w, h = element.bbox
            elif region:
                x, y, w, h = region
            else:
                return ElementFeatures(
                    color_histogram=np.array([]),
                    texture_features=np.array([]),
                    shape_features={},
                    edge_density=0.0,
                    corner_count=0,
                    symmetry_score=0.0
                )
            
            # Extract region
            element_region = screenshot[y:y+h, x:x+w]
            
            # Extract different types of features
            color_features = await self._extract_color_features(element_region)
            texture_features = await self._extract_texture_features(element_region)
            shape_features = await self._extract_shape_features(element_region)
            edge_density = await self._extract_edge_features(element_region)
            corner_count = await self._extract_corner_features(element_region)
            symmetry_score = await self._extract_symmetry_features(element_region)
            
            return ElementFeatures(
                color_histogram=color_features,
                texture_features=texture_features,
                shape_features=shape_features,
                edge_density=edge_density,
                corner_count=corner_count,
                symmetry_score=symmetry_score
            )
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return ElementFeatures(
                color_histogram=np.array([]),
                texture_features=np.array([]),
                shape_features={},
                edge_density=0.0,
                corner_count=0,
                symmetry_score=0.0
            )
    
    async def _extract_color_features(self, region: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        try:
            if not OPENCV_AVAILABLE:
                return np.array([])
            
            # Calculate color histogram for each channel
            hist_r = cv2.calcHist([region], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([region], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([region], [2], None, [32], [0, 256])
            
            # Normalize and concatenate
            hist_r = hist_r.flatten() / np.sum(hist_r)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_b = hist_b.flatten() / np.sum(hist_b)
            
            return np.concatenate([hist_r, hist_g, hist_b])
            
        except Exception as e:
            self.logger.error(f"Color feature extraction error: {e}")
            return np.array([])
    
    async def _extract_texture_features(self, region: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Patterns"""
        try:
            if not OPENCV_AVAILABLE:
                return np.array([])
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            
            # Simple texture features using gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate texture statistics
            features = [
                np.mean(grad_x),
                np.std(grad_x),
                np.mean(grad_y),
                np.std(grad_y),
                np.mean(np.abs(grad_x)),
                np.mean(np.abs(grad_y))
            ]
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Texture feature extraction error: {e}")
            return np.array([])
    
    async def _extract_shape_features(self, region: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features"""
        try:
            features = {}
            
            if not OPENCV_AVAILABLE:
                return features
            
            # Convert to grayscale and find contours
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate shape features
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    features['circularity'] = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    features['circularity'] = 0.0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['aspect_ratio'] = w / h if h > 0 else 0.0
                features['extent'] = area / (w * h) if w * h > 0 else 0.0
                
                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                features['solidity'] = area / hull_area if hull_area > 0 else 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Shape feature extraction error: {e}")
            return {}
    
    async def _extract_edge_features(self, region: np.ndarray) -> float:
        """Extract edge density feature"""
        try:
            if not OPENCV_AVAILABLE:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            
            return edge_pixels / total_pixels if total_pixels > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Edge feature extraction error: {e}")
            return 0.0
    
    async def _extract_corner_features(self, region: np.ndarray) -> int:
        """Extract corner count feature"""
        try:
            if not OPENCV_AVAILABLE:
                return 0
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            
            # Detect corners using Harris corner detection
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            
            # Count significant corners
            corner_count = np.sum(corners > 0.01 * corners.max())
            
            return int(corner_count)
            
        except Exception as e:
            self.logger.error(f"Corner feature extraction error: {e}")
            return 0
    
    async def _extract_symmetry_features(self, region: np.ndarray) -> float:
        """Extract symmetry score"""
        try:
            if not OPENCV_AVAILABLE:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            
            # Check horizontal symmetry
            height, width = gray.shape
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate similarity
            diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0.0, symmetry_score)
            
        except Exception as e:
            self.logger.error(f"Symmetry feature extraction error: {e}")
            return 0.0
    
    async def _classify_by_features(self, features: ElementFeatures) -> Tuple[str, float]:
        """Classify element type based on features"""
        try:
            # Simple rule-based classification
            # In a full implementation, you would use trained ML models
            
            shape_features = features.shape_features
            
            # Button classification
            if (shape_features.get('aspect_ratio', 0) > 1.5 and
                shape_features.get('circularity', 0) < 0.3 and
                features.edge_density > 0.1):
                return 'button', 0.7
            
            # Textbox classification
            if (shape_features.get('aspect_ratio', 0) > 3.0 and
                shape_features.get('extent', 0) > 0.8 and
                features.edge_density > 0.05):
                return 'textbox', 0.6
            
            # Checkbox classification
            if (0.8 <= shape_features.get('aspect_ratio', 0) <= 1.2 and
                shape_features.get('circularity', 0) < 0.5 and
                features.corner_count >= 4):
                return 'checkbox', 0.6
            
            # Default classification
            return 'unknown', 0.3
            
        except Exception as e:
            self.logger.error(f"Feature classification error: {e}")
            return 'unknown', 0.0
    
    # Interaction Analysis
    async def _analyze_element_interactions(self, element: DetectedElement) -> List[InteractionHint]:
        """Analyze possible interactions for element"""
        try:
            hints = []
            
            # Based on element type
            if element.element_type == 'button':
                hints.append(InteractionHint(
                    action_type='click',
                    confidence=0.9,
                    description='Click to activate button',
                    keyboard_shortcut=None,
                    accessibility_info={'role': 'button', 'actionable': True}
                ))
                
                # Check for specific button types
                if any(keyword in element.text.lower() for keyword in ['ok', 'submit', 'save']):
                    hints.append(InteractionHint(
                        action_type='enter_key',
                        confidence=0.7,
                        description='Press Enter to activate',
                        keyboard_shortcut='Enter',
                        accessibility_info={'default_action': True}
                    ))
            
            elif element.element_type == 'textbox':
                hints.append(InteractionHint(
                    action_type='click',
                    confidence=0.8,
                    description='Click to focus and type text',
                    keyboard_shortcut=None,
                    accessibility_info={'role': 'textbox', 'editable': True}
                ))
                
                hints.append(InteractionHint(
                    action_type='type',
                    confidence=0.9,
                    description='Type text content',
                    keyboard_shortcut=None,
                    accessibility_info={'accepts_text': True}
                ))
            
            elif element.element_type == 'checkbox':
                hints.append(InteractionHint(
                    action_type='click',
                    confidence=0.9,
                    description='Click to toggle checkbox state',
                    keyboard_shortcut='Space',
                    accessibility_info={'role': 'checkbox', 'toggleable': True}
                ))
            
            elif element.element_type == 'dropdown':
                hints.append(InteractionHint(
                    action_type='click',
                    confidence=0.9,
                    description='Click to open dropdown menu',
                    keyboard_shortcut='Alt+Down',
                    accessibility_info={'role': 'combobox', 'expandable': True}
                ))
            
            elif element.element_type == 'link':
                hints.append(InteractionHint(
                    action_type='click',
                    confidence=0.9,
                    description='Click to follow link',
                    keyboard_shortcut=None,
                    accessibility_info={'role': 'link', 'navigational': True}
                ))
            
            return hints
            
        except Exception as e:
            self.logger.error(f"Interaction analysis error: {e}")
            return []
    
    async def _analyze_element_accessibility(self, element: DetectedElement):
        """Analyze accessibility features of element"""
        try:
            # Add accessibility information to element properties
            accessibility_info = {
                'has_text_label': bool(element.text.strip()),
                'keyboard_accessible': element.element_type in ['button', 'textbox', 'checkbox', 'dropdown', 'link'],
                'screen_reader_friendly': True,
                'color_contrast_sufficient': True,  # Would need actual analysis
                'size_appropriate': element.bbox[2] >= 44 and element.bbox[3] >= 44  # Minimum touch target size
            }
            
            element.properties['accessibility'] = accessibility_info
            
        except Exception as e:
            self.logger.error(f"Accessibility analysis error: {e}")
    
    # Helper Methods
    def _has_border(self, region: np.ndarray) -> bool:
        """Check if region has a border"""
        try:
            if not OPENCV_AVAILABLE:
                return False
            
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Check edges of the region
            top_edge = np.sum(edges[0, :]) > region.shape[1] * 0.3
            bottom_edge = np.sum(edges[-1, :]) > region.shape[1] * 0.3
            left_edge = np.sum(edges[:, 0]) > region.shape[0] * 0.3
            right_edge = np.sum(edges[:, -1]) > region.shape[0] * 0.3
            
            return sum([top_edge, bottom_edge, left_edge, right_edge]) >= 3
            
        except:
            return False
    
    async def _has_text(self, region: np.ndarray) -> bool:
        """Check if region contains text"""
        try:
            if self.ocr_engine:
                result = await self.ocr_engine.extract_text(region)
                return len(result.text.strip()) > 0
            return False
        except:
            return False
    
    def _has_white_background(self, region: np.ndarray) -> bool:
        """Check if region has predominantly white background"""
        try:
            # Calculate average color
            avg_color = np.mean(region, axis=(0, 1))
            
            # Check if close to white (RGB values > 200)
            return all(c > 200 for c in avg_color)
            
        except:
            return False
    
    def _is_square_shape(self, region: np.ndarray) -> bool:
        """Check if region is roughly square"""
        try:
            height, width = region.shape[:2]
            aspect_ratio = width / height
            return 0.8 <= aspect_ratio <= 1.2
        except:
            return False
    
    def _remove_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping regions"""
        try:
            unique_regions = []
            
            for region in regions:
                is_overlapping = False
                
                for existing in unique_regions:
                    overlap = self._calculate_region_overlap(region, existing)
                    if overlap > self.config['overlap_threshold']:
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    unique_regions.append(region)
            
            return unique_regions
            
        except Exception as e:
            self.logger.error(f"Remove overlapping regions error: {e}")
            return regions
    
    def _calculate_region_overlap(self, region1: Tuple[int, int, int, int], 
                                region2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two regions"""
        try:
            x1, y1, w1, h1 = region1
            x2, y2, w2, h2 = region2
            
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
    
    async def _merge_duplicate_elements(self, elements: List[DetectedElement]) -> List[DetectedElement]:
        """Merge duplicate and overlapping elements"""
        try:
            unique_elements = []
            
            for element in elements:
                merged = False
                
                for i, existing in enumerate(unique_elements):
                    overlap = self._calculate_region_overlap(element.bbox, existing.bbox)
                    
                    if overlap > self.config['overlap_threshold']:
                        # Merge elements - keep the one with higher confidence
                        if element.confidence > existing.confidence:
                            unique_elements[i] = element
                        merged = True
                        break
                
                if not merged:
                    unique_elements.append(element)
            
            return unique_elements
            
        except Exception as e:
            self.logger.error(f"Merge duplicate elements error: {e}")
            return elements
    
    def _adjust_element_coordinates(self, elements: List[DetectedElement], 
                                  offset: Tuple[int, int]) -> List[DetectedElement]:
        """Adjust element coordinates for region offset"""
        try:
            offset_x, offset_y = offset
            
            for element in elements:
                x, y, w, h = element.bbox
                element.bbox = (x + offset_x, y + offset_y, w, h)
                element.center = (element.center[0] + offset_x, element.center[1] + offset_y)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Adjust coordinates error: {e}")
            return elements
    
    def _generate_cache_key(self, screenshot: np.ndarray) -> str:
        """Generate cache key for detection"""
        try:
            # Create hash from screenshot data (downsampled for efficiency)
            small_screenshot = cv2.resize(screenshot, (32, 32)) if OPENCV_AVAILABLE else screenshot[::16, ::16]
            screenshot_hash = hash(small_screenshot.tobytes())
            return f"detection_{screenshot_hash}"
        except:
            return f"detection_unknown_{time.time()}"
    
    def _cache_detection(self, key: str, elements: List[DetectedElement]):
        """Cache detection result"""
        try:
            # Manage cache size
            if len(self.detection_cache) >= self.cache_max_size:
                # Remove oldest entries
                keys_to_remove = list(self.detection_cache.keys())[:self.cache_max_size // 2]
                for k in keys_to_remove:
                    del self.detection_cache[k]
            
            self.detection_cache[key] = elements
            
        except Exception as e:
            self.logger.debug(f"Cache detection error: {e}")
    
    # Configuration and Management
    async def set_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update element detector configuration"""
        try:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
            
            return {
                'success': True,
                'config': self.config.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Set config error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get element detector performance statistics"""
        return {
            'detection_stats': self.detection_stats.copy(),
            'cache_stats': {
                'cache_size': len(self.detection_cache),
                'cache_max_size': self.cache_max_size
            },
            'config': self.config.copy()
        }
    
    async def cleanup(self):
        """Cleanup element detector"""
        try:
            self.logger.info("Cleaning up Element Detector...")
            
            # Clear cache
            self.detection_cache.clear()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            self.logger.info("Element Detector cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Element Detector cleanup error: {e}")
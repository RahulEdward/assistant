"""
OCR Engine
Advanced Optical Character Recognition with multiple backend support.
Provides high-accuracy text extraction, layout analysis, and visual understanding.
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
import base64
import io

# PIL for image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Tesseract OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# OpenCV for image processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class OCRResult:
    """OCR result structure"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    words: List[Dict[str, Any]]
    lines: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]
    language: str
    processing_time: float


@dataclass
class TextRegion:
    """Text region information"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    font_size: Optional[int]
    font_style: Optional[str]
    text_color: Optional[Tuple[int, int, int]]
    background_color: Optional[Tuple[int, int, int]]
    orientation: float
    line_height: float


@dataclass
class LayoutElement:
    """Layout element information"""
    element_type: str  # text, image, table, line, etc.
    bbox: Tuple[int, int, int, int]
    confidence: float
    properties: Dict[str, Any]
    children: List['LayoutElement']


class OCREngine:
    """Advanced OCR engine with multiple backend support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # OCR backends
        self.backends = {}
        self.active_backend = None
        
        # Configuration
        self.config = {
            'default_language': 'eng',
            'supported_languages': ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'chi_tra', 'jpn', 'kor'],
            'confidence_threshold': 0.5,
            'preprocessing_enabled': True,
            'layout_analysis': True,
            'word_detection': True,
            'line_detection': True,
            'paragraph_detection': True
        }
        
        # Performance tracking
        self.operation_stats = {
            'images_processed': 0,
            'text_extracted': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0,
            'backend_usage': {}
        }
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache for processed images
        self.result_cache = {}
        self.cache_max_size = 100
        
        # Image preprocessing pipeline
        self.preprocessing_steps = [
            'noise_reduction',
            'contrast_enhancement',
            'sharpening',
            'binarization',
            'deskewing'
        ]
    
    async def initialize(self):
        """Initialize OCR engine"""
        try:
            self.logger.info("Initializing OCR Engine...")
            
            # Initialize available backends
            await self._initialize_backends()
            
            # Select best available backend
            await self._select_backend()
            
            self.logger.info(f"OCR Engine initialized with backend: {self.active_backend}")
            return True
            
        except Exception as e:
            self.logger.error(f"OCR Engine initialization error: {e}")
            return False
    
    async def _initialize_backends(self):
        """Initialize available OCR backends"""
        try:
            # Initialize Tesseract
            if TESSERACT_AVAILABLE:
                try:
                    # Test Tesseract
                    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                    pytesseract.image_to_string(test_image)
                    
                    self.backends['tesseract'] = {
                        'engine': pytesseract,
                        'available': True,
                        'languages': self._get_tesseract_languages(),
                        'priority': 3
                    }
                    self.logger.info("Tesseract OCR initialized")
                    
                except Exception as e:
                    self.logger.warning(f"Tesseract initialization failed: {e}")
                    self.backends['tesseract'] = {'available': False}
            
            # Initialize EasyOCR
            if EASYOCR_AVAILABLE:
                try:
                    reader = easyocr.Reader(['en'], gpu=False)
                    
                    self.backends['easyocr'] = {
                        'engine': reader,
                        'available': True,
                        'languages': ['en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
                        'priority': 2
                    }
                    self.logger.info("EasyOCR initialized")
                    
                except Exception as e:
                    self.logger.warning(f"EasyOCR initialization failed: {e}")
                    self.backends['easyocr'] = {'available': False}
            
            # Initialize PaddleOCR
            if PADDLEOCR_AVAILABLE:
                try:
                    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
                    
                    self.backends['paddleocr'] = {
                        'engine': ocr,
                        'available': True,
                        'languages': ['en', 'ch', 'fr', 'german', 'korean', 'japan'],
                        'priority': 1
                    }
                    self.logger.info("PaddleOCR initialized")
                    
                except Exception as e:
                    self.logger.warning(f"PaddleOCR initialization failed: {e}")
                    self.backends['paddleocr'] = {'available': False}
            
            # Fallback: Custom OCR implementation
            self.backends['custom'] = {
                'engine': self,
                'available': True,
                'languages': ['en'],
                'priority': 4
            }
            
        except Exception as e:
            self.logger.error(f"Backend initialization error: {e}")
    
    def _get_tesseract_languages(self) -> List[str]:
        """Get available Tesseract languages"""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang in self.config['supported_languages']]
        except:
            return ['eng']
    
    async def _select_backend(self):
        """Select best available backend"""
        try:
            available_backends = [
                (name, info) for name, info in self.backends.items()
                if info.get('available', False)
            ]
            
            if not available_backends:
                raise Exception("No OCR backends available")
            
            # Sort by priority (lower number = higher priority)
            available_backends.sort(key=lambda x: x[1].get('priority', 999))
            
            self.active_backend = available_backends[0][0]
            self.logger.info(f"Selected OCR backend: {self.active_backend}")
            
        except Exception as e:
            self.logger.error(f"Backend selection error: {e}")
            self.active_backend = 'custom'
    
    # Image Preprocessing
    async def preprocess_image(self, image: Union[np.ndarray, Image.Image, str, bytes], 
                             steps: Optional[List[str]] = None) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert input to numpy array
            if isinstance(image, str):
                # File path
                if PIL_AVAILABLE:
                    pil_image = Image.open(image)
                    image_array = np.array(pil_image)
                else:
                    image_array = cv2.imread(image)
            elif isinstance(image, bytes):
                # Bytes data
                if PIL_AVAILABLE:
                    pil_image = Image.open(io.BytesIO(image))
                    image_array = np.array(pil_image)
                else:
                    nparr = np.frombuffer(image, np.uint8)
                    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif PIL_AVAILABLE and isinstance(image, Image.Image):
                # PIL Image
                image_array = np.array(image)
            else:
                # Numpy array
                image_array = image.copy()
            
            # Convert to BGR if needed (OpenCV format)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Assume RGB, convert to BGR for OpenCV
                if not OPENCV_AVAILABLE:
                    # Simple RGB to grayscale conversion
                    image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
                    image_array = image_array.astype(np.uint8)
                else:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Apply preprocessing steps
            if steps is None:
                steps = self.preprocessing_steps if self.config['preprocessing_enabled'] else []
            
            for step in steps:
                if step == 'noise_reduction':
                    image_array = await self._reduce_noise(image_array)
                elif step == 'contrast_enhancement':
                    image_array = await self._enhance_contrast(image_array)
                elif step == 'sharpening':
                    image_array = await self._sharpen_image(image_array)
                elif step == 'binarization':
                    image_array = await self._binarize_image(image_array)
                elif step == 'deskewing':
                    image_array = await self._deskew_image(image_array)
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            # Return original image if preprocessing fails
            if isinstance(image, np.ndarray):
                return image
            else:
                return np.array(image) if PIL_AVAILABLE else np.zeros((100, 100), dtype=np.uint8)
    
    async def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce image noise"""
        try:
            if OPENCV_AVAILABLE:
                # Use OpenCV denoising
                if len(image.shape) == 3:
                    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                else:
                    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            else:
                # Simple noise reduction using median filter
                from scipy import ndimage
                return ndimage.median_filter(image, size=3)
        except:
            return image
    
    async def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        try:
            if OPENCV_AVAILABLE:
                # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
                if len(image.shape) == 3:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    return clahe.apply(image)
            else:
                # Simple contrast enhancement
                return np.clip(image * 1.2, 0, 255).astype(np.uint8)
        except:
            return image
    
    async def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image"""
        try:
            if OPENCV_AVAILABLE:
                # Unsharp masking
                gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
                return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            else:
                # Simple sharpening kernel
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                if len(image.shape) == 3:
                    sharpened = np.zeros_like(image)
                    for i in range(image.shape[2]):
                        sharpened[:,:,i] = cv2.filter2D(image[:,:,i], -1, kernel)
                    return sharpened
                else:
                    return cv2.filter2D(image, -1, kernel)
        except:
            return image
    
    async def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white)"""
        try:
            if OPENCV_AVAILABLE:
                # Convert to grayscale if needed
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # Adaptive thresholding
                return cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                # Simple thresholding
                if len(image.shape) == 3:
                    gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    gray = image
                
                threshold = np.mean(gray)
                return (gray > threshold).astype(np.uint8) * 255
        except:
            return image
    
    async def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew"""
        try:
            if not OPENCV_AVAILABLE:
                return image
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Find edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta - np.pi/2
                    angles.append(angle)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Rotate image to correct skew
                    if abs(avg_angle) > 0.1:  # Only correct if significant skew
                        height, width = gray.shape[:2]
                        center = (width // 2, height // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(avg_angle), 1.0)
                        
                        if len(image.shape) == 3:
                            return cv2.warpAffine(image, rotation_matrix, (width, height), 
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        else:
                            return cv2.warpAffine(image, rotation_matrix, (width, height),
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return image
            
        except:
            return image
    
    # OCR Processing
    async def extract_text(self, image: Union[np.ndarray, Image.Image, str, bytes],
                          language: str = 'eng', backend: Optional[str] = None,
                          preprocess: bool = True) -> OCRResult:
        """Extract text from image using OCR"""
        try:
            start_time = time.time()
            
            # Use specified backend or active backend
            backend_name = backend or self.active_backend
            
            if backend_name not in self.backends or not self.backends[backend_name].get('available'):
                backend_name = self.active_backend
            
            # Preprocess image if enabled
            if preprocess:
                processed_image = await self.preprocess_image(image)
            else:
                if isinstance(image, np.ndarray):
                    processed_image = image
                else:
                    processed_image = np.array(image) if PIL_AVAILABLE else np.zeros((100, 100), dtype=np.uint8)
            
            # Generate cache key
            cache_key = self._generate_cache_key(processed_image, language, backend_name)
            
            # Check cache
            if cache_key in self.result_cache:
                self.logger.debug("Using cached OCR result")
                return self.result_cache[cache_key]
            
            # Perform OCR based on backend
            if backend_name == 'tesseract':
                result = await self._ocr_tesseract(processed_image, language)
            elif backend_name == 'easyocr':
                result = await self._ocr_easyocr(processed_image, language)
            elif backend_name == 'paddleocr':
                result = await self._ocr_paddleocr(processed_image, language)
            else:
                result = await self._ocr_custom(processed_image, language)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Update statistics
            self.operation_stats['images_processed'] += 1
            self.operation_stats['text_extracted'] += len(result.text)
            self.operation_stats['total_processing_time'] += processing_time
            
            if backend_name not in self.operation_stats['backend_usage']:
                self.operation_stats['backend_usage'][backend_name] = 0
            self.operation_stats['backend_usage'][backend_name] += 1
            
            # Update average confidence
            total_ops = self.operation_stats['images_processed']
            current_avg = self.operation_stats['average_confidence']
            self.operation_stats['average_confidence'] = (
                (current_avg * (total_ops - 1) + result.confidence) / total_ops
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text extraction error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                bbox=(0, 0, 0, 0),
                words=[],
                lines=[],
                paragraphs=[],
                language=language,
                processing_time=0.0
            )
    
    async def _ocr_tesseract(self, image: np.ndarray, language: str) -> OCRResult:
        """Perform OCR using Tesseract"""
        try:
            # Convert language code
            lang_map = {
                'eng': 'eng', 'en': 'eng',
                'fra': 'fra', 'fr': 'fra',
                'deu': 'deu', 'de': 'deu',
                'spa': 'spa', 'es': 'spa',
                'chi_sim': 'chi_sim', 'zh': 'chi_sim'
            }
            tesseract_lang = lang_map.get(language, 'eng')
            
            # Configure Tesseract
            config = '--oem 3 --psm 6'  # Use LSTM OCR Engine Mode with uniform text block
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=tesseract_lang, config=config)
            
            # Get detailed data
            data = pytesseract.image_to_data(image, lang=tesseract_lang, config=config, output_type=pytesseract.Output.DICT)
            
            # Process results
            words = []
            lines = []
            paragraphs = []
            
            current_line = []
            current_paragraph = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Valid detection
                    word_info = {
                        'text': data['text'][i],
                        'confidence': float(data['conf'][i]) / 100.0,
                        'bbox': (
                            int(data['left'][i]),
                            int(data['top'][i]),
                            int(data['width'][i]),
                            int(data['height'][i])
                        )
                    }
                    words.append(word_info)
                    current_line.append(word_info)
                    current_paragraph.append(word_info)
                
                # Check for line break
                if data['block_num'][i] != data['block_num'][i+1] if i+1 < len(data['text']) else True:
                    if current_line:
                        lines.append({
                            'words': current_line.copy(),
                            'text': ' '.join([w['text'] for w in current_line]),
                            'bbox': self._calculate_bbox([w['bbox'] for w in current_line])
                        })
                        current_line = []
                
                # Check for paragraph break
                if data['par_num'][i] != data['par_num'][i+1] if i+1 < len(data['text']) else True:
                    if current_paragraph:
                        paragraphs.append({
                            'words': current_paragraph.copy(),
                            'text': ' '.join([w['text'] for w in current_paragraph]),
                            'bbox': self._calculate_bbox([w['bbox'] for w in current_paragraph])
                        })
                        current_paragraph = []
            
            # Calculate overall confidence
            confidences = [w['confidence'] for w in words if w['confidence'] > 0]
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            # Calculate overall bounding box
            if words:
                overall_bbox = self._calculate_bbox([w['bbox'] for w in words])
            else:
                overall_bbox = (0, 0, 0, 0)
            
            return OCRResult(
                text=text.strip(),
                confidence=overall_confidence,
                bbox=overall_bbox,
                words=words,
                lines=lines,
                paragraphs=paragraphs,
                language=language,
                processing_time=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Tesseract OCR error: {e}")
            raise
    
    async def _ocr_easyocr(self, image: np.ndarray, language: str) -> OCRResult:
        """Perform OCR using EasyOCR"""
        try:
            reader = self.backends['easyocr']['engine']
            
            # Convert language code
            lang_map = {
                'eng': 'en', 'en': 'en',
                'fra': 'fr', 'fr': 'fr',
                'deu': 'de', 'de': 'de',
                'spa': 'es', 'es': 'es',
                'chi_sim': 'ch_sim', 'zh': 'ch_sim'
            }
            easyocr_lang = lang_map.get(language, 'en')
            
            # Perform OCR
            results = reader.readtext(image, detail=1, paragraph=False)
            
            # Process results
            words = []
            text_parts = []
            
            for bbox_coords, text, confidence in results:
                # Convert bbox format
                bbox = (
                    int(min([p[0] for p in bbox_coords])),
                    int(min([p[1] for p in bbox_coords])),
                    int(max([p[0] for p in bbox_coords]) - min([p[0] for p in bbox_coords])),
                    int(max([p[1] for p in bbox_coords]) - min([p[1] for p in bbox_coords]))
                )
                
                word_info = {
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': bbox
                }
                words.append(word_info)
                text_parts.append(text)
            
            # Combine text
            full_text = ' '.join(text_parts)
            
            # Create lines (simple grouping by y-coordinate)
            lines = self._group_words_into_lines(words)
            
            # Create paragraphs (simple grouping)
            paragraphs = [{'words': words, 'text': full_text, 'bbox': self._calculate_bbox([w['bbox'] for w in words])}] if words else []
            
            # Calculate overall confidence
            overall_confidence = np.mean([w['confidence'] for w in words]) if words else 0.0
            
            # Calculate overall bounding box
            overall_bbox = self._calculate_bbox([w['bbox'] for w in words]) if words else (0, 0, 0, 0)
            
            return OCRResult(
                text=full_text,
                confidence=overall_confidence,
                bbox=overall_bbox,
                words=words,
                lines=lines,
                paragraphs=paragraphs,
                language=language,
                processing_time=0.0
            )
            
        except Exception as e:
            self.logger.error(f"EasyOCR error: {e}")
            raise
    
    async def _ocr_paddleocr(self, image: np.ndarray, language: str) -> OCRResult:
        """Perform OCR using PaddleOCR"""
        try:
            ocr = self.backends['paddleocr']['engine']
            
            # Perform OCR
            results = ocr.ocr(image, cls=True)
            
            # Process results
            words = []
            text_parts = []
            
            if results and results[0]:
                for line in results[0]:
                    bbox_coords, (text, confidence) = line
                    
                    # Convert bbox format
                    bbox = (
                        int(min([p[0] for p in bbox_coords])),
                        int(min([p[1] for p in bbox_coords])),
                        int(max([p[0] for p in bbox_coords]) - min([p[0] for p in bbox_coords])),
                        int(max([p[1] for p in bbox_coords]) - min([p[1] for p in bbox_coords]))
                    )
                    
                    word_info = {
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': bbox
                    }
                    words.append(word_info)
                    text_parts.append(text)
            
            # Combine text
            full_text = ' '.join(text_parts)
            
            # Create lines (simple grouping by y-coordinate)
            lines = self._group_words_into_lines(words)
            
            # Create paragraphs
            paragraphs = [{'words': words, 'text': full_text, 'bbox': self._calculate_bbox([w['bbox'] for w in words])}] if words else []
            
            # Calculate overall confidence
            overall_confidence = np.mean([w['confidence'] for w in words]) if words else 0.0
            
            # Calculate overall bounding box
            overall_bbox = self._calculate_bbox([w['bbox'] for w in words]) if words else (0, 0, 0, 0)
            
            return OCRResult(
                text=full_text,
                confidence=overall_confidence,
                bbox=overall_bbox,
                words=words,
                lines=lines,
                paragraphs=paragraphs,
                language=language,
                processing_time=0.0
            )
            
        except Exception as e:
            self.logger.error(f"PaddleOCR error: {e}")
            raise
    
    async def _ocr_custom(self, image: np.ndarray, language: str) -> OCRResult:
        """Perform OCR using custom implementation (fallback)"""
        try:
            # Simple template matching or basic character recognition
            # This is a very basic implementation for fallback
            
            # For now, return empty result
            return OCRResult(
                text="[Custom OCR - Limited functionality]",
                confidence=0.1,
                bbox=(0, 0, image.shape[1] if len(image.shape) > 1 else 100, 
                      image.shape[0] if len(image.shape) > 0 else 100),
                words=[],
                lines=[],
                paragraphs=[],
                language=language,
                processing_time=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Custom OCR error: {e}")
            raise
    
    def _group_words_into_lines(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group words into lines based on y-coordinate"""
        try:
            if not words:
                return []
            
            # Sort words by y-coordinate
            sorted_words = sorted(words, key=lambda w: w['bbox'][1])
            
            lines = []
            current_line = [sorted_words[0]]
            current_y = sorted_words[0]['bbox'][1]
            
            for word in sorted_words[1:]:
                word_y = word['bbox'][1]
                
                # If word is on roughly the same line (within threshold)
                if abs(word_y - current_y) < 20:  # 20 pixel threshold
                    current_line.append(word)
                else:
                    # Start new line
                    if current_line:
                        lines.append({
                            'words': current_line,
                            'text': ' '.join([w['text'] for w in current_line]),
                            'bbox': self._calculate_bbox([w['bbox'] for w in current_line])
                        })
                    current_line = [word]
                    current_y = word_y
            
            # Add last line
            if current_line:
                lines.append({
                    'words': current_line,
                    'text': ' '.join([w['text'] for w in current_line]),
                    'bbox': self._calculate_bbox([w['bbox'] for w in current_line])
                })
            
            return lines
            
        except Exception as e:
            self.logger.error(f"Group words into lines error: {e}")
            return []
    
    def _calculate_bbox(self, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Calculate combined bounding box from multiple bboxes"""
        try:
            if not bboxes:
                return (0, 0, 0, 0)
            
            min_x = min(bbox[0] for bbox in bboxes)
            min_y = min(bbox[1] for bbox in bboxes)
            max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
            max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
            
            return (min_x, min_y, max_x - min_x, max_y - min_y)
            
        except Exception as e:
            self.logger.error(f"Calculate bbox error: {e}")
            return (0, 0, 0, 0)
    
    def _generate_cache_key(self, image: np.ndarray, language: str, backend: str) -> str:
        """Generate cache key for image"""
        try:
            # Create hash from image data
            image_hash = hash(image.tobytes())
            return f"{image_hash}_{language}_{backend}"
        except:
            return f"unknown_{language}_{backend}_{time.time()}"
    
    def _cache_result(self, key: str, result: OCRResult):
        """Cache OCR result"""
        try:
            # Manage cache size
            if len(self.result_cache) >= self.cache_max_size:
                # Remove oldest entries
                keys_to_remove = list(self.result_cache.keys())[:self.cache_max_size // 2]
                for k in keys_to_remove:
                    del self.result_cache[k]
            
            self.result_cache[key] = result
            
        except Exception as e:
            self.logger.debug(f"Cache result error: {e}")
    
    # Batch Processing
    async def extract_text_batch(self, images: List[Union[np.ndarray, Image.Image, str, bytes]],
                                language: str = 'eng', backend: Optional[str] = None) -> List[OCRResult]:
        """Extract text from multiple images"""
        try:
            tasks = []
            for image in images:
                task = self.extract_text(image, language, backend)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for result in results:
                if isinstance(result, OCRResult):
                    valid_results.append(result)
                else:
                    # Create empty result for failed extractions
                    valid_results.append(OCRResult(
                        text="", confidence=0.0, bbox=(0, 0, 0, 0),
                        words=[], lines=[], paragraphs=[],
                        language=language, processing_time=0.0
                    ))
            
            return valid_results
            
        except Exception as e:
            self.logger.error(f"Batch text extraction error: {e}")
            return []
    
    # Configuration and Management
    async def set_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update OCR configuration"""
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
    
    async def get_available_languages(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """Get available languages for OCR"""
        try:
            backend_name = backend or self.active_backend
            
            if backend_name in self.backends and self.backends[backend_name].get('available'):
                languages = self.backends[backend_name].get('languages', [])
            else:
                languages = self.config['supported_languages']
            
            return {
                'success': True,
                'languages': languages,
                'backend': backend_name
            }
            
        except Exception as e:
            self.logger.error(f"Get available languages error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get OCR engine performance statistics"""
        return {
            'operation_stats': self.operation_stats.copy(),
            'backend_stats': {
                'active_backend': self.active_backend,
                'available_backends': [name for name, info in self.backends.items() if info.get('available')],
                'backend_priorities': {name: info.get('priority', 999) for name, info in self.backends.items() if info.get('available')}
            },
            'cache_stats': {
                'cache_size': len(self.result_cache),
                'cache_max_size': self.cache_max_size
            },
            'config': self.config.copy()
        }
    
    async def cleanup(self):
        """Cleanup OCR engine"""
        try:
            self.logger.info("Cleaning up OCR Engine...")
            
            # Clear cache
            self.result_cache.clear()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Cleanup backends
            for backend_name, backend_info in self.backends.items():
                if backend_info.get('available') and hasattr(backend_info.get('engine'), 'cleanup'):
                    try:
                        backend_info['engine'].cleanup()
                    except:
                        pass
            
            self.logger.info("OCR Engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"OCR Engine cleanup error: {e}")
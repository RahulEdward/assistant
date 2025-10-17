"""
OCR Package
Optical Character Recognition and screen analysis capabilities.
Provides comprehensive text extraction, element detection, and visual understanding.

Version: 1.0.0
Author: Desktop Assistant Team
"""

from .ocr_engine import OCREngine
from .screen_analyzer import ScreenAnalyzer
from .element_detector import ElementDetector
from .text_extractor import TextExtractor

__version__ = "1.0.0"
__all__ = [
    'OCREngine',
    'ScreenAnalyzer', 
    'ElementDetector',
    'TextExtractor'
]
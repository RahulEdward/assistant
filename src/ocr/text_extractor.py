"""
Advanced Text Extractor for OCR Module

This module provides sophisticated text extraction and processing capabilities,
including text cleaning, formatting, structure analysis, and content understanding.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from collections import defaultdict, Counter
import unicodedata
import string

# Third-party imports
import numpy as np
from PIL import Image
import cv2

# Internal imports
from .ocr_engine import OCREngine, OCRResult, TextRegion


@dataclass
class TextBlock:
    """Represents a structured text block with metadata."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    block_type: str  # paragraph, title, list_item, table_cell, etc.
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    alignment: str = "left"  # left, center, right, justify
    language: Optional[str] = None
    reading_order: int = 0


@dataclass
class DocumentStructure:
    """Represents the hierarchical structure of a document."""
    title: Optional[str] = None
    headings: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    lists: List[List[str]] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextAnalysis:
    """Comprehensive text analysis results."""
    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int
    average_word_length: float
    reading_level: str
    language: str
    sentiment: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)


class TextExtractor:
    """
    Advanced text extractor with sophisticated processing capabilities.
    
    Features:
    - Text cleaning and normalization
    - Structure analysis and document parsing
    - Content understanding and analysis
    - Multi-language support
    - Format detection and preservation
    """
    
    def __init__(self, ocr_engine: Optional[OCREngine] = None):
        """Initialize the text extractor."""
        self.logger = logging.getLogger(__name__)
        self.ocr_engine = ocr_engine or OCREngine()
        
        # Text processing configuration
        self.config = {
            'min_confidence': 0.5,
            'merge_threshold': 10,  # pixels
            'line_height_threshold': 1.5,
            'paragraph_gap_threshold': 2.0,
            'title_size_threshold': 1.3,
            'preserve_formatting': True,
            'auto_correct': True,
            'language_detection': True
        }
        
        # Initialize text processing components
        self._init_text_processors()
        self._init_patterns()
        
        # Performance tracking
        self.stats = {
            'extractions_performed': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0,
            'text_blocks_processed': 0
        }
        
        self.logger.info("TextExtractor initialized successfully")
    
    def _init_text_processors(self):
        """Initialize text processing components."""
        # Common text cleaning patterns
        self.cleaning_patterns = [
            (r'\s+', ' '),  # Multiple whitespace to single space
            (r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ''),  # Remove special chars
            (r'\.{3,}', '...'),  # Multiple dots to ellipsis
            (r'\-{2,}', '--'),  # Multiple dashes to double dash
        ]
        
        # Word correction dictionary (basic)
        self.corrections = {
            'teh': 'the',
            'adn': 'and',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'occured': 'occurred'
        }
        
        # Language detection patterns
        self.language_patterns = {
            'english': r'[a-zA-Z\s\.\,\!\?]+',
            'numeric': r'[\d\s\.\,\-\+\(\)]+',
            'mixed': r'[a-zA-Z\d\s\.\,\!\?\-\+\(\)]+',
        }
    
    def _init_patterns(self):
        """Initialize regex patterns for structure detection."""
        self.patterns = {
            'title': re.compile(r'^[A-Z][A-Za-z\s]{2,50}$'),
            'heading': re.compile(r'^[A-Z][A-Za-z\s\d]{3,100}$'),
            'bullet_point': re.compile(r'^[\•\*\-\+]\s+(.+)$'),
            'numbered_list': re.compile(r'^\d+[\.\)]\s+(.+)$'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'url': re.compile(r'https?://[^\s]+'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'currency': re.compile(r'\$\d+(?:\.\d{2})?'),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
        }
    
    def extract_from_image(self, image: np.ndarray, 
                          region: Optional[Tuple[int, int, int, int]] = None) -> List[TextBlock]:
        """
        Extract structured text blocks from an image.
        
        Args:
            image: Input image as numpy array
            region: Optional region to extract from (x, y, width, height)
            
        Returns:
            List of structured text blocks
        """
        start_time = datetime.now()
        
        try:
            # Crop image if region specified
            if region:
                x, y, w, h = region
                image = image[y:y+h, x:x+w]
            
            # Perform OCR
            ocr_result = self.ocr_engine.extract_text(image)
            
            # Convert OCR results to text blocks
            text_blocks = self._ocr_to_text_blocks(ocr_result, region)
            
            # Post-process text blocks
            text_blocks = self._post_process_blocks(text_blocks)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, text_blocks)
            
            self.logger.info(f"Extracted {len(text_blocks)} text blocks in {processing_time:.2f}s")
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"Error extracting text from image: {e}")
            return []
    
    def _ocr_to_text_blocks(self, ocr_result: OCRResult, 
                           offset: Optional[Tuple[int, int, int, int]] = None) -> List[TextBlock]:
        """Convert OCR results to structured text blocks."""
        text_blocks = []
        
        for region in ocr_result.regions:
            if region.confidence < self.config['min_confidence']:
                continue
            
            # Adjust coordinates if offset provided
            bbox = region.bbox
            if offset:
                x_offset, y_offset = offset[0], offset[1]
                bbox = (bbox[0] + x_offset, bbox[1] + y_offset, bbox[2], bbox[3])
            
            # Determine block type
            block_type = self._classify_text_block(region.text, bbox)
            
            # Create text block
            text_block = TextBlock(
                text=region.text,
                confidence=region.confidence,
                bbox=bbox,
                block_type=block_type,
                font_size=self._estimate_font_size(bbox),
                reading_order=len(text_blocks)
            )
            
            text_blocks.append(text_block)
        
        return text_blocks
    
    def _classify_text_block(self, text: str, bbox: Tuple[int, int, int, int]) -> str:
        """Classify the type of text block based on content and position."""
        text = text.strip()
        
        if not text:
            return "empty"
        
        # Check for specific patterns
        if self.patterns['bullet_point'].match(text):
            return "list_item"
        elif self.patterns['numbered_list'].match(text):
            return "numbered_item"
        elif self.patterns['title'].match(text) and len(text) < 50:
            return "title"
        elif self.patterns['heading'].match(text) and len(text) < 100:
            return "heading"
        elif len(text.split()) > 10:
            return "paragraph"
        elif len(text.split()) > 3:
            return "sentence"
        else:
            return "fragment"
    
    def _estimate_font_size(self, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate font size based on bounding box height."""
        return float(bbox[3])  # Height as font size approximation
    
    def _post_process_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Post-process text blocks for better structure and content."""
        if not text_blocks:
            return text_blocks
        
        # Sort by reading order (top to bottom, left to right)
        text_blocks = self._sort_reading_order(text_blocks)
        
        # Merge adjacent blocks if appropriate
        text_blocks = self._merge_adjacent_blocks(text_blocks)
        
        # Clean and correct text
        for block in text_blocks:
            block.text = self._clean_text(block.text)
            if self.config['auto_correct']:
                block.text = self._correct_text(block.text)
        
        # Detect language
        if self.config['language_detection']:
            self._detect_languages(text_blocks)
        
        return text_blocks
    
    def _sort_reading_order(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Sort text blocks in natural reading order."""
        # Sort by y-coordinate first (top to bottom), then x-coordinate (left to right)
        sorted_blocks = sorted(text_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        
        # Update reading order
        for i, block in enumerate(sorted_blocks):
            block.reading_order = i
        
        return sorted_blocks
    
    def _merge_adjacent_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Merge adjacent text blocks that likely belong together."""
        if len(text_blocks) < 2:
            return text_blocks
        
        merged_blocks = []
        current_block = text_blocks[0]
        
        for next_block in text_blocks[1:]:
            if self._should_merge_blocks(current_block, next_block):
                # Merge blocks
                current_block = self._merge_two_blocks(current_block, next_block)
            else:
                merged_blocks.append(current_block)
                current_block = next_block
        
        merged_blocks.append(current_block)
        return merged_blocks
    
    def _should_merge_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two blocks should be merged."""
        # Don't merge different block types
        if block1.block_type != block2.block_type:
            return False
        
        # Check horizontal alignment
        y1_center = block1.bbox[1] + block1.bbox[3] / 2
        y2_center = block2.bbox[1] + block2.bbox[3] / 2
        
        # Check if blocks are on the same line
        line_threshold = max(block1.bbox[3], block2.bbox[3]) * 0.5
        if abs(y1_center - y2_center) < line_threshold:
            # Check horizontal gap
            gap = block2.bbox[0] - (block1.bbox[0] + block1.bbox[2])
            return gap < self.config['merge_threshold']
        
        return False
    
    def _merge_two_blocks(self, block1: TextBlock, block2: TextBlock) -> TextBlock:
        """Merge two text blocks into one."""
        # Combine text
        combined_text = f"{block1.text} {block2.text}".strip()
        
        # Calculate combined bounding box
        x1 = min(block1.bbox[0], block2.bbox[0])
        y1 = min(block1.bbox[1], block2.bbox[1])
        x2 = max(block1.bbox[0] + block1.bbox[2], block2.bbox[0] + block2.bbox[2])
        y2 = max(block1.bbox[1] + block1.bbox[3], block2.bbox[1] + block2.bbox[3])
        
        # Average confidence
        avg_confidence = (block1.confidence + block2.confidence) / 2
        
        return TextBlock(
            text=combined_text,
            confidence=avg_confidence,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            block_type=block1.block_type,
            font_size=max(block1.font_size or 0, block2.font_size or 0),
            reading_order=block1.reading_order
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return text
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def _correct_text(self, text: str) -> str:
        """Apply basic text corrections."""
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for correction check
            clean_word = word.strip(string.punctuation).lower()
            if clean_word in self.corrections:
                # Preserve original case and punctuation
                corrected = self.corrections[clean_word]
                if word[0].isupper():
                    corrected = corrected.capitalize()
                # Add back punctuation
                for char in word:
                    if char in string.punctuation:
                        corrected += char
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _detect_languages(self, text_blocks: List[TextBlock]):
        """Detect language for each text block."""
        for block in text_blocks:
            # Simple language detection based on character patterns
            text = block.text.lower()
            
            if re.match(self.language_patterns['english'], text):
                block.language = 'en'
            elif re.match(self.language_patterns['numeric'], text):
                block.language = 'numeric'
            else:
                block.language = 'unknown'
    
    def analyze_document_structure(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Analyze document structure from text blocks."""
        structure = DocumentStructure()
        
        # Extract different components
        for block in text_blocks:
            if block.block_type == 'title':
                if not structure.title:
                    structure.title = block.text
            elif block.block_type == 'heading':
                structure.headings.append(block.text)
            elif block.block_type == 'paragraph':
                structure.paragraphs.append(block.text)
            elif block.block_type in ['list_item', 'numbered_item']:
                # Group consecutive list items
                if structure.lists and isinstance(structure.lists[-1], list):
                    structure.lists[-1].append(block.text)
                else:
                    structure.lists.append([block.text])
        
        # Add metadata
        structure.metadata = {
            'total_blocks': len(text_blocks),
            'extraction_time': datetime.now().isoformat(),
            'average_confidence': statistics.mean([b.confidence for b in text_blocks]) if text_blocks else 0
        }
        
        return structure
    
    def analyze_text_content(self, text_blocks: List[TextBlock]) -> TextAnalysis:
        """Perform comprehensive text analysis."""
        # Combine all text
        full_text = ' '.join([block.text for block in text_blocks])
        
        if not full_text.strip():
            return TextAnalysis(0, 0, 0, 0, 0.0, 'unknown', 'unknown')
        
        # Basic statistics
        words = full_text.split()
        word_count = len(words)
        character_count = len(full_text)
        sentence_count = len(re.findall(r'[.!?]+', full_text))
        paragraph_count = len([b for b in text_blocks if b.block_type == 'paragraph'])
        
        # Average word length
        avg_word_length = statistics.mean([len(word.strip(string.punctuation)) for word in words]) if words else 0
        
        # Simple reading level estimation
        if avg_word_length < 4:
            reading_level = 'elementary'
        elif avg_word_length < 6:
            reading_level = 'intermediate'
        else:
            reading_level = 'advanced'
        
        # Language detection (simplified)
        language = 'english' if re.search(r'[a-zA-Z]', full_text) else 'unknown'
        
        # Extract entities (basic patterns)
        entities = {}
        entities['emails'] = self.patterns['email'].findall(full_text)
        entities['phones'] = self.patterns['phone'].findall(full_text)
        entities['urls'] = self.patterns['url'].findall(full_text)
        entities['dates'] = self.patterns['date'].findall(full_text)
        entities['currencies'] = self.patterns['currency'].findall(full_text)
        
        # Extract keywords (simple frequency-based)
        word_freq = Counter(word.lower().strip(string.punctuation) for word in words if len(word) > 3)
        keywords = [word for word, freq in word_freq.most_common(10)]
        
        return TextAnalysis(
            word_count=word_count,
            character_count=character_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            average_word_length=avg_word_length,
            reading_level=reading_level,
            language=language,
            keywords=keywords,
            entities=entities
        )
    
    def extract_tables(self, text_blocks: List[TextBlock]) -> List[List[List[str]]]:
        """Extract table structures from text blocks."""
        tables = []
        current_table = []
        
        for block in text_blocks:
            # Simple table detection based on alignment and structure
            if self._is_table_row(block.text):
                row = self._parse_table_row(block.text)
                current_table.append(row)
            else:
                if current_table:
                    tables.append(current_table)
                    current_table = []
        
        if current_table:
            tables.append(current_table)
        
        return tables
    
    def _is_table_row(self, text: str) -> bool:
        """Check if text represents a table row."""
        # Simple heuristic: multiple words separated by significant whitespace or tabs
        parts = re.split(r'\s{3,}|\t+', text.strip())
        return len(parts) > 1
    
    def _parse_table_row(self, text: str) -> List[str]:
        """Parse a table row into cells."""
        # Split by multiple spaces or tabs
        cells = re.split(r'\s{3,}|\t+', text.strip())
        return [cell.strip() for cell in cells if cell.strip()]
    
    def get_text_by_region(self, text_blocks: List[TextBlock], 
                          region: Tuple[int, int, int, int]) -> List[TextBlock]:
        """Get text blocks within a specific region."""
        x, y, w, h = region
        region_blocks = []
        
        for block in text_blocks:
            bx, by, bw, bh = block.bbox
            
            # Check if block overlaps with region
            if (bx < x + w and bx + bw > x and 
                by < y + h and by + bh > y):
                region_blocks.append(block)
        
        return region_blocks
    
    def search_text(self, text_blocks: List[TextBlock], 
                   query: str, case_sensitive: bool = False) -> List[TextBlock]:
        """Search for text within blocks."""
        if not case_sensitive:
            query = query.lower()
        
        matching_blocks = []
        for block in text_blocks:
            text = block.text if case_sensitive else block.text.lower()
            if query in text:
                matching_blocks.append(block)
        
        return matching_blocks
    
    def _update_stats(self, processing_time: float, text_blocks: List[TextBlock]):
        """Update performance statistics."""
        self.stats['extractions_performed'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['text_blocks_processed'] += len(text_blocks)
        
        if text_blocks:
            avg_confidence = statistics.mean([block.confidence for block in text_blocks])
            # Running average
            total_extractions = self.stats['extractions_performed']
            self.stats['average_confidence'] = (
                (self.stats['average_confidence'] * (total_extractions - 1) + avg_confidence) / 
                total_extractions
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['extractions_performed'] > 0:
            stats['average_processing_time'] = (
                stats['total_processing_time'] / stats['extractions_performed']
            )
        else:
            stats['average_processing_time'] = 0.0
        
        return stats
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update configuration settings."""
        self.config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.ocr_engine:
            self.ocr_engine.cleanup()
        
        self.logger.info("TextExtractor cleanup completed")
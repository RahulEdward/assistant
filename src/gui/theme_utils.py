"""
Theme Utilities for Computer Assistant GUI

Provides utility functions to apply themes to existing widgets
without requiring modifications to each widget class.
"""

from PySide6.QtWidgets import (QWidget, QPushButton, QLabel, QLineEdit, 
                               QTextEdit, QComboBox, QListWidget, QTreeWidget,
                               QTableWidget, QProgressBar, QGroupBox, QFrame,
                               QTabWidget, QScrollArea, QCheckBox, QSpinBox)
from PySide6.QtCore import QObject, Signal
from .theme_manager import theme_manager


class ThemeApplicator(QObject):
    """
    Utility class to apply themes to existing widgets recursively.
    This allows theme application without modifying individual widget classes.
    """
    
    theme_applied = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.theme_manager = theme_manager
        
    def apply_theme_to_widget(self, widget: QWidget, recursive: bool = True):
        """
        Apply current theme to a widget and optionally its children.
        
        Args:
            widget: The widget to apply theme to
            recursive: Whether to apply theme to child widgets
        """
        if not widget:
            return
            
        colors = self.theme_manager.get_current_colors()
        
        # Apply theme based on widget type
        widget_type = type(widget).__name__
        
        if isinstance(widget, QPushButton):
            self._apply_button_theme(widget, colors)
        elif isinstance(widget, QLabel):
            self._apply_label_theme(widget, colors)
        elif isinstance(widget, (QLineEdit, QTextEdit)):
            self._apply_input_theme(widget, colors)
        elif isinstance(widget, QComboBox):
            self._apply_combo_theme(widget, colors)
        elif isinstance(widget, (QListWidget, QTreeWidget, QTableWidget)):
            self._apply_list_theme(widget, colors)
        elif isinstance(widget, QProgressBar):
            self._apply_progress_theme(widget, colors)
        elif isinstance(widget, QGroupBox):
            self._apply_group_theme(widget, colors)
        elif isinstance(widget, QFrame):
            self._apply_frame_theme(widget, colors)
        elif isinstance(widget, QTabWidget):
            self._apply_tab_theme(widget, colors)
        elif isinstance(widget, QScrollArea):
            self._apply_scroll_theme(widget, colors)
        elif isinstance(widget, QCheckBox):
            self._apply_checkbox_theme(widget, colors)
        elif isinstance(widget, QSpinBox):
            self._apply_spinbox_theme(widget, colors)
        else:
            # Apply generic widget theme
            self._apply_generic_theme(widget, colors)
        
        # Recursively apply to children if requested
        if recursive:
            for child in widget.findChildren(QWidget):
                # Skip if child already has specific styling
                if not child.styleSheet():
                    self.apply_theme_to_widget(child, False)
    
    def _apply_button_theme(self, button: QPushButton, colors: dict):
        """Apply theme to button widgets"""
        style = f"""
        QPushButton {{
            background-color: {colors['primary']};
            color: {colors['text_on_primary']};
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 500;
            min-height: 20px;
        }}
        
        QPushButton:hover {{
            background-color: {colors['primary_variant']};
        }}
        
        QPushButton:pressed {{
            background-color: {colors['primary_variant']};
            transform: translateY(1px);
        }}
        
        QPushButton:disabled {{
            background-color: {colors['text_disabled']};
            color: {colors['background']};
        }}
        
        QPushButton:focus {{
            outline: 2px solid {colors['focus']};
            outline-offset: 2px;
        }}
        """
        button.setStyleSheet(style)
    
    def _apply_label_theme(self, label: QLabel, colors: dict):
        """Apply theme to label widgets"""
        style = f"""
        QLabel {{
            color: {colors['text_primary']};
            background-color: transparent;
        }}
        """
        label.setStyleSheet(style)
    
    def _apply_input_theme(self, input_widget: QWidget, colors: dict):
        """Apply theme to input widgets (LineEdit, TextEdit)"""
        style = f"""
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
            border: 2px solid {colors['border']};
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {colors['primary']};
            outline: none;
        }}
        
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
            background-color: {colors['surface']};
            color: {colors['text_disabled']};
        }}
        """
        input_widget.setStyleSheet(style)
    
    def _apply_combo_theme(self, combo: QComboBox, colors: dict):
        """Apply theme to combo box widgets"""
        style = f"""
        QComboBox {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
            border: 2px solid {colors['border']};
            border-radius: 4px;
            padding: 6px 12px;
            min-width: 100px;
        }}
        
        QComboBox:hover {{
            border-color: {colors['primary']};
        }}
        
        QComboBox:focus {{
            border-color: {colors['primary']};
            outline: none;
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {colors['text_secondary']};
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            selection-background-color: {colors['selected']};
        }}
        """
        combo.setStyleSheet(style)
    
    def _apply_list_theme(self, list_widget: QWidget, colors: dict):
        """Apply theme to list-like widgets"""
        style = f"""
        QListWidget, QTreeWidget, QTableWidget {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            alternate-background-color: {colors['surface']};
            selection-background-color: {colors['selected']};
            selection-color: {colors['text_primary']};
        }}
        
        QListWidget::item, QTreeWidget::item, QTableWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {colors['divider']};
        }}
        
        QListWidget::item:hover, QTreeWidget::item:hover, QTableWidget::item:hover {{
            background-color: {colors['hover']};
        }}
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
            background-color: {colors['selected']};
        }}
        
        QHeaderView::section {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            padding: 6px;
            font-weight: bold;
        }}
        """
        list_widget.setStyleSheet(style)
    
    def _apply_progress_theme(self, progress: QProgressBar, colors: dict):
        """Apply theme to progress bar widgets"""
        style = f"""
        QProgressBar {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            border-radius: 4px;
            text-align: center;
        }}
        
        QProgressBar::chunk {{
            background-color: {colors['primary']};
            border-radius: 3px;
        }}
        """
        progress.setStyleSheet(style)
    
    def _apply_group_theme(self, group: QGroupBox, colors: dict):
        """Apply theme to group box widgets"""
        style = f"""
        QGroupBox {{
            color: {colors['text_primary']};
            border: 2px solid {colors['border']};
            border-radius: 8px;
            margin-top: 12px;
            font-weight: bold;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            background-color: {colors['background']};
        }}
        """
        group.setStyleSheet(style)
    
    def _apply_frame_theme(self, frame: QFrame, colors: dict):
        """Apply theme to frame widgets"""
        style = f"""
        QFrame {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
        }}
        """
        frame.setStyleSheet(style)
    
    def _apply_tab_theme(self, tab_widget: QTabWidget, colors: dict):
        """Apply theme to tab widgets"""
        style = f"""
        QTabWidget::pane {{
            border: 1px solid {colors['border']};
            background-color: {colors['background']};
        }}
        
        QTabBar::tab {{
            background-color: {colors['surface']};
            color: {colors['text_secondary']};
            border: 1px solid {colors['border']};
            padding: 8px 16px;
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors['primary']};
            color: {colors['text_on_primary']};
            border-bottom: none;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {colors['hover']};
            color: {colors['text_primary']};
        }}
        """
        tab_widget.setStyleSheet(style)
    
    def _apply_scroll_theme(self, scroll: QScrollArea, colors: dict):
        """Apply theme to scroll area widgets"""
        style = f"""
        QScrollArea {{
            background-color: {colors['background']};
            border: 1px solid {colors['border']};
        }}
        
        QScrollBar:vertical {{
            background-color: {colors['surface']};
            width: 12px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {colors['text_disabled']};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {colors['text_secondary']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        """
        scroll.setStyleSheet(style)
    
    def _apply_checkbox_theme(self, checkbox: QCheckBox, colors: dict):
        """Apply theme to checkbox widgets"""
        style = f"""
        QCheckBox {{
            color: {colors['text_primary']};
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {colors['border']};
            border-radius: 3px;
            background-color: {colors['background']};
        }}
        
        QCheckBox::indicator:hover {{
            border-color: {colors['primary']};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {colors['primary']};
            border-color: {colors['primary']};
        }}
        
        QCheckBox::indicator:checked:hover {{
            background-color: {colors['primary_variant']};
        }}
        """
        checkbox.setStyleSheet(style)
    
    def _apply_spinbox_theme(self, spinbox: QSpinBox, colors: dict):
        """Apply theme to spinbox widgets"""
        style = f"""
        QSpinBox {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
            border: 2px solid {colors['border']};
            border-radius: 4px;
            padding: 4px 8px;
        }}
        
        QSpinBox:focus {{
            border-color: {colors['primary']};
        }}
        
        QSpinBox::up-button, QSpinBox::down-button {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            width: 16px;
        }}
        
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background-color: {colors['hover']};
        }}
        """
        spinbox.setStyleSheet(style)
    
    def _apply_generic_theme(self, widget: QWidget, colors: dict):
        """Apply generic theme to any widget"""
        style = f"""
        QWidget {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
        }}
        """
        widget.setStyleSheet(style)


# Global theme applicator instance
theme_applicator = ThemeApplicator()
"""
Theme Toggle Button Widget

An accessible and intuitive button for switching between dark and light themes
with visual feedback, smooth animations, and proper ARIA attributes.
"""

from PySide6.QtWidgets import (QToolButton, QWidget, QHBoxLayout, QLabel, 
                               QGraphicsOpacityEffect, QSizePolicy)
from PySide6.QtCore import (QPropertyAnimation, QEasingCurve, Property, 
                           QRect, QTimer, Signal, QSize, Qt)
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QIcon, 
                          QPixmap, QPainterPath, QFont)
from ..theme_manager import theme_manager, ThemeMode


class AnimatedToggleButton(QToolButton):
    """
    An animated toggle button that smoothly transitions between states.
    Features smooth sliding animation and visual feedback.
    """
    
    toggled_with_animation = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(60, 32)
        
        # Animation properties
        self._position = 0.0
        self._animation = QPropertyAnimation(self, b"position")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Connect signals
        self.toggled.connect(self._on_toggled)
        
        # Set up accessibility
        self.setAccessibleName("Theme toggle")
        self.setAccessibleDescription("Switch between light and dark themes")
        
        # Set focus policy for keyboard navigation
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    @Property(float)
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value
        self.update()
    
    def _on_toggled(self, checked):
        """Handle toggle state change with animation"""
        target_position = 1.0 if checked else 0.0
        
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(target_position)
        self._animation.start()
        
        # Update accessibility description
        theme_name = "dark" if checked else "light"
        self.setAccessibleDescription(f"Currently in {theme_name} theme. Click to switch.")
        
        # Emit custom signal after animation
        QTimer.singleShot(self._animation.duration(), 
                         lambda: self.toggled_with_animation.emit(checked))
    
    def paintEvent(self, event):
        """Custom paint event for the toggle button"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get current colors from theme manager
        colors = theme_manager.get_current_colors()
        
        # Calculate dimensions
        rect = self.rect()
        track_rect = QRect(2, 6, rect.width() - 4, rect.height() - 12)
        thumb_size = 20
        thumb_margin = 2
        
        # Calculate thumb position based on animation
        max_thumb_x = track_rect.width() - thumb_size - thumb_margin
        thumb_x = track_rect.x() + thumb_margin + (max_thumb_x * self._position)
        thumb_y = track_rect.y() + (track_rect.height() - thumb_size) // 2
        thumb_rect = QRect(int(thumb_x), thumb_y, thumb_size, thumb_size)
        
        # Draw track background
        track_color = QColor(colors['primary'] if self.isChecked() else colors['border'])
        if not self.isEnabled():
            track_color = QColor(colors['text_disabled'])
        
        painter.setBrush(QBrush(track_color))
        painter.setPen(QPen())
        
        track_path = QPainterPath()
        track_path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        painter.drawPath(track_path)
        
        # Draw thumb
        thumb_color = QColor(colors['text_on_primary'] if self.isChecked() else colors['text_primary'])
        if not self.isEnabled():
            thumb_color = QColor(colors['background'])
        
        # Add shadow effect
        shadow_rect = thumb_rect.adjusted(1, 1, 1, 1)
        shadow_color = QColor(colors['shadow'])
        painter.setBrush(QBrush(shadow_color))
        painter.setPen(QPen(shadow_color, 0))
        
        shadow_path = QPainterPath()
        shadow_path.addEllipse(shadow_rect)
        painter.drawPath(shadow_path)
        
        # Draw main thumb
        painter.setBrush(QBrush(thumb_color))
        painter.setPen(QPen(thumb_color, 1))
        
        thumb_path = QPainterPath()
        thumb_path.addEllipse(thumb_rect)
        painter.drawPath(thumb_path)
        
        # Draw icons on thumb
        self._draw_theme_icon(painter, thumb_rect)
        

    
    def _draw_theme_icon(self, painter, thumb_rect):
        """Draw theme-appropriate icon on the thumb"""
        colors = theme_manager.get_current_colors()
        icon_color = QColor(colors['primary'] if self.isChecked() else colors['text_secondary'])
        
        painter.setPen(QPen(icon_color, 2))
        painter.setBrush(QBrush())
        
        # Calculate icon area (smaller than thumb)
        icon_size = 8
        icon_x = thumb_rect.center().x() - icon_size // 2
        icon_y = thumb_rect.center().y() - icon_size // 2
        
        if self.isChecked():
            # Draw moon icon for dark theme
            self._draw_moon_icon(painter, icon_x, icon_y, icon_size)
        else:
            # Draw sun icon for light theme
            self._draw_sun_icon(painter, icon_x, icon_y, icon_size)
    
    def _draw_sun_icon(self, painter, x, y, size):
        """Draw sun icon for light theme"""
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 4
        
        # Draw sun center
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw sun rays
        ray_length = size // 6
        for i in range(8):
            angle = i * 45  # 8 rays at 45-degree intervals
            import math
            start_x = center_x + radius * math.cos(math.radians(angle))
            start_y = center_y + radius * math.sin(math.radians(angle))
            end_x = center_x + (radius + ray_length) * math.cos(math.radians(angle))
            end_y = center_y + (radius + ray_length) * math.sin(math.radians(angle))
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
    
    def _draw_moon_icon(self, painter, x, y, size):
        """Draw moon icon for dark theme"""
        # Draw crescent moon
        moon_path = QPainterPath()
        
        # Main circle
        moon_path.addEllipse(x, y, size, size)
        
        # Subtract smaller circle to create crescent
        offset = size // 4
        moon_path.addEllipse(x + offset, y, size - offset, size)
        
        painter.fillPath(moon_path, painter.pen().color())
    
    def keyPressEvent(self, event):
        """Handle keyboard events for accessibility"""
        if event.key() in (event.Key.Key_Space, event.Key.Key_Return, event.Key.Key_Enter):
            self.toggle()
            event.accept()
        else:
            super().keyPressEvent(event)


class ThemeToggleWidget(QWidget):
    """
    Complete theme toggle widget with label and button.
    Provides context and accessibility information.
    """
    
    theme_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._update_theme_display()
    
    def _setup_ui(self):
        """Set up the widget UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Theme label
        self.theme_label = QLabel("Theme:")
        self.theme_label.setAccessibleName("Theme setting label")
        
        # Current theme indicator
        self.current_theme_label = QLabel()
        self.current_theme_label.setAccessibleName("Current theme indicator")
        font = QFont()
        font.setWeight(QFont.Weight.Medium)
        self.current_theme_label.setFont(font)
        
        # Toggle button
        self.toggle_button = AnimatedToggleButton()
        
        # Add widgets to layout
        layout.addWidget(self.theme_label)
        layout.addWidget(self.current_theme_label)
        layout.addStretch()
        layout.addWidget(self.toggle_button)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(40)
    
    def _connect_signals(self):
        """Connect widget signals"""
        self.toggle_button.toggled_with_animation.connect(self._on_theme_toggled)
        theme_manager.theme_changed.connect(self._on_theme_manager_changed)
    
    def _on_theme_toggled(self, is_dark):
        """Handle theme toggle from button"""
        theme_manager.toggle_theme()
    
    def _on_theme_manager_changed(self, theme_name):
        """Handle theme change from theme manager"""
        self._update_theme_display()
        self.theme_changed.emit(theme_name)
    
    def _update_theme_display(self):
        """Update the display to reflect current theme"""
        current_theme = theme_manager.get_current_theme()
        is_dark = current_theme == ThemeMode.DARK
        
        # Update button state without triggering signals
        self.toggle_button.blockSignals(True)
        self.toggle_button.setChecked(is_dark)
        self.toggle_button.blockSignals(False)
        
        # Update labels
        theme_name = "Dark" if is_dark else "Light"
        self.current_theme_label.setText(theme_name)
        
        # Update accessibility
        self.current_theme_label.setAccessibleDescription(f"Current theme is {theme_name}")
        
        # Apply theme colors
        self._apply_theme_colors()
    
    def _apply_theme_colors(self):
        """Apply current theme colors to the widget"""
        colors = theme_manager.get_current_colors()
        
        # Enhanced focus styling for better accessibility
        focus_style = f"""
        QToolButton:focus {{
            outline: 3px solid {colors['focus']};
            outline-offset: 2px;
            border-radius: 20px;
        }}
        """
        
        # Apply enhanced styling to toggle button
        button_style = f"""
        QToolButton {{
            background-color: transparent;
            border: 2px solid {colors['border']};
            border-radius: 20px;
            padding: 2px;
            min-width: 60px;
            min-height: 30px;
        }}
        
        QToolButton:hover {{
            border-color: {colors['primary']};
            background-color: {colors['hover']};
        }}
        
        QToolButton:pressed {{
            background-color: {colors['selected']};
        }}
        
        {focus_style}
        """
        
        self.toggle_button.setStyleSheet(button_style)
        
        # Apply label styling
        label_style = f"""
        QLabel {{
            color: {colors['text_primary']};
            font-weight: 500;
            font-size: 14px;
            background-color: transparent;
        }}
        """
        
        self.theme_label.setStyleSheet(label_style)
        
        # Highlight current theme label
        current_theme_style = f"""
        QLabel {{
            color: {colors['primary']};
            background-color: transparent;
            font-weight: 600;
        }}
        """
        
        self.current_theme_label.setStyleSheet(current_theme_style)
    
    def get_toggle_button(self) -> AnimatedToggleButton:
        """Get the toggle button for external access"""
        return self.toggle_button
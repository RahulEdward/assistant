"""
Specialized Modules Package for Computer Assistant

This package contains specialized modules for domain-specific tasks including:
- Financial analysis and trading automation
- Code generation and development assistance
- Automated testing and quality assurance
"""

__version__ = "1.0.0"
__author__ = "Computer Assistant Team"

# Import specialized modules
from .financial_analyzer import FinancialAnalyzer, MarketData, TradingSignal
from .code_generator import CodeGenerator, CodeTemplate, GeneratedCode
from .test_automator import TestAutomator, TestCase, TestExecution
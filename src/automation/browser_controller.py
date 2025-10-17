"""
Browser Controller
Advanced web browser automation with support for multiple browsers.
Provides comprehensive web interaction, scraping, and automation capabilities.
"""

import asyncio
import logging
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import base64

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.common.exceptions import (
        TimeoutException, NoSuchElementException, WebDriverException,
        ElementNotInteractableException, StaleElementReferenceException
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    import requests
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


@dataclass
class BrowserSession:
    """Browser session information"""
    session_id: str
    browser_type: str
    driver: Any
    start_time: datetime
    current_url: str
    title: str
    window_handles: List[str]
    cookies: List[Dict[str, Any]]
    local_storage: Dict[str, Any]
    session_storage: Dict[str, Any]


@dataclass
class WebElement:
    """Web element information"""
    tag_name: str
    text: str
    attributes: Dict[str, str]
    location: Dict[str, int]
    size: Dict[str, int]
    is_displayed: bool
    is_enabled: bool
    is_selected: bool


@dataclass
class PageInfo:
    """Web page information"""
    url: str
    title: str
    source_length: int
    load_time: float
    elements_count: int
    images_count: int
    links_count: int
    forms_count: int


class BrowserController:
    """Advanced browser automation controller"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Browser sessions
        self.sessions = {}  # session_id -> BrowserSession
        self.active_session = None
        
        # Browser configurations
        self.browser_configs = {
            'chrome': {
                'driver_class': webdriver.Chrome if SELENIUM_AVAILABLE else None,
                'service_class': ChromeService if SELENIUM_AVAILABLE else None,
                'options_class': webdriver.ChromeOptions if SELENIUM_AVAILABLE else None
            },
            'firefox': {
                'driver_class': webdriver.Firefox if SELENIUM_AVAILABLE else None,
                'service_class': FirefoxService if SELENIUM_AVAILABLE else None,
                'options_class': webdriver.FirefoxOptions if SELENIUM_AVAILABLE else None
            },
            'edge': {
                'driver_class': webdriver.Edge if SELENIUM_AVAILABLE else None,
                'service_class': EdgeService if SELENIUM_AVAILABLE else None,
                'options_class': webdriver.EdgeOptions if SELENIUM_AVAILABLE else None
            }
        }
        
        # Performance monitoring
        self.operation_stats = {
            'sessions_created': 0,
            'pages_loaded': 0,
            'elements_interacted': 0,
            'screenshots_taken': 0,
            'downloads_completed': 0
        }
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Default timeouts
        self.default_timeout = 10
        self.page_load_timeout = 30
        self.script_timeout = 30
        
        # Download directory
        self.download_dir = Path.cwd() / 'downloads'
        self.download_dir.mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize browser controller"""
        try:
            self.logger.info("Initializing Browser Controller...")
            
            if not SELENIUM_AVAILABLE:
                self.logger.warning("Selenium not available - browser automation limited")
            
            if not BS4_AVAILABLE:
                self.logger.warning("BeautifulSoup not available - HTML parsing limited")
            
            # Check for browser drivers
            await self._check_browser_drivers()
            
            self.logger.info("Browser Controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Browser Controller initialization error: {e}")
            return False
    
    async def _check_browser_drivers(self):
        """Check availability of browser drivers"""
        try:
            available_browsers = []
            
            for browser_name, config in self.browser_configs.items():
                if config['driver_class'] is None:
                    continue
                
                try:
                    # Try to create a driver instance
                    options = config['options_class']()
                    options.add_argument('--headless')
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    
                    driver = config['driver_class'](options=options)
                    driver.quit()
                    
                    available_browsers.append(browser_name)
                    
                except Exception as e:
                    self.logger.debug(f"{browser_name} driver not available: {e}")
            
            self.available_browsers = available_browsers
            self.logger.info(f"Available browsers: {available_browsers}")
            
        except Exception as e:
            self.logger.error(f"Browser driver check error: {e}")
            self.available_browsers = []
    
    # Session Management
    async def create_session(self, browser_type: str = 'chrome', headless: bool = False, **options) -> Dict[str, Any]:
        """Create new browser session"""
        try:
            if not SELENIUM_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Selenium not available'
                }
            
            if browser_type not in self.browser_configs:
                return {
                    'success': False,
                    'error': f'Unsupported browser: {browser_type}',
                    'available_browsers': list(self.browser_configs.keys())
                }
            
            config = self.browser_configs[browser_type]
            if config['driver_class'] is None:
                return {
                    'success': False,
                    'error': f'{browser_type} driver not available'
                }
            
            # Configure browser options
            browser_options = config['options_class']()
            
            if headless:
                browser_options.add_argument('--headless')
            
            # Common options
            browser_options.add_argument('--no-sandbox')
            browser_options.add_argument('--disable-dev-shm-usage')
            browser_options.add_argument('--disable-gpu')
            browser_options.add_argument('--window-size=1920,1080')
            
            # Download preferences
            if browser_type == 'chrome':
                prefs = {
                    'download.default_directory': str(self.download_dir),
                    'download.prompt_for_download': False,
                    'download.directory_upgrade': True,
                    'safebrowsing.enabled': True
                }
                browser_options.add_experimental_option('prefs', prefs)
            
            # Custom options
            for option_key, option_value in options.items():
                if option_key == 'user_agent':
                    browser_options.add_argument(f'--user-agent={option_value}')
                elif option_key == 'proxy':
                    browser_options.add_argument(f'--proxy-server={option_value}')
                elif option_key == 'window_size':
                    browser_options.add_argument(f'--window-size={option_value}')
            
            # Create driver
            driver = config['driver_class'](options=browser_options)
            
            # Set timeouts
            driver.implicitly_wait(self.default_timeout)
            driver.set_page_load_timeout(self.page_load_timeout)
            driver.set_script_timeout(self.script_timeout)
            
            # Create session
            session_id = f"{browser_type}_{int(time.time())}"
            session = BrowserSession(
                session_id=session_id,
                browser_type=browser_type,
                driver=driver,
                start_time=datetime.now(),
                current_url='',
                title='',
                window_handles=[],
                cookies=[],
                local_storage={},
                session_storage={}
            )
            
            self.sessions[session_id] = session
            self.active_session = session_id
            
            self.operation_stats['sessions_created'] += 1
            
            return {
                'success': True,
                'session_id': session_id,
                'browser_type': browser_type,
                'headless': headless
            }
            
        except Exception as e:
            self.logger.error(f"Create session error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Close browser session"""
        try:
            if session_id is None:
                session_id = self.active_session
            
            if session_id not in self.sessions:
                return {
                    'success': False,
                    'error': f'Session not found: {session_id}'
                }
            
            session = self.sessions[session_id]
            
            # Close browser
            try:
                session.driver.quit()
            except Exception as e:
                self.logger.warning(f"Error closing driver: {e}")
            
            # Remove session
            del self.sessions[session_id]
            
            # Update active session
            if self.active_session == session_id:
                self.active_session = next(iter(self.sessions.keys())) if self.sessions else None
            
            return {
                'success': True,
                'session_id': session_id,
                'active_session': self.active_session
            }
            
        except Exception as e:
            self.logger.error(f"Close session error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def switch_session(self, session_id: str) -> Dict[str, Any]:
        """Switch active session"""
        try:
            if session_id not in self.sessions:
                return {
                    'success': False,
                    'error': f'Session not found: {session_id}',
                    'available_sessions': list(self.sessions.keys())
                }
            
            self.active_session = session_id
            
            return {
                'success': True,
                'session_id': session_id,
                'browser_type': self.sessions[session_id].browser_type
            }
            
        except Exception as e:
            self.logger.error(f"Switch session error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_active_driver(self):
        """Get active driver instance"""
        if not self.active_session or self.active_session not in self.sessions:
            return None
        return self.sessions[self.active_session].driver
    
    # Navigation
    async def navigate_to(self, url: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Navigate to URL"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {
                    'success': False,
                    'error': 'No active session'
                }
            
            start_time = time.time()
            
            # Navigate to URL
            driver.get(url)
            
            load_time = time.time() - start_time
            
            # Update session info
            session = self.sessions[session_id or self.active_session]
            session.current_url = driver.current_url
            session.title = driver.title
            session.window_handles = driver.window_handles
            
            self.operation_stats['pages_loaded'] += 1
            
            return {
                'success': True,
                'url': driver.current_url,
                'title': driver.title,
                'load_time': load_time
            }
            
        except Exception as e:
            self.logger.error(f"Navigate error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def go_back(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Go back in browser history"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            driver.back()
            
            # Update session info
            session = self.sessions[session_id or self.active_session]
            session.current_url = driver.current_url
            session.title = driver.title
            
            return {
                'success': True,
                'url': driver.current_url,
                'title': driver.title
            }
            
        except Exception as e:
            self.logger.error(f"Go back error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def go_forward(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Go forward in browser history"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            driver.forward()
            
            # Update session info
            session = self.sessions[session_id or self.active_session]
            session.current_url = driver.current_url
            session.title = driver.title
            
            return {
                'success': True,
                'url': driver.current_url,
                'title': driver.title
            }
            
        except Exception as e:
            self.logger.error(f"Go forward error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def refresh_page(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Refresh current page"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            driver.refresh()
            
            return {
                'success': True,
                'url': driver.current_url,
                'title': driver.title
            }
            
        except Exception as e:
            self.logger.error(f"Refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_session_driver(self, session_id: Optional[str] = None):
        """Get driver for session"""
        if session_id is None:
            session_id = self.active_session
        
        if not session_id or session_id not in self.sessions:
            return None
        
        return self.sessions[session_id].driver
    
    # Element Interaction
    async def find_element(self, selector: str, by: str = 'css', session_id: Optional[str] = None) -> Dict[str, Any]:
        """Find element on page"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            # Map selector types
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'name': By.NAME,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
                'link_text': By.LINK_TEXT,
                'partial_link_text': By.PARTIAL_LINK_TEXT
            }
            
            if by not in by_map:
                return {
                    'success': False,
                    'error': f'Invalid selector type: {by}',
                    'available_types': list(by_map.keys())
                }
            
            # Find element
            element = driver.find_element(by_map[by], selector)
            
            # Get element information
            element_info = WebElement(
                tag_name=element.tag_name,
                text=element.text,
                attributes={},
                location=element.location,
                size=element.size,
                is_displayed=element.is_displayed(),
                is_enabled=element.is_enabled(),
                is_selected=element.is_selected()
            )
            
            # Get attributes
            common_attributes = ['id', 'class', 'name', 'value', 'href', 'src', 'alt', 'title']
            for attr in common_attributes:
                try:
                    value = element.get_attribute(attr)
                    if value:
                        element_info.attributes[attr] = value
                except:
                    pass
            
            return {
                'success': True,
                'element': {
                    'tag_name': element_info.tag_name,
                    'text': element_info.text,
                    'attributes': element_info.attributes,
                    'location': element_info.location,
                    'size': element_info.size,
                    'is_displayed': element_info.is_displayed,
                    'is_enabled': element_info.is_enabled,
                    'is_selected': element_info.is_selected
                }
            }
            
        except NoSuchElementException:
            return {
                'success': False,
                'error': f'Element not found: {selector}'
            }
        except Exception as e:
            self.logger.error(f"Find element error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def find_elements(self, selector: str, by: str = 'css', session_id: Optional[str] = None) -> Dict[str, Any]:
        """Find multiple elements on page"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            # Map selector types
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'name': By.NAME,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
                'link_text': By.LINK_TEXT,
                'partial_link_text': By.PARTIAL_LINK_TEXT
            }
            
            if by not in by_map:
                return {
                    'success': False,
                    'error': f'Invalid selector type: {by}',
                    'available_types': list(by_map.keys())
                }
            
            # Find elements
            elements = driver.find_elements(by_map[by], selector)
            
            # Get element information
            elements_info = []
            for element in elements:
                try:
                    element_info = {
                        'tag_name': element.tag_name,
                        'text': element.text,
                        'attributes': {},
                        'location': element.location,
                        'size': element.size,
                        'is_displayed': element.is_displayed(),
                        'is_enabled': element.is_enabled(),
                        'is_selected': element.is_selected()
                    }
                    
                    # Get attributes
                    common_attributes = ['id', 'class', 'name', 'value', 'href', 'src', 'alt', 'title']
                    for attr in common_attributes:
                        try:
                            value = element.get_attribute(attr)
                            if value:
                                element_info['attributes'][attr] = value
                        except:
                            pass
                    
                    elements_info.append(element_info)
                    
                except Exception:
                    continue
            
            return {
                'success': True,
                'elements': elements_info,
                'count': len(elements_info)
            }
            
        except Exception as e:
            self.logger.error(f"Find elements error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def click_element(self, selector: str, by: str = 'css', session_id: Optional[str] = None) -> Dict[str, Any]:
        """Click element"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            # Map selector types
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'name': By.NAME,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
                'link_text': By.LINK_TEXT,
                'partial_link_text': By.PARTIAL_LINK_TEXT
            }
            
            if by not in by_map:
                return {
                    'success': False,
                    'error': f'Invalid selector type: {by}'
                }
            
            # Wait for element to be clickable
            wait = WebDriverWait(driver, self.default_timeout)
            element = wait.until(EC.element_to_be_clickable((by_map[by], selector)))
            
            # Click element
            element.click()
            
            self.operation_stats['elements_interacted'] += 1
            
            return {
                'success': True,
                'selector': selector,
                'action': 'clicked'
            }
            
        except TimeoutException:
            return {
                'success': False,
                'error': f'Element not clickable within timeout: {selector}'
            }
        except Exception as e:
            self.logger.error(f"Click element error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def type_text(self, selector: str, text: str, by: str = 'css', clear_first: bool = True, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Type text into element"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            # Map selector types
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'name': By.NAME,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME
            }
            
            if by not in by_map:
                return {
                    'success': False,
                    'error': f'Invalid selector type: {by}'
                }
            
            # Wait for element to be present
            wait = WebDriverWait(driver, self.default_timeout)
            element = wait.until(EC.presence_of_element_located((by_map[by], selector)))
            
            # Clear and type text
            if clear_first:
                element.clear()
            
            element.send_keys(text)
            
            self.operation_stats['elements_interacted'] += 1
            
            return {
                'success': True,
                'selector': selector,
                'text_length': len(text),
                'action': 'typed'
            }
            
        except TimeoutException:
            return {
                'success': False,
                'error': f'Element not found within timeout: {selector}'
            }
        except Exception as e:
            self.logger.error(f"Type text error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_element_text(self, selector: str, by: str = 'css', session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get text from element"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            # Map selector types
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'name': By.NAME,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
                'link_text': By.LINK_TEXT,
                'partial_link_text': By.PARTIAL_LINK_TEXT
            }
            
            if by not in by_map:
                return {
                    'success': False,
                    'error': f'Invalid selector type: {by}'
                }
            
            # Find element and get text
            element = driver.find_element(by_map[by], selector)
            text = element.text
            
            return {
                'success': True,
                'selector': selector,
                'text': text,
                'text_length': len(text)
            }
            
        except NoSuchElementException:
            return {
                'success': False,
                'error': f'Element not found: {selector}'
            }
        except Exception as e:
            self.logger.error(f"Get element text error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_element_attribute(self, selector: str, attribute: str, by: str = 'css', session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get attribute from element"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            # Map selector types
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'name': By.NAME,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME
            }
            
            if by not in by_map:
                return {
                    'success': False,
                    'error': f'Invalid selector type: {by}'
                }
            
            # Find element and get attribute
            element = driver.find_element(by_map[by], selector)
            value = element.get_attribute(attribute)
            
            return {
                'success': True,
                'selector': selector,
                'attribute': attribute,
                'value': value
            }
            
        except NoSuchElementException:
            return {
                'success': False,
                'error': f'Element not found: {selector}'
            }
        except Exception as e:
            self.logger.error(f"Get element attribute error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Page Information
    async def get_page_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current page information"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            start_time = time.time()
            
            # Get basic page info
            url = driver.current_url
            title = driver.title
            source = driver.page_source
            
            # Parse HTML for additional info
            if BS4_AVAILABLE:
                soup = BeautifulSoup(source, 'html.parser')
                
                elements_count = len(soup.find_all())
                images_count = len(soup.find_all('img'))
                links_count = len(soup.find_all('a'))
                forms_count = len(soup.find_all('form'))
            else:
                elements_count = 0
                images_count = 0
                links_count = 0
                forms_count = 0
            
            load_time = time.time() - start_time
            
            page_info = PageInfo(
                url=url,
                title=title,
                source_length=len(source),
                load_time=load_time,
                elements_count=elements_count,
                images_count=images_count,
                links_count=links_count,
                forms_count=forms_count
            )
            
            return {
                'success': True,
                'page_info': {
                    'url': page_info.url,
                    'title': page_info.title,
                    'source_length': page_info.source_length,
                    'load_time': page_info.load_time,
                    'elements_count': page_info.elements_count,
                    'images_count': page_info.images_count,
                    'links_count': page_info.links_count,
                    'forms_count': page_info.forms_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"Get page info error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_page_source(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get page source HTML"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            source = driver.page_source
            
            return {
                'success': True,
                'source': source,
                'length': len(source)
            }
            
        except Exception as e:
            self.logger.error(f"Get page source error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def take_screenshot(self, filename: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Take screenshot of current page"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'screenshot_{timestamp}.png'
            
            # Ensure screenshots directory exists
            screenshots_dir = Path.cwd() / 'screenshots'
            screenshots_dir.mkdir(exist_ok=True)
            
            screenshot_path = screenshots_dir / filename
            
            # Take screenshot
            driver.save_screenshot(str(screenshot_path))
            
            self.operation_stats['screenshots_taken'] += 1
            
            return {
                'success': True,
                'filename': filename,
                'path': str(screenshot_path),
                'size': screenshot_path.stat().st_size
            }
            
        except Exception as e:
            self.logger.error(f"Take screenshot error: {e}")
            return {'success': False, 'error': str(e)}
    
    # JavaScript Execution
    async def execute_script(self, script: str, *args, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute JavaScript in browser"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            result = driver.execute_script(script, *args)
            
            return {
                'success': True,
                'result': result,
                'script_length': len(script)
            }
            
        except Exception as e:
            self.logger.error(f"Execute script error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Cookie Management
    async def get_cookies(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get all cookies"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            cookies = driver.get_cookies()
            
            return {
                'success': True,
                'cookies': cookies,
                'count': len(cookies)
            }
            
        except Exception as e:
            self.logger.error(f"Get cookies error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def add_cookie(self, cookie_dict: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add cookie"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            driver.add_cookie(cookie_dict)
            
            return {
                'success': True,
                'cookie': cookie_dict
            }
            
        except Exception as e:
            self.logger.error(f"Add cookie error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def delete_cookie(self, name: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete cookie by name"""
        try:
            driver = self._get_session_driver(session_id)
            if not driver:
                return {'success': False, 'error': 'No active session'}
            
            driver.delete_cookie(name)
            
            return {
                'success': True,
                'deleted_cookie': name
            }
            
        except Exception as e:
            self.logger.error(f"Delete cookie error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Session Information
    async def get_sessions(self) -> Dict[str, Any]:
        """Get all browser sessions"""
        try:
            sessions_info = []
            
            for session_id, session in self.sessions.items():
                try:
                    # Update session info
                    session.current_url = session.driver.current_url
                    session.title = session.driver.title
                    session.window_handles = session.driver.window_handles
                    
                    sessions_info.append({
                        'session_id': session_id,
                        'browser_type': session.browser_type,
                        'start_time': session.start_time.isoformat(),
                        'current_url': session.current_url,
                        'title': session.title,
                        'window_count': len(session.window_handles),
                        'is_active': session_id == self.active_session
                    })
                    
                except Exception:
                    # Session might be closed
                    sessions_info.append({
                        'session_id': session_id,
                        'browser_type': session.browser_type,
                        'start_time': session.start_time.isoformat(),
                        'status': 'closed',
                        'is_active': session_id == self.active_session
                    })
            
            return {
                'success': True,
                'sessions': sessions_info,
                'active_session': self.active_session,
                'total_sessions': len(sessions_info)
            }
            
        except Exception as e:
            self.logger.error(f"Get sessions error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get browser controller performance statistics"""
        return {
            'operation_stats': self.operation_stats.copy(),
            'session_stats': {
                'total_sessions': len(self.sessions),
                'active_session': self.active_session,
                'available_browsers': self.available_browsers
            },
            'capabilities': {
                'selenium_available': SELENIUM_AVAILABLE,
                'bs4_available': BS4_AVAILABLE
            }
        }
    
    async def cleanup(self):
        """Cleanup browser controller"""
        try:
            self.logger.info("Cleaning up Browser Controller...")
            
            # Close all sessions
            for session_id in list(self.sessions.keys()):
                await self.close_session(session_id)
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            self.logger.info("Browser Controller cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Browser Controller cleanup error: {e}")
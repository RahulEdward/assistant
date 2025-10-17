#!/usr/bin/env python3
"""
Enhanced WindowManager Test Suite with Result Saving
Comprehensive testing with structured result storage
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from automation.window_manager import WindowManager

class TestResultCollector:
    def __init__(self):
        self.results = {
            'test_session': {
                'timestamp': datetime.now().isoformat(),
                'test_suite': 'WindowManager Comprehensive Test',
                'version': '1.0',
                'environment': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'working_directory': os.getcwd()
                }
            },
            'tests': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': []
            }
        }
    
    def add_test_result(self, test_name, status, details=None, error=None):
        """Add a test result to the collection"""
        test_result = {
            'test_name': test_name,
            'status': status,  # 'PASS', 'FAIL', 'ERROR'
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
            'error': error
        }
        
        self.results['tests'].append(test_result)
        self.results['summary']['total_tests'] += 1
        
        if status == 'PASS':
            self.results['summary']['passed'] += 1
        elif status == 'FAIL':
            self.results['summary']['failed'] += 1
        elif status == 'ERROR':
            self.results['summary']['failed'] += 1
            if error:
                self.results['summary']['errors'].append(f"{test_name}: {error}")
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        return filepath

async def run_comprehensive_tests():
    """Run comprehensive WindowManager tests and collect results"""
    collector = TestResultCollector()
    
    print("=== Enhanced WindowManager Test Suite ===")
    
    # Test 1: Instantiation
    print("1. Testing instantiation...")
    try:
        wm = WindowManager()
        collector.add_test_result("instantiation", "PASS", {"message": "WindowManager created successfully"})
        print("   ✓ WindowManager created successfully")
    except Exception as e:
        collector.add_test_result("instantiation", "ERROR", error=str(e))
        print(f"   ✗ Error: {e}")
        return collector
    
    # Test 2: Initialization
    print("2. Testing initialization...")
    try:
        await wm.initialize()
        collector.add_test_result("initialization", "PASS", {"message": "WindowManager initialized successfully"})
        print("   ✓ WindowManager initialized successfully")
    except Exception as e:
        collector.add_test_result("initialization", "ERROR", error=str(e))
        print(f"   ✗ Error: {e}")
    
    # Test 3: Window enumeration
    print("3. Testing window enumeration...")
    try:
        windows_result = await wm.get_all_windows()
        if windows_result['success']:
            window_count = windows_result['data']['count']
            windows = windows_result['data']['windows']
            
            details = {
                'window_count': window_count,
                'sample_windows': []
            }
            
            # Add sample window details
            sample_windows = windows[:3] if isinstance(windows, list) else []
            for i, window in enumerate(sample_windows):  # First 3 windows
                details['sample_windows'].append({
                    'title': window['title'][:50] + '...' if len(window['title']) > 50 else window['title'],
                    'pid': window['process_id'],
                    'class_name': window['class_name']
                })
            
            collector.add_test_result("window_enumeration", "PASS", details)
            print(f"   ✓ Found {window_count} windows")
            
            if windows:
                print("   Sample windows:")
                sample_windows = windows[:3] if isinstance(windows, list) else []
                for window in sample_windows:
                    title = window['title'][:50] + '...' if len(window['title']) > 50 else window['title']
                    print(f"     - {title} (PID: {window['process_id']})")
        else:
            collector.add_test_result("window_enumeration", "FAIL", error=windows_result.get('error', 'Unknown error'))
            print(f"   ✗ Error: {windows_result.get('error', 'Unknown error')}")
    except Exception as e:
        collector.add_test_result("window_enumeration", "ERROR", error=str(e))
        print(f"   ✗ Exception: {e}")
    
    # Test 4: Monitor detection
    print("4. Testing monitor detection...")
    try:
        monitors_result = await wm.get_monitors()
        if monitors_result['success']:
            monitor_count = monitors_result['data']['count']
            monitors = monitors_result['data']['monitors']
            
            details = {
                'monitor_count': monitor_count,
                'monitors': []
            }
            
            for i, monitor in enumerate(monitors):
                details['monitors'].append({
                    'index': i + 1,
                    'width': monitor['width'],
                    'height': monitor['height'],
                    'dpi': monitor['dpi'],
                    'is_primary': monitor['is_primary']
                })
            
            collector.add_test_result("monitor_detection", "PASS", details)
            print(f"   ✓ Found {monitor_count} monitors")
            
            for i, monitor in enumerate(monitors):
                print(f"     Monitor {i+1}: {monitor['width']}x{monitor['height']} @ {monitor['dpi']} DPI")
        else:
            collector.add_test_result("monitor_detection", "FAIL", error=monitors_result.get('error', 'Unknown error'))
            print(f"   ✗ Error: {monitors_result.get('error', 'Unknown error')}")
    except Exception as e:
        collector.add_test_result("monitor_detection", "ERROR", error=str(e))
        print(f"   ✗ Exception: {e}")
    
    # Test 5: Primary monitor detection
    print("5. Testing primary monitor detection...")
    try:
        primary_result = await wm.get_primary_monitor()
        if primary_result['success']:
            monitor = primary_result['data']['monitor']
            details = {
                'width': monitor['width'],
                'height': monitor['height'],
                'dpi': monitor['dpi'],
                'device_name': monitor['device_name']
            }
            
            collector.add_test_result("primary_monitor_detection", "PASS", details)
            print(f"   ✓ Primary monitor: {monitor['width']}x{monitor['height']} @ {monitor['dpi']} DPI")
        else:
            collector.add_test_result("primary_monitor_detection", "FAIL", error=primary_result.get('error', 'Unknown error'))
            print(f"   ✗ Error: {primary_result.get('error', 'Unknown error')}")
    except Exception as e:
        collector.add_test_result("primary_monitor_detection", "ERROR", error=str(e))
        print(f"   ✗ Exception: {e}")
    
    # Test 6: Window search
    print("6. Testing window search...")
    try:
        search_result = await wm.find_windows(title="explorer", partial_match=True)
        if search_result['success']:
            found_count = search_result['data']['count']
            details = {
                'search_term': 'explorer',
                'found_count': found_count,
                'partial_match': True
            }
            
            collector.add_test_result("window_search", "PASS", details)
            print(f"   ✓ Found {found_count} windows matching \"explorer\"")
        else:
            collector.add_test_result("window_search", "FAIL", error=search_result.get('error', 'Unknown error'))
            print(f"   ✗ Error: {search_result.get('error', 'Unknown error')}")
    except Exception as e:
        collector.add_test_result("window_search", "ERROR", error=str(e))
        print(f"   ✗ Exception: {e}")
    
    # Test 7: Performance statistics
    print("7. Testing performance statistics...")
    try:
        stats_result = await wm.get_performance_stats()
        if stats_result['success']:
            stats = stats_result['data']['stats']
            details = {
                'total_operations': stats.get('total_operations', 0),
                'total_errors': stats.get('total_errors', 0),
                'windows_enumerated': stats.get('windows_enumerated', 0),
                'windows_manipulated': stats.get('windows_manipulated', 0)
            }
            
            collector.add_test_result("performance_statistics", "PASS", details)
            print(f"   ✓ Performance stats - Operations: {details['total_operations']}, Errors: {details['total_errors']}")
        else:
            collector.add_test_result("performance_statistics", "FAIL", error=stats_result.get('error', 'Unknown error'))
            print(f"   ✗ Error: {stats_result.get('error', 'Unknown error')}")
    except Exception as e:
        collector.add_test_result("performance_statistics", "ERROR", error=str(e))
        print(f"   ✗ Exception: {e}")
    
    # Test 8: Cleanup
    print("8. Testing cleanup...")
    try:
        await wm.cleanup()
        collector.add_test_result("cleanup", "PASS", {"message": "Cleanup completed successfully"})
        print("   ✓ Cleanup completed successfully")
    except Exception as e:
        collector.add_test_result("cleanup", "ERROR", error=str(e))
        print(f"   ✗ Error: {e}")
    
    print("\n=== All Tests Completed ===")
    
    return collector

def main():
    """Main function to run tests and save results"""
    try:
        # Run the tests
        collector = asyncio.run(run_comprehensive_tests())
        
        # Save results
        print("\n=== Saving Test Results ===")
        filepath = collector.save_results()
        print(f"✓ Test results saved to: {filepath}")
        
        # Print summary
        summary = collector.results['summary']
        print(f"\n=== Test Summary ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {(summary['passed']/summary['total_tests']*100):.1f}%")
        
        if summary['errors']:
            print(f"\nErrors encountered:")
            for error in summary['errors']:
                print(f"  - {error}")
        
        return 0 if summary['failed'] == 0 else 1
        
    except Exception as e:
        print(f"Critical error during test execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
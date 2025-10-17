#!/usr/bin/env python3
"""
WindowManager Test Suite
Comprehensive testing of the WindowManager functionality
"""

import asyncio
import sys
import os
sys.path.append('src')

from automation.window_manager import WindowManager

async def test_window_manager():
    print('=== WindowManager Test Suite ===')
    
    # Test 1: Instantiation
    print('1. Testing instantiation...')
    try:
        wm = WindowManager()
        print('   ✓ WindowManager created successfully')
    except Exception as e:
        print(f'   ✗ Error creating WindowManager: {e}')
        return
    
    # Test 2: Initialization
    print('2. Testing initialization...')
    try:
        await wm.initialize()
        print('   ✓ WindowManager initialized successfully')
    except Exception as e:
        print(f'   ✗ Error initializing WindowManager: {e}')
        return
    
    # Test 3: Get all windows
    print('3. Testing window enumeration...')
    try:
        windows_result = await wm.get_all_windows()
        if windows_result['success']:
            window_count = len(windows_result['data']['windows'])
            print(f'   ✓ Found {window_count} windows')
            
            # Show first few windows as examples
            if window_count > 0:
                print('   Sample windows:')
                for i, (hwnd, window) in enumerate(list(windows_result['data']['windows'].items())[:3]):
                    print(f'     - {window["title"][:50]}... (PID: {window["process_id"]})')
        else:
            print(f'   ✗ Error: {windows_result["error"]}')
    except Exception as e:
        print(f'   ✗ Exception during window enumeration: {e}')
    
    # Test 4: Get monitors
    print('4. Testing monitor detection...')
    try:
        monitors_result = await wm.get_monitors()
        if monitors_result['success']:
            monitor_count = len(monitors_result['data']['monitors'])
            print(f'   ✓ Found {monitor_count} monitors')
            
            # Show monitor details
            for i, monitor in enumerate(monitors_result['data']['monitors']):
                print(f'     Monitor {i+1}: {monitor["width"]}x{monitor["height"]} @ {monitor["dpi"]} DPI')
        else:
            print(f'   ✗ Error: {monitors_result["error"]}')
    except Exception as e:
        print(f'   ✗ Exception during monitor detection: {e}')
    
    # Test 5: Get primary monitor
    print('5. Testing primary monitor detection...')
    try:
        primary_result = await wm.get_primary_monitor()
        if primary_result['success']:
            monitor = primary_result['data']['monitor']
            print(f'   ✓ Primary monitor: {monitor["width"]}x{monitor["height"]} @ {monitor["dpi"]} DPI')
        else:
            print(f'   ✗ Error: {primary_result["error"]}')
    except Exception as e:
        print(f'   ✗ Exception during primary monitor detection: {e}')
    
    # Test 6: Find specific windows
    print('6. Testing window search...')
    try:
        search_result = await wm.find_windows(title="explorer", partial_match=True)
        if search_result['success']:
            found_count = len(search_result['data']['windows'])
            print(f'   ✓ Found {found_count} windows matching "explorer"')
        else:
            print(f'   ✗ Error: {search_result["error"]}')
    except Exception as e:
        print(f'   ✗ Exception during window search: {e}')
    
    # Test 7: Performance stats
    print('7. Testing performance statistics...')
    try:
        stats_result = await wm.get_performance_stats()
        if stats_result['success']:
            stats = stats_result['data']['stats']
            print(f'   ✓ Performance stats - Operations: {stats["total_operations"]}, Errors: {stats["total_errors"]}')
        else:
            print(f'   ✗ Error: {stats_result["error"]}')
    except Exception as e:
        print(f'   ✗ Exception during performance stats: {e}')
    
    # Test 8: Cleanup
    print('8. Testing cleanup...')
    try:
        await wm.cleanup()
        print('   ✓ Cleanup completed successfully')
    except Exception as e:
        print(f'   ✗ Error during cleanup: {e}')
    
    print('\n=== All Tests Completed ===')

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_window_manager())
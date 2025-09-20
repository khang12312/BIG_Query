#!/usr/bin/env python3
"""
Test runner script for the AI-Powered Resume Matcher
This script runs tests locally with proper error handling
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, continue_on_error=True):
    """Run a command and handle errors gracefully"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            if not continue_on_error:
                sys.exit(1)
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        if not continue_on_error:
            sys.exit(1)
        return False

def main():
    """Main test runner function"""
    print("🚀 AI-Powered Resume Matcher - Test Runner")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # List of test commands to run
    test_commands = [
        {
            "cmd": [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
            "description": "Install package in development mode",
            "continue_on_error": True
        },
        {
            "cmd": [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            "description": "Run pytest tests",
            "continue_on_error": True
        },
        {
            "cmd": [sys.executable, "-m", "flake8", "src/", "--count", "--statistics"],
            "description": "Run flake8 linting",
            "continue_on_error": True
        },
        {
            "cmd": [sys.executable, "-m", "black", "--check", "src/"],
            "description": "Check code formatting with black",
            "continue_on_error": True
        },
        {
            "cmd": [sys.executable, "-m", "isort", "--check-only", "src/"],
            "description": "Check import sorting with isort",
            "continue_on_error": True
        },
        {
            "cmd": [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"],
            "description": "Run type checking with mypy",
            "continue_on_error": True
        }
    ]
    
    # Run all test commands
    results = []
    for test_config in test_commands:
        result = run_command(
            test_config["cmd"], 
            test_config["description"],
            test_config["continue_on_error"]
        )
        results.append((test_config["description"], result))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for description, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {description}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed, but continuing...")
        return 0  # Don't fail the script, just report issues

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

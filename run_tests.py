#!/usr/bin/env python
"""Run tests for Instant Data Connector."""

import sys
import subprocess


def run_tests():
    """Run the test suite."""
    # Basic test run
    cmd = [sys.executable, "-m", "pytest", "-v"]
    
    # Add coverage if pytest-cov is installed
    try:
        import pytest_cov
        cmd.extend(["--cov=instant_connector", "--cov-report=term-missing"])
    except ImportError:
        print("Note: Install pytest-cov for coverage reports")
    
    # Run tests
    result = subprocess.run(cmd)
    return result.returncode


def run_specific_tests(test_module):
    """Run tests for a specific module."""
    cmd = [sys.executable, "-m", "pytest", "-v", f"tests/test_{test_module}.py"]
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        exit_code = run_specific_tests(sys.argv[1])
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
TBRGS Test Runner

This script runs the TBRGS test suite from the correct directory.
"""

import os
import sys
import subprocess

def main():
    """Run the TBRGS test suite."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the current directory to the Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print("Running TBRGS test suite...")
    print(f"Current directory: {current_dir}")
    
    # Import and run the test suite
    try:
        from app.tests.run_tests import main as run_tests
        run_tests()
    except Exception as e:
        print(f"Error running test suite: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

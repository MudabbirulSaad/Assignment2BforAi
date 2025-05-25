#!/usr/bin/env python3
"""
Run script for the TBRGS Gradio web interface.
This script adds the project root to the Python path before importing the modules.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the Gradio app
from app.gui.gradio_app import main

if __name__ == "__main__":
    main()

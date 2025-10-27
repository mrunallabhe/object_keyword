#!/usr/bin/env python3
"""
Simple startup script for the OCR & Object Detection System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import cv2
        import easyocr
        import torch
        import numpy
        from PIL import Image
        from reportlab.lib.pagesizes import A4
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['uploads', 'outputs', 'templates']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Directory '{directory}' ready")

def main():
    """Main startup function"""
    print("🚀 Starting OCR & Object Detection System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('app_integrated.py').exists():
        print("❌ app_integrated.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("\n🌐 Starting Flask application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from app_integrated import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

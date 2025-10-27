#!/usr/bin/env python3
"""
Test script to verify the OCR & Object Detection System installation
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_imports():
    """Test required package imports"""
    print("\n📦 Testing package imports...")
    
    packages = [
        ('flask', 'Flask'),
        ('cv2', 'OpenCV'),
        ('easyocr', 'EasyOCR'),
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('reportlab', 'ReportLab'),
        ('sqlite3', 'SQLite3 (built-in)'),
        ('json', 'JSON (built-in)'),
        ('hashlib', 'Hashlib (built-in)'),
        ('glob', 'Glob (built-in)'),
        ('time', 'Time (built-in)'),
        ('datetime', 'Datetime (built-in)'),
        ('pathlib', 'Pathlib (built-in)')
    ]
    
    failed_imports = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - FAILED: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def test_optional_imports():
    """Test optional package imports"""
    print("\n🔧 Testing optional packages...")
    
    optional_packages = [
        ('clip', 'CLIP (for object detection fallback)'),
        ('groundingdino', 'GroundingDINO (for advanced object detection)')
    ]
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name} - Available")
        except ImportError:
            print(f"⚠️  {name} - Not available (will use fallback)")

def test_directories():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = ['uploads', 'outputs', 'templates']
    required_files = ['app_integrated.py', 'requirements.txt', 'README.md']
    
    all_good = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory '{directory}' - OK")
        else:
            print(f"❌ Directory '{directory}' - Missing")
            all_good = False
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ File '{file}' - OK")
        else:
            print(f"❌ File '{file}' - Missing")
            all_good = False
    
    return all_good

def test_cuda():
    """Test CUDA availability"""
    print("\n🚀 Testing CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not available - cannot test CUDA")
        return False

def test_easyocr():
    """Test EasyOCR initialization"""
    print("\n👁️  Testing EasyOCR...")
    
    try:
        import easyocr
        import torch
        
        # Test with CPU first to avoid GPU memory issues during testing
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR initialized successfully")
        return True
    except Exception as e:
        print(f"❌ EasyOCR initialization failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 OCR & Object Detection System - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Directory Structure", test_directories),
        ("CUDA Availability", test_cuda),
        ("EasyOCR", test_easyocr)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Test optional imports
    test_optional_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("🚀 Run 'python run_app.py' to start the application.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("💡 Try running: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

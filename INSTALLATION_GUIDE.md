# 🔧 Installation Guide for Python 3.12

## The Problem
You're encountering a compatibility issue with Python 3.12 and certain packages. This is a known issue with the `pkgutil.ImpImporter` attribute that was removed in Python 3.12.

## 🚀 Solution: Step-by-Step Installation

### Step 1: Update pip first
```bash
python.exe -m pip install --upgrade pip
```

### Step 2: Install packages individually (Recommended)
Instead of installing all packages at once, install them individually to identify any problematic packages:

```bash
# Core packages first
pip install Flask>=3.0.0
pip install Werkzeug>=3.0.0
pip install opencv-python-headless
pip install Pillow
pip install numpy

# OCR
pip install easyocr

# PyTorch (CPU version for better compatibility)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PDF generation
pip install reportlab

# Text processing
pip install ftfy regex tqdm

# Transformers (CLIP alternative)
pip install transformers
pip install sentence-transformers

# Additional utilities
pip install pathlib2
```

### Step 3: Test the installation
```bash
python test_installation.py
```

### Step 4: If you still have issues, try the minimal version
If some packages still fail, you can run the app with minimal dependencies:

```bash
# Minimal installation
pip install Flask opencv-python-headless Pillow numpy easyocr torch torchvision reportlab
```

## 🔄 Alternative: Use Python 3.11

If you continue to have issues with Python 3.12, consider using Python 3.11:

1. **Download Python 3.11** from [python.org](https://www.python.org/downloads/)
2. **Create a virtual environment** with Python 3.11:
   ```bash
   python3.11 -m venv venv311
   venv311\Scripts\activate  # Windows
   # or
   source venv311/bin/activate  # macOS/Linux
   ```
3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Troubleshooting Specific Errors

### Error: `pkgutil.ImpImporter`
This is a Python 3.12 compatibility issue. Solutions:
1. Use the individual package installation method above
2. Use Python 3.11 instead
3. Wait for package updates that support Python 3.12

### Error: `torch` installation fails
Try the CPU-only version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Error: `easyocr` installation fails
Try installing dependencies first:
```bash
pip install torch torchvision
pip install easyocr
```

### Error: `transformers` installation fails
Try installing without optional dependencies:
```bash
pip install transformers --no-deps
pip install tokenizers safetensors
```

## 🎯 Quick Test Installation

Create a simple test script to verify your installation:

```python
# test_simple.py
try:
    import flask
    print("✅ Flask OK")
except ImportError as e:
    print(f"❌ Flask: {e}")

try:
    import cv2
    print("✅ OpenCV OK")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

try:
    import easyocr
    print("✅ EasyOCR OK")
except ImportError as e:
    print(f"❌ EasyOCR: {e}")

try:
    import torch
    print("✅ PyTorch OK")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    from reportlab.lib.pagesizes import A4
    print("✅ ReportLab OK")
except ImportError as e:
    print(f"❌ ReportLab: {e}")
```

Run it with:
```bash
python test_simple.py
```

## 🚀 Once Installation is Complete

1. **Test the installation**:
   ```bash
   python test_installation.py
   ```

2. **Start the application**:
   ```bash
   python run_app.py
   ```

3. **Open your browser** to `http://localhost:5000`

## 📞 Still Having Issues?

If you're still encountering problems:

1. **Check Python version**: `python --version`
2. **Check pip version**: `pip --version`
3. **Try creating a fresh virtual environment**
4. **Consider using Python 3.11** for better compatibility

The application will work with minimal dependencies - the object detection features are optional and the app will still function for OCR and PDF generation even if some packages fail to install.

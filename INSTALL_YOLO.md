# YOLO Installation Guide

This guide will help you install and run the YOLO-based object detection system for crime scene evidence analysis.

## Quick Start

### Option 1: Simple Installation (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app_yolo.py

# 3. Open browser to http://localhost:5000
```

### Option 2: Anaconda Environment (Recommended for Research)

```bash
# 1. Create environment
conda create -n yolo_env python=3.9
conda activate yolo_env

# 2. Install PyTorch (with GPU support if available)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Run application
python app_yolo.py
```

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Processor | Intel Pentium 2.0GHz | Intel i5/i7 or AMD equivalent |
| RAM | 4GB | 8GB or more |
| Storage | 2GB free | 5GB free (for models) |
| GPU | Not required | NVIDIA GPU with 2GB+ VRAM |
| Display | 1024×768 | 1920×1080+ |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8+ | Programming language |
| Flask | 3.0+ | Web framework |
| PyTorch | 2.0+ | Deep learning framework |
| OpenCV | 4.0+ | Image processing |
| EasyOCR | 1.7+ | OCR text extraction |

## Step-by-Step Installation

### Windows

#### 1. Install Python
```powershell
# Download from https://www.python.org/downloads/
# Make sure to check "Add Python to PATH"
```

#### 2. Install Git (Optional)
```powershell
# Download from https://git-scm.com/download/win
```

#### 3. Open Command Prompt or PowerShell
```powershell
# Navigate to project directory
cd C:\Users\HP\OneDrive\Desktop\VII_Sem
```

#### 4. Install Dependencies
```powershell
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Run Application
```powershell
python app_yolo.py
```

### Linux/Ubuntu

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# 3. Install system dependencies for YOLO
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 -y

# 4. Navigate to project
cd ~/VII_Sem

# 5. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 6. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 7. Run application
python app_yolo.py
```

### macOS

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python@3.9

# 3. Navigate to project
cd ~/VII_Sem

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Run application
python app_yolo.py
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU with CUDA

1. **Check GPU Compatibility**
```bash
# Check NVIDIA drivers
nvidia-smi
```

2. **Install CUDA Toolkit** (if not installed)
```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Follow installation instructions for your OS
```

3. **Install PyTorch with CUDA**
```bash
# Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Linux
pip install torch torchvision torchaudio

# Mac (Apple Silicon)
pip install torch torchvision torchaudio
```

4. **Verify CUDA Installation**
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Print GPU name
```

## YOLO Model Download

### Automatic Download (Recommended)

The application will automatically download YOLO weights on first run:

```bash
# First run downloads yolov8n.pt (~6MB)
python app_yolo.py
```

### Manual Download

If automatic download fails:

1. **Download YOLOv8n (nano)**
```bash
# Create weights directory
mkdir -p weights

# Download weights
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt -O weights/yolov8n.pt
```

2. **Download YOLOv5s (small)** (Alternative)
```bash
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O weights/yolov5s.pt
```

3. **Update detector initialization**
```python
# In app_yolo.py or your script
yolo_detector = YOLODetector(weights_path='weights/yolov8n.pt')
```

## Verification

### Test YOLO Installation

```python
# Create test script: test_yolo.py
from yolo_detector import YOLODetector
import cv2
import numpy as np

# Initialize
detector = YOLODetector(conf_threshold=0.25)

# Create test image
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
cv2.imwrite('test_image.jpg', img)

# Test detection
detections = detector.detect('test_image.jpg')
print(f"✅ YOLO working! Found {len(detections)} objects")

# Clean up
import os
os.remove('test_image.jpg')
```

### Test Flask Application

```bash
# Start server
python app_yolo.py

# In another terminal, test API
curl http://localhost:5000/status
# Should return: {"status": "running", "device": "cuda", ...}
```

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:**
```bash
pip install ultralytics
```

### Issue 2: "CUDA out of memory"

**Solution:**
```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or use smaller batch size
yolo_detector = YOLODetector(img_size=416)  # Smaller input
```

### Issue 3: "Permission denied" errors

**Solution:**
```bash
# Linux/macOS
sudo chmod +x app_yolo.py
chmod -R 755 uploads outputs

# Windows
# Run PowerShell as Administrator
```

### Issue 4: Model download fails

**Solution:**
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt

# Or use smaller model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Issue 5: "OpenCV not found"

**Solution:**
```bash
pip install opencv-python-headless
```

## Performance Optimization

### For Slow Systems

```python
# Use smaller model and input size
yolo_detector = YOLODetector(
    conf_threshold=0.5,  # Higher threshold = faster
    img_size=416  # Smaller input
)
```

### For Fast GPU Systems

```python
# Use larger model for better accuracy
detector = YOLODetector(
    conf_threshold=0.25,
    img_size=640  # Full resolution
)
```

## Production Deployment

### Using Gunicorn (Linux)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_yolo:app
```

### Using Waitress (Windows)

```bash
# Install Waitress
pip install waitress

# Create run script: run_production.py
from waitress import serve
from app_yolo import app
serve(app, host="0.0.0.0", port=5000)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app_yolo.py"]
```

```bash
# Build and run
docker build -t yolo-detection .
docker run -p 5000:5000 yolo-detection
```

## Next Steps

1. ✅ Installation complete
2. 🚀 Run `python app_yolo.py`
3. 🌐 Open `http://localhost:5000`
4. 📤 Upload test images
5. 📊 Process a case
6. 📥 Download generated reports

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error logs
3. Verify all dependencies are installed
4. Test with a simple image first

## References

- YOLO Paper: https://arxiv.org/abs/1506.02640
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- COCO Dataset: https://cocodataset.org/


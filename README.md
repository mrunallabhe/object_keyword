# YOLO Object Detection System for Crime Scene Evidence Analysis

A comprehensive Flask-based web application that implements **YOLO (You Only Look Once)** object detection algorithm, combined with OCR for automated crime scene evidence analysis and report generation.

Based on the research paper: "YOLO: Real-Time Object Detection" by Redmon et al., 2016

## 🚀 Features

- **YOLO Object Detection**: Real-time object detection using YOLOv5/YOLOv8 (45+ FPS)
- **80 Object Classes**: Detects 80 common objects from COCO dataset (person, vehicle, weapon, etc.)
- **Advanced OCR**: Text extraction from images with preprocessing and caching
- **License Plate Recognition**: Specialized detection and reading of license plates
- **PDF Report Generation**: Automated case reports with annotated images and extracted data
- **Web Interface**: Modern, responsive web UI for easy interaction
- **Case Management**: SQLite database for evidence and detection tracking
- **Batch Processing**: Process multiple images simultaneously
- **GPU Acceleration**: Automatic GPU detection and utilization

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- At least 4GB RAM
- 2GB free disk space

### Python Dependencies
All dependencies are listed in `requirements.txt`. Key packages include:
- Flask (web framework)
- OpenCV (image processing)
- EasyOCR (text recognition)
- PyTorch (deep learning framework)
- Ultralytics YOLO (object detection)
- ReportLab (PDF generation)

**New**: YOLO implementation requires `ultralytics` package for object detection.

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd ocr-object-detection-system

# Or download and extract the files
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation

Test YOLO installation:
```bash
python yolo_detector.py
```

Expected output:
```
🔍 YOLO Object Detector - Test Mode
==================================================
✅ Loaded YOLOv8n (nano) pre-trained model
📸 Processing: uploads/test_image.jpg
✅ Found 3 objects:
  - person: 0.85 at (100, 200, 300, 400)
  - car: 0.78 at (50, 150, 250, 300)
  - backpack: 0.65 at (400, 100, 500, 250)
```

## 🚀 Usage

### 1. Start the YOLO Application
```bash
# Windows
start_yolo.bat

# Linux/macOS
python app_yolo.py

# Or use the simple version
python app_simple.py
```

The application will start on `http://localhost:5000`

### 2. Web Interface Usage

#### Upload Images
1. Open your web browser and go to `http://localhost:5000`
2. Click the upload area or drag & drop images
3. Supported formats: JPG, JPEG, PNG, BMP, TIFF
4. Images are automatically processed and cached

#### Process a Case
1. Enter a case name (e.g., "Case_001")
2. Specify keywords to detect (comma-separated):
   - `gun, knife, weapon`
   - `license plate, car, vehicle`
   - `person, face, suspect`
3. Choose whether to generate a PDF report
4. Click "Process Case"

### 3. API Endpoints

#### Upload Single Image
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/upload
```

#### Process Case (JSON API)
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "case_name": "Case_001",
    "keywords": ["gun", "knife", "license plate"],
    "generate_pdf": true
  }' \
  http://localhost:5000/process_case
```

## 📁 Project Structure

```
├── app_yolo.py              # YOLO-based Flask application (RECOMMENDED)
├── app_integrated.py        # Integrated Flask application
├── app_simple.py           # Simple version for testing
├── yolo_detector.py        # YOLO object detection implementation
├── templates/
│   └── index.html          # Web interface template
├── uploads/                # Uploaded images (auto-created)
├── outputs/                # Generated reports (auto-created)
├── ocr_database.db        # SQLite cache database (auto-created)
├── requirements.txt        # Python dependencies
├── YOLO_IMPLEMENTATION_GUIDE.md  # Detailed YOLO documentation
├── INSTALL_YOLO.md        # Installation guide
└── README.md              # This file
```

## 🔍 YOLO Object Detection

This implementation uses the **YOLO (You Only Look Once)** algorithm for real-time object detection:

### How YOLO Works

1. **Grid Division**: Image is divided into S × S grid cells (e.g., 19×19)
2. **Cell Prediction**: Each cell predicts K bounding boxes
3. **Confidence Scores**: Each box has class confidence scores
4. **Non-Max Suppression**: Removes duplicate detections using IoU
5. **Output**: Final object detections with bounding boxes

### Key Features

- **Real-time Performance**: 45+ FPS processing speed
- **80 Object Classes**: Detects person, vehicle, weapon, and more
- **High Accuracy**: Optimized architecture for evidence analysis
- **GPU Acceleration**: Automatic CUDA support

### Supported Objects

People, vehicles, weapons, tools, electronics, furniture, containers, and 70+ other object classes from COCO dataset.

## 🔧 Configuration

### Environment Variables
You can set these environment variables to customize the application:

```bash
# Set upload folder (default: uploads)
export UPLOAD_FOLDER="custom_uploads"

# Set output folder (default: outputs)
export OUTPUT_FOLDER="custom_outputs"

# Set maximum file size (default: 32MB)
export MAX_CONTENT_LENGTH="67108864"  # 64MB
```

### GPU Configuration
The application automatically detects CUDA availability:
- If CUDA is available, it uses GPU acceleration
- If not available, it falls back to CPU processing

## 📊 Output Formats

### PDF Reports
Generated PDF reports include:
- Case information and timestamp
- Annotated images with detection boxes
- Extracted text excerpts
- Detection results table with:
  - Object labels
  - Bounding box coordinates
  - Confidence scores
  - Additional data (e.g., license plate text)

### Detection Results
JSON format for programmatic access:
```json
{
  "label": "license plate",
  "bbox": [100, 200, 300, 250],
  "score": 0.85,
  "extra": {
    "plate_text": "ABC123"
  }
}
```

## 🔍 Object Detection Methods

### Primary: YOLO (You Only Look Once)
- **Algorithm**: YOLOv5/YOLOv8 pre-trained on COCO dataset
- **Speed**: Real-time (45+ FPS on GPU)
- **Accuracy**: High accuracy with optimized architecture
- **Classes**: 80 object categories
- **Features**: Non-max suppression, IoU calculation, keyword filtering

### Secondary: OCR Text Extraction
- EasyOCR for text recognition
- Specialized license plate reading
- Text-guided evidence matching

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
python app_integrated.py
```

#### 2. Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

#### 3. GroundingDINO Import Error
- Ensure GroundingDINO is properly installed
- Check that model weights are in the correct location
- The application will automatically fall back to CLIP

#### 4. Permission Errors
```bash
# Ensure write permissions for uploads and outputs folders
chmod 755 uploads outputs
```

### Performance Optimization

#### For CPU-only Systems
```python
# In app_integrated.py, modify:
DEVICE = "cpu"  # Force CPU usage
reader = easyocr.Reader(['en'], gpu=False)  # Disable GPU for EasyOCR
```

#### For Large Images
- Images are automatically resized if larger than 2000px
- Consider preprocessing images before upload
- Use appropriate image formats (JPEG for photos, PNG for graphics)

## 🔒 Security Considerations

- File uploads are validated for image formats only
- Filenames are sanitized using Werkzeug's secure_filename
- Maximum file size limits are enforced
- No direct file system access from web interface

## 📈 Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed for best performance
2. **Batch Processing**: Process multiple images in a single case for efficiency
3. **Caching**: The SQLite cache prevents reprocessing identical images
4. **Image Preprocessing**: Pre-resize very large images before upload
5. **Memory Management**: Close browser tabs with large images to free memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error logs in the console
3. Ensure all dependencies are properly installed
4. Verify system requirements are met

## 🔄 Updates

To update the application:
1. Backup your data (uploads, outputs, database)
2. Update dependencies: `pip install --upgrade -r requirements.txt`
3. Restart the application
4. Test with sample images

---

**Note**: This application is designed for research and educational purposes. Ensure compliance with local laws and regulations when processing sensitive images.

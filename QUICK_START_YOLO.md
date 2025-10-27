# 🚀 Quick Start Guide - YOLO Object Detection

Get started with YOLO object detection in minutes!

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- OpenCV (image processing)
- EasyOCR (text extraction)
- Ultralytics YOLO (object detection)
- PyTorch (deep learning)

### Step 2: Run the Application

**Windows:**
```bash
start_yolo.bat
```

**Linux/macOS:**
```bash
python app_yolo.py
```

**Or manually:**
```bash
python app_yolo.py
```

### Step 3: Open Your Browser

Go to: **http://localhost:5000**

## Using the Web Interface

### 1. Upload Evidence Images

- Click the upload area or drag & drop images
- Supported formats: JPG, PNG, BMP, TIFF
- Multiple images can be uploaded

### 2. Process a Case

Fill out the form:
- **Case Name**: e.g., "Crime Scene 001"
- **Investigator ID**: (optional) your ID
- **Description**: (optional) case details
- **Keywords**: Objects to detect (comma-separated)
  - Examples: `person, gun, knife, vehicle, bag`
- **Generate PDF**: Check to create report

### 3. Download Report

- Click "Process Case"
- Wait for processing (may take a few minutes)
- Download the generated PDF report

## Example Keywords

Common objects for crime scene analysis:

```
person, vehicle, car, gun, knife, bag, backpack, phone, laptop, 
document, weapon, tool, bottle, clothing, shoe, hat
```

## Command Line Usage

### Test YOLO Detection

```bash
python yolo_detector.py
```

### Process via API

```bash
curl -X POST http://localhost:5000/process_case \
  -H "Content-Type: application/json" \
  -d '{
    "case_name": "Test Case",
    "keywords": ["person", "gun"],
    "generate_pdf": true
  }'
```

## Troubleshooting

### Issue: "Module 'ultralytics' not found"

**Solution:**
```bash
pip install ultralytics
```

### Issue: "Out of memory"

**Solution:**
Use smaller model or process images individually:
```python
# In app_yolo.py
yolo_detector = YOLODetector(img_size=416)  # Smaller input
```

### Issue: Slow processing

**Solutions:**
1. Use GPU if available (automatic)
2. Reduce input image size
3. Process fewer images at once

### Issue: "Cannot download model"

**Solution:**
Download manually:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt
```

## What YOLO Detects (80 Objects)

**People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Weapons & Tools**: knife, scissors, baseball bat, baseball glove, tennis racket

**Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven

**Furniture**: chair, couch, bed, dining table, toilet

**Containers**: bottle, wine glass, cup, bowl, backpack, handbag, suitcase

**And 50+ more!**

## Next Steps

1. ✅ **Installation complete**
2. ✅ **Run application**: `python app_yolo.py`
3. ✅ **Open browser**: http://localhost:5000
4. ✅ **Upload test images**
5. ✅ **Process a case**
6. ✅ **Download PDF report**

## Need Help?

- 📖 Read `YOLO_IMPLEMENTATION_GUIDE.md` for technical details
- 📖 Read `INSTALL_YOLO.md` for detailed installation
- 📖 Read `README.md` for full documentation

## Performance Tips

1. **Use GPU**: Faster processing (automatic if available)
2. **Batch Processing**: Process multiple images together
3. **Reduce Image Size**: Resize large images before upload
4. **Specific Keywords**: Use targeted keywords for better filtering

## Example Workflow

1. **Upload** crime scene photos
2. **Enter** case details and keywords
3. **Process** case (detects objects)
4. **Review** annotated images
5. **Download** PDF report
6. **Analyze** findings

---

**Ready to start? Run `python app_yolo.py` and open http://localhost:5000**


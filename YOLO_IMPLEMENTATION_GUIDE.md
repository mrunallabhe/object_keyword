# YOLO Object Detection Implementation Guide

## Overview

This implementation provides a complete YOLO (You Only Look Once) object detection system for crime scene evidence analysis, based on the research paper by Redmon et al. (2016).

## What is YOLO?

YOLO (You Only Look Once) is a state-of-the-art algorithm for real-time object detection that was presented in a 2016 research paper by Redmon et al. The algorithm processes images at about 45 frames per second, making it much faster than previous approaches like R-CNN and Faster R-CNN.

## How YOLO Works

### Algorithm Overview

1. **Grid Division**: The YOLO algorithm divides an input image into an S × S matrix grid.
2. **Cell Prediction**: Each cell of the matrix predicts a fixed number of boundary boxes to localize objects in the image.
3. **Output Elements**: For each boundary box, YOLO outputs 5 elements:
   - `(x, y)`: coordinates for the center of the box (object localization)
   - `w`: width of the object in the image
   - `h`: height of the object in the image
   - `C`: class confidence score of the prediction

### Network Architecture

YOLO predicts a (7, 7, 30) tensor with a CNN network having:
- 24 convolutional layers
- 2 fully connected layers

The algorithm predicts several boundary boxes and outputs boxes with a class confidence score greater than 0.25 as the final prediction.

**Class Confidence Score = Box Confidence Score × Conditional Class Probability**

### Loss Function

The YOLO algorithm calculates the loss function as a sum-squared of three errors for each grid cell prediction:

1. **Localization Loss**: Error in bounding box position and size
2. **Confidence Loss**: Error in object confidence scores
3. **Classification Loss**: Squared sum error of class conditional probabilities for each class

### Non-Max Suppression (NMS)

NMS is crucial for eliminating duplicate detections:

1. Calculate IoU (Intersection over Union) for all bounding boxes
2. Remove boxes with IoU greater than a threshold (typically 0.4-0.5)
3. Keep the box with the highest confidence score
4. Repeat until all objects are uniquely identified

**IoU Formula**:
```
IoU = Intersection Area / Union Area
```

## Implementation Features

Our YOLO implementation includes:

### 1. YOLODetector Class (`yolo_detector.py`)

- **Model Loading**: Automatically downloads and loads YOLOv5 or YOLOv8 weights
- **GPU/CPU Support**: Uses GPU when available, falls back to CPU
- **Confidence Thresholding**: Configurable confidence threshold (default: 0.25)
- **NMS Implementation**: Custom NMS to eliminate duplicate detections
- **80 Class Detection**: Detects 80 common objects from COCO dataset

### 2. Key Methods

#### `detect(image_path)`
Performs object detection on an image:
- Divides image into grid cells
- Predicts bounding boxes for each cell
- Calculates class confidence scores
- Applies NMS to remove duplicates
- Returns list of detections

#### `filter_by_keywords(detections, keywords)`
Filters detections based on specific keywords:
- Matches detected objects to keywords
- Uses synonym matching for better results
- Returns filtered detection list

#### `annotate_image(image_path, detections, output_path)`
Draws bounding boxes and labels on image:
- Draws colored bounding boxes
- Adds class labels with confidence scores
- Saves annotated image

#### `_calculate_iou(box1, box2)`
Calculates Intersection over Union between two boxes:
- Used in NMS algorithm
- Returns value between 0 and 1

### 3. Flask Integration (`app_yolo.py`)

The Flask application integrates YOLO detection with:
- **OCR**: Text extraction from images using EasyOCR
- **Case Management**: Database storage for evidence and detections
- **PDF Reports**: Automated report generation
- **Web Interface**: Modern UI for case processing

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `opencv-python-headless` - Image processing
- `easyocr` - OCR functionality
- `flask` - Web framework
- `reportlab` - PDF generation

### 2. Download YOLO Weights (Automatic)

The first time you run the application, YOLO will automatically download:
- `yolov8n.pt` - YOLOv8 nano (smallest, fastest model)
- Or `yolov5s.pt` - YOLOv5 small model

Models are downloaded from:
- YOLOv8: https://github.com/ultralytics/assets/releases
- YOLOv5: https://github.com/ultralytics/yolov5/releases

### 3. Run the Application

```bash
python app_yolo.py
```

Then open your browser to: `http://localhost:5000`

## Usage

### Basic Detection

```python
from yolo_detector import YOLODetector

# Initialize detector
detector = YOLODetector(conf_threshold=0.25, nms_threshold=0.4)

# Detect objects
detections = detector.detect('image.jpg')

# Print results
for det in detections:
    print(f"{det['label']}: {det['score']:.2f} at {det['bbox']}")

# Annotate image
detector.annotate_image('image.jpg', detections, 'annotated_image.jpg')
```

### Keyword Filtering

```python
# Filter by keywords
keywords = ['gun', 'knife', 'person']
filtered = detector.filter_by_keywords(detections, keywords)
```

### Web Interface

1. Upload evidence images
2. Enter case information
3. Specify keywords to detect (e.g., "gun, knife, person")
4. Click "Process Case"
5. Download generated PDF report

## Supported Object Classes (80 classes)

### People & Animals
- person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

### Vehicles
- bicycle, car, motorcycle, airplane, bus, train, truck, boat

### Weapons & Tools
- knife, scissors, baseball bat, tennis racket, skateboard, surfboard

### Electronics
- tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster

### Furniture
- chair, couch, bed, dining table, toilet

### Containers
- bottle, wine glass, cup, bowl, backpack, handbag, suitcase

### Other
- traffic light, fire hydrant, stop sign, parking meter, bench, backpack, umbrella, tie, frisbee, book, clock, vase, scissors, teddy bear

And many more! (See `yolo_detector.py` for complete list)

## Performance

### Speed
- **YOLOv8n (nano)**: ~50-60 FPS on modern GPU, ~5-10 FPS on CPU
- **YOLOv5s (small)**: ~40-50 FPS on GPU, ~3-8 FPS on CPU

### Accuracy
- **mAP @ 0.5 IoU**: ~50-60% on COCO dataset
- **Real-time Performance**: Optimized for speed while maintaining good accuracy

### Memory Usage
- **GPU Memory**: ~500MB - 2GB (depending on model)
- **CPU RAM**: ~2-4GB during detection

## Configuration

### Confidence Threshold

Controls minimum confidence for detections (default: 0.25):

```python
detector = YOLODetector(conf_threshold=0.5)  # Higher threshold = fewer but more confident detections
```

### NMS Threshold

Controls overlap tolerance (default: 0.4):

```python
detector = YOLODetector(nms_threshold=0.5)  # Higher = more aggressive deduplication
```

### Image Size

Adjust input image size for speed vs accuracy trade-off:

```python
detector = YOLODetector(img_size=640)  # Larger = more accurate but slower
```

## Testing

```bash
# Test YOLO detector
python yolo_detector.py

# Run Flask app
python app_yolo.py

# Test API
curl -X POST http://localhost:5000/process_case \
  -H "Content-Type: application/json" \
  -d '{
    "case_name": "Test Case",
    "keywords": ["person", "gun"],
    "generate_pdf": true
  }'
```

## Troubleshooting

### Model Download Issues

If automatic download fails:
```bash
# Manually download weights
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt
# Place in project directory
```

### CUDA/GPU Issues

Force CPU usage:
```python
# In yolo_detector.py or app_yolo.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

### Memory Issues

Use smaller model:
```python
detector = YOLODetector(img_size=416)  # Smaller input size
```

## References

1. **YOLO v1 (2016)**: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You Only Look Once: Unified, Real-Time Object Detection"
2. **YOLOv8 (2023)**: Ultralytics - https://github.com/ultralytics/ultralytics
3. **YOLOv5 (2020)**: Ultralytics - https://github.com/ultralytics/yolov5
4. **COCO Dataset**: https://cocodataset.org/

## Architecture Comparison

### YOLO vs R-CNN
- **R-CNN**: Selective search (slow), ~40 seconds/image
- **YOLO**: Single pass (fast), ~0.02 seconds/image (45 FPS)

### YOLO Algorithm Steps
1. **Input**: Image (608×608 or 640×640)
2. **Grid**: Divide into 19×19 or 7×7 grid cells
3. **Prediction**: Each cell predicts B bounding boxes
4. **Output**: Class probabilities + bounding box coordinates
5. **NMS**: Remove overlapping boxes
6. **Result**: Final object detections

## Research Paper Implementation Details

Based on the original 2016 YOLO paper:

### Grid Size
- Original YOLO: 7×7 grid for 448×448 input
- Modern YOLO: 19×19 grid for 608×608 input
- Our implementation: Uses YOLOv5/YOLOv8 (19×19 grid)

### Bounding Boxes per Cell
- Original YOLO: 2 boxes per cell (B=2)
- Modern YOLO: 3 boxes per cell (B=3)

### Classes
- Original YOLO: 20 classes (PASCAL VOC)
- Modern YOLO: 80 classes (COCO dataset)
- Our implementation: 80 classes (COCO)

### Loss Components
1. **Localization**: (x, y, w, h) coordinate errors
2. **Confidence**: Object presence confidence
3. **Classification**: Class probability errors

## License

This implementation is based on open-source YOLO models and is provided for research and educational purposes.


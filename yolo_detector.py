"""
YOLO Object Detection Implementation for Crime Scene Evidence Analysis
Based on research paper: "YOLO: Real-Time Object Detection" by Redmon et al., 2016

This module implements YOLO (You Only Look Once) algorithm for real-time object detection
as described in the research paper.
"""

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import os

class YOLODetector:
    """
    YOLO Object Detection Class
    
    The YOLO algorithm divides an input image into an S x S matrix grid.
    Each cell of the matrix predicts boundary boxes to localize objects.
    
    For each boundary box, we output:
    - (x, y): coordinates for the center of the box (object localization)
    - (w, h): width and height of the object
    - C: class confidence score
    """
    
    def __init__(self, weights_path=None, conf_threshold=0.15, nms_threshold=0.4, img_size=640):
        """
        Initialize YOLO detector
        
        Args:
            weights_path: Path to YOLO weights file (.weights or .pt)
            conf_threshold: Confidence threshold (default: 0.15 for better detection)
            nms_threshold: Non-max suppression threshold
            img_size: Input image size for the model
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.img_size = img_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # COCO dataset class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.colors = self._generate_colors()
        
        # Initialize model
        self._load_model(weights_path)
    
    def _generate_colors(self):
        """Generate distinct colors for each class"""
        np.random.seed(42)
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
        return colors
    
    def _load_model(self, weights_path):
        """Load YOLO model"""
        try:
            # Try to load YOLOv5 or YOLOv8
            if weights_path and os.path.exists(weights_path):
                if weights_path.endswith('.pt'):
                    # Try loading with ultralytics YOLOv8
                    try:
                        from ultralytics import YOLO
                        self.model = YOLO(weights_path)
                        self.model_type = 'ultralytics'
                        print(f"✅ Loaded YOLO model from {weights_path}")
                        return
                    except ImportError:
                        pass
                
                # Try loading with torch.hub (YOLOv5)
                try:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True)
                    self.model_type = 'yolov5'
                    print(f"✅ Loaded YOLO model from {weights_path}")
                    return
                except Exception as e:
                    print(f"⚠️ Error loading from path: {e}")
            
            # Try to load YOLOv8 nano (smallest, fastest model)
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')  # nano version for speed
                self.model_type = 'ultralytics'
                print("✅ Loaded YOLOv8n (nano) pre-trained model")
                return
            except ImportError:
                pass
            
            # Try to load YOLOv5s from torch.hub
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                self.model_type = 'yolov5'
                print("✅ Loaded YOLOv5s pre-trained model from torch.hub")
                return
            except Exception as e:
                print(f"⚠️ Error loading YOLO: {e}")
                print("Attempting to install ultralytics...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.model_type = 'ultralytics'
                print("✅ Loaded YOLOv8n after installation")
                
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("⚠️ Using simplified detection fallback")
            self.model = None
    
    def detect(self, image_path):
        """
        Detect objects in an image using YOLO algorithm
        
        YOLO algorithm performs:
        1. Grid division: Split image into S x S grid (typically 19x19)
        2. Each cell predicts K bounding boxes
        3. Calculate class confidence scores
        4. Apply non-max suppression to eliminate duplicate detections
        5. Output final detections with bounding boxes
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detection dictionaries with format:
            [{'label': str, 'bbox': (x1, y1, x2, y2), 'score': float}, ...]
        """
        if self.model is None:
            return []
        
        detections = []
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not read image: {image_path}")
                return []
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Run YOLO detection
            print(f"🔍 Running YOLO detection with conf_threshold={self.conf_threshold}")
            
            if self.model_type == 'ultralytics':
                # YOLOv8 detection
                results = self.model(img_rgb, conf=self.conf_threshold, iou=0.45)
                
                # Extract detections
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    print(f"📊 Found {len(boxes)} raw detections")
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        x1, y1, x2, y2 = box
                        
                        if cls < len(self.class_names):
                            label = self.class_names[cls]
                        else:
                            label = f"class_{cls}"
                        
                        detections.append({
                            'label': label,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'score': conf
                        })
                        
                        print(f"  ✅ Detected: {label} (score={conf:.3f})")
                else:
                    print(f"⚠️ No objects detected by YOLO")
            
            elif self.model_type == 'yolov5':
                # YOLOv5 detection
                results = self.model(img_rgb, size=self.img_size)
                pred = results.pred[0]
                
                if len(pred) > 0:
                    for det in pred:
                        x1, y1, x2, y2, conf, cls = det[:6]
                        
                        if conf >= self.conf_threshold:
                            cls = int(cls)
                            if 0 <= cls < len(self.class_names):
                                label = self.class_names[cls]
                                detections.append({
                                    'label': label,
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'score': float(conf)
                                })
            
        except Exception as e:
            print(f"❌ Error during YOLO detection: {e}")
            return []
        
        # Apply non-max suppression manually if needed
        if detections:
            detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections):
        """
        Apply Non-Max Suppression to eliminate duplicate detections
        
        NMS helps remove overlapping bounding boxes that cover the same object.
        It keeps only the detection with the highest confidence score for each object.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []
        
        # Separate by class
        class_detections = {}
        for det in detections:
            label = det['label']
            if label not in class_detections:
                class_detections[label] = []
            class_detections[label].append(det)
        
        # Apply NMS within each class
        filtered_detections = []
        for label, dets in class_detections.items():
            if len(dets) == 1:
                filtered_detections.extend(dets)
                continue
            
            # Sort by confidence
            dets_sorted = sorted(dets, key=lambda x: x['score'], reverse=True)
            
            # Greedy NMS
            keep = []
            used = [False] * len(dets_sorted)
            
            for i in range(len(dets_sorted)):
                if used[i]:
                    continue
                
                keep.append(i)
                x1_i, y1_i, x2_i, y2_i = dets_sorted[i]['bbox']
                
                for j in range(i + 1, len(dets_sorted)):
                    if used[j]:
                        continue
                    
                    x1_j, y1_j, x2_j, y2_j = dets_sorted[j]['bbox']
                    
                    # Calculate IoU (Intersection over Union)
                    iou = self._calculate_iou(
                        (x1_i, y1_i, x2_i, y2_i),
                        (x1_j, y1_j, x2_j, y2_j)
                    )
                    
                    if iou > self.nms_threshold:
                        used[j] = True
            
            for idx in keep:
                filtered_detections.append(dets_sorted[idx])
        
        return filtered_detections
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        IoU is used in non-max suppression to determine if two boxes cover the same object.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2
        
        # Calculate intersection area
        x1_inter = max(x1_box1, x1_box2)
        y1_inter = max(y1_box1, y1_box2)
        x2_inter = min(x2_box1, x2_box2)
        y2_inter = min(y2_box1, y2_box2)
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # Calculate union area
        box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def annotate_image(self, image_path, detections, output_path):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            output_path: Path to save annotated image
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image for annotation: {image_path}")
            return
        
        h, w = img.shape[:2]
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            score = det['score']
            
            # Get color for this class
            class_idx = self.class_names.index(label) if label in self.class_names else 0
            color = tuple(map(int, self.colors[class_idx % len(self.colors)]))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            text = f"{label} {score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save annotated image
        cv2.imwrite(output_path, img)
        print(f"✅ Saved annotated image to: {output_path}")
    
    def filter_by_keywords(self, detections, keywords):
        """
        Filter detections based on keyword matching
        
        Args:
            detections: List of detection dictionaries
            keywords: List of keywords to match against
            
        Returns:
            Filtered detections
        """
        if not keywords:
            print("⚠️ No keywords provided, returning all detections")
            return detections
        
        keywords_lower = [kw.lower().strip() for kw in keywords]
        filtered = []
        
        print(f"🔍 Filtering {len(detections)} detections for keywords: {keywords_lower}")
        
        for det in detections:
            label_lower = det['label'].lower().strip()
            
            # Direct match
            for kw in keywords_lower:
                if kw in label_lower:
                    filtered.append(det)
                    print(f"  ✅ Matched '{det['label']}' for keyword '{kw}'")
                    break
            
            # Check synonyms if no direct match
            if det not in filtered:
                synonyms = {
                    'gun': ['weapon', 'firearm', 'pistol', 'rifle', 'handgun'],
                    'knife': ['blade', 'cutting tool'],
                    'vehicle': ['car', 'truck', 'automobile', 'motorcycle', 'bicycle', 'bus', 'boat'],
                    'person': ['human', 'people', 'individual'],
                    'weapon': ['gun', 'knife', 'blade', 'firearm'],
                    'bag': ['backpack', 'purse', 'suitcase', 'handbag', 'bag'],
                    'plate': ['license', 'number plate', 'registration'],
                    'cell phone': ['phone', 'mobile', 'smartphone'],
                    'laptop': ['computer', 'notebook'],
                    'cake': ['dessert', 'cake'],
                    'cup': ['mug', 'cup'],
                    'bottle': ['bottle', 'container'],
                    'chair': ['seat', 'chair'],
                    'couch': ['sofa', 'couch'],
                    'bed': ['bed', 'mattress'],
                    'book': ['book', 'novel'],
                    'scissors': ['scissors', 'cutter'],
                }
                
                # Check synonyms
                for keyword in keywords_lower:
                    if keyword in synonyms:
                        for syn in synonyms[keyword]:
                            if syn in label_lower:
                                filtered.append(det)
                                print(f"  ✅ Matched '{det['label']}' via synonym '{keyword}' -> '{syn}'")
                                break
        
        print(f"📊 Filtered to {len(filtered)} detections")
        return filtered


# Example usage and testing
if __name__ == "__main__":
    print("🔍 YOLO Object Detector - Test Mode")
    print("=" * 50)
    
    # Initialize detector
    detector = YOLODetector(conf_threshold=0.25, nms_threshold=0.4)
    
    # Test with a sample image if available
    test_images = [
        "uploads/test_image.jpg",
        "uploads/sample.jpg",
        "test_image.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Processing: {img_path}")
            detections = detector.detect(img_path)
            
            if detections:
                print(f"✅ Found {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['label']}: {det['score']:.2f} at {det['bbox']}")
                
                # Annotate image
                output_path = img_path.replace('.jpg', '_yolo_detected.jpg')
                detector.annotate_image(img_path, detections, output_path)
            else:
                print("⚠️ No objects detected")
            break
    
    print("\n✅ YOLO detector test completed!")


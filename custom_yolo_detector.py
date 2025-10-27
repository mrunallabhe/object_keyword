"""
Custom YOLO Detector for Trained Model
Uses custom trained YOLOv12 model with 3 classes: BUS, People, Car
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class CustomYOLODetector:
    """
    Custom YOLO detector for your trained model
    Classes: BUS (0), People (1), Car (2)
    """
    
    def __init__(self, model_path=None, conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize custom YOLO detector
        
        Args:
            model_path: Path to your trained .pt model file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Default class names for your model
        self.class_names = ['BUS', 'People', 'Car']
        
        # Try to load custom model
        if model_path and os.path.exists(model_path):
            print(f"🔍 Loading custom trained model from: {model_path}")
            self.model = YOLO(model_path)
            self.model_type = 'custom'
            print(f"✅ Custom model loaded!")
        else:
            print(f"⚠️ Custom model not found at {model_path}")
            print("⚠️ Using default YOLOv8 model instead")
            self.model = YOLO('yolov8n.pt')
            self.model_type = 'default'
            
            # If using default, use full COCO classes
            # But map your classes
            self.class_mapping = {
                'bus': 'BUS',
                'person': 'People',
                'car': 'Car'
            }
    
    def detect(self, image_path):
        """
        Detect objects using custom trained model
        
        Args:
            image_path: Path to image
            
        Returns:
            List of detections with format:
            [{'label': str, 'bbox': (x1, y1, x2, y2), 'score': float}, ...]
        """
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not read image: {image_path}")
                return []
            
            print(f"🔍 Running detection with custom model...")
            
            # Run inference
            results = self.model.predict(
                img, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Extract detections
            result = results[0]
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"📊 Found {len(result.boxes)} raw detections")
                
                for i in range(len(result.boxes)):
                    # Get box coordinates
                    box = result.boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    # Get confidence
                    conf = float(result.boxes.conf[i].cpu().numpy())
                    
                    # Get class
                    cls = int(result.boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    if cls < len(result.names):
                        label = result.names[cls]
                    elif cls < len(self.class_names):
                        label = self.class_names[cls]
                    else:
                        label = f"class_{cls}"
                    
                    detections.append({
                        'label': label,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'score': conf
                    })
                    
                    print(f"  ✅ {label}: {conf:.3f} at ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
            else:
                print("⚠️ No objects detected")
            
            return detections
            
        except Exception as e:
            print(f"❌ Error in custom detection: {e}")
            return []
    
    def detect_enhanced(self, image_path):
        """
        Enhanced detection with custom model
        Includes additional processing for better accuracy
        """
        # Basic detection
        detections = self.detect(image_path)
        
        # Apply custom post-processing
        if self.model_type == 'custom':
            # Your trained model's classes
            enhanced_detections = self._enhance_detections(detections, image_path)
            return enhanced_detections
        
        return detections
    
    def _enhance_detections(self, detections, image_path):
        """
        Enhance detections with additional logic
        """
        enhanced = detections.copy()
        
        # Check for specific patterns in your custom model
        # For example: BUS detection
        has_people = any(d['label'] == 'People' or 'person' in d['label'].lower() for d in detections)
        has_vehicle = any(d['label'] in ['BUS', 'Car', 'car'] for d in detections)
        
        # If people and vehicle together, might be a bus stop or traffic scene
        if has_people and has_vehicle:
            for det in detections:
                # Enhance vehicle labels if context suggests bus
                if det['label'] in ['car', 'Car'] and has_people:
                    # Check if it's actually a bus based on size/context
                    x1, y1, x2, y2 = det['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Buses are typically larger
                    if width * height > 20000:  # Large object
                        det['label'] = 'BUS'
        
        return enhanced
    
    def annotate_image(self, image_path, detections, output_path):
        """
        Draw annotations on image with custom styling
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return
        
        h, w = img.shape[:2]
        
        # Define colors for your custom classes
        colors = {
            'BUS': (0, 0, 255),        # Red
            'People': (255, 0, 0),     # Blue
            'Car': (0, 255, 0),        # Green
            'person': (255, 0, 0),     # Blue
            'car': (0, 255, 0),        # Green
            'bus': (0, 0, 255)         # Red
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            score = det['score']
            
            # Get color for this class
            color = colors.get(label, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            text = f"{label} {score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for text
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 8), 
                         (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, img)
        print(f"✅ Saved annotated image: {output_path}")
    
    def filter_by_keywords(self, detections, keywords):
        """
        Filter detections based on keywords
        Supports your custom classes
        """
        if not keywords:
            return detections
        
        keywords_lower = [kw.lower().strip() for kw in keywords]
        filtered = []
        
        # Custom synonyms for your classes
        synonyms = {
            'bus': ['bus', 'bus', 'public transport', 'transit'],
            'people': ['person', 'people', 'human', 'individual', 'pedestrian'],
            'car': ['car', 'vehicle', 'automobile', 'auto']
        }
        
        for det in detections:
            label_lower = det['label'].lower()
            
            # Direct match
            for kw in keywords_lower:
                if kw in label_lower:
                    filtered.append(det)
                    break
            
            # Synonym match
            if det not in filtered:
                for keyword in keywords_lower:
                    if keyword in synonyms:
                        for syn in synonyms[keyword]:
                            if syn in label_lower:
                                filtered.append(det)
                                break
        
        return filtered


# Example usage
if __name__ == "__main__":
    print("🔍 Custom YOLO Detector Test")
    print("=" * 50)
    
    # Try to load custom model (adjust path as needed)
    custom_model_path = "yolo11m.pt"  # Your trained model
    if not os.path.exists(custom_model_path):
        custom_model_path = None
        print("⚠️ Custom model not found - using default YOLOv8")
    
    detector = CustomYOLODetector(model_path=custom_model_path, conf_threshold=0.4)
    
    # Test with images
    test_images = [
        "uploads/policeman.jpg",
        "uploads/cake.jpg",
        "uploads/test_image.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Testing: {img_path}")
            detections = detector.detect_enhanced(img_path)
            
            if detections:
                print(f"✅ Found {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['label']}: {det['score']:.3f}")
            else:
                print("⚠️ No objects detected")
    
    print("\n✅ Test completed!")


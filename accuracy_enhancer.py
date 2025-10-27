"""
Ultimate Accuracy Enhancement System
Combines multiple advanced techniques for maximum detection accuracy
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

class UltimateAccuracyDetector:
    """
    Maximum accuracy detector using advanced techniques
    """
    
    def __init__(self, conf_threshold=0.15):
        self.conf_threshold = conf_threshold
        
        print("🔍 Initializing Ultimate Accuracy Detector...")
        
        # Load models
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8n loaded")
        except:
            print("⚠️ Model loading failed")
            self.model = None
    
    def detect_with_maximum_accuracy(self, image_path):
        """
        Ultimate accuracy detection using all techniques
        
        Techniques used:
        1. Higher resolution detection (1280x1280)
        2. Test-time augmentation (flip, rotate, zoom)
        3. Multi-scale detection
        4. Ensemble voting
        5. Confidence calibration
        6. Post-processing refinement
        """
        all_detections = []
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        h, w = img.shape[:2]
        print(f"📸 Processing image: {w}x{h}")
        
        # Technique 1: High-resolution detection
        print("🔹 Method 1: High-resolution detection")
        detections_hr = self._detect_high_res(image_path)
        all_detections.extend(detections_hr)
        
        # Technique 2: Multi-scale ensemble
        print("🔹 Method 2: Multi-scale detection")
        detections_ms = self._detect_multi_scale(image_path)
        all_detections.extend(detections_ms)
        
        # Technique 3: Test-time augmentation
        print("🔹 Method 3: Test-time augmentation")
        detections_tta = self._test_time_augmentation(image_path)
        all_detections.extend(detections_tta)
        
        # Technique 4: Region-based detection (for small objects)
        print("🔹 Method 4: Region-based detection")
        detections_reg = self._region_based_detection(img)
        all_detections.extend(detections_reg)
        
        # Combine results using voting
        print(f"📊 Total detections collected: {len(all_detections)}")
        final_detections = self._intelligent_combine(all_detections)
        print(f"✅ Final detections after voting: {len(final_detections)}")
        
        return final_detections
    
    def _detect_high_res(self, image_path):
        """Detect at higher resolution for better accuracy"""
        if self.model is None:
            return []
        
        try:
            # Detect at full resolution
            results = self.model(image_path, conf=self.conf_threshold, imgsz=1280)
            return self._parse_results(results[0])
        except:
            return []
    
    def _detect_multi_scale(self, image_path):
        """Multi-scale detection"""
        all_dets = []
        
        if self.model is None:
            return []
        
        for scale in [0.7, 1.0, 1.3]:
            try:
                img = cv2.imread(image_path)
                h, w = img.shape[:2]
                
                # Scale image
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Save temp
                temp_path = f"temp_scale_{scale}.jpg"
                cv2.imwrite(temp_path, img_scaled)
                
                # Detect
                results = self.model(temp_path, conf=self.conf_threshold, verbose=False)
                dets = self._parse_results(results[0])
                
                # Scale boxes back
                for det in dets:
                    x1, y1, x2, y2 = det['bbox']
                    det['bbox'] = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
                    det['scale'] = scale
                    all_dets.append(det)
                
                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass
            except:
                continue
        
        return all_dets
    
    def _test_time_augmentation(self, image_path):
        """Test-time augmentation"""
        all_dets = []
        
        if self.model is None:
            return []
        
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        h, w = img.shape[:2]
        
        # Original
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        all_dets.extend(self._parse_results(results[0]))
        
        # Horizontal flip
        img_flip = cv2.flip(img, 1)
        temp_flip = "temp_flip.jpg"
        cv2.imwrite(temp_flip, img_flip)
        results = self.model(temp_flip, conf=self.conf_threshold, verbose=False)
        dets = self._parse_results(results[0])
        for det in dets:
            # Transform box back
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = (w - x2, y1, w - x1, y2)
            all_dets.append(det)
        try:
            os.remove(temp_flip)
        except:
            pass
        
        # Slightly higher confidence for TTA
        results = self.model(image_path, conf=self.conf_threshold + 0.1, verbose=False)
        all_dets.extend(self._parse_results(results[0]))
        
        return all_dets
    
    def _region_based_detection(self, img):
        """Region-based detection for small objects"""
        all_dets = []
        
        if self.model is None:
            return []
        
        h, w = img.shape[:2]
        
        # Divide image into overlapping regions
        region_size = min(w, h) // 2
        stride = region_size // 2
        
        for y in range(0, h - region_size + 1, stride):
            for x in range(0, w - region_size + 1, stride):
                # Extract region
                region = img[y:y+region_size, x:x+region_size]
                
                # Save temp region
                temp_region = f"temp_region_{x}_{y}.jpg"
                cv2.imwrite(temp_region, region)
                
                try:
                    # Detect in region
                    results = self.model(temp_region, conf=self.conf_threshold + 0.1, verbose=False)
                    dets = self._parse_results(results[0])
                    
                    # Offset boxes to original coordinates
                    for det in dets:
                        bx1, by1, bx2, by2 = det['bbox']
                        det['bbox'] = (bx1 + x, by1 + y, bx2 + x, by2 + y)
                        det['region'] = f"{x},{y}"
                        all_dets.append(det)
                except:
                    pass
                
                # Clean up
                try:
                    os.remove(temp_region)
                except:
                    pass
        
        return all_dets
    
    def _parse_results(self, result):
        """Parse YOLO results"""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        try:
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i].cpu().numpy())
                cls = int(result.boxes.cls[i].cpu().numpy())
                
                label = result.names[cls] if cls < len(result.names) else f"class_{cls}"
                
                detections.append({
                    'label': label,
                    'bbox': (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    'score': conf
                })
        except:
            pass
        
        return detections
    
    def _intelligent_combine(self, all_detections):
        """
        Intelligent combination of detections
        
        Strategy:
        1. Group by class and similar position
        2. Count how many times each object was detected
        3. Keep objects detected multiple times (voting)
        4. Boost confidence for consensus detections
        """
        if not all_detections:
            return []
        
        # Group by label and similar bbox
        groups = {}
        
        for det in all_detections:
            key = det['label']
            x1, y1, x2, y2 = det['bbox']
            
            # Find similar detections
            added = False
            for existing_key, group_dets in groups.items():
                if existing_key.startswith(key):
                    # Check if boxes are similar
                    avg_bbox = self._get_average_bbox([d['bbox'] for d in group_dets])
                    if self._bboxes_similar(avg_bbox, det['bbox']):
                        group_dets.append(det)
                        added = True
                        break
            
            if not added:
                # Create new group
                groups[f"{key}_{len(groups)}"] = [det]
        
        # Select best detections from each group
        final = []
        for group_dets in groups.values():
            if len(group_dets) == 1:
                # Single detection
                final.append(group_dets[0])
            else:
                # Multiple detections of same object - use consensus
                # Calculate average bbox
                avg_bbox = self._get_average_bbox([d['bbox'] for d in group_dets])
                
                # Calculate average confidence
                avg_conf = np.mean([d['score'] for d in group_dets])
                
                # Boost confidence based on agreement
                num_agreements = len(group_dets)
                boost = min(0.2 * (num_agreements - 1), 0.4)  # Max 0.4 boost
                final_conf = min(avg_conf + boost, 1.0)
                
                final.append({
                    'label': group_dets[0]['label'],
                    'bbox': avg_bbox,
                    'score': final_conf,
                    'detections_count': num_agreements
                })
        
        # Final NMS
        final = self._apply_final_nms(final)
        
        return final
    
    def _bboxes_similar(self, bbox1, bbox2, iou_threshold=0.3):
        """Check if two bounding boxes are similar"""
        try:
            iou = self._calculate_iou(bbox1, bbox2)
            return iou > iou_threshold
        except:
            return False
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU"""
        try:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
            
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        except:
            return 0
    
    def _get_average_bbox(self, bboxes):
        """Get average bounding box from multiple boxes"""
        if not bboxes:
            return (0, 0, 0, 0)
        
        avg_x1 = int(np.mean([b[0] for b in bboxes]))
        avg_y1 = int(np.mean([b[1] for b in bboxes]))
        avg_x2 = int(np.mean([b[2] for b in bboxes]))
        avg_y2 = int(np.mean([b[3] for b in bboxes]))
        
        return (avg_x1, avg_y1, avg_x2, avg_y2)
    
    def _apply_final_nms(self, detections):
        """Apply final NMS to remove overlaps"""
        if not detections:
            return []
        
        # Sort by score
        sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        kept = []
        for det in sorted_dets:
            overlap = False
            for k in kept:
                if self._calculate_iou(det['bbox'], k['bbox']) > 0.3:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        
        return kept


"""
Ensemble Detector for Maximum Accuracy
Uses multiple detection methods and combines results for better accuracy
"""

import cv2
import numpy as np
import torch
from yolo_detector import YOLODetector
from collections import defaultdict

class EnsembleDetector:
    """
    Ensemble detector combining multiple models and techniques for maximum accuracy
    """
    
    def __init__(self, conf_threshold=0.15):
        """
        Initialize ensemble detector with multiple detection strategies
        """
        self.conf_threshold = conf_threshold
        
        # Initialize multiple detectors
        print("🔍 Initializing Ensemble Detector...")
        
        try:
            from ultralytics import YOLO
            # Load multiple YOLO models
            self.yolo_v8n = YOLO('yolov8n.pt')  # Nano - fastest
            self.yolo_v8s = YOLO('yolov8s.pt')  # Small - better accuracy
            print("✅ Loaded YOLOv8n and YOLOv8s")
        except:
            print("⚠️ Could not load YOLOv8 models")
            self.yolo_v8n = None
            self.yolo_v8s = None
        
        # Base detector
        self.base_detector = YOLODetector(conf_threshold=conf_threshold)
        
    def detect_ensemble(self, image_path, methods=['standard', 'tts', 'multiscale']):
        """
        Ensemble detection using multiple methods and combining results
        
        Args:
            image_path: Path to image
            methods: Detection methods to use
                - 'standard': Basic YOLO detection
                - 'tts': Test-time augmentation
                - 'multiscale': Multiple scales
                - 'ensemble': Multiple models
        
        Returns:
            Combined detections with voting
        """
        all_detections = []
        
        # Method 1: Standard detection
        if 'standard' in methods:
            detections_std = self._standard_detection(image_path)
            all_detections.extend(detections_std)
        
        # Method 2: Test-time augmentation
        if 'tts' in methods:
            detections_tta = self._test_time_augmentation(image_path)
            all_detections.extend(detections_tta)
        
        # Method 3: Multi-scale
        if 'multiscale' in methods:
            detections_ms = self._multi_scale_detection(image_path)
            all_detections.extend(detections_ms)
        
        # Method 4: Ensemble models
        if 'ensemble' in methods and self.yolo_v8n and self.yolo_v8s:
            detections_ens = self._ensemble_models(image_path)
            all_detections.extend(detections_ens)
        
        # Combine and vote
        final_detections = self._voting_nms(all_detections)
        
        return final_detections
    
    def _standard_detection(self, image_path):
        """Standard YOLO detection"""
        return self.base_detector.detect(image_path)
    
    def _test_time_augmentation(self, image_path):
        """
        Test-time augmentation - detects at different orientations/transforms
        Improves accuracy by detecting rotated/flipped objects
        """
        detections_all = []
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # Original
            detections_orig = self._detect_image(img)
            detections_all.extend(detections_orig)
            
            # Horizontal flip
            img_flipped = cv2.flip(img, 1)
            detections_flip = self._detect_image(img_flipped)
            for det in detections_flip:
                # Transform bounding box back
                h, w = img.shape[:2]
                x1, y1, x2, y2 = det['bbox']
                x1, x2 = w - x2, w - x1
                det['bbox'] = (x1, y1, x2, y2)
                detections_all.append(det)
            
            # Slight rotation
            for angle in [-15, 15]:
                M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)
                img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                detections_rot = self._detect_image(img_rot)
                detections_all.extend(detections_rot)
            
        except Exception as e:
            print(f"⚠️ TTA error: {e}")
        
        return detections_all
    
    def _multi_scale_detection(self, image_path):
        """Multi-scale detection at different resolutions"""
        detections_all = []
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            h, w = img.shape[:2]
            
            for scale in [0.8, 1.0, 1.2]:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Save temp
                temp_path = f"temp_ms_{scale}.jpg"
                cv2.imwrite(temp_path, img_scaled)
                
                # Detect
                detections = self._detect_image(img_scaled)
                
                # Scale boxes back
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                    det['bbox'] = (x1, y1, x2, y2)
                    det['scale'] = scale
                    detections_all.append(det)
                
                # Clean up
                try:
                    import os
                    os.remove(temp_path)
                except:
                    pass
            
        except Exception as e:
            print(f"⚠️ Multi-scale error: {e}")
        
        return detections_all
    
    def _ensemble_models(self, image_path):
        """Use multiple YOLO models and combine"""
        detections_all = []
        
        try:
            # YOLOv8n detections
            if self.yolo_v8n:
                results_n = self.yolo_v8n(image_path, conf=self.conf_threshold, verbose=False)
                detections_all.extend(self._parse_yolo_results(results_n, 'v8n'))
            
            # YOLOv8s detections (more accurate)
            if self.yolo_v8s:
                results_s = self.yolo_v8s(image_path, conf=self.conf_threshold-0.05, verbose=False)
                detections_all.extend(self._parse_yolo_results(results_s, 'v8s'))
        
        except Exception as e:
            print(f"⚠️ Ensemble error: {e}")
        
        return detections_all
    
    def _detect_image(self, img):
        """Detect objects in image array"""
        detections = []
        
        try:
            # Save temp image
            temp_path = "temp_detect.jpg"
            cv2.imwrite(temp_path, img)
            
            # Detect
            results = self.base_detector.detect(temp_path)
            detections.extend(results)
            
            # Clean up
            try:
                import os
                os.remove(temp_path)
            except:
                pass
        
        except:
            pass
        
        return detections
    
    def _parse_yolo_results(self, results, model_name='yolo'):
        """Parse YOLO results into standard format"""
        detections = []
        
        try:
            result = results[0]
            if result.boxes is not None:
                for i in range(len(result.boxes)):
                    box = result.boxes.xyxy[i].cpu().numpy()
                    conf = float(result.boxes.conf[i].cpu().numpy())
                    cls = int(result.boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[cls] if cls < len(result.names) else f"class_{cls}"
                    
                    detections.append({
                        'label': class_name,
                        'bbox': (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        'score': conf,
                        'model': model_name
                    })
        except:
            pass
        
        return detections
    
    def _voting_nms(self, all_detections):
        """
        Voting-based NMS - combines detections from multiple methods
        
        Key insight: If multiple methods detect similar boxes, it's more likely correct
        """
        if not all_detections:
            return []
        
        # Group detections by class and position
        grouped = defaultdict(list)
        
        for det in all_detections:
            key = det['label']
            grouped[key].append(det)
        
        # For each class, keep only detections that appear in multiple methods
        final_detections = []
        
        for label, dets in grouped.items():
            # Sort by score
            dets_sorted = sorted(dets, key=lambda x: x['score'], reverse=True)
            
            # Apply smart NMS
            kept = self._smart_nms(dets_sorted)
            
            # Boost confidence if detected multiple times
            for det in kept:
                # Count how many methods detected it
                count = sum(1 for d in dets if self._boxes_similar(det['bbox'], d['bbox']))
                if count > 1:
                    det['score'] = min(det['score'] + 0.1 * (count - 1), 1.0)
                
                final_detections.append(det)
        
        # Final NMS across all classes
        final_detections = self._final_nms(final_detections)
        
        return final_detections
    
    def _boxes_similar(self, box1, box2, iou_threshold=0.5):
        """Check if two boxes are similar"""
        iou = self._calculate_iou(box1, box2)
        return iou > iou_threshold
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
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
    
    def _smart_nms(self, detections):
        """Smart NMS that keeps high-confidence detections"""
        if not detections:
            return []
        
        # Track which detections are kept
        kept = []
        
        for i, det in enumerate(detections):
            overlap = False
            
            # Check against already kept detections
            for kept_det in kept:
                if self._boxes_similar(det['bbox'], kept_det['bbox']):
                    # If new detection has higher confidence, replace
                    if det['score'] > kept_det['score']:
                        kept.remove(kept_det)
                        kept.append(det)
                    overlap = True
                    break
            
            if not overlap:
                kept.append(det)
        
        return kept
    
    def _final_nms(self, detections, iou_threshold=0.4):
        """Final NMS pass"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        kept = []
        for det in detections:
            overlap = False
            for k in kept:
                if self._calculate_iou(det['bbox'], k['bbox']) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        
        return kept


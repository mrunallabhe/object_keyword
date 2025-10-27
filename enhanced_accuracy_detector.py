"""
Enhanced Accuracy Detector
- Handles low quality images (upscaling, denoising, sharpening)
- Detects tiny objects (multi-scale detection)
- Better post-processing
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from yolo_detector import YOLODetector
import torch

class EnhancedAccuracyDetector(YOLODetector):
    """
    Enhanced detector for low quality images and tiny objects
    """
    
    def __init__(self, weights_path=None, conf_threshold=0.15, nms_threshold=0.4, img_size=640):
        super().__init__(weights_path, conf_threshold, nms_threshold, img_size)
        
        # Multi-scale detection settings
        self.scales = [0.5, 1.0, 1.5]  # Test at multiple scales
        self.tiny_object_threshold = 50  # Pixels - objects smaller than this
        
    def preprocess_low_quality_image(self, image_path):
        """
        Enhance low quality images for better detection
        """
        try:
            # Skip if already enhanced (avoid recursion)
            if '_enhanced' in image_path:
                return image_path
            
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            h, w = img.shape[:2]
            
            # Check if image is low resolution
            if h * w < 250000:  # Less than 500x500
                print(f"🔧 Low resolution detected ({w}x{h}) - enhancing...")
                
                # Upscale using super-resolution
                scale_factor = max(640 / w, 640 / h)
                
                if scale_factor > 1.5:
                    # Use inter-cubic for upscaling
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                    img_upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    
                    # Apply denoising
                    img_denoised = cv2.fastNlMeansDenoisingColored(img_upscaled, None, 10, 10, 7, 21)
                    
                    # Apply sharpening
                    kernel = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
                    img_sharp = cv2.filter2D(img_denoised, -1, kernel)
                    
                    # Save enhanced image (only once)
                    enhanced_path = image_path.replace('.jpg', '_enhanced.jpg').replace('.jpeg', '_enhanced.jpeg')
                    enhanced_path = enhanced_path.replace('.png', '_enhanced.png')
                    
                    # Check if already exists
                    import os
                    if not os.path.exists(enhanced_path):
                        cv2.imwrite(enhanced_path, img_sharp)
                        print(f"✅ Enhanced image saved: {enhanced_path}")
                    
                    return enhanced_path
                
                return image_path
            else:
                # Use original image without modification
                return image_path
                
        except Exception as e:
            print(f"⚠️ Enhancement error: {e}")
            return image_path
    
    def detect_tiny_objects(self, image_path):
        """
        Multi-scale detection for tiny objects
        Detects objects at different scales to catch small objects
        """
        all_detections = []
        
        try:
            # Original detection
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            h, w = img.shape[:2]
            print(f"🔍 Multi-scale detection for tiny objects (image: {w}x{h})")
            
            # Test at multiple scales
            for scale in self.scales:
                # Resize image
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize with quality preservation
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Save temporary scaled image
                temp_path = f"temp_scaled_{scale}.jpg"
                cv2.imwrite(temp_path, img_scaled)
                
                try:
                    # Get detections at this scale
                    detections_scale = super().detect(temp_path)
                    
                    # Scale bounding boxes back to original size
                    for det in detections_scale:
                        x1, y1, x2, y2 = det['bbox']
                        # Scale back to original coordinates
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                        
                        # Only keep if boxes make sense
                        if 0 <= x1 < w and 0 <= y1 < h and x2 > x1 and y2 > y1:
                            det['bbox'] = (x1, y1, x2, y2)
                            det['scale'] = scale
                            all_detections.append(det)
                    
                except Exception as e:
                    print(f"⚠️ Error at scale {scale}: {e}")
                finally:
                    # Clean up temp file
                    try:
                        import os
                        os.remove(temp_path)
                    except:
                        pass
            
            # Remove duplicates using NMS
            if all_detections:
                all_detections = self._apply_nms_multi_scale(all_detections)
            
            print(f"📊 Multi-scale found {len(all_detections)} objects")
            return all_detections
            
        except Exception as e:
            print(f"❌ Error in tiny object detection: {e}")
            return []
    
    def _apply_nms_multi_scale(self, detections):
        """
        Apply NMS to multi-scale detections
        """
        if not detections:
            return []
        
        # Group by label
        by_label = {}
        for det in detections:
            label = det['label']
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(det)
        
        # Apply NMS within each label
        kept = []
        for label, dets in by_label.items():
            if len(dets) == 1:
                kept.extend(dets)
                continue
            
            # Sort by score
            dets_sorted = sorted(dets, key=lambda x: x['score'], reverse=True)
            
            # Keep highest confidence detections
            seen_boxes = []
            for det in dets_sorted:
                # Check overlap with already kept boxes
                x1, y1, x2, y2 = det['bbox']
                overlap = False
                
                for kept_det in seen_boxes:
                    kx1, ky1, kx2, ky2 = kept_det['bbox']
                    iou = self._calculate_iou((x1, y1, x2, y2), (kx1, ky1, kx2, ky2))
                    
                    if iou > self.nms_threshold:
                        overlap = True
                        break
                
                if not overlap:
                    kept.append(det)
                    seen_boxes.append(det)
        
        return kept
    
    def detect_enhanced(self, image_path):
        """
        Enhanced detection combining:
        - Low quality image preprocessing
        - Tiny object detection
        - Multi-scale processing
        """
        try:
            # 1. Enhance low quality images
            enhanced_path = self.preprocess_low_quality_image(image_path)
            
            # 2. Standard detection
            detections = super().detect(enhanced_path)
            
            # 3. Multi-scale detection for tiny objects
            tiny_detections = self.detect_tiny_objects(image_path)
            
            # 4. Combine results
            all_detections = detections.copy()
            
            # Add tiny object detections
            for tiny_det in tiny_detections:
                # Check if similar detection already exists
                x1_t, y1_t, x2_t, y2_t = tiny_det['bbox']
                exists = False
                
                for det in all_detections:
                    x1, y1, x2, y2 = det['bbox']
                    iou = self._calculate_iou((x1, y1, x2, y2), (x1_t, y1_t, x2_t, y2_t))
                    
                    if iou > 0.3 and det['label'] == tiny_det['label']:
                        # Similar detection exists, keep the one with higher confidence
                        if tiny_det['score'] > det['score']:
                            det['bbox'] = tiny_det['bbox']
                            det['score'] = tiny_det['score']
                        exists = True
                        break
                
                if not exists:
                    all_detections.append(tiny_det)
            
            # 5. Mark tiny objects
            for det in all_detections:
                x1, y1, x2, y2 = det['bbox']
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                if area < self.tiny_object_threshold:
                    det['is_tiny'] = True
            
            return all_detections
            
        except Exception as e:
            print(f"❌ Error in enhanced detection: {e}")
            return []
    
    def annotate_image(self, image_path, detections, output_path):
        """
        Enhanced annotation with special styling for tiny objects
        """
        img = cv2.imread(image_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            score = det['score']
            is_tiny = det.get('is_tiny', False)
            
            # Special styling for tiny objects
            if is_tiny:
                color = (0, 165, 255)  # Orange for tiny objects
                thickness = 4
            else:
                # Get color for class
                class_idx = self.class_names.index(label) if label in self.class_names else 0
                color = tuple(map(int, self.colors[class_idx % len(self.colors)]))
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            text = f"{label} {score:.2f}"
            if is_tiny:
                text += " [tiny]"
            
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, img)
        print(f"✅ Saved enhanced annotated image: {output_path}")


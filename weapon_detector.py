"""
Enhanced Weapon and Crime-Related Object Detection
Adds custom weapon detection beyond YOLO's standard 80 classes
"""

import cv2
import numpy as np
import torch
from yolo_detector import YOLODetector

class EnhancedWeaponDetector(YOLODetector):
    """
    Enhanced detector that adds weapon detection capabilities
    """
    
    def __init__(self, weights_path=None, conf_threshold=0.15, nms_threshold=0.4, img_size=640):
        super().__init__(weights_path, conf_threshold, nms_threshold, img_size)
        
        # Add weapon-related synonyms
        self.weapon_synonyms = {
            'gun': ['weapon', 'firearm', 'pistol', 'rifle', 'handgun', 'gun', 'machine gun', 'mp5', 'ak47', 'm16'],
            'weapon': ['gun', 'firearm', 'pistol', 'rifle', 'handgun', 'weapon', 'knife', 'blade'],
            'knife': ['blade', 'knife', 'weapon', 'cutting tool', 'sword'],
            'bat': ['baseball bat', 'bat', 'club', 'weapon'],
            'glove': ['baseball glove', 'glove'],
            'scissors': ['scissors', 'cutter', 'tool'],
        }
        
        # Crime-related detection patterns
        self.crime_patterns = {
            'policeman': ['person', 'police', 'officer'],
            'weapon_visible': ['person'],
            'suspicious_activity': ['person', 'person']
        }
    
    def detect_enhanced(self, image_path):
        """
        Enhanced detection with weapon analysis
        """
        # Get standard YOLO detections
        detections = self.detect(image_path)
        
        # Read image for advanced analysis
        img = cv2.imread(image_path)
        if img is None:
            return detections
        
        # Analyze image for weapon-like objects
        enhanced_detections = detections.copy()
        
        # Check for person objects with potential weapons
        person_detections = [d for d in detections if d['label'] == 'person']
        
        if person_detections:
            # Analyze each person detection for weapon characteristics
            for person in person_detections:
                x1, y1, x2, y2 = person['bbox']
                
                # Crop person region
                person_crop = img[y1:y2, x1:x2]
                
                # Detect elongated objects (potential guns/weapons)
                gun_candidates = self._detect_elongated_objects(person_crop)
                
                if gun_candidates:
                    for candidate in gun_candidates:
                        # Offset coordinates back to original image
                        gun_x1 = x1 + candidate[0]
                        gun_y1 = y1 + candidate[1]
                        gun_x2 = x1 + candidate[2]
                        gun_y2 = y1 + candidate[3]
                        
                        enhanced_detections.append({
                            'label': 'weapon',  # Generic weapon class
                            'bbox': (gun_x1, gun_y1, gun_x2, gun_y2),
                            'score': 0.6,  # Medium confidence
                            'type': 'gun_like_object',
                            'description': 'Potential weapon detected near person'
                        })
        
        return enhanced_detections
    
    def _detect_elongated_objects(self, img_crop):
        """
        Detect elongated, gun-like objects in image
        """
        if img_crop.size == 0:
            return []
        
        candidates = []
        
        try:
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (weapons are typically medium-sized)
                if 500 < area < 50000:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check for elongated shape (guns are long and narrow)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Typical gun: long and thin (aspect ratio > 2.0 or < 0.5)
                    if aspect_ratio > 1.8 or (aspect_ratio < 0.6 and aspect_ratio > 0):
                        candidates.append((x, y, x + w, y + h))
            
            return candidates[:2]  # Return top 2 candidates
            
        except Exception as e:
            print(f"Error in elongated object detection: {e}")
            return []
    
    def filter_by_keywords(self, detections, keywords):
        """
        Enhanced keyword filtering with weapon detection
        """
        if not keywords:
            return detections
        
        keywords_lower = [kw.lower().strip() for kw in keywords]
        filtered = []
        
        for det in detections:
            label_lower = det['label'].lower().strip()
            
            # Direct match
            matched = False
            for kw in keywords_lower:
                if kw in label_lower:
                    filtered.append(det)
                    matched = True
                    break
            
            # Check synonyms
            if not matched:
                for keyword in keywords_lower:
                    if keyword in self.weapon_synonyms:
                        synonyms = self.weapon_synonyms[keyword]
                        for syn in synonyms:
                            if syn in label_lower:
                                filtered.append(det)
                                matched = True
                                break
                        if matched:
                            break
            
            # Special case: "gun" or "weapon" keywords
            if not matched:
                if any(kw in ['gun', 'weapon', 'pistol', 'rifle', 'firearm'] for kw in keywords_lower):
                    if 'weapon' in label_lower or 'bat' in label_lower or 'knife' in label_lower:
                        filtered.append(det)
                        matched = True
            
            # Special case: "policeman" keyword
            if not matched:
                if 'police' in label_lower or 'officer' in label_lower:
                    if 'person' in label_lower or any(kw in ['police', 'officer'] for kw in keywords_lower):
                        filtered.append(det)
                        matched = True
        
        return filtered
    
    def annotate_with_weapons(self, image_path, detections, output_path):
        """
        Enhanced annotation that highlights weapons specially
        """
        img = cv2.imread(image_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            score = det['score']
            
            # Use different colors for different object types
            if label in ['weapon', 'gun']:
                color = (0, 0, 255)  # Red for weapons
                thickness = 4
            elif label == 'person':
                color = (255, 0, 0)  # Blue for people
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for other objects
                thickness = 2
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            text = f"{label} {score:.2f}"
            if 'description' in det:
                text += f" ({det['description']})"
            
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw background for text
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, img)
        print(f"✅ Saved enhanced annotated image: {output_path}")


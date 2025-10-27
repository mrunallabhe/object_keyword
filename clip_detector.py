"""
CLIP-based Semantic Object Detection
Handles natural language queries with relationships and context

Example queries:
- "person holding cup"
- "dog sitting on sofa"
- "man wearing blue jeans"
- "woman with red bag"
- "vehicle at intersection"

This uses CLIP (Contrastive Language–Image Pretraining) to understand
semantic relationships and context, not just keyword matching.
"""

import torch
import clip
import cv2
import numpy as np
from PIL import Image
import os

class CLIPSemanticDetector:
    """
    CLIP-based detector for understanding semantic queries with relationships
    """
    
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-L/14, RN50)
            device: 'cuda' or 'cpu'
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔍 Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"✅ CLIP loaded on {self.device}")
        
        self.model_name = model_name
    
    def get_text_embedding(self, text_queries):
        """
        Get CLIP embeddings for text queries
        
        Args:
            text_queries: List of text queries or single query string
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        # Tokenize text
        text_tokens = clip.tokenize(text_queries).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def get_image_embedding(self, image_path):
        """
        Get CLIP embedding for an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding tensor
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def semantic_search(self, image_path, text_query, threshold=0.25):
        """
        Search for semantic matches in image using natural language query
        
        Args:
            image_path: Path to image
            text_query: Natural language query (e.g., "person holding cup")
            threshold: Similarity threshold
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Get embeddings
            text_embedding = self.get_text_embedding(text_query)
            image_embedding = self.get_image_embedding(image_path)
            
            # Calculate cosine similarity
            similarity = (image_embedding @ text_embedding.T).item()
            
            # Return match if above threshold
            return similarity >= threshold, similarity
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return False, 0.0
    
    def detect_semantic_objects(self, image_path, text_queries, tile_size=224, stride=100):
        """
        Detect objects in image using semantic query with sliding window approach
        
        Args:
            image_path: Path to image
            text_queries: List of natural language queries
            tile_size: Size of tiles for sliding window
            stride: Stride for sliding window
            
        Returns:
            List of detections with bounding boxes
        """
        detections = []
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        
        # Get text embeddings for all queries
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        text_features = self.get_text_embedding(text_queries)
        
        # Sliding window approach
        patches = []
        bboxes = []
        
        for y in range(0, img_height - tile_size + 1, stride):
            for x in range(0, img_width - tile_size + 1, stride):
                # Extract patch
                patch = img.crop((x, y, x + tile_size, y + tile_size))
                
                # Preprocess patch
                patch_tensor = self.preprocess(patch).unsqueeze(0).to(self.device)
                
                # Get patch embedding
                with torch.no_grad():
                    patch_features = self.model.encode_image(patch_tensor)
                    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity with each query
                similarities = (patch_features @ text_features.T).squeeze().cpu()
                
                # Check each query
                if isinstance(similarities, torch.Tensor):
                    similarities = similarities.tolist()
                    if not isinstance(similarities, list):
                        similarities = [similarities]
                
                for i, sim in enumerate(similarities):
                    if sim > 0.22:  # Similarity threshold
                        label = text_queries[i] if i < len(text_queries) else "object"
                        detections.append({
                            'label': label,
                            'bbox': (x, y, x + tile_size, y + tile_size),
                            'score': float(sim)
                        })
        
        # Apply Non-Max Suppression to remove overlapping boxes
        if detections:
            detections = self._apply_nms(detections, iou_threshold=0.5)
        
        return detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Max Suppression to remove overlapping detections
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        # Sort by score
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        filtered = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]:
                continue
            
            filtered.append(detections[i])
            
            # Suppress overlapping boxes
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                
                iou = self._calculate_iou(detections[i]['bbox'], detections[j]['bbox'])
                
                if iou > iou_threshold:
                    used[j] = True
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def annotate_image(self, image_path, detections, output_path):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image_path: Input image path
            detections: List of detections
            output_path: Output path for annotated image
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return
        
        height, width = img.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            score = det['score']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label} {score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, img)
        print(f"✅ Saved annotated image: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    print("🔍 CLIP Semantic Detector - Test Mode")
    print("=" * 50)
    
    # Initialize detector
    detector = CLIPSemanticDetector(model_name="ViT-B/32")
    
    # Test semantic queries
    print("\n📝 Testing semantic queries:")
    
    test_queries = [
        "person holding cup",
        "dog sitting on sofa",
        "man wearing blue jeans",
        "woman with red bag",
        "vehicle at intersection"
    ]
    
    # Test with a sample image
    test_images = [
        "uploads/test_image.jpg",
        "uploads/sample.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Processing: {img_path}")
            
            for query in test_queries:
                # Test semantic search
                is_match, similarity = detector.semantic_search(img_path, query)
                
                if is_match:
                    print(f"  ✅ '{query}': similarity={similarity:.3f}")
                else:
                    print(f"  ❌ '{query}': similarity={similarity:.3f}")
            
            break
    
    print("\n✅ CLIP detector test completed!")


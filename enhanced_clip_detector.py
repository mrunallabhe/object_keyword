"""
Enhanced CLIP-based Semantic Object Detection
Uses multi-scale tiling and batch processing for better accuracy
"""

import torch
import clip
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class EnhancedCLIPDetector:
    """
    Enhanced CLIP detector with multi-scale tiling and batch processing
    """
    
    def __init__(self, model_name="ViT-B/32", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔍 Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"✅ CLIP loaded on {self.device}")
        
        self.model_name = model_name
        self.model.eval()
    
    def generate_tiles(self, image, scales=[1.0, 0.75, 0.5], base_size=224, stride_factor=0.5):
        """
        Generate tiles across multiple scales for better detection
        Returns list of (tile_image_pil, bbox)
        """
        w, h = image.size
        tiles = []
        
        for scale in scales:
            tile_size = int(base_size / scale)
            stride = max(16, int(tile_size * stride_factor))
            
            # If tile_size larger than image, use whole image
            if tile_size >= max(w, h):
                tile = image.resize((base_size, base_size))
                tiles.append((tile, (0, 0, w, h)))
                continue
            
            for top in range(0, max(1, h - tile_size + 1), stride):
                for left in range(0, max(1, w - tile_size + 1), stride):
                    right = min(left + tile_size, w)
                    bottom = min(top + tile_size, h)
                    crop = image.crop((left, top, right, bottom)).resize((base_size, base_size))
                    tiles.append((crop, (left, top, right, bottom)))
        
        return tiles
    
    def detect_text_guided(self, image_path, text_prompt, 
                            scales=[1.0, 0.75, 0.5], base_size=224, stride_factor=0.5,
                            topk=20, score_threshold=0.25, nms_iou=0.35, batch_size=32):
        """
        Enhanced text-guided detection with multi-scale tiling
        
        Returns:
            List of detections with bbox and score
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tiles = self.generate_tiles(image, scales, base_size, stride_factor)
            
            # Encode text
            text_tokens = clip.tokenize([text_prompt]).to(self.device)
            with torch.no_grad():
                text_feat = self.model.encode_text(text_tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                text_feat_cpu = text_feat.cpu().numpy().reshape(-1)
            
            # Collect patch features in batches
            feats = []
            bboxes = []
            batch = []
            batch_bboxes = []
            
            for tile_img, bbox in tiles:
                batch.append(self.preprocess(tile_img).unsqueeze(0).to(self.device))
                batch_bboxes.append(bbox)
                
                if len(batch) == batch_size:
                    inp = torch.cat(batch, dim=0)
                    with torch.no_grad():
                        img_feats = self.model.encode_image(inp)
                        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    feats.append(img_feats.cpu().numpy())
                    bboxes.extend(batch_bboxes)
                    batch = []
                    batch_bboxes = []
            
            # Process remaining
            if batch:
                inp = torch.cat(batch, dim=0)
                with torch.no_grad():
                    img_feats = self.model.encode_image(inp)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                feats.append(img_feats.cpu().numpy())
                bboxes.extend(batch_bboxes)
            
            if not feats:
                return []
            
            feats = np.concatenate(feats, axis=0)
            
            # Compute cosine similarities
            sims = (feats @ text_feat_cpu).reshape(-1)
            
            # Filter by threshold and take topk
            candidate_inds = np.where(sims >= score_threshold)[0]
            if candidate_inds.size == 0:
                candidate_inds = np.argsort(-sims)[:topk]
            else:
                candidate_inds = candidate_inds[np.argsort(-sims[candidate_inds])][:topk]
            
            candidate_scores = sims[candidate_inds]
            candidate_boxes = [bboxes[i] for i in candidate_inds]
            
            # Prepare for NMS
            boxes_tensor = torch.tensor(candidate_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(candidate_scores, dtype=torch.float32)
            
            if boxes_tensor.shape[0] == 0:
                return []
            
            # Apply NMS
            try:
                from torchvision.ops import nms
                keep = nms(boxes_tensor, scores_tensor, nms_iou).cpu().numpy()
            except:
                # Simple NMS fallback
                keep = self._simple_nms(boxes_tensor, scores_tensor, nms_iou)
            
            kept_boxes = boxes_tensor[keep].numpy().astype(int).tolist()
            kept_scores = scores_tensor[keep].numpy().tolist()
            
            detections = [{"bbox": b, "score": float(s)} for b, s in zip(kept_boxes, kept_scores)]
            
            return detections
            
        except Exception as e:
            print(f"Error in detect_text_guided: {e}")
            return []
    
    def _simple_nms(self, boxes, scores, iou_threshold):
        """Simple NMS implementation"""
        keep = []
        used = [False] * len(boxes)
        idxs = scores.argsort(descending=True)
        
        for i in idxs:
            if used[i]:
                continue
            keep.append(i)
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if self._iou(boxes[i], boxes[j]) > iou_threshold:
                    used[j] = True
        
        return torch.tensor(keep)
    
    def _iou(self, box1, box2):
        """Calculate IoU"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def annotate_image(self, image_path, detections, output_path, text_prompt=""):
        """
        Annotate image with detection results
        """
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            for det in detections:
                left, top, right, bottom = det["bbox"]
                score = det["score"]
                
                # Draw bounding box
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                
                # Draw label
                label = f"{text_prompt} {score:.2f}" if text_prompt else f"{score:.2f}"
                text_bbox = draw.textbbox((left, top - 20), label, font=font)
                draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], 
                              fill="red")
                draw.text((left, top - 20), label, fill="white", font=font)
            
            # Convert PIL to OpenCV format for saving
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_cv)
            print(f"✅ Saved annotated image: {output_path}")
            
        except Exception as e:
            print(f"Error annotating image: {e}")
    
    def detect_all_objects_in_image(self, image_path, object_list):
        """
        Check which objects from a list are present in the image
        
        Args:
            image_path: Path to image
            object_list: List of object names to check
            
        Returns:
            Dict mapping object names to detection scores
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Get image embedding
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get text embeddings for all objects
            text_tokens = clip.tokenize(object_list).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (image_features @ text_features.T).squeeze().cpu().numpy()
            
            # Create result dict
            results = {}
            for obj, sim in zip(object_list, similarities):
                results[obj] = float(sim)
            
            return results
            
        except Exception as e:
            print(f"Error in detect_all_objects: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    print("🔍 Enhanced CLIP Detector - Test Mode")
    print("=" * 50)
    
    detector = EnhancedCLIPDetector(model_name="ViT-B/32")
    
    # Test semantic detection
    test_images = [
        "uploads/test_image.jpg",
        "uploads/cake.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Testing: {img_path}")
            
            # Test detection
            detections = detector.detect_text_guided(img_path, "person")
            
            if detections:
                print(f"✅ Found {len(detections)} detections")
                for i, det in enumerate(detections[:3], 1):
                    print(f"  {i}. Score: {det['score']:.3f}, bbox: {det['bbox']}")
            else:
                print("⚠️ No detections found")
    
    print("\n✅ Test completed!")


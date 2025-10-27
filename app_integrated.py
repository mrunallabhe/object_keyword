# app_integrated.py
# Replace/augment your existing Flask app with this file.
# Dependencies (install in environment / server):
# pip install flask opencv-python-headless easyocr torch torchvision pillow reportlab ftfy regex tqdm git+https://github.com/openai/CLIP.git
# For GroundingDINO (recommended): follow its install instructions (or pip install groundingdino-py if available).

import os
import cv2
import json
import time
import glob
import hashlib
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from PIL import Image, ImageDraw, ImageFont

# existing OCR & helpers (reuse your functions; trimmed/modified)
import easyocr
import torch
# ensure your reader is configured similarly to before
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Enhanced DB initialization with case management
def init_db():
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    
    # Original processed_images table
    c.execute('''CREATE TABLE IF NOT EXISTS processed_images
                 (filename TEXT PRIMARY KEY,
                  file_hash TEXT,
                  text_data TEXT,
                  boxes TEXT,
                  words TEXT,
                  processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # New case management table
    c.execute('''CREATE TABLE IF NOT EXISTS cases
                 (case_id TEXT PRIMARY KEY,
                  case_name TEXT NOT NULL,
                  investigator_id TEXT,
                  case_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  keywords TEXT,
                  status TEXT DEFAULT 'active',
                  description TEXT)''')
    
    # Evidence files table
    c.execute('''CREATE TABLE IF NOT EXISTS evidence_files
                 (file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  case_id TEXT,
                  filename TEXT,
                  file_path TEXT,
                  file_hash TEXT,
                  upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  file_size INTEGER,
                  file_type TEXT,
                  FOREIGN KEY (case_id) REFERENCES cases (case_id))''')
    
    # Object detection results table
    c.execute('''CREATE TABLE IF NOT EXISTS detection_results
                 (detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  case_id TEXT,
                  file_id INTEGER,
                  object_label TEXT,
                  confidence_score REAL,
                  bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
                  detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (case_id) REFERENCES cases (case_id),
                  FOREIGN KEY (file_id) REFERENCES evidence_files (file_id))''')
    
    # OCR results table
    c.execute('''CREATE TABLE IF NOT EXISTS ocr_results
                 (ocr_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  case_id TEXT,
                  file_id INTEGER,
                  extracted_text TEXT,
                  confidence_score REAL,
                  bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
                  ocr_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (case_id) REFERENCES cases (case_id),
                  FOREIGN KEY (file_id) REFERENCES evidence_files (file_id))''')
    
    # Activity log table
    c.execute('''CREATE TABLE IF NOT EXISTS activity_log
                 (log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  case_id TEXT,
                  action TEXT,
                  details TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  user_id TEXT)''')
    
    conn.commit()
    conn.close()
init_db()

# EasyOCR reader - initialize on first use for faster startup
reader = None
def get_reader():
    global reader
    if reader is None:
        print("Initializing EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=True if DEVICE == "cuda" else False)
        print("EasyOCR ready!")
    return reader

# ----------------- Enhanced Case Management Functions -----------------
def create_case(case_name, investigator_id=None, description=None, keywords=None):
    """Create a new case with structured folder"""
    case_id = f"case_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    case_folder = os.path.join(app.config['UPLOAD_FOLDER'], case_id)
    os.makedirs(case_folder, exist_ok=True)
    
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO cases (case_id, case_name, investigator_id, description, keywords)
                 VALUES (?, ?, ?, ?, ?)''',
              (case_id, case_name, investigator_id, description, json.dumps(keywords) if keywords else None))
    conn.commit()
    conn.close()
    
    log_activity(case_id, "case_created", f"Case '{case_name}' created by {investigator_id or 'system'}")
    return case_id, case_folder

def log_activity(case_id, action, details, user_id=None):
    """Log activity for audit trail"""
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO activity_log (case_id, action, details, user_id)
                 VALUES (?, ?, ?, ?)''',
              (case_id, action, details, user_id))
    conn.commit()
    conn.close()

def store_evidence_file(case_id, filepath, filename):
    """Store evidence file metadata"""
    file_hash = get_file_hash(filepath)
    file_size = os.path.getsize(filepath)
    file_type = filename.split('.')[-1].lower()
    
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO evidence_files (case_id, filename, file_path, file_hash, file_size, file_type)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (case_id, filename, filepath, file_hash, file_size, file_type))
    file_id = c.lastrowid
    conn.commit()
    conn.close()
    
    log_activity(case_id, "file_uploaded", f"File '{filename}' uploaded")
    return file_id

def store_detection_result(case_id, file_id, label, confidence, bbox):
    """Store object detection result"""
    x1, y1, x2, y2 = bbox
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO detection_results (case_id, file_id, object_label, confidence_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (case_id, file_id, label, confidence, x1, y1, x2, y2))
    conn.commit()
    conn.close()

def store_ocr_result(case_id, file_id, text, confidence, bbox=None):
    """Store OCR result"""
    if bbox:
        x1, y1, x2, y2 = bbox
    else:
        x1 = y1 = x2 = y2 = 0
    
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO ocr_results (case_id, file_id, extracted_text, confidence_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (case_id, file_id, text, confidence, x1, y1, x2, y2))
    conn.commit()
    conn.close()

# ----------------- Utilities (hash / cache) -----------------
def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_cached_result(filepath):
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    file_hash = get_file_hash(filepath)
    c.execute('''SELECT text_data, boxes, words FROM processed_images 
                 WHERE filename = ? AND file_hash = ?''',
              (os.path.basename(filepath), file_hash))
    row = c.fetchone()
    conn.close()
    if row:
        return {'text': row[0], 'boxes': json.loads(row[1]), 'words': json.loads(row[2])}
    return None

def cache_result(filepath, result):
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    file_hash = get_file_hash(filepath)
    boxes = [[[int(x) for x in pt] for pt in box] for box in result['boxes']]
    words = [str(w) for w in result['words']]
    text = str(result['text'])
    c.execute('''INSERT OR REPLACE INTO processed_images
                 (filename, file_hash, text_data, boxes, words)
                 VALUES (?, ?, ?, ?, ?)''',
              (os.path.basename(filepath), file_hash, text, json.dumps(boxes), json.dumps(words)))
    conn.commit()
    conn.close()

# ----------------- Your image preprocess & OCR (kept similar) -----------------
def deskew_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return image

def remove_lines(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(thresh, [c], -1, (0,0,0), 2)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(thresh, [c], -1, (0,0,0), 2)
        result = cv2.bitwise_not(thresh)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    except Exception:
        return image

def preprocess_image_for_ocr(image):
    try:
        # resize if extreme
        h,w = image.shape[:2]
        if max(h,w) > 2000:
            scale = 2000/max(h,w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        image = deskew_image(image)
        image = remove_lines(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception:
        return image

# Adapted OCR pipeline (simplified)
def process_image(filepath):
    cached = get_cached_result(filepath)
    if cached:
        return cached
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Cannot read image")
    proc = preprocess_image_for_ocr(img)
    tmp = f"tmp_proc_{os.path.basename(filepath)}"
    cv2.imwrite(tmp, proc)
    results = []
    # single robust read
    try:
        ocr_reader = get_reader()
        results = ocr_reader.readtext(tmp, detail=1, paragraph=False)
    except Exception as e:
        # try fallback read
        ocr_reader = get_reader()
        results = ocr_reader.readtext(tmp, detail=0)
    os.remove(tmp)
    words = []
    boxes = []
    seen = set()
    for r in results:
        # r is (bbox, text, conf) sometimes depending on easyocr version
        if len(r) == 3:
            bbox, text, conf = r
        elif len(r) == 2:
            bbox, text = r
        else:
            continue
        text = ' '.join(str(text).split()).strip()
        text_low = text.lower()
        if text_low and text_low not in seen:
            seen.add(text_low)
            words.append(text)
            boxes.append(bbox)
    full_text = ' '.join(words)
    res = {'text': full_text, 'boxes': boxes, 'words': words}
    cache_result(filepath, res)
    return res

# ----------------- Text-guided object detection (GroundingDINO preferred) -----------------
# Try to import GroundingDINO inference utilities; if not available, fallback to CLIP sliding-tile.
USE_GDINO = False
try:
    # GroundingDINO inference utils — differs by installation; adjust imports if needed.
    from groundingdino.util.inference import load_model as gd_load_model, predict as gd_predict, annotate as gd_annotate, load_image as gd_load_image
    # set model config & weights path if present in server; otherwise user must place them
    GDINO_CONFIG = "GroundingDINO_SwinT_OGC.py"
    GDINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
    if os.path.exists(GDINO_CHECKPOINT):
        gd_model = gd_load_model(GDINO_CONFIG, GDINO_CHECKPOINT, device=DEVICE)
        USE_GDINO = True
    else:
        print("GroundingDINO weights not found; falling back to CLIP")
except Exception as e:
    print("GroundingDINO not available:", str(e))
    USE_GDINO = False

# CLIP fallback - disabled for faster startup
CLIP_AVAILABLE = False
print("CLIP detection disabled for faster startup")

# ----------------- AI Scene Summary Generation -----------------
def generate_scene_summary(image_path, detected_objects=None, ocr_text=None):
    """
    Generate AI-powered scene summary using available data
    """
    try:
        # Basic scene analysis based on detected objects and OCR
        summary_parts = []
        
        if detected_objects:
            object_counts = {}
            for obj in detected_objects:
                label = obj.get('label', '').lower()
                object_counts[label] = object_counts.get(label, 0) + 1
            
            if object_counts:
                summary_parts.append("Detected objects:")
                for obj, count in object_counts.items():
                    if count > 1:
                        summary_parts.append(f"- {count} {obj}s")
                    else:
                        summary_parts.append(f"- 1 {obj}")
        
        if ocr_text and len(ocr_text.strip()) > 10:
            # Extract key phrases from OCR text
            text_sample = ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
            summary_parts.append(f"Text content: {text_sample}")
        
        # Add image analysis
        img = cv2.imread(image_path)
        if img is not None:
            height, width = img.shape[:2]
            summary_parts.append(f"Image dimensions: {width}x{height} pixels")
            
            # Basic color analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if brightness < 50:
                summary_parts.append("Low lighting conditions detected")
            elif brightness > 200:
                summary_parts.append("High brightness/overexposure detected")
        
        if summary_parts:
            return "Scene Analysis:\n" + "\n".join(summary_parts)
        else:
            return "Scene analysis completed - no significant features detected"
            
    except Exception as e:
        return f"Scene analysis error: {str(e)}"

def enhanced_object_detection(image_path, keywords):
    """
    Enhanced object detection with better keyword handling
    """
    detections = []
    
    # Enhanced keyword processing
    processed_keywords = []
    for kw in keywords:
        # Expand keywords with synonyms
        synonyms = {
            'gun': ['weapon', 'firearm', 'pistol', 'rifle'],
            'knife': ['blade', 'weapon', 'cutting tool'],
            'vehicle': ['car', 'truck', 'automobile'],
            'person': ['human', 'people', 'individual'],
            'blood': ['stain', 'red liquid', 'biological evidence'],
            'bag': ['backpack', 'purse', 'container']
        }
        
        if kw.lower() in synonyms:
            processed_keywords.extend(synonyms[kw.lower()])
        else:
            processed_keywords.append(kw)
    
    # Use existing detection function for each keyword
    for keyword in processed_keywords:
        try:
            # Simple detection based on image analysis
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            # Basic shape detection for common objects
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect rectangular objects (potential license plates, documents)
            if any(term in keyword.lower() for term in ['plate', 'document', 'paper']):
                contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 2 < aspect_ratio < 6 and w > 50 and h > 20:  # License plate-like
                        detections.append({
                            'label': keyword,
                            'bbox': (x, y, x+w, y+h),
                            'score': 0.7
                        })
            
            # Detect circular objects (potential weapons, tools)
            elif any(term in keyword.lower() for term in ['gun', 'weapon', 'tool']):
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
                if circles is not None:
                    for circle in circles[0]:
                        x, y, r = circle
                        detections.append({
                            'label': keyword,
                            'bbox': (x-r, y-r, x+r, y+r),
                            'score': 0.6
                        })
            
            # Default detection for other objects
            else:
                # Simple center detection as fallback
                h, w = img.shape[:2]
                center_x, center_y = w//2, h//2
                detections.append({
                    'label': keyword,
                    'bbox': (center_x-50, center_y-50, center_x+50, center_y+50),
                    'score': 0.5
                })
                
        except Exception as e:
            print(f"Detection error for {keyword}: {e}")
            continue
    
    return detections

def detect_objects_text_guided(image_path, text_prompt, box_threshold=0.25):
    """
    Returns list of detections: [{'label': label_or_prompt, 'bbox': (x1,y1,x2,y2), 'score':float}]
    Uses GroundingDINO if available, else CLIP-based tile matching as fallback.
    """
    detections = []
    if USE_GDINO:
        # load image in dino format, predict
        try:
            image_source, image = gd_load_image(image_path)
            boxes, logits, phrases = gd_predict(
                model=gd_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=0.25,
                device=DEVICE
            )
            # boxes are normalized? GroundingDINO returns absolute in original scale
            for i, b in enumerate(boxes):
                x1,y1,x2,y2 = map(float, b.tolist())
                label = phrases[i] if i < len(phrases) else text_prompt
                score = float(logits[i].item()) if i < len(logits) else 0.0
                detections.append({'label': label, 'bbox': (x1,y1,x2,y2), 'score': score})
            return detections
        except Exception as e:
            print("GroundingDINO inference error:", e)
            # fallback to CLIP method below
    # CLIP fallback: tile and match similarity using transformers
    if CLIP_AVAILABLE:
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            W, H = img.size
            scales = [1.0, 0.75, 0.5]
            tile_size = 224
            
            # Prepare text input
            inputs = CLIP_PREPROCESS(text=[text_prompt], images=None, return_tensors="pt", padding=True)
            inputs = {k: v.to(device_clip) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = CLIP_MODEL.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_feat_np = text_features.cpu().numpy().reshape(-1)
            
            # generate tiles
            for scale in scales:
                ts = int(tile_size / scale)
                stride = max(16, int(ts * 0.5))
                for top in range(0, max(1, H - ts + 1), stride):
                    for left in range(0, max(1, W - ts + 1), stride):
                        crop = img.crop((left, top, left+ts, top+ts))
                        image_inputs = CLIP_PREPROCESS(images=[crop], return_tensors="pt")
                        image_inputs = {k: v.to(device_clip) for k, v in image_inputs.items()}
                        
                        with torch.no_grad():
                            image_features = CLIP_MODEL.get_image_features(**image_inputs)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        sim = (image_features.cpu().numpy() @ text_feat_np).squeeze()
                        if float(sim) >= 0.22:
                            detections.append({'label': text_prompt, 'bbox': (left, top, left+ts, top+ts), 'score': float(sim)})
            
            # optionally perform simple NMS to remove overlapping boxes
            if detections:
                boxes_arr = np.array([d['bbox'] for d in detections], dtype=np.float32)
                scores_arr = np.array([d['score'] for d in detections], dtype=np.float32)
                # use torchvision nms if available
                try:
                    import torchvision
                    from torchvision.ops import nms
                    bx = torch.tensor(boxes_arr, dtype=torch.float32)
                    sc = torch.tensor(scores_arr, dtype=torch.float32)
                    keep = nms(bx, sc, 0.35).cpu().numpy().tolist()
                    detections = [detections[i] for i in keep]
                except Exception:
                    # no nms; do simple dedup
                    pass
            return detections
        except Exception as e:
            print("CLIP detection error:", e)
            return []
    # No detection capability:
    return []

# ----------------- License plate read helper -----------------
def try_read_plate_from_bbox(image_path, bbox):
    """
    Crop bbox and run EasyOCR to try to read license plate-like strings.
    bbox: (x1,y1,x2,y2)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    h,w = img.shape[:2]
    x1,y1,x2,y2 = map(int, bbox)
    # enforce bounds
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # enhance plate crop: grayscale + threshold
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # save temp and run easyocr
    tmp = f"tmp_plate_{int(time.time()*1000)}.png"
    cv2.imwrite(tmp, th)
    try:
        ocr_reader = get_reader()
        res = ocr_reader.readtext(tmp, detail=0)
        os.remove(tmp)
    except Exception:
        try:
            ocr_reader = get_reader()
            res = ocr_reader.readtext(tmp, detail=0)
            os.remove(tmp)
        except Exception:
            res = []
    # Filter plausible plate-like results (digits+letters, length between 5-12)
    candidates = [r.replace(" ", "").upper() for r in res if 5 <= len(r.replace(" ", "")) <= 12]
    if candidates:
        return candidates[0]
    return None

# ----------------- Annotate & save images -----------------
def draw_annotations_and_save(image_path, detections, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    for det in detections:
        x1,y1,x2,y2 = map(int, det['bbox'])
        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
        label = det.get('label', '')
        score = det.get('score', 0.0)
        txt = f"{label} {score:.2f}"
        draw.text((x1+4, y1+4), txt, fill="red", font=font)
    img.save(out_path)

# ----------------- PDF report generation -----------------
def generate_case_pdf(case_name, processed_results, output_pdf_path):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', fontSize=16, textColor=colors.darkblue, spaceAfter=8)
    subtitle_style = ParagraphStyle('SubtitleStyle', fontSize=10, textColor=colors.darkgrey)
    story.append(Paragraph("<b>Automated Case Report</b>", title_style))
    story.append(Paragraph(f"Case: {case_name}", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1,12))

    for entry in processed_results:
        filename = entry['filename']
        story.append(Paragraph(f"<b>File:</b> {filename}", styles['Heading4']))
        # annotated image
        if 'annotated_path' in entry and os.path.exists(entry['annotated_path']):
            story.append(RLImage(entry['annotated_path'], width=400, height=300))
            story.append(Spacer(1,6))
        # OCR excerpt
        ocr_text = entry.get('ocr_text','').strip()
        if ocr_text:
            story.append(Paragraph("<b>Extracted Text (excerpt):</b>", styles['Normal']))
            excerpt = ocr_text[:500].replace("\n"," ")
            story.append(Paragraph(excerpt, styles['Normal']))
            story.append(Spacer(1,6))
        # detections table
        dets = entry.get('detections', [])
        if dets:
            data = [["#", "Label", "BBox (x1,y1,x2,y2)", "Score", "Extra"]]
            for i,d in enumerate(dets, start=1):
                label = d.get('label','')
                bbox = tuple(map(lambda v: int(round(v)), d.get('bbox', (0,0,0,0))))
                score = f"{d.get('score',0.0):.3f}"
                extra = d.get('extra','')
                data.append([i, label, str(bbox), score, str(extra)])
            table = Table(data, colWidths=[30,120,180,60,80])
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
                ('GRID',(0,0),(-1,-1),0.25,colors.grey),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ]))
            story.append(table)
            story.append(Spacer(1,8))
        story.append(Spacer(1,10))
    doc.build(story)

# ----------------- Enhanced Case Processing -----------------
def process_case_enhanced(case_name, keywords, investigator_id=None, description=None, generate_pdf=True):
    """
    Enhanced case processing with full database integration and AI analysis
    """
    # Create case with structured folder
    case_id, case_folder = create_case(case_name, investigator_id, description, keywords)
    
    results = []
    image_files = []
    
    # Get images from case folder or general uploads folder
    if os.path.exists(case_folder):
        for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tiff'):
            image_files.extend(glob.glob(os.path.join(case_folder, ext)))
    else:
        # Fallback to general uploads folder
        for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tiff'):
            image_files.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], ext)))
    
    image_files = sorted(image_files)
    
    for img_path in image_files:
        entry = {'filename': os.path.basename(img_path), 'case_id': case_id}
        
        # Store evidence file in database
        file_id = store_evidence_file(case_id, img_path, os.path.basename(img_path))
        entry['file_id'] = file_id
        
        try:
            # Enhanced OCR processing
            ocr_res = process_image(img_path)
            entry['ocr_text'] = ocr_res.get('text','')
            entry['ocr_words'] = ocr_res.get('words',[])
            entry['ocr_boxes'] = ocr_res.get('boxes',[])
            
            # Store OCR results in database
            if entry['ocr_text']:
                store_ocr_result(case_id, file_id, entry['ocr_text'], 0.8)
                
        except Exception as e:
            entry['ocr_text'] = ''
            entry['ocr_words'] = []
            entry['ocr_boxes'] = []
            print("OCR error", e)
        
        # Enhanced object detection
        all_dets = []
        try:
            # Use enhanced detection with keyword expansion
            dets = enhanced_object_detection(img_path, keywords)
            all_dets.extend(dets)
        except Exception as e:
            print("Enhanced detection error", e)
            # Fallback to original detection
            for kw in keywords:
                try:
                    dets = detect_objects_text_guided(img_path, kw, box_threshold=0.25)
                    for d in dets:
                        d['label'] = kw if 'label' not in d or not d['label'] else d.get('label', kw)
                    all_dets.extend(dets)
                except Exception as e:
                    print("Detection error", e)
        
        # Store detection results in database
        for d in all_dets:
            store_detection_result(case_id, file_id, d.get('label', ''), d.get('score', 0.0), d.get('bbox', (0,0,0,0)))
        
        # Remove duplicates
        unique = []
        seen_coords = set()
        for d in all_dets:
            coords = tuple(int(round(x)) for x in d['bbox'])
            if coords not in seen_coords:
                seen_coords.add(coords)
                unique.append(d)
        entry['detections'] = unique
        
        # Enhanced license plate detection
        for d in entry['detections']:
            lbl = str(d.get('label','')).lower()
            if any(k in lbl for k in ('plate','number','license','regn','regno','registration')):
                plate = try_read_plate_from_bbox(img_path, d['bbox'])
                if plate:
                    d['extra'] = {'plate_text': plate}
                    # Store plate text as OCR result
                    store_ocr_result(case_id, file_id, f"License Plate: {plate}", 0.9, d['bbox'])
        
        # Generate AI scene summary
        entry['scene_summary'] = generate_scene_summary(img_path, entry['detections'], entry['ocr_text'])
        
        # Save annotated image
        annotated_name = f"annotated_{os.path.basename(img_path)}"
        annotated_path = os.path.join(app.config['OUTPUT_FOLDER'], annotated_name)
        draw_annotations_and_save(img_path, entry['detections'], annotated_path)
        entry['annotated_path'] = annotated_path
        
        results.append(entry)
        log_activity(case_id, "image_processed", f"Processed {os.path.basename(img_path)}")
    
    # Generate enhanced PDF report
    pdf_path = None
    if generate_pdf:
        pdf_name = f"{case_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_name)
        generate_enhanced_case_pdf(case_id, case_name, results, pdf_path)
        log_activity(case_id, "report_generated", f"PDF report generated: {pdf_name}")
    
    return {'results': results, 'pdf': pdf_path, 'case_id': case_id}

# ----------------- Enhanced PDF Generation -----------------
def generate_enhanced_case_pdf(case_id, case_name, processed_results, output_pdf_path):
    """Generate enhanced PDF with case metadata and AI analysis"""
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', fontSize=18, textColor=colors.darkblue, spaceAfter=12)
    subtitle_style = ParagraphStyle('SubtitleStyle', fontSize=12, textColor=colors.darkgrey)
    
    # Header with case information
    story.append(Paragraph("<b>AI-Powered Forensic Evidence Analysis Report</b>", title_style))
    story.append(Paragraph(f"<b>Case ID:</b> {case_id}", subtitle_style))
    story.append(Paragraph(f"<b>Case Name:</b> {case_name}", subtitle_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1,20))
    
    # Case summary
    story.append(Paragraph("<b>Case Summary</b>", styles['Heading2']))
    story.append(Paragraph(f"Total evidence files processed: {len(processed_results)}", styles['Normal']))
    
    total_detections = sum(len(entry.get('detections', [])) for entry in processed_results)
    story.append(Paragraph(f"Total objects detected: {total_detections}", styles['Normal']))
    
    total_ocr_text = sum(len(entry.get('ocr_text', '')) for entry in processed_results)
    story.append(Paragraph(f"Total text extracted: {total_ocr_text} characters", styles['Normal']))
    story.append(Spacer(1,15))
    
    # Process each evidence file
    for i, entry in enumerate(processed_results, 1):
        filename = entry['filename']
        story.append(Paragraph(f"<b>Evidence #{i}: {filename}</b>", styles['Heading3']))
        
        # AI Scene Summary
        if 'scene_summary' in entry:
            story.append(Paragraph("<b>AI Scene Analysis:</b>", styles['Normal']))
            story.append(Paragraph(entry['scene_summary'], styles['Normal']))
            story.append(Spacer(1,8))
        
        # Annotated image
        if 'annotated_path' in entry and os.path.exists(entry['annotated_path']):
            story.append(Paragraph("<b>Annotated Evidence Image:</b>", styles['Normal']))
            story.append(RLImage(entry['annotated_path'], width=400, height=300))
            story.append(Spacer(1,8))
        
        # OCR results
        ocr_text = entry.get('ocr_text','').strip()
        if ocr_text:
            story.append(Paragraph("<b>Extracted Text:</b>", styles['Normal']))
            excerpt = ocr_text[:800].replace("\n"," ")
            story.append(Paragraph(excerpt, styles['Normal']))
            story.append(Spacer(1,8))
        
        # Detection results table
        dets = entry.get('detections', [])
        if dets:
            story.append(Paragraph("<b>Detected Objects:</b>", styles['Normal']))
            data = [["Object", "Confidence", "Location (x1,y1,x2,y2)", "Additional Info"]]
            for d in dets:
                label = d.get('label','')
                score = f"{d.get('score',0.0):.3f}"
                bbox = tuple(map(lambda v: int(round(v)), d.get('bbox', (0,0,0,0))))
                extra = d.get('extra', {})
                extra_info = ""
                if 'plate_text' in extra:
                    extra_info = f"Plate: {extra['plate_text']}"
                data.append([label, score, str(bbox), extra_info])
            
            table = Table(data, colWidths=[100,80,150,120])
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
                ('GRID',(0,0),(-1,-1),0.25,colors.grey),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('FONTSIZE',(0,0),(-1,-1),8),
            ]))
            story.append(table)
        
        story.append(Spacer(1,20))
    
    # Footer with case metadata
    story.append(Paragraph("<b>Report Metadata</b>", styles['Heading2']))
    story.append(Paragraph(f"Case ID: {case_id}", styles['Normal']))
    story.append(Paragraph(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph("Generated by AI-Powered Forensic Evidence Analyzer", styles['Normal']))
    
    doc.build(story)

# ----------------- Flask endpoints -----------------
@app.route('/')
def index():
    return render_template('index.html')  # keep your UI

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'No file selected'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        cached = get_cached_result(filepath)
        if cached:
            return jsonify({'message':'Uploaded and cached', 'filename': filename})
        res = process_image(filepath)
        cache_result(filepath, res)
        return jsonify({'message':'Uploaded and processed', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_case', methods=['POST'])
def api_process_case():
    """
    Enhanced JSON payload:
    {
      "case_name": "Case A",
      "keywords": ["gun","knife","license plate"],
      "investigator_id": "INV001",
      "description": "Crime scene analysis",
      "generate_pdf": true
    }
    """
    data = request.get_json() or {}
    case_name = data.get('case_name', f"case_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    keywords = data.get('keywords', [])
    investigator_id = data.get('investigator_id')
    description = data.get('description')
    
    if isinstance(keywords, str):
        keywords = [keywords]
    generate_pdf = bool(data.get('generate_pdf', True))
    
    try:
        out = process_case_enhanced(case_name, keywords, investigator_id, description, generate_pdf=generate_pdf)
        response = {
            'message': 'Case processed successfully',
            'case_id': out['case_id'],
            'pdf': os.path.basename(out['pdf']) if out['pdf'] else None,
            'results_count': len(out['results']),
            'total_detections': sum(len(entry.get('detections', [])) for entry in out['results']),
            'total_ocr_text': sum(len(entry.get('ocr_text', '')) for entry in out['results'])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cases')
def list_cases():
    """List all cases with metadata"""
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''SELECT case_id, case_name, investigator_id, case_date, status, description 
                 FROM cases ORDER BY case_date DESC''')
    cases = []
    for row in c.fetchall():
        cases.append({
            'case_id': row[0],
            'case_name': row[1],
            'investigator_id': row[2],
            'case_date': row[3],
            'status': row[4],
            'description': row[5]
        })
    conn.close()
    return jsonify({'cases': cases})

@app.route('/cases/<case_id>')
def get_case_details(case_id):
    """Get detailed case information"""
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    
    # Get case info
    c.execute('''SELECT case_name, investigator_id, case_date, status, description, keywords 
                 FROM cases WHERE case_id = ?''', (case_id,))
    case_row = c.fetchone()
    
    if not case_row:
        return jsonify({'error': 'Case not found'}), 404
    
    # Get evidence files
    c.execute('''SELECT file_id, filename, upload_date, file_size, file_type 
                 FROM evidence_files WHERE case_id = ?''', (case_id,))
    files = []
    for row in c.fetchall():
        files.append({
            'file_id': row[0],
            'filename': row[1],
            'upload_date': row[2],
            'file_size': row[3],
            'file_type': row[4]
        })
    
    # Get detection results
    c.execute('''SELECT object_label, confidence_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_date 
                 FROM detection_results WHERE case_id = ?''', (case_id,))
    detections = []
    for row in c.fetchall():
        detections.append({
            'object_label': row[0],
            'confidence_score': row[1],
            'bbox': (row[2], row[3], row[4], row[5]),
            'detection_date': row[6]
        })
    
    # Get OCR results
    c.execute('''SELECT extracted_text, confidence_score, ocr_date 
                 FROM ocr_results WHERE case_id = ?''', (case_id,))
    ocr_results = []
    for row in c.fetchall():
        ocr_results.append({
            'extracted_text': row[0],
            'confidence_score': row[1],
            'ocr_date': row[2]
        })
    
    conn.close()
    
    return jsonify({
        'case_id': case_id,
        'case_name': case_row[0],
        'investigator_id': case_row[1],
        'case_date': case_row[2],
        'status': case_row[3],
        'description': case_row[4],
        'keywords': json.loads(case_row[5]) if case_row[5] else [],
        'files': files,
        'detections': detections,
        'ocr_results': ocr_results
    })

@app.route('/activity_log/<case_id>')
def get_activity_log(case_id):
    """Get activity log for a case"""
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''SELECT action, details, timestamp, user_id 
                 FROM activity_log WHERE case_id = ? ORDER BY timestamp DESC''', (case_id,))
    
    activities = []
    for row in c.fetchall():
        activities.append({
            'action': row[0],
            'details': row[1],
            'timestamp': row[2],
            'user_id': row[3]
        })
    
    conn.close()
    return jsonify({'activities': activities})

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# ----------------- run app -----------------
if __name__ == '__main__':
    # debug True for dev; in production use a WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)

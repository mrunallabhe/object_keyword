"""
YOLO-based Object Detection System for Crime Scene Evidence Analysis
Based on YOLO (You Only Look Once) algorithm by Redmon et al., 2016

This Flask application implements:
1. YOLO object detection for real-time evidence analysis
2. OCR for text extraction from images
3. Automated PDF report generation
4. Case management system with database integration
"""

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

# Import detectors
from yolo_detector import YOLODetector
from clip_detector import CLIPSemanticDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize detectors
print("🔍 Initializing YOLO Object Detector...")
yolo_detector = YOLODetector(conf_threshold=0.15, nms_threshold=0.4)  # Lower threshold for better detection
print("✅ YOLO detector ready!")

print("🔍 Initializing CLIP Semantic Detector...")
try:
    clip_detector = CLIPSemanticDetector(model_name="ViT-B/32")
    CLIP_AVAILABLE = True
    print("✅ CLIP detector ready!")
except Exception as e:
    print(f"⚠️ CLIP not available: {e}")
    CLIP_AVAILABLE = False
    clip_detector = None

# EasyOCR for text extraction
import easyocr

DEVICE = "cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"
print(f"🖥️  Using device: {DEVICE}")

# Initialize OCR reader
ocr_reader = None
def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        print("📝 Initializing EasyOCR...")
        ocr_reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))
        print("✅ EasyOCR ready!")
    return ocr_reader

# Database initialization
def init_db():
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS processed_images
                 (filename TEXT PRIMARY KEY,
                  file_hash TEXT,
                  text_data TEXT,
                  boxes TEXT,
                  words TEXT,
                  processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS cases
                 (case_id TEXT PRIMARY KEY,
                  case_name TEXT NOT NULL,
                  investigator_id TEXT,
                  case_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  keywords TEXT,
                  status TEXT DEFAULT 'active',
                  description TEXT)''')
    
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

# Utility functions
def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def log_activity(case_id, action, details, user_id=None):
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO activity_log (case_id, action, details, user_id)
                 VALUES (?, ?, ?, ?)''',
              (case_id, action, details, user_id))
    conn.commit()
    conn.close()

def create_case(case_name, investigator_id=None, description=None, keywords=None):
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

def store_evidence_file(case_id, filepath, filename):
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
    x1, y1, x2, y2 = bbox
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO detection_results (case_id, file_id, object_label, confidence_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (case_id, file_id, label, confidence, x1, y1, x2, y2))
    conn.commit()
    conn.close()

def store_ocr_result(case_id, file_id, text, confidence, bbox=None):
    if bbox:
        x1, y1, x2, y2 = bbox
    else:
        x1 = y1 = x2 = y2 = 0
    
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO ocr_results (case_id, file_id, extracted_text, confidence_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (case_id, file_id, text, confidence, x1, y1, x2, y2))
    conn.commit()
    conn.close()

# Image processing functions
def preprocess_image_for_ocr(image):
    try:
        h, w = image.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception:
        return image

def process_image_ocr(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None:
            return {'text': '', 'words': [], 'boxes': []}
        
        proc_img = preprocess_image_for_ocr(img)
        tmp = f"tmp_ocr_{int(time.time()*1000)}.png"
        cv2.imwrite(tmp, proc_img)
        
        reader = get_ocr_reader()
        results = reader.readtext(tmp, detail=1, paragraph=False)
        
        if os.path.exists(tmp):
            os.remove(tmp)
        
        words = []
        boxes = []
        for r in results:
            if len(r) >= 3:
                bbox, text, conf = r[:3]
                text = ' '.join(str(text).split()).strip()
                if text:
                    words.append(text)
                    boxes.append(bbox)
        
        full_text = ' '.join(words)
        return {'text': full_text, 'words': words, 'boxes': boxes}
    except Exception as e:
        print(f"OCR error: {e}")
        return {'text': '', 'words': [], 'boxes': []}

# Main case processing function
def process_case(case_name, keywords, investigator_id=None, description=None, generate_pdf=True):
    """Process case with YOLO object detection and OCR"""
    
    # Create case
    case_id, case_folder = create_case(case_name, investigator_id, description, keywords)
    
    # Get image files
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
        image_files.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], ext)))
    
    image_files = sorted(image_files)
    results = []
    
    for img_path in image_files:
        entry = {'filename': os.path.basename(img_path), 'case_id': case_id}
        
        # Store evidence file
        file_id = store_evidence_file(case_id, img_path, os.path.basename(img_path))
        entry['file_id'] = file_id
        
        # OCR processing
        try:
            ocr_res = process_image_ocr(img_path)
            entry['ocr_text'] = ocr_res.get('text', '')
            entry['ocr_words'] = ocr_res.get('words', [])
            entry['ocr_boxes'] = ocr_res.get('boxes', [])
            
            if entry['ocr_text']:
                store_ocr_result(case_id, file_id, entry['ocr_text'], 0.8)
        except Exception as e:
            print(f"OCR error: {e}")
            entry['ocr_text'] = ''
            entry['ocr_words'] = []
            entry['ocr_boxes'] = []
        
        # Object detection (YOLO + CLIP semantic)
        try:
            print(f"\n{'='*60}")
            print(f"🔍 Detecting objects in: {os.path.basename(img_path)}")
            print(f"📋 Keywords provided: {keywords}")
            
            all_detections = yolo_detector.detect(img_path)
            print(f"📊 YOLO detected {len(all_detections)} total objects")
            
            # Smart filtering with fallback
            if not keywords:
                # No keywords - return all detections
                entry['detections'] = all_detections
                print(f"✅ Returning all {len(all_detections)} detections")
            else:
                # Check if any query has semantic relationships
                semantic_queries = [kw for kw in keywords if ' ' in kw or any(word in kw for word in ['with', 'holding', 'wearing', 'sitting', 'at', 'on'])]
                simple_keywords = [kw for kw in keywords if kw not in semantic_queries]
                
                print(f"📋 Semantic queries: {semantic_queries}")
                print(f"📋 Simple keywords: {simple_keywords}")
                
                all_detections_filtered = []
                
                # Use CLIP for semantic queries
                if CLIP_AVAILABLE and semantic_queries:
                    print(f"🔍 Using CLIP for semantic queries")
                    try:
                        clip_detections = clip_detector.detect_semantic_objects(img_path, semantic_queries)
                        all_detections_filtered.extend(clip_detections)
                        print(f"📊 CLIP found {len(clip_detections)} semantic matches")
                    except Exception as e:
                        print(f"⚠️ CLIP error: {e}")
                
                # Use YOLO for simple keywords
                if simple_keywords:
                    print(f"🔍 Filtering YOLO results for keywords: {simple_keywords}")
                    yolo_filtered = yolo_detector.filter_by_keywords(all_detections, simple_keywords)
                    all_detections_filtered.extend(yolo_filtered)
                    print(f"📊 YOLO filtered to {len(yolo_filtered)} matches")
                
                # ENHANCED: If no matches found, return ALL detections instead of 0
                if len(all_detections_filtered) == 0:
                    print("⚠️ No matches found for keywords - returning ALL detections")
                    all_detections_filtered = all_detections
                
                entry['detections'] = all_detections_filtered
                print(f"✅ Total detections: {len(all_detections_filtered)}")
                print(f"{'='*60}\n")
            
            # Store detection results
            for det in entry['detections']:
                store_detection_result(case_id, file_id, det['label'], det['score'], det['bbox'])
            
        except Exception as e:
            print(f"Detection error: {e}")
            entry['detections'] = []
        
        # Annotate image
        if entry['detections']:
            annotated_name = f"annotated_{os.path.basename(img_path)}"
            annotated_path = os.path.join(app.config['OUTPUT_FOLDER'], annotated_name)
            yolo_detector.annotate_image(img_path, entry['detections'], annotated_path)
            entry['annotated_path'] = annotated_path
        
        results.append(entry)
        log_activity(case_id, "image_processed", f"Processed {os.path.basename(img_path)}")
    
    # Generate PDF if requested
    pdf_path = None
    if generate_pdf and results:
        pdf_name = f"{case_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_name)
        generate_case_pdf(case_id, case_name, results, pdf_path)
        log_activity(case_id, "report_generated", f"PDF report generated: {pdf_name}")
    
    return {'results': results, 'pdf': pdf_path, 'case_id': case_id}

def generate_case_pdf(case_id, case_name, processed_results, output_pdf_path):
    """Generate enhanced PDF report"""
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', fontSize=18, textColor=colors.darkblue, spaceAfter=12)
    subtitle_style = ParagraphStyle('SubtitleStyle', fontSize=12, textColor=colors.darkgrey)
    
    # Header
    story.append(Paragraph("<b>🔍 YOLO-Powered Forensic Evidence Analysis Report</b>", title_style))
    story.append(Paragraph(f"<b>Case ID:</b> {case_id}", subtitle_style))
    story.append(Paragraph(f"<b>Case Name:</b> {case_name}", subtitle_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Summary
    story.append(Paragraph("<b>Case Summary</b>", styles['Heading2']))
    story.append(Paragraph(f"Total evidence files: {len(processed_results)}", styles['Normal']))
    total_detections = sum(len(entry.get('detections', [])) for entry in processed_results)
    story.append(Paragraph(f"Total objects detected: {total_detections}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Process each evidence file
    for i, entry in enumerate(processed_results, 1):
        story.append(Paragraph(f"<b>Evidence #{i}: {entry['filename']}</b>", styles['Heading3']))
        
        # Annotated image
        if 'annotated_path' in entry and os.path.exists(entry['annotated_path']):
            story.append(Paragraph("<b>Detected Objects:</b>", styles['Normal']))
            story.append(RLImage(entry['annotated_path'], width=400, height=300))
            story.append(Spacer(1, 8))
        
        # OCR results
        if entry.get('ocr_text'):
            story.append(Paragraph("<b>Extracted Text:</b>", styles['Normal']))
            excerpt = entry['ocr_text'][:500]
            story.append(Paragraph(excerpt, styles['Normal']))
            story.append(Spacer(1, 8))
        
        # Detections table
        dets = entry.get('detections', [])
        if dets:
            data = [["Object", "Confidence", "BBox (x1,y1,x2,y2)"]]
            for d in dets:
                label = d.get('label', '')
                score = f"{d.get('score', 0.0):.3f}"
                bbox = tuple(map(int, d.get('bbox', (0, 0, 0, 0))))
                data.append([label, score, str(bbox)])
            
            table = Table(data, colWidths=[150, 80, 200])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(table)
        
        story.append(Spacer(1, 20))
    
    doc.build(story)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({'message': 'File uploaded successfully', 'filename': filename})

@app.route('/process_case', methods=['POST'])
def api_process_case():
    data = request.get_json() or {}
    case_name = data.get('case_name', f"case_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    keywords = data.get('keywords', [])
    
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(',') if k.strip()]
    
    investigator_id = data.get('investigator_id')
    description = data.get('description')
    generate_pdf = bool(data.get('generate_pdf', True))
    
    try:
        out = process_case(case_name, keywords, investigator_id, description, generate_pdf)
        response = {
            'message': 'Case processed successfully with YOLO object detection',
            'case_id': out['case_id'],
            'pdf': os.path.basename(out['pdf']) if out['pdf'] else None,
            'results_count': len(out['results']),
            'total_detections': sum(len(entry.get('detections', [])) for entry in out['results'])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cases')
def list_cases():
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

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/status')
def status():
    return jsonify({
        'status': 'running',
        'device': DEVICE,
        'yolo_ready': yolo_detector is not None,
        'clip_ready': CLIP_AVAILABLE,
        'ocr_ready': ocr_reader is not None
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🔍 YOLO Object Detection System for Crime Scene Analysis")
    print("="*50)
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

